use super::*;
use crate::backend::cuda::{
    index_score::{
        IndexCandidatesPlan, IndexCandidatesViews, IndexPositionOrder, IndexScorePlan,
        IndexScoreViews,
    },
    latent_compress::{CompressionBuffers, CompressionInputs, LatentCompressPlan},
    latent_window_attention::{IndexedAttentionPlan, IndexedInputs},
    linear::{Bf16F32Plan, Fp8BlockPlan, Fp8ScaleRef, LinearWeightRef},
    quantize::{Fp4PackPlan, Fp4PackViews},
    selection::{DsaSelectionPlan, DsaTopKInput, DsaTopKOutput},
    state::{LatentCompressedState, LatentOwners, StateStore},
};
use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use std::collections::BTreeMap;

pub(super) type Weights = BTreeMap<String, CudaSlice<u8>>;

pub(super) fn scratch_bytes(
    rows: usize,
    capacity: usize,
    shape: &crate::model_spec::DeepseekSelectionShape,
) -> usize {
    let index_topk = shape.index_topk as usize;
    let candidate_blocks = shape.candidate_topk_blocks as usize;
    let candidate_block_size = shape.candidate_block_size as usize;
    32 + rows
        * (64 * 4
            + 2 * 512 * 4
            + 512 * 2
            + 128 * 2
            + 4096 * 2
            + 32 * 2
            + 256
            + 32
            + 64
            + 4
            + candidate_blocks
            + 128
            + capacity * 8
            + 16384 * 8
            + capacity.div_ceil(candidate_block_size) * 8
            + (candidate_blocks - 1) * 8
            + index_topk * 8
            + 4)
}

pub(super) fn load_weights(
    loader: &mut super::super::super::weights::Loader,
    spec: &ModelSpec,
) -> Result<Weights> {
    let mut weights = Weights::new();
    let owners = crate::backend::cuda::state::latent_owners(spec)?;
    let ds = spec.deepseek_v41.as_ref().context("DSA geometry missing")?;
    for (layer, mapping) in owners.iter().enumerate() {
        let Some(mapping) = mapping else { continue };
        let mut entries = Vec::new();
        if mapping.kv == layer {
            entries.extend([
                (
                    "compressor.wkv.weight",
                    vec![512, spec.hidden as usize],
                    safetensors::Dtype::BF16,
                ),
                (
                    "compressor.norm.weight",
                    vec![512],
                    safetensors::Dtype::BF16,
                ),
                (
                    "indexer.wk.weight",
                    vec![128, 512],
                    safetensors::Dtype::BF16,
                ),
                ("indexer.k_norm.weight", vec![128], safetensors::Dtype::BF16),
            ]);
            if mapping.ratio == 2 {
                entries.push((
                    "compressor.wgate.weight",
                    vec![512, spec.hidden as usize],
                    safetensors::Dtype::BF16,
                ));
            }
        }
        if mapping.index == layer {
            entries.extend([
                (
                    "indexer.wq_b.weight",
                    vec![4096, ds.q_lora_rank as usize],
                    safetensors::Dtype::F8_E4M3,
                ),
                (
                    "indexer.wq_b.scale",
                    vec![128, ds.q_lora_rank as usize / 32],
                    safetensors::Dtype::F8_E8M0,
                ),
                (
                    "indexer.weights_proj.weight",
                    vec![32, spec.hidden as usize],
                    safetensors::Dtype::BF16,
                ),
            ]);
        }
        for (name, shape, dtype) in entries {
            let name = format!("layers.{layer}.attn.{name}");
            weights.insert(name.clone(), loader.load_bytes(&name, &shape, dtype)?);
        }
    }
    Ok(weights)
}

fn dense<T: cudarc::driver::DeviceRepr + Copy, const R: usize>(
    data: &CudaSlice<T>,
    s: &Arc<CudaStream>,
    shape: [u32; R],
) -> Result<Tensor<T, R>> {
    view(data, s, 0, shape, dense_layout(shape)?.0)
}

fn reshape<T: Copy, const R: usize, const S: usize>(
    input: Tensor<T, R>,
    shape: [u32; S],
) -> Result<Tensor<T, S>> {
    let (stride, count) = dense_layout(shape)?;
    ensure!(
        count * std::mem::size_of::<T>() as u64 <= input.span_bytes,
        "DSA reshape exceeds allocation"
    );
    unsafe {
        Tensor::new(
            input.ptr,
            shape,
            stride,
            count * std::mem::size_of::<T>() as u64,
        )
    }
}

struct Bf16Projection {
    input: Tensor<bf16, 2>,
    weight: Tensor<bf16, 2>,
    output: Tensor<bf16, 2>,
}
impl Bf16Projection {
    fn new(
        blas: &Blas,
        input: Tensor<bf16, 2>,
        weight: Tensor<bf16, 2>,
        output: Tensor<bf16, 2>,
    ) -> Result<Self> {
        blas.prepare_skinny(crate::backend::setup::LinearShape {
            rows: input.shape[0] as usize,
            input: input.shape[1] as usize,
            output: output.shape[1] as usize,
        })?;
        Ok(Self {
            input,
            weight,
            output,
        })
    }
    fn run(&self, blas: &Blas) -> Result<()> {
        blas.linear_typed_bound(
            crate::op_contract::ExecutionPhase::Prefill,
            LinearWeightRef::Bf16(self.weight),
            self.input,
            &mut TensorMut(self.output),
            16,
        )
    }
}

struct Norm {
    values: Tensor<bf16, 3>,
    weight: Tensor<bf16, 1>,
    sequence: SeqRef,
}
impl Norm {
    fn run(&self, common: &CommonOpsRef, stream: &Arc<CudaStream>, eps: f32) -> Result<()> {
        common.norm.rms_with(
            stream,
            RowDomain {
                meta: self.sequence,
                row_base: 0,
                row_capacity: self.values.shape[0],
            },
            self.values,
            self.weight,
            &mut TensorMut(self.values),
            RmsOptions {
                mode: RmsMode::Unrounded,
                eps,
            },
        )
    }
}

struct Scratch {
    group_meta: CudaSlice<i32>,
    group_angles: CudaSlice<f32>,
    values: CudaSlice<f32>,
    gates: CudaSlice<f32>,
    pooled: CudaSlice<bf16>,
    key: CudaSlice<bf16>,
    query: CudaSlice<bf16>,
    head_weights: CudaSlice<bf16>,
    kv_codes: CudaSlice<u8>,
    kv_scales: CudaSlice<u8>,
    key_codes: CudaSlice<u8>,
    key_scales: CudaSlice<u8>,
    query_codes: CudaSlice<u8>,
    query_scales: CudaSlice<u8>,
    full_scores: CudaSlice<f32>,
    full_ids: CudaSlice<i32>,
    restricted_scores: CudaSlice<f32>,
    restricted_ids: CudaSlice<i32>,
    block_scores: CudaSlice<f32>,
    block_ids: CudaSlice<i32>,
    block_selected_scores: CudaSlice<f32>,
    block_selected_ids: CudaSlice<i32>,
    selected_scores: CudaSlice<f32>,
    selected_ids: CudaSlice<i32>,
    margins: CudaSlice<f32>,
}
impl Scratch {
    fn new(
        s: &Arc<CudaStream>,
        rows: usize,
        capacity: usize,
        shape: &crate::model_spec::DeepseekSelectionShape,
    ) -> Result<Self> {
        let index_topk = shape.index_topk as usize;
        let candidate_blocks = shape.candidate_topk_blocks as usize;
        let candidate_block_size = shape.candidate_block_size as usize;
        Ok(Self {
            group_meta: s.alloc_zeros(8)?,
            group_angles: s.alloc_zeros(rows * 64)?,
            values: s.alloc_zeros(rows * 512)?,
            gates: s.alloc_zeros(rows * 512)?,
            pooled: s.alloc_zeros(rows * 512)?,
            key: s.alloc_zeros(rows * 128)?,
            query: s.alloc_zeros(rows * 4096)?,
            head_weights: s.alloc_zeros(rows * 32)?,
            kv_codes: s.alloc_zeros(rows * 256)?,
            kv_scales: s.alloc_zeros(rows * 32)?,
            key_codes: s.alloc_zeros(rows * 64)?,
            key_scales: s.alloc_zeros(rows * 4)?,
            query_codes: s.alloc_zeros(rows * 2048)?,
            query_scales: s.alloc_zeros(rows * 128)?,
            full_scores: s.alloc_zeros(rows * capacity)?,
            full_ids: s.alloc_zeros(rows * capacity)?,
            restricted_scores: s.alloc_zeros(rows * 16384)?,
            restricted_ids: s.alloc_zeros(rows * 16384)?,
            block_scores: s.alloc_zeros(rows * capacity.div_ceil(candidate_block_size))?,
            block_ids: s.alloc_zeros(rows * capacity.div_ceil(candidate_block_size))?,
            block_selected_scores: s.alloc_zeros(rows * (candidate_blocks - 1))?,
            block_selected_ids: s.alloc_zeros(rows * (candidate_blocks - 1))?,
            selected_scores: s.alloc_zeros(rows * index_topk)?,
            selected_ids: s.alloc_zeros(rows * index_topk)?,
            margins: s.alloc_zeros(rows)?,
        })
    }
}

struct Publication {
    sequence: SeqRef,
    generation: u64,
    ticket: u64,
    cursor: u64,
    count: u64,
    groups: u64,
    frequencies: u64,
    angles: u64,
    payload: [u64; 4],
    cache: [u64; 4],
    owner: i32,
    ratio: i32,
    rows: i32,
    capacity: i32,
}
impl Publication {
    fn begin(&self, s: &Arc<CudaStream>, function: &CudaFunction) -> Result<()> {
        unsafe {
            s.launch_builder(function)
                .arg(&self.sequence.ptr)
                .arg(&self.generation)
                .arg(&self.ticket)
                .arg(&self.cursor)
                .arg(&self.count)
                .arg(&self.groups)
                .arg(&self.frequencies)
                .arg(&self.angles)
                .arg(&self.owner)
                .arg(&self.ratio)
                .arg(&self.rows)
                .arg(&64i32)
                .launch(one_block())?;
        }
        Ok(())
    }
    fn commit(&self, s: &Arc<CudaStream>, function: &CudaFunction) -> Result<()> {
        unsafe {
            s.launch_builder(function)
                .arg(&self.sequence.ptr)
                .arg(&self.generation)
                .arg(&self.ticket)
                .arg(&self.cursor)
                .arg(&self.count)
                .arg(&self.payload[0])
                .arg(&self.payload[1])
                .arg(&self.payload[2])
                .arg(&self.payload[3])
                .arg(&self.cache[0])
                .arg(&self.cache[1])
                .arg(&self.cache[2])
                .arg(&self.cache[3])
                .arg(&self.owner)
                .arg(&self.ratio)
                .arg(&self.capacity)
                .launch(one_block())?;
        }
        Ok(())
    }
}
fn one_block() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

struct Ticket {
    sequence: SeqRef,
    generation: u64,
    kv: u64,
    index: u64,
    owners: LatentOwners,
}
impl Ticket {
    fn run(&self, s: &Arc<CudaStream>, function: &CudaFunction, publish: i32) -> Result<()> {
        unsafe {
            s.launch_builder(function)
                .arg(&self.sequence.ptr)
                .arg(&self.generation)
                .arg(&self.kv)
                .arg(&self.index)
                .arg(&(self.owners.kv as i32))
                .arg(&(self.owners.index as i32))
                .arg(&(self.owners.ratio as i32))
                .arg(&publish)
                .launch(LaunchConfig {
                    block_dim: (1, 1, 1),
                    ..one_block()
                })?;
        }
        Ok(())
    }
}

struct Compressor {
    publication: Publication,
    float_values: Option<Bf16F32Plan>,
    float_gates: Option<Bf16F32Plan>,
    plain_values: Option<Bf16Projection>,
    pool: Option<LatentCompressPlan>,
    norm: Norm,
    key_projection: Bf16Projection,
    key_norm: Norm,
    key: Tensor<bf16, 3>,
    latent: Tensor<bf16, 3>,
    angles: Tensor<f32, 2>,
    key_pack: Fp4PackPlan,
    latent_pack: Fp4PackPlan,
}
impl Compressor {
    fn run(
        &self,
        runtime: &Runtime,
        mixer: &DeepseekMixer,
        s: &Arc<CudaStream>,
        blas: &Blas,
        rope: &Ops,
    ) -> Result<()> {
        self.publication.begin(s, &runtime.begin)?;
        if let (Some(values), Some(gates), Some(pool)) =
            (&self.float_values, &self.float_gates, &self.pool)
        {
            values.run(s)?;
            gates.run(s)?;
            pool.run(s)?;
        } else if let Some(values) = &self.plain_values {
            values.run(blas)?;
        } else {
            anyhow::bail!("DSA compressor projection not bound");
        }
        self.norm.run(&mixer.common, s, mixer.eps)?;
        self.key_projection.run(blas)?;
        self.key_norm.run(&mixer.common, s, mixer.eps)?;
        rope.rope_pair_tail(s, &mut TensorMut(self.key), self.angles)?;
        self.key_pack.run(s)?;
        rope.rope_pair_tail(s, &mut TensorMut(self.latent), self.angles)?;
        self.latent_pack.run(s)?;
        self.publication.commit(s, &runtime.commit)
    }
}

struct Indexer {
    query_projection: Fp8BlockPlan,
    weight_projection: Bf16Projection,
    query: Tensor<bf16, 3>,
    angles: Tensor<f32, 2>,
    pack: Fp4PackPlan,
    full: Option<IndexScorePlan>,
    candidates: Option<IndexCandidatesPlan>,
    candidate_ticket: Option<Ticket>,
    restricted: Option<IndexScorePlan>,
    selector: DsaSelectionPlan,
    input: DsaTopKInput,
    output: DsaTopKOutput,
    order: IndexPositionOrder,
}
impl Indexer {
    fn run(&self, runtime: &Runtime, s: &Arc<CudaStream>, blas: &Blas, rope: &Ops) -> Result<()> {
        self.query_projection.run(s)?;
        self.weight_projection.run(blas)?;
        rope.rope_pair_tail(s, &mut TensorMut(self.query), self.angles)?;
        self.pack.run(s)?;
        if let Some(plan) = &self.full {
            plan.run(s)?;
        }
        if let Some(plan) = &self.candidates {
            plan.run(s)?;
        }
        if let Some(ticket) = &self.candidate_ticket {
            ticket.run(s, &runtime.ticket, i32::from(self.candidates.is_some()))?;
        }
        if let Some(plan) = &self.restricted {
            plan.run(s)?;
        }
        self.selector.select_scores(&self.input, &self.output, s)?;
        self.order.run(s)
    }
}

struct RowPlan {
    compressor: Option<Compressor>,
    indexer: Option<Indexer>,
    ticket: Ticket,
    attention: Vec<IndexedAttentionPlan>,
}

pub(super) struct Runtime {
    plans: Vec<Vec<RowPlan>>,
    begin: CudaFunction,
    commit: CudaFunction,
    ticket: CudaFunction,
    _scratch: Scratch,
    _weights: Weights,
}

struct Binding<'a> {
    device: &'a Device,
    blas: &'a Blas,
    mixer: &'a DeepseekMixer,
    state: &'a StateStore,
    work: &'a Work,
    scratch: &'a Scratch,
    weights: &'a Weights,
}

impl Runtime {
    pub(super) fn new(
        binding: (
            &Device,
            &Blas,
            &ModelSpec,
            &StateStore,
            &DeepseekMixer,
            &Work,
        ),
        weights: Weights,
    ) -> Result<Self> {
        let (device, blas, spec, state, mixer, work) = binding;
        let compressed = state
            .compressed
            .as_ref()
            .context("DSA state allocation missing")?;
        let capacity = compressed
            .kv
            .iter()
            .map(|v| v.codes.len() / 256)
            .max()
            .context("DSA KV owners absent")?;
        let scratch = Scratch::new(
            &device.stream,
            compressed.rows,
            capacity,
            &spec
                .deepseek_v41
                .as_ref()
                .context("DSA geometry missing")?
                .selection,
        )?;
        let bind = Binding {
            device,
            blas,
            mixer,
            state,
            work,
            scratch: &scratch,
            weights: &weights,
        };
        let mut plans = Vec::new();
        for (layer, mapping) in compressed.owners.iter().enumerate() {
            let mut layer_plans = Vec::new();
            if let Some(mapping) = mapping {
                for rows in 1..=compressed.rows {
                    layer_plans.push(bind.row(spec, layer, rows, *mapping)?);
                }
            }
            plans.push(layer_plans);
        }
        let module = crate::backend::cuda::cubin::module(&device.ctx, "latent_compress")?;
        Ok(Self {
            plans,
            begin: module.load_function("latent_publication_begin")?,
            commit: module.load_function("latent_publication_commit")?,
            ticket: module.load_function("latent_selection_ticket")?,
            _scratch: scratch,
            _weights: weights,
        })
    }
    #[cfg(test)]
    pub(super) fn isolate_published_attention(&mut self, layer: usize) -> Result<()> {
        let plans = self.plans.get_mut(layer).context("DSA layer unbound")?;
        ensure!(!plans.is_empty(), "DSA layer has no attention plans");
        for plan in plans {
            plan.compressor = None;
            plan.indexer = None;
        }
        Ok(())
    }

    pub(super) fn run(
        &self,
        mixer: &DeepseekMixer,
        s: &Arc<CudaStream>,
        blas: &Blas,
        rope: &Ops,
        layer: usize,
        rows: usize,
    ) -> Result<()> {
        let plan = self
            .plans
            .get(layer)
            .and_then(|p| p.get(rows.wrapping_sub(1)))
            .context("DSA row plan unbound")?;
        if let Some(compressor) = &plan.compressor {
            compressor.run(self, mixer, s, blas, rope)?;
        }
        plan.ticket.run(s, &self.ticket, 2)?;
        if let Some(indexer) = &plan.indexer {
            indexer.run(self, s, blas, rope)?;
            plan.ticket.run(s, &self.ticket, 1)?;
        }
        plan.ticket.run(s, &self.ticket, 0)
    }
    pub(super) fn attention(
        &self,
        s: &Arc<CudaStream>,
        layer: usize,
        rows: usize,
        row: usize,
    ) -> Result<()> {
        self.plans
            .get(layer)
            .and_then(|p| p.get(rows.wrapping_sub(1)))
            .and_then(|p| p.attention.get(row))
            .context("DSA attention row unbound")?
            .run(s)
    }
}

impl Binding<'_> {
    fn weight<T: Copy, const R: usize>(
        &self,
        layer: usize,
        name: &str,
        shape: [u32; R],
    ) -> Result<Tensor<T, R>> {
        let name = format!("layers.{layer}.attn.{name}");
        weight_view(
            self.weights
                .get(&name)
                .with_context(|| format!("DSA missing {name}"))?,
            &self.device.stream,
            shape,
            &name,
        )
    }
    fn row(
        &self,
        spec: &ModelSpec,
        layer: usize,
        rows: usize,
        owners: LatentOwners,
    ) -> Result<RowPlan> {
        let s = &self.device.stream;
        let state = self
            .state
            .compressed
            .as_ref()
            .context("DSA state missing")?;
        let kv = state
            .kv
            .iter()
            .find(|v| v.owner == owners.kv)
            .context("DSA KV owner missing")?;
        let index = state
            .index
            .iter()
            .find(|v| v.owner == owners.index)
            .context("DSA selection owner missing")?;
        let seq = self.state.seq_ref(s)?;
        let ticket = Ticket {
            sequence: seq,
            generation: state.generation.device_ptr(s).0,
            kv: kv.ticket.device_ptr(s).0,
            index: index.ticket.device_ptr(s).0,
            owners,
        };
        let compressor = if layer == owners.kv {
            Some(self.compressor(layer, rows, state, owners)?)
        } else {
            None
        };
        let indexer = if layer == owners.index {
            Some(self.indexer(spec, layer, rows, state, owners)?)
        } else {
            None
        };
        let g = &self.mixer.geometry;
        let ring = self.state.latent_ring(layer)?;
        let weights = self.mixer.weights[layer]
            .as_ref()
            .context("DSA attention weights missing")?;
        let mut attention = Vec::new();
        for row in 0..rows {
            attention.push(IndexedAttentionPlan::new(
                &self.device.ctx,
                IndexedInputs {
                    window: LwaInputs {
                        q: view(
                            &self.work.qkv,
                            s,
                            row * g.heads * g.head_dim,
                            [g.heads as u32, g.head_dim as u32],
                            [g.head_dim as u64, 1],
                        )?,
                        k_codes: dense(&ring.codes, s, [128, 512])?,
                        k_scales: dense(&ring.scales, s, [128, 16])?,
                        slot_ids: dense(&self.mixer.slots, s, [128])?,
                        sink: weight_view(&weights.attn_sink, s, [g.heads as u32], "sink")?,
                        inv_rotary: view(&self.mixer.rotary_inv, s, row * 64, [64], [1])?,
                    },
                    packed: dense(&kv.codes, s, [(kv.codes.len() / 256) as u32, 256])?,
                    scales: dense(&kv.scales, s, [(kv.codes.len() / 256) as u32, 32])?,
                    selected: dense(&index.positions, s, [rows as u32, 512])?,
                    committed: dense(&kv.count, s, [1])?,
                    sequence: seq,
                    ratio: owners.ratio as u32,
                },
                view(
                    &self.mixer.o,
                    s,
                    row * g.heads * g.head_dim,
                    [g.heads as u32, 512],
                    [512, 1],
                )?,
                row as u32,
            )?);
        }
        Ok(RowPlan {
            compressor,
            indexer,
            ticket,
            attention,
        })
    }
    fn compressor(
        &self,
        layer: usize,
        rows: usize,
        state: &LatentCompressedState,
        owners: LatentOwners,
    ) -> Result<Compressor> {
        let s = &self.device.stream;
        let ctx = &self.device.ctx;
        let a = self.scratch;
        let kv = state
            .kv
            .iter()
            .find(|v| v.owner == layer)
            .context("DSA compressor owner missing")?;
        ensure!(
            kv.ratio == owners.ratio,
            "DSA compressor owner ratio mismatch"
        );
        let groups = rows.div_ceil(owners.ratio) as u32;
        let seq = self.state.seq_ref(s)?;
        let group_seq = SeqRef {
            ptr: a.group_meta.device_ptr(s).0,
        };
        let x = dense(
            &self.work.n,
            s,
            [rows as u32, self.mixer.geometry.hidden as u32],
        )?;
        let pooled = dense(&a.pooled, s, [groups, 512])?;
        let weight = self.weight(layer, "compressor.wkv.weight", [512, x.shape[1]])?;
        let mut float_values = None;
        let mut float_gates = None;
        let mut plain_values = None;
        let mut pool = None;
        if owners.ratio == 2 {
            let values = dense(&a.values, s, [rows as u32, 512])?;
            let scores = dense(&a.gates, s, [rows as u32, 512])?;
            float_values = Some(Bf16F32Plan::new(ctx, x, weight, values)?);
            float_gates = Some(Bf16F32Plan::new(
                ctx,
                x,
                self.weight(layer, "compressor.wgate.weight", [512, x.shape[1]])?,
                scores,
            )?);
            pool = Some(LatentCompressPlan::new(
                ctx,
                CompressionInputs {
                    values,
                    scores,
                    sequence: seq,
                },
                CompressionBuffers {
                    tail_values: dense(&kv.tail_values, s, [2, 512])?,
                    tail_scores: dense(&kv.tail_scores, s, [2, 512])?,
                    cursor: dense(&kv.cursor, s, [1])?,
                    pooled,
                },
            )?);
        } else {
            plain_values = Some(Bf16Projection::new(self.blas, x, weight, pooled)?);
        }
        let key = dense(&a.key, s, [groups, 128])?;
        let norm = Norm {
            values: reshape(pooled, [groups, 1, 512])?,
            weight: self.weight(layer, "compressor.norm.weight", [512])?,
            sequence: group_seq,
        };
        let key_projection = Bf16Projection::new(
            self.blas,
            pooled,
            self.weight(layer, "indexer.wk.weight", [128, 512])?,
            key,
        )?;
        let key_norm = Norm {
            values: reshape(key, [groups, 1, 128])?,
            weight: self.weight(layer, "indexer.k_norm.weight", [128])?,
            sequence: group_seq,
        };
        let key_pack = Fp4PackPlan::new(
            ctx,
            Fp4PackViews {
                input: key,
                codes: dense(&a.key_codes, s, [groups, 64])?,
                scales: dense(&a.key_scales, s, [groups, 4])?,
                sequence: seq,
            },
            32,
            owners.ratio as u32,
        )?;
        let latent_pack = Fp4PackPlan::new(
            ctx,
            Fp4PackViews {
                input: pooled,
                codes: dense(&a.kv_codes, s, [groups, 256])?,
                scales: dense(&a.kv_scales, s, [groups, 32])?,
                sequence: seq,
            },
            16,
            owners.ratio as u32,
        )?;
        let publication = Publication {
            sequence: seq,
            generation: state.generation.device_ptr(s).0,
            ticket: kv.ticket.device_ptr(s).0,
            cursor: kv.cursor.device_ptr(s).0,
            count: kv.count.device_ptr(s).0,
            groups: group_seq.ptr,
            frequencies: self.mixer.inv_freq[layer].device_ptr(s).0,
            angles: a.group_angles.device_ptr(s).0,
            payload: [
                a.kv_codes.device_ptr(s).0,
                a.kv_scales.device_ptr(s).0,
                a.key_codes.device_ptr(s).0,
                a.key_scales.device_ptr(s).0,
            ],
            cache: [
                kv.codes.device_ptr(s).0,
                kv.scales.device_ptr(s).0,
                kv.index_codes.device_ptr(s).0,
                kv.index_scales.device_ptr(s).0,
            ],
            owner: layer as i32,
            ratio: owners.ratio as i32,
            rows: rows as i32,
            capacity: (kv.codes.len() / 256) as i32,
        };
        Ok(Compressor {
            publication,
            float_values,
            float_gates,
            plain_values,
            pool,
            norm,
            key_projection,
            key_norm,
            key: reshape(key, [groups, 1, 128])?,
            latent: reshape(pooled, [groups, 1, 512])?,
            angles: dense(&a.group_angles, s, [groups, 64])?,
            key_pack,
            latent_pack,
        })
    }
    fn indexer(
        &self,
        spec: &ModelSpec,
        layer: usize,
        rows: usize,
        state: &LatentCompressedState,
        owners: LatentOwners,
    ) -> Result<Indexer> {
        let s = &self.device.stream;
        let ctx = &self.device.ctx;
        let a = self.scratch;
        let kv = state
            .kv
            .iter()
            .find(|v| v.owner == owners.kv)
            .context("DSA index KV owner missing")?;
        let index = state
            .index
            .iter()
            .find(|v| v.owner == layer)
            .context("DSA index owner missing")?;
        let capacity = (kv.codes.len() / 256) as u32;
        let rows = rows as u32;
        let ds = spec.deepseek_v41.as_ref().context("DSA geometry missing")?;
        let seq = self.state.seq_ref(s)?;
        let qrank = self.mixer.geometry.q_lora as u32;
        let query = dense(&a.query, s, [rows, 4096])?;
        let query_projection = self.blas.fp8_ops()?.bind_block(
            LinearWeightRef::Fp8 {
                codes: self.weight(layer, "indexer.wq_b.weight", [4096, qrank])?,
                scale: Fp8ScaleRef::E8m0(self.weight(
                    layer,
                    "indexer.wq_b.scale",
                    [128, qrank / 32],
                )?),
                layout: self.mixer.layout,
            },
            dense(&self.mixer.act_lora_codes, s, [rows, qrank])?,
            Fp8ScaleRef::E8m0(dense(&self.mixer.act_lora_scales, s, [rows, qrank / 32])?),
            &TensorMut(query),
        )?;
        let weight_projection = Bf16Projection::new(
            self.blas,
            dense(&self.work.n, s, [rows, self.mixer.geometry.hidden as u32])?,
            self.weight(
                layer,
                "indexer.weights_proj.weight",
                [32, self.mixer.geometry.hidden as u32],
            )?,
            dense(&a.head_weights, s, [rows, 32])?,
        )?;
        let pack = Fp4PackPlan::new(
            ctx,
            Fp4PackViews {
                input: query,
                codes: dense(&a.query_codes, s, [rows, 2048])?,
                scales: dense(&a.query_scales, s, [rows, 128])?,
                sequence: seq,
            },
            32,
            1,
        )?;
        let score = |restricted: bool| -> Result<IndexScoreViews> {
            let (scores, ids, cols) = if restricted {
                (&a.restricted_scores, &a.restricted_ids, 16384)
            } else {
                (&a.full_scores, &a.full_ids, capacity)
            };
            Ok(IndexScoreViews {
                query: dense(&a.query_codes, s, [rows, 32, 64])?,
                query_scales: dense(&a.query_scales, s, [rows, 32, 4])?,
                keys: dense(&kv.index_codes, s, [capacity, 64])?,
                key_scales: dense(&kv.index_scales, s, [capacity, 4])?,
                weights: dense(&a.head_weights, s, [rows, 32])?,
                committed: dense(&kv.count, s, [1])?,
                scores: dense(scores, s, [rows, cols])?,
                positions: dense(ids, s, [rows, cols])?,
                sequence: seq,
                candidates: if restricted {
                    Some(dense(&state.candidates, s, [rows, 2048])?)
                } else {
                    None
                },
            })
        };
        let candidate_owner = ds.candidate_source_layer_id as usize;
        let full = if layer <= candidate_owner {
            Some(IndexScorePlan::new(
                ctx,
                score(false)?,
                owners.ratio as u32,
            )?)
        } else {
            None
        };
        let candidates = if layer == candidate_owner {
            Some(IndexCandidatesPlan::new(
                ctx,
                IndexCandidatesViews {
                    scores: dense(&a.full_scores, s, [rows, capacity])?,
                    block_scores: dense(
                        &a.block_scores,
                        s,
                        [rows, capacity.div_ceil(ds.selection.candidate_block_size)],
                    )?,
                    block_ids: dense(
                        &a.block_ids,
                        s,
                        [rows, capacity.div_ceil(ds.selection.candidate_block_size)],
                    )?,
                    selected_scores: dense(
                        &a.block_selected_scores,
                        s,
                        [rows, ds.selection.candidate_topk_blocks - 1],
                    )?,
                    selected_ids: dense(
                        &a.block_selected_ids,
                        s,
                        [rows, ds.selection.candidate_topk_blocks - 1],
                    )?,
                    margins: dense(&a.margins, s, [rows])?,
                    candidates: dense(
                        &state.candidates,
                        s,
                        [rows, ds.selection.candidate_topk_blocks],
                    )?,
                    sequence: seq,
                },
                owners.ratio as u32,
            )?)
        } else {
            None
        };
        let candidate_ticket = if layer >= candidate_owner {
            Some(Ticket {
                sequence: seq,
                generation: state.generation.device_ptr(s).0,
                kv: kv.ticket.device_ptr(s).0,
                index: state.candidate_ticket.device_ptr(s).0,
                owners: LatentOwners {
                    kv: owners.kv,
                    index: candidate_owner,
                    ratio: owners.ratio,
                },
            })
        } else {
            None
        };
        let restricted = if layer >= candidate_owner {
            Some(IndexScorePlan::new(ctx, score(true)?, owners.ratio as u32)?)
        } else {
            None
        };
        let values = score(restricted.is_some())?;
        let selector =
            DsaSelectionPlan::new(ctx, rows, values.scores.shape[1], ds.selection.index_topk)?;
        let input = DsaTopKInput {
            scores: values.scores,
            ids: values.positions,
        };
        let output = DsaTopKOutput {
            scores: TensorMut(dense(
                &a.selected_scores,
                s,
                [rows, ds.selection.index_topk],
            )?),
            ids: TensorMut(dense(&a.selected_ids, s, [rows, ds.selection.index_topk])?),
            margins: TensorMut(dense(&a.margins, s, [rows])?),
        };
        let order = IndexPositionOrder::new(
            ctx,
            output.ids.0,
            dense(&index.positions, s, [rows, ds.selection.index_topk])?,
        )?;
        Ok(Indexer {
            query_projection,
            weight_projection,
            query: reshape(query, [rows, 32, 128])?,
            angles: dense(&self.mixer.rotary_fwd, s, [rows, 64])?,
            pack,
            full,
            candidates,
            candidate_ticket,
            restricted,
            selector,
            input,
            output,
            order,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Deserialize)]
    struct Pool {
        owner: usize,
        ratio: usize,
        values: Vec<f32>,
        scores: Vec<f32>,
        pooled: Vec<u16>,
        bound: f64,
    }
    #[derive(serde::Deserialize)]
    struct Codec {
        owner: usize,
        kind: String,
        codes: Vec<u8>,
        scales: Vec<u8>,
    }
    #[derive(serde::Deserialize)]
    struct Index {
        owner: usize,
        keys: Vec<u8>,
        key_scales: Vec<u8>,
    }
    #[derive(serde::Deserialize)]
    struct Record {
        reference_only: bool,
        candidate_seen: bool,
        compressions: Vec<Pool>,
        codecs: Vec<Codec>,
        indexes: Vec<Index>,
    }

    fn spec() -> Result<ModelSpec> {
        ModelSpec::deepseek_v41(
            &serde_json::from_str(include_str!(
                "../../../tests/data/deepseek41-flash-config.json"
            ))?,
            &serde_json::Value::Null,
        )
    }

    struct PublicationFixture<'a> {
        record: &'a Record,
        owner: usize,
        width: usize,
        graph: bool,
    }
    fn publication_case(
        ctx: &Arc<CudaContext>,
        fixture: PublicationFixture<'_>,
    ) -> Result<(usize, usize)> {
        let PublicationFixture {
            record,
            owner,
            width,
            graph,
        } = fixture;
        let s = ctx.new_stream()?;
        let spec = spec()?;
        let mut state = LatentCompressedState::new(&spec, 136, width, &s)?;
        let pool = record
            .compressions
            .iter()
            .find(|v| v.owner == owner)
            .context("pool fixture absent")?;
        let codec = record
            .codecs
            .iter()
            .find(|v| v.owner == owner && v.kind == "compressed_kv")
            .context("codec fixture absent")?;
        let keys = record
            .indexes
            .iter()
            .find(|v| v.owner == owner)
            .context("index fixture absent")?;
        let mut metadata = s.clone_htod(&[0i32, 1, 1, 136, 1, 0, 0, 0])?;
        let seq = SeqRef {
            ptr: metadata.device_ptr(&s).0,
        };
        let mut scratch = Scratch::new(
            &s,
            width,
            136,
            &crate::model_spec::DeepseekSelectionShape::IMPLEMENTED,
        )?;
        let frequency = s.clone_htod(&rotary_inv_freq(
            &spec,
            spec.deepseek_v41.as_ref().unwrap(),
            pool.ratio as u8,
        )?)?;
        let kv = state
            .kv
            .iter()
            .find(|v| v.owner == owner)
            .context("owner absent")?;
        let module = crate::backend::cuda::cubin::module(ctx, "latent_compress")?;
        let begin = module.load_function("latent_publication_begin")?;
        let commit = module.load_function("latent_publication_commit")?;
        let ticket_kernel = module.load_function("latent_selection_ticket")?;
        let p = Publication {
            sequence: seq,
            generation: state.generation.device_ptr(&s).0,
            ticket: kv.ticket.device_ptr(&s).0,
            cursor: kv.cursor.device_ptr(&s).0,
            count: kv.count.device_ptr(&s).0,
            groups: scratch.group_meta.device_ptr(&s).0,
            frequencies: frequency.device_ptr(&s).0,
            angles: scratch.group_angles.device_ptr(&s).0,
            payload: [
                scratch.kv_codes.device_ptr(&s).0,
                scratch.kv_scales.device_ptr(&s).0,
                scratch.key_codes.device_ptr(&s).0,
                scratch.key_scales.device_ptr(&s).0,
            ],
            cache: [
                kv.codes.device_ptr(&s).0,
                kv.scales.device_ptr(&s).0,
                kv.index_codes.device_ptr(&s).0,
                kv.index_scales.device_ptr(&s).0,
            ],
            owner: owner as i32,
            ratio: pool.ratio as i32,
            rows: width as i32,
            capacity: (kv.codes.len() / 256) as i32,
        };
        let compress = if pool.ratio == 2 {
            Some(LatentCompressPlan::new(
                ctx,
                CompressionInputs {
                    values: dense(&scratch.values, &s, [width as u32, 512])?,
                    scores: dense(&scratch.gates, &s, [width as u32, 512])?,
                    sequence: seq,
                },
                CompressionBuffers {
                    tail_values: dense(&kv.tail_values, &s, [2, 512])?,
                    tail_scores: dense(&kv.tail_scores, &s, [2, 512])?,
                    cursor: dense(&kv.cursor, &s, [1])?,
                    pooled: dense(&scratch.pooled, &s, [width.div_ceil(2) as u32, 512])?,
                },
            )?)
        } else {
            None
        };
        let tickets = state
            .index
            .iter()
            .filter(|v| state.owners[v.owner].is_some_and(|m| m.kv == owner))
            .map(|v| Ticket {
                sequence: seq,
                generation: p.generation,
                kv: p.ticket,
                index: v.ticket.device_ptr(&s).0,
                owners: LatentOwners {
                    kv: owner,
                    index: v.owner,
                    ratio: pool.ratio,
                },
            })
            .collect::<Vec<_>>();
        let launch = || -> Result<()> {
            p.begin(&s, &begin)?;
            if let Some(compress) = &compress {
                compress.run(&s)?;
            }
            p.commit(&s, &commit)?;
            for ticket in &tickets {
                ticket.run(&s, &ticket_kernel, 1)?;
                ticket.run(&s, &ticket_kernel, 0)?;
            }
            Ok(())
        };
        let captured = if graph {
            s.begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )?;
            launch()?;
            Some(
                s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))?
                    .context("publication graph absent")?,
            )
        } else {
            None
        };
        let mut cases = 0;
        let mut defects = 0;
        for repeat in 0..2 {
            state.reset(&s)?;
            for position in (0..129).step_by(width) {
                let rows = width.min(129 - position);
                let first = position / pool.ratio;
                let end = (position + rows) / pool.ratio;
                s.memcpy_htod(
                    &[
                        position as i32,
                        rows as i32,
                        (position + rows) as i32,
                        136,
                        1,
                        0,
                        0,
                        0,
                    ],
                    &mut metadata,
                )?;
                let mut values = vec![0.0f32; width * 512];
                values[..rows * 512]
                    .copy_from_slice(&pool.values[position * 512..(position + rows) * 512]);
                s.memcpy_htod(&values, &mut scratch.values)?;
                if pool.ratio == 2 {
                    let mut scores = vec![0.0f32; width * 512];
                    scores[..rows * 512]
                        .copy_from_slice(&pool.scores[position * 512..(position + rows) * 512]);
                    s.memcpy_htod(&scores, &mut scratch.gates)?;
                }
                for (source, destination, stride) in [
                    (&codec.codes, &mut scratch.kv_codes, 256),
                    (&codec.scales, &mut scratch.kv_scales, 32),
                    (&keys.keys, &mut scratch.key_codes, 64),
                    (&keys.key_scales, &mut scratch.key_scales, 4),
                ] {
                    let mut bytes = vec![0; destination.len()];
                    bytes[..(end - first) * stride]
                        .copy_from_slice(&source[first * stride..end * stride]);
                    s.memcpy_htod(&bytes, destination)?;
                }
                if let Some(graph) = &captured {
                    graph.launch()?;
                } else {
                    launch()?;
                }
                let kv = state.kv.iter().find(|v| v.owner == owner).unwrap();
                ensure!(
                    s.clone_dtoh(&kv.count)? == [end as u32]
                        && s.clone_dtoh(&kv.cursor)? == [(position + rows) as u32],
                    "publication count/cursor DEFECT"
                );
                let generation = s.clone_dtoh(&state.generation)?[0];
                ensure!(
                    s.clone_dtoh(&kv.ticket)?
                        == [
                            owner as u32,
                            generation,
                            (position + rows) as u32,
                            end as u32
                        ],
                    "publication ticket DEFECT"
                );
                for (expected, actual, stride) in [
                    (&codec.codes, &kv.codes, 256),
                    (&codec.scales, &kv.scales, 32),
                    (&keys.keys, &kv.index_codes, 64),
                    (&keys.key_scales, &kv.index_scales, 4),
                ] {
                    let got = s.clone_dtoh(actual)?;
                    ensure!(
                        got[..end * stride] == expected[..end * stride],
                        "published native cache bytes DEFECT owner{owner} position{position}"
                    );
                    ensure!(
                        got[end * stride..].iter().all(|&v| v == 0) || repeat > 0,
                        "publication wrote beyond committed capacity"
                    );
                }
                if pool.ratio == 2 {
                    let got = s.clone_dtoh(&scratch.pooled)?;
                    for (value, want) in got[..(end - first) * 512]
                        .iter()
                        .zip(&pool.pooled[first * 512..end * 512])
                    {
                        defects += usize::from(
                            f64::from((value.to_f32() - bf16::from_bits(*want).to_f32()).abs())
                                > pool.bound,
                        );
                    }
                    for (tail, values) in [
                        (&kv.tail_values, &pool.values),
                        (&kv.tail_scores, &pool.scores),
                    ] {
                        let actual = s.clone_dtoh(tail)?;
                        for parity in 0..2 {
                            let last = (0..position + rows).rev().find(|p| p % 2 == parity);
                            if let Some(last) = last {
                                ensure!(
                                    actual[parity * 512..(parity + 1) * 512]
                                        == values[last * 512..(last + 1) * 512],
                                    "publication tail DEFECT"
                                );
                            }
                        }
                    }
                }
                cases += 1;
            }
        }
        println!(
            "DSA_PUBLICATION owner={owner} width={width} graph={graph} cases={cases} defects={defects} reset_repeats=2"
        );
        Ok((cases, defects))
    }

    fn trap_control(kind: &str) -> Result<()> {
        let device = Device::new(0)?;
        let s = &device.stream;
        let spec = spec()?;
        let mut state = LatentCompressedState::new(&spec, 136, 1, s)?;
        let mut meta = s.clone_htod(&[2i32, 1, 3, 136, 1, 0, 0, 0])?;
        let seq = SeqRef {
            ptr: meta.device_ptr(s).0,
        };
        let module = crate::backend::cuda::cubin::module(&device.ctx, "latent_compress")?;
        let kv = &mut state.kv[0];
        s.memcpy_htod(&[2, 1, 3, 1], &mut kv.ticket)?;
        s.memcpy_htod(&[1], &mut kv.count)?;
        s.memcpy_htod(&[3], &mut kv.cursor)?;
        let index = &mut state.index[0];
        s.memcpy_htod(&[2, 2, 1, 2, 3], &mut index.ticket)?;
        println!("DSA_CONTROL_ARMED kind={kind}");
        if kind == "stale-cursor" {
            s.memcpy_htod(&[2, 1, 2, 1], &mut kv.ticket)?;
            s.memcpy_htod(&[2, 1, 3, 136, 1, 0, 0, 0], &mut meta)?;
            let scratch = Scratch::new(
                s,
                1,
                136,
                &crate::model_spec::DeepseekSelectionShape::IMPLEMENTED,
            )?;
            let frequencies = s.clone_htod(&[1.0f32; 32])?;
            let p = Publication {
                sequence: seq,
                generation: state.generation.device_ptr(s).0,
                ticket: kv.ticket.device_ptr(s).0,
                cursor: kv.cursor.device_ptr(s).0,
                count: kv.count.device_ptr(s).0,
                groups: scratch.group_meta.device_ptr(s).0,
                frequencies: frequencies.device_ptr(s).0,
                angles: scratch.group_angles.device_ptr(s).0,
                payload: [0; 4],
                cache: [0; 4],
                owner: 2,
                ratio: 2,
                rows: 1,
                capacity: 68,
            };
            p.begin(s, &module.load_function("latent_publication_begin")?)?;
        } else {
            let owners = match kind {
                "wrong-kv-owner" => LatentOwners {
                    kv: 8,
                    index: 2,
                    ratio: 2,
                },
                "wrong-index-owner" => LatentOwners {
                    kv: 2,
                    index: 8,
                    ratio: 2,
                },
                "stale-generation" => {
                    s.memcpy_htod(&[2], &mut state.generation)?;
                    LatentOwners {
                        kv: 2,
                        index: 2,
                        ratio: 2,
                    }
                }
                _ => anyhow::bail!("unknown DSA control"),
            };
            Ticket {
                sequence: seq,
                generation: state.generation.device_ptr(s).0,
                kv: kv.ticket.device_ptr(s).0,
                index: index.ticket.device_ptr(s).0,
                owners,
            }
            .run(s, &module.load_function("latent_selection_ticket")?, 0)?;
        }
        s.synchronize()?;
        anyhow::bail!("DSA control escaped device validation")
    }

    #[test]
    #[ignore]
    fn deepseek_dsa_publication_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_DS41_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        if let Ok(kind) = std::env::var("D1_DSA_CONTROL") {
            return trap_control(&kind);
        }
        let record: Record =
            serde_json::from_slice(&std::fs::read(std::env::var("D1_OP_REFERENCE")?)?)?;
        ensure!(
            record.reference_only && !record.candidate_seen,
            "publication fixture not reference-only"
        );
        let ctx = CudaContext::new(0)?;
        let mut cases = 0;
        let mut defects = 0;
        for owner in [2, 8, 14, 20] {
            for width in [1, 5, 8] {
                for graph in [false, true] {
                    let (n, bad) = publication_case(
                        &ctx,
                        PublicationFixture {
                            record: &record,
                            owner,
                            width,
                            graph,
                        },
                    )?;
                    cases += n;
                    defects += bad;
                }
            }
        }
        for kind in [
            "wrong-kv-owner",
            "wrong-index-owner",
            "stale-cursor",
            "stale-generation",
        ] {
            let output = std::process::Command::new(std::env::current_exe()?)
                .args([
                    "backend::cuda::engine::deepseek::dsa::tests::deepseek_dsa_publication_gpu",
                    "--exact",
                    "--include-ignored",
                    "--test-threads=1",
                    "--nocapture",
                ])
                .env("D1_DSA_CONTROL", kind)
                .output()?;
            let raw = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            println!("DSA_CONTROL {kind} rc={:?}\n{raw}", output.status.code());
            ensure!(
                !output.status.success()
                    && raw.contains("DSA_CONTROL_ARMED")
                    && raw.contains("CUDA_ERROR_LAUNCH_FAILED"),
                "DSA control failed validity or escaped: {kind}"
            );
        }
        println!("DSA_PUBLICATION_RESULT cases={cases} defects={defects} controls=4");
        ensure!(defects == 0, "DSA publication DEFECT");
        Ok(())
    }
}

#[cfg(test)]
mod live_tests {
    use super::*;
    #[derive(serde::Deserialize)]
    struct LayerInput {
        layer: usize,
        input: Vec<u16>,
    }
    #[derive(serde::Deserialize)]
    struct Inputs {
        reference_only: bool,
        rows: usize,
        width: usize,
        layers: Vec<LayerInput>,
    }

    fn mixer_weights(
        loader: &mut crate::backend::cuda::weights::Loader,
        layer: usize,
        g: &MixerGeometry,
    ) -> Result<MixerWeights> {
        let p = format!("layers.{layer}.attn");
        let fp8 = |loader: &mut crate::backend::cuda::weights::Loader,
                   name: &str,
                   n: usize,
                   k: usize|
         -> Result<Fp8Projection> {
            Ok(Fp8Projection {
                codes: loader.load_bytes(
                    &format!("{p}.{name}.weight"),
                    &[n, k],
                    safetensors::Dtype::F8_E4M3,
                )?,
                scale: loader.load_bytes(
                    &format!("{p}.{name}.scale"),
                    &[n.div_ceil(32), k.div_ceil(32)],
                    safetensors::Dtype::F8_E8M0,
                )?,
            })
        };
        Ok(MixerWeights {
            wq_a: fp8(loader, "wq_a", g.q_lora, g.hidden)?,
            wq_b: fp8(loader, "wq_b", g.heads * g.head_dim, g.q_lora)?,
            wkv: fp8(loader, "wkv", g.head_dim, g.hidden)?,
            wo_b: fp8(loader, "wo_b", g.hidden, g.o_groups * g.o_rank)?,
            q_norm: loader.load_bytes(
                &format!("{p}.q_norm.weight"),
                &[g.q_lora],
                safetensors::Dtype::BF16,
            )?,
            kv_norm: loader.load_bytes(
                &format!("{p}.kv_norm.weight"),
                &[g.head_dim],
                safetensors::Dtype::BF16,
            )?,
            attn_sink: loader.load_bytes(
                &format!("{p}.attn_sink"),
                &[g.heads],
                safetensors::Dtype::F32,
            )?,
            wo_a: loader.load_bytes(
                &format!("{p}.wo_a.weight"),
                &[g.o_groups * g.o_rank, g.o_in],
                safetensors::Dtype::F8_E4M3,
            )?,
            wo_a_scale: loader.load_bytes(
                &format!("{p}.wo_a.scale"),
                &[g.o_groups * g.o_rank / 32, g.o_in / 32],
                safetensors::Dtype::F8_E8M0,
            )?,
        })
    }

    fn final_state(
        state: &LatentCompressedState,
        s: &Arc<CudaStream>,
        last_rows: usize,
    ) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        for kv in &state.kv {
            let count = s.clone_dtoh(&kv.count)?[0] as usize;
            for (data, width) in [
                (&kv.codes, 256),
                (&kv.scales, 32),
                (&kv.index_codes, 64),
                (&kv.index_scales, 4),
            ] {
                bytes.extend(s.clone_dtoh(&data.slice(..count * width))?);
            }
            for data in [&kv.tail_values, &kv.tail_scores] {
                for value in s.clone_dtoh(data)? {
                    bytes.extend(value.to_bits().to_le_bytes());
                }
            }
            ensure!(
                s.clone_dtoh(&kv.cursor)? == [129],
                "live owner cursor missed a token"
            );
            ensure!(
                count == 129 / kv.ratio,
                "live owner committed count differs"
            );
        }
        for selection in &state.index {
            for value in s.clone_dtoh(
                &selection
                    .positions
                    .slice((last_rows - 1) * 512..last_rows * 512),
            )? {
                bytes.extend(value.to_le_bytes());
            }
        }
        Ok(bytes)
    }

    #[test]
    #[ignore]
    fn deepseek_dsa_live_chain_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_DS41_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let dir = std::env::var_os("F3_MODEL").context("F3_MODEL required")?;
        let input: Inputs =
            serde_json::from_slice(&std::fs::read(std::env::var("D1_LIVE_INPUTS")?)?)?;
        ensure!(
            input.reference_only && input.rows == 129 && input.width == 5120,
            "live DSA fixture shape"
        );
        let expected = [2, 3, 8, 9, 14, 15, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37];
        ensure!(
            input.layers.iter().map(|v| v.layer).eq(expected),
            "live DSA owner/consumer fixture coverage"
        );
        ensure!(
            input.layers.iter().all(|v| v.input.len() == 129 * 5120),
            "live DSA input extent"
        );
        let root: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/data/deepseek41-flash-config.json"
        ))?;
        let spec = ModelSpec::deepseek_v41(&root, &serde_json::Value::Null)?;
        let device = Device::new(0)?;
        let s = &device.stream;
        let ops = Ops::new_with_swiglu_rounding(&device.ctx, spec.swiglu_rounding())?;
        let runtime = PathBuf::from(
            std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("runtime directory required")?,
        );
        let mut blas = Blas::new(
            &device,
            &ops,
            &runtime,
            Rc::new(crate::backend::cuda::closure::Closure::new(
                false,
                "DSA live gate".into(),
            )),
        )?;
        blas.enable_fp8(&device.ctx)?;
        let mut state = StateStore::new(&spec, 136, s)?;
        state.compressed = Some(LatentCompressedState::new(&spec, 136, 8, s)?);
        let mut mixer = DeepseekMixer::new(&device.ctx, s, &spec, 40, 8)?;
        let mut work = Work::new(s, &spec, 8)?;
        let checkpoint = crate::checkpoint::MappedCheckpoint::open(Path::new(&dir))?;
        let manifest = checkpoint.manifest(&spec, crate::checkpoint::LoadPolicy::TextOnly)?;
        let weights = crate::backend::cuda::weights::checkpoint_mapped(
            &checkpoint,
            &device,
            Some(&manifest),
            false,
            |loader| {
                let weights = load_weights(loader, &spec)?;
                for layer in 0..40 {
                    mixer.bind_layer(layer, mixer_weights(loader, layer, &mixer.geometry)?)?;
                }
                for (name, entry) in &manifest.entries {
                    if entry.load
                        && entry.scope == crate::checkpoint::TensorScope::Text
                        && !name.contains(".attn.")
                    {
                        loader.defer(name)?;
                    }
                }
                Ok(weights)
            },
        )?
        .0;
        mixer.dsa = Some(Runtime::new(
            (&device, &blas, &spec, &state, &mixer, &work),
            weights,
        )?);
        let mut baseline: Option<(Vec<Vec<u16>>, Vec<u8>)> = None;
        let mut defects = 0;
        let mut cases = 0;
        for width in [1, 5, 8] {
            for repeat in 0..2 {
                state.reset(s)?;
                let mut outputs = vec![Vec::<u16>::new(); input.layers.len()];
                for position in (0..129).step_by(width) {
                    let rows = width.min(129 - position);
                    state.set_sequence(s, position as u32, rows as u32, 8)?;
                    for (slot, layer) in input.layers.iter().enumerate() {
                        let x = layer.input[position * 5120..(position + rows) * 5120]
                            .iter()
                            .copied()
                            .map(bf16::from_bits)
                            .collect::<Vec<_>>();
                        s.memcpy_htod(&x, &mut work.n.slice_mut(..rows * 5120))?;
                        mixer.run(
                            s,
                            &blas,
                            &ops,
                            state.latent_ring(layer.layer)?,
                            MixerStep {
                                position,
                                rows,
                                x: &work.n,
                                qkv: &mut work.qkv,
                                out: &mut work.out,
                                seq: state.seq_ref(s)?,
                            },
                            layer.layer,
                        )?;
                        let result = s.clone_dtoh(&work.out.slice(..rows * 5120))?;
                        ensure!(
                            result.iter().all(|v| v.is_finite()),
                            "live DSA nonfinite model output"
                        );
                        outputs[slot].extend(result.iter().map(|v| v.to_bits()));
                        cases += 1;
                    }
                }
                let state_bytes =
                    final_state(state.compressed.as_ref().unwrap(), s, 1 + (129 - 1) % width)?;
                if let Some((base, state_base)) = &baseline {
                    let output_bad = outputs
                        .iter()
                        .flatten()
                        .zip(base.iter().flatten())
                        .filter(|(a, b)| a != b)
                        .count();
                    let state_bad = state_bytes
                        .iter()
                        .zip(state_base)
                        .filter(|(a, b)| a != b)
                        .count();
                    defects += output_bad + state_bad;
                    println!(
                        "DSA_LIVE_CHAIN width={width} repeat={repeat} output_bits_differ={output_bad} state_bytes_differ={state_bad}"
                    );
                } else {
                    baseline = Some((outputs, state_bytes));
                }
            }
        }
        println!("DSA_LIVE_CHAIN_RESULT cases={cases} owners=8 consumers=8 defects={defects}");
        ensure!(defects == 0, "live DSA chunk/reset invariance DEFECT");
        Ok(())
    }
}
