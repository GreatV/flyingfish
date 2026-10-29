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

pub(super) fn scratch_bytes(rows: usize, capacity: usize) -> usize {
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
            + 2048
            + 128
            + capacity * 8
            + 16384 * 8
            + capacity.div_ceil(8) * 8
            + 2047 * 8
            + 512 * 8
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
                mode: deepseek_rms_mode(),
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
    fn new(s: &Arc<CudaStream>, rows: usize, capacity: usize) -> Result<Self> {
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
            block_scores: s.alloc_zeros(rows * capacity.div_ceil(8))?,
            block_ids: s.alloc_zeros(rows * capacity.div_ceil(8))?,
            block_selected_scores: s.alloc_zeros(rows * 2047)?,
            block_selected_ids: s.alloc_zeros(rows * 2047)?,
            selected_scores: s.alloc_zeros(rows * 512)?,
            selected_ids: s.alloc_zeros(rows * 512)?,
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
        let scratch = Scratch::new(&device.stream, compressed.rows, capacity)?;
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
        let candidate_owner = spec
            .deepseek_v41
            .as_ref()
            .context("DSA geometry missing")?
            .candidate_source_layer_id as usize;
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
                    block_scores: dense(&a.block_scores, s, [rows, capacity.div_ceil(8)])?,
                    block_ids: dense(&a.block_ids, s, [rows, capacity.div_ceil(8)])?,
                    selected_scores: dense(&a.block_selected_scores, s, [rows, 2047])?,
                    selected_ids: dense(&a.block_selected_ids, s, [rows, 2047])?,
                    margins: dense(&a.margins, s, [rows])?,
                    candidates: dense(&state.candidates, s, [rows, 2048])?,
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
        let selector = DsaSelectionPlan::new(ctx, rows, values.scores.shape[1], 512)?;
        let input = DsaTopKInput {
            scores: values.scores,
            ids: values.positions,
        };
        let output = DsaTopKOutput {
            scores: TensorMut(dense(&a.selected_scores, s, [rows, 512])?),
            ids: TensorMut(dense(&a.selected_ids, s, [rows, 512])?),
            margins: TensorMut(dense(&a.margins, s, [rows])?),
        };
        let order =
            IndexPositionOrder::new(ctx, output.ids.0, dense(&index.positions, s, [rows, 512])?)?;
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
