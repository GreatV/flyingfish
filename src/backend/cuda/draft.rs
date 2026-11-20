use super::{
    Device,
    blas::Blas,
    copy::{Rect, columns},
    ops::{Ops, flat, grid},
    verification::Shape,
    verification::Verification,
    weights::{Layer, checkpoint},
};
use super::{
    attention_d256::{
        AttentionBindings, FullAttentionOps, FullAttentionPlan, MASK_LEN, PositionBindings,
    },
    contracts::{Scratch, SeqRef},
};
use super::{
    contracts::{TensorMut, view},
    ops::{CommonOpsRef, RmsMode},
};
use crate::backend::gpu::DevicePtr;
use crate::backend::gpu::{CudaSlice, CudaStream, DevicePtrMut, PushKernelArg};
use crate::{config::Config, dspark::DraftConfig, trace::Trace};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::{path::Path, sync::Arc};

pub struct Compute<'a> {
    pub ops: &'a Ops,
    pub blas: &'a mut Blas,
}

enum Output {
    Tokens,
    Device,
    Base,
}

enum Attention {
    Cached(Verification),
    Block(BlockAttention),
}

struct BlockAttention {
    layers: Vec<FullAttentionPlan>,
    metadata: CudaSlice<i32>,
    positions: CudaSlice<i32>,
    mask: CudaSlice<u64>,
    scratch: CudaSlice<u8>,
    prepare: crate::backend::gpu::CudaFunction,
}

impl BlockAttention {
    fn new(draft: &Dense, device: &Device) -> Result<Self> {
        use crate::op_contract::{AttentionLimits, FullAttentionSpec};
        let c = &draft.config;
        let s = &device.stream;
        let rows = u32::try_from(c.block_size)?;
        let groups = c.num_key_value_heads
            * ((c.num_attention_heads / c.num_key_value_heads) * c.block_size).div_ceil(64);
        let partitions = device.info.sms.div_ceil(groups).max(1);
        let key_chunk = draft.capacity.div_ceil(partitions).next_multiple_of(32);
        let limits = AttentionLimits {
            max_rows: rows,
            capacity: u32::try_from(draft.capacity)?,
            key_chunk: u32::try_from(key_chunk)?,
            causal: 0,
        };
        let spec = FullAttentionSpec {
            hq: u32::try_from(c.num_attention_heads)?,
            hkv: u32::try_from(c.num_key_value_heads)?,
            d: u32::try_from(c.head_dim)?,
            rotary_dim: u32::try_from(c.head_dim)?,
            scale: (c.head_dim as f32).sqrt().recip(),
            rms_eps: c.rms_norm_eps,
            rope_theta: c.rope_parameters.rope_theta,
        };
        let request = FullAttentionOps::workspace(&spec, limits)?;
        let live = if rows == MASK_LEN {
            u64::MAX
        } else {
            (1u64 << rows) - 1
        };
        let mask = s.clone_htod(&std::array::from_fn::<_, { MASK_LEN as usize }, _>(|row| {
            if row < c.block_size { live } else { 0 }
        }))?;
        let positions = s.alloc_zeros::<i32>(c.block_size * 3)?;
        let metadata = s.alloc_zeros::<i32>(8)?;
        let scratch = s.alloc_zeros::<u8>(usize::try_from(request.bytes)?)?;
        let common = CommonOpsRef::new(&device.ctx)?;
        let width = c.num_attention_heads * c.head_dim;
        let kv_width = c.kv_dim();
        let head = c.head_dim as u32;
        let q_shape = [rows, spec.hq, head];
        let q_stride = [width as u64, u64::from(head), 1];
        let kv_shape = [rows, spec.hkv, head];
        let kv_stride = [c.qkv_dim() as u64, u64::from(head), 1];
        let cache_shape = [spec.hkv, limits.capacity, head];
        let cache_stride = [(draft.capacity * c.head_dim) as u64, u64::from(head), 1];
        let mut layers = Vec::new();
        for (layer, cache) in draft.weights.layers.iter().zip(&draft.kv) {
            let bindings = AttentionBindings {
                compact: None,
                q_gate: view(
                    &draft.qkv,
                    s,
                    0,
                    q_shape,
                    [c.qkv_dim() as u64, u64::from(head), 1],
                )?,
                k: view(&draft.qkv, s, width, kv_shape, kv_stride)?,
                v: view(&draft.qkv, s, width + kv_width, kv_shape, kv_stride)?,
                q: TensorMut(view(&draft.attn, s, 0, q_shape, q_stride)?),
                gate: None,
                norm_mode: RmsMode::Standard,
                core_out: TensorMut(view(&draft.attn, s, 0, q_shape, q_stride)?),
                k_cache: TensorMut(view(&cache.k, s, 0, cache_shape, cache_stride)?),
                v_cache: TensorMut(view(&cache.v, s, 0, cache_shape, cache_stride)?),
                q_norm: view(&layer.q_norm, s, 0, [head], [1])?,
                k_norm: view(&layer.k_norm, s, 0, [head], [1])?,
                causal_mask: view(&mask, s, 0, [MASK_LEN], [1])?,
                positions: PositionBindings::Axes(view(&positions, s, 0, [rows, 3], [3, 1])?),
            };
            layers.push(FullAttentionOps::prepare_reusing(
                &device.ctx,
                &spec,
                limits,
                bindings,
                common.clone(),
                layers.first(),
            )?);
        }
        Ok(Self {
            layers,
            metadata,
            positions,
            mask,
            scratch,
            prepare: super::cubin::module(&device.ctx, "rope")?
                .load_function("sequence_positions")?,
        })
    }
}

struct LayerWeights {
    block: Layer,
    q_norm: CudaSlice<bf16>,
    k_norm: CudaSlice<bf16>,
}

struct DraftWeights {
    layers: Vec<LayerWeights>,
    fc: CudaSlice<bf16>,
    hidden_norm: CudaSlice<bf16>,
    norm: CudaSlice<bf16>,
    w1: CudaSlice<bf16>,
    w2: CudaSlice<bf16>,
    confidence: CudaSlice<bf16>,
    confidence_bias: CudaSlice<bf16>,
    bytes: usize,
}

impl DraftWeights {
    fn load(path: &Path, c: &DraftConfig, d: &Device) -> Result<Self> {
        checkpoint(path, d, |l| {
            let h = c.hidden_size;
            let f = c.intermediate_size;
            let kv = c.num_key_value_heads * c.head_dim;
            let fc = l.load(&[("fc.weight".into(), vec![h, c.target_layer_ids.len() * h])])?;
            let hidden_norm = l.load(&[("hidden_norm.weight".into(), vec![h])])?;
            let mut layers = Vec::new();
            for i in 0..c.num_hidden_layers {
                let prefix = format!("layers.{i}");
                let mut one = |name: &str, shape: Vec<usize>| {
                    l.load(&[(format!("{prefix}.{name}.weight"), shape)])
                };
                let input_norm = one("input_layernorm", vec![h])?;
                let post_norm = one("post_attention_layernorm", vec![h])?;
                let o = one("self_attn.o_proj", vec![h, h])?;
                let down = one("mlp.down_proj", vec![h, f])?;
                let q_norm = one("self_attn.q_norm", vec![c.head_dim])?;
                let k_norm = one("self_attn.k_norm", vec![c.head_dim])?;
                let qkv = l.load(&[
                    (format!("{prefix}.self_attn.q_proj.weight"), vec![h, h]),
                    (format!("{prefix}.self_attn.k_proj.weight"), vec![kv, h]),
                    (format!("{prefix}.self_attn.v_proj.weight"), vec![kv, h]),
                ])?;
                let gu = l.load(&[
                    (format!("{prefix}.mlp.gate_proj.weight"), vec![f, h]),
                    (format!("{prefix}.mlp.up_proj.weight"), vec![f, h]),
                ])?;
                layers.push(LayerWeights {
                    block: Layer {
                        input_norm: Some(input_norm),
                        qkv,
                        o,
                        post_norm: Some(post_norm),
                        gu,
                        down,
                    },
                    q_norm,
                    k_norm,
                });
            }
            let norm = l.load(&[("norm.weight".into(), vec![h])])?;
            let w1 = l.load(&[(
                "markov_head.markov_w1.weight".into(),
                vec![c.vocab_size, c.markov_rank],
            )])?;
            let w2 = l.load(&[(
                "markov_head.markov_w2.weight".into(),
                vec![c.vocab_size, c.markov_rank],
            )])?;
            let confidence = l.load(&[(
                "confidence_head.proj.weight".into(),
                vec![1, h + c.markov_rank],
            )])?;
            let confidence_bias = l.load(&[("confidence_head.proj.bias".into(), vec![1])])?;
            Ok(Self {
                layers,
                fc,
                hidden_norm,
                norm,
                w1,
                w2,
                confidence,
                confidence_bias,
                bytes: l.bytes,
            })
        })
    }
}

struct Kv {
    k: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
}

type StagedDraft = super::engine::deepseek::staged::StagedDraft;

pub struct Draft {
    pub config: DraftConfig,
    runtime: Runtime,
}

enum Runtime {
    Dense(Box<Dense>),
    Staged(Box<StagedDraft>),
}

impl Draft {
    #[cfg(all(test, feature = "cuda"))]
    pub(super) fn staged_bits(&self) -> Result<Vec<u8>> {
        match &self.runtime {
            Runtime::Staged(d) => d.test_bits(),
            Runtime::Dense(_) => anyhow::bail!("staged output gate requires a native draft"),
        }
    }

    pub fn new(
        path: &Path,
        device: &Device,
        target: &Config,
        capacity: usize,
        rows: usize,
        blas: &mut Blas,
        rounding: super::markov::ScoreRounding,
    ) -> Result<Self> {
        let config = DraftConfig::read(path, target)?;
        let runtime = if config.staged.is_some() {
            ensure!(
                rounding == super::markov::ScoreRounding::F32,
                "native DSpark requires F32 scores"
            );
            Runtime::Staged(Box::new(StagedDraft::load(
                path,
                device,
                config.clone(),
                capacity,
                rows,
                blas,
            )?))
        } else {
            Runtime::Dense(Box::new(Dense::new(
                path, device, target, capacity, rows, blas, rounding,
            )?))
        };
        Ok(Self { config, runtime })
    }

    #[cfg(feature = "cuda")]
    pub(super) fn prepare_linears(
        &self,
        head: &CudaSlice<bf16>,
        blas: &mut Blas,
        profile: &super::closure::Profile,
        write: bool,
    ) -> Result<()> {
        match &self.runtime {
            Runtime::Dense(d) => d.prepare_linears(head, blas, profile, write),
            Runtime::Staged(_) => {
                ensure!(!write, "native DSpark uses derived projection plans");
                Ok(())
            }
        }
    }

    #[cfg(feature = "cuda")]
    pub(super) fn calibrate_shape(
        &self,
        blas: &mut Blas,
        input: &crate::backend::gpu::CudaView<'_, bf16>,
        shape: crate::backend::setup::LinearShape,
    ) -> Result<()> {
        match &self.runtime {
            Runtime::Dense(d) => d.calibrate_shape(blas, input, shape),
            Runtime::Staged(_) => anyhow::bail!("native DSpark uses derived projection plans"),
        }
    }

    #[cfg(feature = "rocm")]
    pub(super) fn calibration_weight(
        &self,
        shape: crate::backend::setup::LinearShape,
    ) -> Result<crate::backend::gpu::CudaView<'_, bf16>> {
        match &self.runtime {
            Runtime::Dense(d) => d.calibration_weight(shape),
            Runtime::Staged(_) => anyhow::bail!("native DSpark uses original FP8 projections"),
        }
    }

    pub(super) fn reset_setup(&mut self) -> Result<()> {
        match &mut self.runtime {
            Runtime::Dense(d) => d.reset_setup(),
            Runtime::Staged(d) => d.reset_setup(),
        }
    }
    pub(super) fn grow(&mut self, rows: usize) -> Result<()> {
        match &mut self.runtime {
            Runtime::Dense(d) => d.grow(rows),
            Runtime::Staged(d) => d.grow(rows),
        }
    }
    pub(super) fn checkpoint_kv(&mut self, start: usize) -> Result<()> {
        match &mut self.runtime {
            Runtime::Dense(d) => d.checkpoint_kv(start),
            Runtime::Staged(d) => d.checkpoint_kv(start),
        }
    }
    pub(super) fn restore_kv(&mut self, start: usize, accepted: usize) -> Result<()> {
        match &mut self.runtime {
            Runtime::Dense(d) => d.restore_kv(start, accepted),
            Runtime::Staged(d) => d.restore_kv(start, accepted),
        }
    }
    pub fn inject(
        &mut self,
        input: &CudaSlice<bf16>,
        rows: usize,
        start: usize,
        compute: Compute<'_>,
        trace: Option<(&mut Trace, &str)>,
    ) -> Result<()> {
        match &mut self.runtime {
            Runtime::Dense(d) => d.inject(input, rows, start, compute, trace),
            Runtime::Staged(d) => {
                d.inject(input, rows, start, None, compute, trace.map(|(t, _)| t))
            }
        }
    }
    pub fn inject_device(
        &mut self,
        input: &CudaSlice<bf16>,
        rows: usize,
        start: usize,
        position: &CudaSlice<i32>,
        compute: Compute<'_>,
    ) -> Result<()> {
        match &mut self.runtime {
            Runtime::Dense(d) => d.inject_device(input, rows, start, position, compute),
            Runtime::Staged(d) => d.inject(input, rows, start, Some(position), compute, None),
        }
    }
    pub fn setup_attention(
        &mut self,
        device: &Device,
        ops: &Ops,
        target: &Config,
        dir: &Path,
        closure: &super::closure::Closure,
    ) -> Result<()> {
        match &mut self.runtime {
            Runtime::Dense(d) => d.setup_attention(device, ops, target, dir, closure),
            Runtime::Staged(_) => Ok(()),
        }
    }
    pub fn propose(
        &mut self,
        anchor: u32,
        start: usize,
        target: (&CudaSlice<bf16>, &CudaSlice<bf16>),
        compute: Compute<'_>,
        trace: Option<(&mut Trace, &str)>,
        forced: Option<&[u32]>,
    ) -> Result<Vec<u32>> {
        match &mut self.runtime {
            Runtime::Dense(d) => d.propose(anchor, start, target, compute, trace, forced),
            Runtime::Staged(d) => d.propose(
                anchor,
                start,
                target,
                compute,
                trace.map(|(t, _)| t),
                forced,
            ),
        }
    }
    #[cfg(feature = "cuda")]
    pub(super) fn capture_staged(
        &mut self,
        target: (&CudaSlice<bf16>, &CudaSlice<bf16>),
        compute: Compute<'_>,
    ) -> Result<()> {
        match &mut self.runtime {
            Runtime::Staged(d) => d.capture(target, compute),
            Runtime::Dense(_) => anyhow::bail!("staged capture requires a native draft"),
        }
    }

    #[cfg(feature = "cuda")]
    pub(super) fn bind_staged_slots(&mut self, slots: super::engine::moe::Slots) -> Result<()> {
        match &mut self.runtime {
            Runtime::Staged(d) => d.bind_slots(slots),
            Runtime::Dense(_) => anyhow::bail!("dense draft cannot bind a staged expert catalogue"),
        }
    }

    #[cfg(feature = "cuda")]
    pub(super) fn propose_joint(
        &mut self,
        owner: &mut super::engine::moe::Slots,
        anchor: u32,
        start: usize,
        target: (&CudaSlice<bf16>, &CudaSlice<bf16>),
        compute: Compute<'_>,
        graph: bool,
    ) -> Result<Vec<u32>> {
        match &mut self.runtime {
            Runtime::Staged(d) => d.propose_joint(owner, anchor, start, target, compute, graph),
            Runtime::Dense(_) => anyhow::bail!("joint native pool requires a staged draft"),
        }
    }

    pub fn propose_device(
        &mut self,
        anchor: &CudaSlice<u32>,
        position: &CudaSlice<i32>,
        start: usize,
        target: (&CudaSlice<bf16>, &CudaSlice<bf16>),
        compute: Compute<'_>,
    ) -> Result<()> {
        match &mut self.runtime {
            Runtime::Dense(d) => d.propose_device(anchor, position, start, target, compute),
            Runtime::Staged(d) => d.propose_device(anchor, position, start, target, compute),
        }
    }
    pub fn forward_base(
        &mut self,
        anchor: u32,
        start: usize,
        target: (&CudaSlice<bf16>, &CudaSlice<bf16>),
        compute: Compute<'_>,
    ) -> Result<()> {
        match &mut self.runtime {
            Runtime::Dense(d) => d.forward_base(anchor, start, target, compute),
            Runtime::Staged(_) => {
                anyhow::bail!("native DSpark exposes a greedy chain, not tree distributions")
            }
        }
    }
    pub fn forward_base_device(
        &mut self,
        anchor: &CudaSlice<u32>,
        position: &CudaSlice<i32>,
        start: usize,
        target: (&CudaSlice<bf16>, &CudaSlice<bf16>),
        compute: Compute<'_>,
    ) -> Result<()> {
        match &mut self.runtime {
            Runtime::Dense(d) => d.forward_base_device(anchor, position, start, target, compute),
            Runtime::Staged(_) => {
                anyhow::bail!("native DSpark exposes a greedy chain, not tree distributions")
            }
        }
    }
    pub fn setup_markov(
        &mut self,
        device: &Device,
        ops: &Ops,
        dir: &Path,
        closure: &super::closure::Closure,
    ) -> Result<()> {
        match &mut self.runtime {
            Runtime::Dense(d) => d.setup_markov(device, ops, dir, closure),
            Runtime::Staged(_) => Ok(()),
        }
    }
    #[cfg(feature = "cuda")]
    pub(super) fn markov_shape(&self) -> super::markov::MarkovShape {
        match &self.runtime {
            Runtime::Dense(d) => d.markov_shape(),
            Runtime::Staged(_) => super::markov::MarkovShape {
                rows: self.config.block_size,
                vocab: self.config.vocab_size,
                rank: self.config.markov_rank,
                rounding: super::markov::ScoreRounding::F32,
            },
        }
    }
    pub fn distributions_batch(
        &mut self,
        requests: &[(u8, u32)],
    ) -> Result<Vec<crate::backend::Top4>> {
        match &mut self.runtime {
            Runtime::Dense(d) => d.distributions_batch(requests),
            Runtime::Staged(_) => {
                anyhow::bail!("native DSpark exposes a greedy chain, not tree distributions")
            }
        }
    }
    pub fn proposals(&mut self) -> Result<Vec<u32>> {
        match &mut self.runtime {
            Runtime::Dense(d) => d.proposals(),
            Runtime::Staged(d) => d.proposals(),
        }
    }
    pub fn confidences(&self) -> Result<Vec<f32>> {
        match &self.runtime {
            Runtime::Dense(d) => d.confidences(),
            Runtime::Staged(d) => Ok(d.confidences()),
        }
    }
    pub fn copy_proposals(
        &self,
        ids: &mut crate::backend::gpu::CudaViewMut<'_, u32>,
    ) -> Result<()> {
        match &self.runtime {
            Runtime::Dense(d) => d.copy_proposals(ids),
            Runtime::Staged(d) => d.copy_proposals(ids),
        }
    }
    pub fn state_bits(&self, start: usize, rows: usize, output: &mut Vec<u8>) -> Result<()> {
        match &self.runtime {
            Runtime::Dense(d) => d.state_bits(start, rows, output),
            Runtime::Staged(d) => d.state_bits(start, rows, output),
        }
    }
    pub fn snapshot_kv(&self, trace: &mut Trace, start: usize, rows: usize) -> Result<()> {
        match &self.runtime {
            Runtime::Dense(d) => d.snapshot_kv(trace, start, rows),
            Runtime::Staged(d) => d.cache_trace(&mut Some(trace), "cache"),
        }
    }
}

struct Dense {
    markov: Option<super::markov::Markov>,
    score_rounding: super::markov::ScoreRounding,
    pub config: DraftConfig,
    weights: DraftWeights,
    kv: Vec<Kv>,
    kv_checkpoint: Vec<Kv>,
    capacity: usize,
    rows: usize,
    stream: Arc<CudaStream>,
    attention: Option<Attention>,
    #[cfg(test)]
    position_shift: i32,
    qknorm_rope_kv: crate::backend::gpu::CudaFunction,
    norm: crate::backend::gpu::CudaFunction,
    add_norm: crate::backend::gpu::CudaFunction,
    context_rope: crate::backend::gpu::CudaFunction,
    ids: CudaSlice<u32>,
    mask_ids: Vec<u32>,
    position: CudaSlice<i32>,
    x: CudaSlice<bf16>,
    n: CudaSlice<bf16>,
    qkv: CudaSlice<bf16>,
    attn: CudaSlice<bf16>,
    out: CudaSlice<bf16>,
    gu: CudaSlice<bf16>,
    act: CudaSlice<bf16>,
    k_raw: CudaSlice<bf16>,
    k_norm: CudaSlice<bf16>,
    v_raw: CudaSlice<bf16>,
    projected: CudaSlice<bf16>,
    context: CudaSlice<bf16>,
    packed: CudaSlice<bf16>,
    base: CudaSlice<bf16>,
    scores: CudaSlice<bf16>,
    combined: Option<CudaSlice<f32>>,
    confidence_values: CudaSlice<f32>,
    host_confidence: crate::backend::gpu::PinnedHostSlice<u8>,
    forced_tokens: CudaSlice<u32>,
    sum_argmax: crate::backend::gpu::CudaFunction,
    concat_dot_sigmoid: crate::backend::gpu::CudaFunction,
    prev: CudaSlice<bf16>,
    token: CudaSlice<u32>,
    tokens: CudaSlice<u32>,
}

fn norm(
    s: &Arc<CudaStream>,
    kernel: &crate::backend::gpu::CudaFunction,
    input: &CudaSlice<bf16>,
    weight: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    shape: (usize, usize),
    eps: f32,
) -> Result<()> {
    let (rows, dim) = shape;
    unsafe {
        s.launch_builder(kernel)
            .arg(input)
            .arg(weight)
            .arg(output)
            .arg(&(rows as i32))
            .arg(&(dim as i32))
            .arg(&eps)
            .launch(grid(rows, 256))?;
    }
    Ok(())
}

impl Dense {
    #[cfg(feature = "cuda")]
    pub(super) fn prepare_linears(
        &self,
        head: &CudaSlice<bf16>,
        blas: &mut Blas,
        profile: &super::closure::Profile,
        write: bool,
    ) -> Result<()> {
        blas.declare(self.rows, &profile.pairs);
        if self.score_rounding == super::markov::ScoreRounding::F32 {
            let buckets = (1..=self.rows)
                .map(|rows| super::engine::bucketed_prefill_rows(rows as u32, self.rows as u32))
                .collect::<Result<std::collections::BTreeSet<_>>>()?;
            for rows in buckets {
                blas.prepare(
                    rows as usize,
                    self.config.hidden_size,
                    self.config.capture_width(),
                )?;
                blas.prepare(rows as usize, self.config.kv_dim(), self.config.hidden_size)?;
            }
        }
        if !write {
            return blas.load_choices(&profile.linear);
        }
        for &shape in &profile.linear {
            let values: Vec<_> = (0..shape.rows * shape.input)
                .map(|i| bf16::from_f32(((i % 257) as f32 - 128.0) / 128.0))
                .collect();
            let input = self.stream.clone_htod(&values)?;
            if shape.output == self.config.vocab_size && shape.input == self.config.hidden_size {
                blas.calibrate_shape(head, &input.slice(..), shape)?;
            } else {
                self.calibrate_shape(blas, &input.slice(..), shape)?;
            }
        }
        Ok(())
    }
    #[cfg(feature = "cuda")]
    pub(super) fn calibrate_shape(
        &self,
        blas: &mut Blas,
        input: &crate::backend::gpu::CudaView<'_, bf16>,
        shape: crate::backend::setup::LinearShape,
    ) -> Result<()> {
        blas.calibrate_shape(&self.calibration_weight(shape)?, input, shape)
    }
    pub(super) fn calibration_weight(
        &self,
        shape: crate::backend::setup::LinearShape,
    ) -> Result<crate::backend::gpu::CudaView<'_, bf16>> {
        let c = &self.config;
        let h = c.hidden_size;
        let kv = c.kv_dim();
        let layer = &self.weights.layers[0].block;
        let weights = [
            (c.qkv_dim(), h, layer.qkv.slice(..)),
            (h, h, layer.o.slice(..)),
            (2 * c.intermediate_size, h, layer.gu.slice(..)),
            (h, c.intermediate_size, layer.down.slice(..)),
            (h, c.capture_width(), self.weights.fc.slice(..)),
            (kv, h, layer.qkv.slice(h * h..(h + kv) * h)),
            (c.vocab_size, c.markov_rank, self.weights.w2.slice(..)),
        ];
        let (_, _, weight) = weights
            .into_iter()
            .find(|(output, input, _)| *output == shape.output && *input == shape.input)
            .with_context(|| format!("unknown draft calibration shape: {shape:?}"))?;
        Ok(weight)
    }
    pub(super) fn reset_setup(&mut self) -> Result<()> {
        let block_rows = self.config.block_size;
        self.mask_ids.fill(self.config.mask_token_id);
        let s = &self.stream;
        let dim = self.config.head_dim;
        for kv in &mut self.kv {
            for head in 0..self.config.num_key_value_heads {
                let lo = head * self.capacity * dim;
                s.memset_zeros(&mut kv.k.slice_mut(lo..lo + (block_rows + 1) * dim))?;
                s.memset_zeros(&mut kv.v.slice_mut(lo..lo + (block_rows + 1) * dim))?;
            }
        }
        for value in [
            &mut self.x,
            &mut self.n,
            &mut self.qkv,
            &mut self.attn,
            &mut self.out,
            &mut self.gu,
            &mut self.act,
            &mut self.k_raw,
            &mut self.k_norm,
            &mut self.v_raw,
            &mut self.projected,
            &mut self.context,
            &mut self.packed,
            &mut self.base,
            &mut self.scores,
            &mut self.prev,
        ] {
            s.memset_zeros(value)?;
        }
        for value in [&mut self.token, &mut self.tokens] {
            s.memset_zeros(value)?;
        }
        s.memset_zeros(&mut self.position)?;
        if let Some(combined) = &mut self.combined {
            s.memset_zeros(combined)?;
        }
        s.memset_zeros(&mut self.confidence_values)?;
        self.host_confidence.as_mut_slice()?.fill(0);
        s.memset_zeros(&mut self.forced_tokens)?;
        s.memcpy_htod(&self.mask_ids, &mut self.ids)?;
        Ok(())
    }
    pub fn new(
        path: &Path,
        d: &Device,
        c: &Config,
        capacity: usize,
        rows: usize,
        blas: &mut Blas,
        score_rounding: super::markov::ScoreRounding,
    ) -> Result<Self> {
        #[cfg(feature = "rocm")]
        let _ = blas;
        let config = DraftConfig::read(path, c)?;
        ensure!(
            rows > config.block_size && capacity >= rows,
            "DSpark workspace must fit block_size+1 rows within capacity"
        );
        let weights = DraftWeights::load(path, &config, d)?;
        let s = &d.stream;
        let h = config.hidden_size;
        let f = config.intermediate_size;
        let kv = config.kv_dim();
        let qkv = config.qkv_dim();
        let vocab = config.vocab_size;
        let rank = config.markov_rank;
        let block_rows = config.block_size;
        let mut caches = Vec::new();
        let mut kv_checkpoint = Vec::new();
        let checkpoint_words = config
            .verify_rows()?
            .checked_mul(kv)
            .context("draft KV checkpoint extent overflow")?;
        for _ in 0..config.num_hidden_layers {
            caches.push(Kv {
                k: s.alloc_zeros(capacity * kv)?,
                v: s.alloc_zeros(capacity * kv)?,
            });
            if score_rounding == super::markov::ScoreRounding::F32 {
                kv_checkpoint.push(Kv {
                    k: s.alloc_zeros(checkpoint_words)?,
                    v: s.alloc_zeros(checkpoint_words)?,
                });
            }
        }
        #[cfg(feature = "cuda")]
        for (n, o, i) in [
            (block_rows, qkv, h),
            (block_rows, h, h),
            (block_rows, 2 * f, h),
            (block_rows, h, f),
            (block_rows, vocab, h),
            (rows, h, config.capture_width()),
            (rows, kv, h),
        ] {
            blas.prepare(n, o, i)?;
        }
        eprintln!(
            "draft_weights_bytes={} draft_kv_bytes={} draft_kv_checkpoint_bytes={}",
            weights.bytes,
            config.num_hidden_layers * capacity * kv * 4,
            kv_checkpoint
                .iter()
                .map(|cache| (cache.k.len() + cache.v.len()) * size_of::<bf16>())
                .sum::<usize>()
        );
        let mask_ids = vec![config.mask_token_id; block_rows];
        let qknorm_rope_kv =
            super::cubin::module(&d.ctx, "rope")?.load_function("draft_qknorm_rope_kv")?;
        let norm_module = super::cubin::module(&d.ctx, "norm")?;
        let rope_module = super::cubin::module(&d.ctx, "rope")?;
        let (norm_name, add_norm_name, rope_name) =
            if score_rounding == super::markov::ScoreRounding::F32 {
                ("rms_norm_bf16", "add_rmsnorm_bf16", "rope")
            } else {
                ("rms_norm", "add_rmsnorm", "rope_float")
            };
        Ok(Self {
            markov: None,
            score_rounding,
            config,
            weights,
            kv: caches,
            kv_checkpoint,
            capacity,
            rows,
            stream: s.clone(),
            attention: None,
            #[cfg(test)]
            position_shift: 0,
            qknorm_rope_kv,
            norm: norm_module.load_function(norm_name)?,
            add_norm: norm_module.load_function(add_norm_name)?,
            context_rope: rope_module.load_function(rope_name)?,
            ids: s.clone_htod(&mask_ids)?,
            mask_ids,
            position: s.alloc_zeros(1)?,
            x: s.alloc_zeros(block_rows * h)?,
            n: s.alloc_zeros(block_rows * h)?,
            qkv: s.alloc_zeros(block_rows * qkv)?,
            attn: s.alloc_zeros(block_rows * h)?,
            out: s.alloc_zeros(block_rows * h)?,
            gu: s.alloc_zeros(block_rows * 2 * f)?,
            act: s.alloc_zeros(block_rows * f)?,
            k_raw: s.alloc_zeros(rows * kv)?,
            k_norm: s.alloc_zeros(rows * kv)?,
            v_raw: s.alloc_zeros(rows * kv)?,
            projected: s.alloc_zeros(rows * h)?,
            context: s.alloc_zeros(rows * h)?,
            packed: s.alloc_zeros(rows * 2 * kv)?,
            base: s.alloc_zeros(block_rows * vocab)?,
            scores: s.alloc_zeros(block_rows * vocab)?,
            combined: if score_rounding == super::markov::ScoreRounding::F32 {
                Some(s.alloc_zeros(block_rows * vocab)?)
            } else {
                None
            },
            confidence_values: s.alloc_zeros(block_rows)?,
            host_confidence: super::copy::pinned_zeros(&d.ctx, block_rows * size_of::<f32>())?,
            forced_tokens: s.alloc_zeros(block_rows)?,
            sum_argmax: super::cubin::module(&d.ctx, "markov")?.load_function("sum_argmax")?,
            concat_dot_sigmoid: super::cubin::module(&d.ctx, "markov")?
                .load_function("concat_dot_sigmoid")?,
            prev: s.alloc_zeros(rank)?,
            token: s.alloc_zeros(1)?,
            tokens: s.alloc_zeros(block_rows)?,
        })
    }

    pub(super) fn grow(&mut self, rows: usize) -> Result<()> {
        ensure!(
            rows >= self.rows && rows <= self.capacity,
            "draft workspace rows outside cache bound"
        );
        let c = &self.config;
        let s = &self.stream;
        let planned = rows
            .checked_mul(c.workspace_row_bytes()?)
            .context("draft row workspace byte count overflow")?;
        let k_raw = s.alloc_zeros(rows * c.kv_dim())?;
        let k_norm = s.alloc_zeros(rows * c.kv_dim())?;
        let v_raw = s.alloc_zeros(rows * c.kv_dim())?;
        let projected = s.alloc_zeros(rows * c.hidden_size)?;
        let context = s.alloc_zeros(rows * c.hidden_size)?;
        let packed = s.alloc_zeros(rows * 2 * c.kv_dim())?;
        let allocated = [&k_raw, &k_norm, &v_raw, &projected, &context, &packed]
            .iter()
            .map(|buffer| buffer.len() * size_of::<bf16>())
            .sum::<usize>();
        ensure!(
            allocated == planned,
            "draft row workspace allocation={allocated} differs from planned={planned} rows={rows}"
        );
        self.k_raw = k_raw;
        self.k_norm = k_norm;
        self.v_raw = v_raw;
        self.projected = projected;
        self.context = context;
        self.packed = packed;
        self.rows = rows;
        Ok(())
    }

    pub(super) fn checkpoint_kv(&mut self, start: usize) -> Result<()> {
        let rows = self.config.verify_rows()?;
        ensure!(
            self.kv_checkpoint.len() == self.kv.len()
                && !self.kv_checkpoint.is_empty()
                && start
                    .checked_add(rows)
                    .is_some_and(|end| end <= self.capacity),
            "draft KV checkpoint start={start} rows={rows} capacity={} layers={}",
            self.capacity,
            self.kv_checkpoint.len()
        );
        let d = self.config.head_dim;
        for (live, saved) in self.kv.iter().zip(&mut self.kv_checkpoint) {
            for (source, destination) in [(&live.k, &mut saved.k), (&live.v, &mut saved.v)] {
                columns(
                    &self.stream,
                    source,
                    destination,
                    Rect {
                        rows: self.config.num_key_value_heads,
                        width: rows * d,
                        src_pitch: self.capacity * d,
                        src_col: start * d,
                        dst_pitch: rows * d,
                        dst_col: 0,
                    },
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn restore_kv(&mut self, start: usize, accepted: usize) -> Result<()> {
        let rows = self.config.verify_rows()?;
        ensure!(
            accepted <= rows
                && self.kv_checkpoint.len() == self.kv.len()
                && !self.kv_checkpoint.is_empty()
                && start
                    .checked_add(rows)
                    .is_some_and(|end| end <= self.capacity),
            "draft KV restore start={start} accepted={accepted} rows={rows} capacity={}",
            self.capacity
        );
        if accepted == rows {
            return Ok(());
        }
        let d = self.config.head_dim;
        for (saved, live) in self.kv_checkpoint.iter().zip(&mut self.kv) {
            for (source, destination) in [(&saved.k, &mut live.k), (&saved.v, &mut live.v)] {
                columns(
                    &self.stream,
                    source,
                    destination,
                    Rect {
                        rows: self.config.num_key_value_heads,
                        width: (rows - accepted) * d,
                        src_pitch: rows * d,
                        src_col: accepted * d,
                        dst_pitch: self.capacity * d,
                        dst_col: (start + accepted) * d,
                    },
                )?;
            }
        }
        Ok(())
    }

    pub fn inject(
        &mut self,
        input: &CudaSlice<bf16>,
        rows: usize,
        start: usize,
        compute: Compute<'_>,
        trace: Option<(&mut Trace, &str)>,
    ) -> Result<()> {
        self.stream
            .memcpy_htod(&[start as i32], &mut self.position)?;
        self.inject_block(input, rows, start, compute, trace)
    }

    pub fn inject_device(
        &mut self,
        input: &CudaSlice<bf16>,
        rows: usize,
        start: usize,
        position: &CudaSlice<i32>,
        compute: Compute<'_>,
    ) -> Result<()> {
        self.stream.memcpy_dtod(position, &mut self.position)?;
        self.inject_block(input, rows, start, compute, None)
    }

    fn inject_block(
        &mut self,
        input: &CudaSlice<bf16>,
        rows: usize,
        start: usize,
        compute: Compute<'_>,
        trace: Option<(&mut Trace, &str)>,
    ) -> Result<()> {
        let Compute { ops, blas } = compute;
        let c = &self.config;
        ensure!(
            rows > 0 && rows <= self.rows && start + rows <= self.capacity,
            "draft injection exceeds workspace/cache"
        );
        let s = &self.stream;
        let h = c.hidden_size;
        let kv = c.kv_dim();
        let linear_rows =
            if cfg!(feature = "rocm") || self.score_rounding == super::markov::ScoreRounding::F32 {
                super::engine::bucketed_prefill_rows(rows as u32, self.rows as u32)? as usize
            } else {
                blas.prepare(rows, h, c.capture_width())?;
                blas.prepare(rows, kv, h)?;
                rows
            };
        ensure!(
            input.len() >= linear_rows * c.capture_width(),
            "draft injection input={} rows={rows} linear_rows={linear_rows} capture_width={}",
            input.len(),
            c.capture_width()
        );
        blas.linear(
            &self.weights.fc,
            &input.slice(..linear_rows * c.capture_width()),
            &mut self.projected.slice_mut(..linear_rows * h),
            linear_rows,
            h,
            c.capture_width(),
        )?;
        norm(
            s,
            &self.norm,
            &self.projected,
            &self.weights.hidden_norm,
            &mut self.context,
            (rows, h),
            self.config.rms_norm_eps,
        )?;
        for (i, layer) in self.weights.layers.iter().enumerate() {
            let k_weight = layer.block.qkv.slice(h * h..(h + kv) * h);
            let v_weight = layer.block.qkv.slice((h + kv) * h..(h + 2 * kv) * h);
            blas.linear(
                &k_weight,
                &self.context.slice(..linear_rows * h),
                &mut self.k_raw.slice_mut(..linear_rows * kv),
                linear_rows,
                kv,
                h,
            )?;
            blas.linear(
                &v_weight,
                &self.context.slice(..linear_rows * h),
                &mut self.v_raw.slice_mut(..linear_rows * kv),
                linear_rows,
                kv,
                h,
            )?;
            norm(
                s,
                &self.norm,
                &self.k_raw,
                &layer.k_norm,
                &mut self.k_norm,
                (rows * c.num_key_value_heads, c.head_dim),
                self.config.rms_norm_eps,
            )?;
            unsafe {
                s.launch_builder(&self.context_rope)
                    .arg(&mut self.k_norm)
                    .arg(&self.position)
                    .arg(&(rows as i32))
                    .arg(&(c.num_key_value_heads as i32))
                    .arg(&0i32)
                    .arg(&(c.head_dim as i32))
                    .arg(&c.rope_parameters.rope_theta)
                    .launch(flat(rows * kv / 2))?;
            }
            columns(
                s,
                &self.k_norm,
                &mut self.packed,
                Rect {
                    rows,
                    width: kv,
                    src_pitch: kv,
                    src_col: 0,
                    dst_pitch: 2 * kv,
                    dst_col: 0,
                },
            )?;
            columns(
                s,
                &self.v_raw,
                &mut self.packed,
                Rect {
                    rows,
                    width: kv,
                    src_pitch: kv,
                    src_col: 0,
                    dst_pitch: 2 * kv,
                    dst_col: kv,
                },
            )?;
            let cache = &mut self.kv[i];
            unsafe {
                s.launch_builder(&ops.kv_write)
                    .arg(&self.packed)
                    .arg(&mut cache.k)
                    .arg(&mut cache.v)
                    .arg(&self.position)
                    .arg(&(rows as i32))
                    .arg(&0i32)
                    .arg(&(c.num_key_value_heads as i32))
                    .arg(&(c.head_dim as i32))
                    .arg(&(self.capacity as i32))
                    .launch(flat(rows * kv))?;
            }
        }
        if let Some((trace, prefix)) = trace {
            add_trace(
                s,
                trace,
                format!("{prefix}.c_projected"),
                &self.projected,
                rows,
                h,
            )?;
            add_trace(
                s,
                trace,
                format!("{prefix}.c_committed"),
                &self.context,
                rows,
                h,
            )?;
        }
        Ok(())
    }
}

fn add_trace(
    s: &Arc<CudaStream>,
    trace: &mut Trace,
    name: String,
    data: &CudaSlice<bf16>,
    rows: usize,
    width: usize,
) -> Result<()> {
    let values = s.clone_dtoh(&data.slice(..rows * width))?;
    trace.add(
        name,
        vec![rows, width],
        values.into_iter().map(bf16::to_f32).collect(),
    )
}

impl Dense {
    pub fn setup_attention(
        &mut self,
        d: &Device,
        ops: &Ops,
        c: &Config,
        dir: &Path,
        closure: &super::closure::Closure,
    ) -> Result<()> {
        if self.attention.is_some() {
            return Ok(());
        }
        if self.config.head_dim == 256 {
            self.attention = Some(Attention::Block(BlockAttention::new(self, d)?));
            return Ok(());
        }
        let block_rows = self.config.block_size;
        let mut geometry = c.clone();
        geometry.num_attention_heads = self.config.num_attention_heads;
        geometry.num_key_value_heads = self.config.num_key_value_heads;
        geometry.head_dim = self.config.head_dim;
        geometry.rope_theta = self.config.rope_parameters.rope_theta;
        geometry.rms_norm_eps = self.config.rms_norm_eps;
        let c = &geometry;
        #[cfg(feature = "cuda")]
        let measured = super::multi_calibrate::load(
            d,
            ops,
            c,
            dir,
            crate::backend::setup::MultiShape {
                rows: block_rows,
                causal: false,
                capacity: self.capacity,
            },
            closure,
        )?;
        #[cfg(feature = "cuda")]
        let (plan, selection) = measured.choose(self.capacity)?;
        #[cfg(feature = "rocm")]
        let plan = {
            let _ = (ops, dir);
            closure.bind_fixed()?;
            super::verification::hip_tcmqa_plan(&d.ctx, self.capacity, block_rows)?
        };
        #[cfg(feature = "cuda")]
        eprintln!(
            "{}",
            serde_json::json!({"backend_setup":{"role":"draft","rows":block_rows,"causal":false,"attention":plan,"selection":selection,"calibration":measured}})
        );
        self.attention = Some(Attention::Cached(Verification::with_plan(
            &d.ctx,
            &d.stream,
            c,
            crate::backend::setup::MultiShape {
                capacity: self.capacity,
                rows: block_rows,
                causal: false,
            },
            plan,
        )?));
        Ok(())
    }
    pub fn propose(
        &mut self,
        anchor: u32,
        start: usize,
        target: (&CudaSlice<bf16>, &CudaSlice<bf16>),
        compute: Compute<'_>,
        trace: Option<(&mut Trace, &str)>,
        forced: Option<&[u32]>,
    ) -> Result<Vec<u32>> {
        self.mask_ids[0] = anchor;
        self.stream.memcpy_htod(&self.mask_ids, &mut self.ids)?;
        self.stream
            .memcpy_htod(&[start as i32], &mut self.position)?;
        self.block(start, target, compute, trace, forced, Output::Tokens)
    }

    pub fn propose_device(
        &mut self,
        anchor: &CudaSlice<u32>,
        position: &CudaSlice<i32>,
        start: usize,
        target: (&CudaSlice<bf16>, &CudaSlice<bf16>),
        compute: Compute<'_>,
    ) -> Result<()> {
        self.stream
            .memcpy_dtod(anchor, &mut self.ids.slice_mut(..1))?;
        self.stream.memcpy_dtod(position, &mut self.position)?;
        self.block(start, target, compute, None, None, Output::Device)?;
        Ok(())
    }

    pub fn forward_base(
        &mut self,
        anchor: u32,
        start: usize,
        target: (&CudaSlice<bf16>, &CudaSlice<bf16>),
        compute: Compute<'_>,
    ) -> Result<()> {
        self.mask_ids[0] = anchor;
        self.stream.memcpy_htod(&self.mask_ids, &mut self.ids)?;
        self.stream
            .memcpy_htod(&[start as i32], &mut self.position)?;
        self.block(start, target, compute, None, None, Output::Base)?;
        Ok(())
    }

    pub fn forward_base_device(
        &mut self,
        anchor: &CudaSlice<u32>,
        position: &CudaSlice<i32>,
        start: usize,
        target: (&CudaSlice<bf16>, &CudaSlice<bf16>),
        compute: Compute<'_>,
    ) -> Result<()> {
        self.stream
            .memcpy_dtod(anchor, &mut self.ids.slice_mut(..1))?;
        self.stream.memcpy_dtod(position, &mut self.position)?;
        self.block(start, target, compute, None, None, Output::Base)?;
        Ok(())
    }

    pub fn setup_markov(
        &mut self,
        d: &Device,
        ops: &super::ops::Ops,
        dir: &std::path::Path,
        closure: &super::closure::Closure,
    ) -> Result<()> {
        if self.markov.is_none() {
            self.markov = Some(super::markov::Markov::new(
                d,
                ops,
                dir,
                closure,
                self.markov_shape(),
            )?);
        }
        Ok(())
    }

    pub(super) fn markov_shape(&self) -> super::markov::MarkovShape {
        super::markov::MarkovShape {
            rows: self.config.block_size,
            vocab: self.config.vocab_size,
            rank: self.config.markov_rank,
            rounding: self.score_rounding,
        }
    }

    pub fn distributions_batch(
        &mut self,
        requests: &[(u8, u32)],
    ) -> Result<Vec<crate::backend::Top4>> {
        self.markov
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Markov module is not initialized"))?
            .distributions_batch(&self.base, &self.weights.w1, &self.weights.w2, requests)
    }

    pub fn proposals(&mut self) -> Result<Vec<u32>> {
        super::copy::dtoh_bytes(
            &self.stream,
            &self.confidence_values,
            0,
            &mut self.host_confidence,
            0,
            self.config.block_size * size_of::<f32>(),
        )?;
        let values = self.stream.clone_dtoh(&self.tokens)?;
        Ok(values)
    }

    pub fn confidences(&self) -> Result<Vec<f32>> {
        let bytes = self.host_confidence.as_slice()?;
        let (chunks, remainder) = bytes.as_chunks::<4>();
        ensure!(
            remainder.is_empty() && chunks.len() == self.config.block_size,
            "draft host confidence extent={} block_size={}",
            bytes.len(),
            self.config.block_size
        );
        Ok(chunks.iter().map(|b| f32::from_le_bytes(*b)).collect())
    }

    pub fn copy_proposals(
        &self,
        ids: &mut crate::backend::gpu::CudaViewMut<'_, u32>,
    ) -> Result<()> {
        self.stream.memcpy_dtod(&self.tokens, ids)?;
        Ok(())
    }

    pub fn state_bits(&self, start: usize, rows: usize, out: &mut Vec<u8>) -> Result<()> {
        ensure!(
            rows > 0 && start + rows <= self.capacity,
            "draft bit comparison range exceeds KV"
        );
        let s = &self.stream;
        if let Some(Attention::Block(attention)) = &self.attention {
            for values in [&attention.metadata, &attention.positions] {
                out.extend(s.clone_dtoh(values)?.iter().flat_map(|v| v.to_le_bytes()));
            }
            out.extend(
                s.clone_dtoh(&attention.mask)?
                    .iter()
                    .flat_map(|v| v.to_le_bytes()),
            );
        }
        for values in [&self.n, &self.base, &self.scores] {
            super::copy::bits(s, values, out)?;
        }
        if let Some(combined) = &self.combined {
            out.extend(
                s.clone_dtoh(combined)?
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes()),
            );
        }
        out.extend(
            s.clone_dtoh(&self.confidence_values)?
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes()),
        );
        let d = self.config.head_dim;
        for kv in &self.kv {
            for head in 0..self.config.num_key_value_heads {
                let lo = (head * self.capacity + start) * d;
                let hi = lo + rows * d;
                super::copy::bits(s, &kv.k.slice(lo..hi), out)?;
                super::copy::bits(s, &kv.v.slice(lo..hi), out)?;
            }
        }
        Ok(())
    }

    fn block(
        &mut self,
        start: usize,
        target: (&CudaSlice<bf16>, &CudaSlice<bf16>),
        compute: Compute<'_>,
        mut trace: Option<(&mut Trace, &str)>,
        forced: Option<&[u32]>,
        output: Output,
    ) -> Result<Vec<u32>> {
        if let Some(tokens) = forced {
            ensure!(
                tokens.len() == self.config.block_size
                    && tokens
                        .iter()
                        .all(|&v| (v as usize) < self.config.vocab_size),
                "invalid teacher-forced Markov tokens"
            );
        }
        let Compute { ops, blas } = compute;
        let c = &self.config;
        let block_rows = c.block_size;
        ensure!(
            start + block_rows <= self.capacity,
            "draft block exceeds KV capacity"
        );
        let s = &self.stream;
        let h = c.hidden_size;
        let f = c.intermediate_size;
        let qkv = c.qkv_dim();
        let eps = self.config.rms_norm_eps;
        let add_norm_kernel = &self.add_norm;
        let add_norm = |res: &CudaSlice<bf16>,
                        x: &mut CudaSlice<bf16>,
                        weight: &CudaSlice<bf16>,
                        output: &mut CudaSlice<bf16>|
         -> Result<()> {
            let (xp, _x) = x.device_ptr_mut(s);
            unsafe {
                s.launch_builder(add_norm_kernel)
                    .arg(res)
                    .arg(&xp)
                    .arg(weight)
                    .arg(output)
                    .arg(&xp)
                    .arg(&(block_rows as i32))
                    .arg(&(h as i32))
                    .arg(&eps)
                    .launch(grid(block_rows, 256))?;
            }
            Ok(())
        };
        unsafe {
            s.launch_builder(&ops.embed)
                .arg(target.0)
                .arg(&self.ids)
                .arg(&mut self.x)
                .arg(&(block_rows as i32))
                .arg(&(h as i32))
                .launch(flat(block_rows * h))?;
        }
        if let Some(Attention::Block(attention)) = &mut self.attention {
            let shift = {
                #[cfg(test)]
                {
                    self.position_shift
                }
                #[cfg(not(test))]
                {
                    0i32
                }
            };
            unsafe {
                s.launch_builder(&attention.prepare)
                    .arg(&self.position)
                    .arg(&mut attention.metadata)
                    .arg(&mut attention.positions)
                    .arg(&(block_rows as i32))
                    .arg(&(self.capacity as i32))
                    .arg(&shift)
                    .launch(grid(1, 256))?;
            }
        }
        for (i, layer) in self.weights.layers.iter().enumerate() {
            if i == 0 {
                norm(
                    s,
                    &self.norm,
                    &self.x,
                    layer
                        .block
                        .input_norm
                        .as_ref()
                        .context("draft input norm missing")?,
                    &mut self.n,
                    (block_rows, h),
                    self.config.rms_norm_eps,
                )?;
            }
            blas.linear(
                &layer.block.qkv,
                &self.n.slice(..block_rows * h),
                &mut self.qkv.slice_mut(..block_rows * qkv),
                block_rows,
                qkv,
                h,
            )?;
            match self
                .attention
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("draft attention has not been initialized"))?
            {
                Attention::Cached(attention) => {
                    let cache = &mut self.kv[i];
                    unsafe {
                        s.launch_builder(&self.qknorm_rope_kv)
                            .arg(&mut self.qkv)
                            .arg(&cache.k)
                            .arg(&cache.v)
                            .arg(&layer.q_norm)
                            .arg(&layer.k_norm)
                            .arg(&self.position)
                            .arg(&(block_rows as i32))
                            .arg(&(c.num_attention_heads as i32))
                            .arg(&(c.num_key_value_heads as i32))
                            .arg(&(c.head_dim as i32))
                            .arg(&(self.capacity as i32))
                            .arg(&0i32)
                            .arg(&c.rope_parameters.rope_theta)
                            .arg(&self.config.rms_norm_eps)
                            .launch(grid(
                                block_rows * (c.num_attention_heads + 2 * c.num_key_value_heads),
                                256,
                            ))?;
                    }
                    attention.run(
                        &self.qkv,
                        &self.kv[i].k,
                        &self.kv[i].v,
                        &mut self.attn,
                        &self.position,
                        Shape {
                            rows: block_rows,
                            start,
                            q_heads: c.num_attention_heads,
                            kv_heads: c.num_key_value_heads,
                            capacity: self.capacity,
                            head_dim: c.head_dim,
                            causal: false,
                            window_left: None,
                        },
                    )?;
                }
                Attention::Block(attention) => {
                    let (metadata, _meta_hold) = attention.metadata.device_ptr(s);
                    let (scratch, _scratch_hold) = attention.scratch.device_ptr(s);
                    FullAttentionOps::run(
                        &attention.layers[i],
                        s,
                        SeqRef { ptr: metadata },
                        Scratch {
                            ptr: scratch,
                            bytes: attention.scratch.len() as u64,
                        },
                    )?;
                }
            }
            blas.linear(
                &layer.block.o,
                &self.attn.slice(..block_rows * h),
                &mut self.out.slice_mut(..block_rows * h),
                block_rows,
                h,
                h,
            )?;
            add_norm(
                &self.out,
                &mut self.x,
                layer
                    .block
                    .post_norm
                    .as_ref()
                    .context("draft post norm missing")?,
                &mut self.n,
            )?;
            blas.linear(
                &layer.block.gu,
                &self.n.slice(..block_rows * h),
                &mut self.gu.slice_mut(..block_rows * 2 * f),
                block_rows,
                2 * f,
                h,
            )?;
            let act_cfg = super::ops::silu_mul_grid(
                block_rows,
                f,
                self.gu.device_ptr(s).0,
                self.act.device_ptr(s).0,
            )?;
            unsafe {
                s.launch_builder(&ops.act)
                    .arg(&self.gu)
                    .arg(&mut self.act)
                    .arg(&(block_rows as i32))
                    .arg(&(f as i32))
                    .arg(&(ops.swiglu_rounding() as i32))
                    .launch(act_cfg)?;
            }
            blas.linear(
                &layer.block.down,
                &self.act.slice(..block_rows * f),
                &mut self.out.slice_mut(..block_rows * h),
                block_rows,
                h,
                f,
            )?;
            let weight = if i + 1 < self.weights.layers.len() {
                self.weights.layers[i + 1]
                    .block
                    .input_norm
                    .as_ref()
                    .context("draft input norm missing")?
            } else {
                &self.weights.norm
            };
            add_norm(&self.out, &mut self.x, weight, &mut self.n)?;
            if let Some((trace, prefix)) = trace.as_mut() {
                add_trace(
                    s,
                    trace,
                    format!("{prefix}.layer.{i}"),
                    &self.x,
                    block_rows,
                    h,
                )?;
            }
        }
        blas.linear(
            target.1,
            &self.n.slice(..block_rows * h),
            &mut self.base.slice_mut(..block_rows * c.vocab_size),
            block_rows,
            c.vocab_size,
            h,
        )?;
        if matches!(output, Output::Base) {
            return Ok(Vec::new());
        }
        s.memcpy_dtod(&self.ids.slice(..1), &mut self.token)?;
        if let Some(tokens) = forced {
            s.memcpy_htod(tokens, &mut self.forced_tokens)?;
        }
        for row in 0..block_rows {
            unsafe {
                s.launch_builder(&ops.embed)
                    .arg(&self.weights.w1)
                    .arg(&self.token)
                    .arg(&mut self.prev)
                    .arg(&1i32)
                    .arg(&(self.config.markov_rank as i32))
                    .launch(flat(self.config.markov_rank))?;
            }
            blas.linear(
                &self.weights.w2,
                &self.prev.slice(..self.config.markov_rank),
                &mut self
                    .scores
                    .slice_mut(row * c.vocab_size..(row + 1) * c.vocab_size),
                1,
                c.vocab_size,
                self.config.markov_rank,
            )?;
            let (bias, _bias_hold) = self.scores.device_ptr_mut(s);
            let (combined, _combined_hold) = if let Some(values) = &mut self.combined {
                let (ptr, hold) = values.device_ptr_mut(s);
                (Some(ptr), Some(hold))
            } else {
                (None, None)
            };
            let bias_row = bias + (row * c.vocab_size * 2) as u64;
            let score_row = combined.map_or_else(
                || if trace.is_some() { bias_row } else { 0 },
                |ptr| ptr + (row * c.vocab_size * 4) as u64,
            );
            unsafe {
                s.launch_builder(&self.sum_argmax)
                    .arg(
                        &self
                            .base
                            .slice(row * c.vocab_size..(row + 1) * c.vocab_size),
                    )
                    .arg(&bias_row)
                    .arg(&score_row)
                    .arg(&mut self.token)
                    .arg(&(c.vocab_size as i32))
                    .arg(&(self.score_rounding as i32))
                    .launch(grid(1, 1024))?;
            }
            s.memcpy_dtod(&self.token, &mut self.tokens.slice_mut(row..row + 1))?;
            if forced.is_some() {
                s.memcpy_dtod(&self.forced_tokens.slice(row..row + 1), &mut self.token)?;
            }
        }
        unsafe {
            s.launch_builder(&self.concat_dot_sigmoid)
                .arg(&self.n)
                .arg(&self.weights.w1)
                .arg(&self.ids)
                .arg(if forced.is_some() {
                    &self.forced_tokens
                } else {
                    &self.tokens
                })
                .arg(&self.weights.confidence)
                .arg(&self.weights.confidence_bias)
                .arg(&mut self.confidence_values)
                .arg(&(block_rows as i32))
                .arg(&(h as i32))
                .arg(&(c.markov_rank as i32))
                .arg(&(c.vocab_size as i32))
                .launch(grid(block_rows, 256))?;
        }
        let proposals = if matches!(output, Output::Tokens) {
            super::copy::dtoh_bytes(
                s,
                &self.confidence_values,
                0,
                &mut self.host_confidence,
                0,
                block_rows * size_of::<f32>(),
            )?;
            s.clone_dtoh(&self.tokens)?
        } else {
            Vec::new()
        };
        if let Some((trace, prefix)) = trace.as_mut() {
            add_trace(s, trace, format!("{prefix}.H"), &self.n, block_rows, h)?;
            add_trace(
                s,
                trace,
                format!("{prefix}.B"),
                &self.base,
                block_rows,
                c.vocab_size,
            )?;
            if let Some(combined) = &self.combined {
                add_trace(
                    s,
                    trace,
                    format!("{prefix}.M"),
                    &self.scores,
                    block_rows,
                    c.vocab_size,
                )?;
                trace.add(
                    format!("{prefix}.s"),
                    vec![block_rows, c.vocab_size],
                    s.clone_dtoh(combined)?,
                )?;
            } else {
                add_trace(
                    s,
                    trace,
                    format!("{prefix}.s"),
                    &self.scores,
                    block_rows,
                    c.vocab_size,
                )?;
            }
            trace.add(
                format!("{prefix}.conf"),
                vec![block_rows],
                s.clone_dtoh(&self.confidence_values)?,
            )?;
            trace.add(
                format!("{prefix}.proposed"),
                vec![block_rows],
                proposals.iter().map(|&v| v as f32).collect(),
            )?;
        }
        Ok(proposals)
    }

    pub fn snapshot_kv(&self, trace: &mut Trace, start: usize, rows: usize) -> Result<()> {
        let c = &self.config;
        ensure!(
            start + rows <= self.capacity,
            "draft KV snapshot exceeds capacity"
        );
        for (i, kv) in self.kv.iter().enumerate() {
            for (name, values) in [("k", &kv.k), ("v", &kv.v)] {
                let mut output = vec![0f32; rows * c.kv_dim()];
                for head in 0..c.num_key_value_heads {
                    let values = self.stream.clone_dtoh(&values.slice(
                        (head * self.capacity + start) * c.head_dim
                            ..(head * self.capacity + start + rows) * c.head_dim,
                    ))?;
                    for row in 0..rows {
                        for dim in 0..c.head_dim {
                            output[row * c.kv_dim() + head * c.head_dim + dim] =
                                values[row * c.head_dim + dim].to_f32();
                        }
                    }
                }
                trace.add(
                    format!("kv_{name}_{i}"),
                    vec![rows, c.num_key_value_heads, c.head_dim],
                    output,
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use anyhow::Context;
    use safetensors::{Dtype, SafeTensors};
    use std::{path::PathBuf, rc::Rc};

    fn values(tensors: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let tensor = tensors.tensor(name)?;
        ensure!(tensor.shape() == shape, "reference {name} shape mismatch");
        let result: Vec<_> = match tensor.dtype() {
            Dtype::BF16 => tensor
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| bf16::from_bits(u16::from_le_bytes(*v)).to_f32())
                .collect(),
            Dtype::F32 => tensor
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect(),
            _ => anyhow::bail!("reference {name} is not BF16/FP32"),
        };
        ensure!(
            result.iter().all(|v| v.is_finite()),
            "nonfinite reference {name}"
        );
        Ok(result)
    }

    fn ids(tensors: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<u32>> {
        let tensor = tensors.tensor(name)?;
        ensure!(
            tensor.dtype() == Dtype::I64 && tensor.shape() == shape,
            "reference {name} id layout mismatch"
        );
        tensor
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|v| Ok(u32::try_from(i64::from_le_bytes(*v))?))
            .collect()
    }

    fn native(
        device: &Device,
        path: &Path,
        name: &str,
        shape: &[usize],
    ) -> Result<CudaSlice<bf16>> {
        let mapped = crate::checkpoint::MappedCheckpoint::open(path)?;
        let views = mapped.views()?;
        let shard = mapped
            .index
            .weight_map
            .get(name)
            .context("native tensor absent")?;
        let tensor = views[shard].tensor(name)?;
        ensure!(
            tensor.dtype() == Dtype::BF16 && tensor.shape() == shape,
            "native {name} layout mismatch"
        );
        let data: Vec<_> = tensor
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| bf16::from_bits(u16::from_le_bytes(*v)))
            .collect();
        ensure!(
            data.iter().all(|v| v.to_f32().is_finite()),
            "nonfinite native {name}"
        );
        Ok(device.stream.clone_htod(&data)?)
    }

    fn coefficient(config: &DraftConfig, context: usize, layers: usize, extra: u32) -> f64 {
        let h = config.hidden_size;
        let d = config.head_dim;
        let keys = context + config.block_size;
        let mut reductions = vec![config.capture_width(), h];
        for _ in 0..layers {
            reductions.extend([h, h, d, d, keys, keys, h, h, h, config.intermediate_size]);
        }
        // Per layer: input norm2, QKV1, Q/K norm2, RoPE3, probability1, PV1, O1, add1,
        // post norm2, gate/up1, SiLU1, product1, down1, add1.
        let rounds = 3 + 19 * layers as u32 + extra;
        let chain = super::super::bf16_chain_bound(rounds, &reductions);
        let norm = super::super::bf16_chain_bound(2 * layers as u32, &[h, d]);
        let silu = super::super::bf16_chain_bound(2 * layers as u32, &[h]) / std::f64::consts::E;
        let softmax = super::super::gamma(keys, 2f64.powi(-24)) * 0.5;
        chain + norm + silu + softmax
    }

    fn compare(
        name: &str,
        actual: &[f32],
        expected: (&[f32], &[f32]),
        coefficient: f64,
        width: usize,
    ) -> Result<usize> {
        let (reference, floor) = expected;
        ensure!(
            actual.len() == reference.len()
                && floor.len() == reference.len()
                && width > 0
                && reference.len().is_multiple_of(width),
            "{name} extent mismatch"
        );
        let (error, _) = crate::verify::max_abs_rmse(name, actual, reference)?;
        ensure!(
            floor.iter().all(|v| v.is_finite() && *v >= 0.0),
            "Runner FAIL {name}: non-finite floor"
        );
        let mut failures = 0;
        let mut maximum = 0f64;
        for ((actual, reference), floor) in actual
            .chunks_exact(width)
            .zip(reference.chunks_exact(width))
            .zip(floor.chunks_exact(width))
        {
            let rms = (reference.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / width as f64)
                .sqrt();
            for ((a, b), f) in actual.iter().zip(reference).zip(floor) {
                let delta = (f64::from(*a) - f64::from(*b)).abs();
                let bound = coefficient * f64::from(*b).abs().max(rms) + f64::from(*f);
                failures += usize::from(delta > bound);
                maximum = maximum.max(if bound == 0.0 {
                    if delta == 0.0 { 0.0 } else { f64::INFINITY }
                } else {
                    delta / bound
                });
            }
        }
        println!(
            "T19_TENSOR name={name} max_abs={error} max_bound_ratio={maximum} coefficient={coefficient} violations={failures}"
        );
        Ok(failures)
    }

    fn confidence_bounds(
        c: &DraftConfig,
        path: &Path,
        reference: (&[f32], &[f32]),
        parents: &[u32],
        propagated: f64,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let (hidden, floor) = reference;
        ensure!(
            hidden.len() == c.block_size * c.hidden_size
                && floor.len() == c.block_size
                && parents.len() == c.block_size,
            "confidence reference extent mismatch"
        );
        ensure!(
            parents.iter().all(|p| (*p as usize) < c.vocab_size),
            "confidence parent outside vocabulary"
        );
        let mapped = crate::checkpoint::MappedCheckpoint::open(path)?;
        let views = mapped.views()?;
        let table = &views[mapped
            .index
            .weight_map
            .get("confidence_head.proj.weight")
            .context("confidence shard absent")?];
        let weight = values(
            table,
            "confidence_head.proj.weight",
            &[1, c.hidden_size + c.markov_rank],
        )?;
        let bias = values(table, "confidence_head.proj.bias", &[1])?[0].abs();
        let embedding_table = &views[mapped
            .index
            .weight_map
            .get("markov_head.markov_w1.weight")
            .context("embedding shard absent")?];
        let embedding_tensor = embedding_table.tensor("markov_head.markov_w1.weight")?;
        ensure!(
            embedding_tensor.dtype() == Dtype::BF16
                && embedding_tensor.shape() == [c.vocab_size, c.markov_rank],
            "confidence embedding layout mismatch"
        );
        let local = super::super::bf16_chain_bound(1, &[c.hidden_size + c.markov_rank]);
        let mut complete = Vec::new();
        let mut isolated = Vec::new();
        for (row, &parent) in parents.iter().enumerate() {
            let begin = parent as usize * c.markov_rank;
            let embedding: Vec<_> = embedding_tensor.data()[begin * 2..(begin + c.markov_rank) * 2]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| bf16::from_bits(u16::from_le_bytes(*v)).to_f32())
                .collect();
            let h_product: f64 = hidden[row * c.hidden_size..(row + 1) * c.hidden_size]
                .iter()
                .zip(&weight[..c.hidden_size])
                .map(|(v, w)| f64::from(*v).abs() * f64::from(*w).abs())
                .sum();
            let m_product: f64 = embedding
                .iter()
                .zip(&weight[c.hidden_size..])
                .map(|(v, w)| f64::from(*v).abs() * f64::from(*w).abs())
                .sum();
            let budget =
                0.25 * local * (h_product + m_product + f64::from(bias)) + f64::from(floor[row]);
            isolated.push(budget as f32);
            complete.push((budget + 0.25 * propagated * h_product) as f32);
        }
        Ok((complete, isolated))
    }

    fn staged_fixture(root: &Path, name: &str, shape: &[usize], dtype: &str) -> Result<Vec<u8>> {
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        let entry = &manifest[name];
        ensure!(
            entry["dtype"] == dtype && entry["shape"] == serde_json::json!(shape),
            "staged fixture {name} layout"
        );
        let file = entry["file"].as_str().context("staged fixture filename")?;
        ensure!(
            Path::new(file)
                .file_name()
                .is_some_and(|name| name == std::ffi::OsStr::new(file)),
            "staged fixture filename escapes case"
        );
        let bytes = std::fs::read(root.join(file))?;
        ensure!(
            Some(bytes.len() as u64) == entry["bytes"].as_u64(),
            "staged fixture {name} byte extent"
        );
        Ok(bytes)
    }

    fn staged_bf16(root: &Path, name: &str, shape: &[usize]) -> Result<Vec<bf16>> {
        let raw = staged_fixture(root, name, shape, "torch.bfloat16")?;
        ensure!(raw.len() == shape.iter().product::<usize>() * 2);
        let values: Vec<_> = raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|word| bf16::from_bits(u16::from_le_bytes(*word)))
            .collect();
        ensure!(values.iter().all(|v| v.is_finite()), "nonfinite {name}");
        Ok(values)
    }

    fn staged_f64(path: &Path, cells: usize) -> Result<Vec<f64>> {
        let raw = std::fs::read(path)?;
        ensure!(raw.len() == cells * 8, "staged FP64 certificate extent");
        let result: Vec<_> = raw
            .as_chunks::<8>()
            .0
            .iter()
            .map(|word| f64::from_le_bytes(*word))
            .collect();
        ensure!(
            result.iter().all(|v| v.is_finite()),
            "nonfinite FP64 oracle"
        );
        Ok(result)
    }

    #[test]
    #[ignore = "GPU lease; FF_DS41_MODEL and FF_MARKOV_FIXTURES with the frozen staged oracle"]
    fn staged_native_heads_gpu() -> Result<()> {
        use super::super::{contracts::view, linear::Bf16F32Plan, moe::testing::capture};
        ensure!(
            std::env::var("FF_DS41_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let model = PathBuf::from(std::env::var("FF_DS41_MODEL")?);
        let root = PathBuf::from(std::env::var("FF_MARKOV_FIXTURES")?);
        let oracle = root.join("head-oracle-r2");
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(oracle.join("manifest.json"))?)?;
        let frozen: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("reference-tolerances-r2.json"))?)?;
        ensure!(
            manifest["candidate_seen"] == false
                && manifest["factor"] == 2
                && manifest["freeze"] == "reference-tolerances-r2.json"
                && frozen["candidate_seen"] == false
                && frozen["tolerances_frozen"] == true
                && frozen["factor"] == 2,
            "staged numerical contract is not frozen r2"
        );
        let spec = crate::model_spec::ModelSpec::read(&model)?;
        let ds = spec.deepseek_v41.as_ref().context("staged geometry")?;
        let (h, v, rank, rows) = (
            spec.hidden as usize,
            spec.vocab as usize,
            ds.dspark_markov_rank as usize,
            ds.dspark_block_size as usize,
        );
        ensure!(
            (h, v, rank, rows) == (5120, 129280, 256, 5),
            "staged head production geometry"
        );
        let cases = manifest["fixtures"].as_array().context("head fixtures")?;
        ensure!(cases.len() == 20, "staged head fixture count");
        for (name, width, columns) in [
            ("base_head", h, v),
            ("markov_head", rank, v),
            ("confidence", h + rank, 1),
        ] {
            ensure!(
                manifest["linear"][name]["rows"] == 100
                    && manifest["linear"][name]["columns"] == columns
                    && manifest["linear"][name]["width"] == width
                    && frozen["linear_certificates"][name]["width"] == width,
                "staged {name} certificate geometry"
            );
        }
        let read_certificate = |name: &str, kind: &str, columns| {
            staged_f64(&oracle.join(format!("{name}.{kind}.f64")), 100 * columns)
        };
        let base_exact = read_certificate("base_head", "exact", v)?;
        let base_bound = read_certificate("base_head", "bound", v)?;
        let markov_exact = read_certificate("markov_head", "exact", v)?;
        let markov_bound = read_certificate("markov_head", "bound", v)?;
        let confidence_exact = read_certificate("confidence", "exact", 1)?;
        let confidence_bound = read_certificate("confidence", "bound", 1)?;
        ensure!(
            base_bound
                .iter()
                .chain(&markov_bound)
                .chain(&confidence_bound)
                .all(|v| *v >= 0.0),
            "negative staged arithmetic bound"
        );
        let d = Device::new(0)?;
        let s = &d.stream;
        let head = native(&d, &model, "head.weight", &[v, h])?;
        let embedding = native(&d, &model, "mtp.2.markov_head.embed.weight", &[v, rank])?;
        let markov = native(&d, &model, "mtp.2.markov_head.head.weight", &[v, rank])?;
        let confidence = native(
            &d,
            &model,
            "mtp.2.confidence_head.proj.weight",
            &[1, h + rank],
        )?;
        let module = super::super::cubin::module(&d.ctx, "markov")?;
        let sum = module.load_function("sum_argmax_f32")?;
        let old_sum = module.load_function("sum_argmax")?;
        let dot = module.load_function("concat_dot")?;
        d.finish_upload()?;
        let mut defects = 0usize;
        let mut projections = 0usize;
        let mut graph_cases = 0usize;
        let mut control = 0usize;
        let mut min_margin = f32::INFINITY;
        let mut check = |name: &str, actual: &[f32], exact: &[f64], bound: &[f64]| -> Result<()> {
            ensure!(
                actual.len() == exact.len() && actual.len() == bound.len(),
                "{name} certificate layout"
            );
            ensure!(actual.iter().all(|v| v.is_finite()), "nonfinite {name}");
            let violations = actual
                .iter()
                .zip(exact.iter().zip(bound))
                .filter(|(a, (e, b))| (f64::from(**a) - **e).abs() > **b)
                .count();
            defects += violations;
            println!(
                "STAGED_HEAD name={name} cells={} violations={violations}",
                actual.len()
            );
            Ok(())
        };
        let mut bit_defects = 0usize;
        for (case_index, case) in cases.iter().enumerate() {
            let folder = case["folder"].as_str().context("head case folder")?;
            let case_root = root.join(folder);
            ensure!(
                case_root.canonicalize()?.starts_with(root.canonicalize()?),
                "head case outside fixture root"
            );
            let start = case["row_start"].as_u64().context("head row start")? as usize;
            ensure!(start == case_index * rows && case["rows"] == rows);
            let x = s.clone_htod(&staged_bf16(&case_root, "final_norm.0", &[1, rows, h])?)?;
            let hidden = s.clone_htod(&staged_bf16(
                &case_root,
                "stage2.collapse.2",
                &[1, rows, h],
            )?)?;
            let mut parents = Vec::with_capacity(rows * rank);
            for row in 0..rows {
                parents.extend(staged_bf16(
                    &case_root,
                    &format!("markov.{row}.1"),
                    &[1, rank],
                )?);
            }
            let parent_embeddings = s.clone_htod(&parents)?;
            let raw_ids = staged_fixture(&case_root, "output.0.0", &[1, rows + 1], "torch.int64")?;
            ensure!(raw_ids.len() == (rows + 1) * 8);
            let ids = raw_ids
                .as_chunks::<8>()
                .0
                .iter()
                .map(|word| Ok(u32::try_from(i64::from_le_bytes(*word))?))
                .collect::<Result<Vec<_>>>()?;
            ensure!(ids.iter().all(|id| (*id as usize) < v), "head parent id");
            let ids = s.clone_htod(&ids)?;
            let mut base = s.clone_htod(&vec![123.0f32; rows * v + 16])?;
            let mut biases = s.clone_htod(&vec![123.0f32; rows * v + 16])?;
            let mut combined = s.alloc_zeros::<f32>(rows * v)?;
            let mut tokens = s.alloc_zeros::<u32>(rows)?;
            let mut confidences = s.alloc_zeros::<f32>(rows)?;
            let mut plans = Vec::new();
            for (name, inputs, weights, width, output, exact, bound) in [
                ("base", &x, &head, h, &base, &base_exact, &base_bound),
                (
                    "markov",
                    &parent_embeddings,
                    &markov,
                    rank,
                    &biases,
                    &markov_exact,
                    &markov_bound,
                ),
            ] {
                for m in [rows, 1] {
                    for row in (0..rows).step_by(m) {
                        let plan = Bf16F32Plan::new(
                            &d.ctx,
                            view(
                                inputs,
                                s,
                                row * width,
                                [m as u32, width as u32],
                                [width as u64, 1],
                            )?,
                            view(weights, s, 0, [v as u32, width as u32], [width as u64, 1])?,
                            view(output, s, row * v, [m as u32, v as u32], [v as u64, 1])?,
                        )?;
                        plan.run(s)?;
                        let actual = s.clone_dtoh(&output.slice(row * v..(row + m) * v))?;
                        let range = (start + row) * v..(start + row + m) * v;
                        check(name, &actual, &exact[range.clone()], &bound[range])?;
                        projections += 1;
                        if m == rows {
                            plans.push(plan);
                        }
                    }
                }
                bit_defects += s
                    .clone_dtoh(&output.slice(rows * v..))?
                    .iter()
                    .filter(|v| **v != 123.0)
                    .count();
            }
            let run_heads = |combined: &mut CudaSlice<f32>,
                             tokens: &mut CudaSlice<u32>,
                             confidences: &mut CudaSlice<f32>|
             -> Result<()> {
                for row in 0..rows {
                    unsafe {
                        s.launch_builder(&sum)
                            .arg(&base.slice(row * v..(row + 1) * v))
                            .arg(&biases.slice(row * v..(row + 1) * v))
                            .arg(&mut combined.slice_mut(row * v..(row + 1) * v))
                            .arg(&mut tokens.slice_mut(row..row + 1))
                            .arg(&(v as i32))
                            .launch(grid(1, 1024))?;
                    }
                }
                unsafe {
                    s.launch_builder(&dot)
                        .arg(&hidden)
                        .arg(&embedding)
                        .arg(&ids.slice(..1))
                        .arg(&ids.slice(1..))
                        .arg(&confidence)
                        .arg(&mut *confidences)
                        .arg(&(rows as i32))
                        .arg(&(h as i32))
                        .arg(&(rank as i32))
                        .arg(&(v as i32))
                        .launch(grid(rows, 256))?;
                }
                Ok(())
            };
            run_heads(&mut combined, &mut tokens, &mut confidences)?;
            let scores = s.clone_dtoh(&combined)?;
            let eager_confidence = s.clone_dtoh(&confidences)?;
            check(
                "confidence",
                &eager_confidence,
                &confidence_exact[start..start + rows],
                &confidence_bound[start..start + rows],
            )?;
            let actual_tokens = s.clone_dtoh(&tokens)?;
            let base_values = s.clone_dtoh(&base.slice(..rows * v))?;
            let bias_values = s.clone_dtoh(&biases.slice(..rows * v))?;
            for (row, &actual_token) in actual_tokens.iter().enumerate() {
                let mut top = [(f32::NEG_INFINITY, u32::MAX); 2];
                for token in 0..v {
                    let index = row * v + token;
                    let expected = base_values[index] + bias_values[index];
                    bit_defects += usize::from(scores[index].to_bits() != expected.to_bits());
                    if let Some(slot) = top.iter().position(|&(score, id)| {
                        expected > score || (expected == score && (token as u32) < id)
                    }) {
                        top[slot..].rotate_right(1);
                        top[slot] = (expected, token as u32);
                    }
                }
                bit_defects += usize::from(actual_token != top[0].1);
                min_margin = min_margin.min(top[0].0 - top[1].0);
            }
            let graph = capture(s, || {
                for plan in &plans {
                    plan.run(s)?;
                }
                run_heads(&mut combined, &mut tokens, &mut confidences)
            })
            .context("staged head graph capture")?;
            let poison = vec![f32::NAN; rows * v];
            s.memcpy_htod(&poison, &mut base.slice_mut(..rows * v))?;
            s.memcpy_htod(&poison, &mut biases.slice_mut(..rows * v))?;
            s.memcpy_htod(&poison, &mut combined)?;
            s.memcpy_htod(&vec![f32::NAN; rows], &mut confidences)?;
            s.memcpy_htod(&vec![u32::MAX; rows], &mut tokens)?;
            graph.launch()?;
            let replay = s.clone_dtoh(&combined)?;
            let graph_confidence = s.clone_dtoh(&confidences)?;
            ensure!(
                replay
                    .iter()
                    .chain(&graph_confidence)
                    .all(|v| v.is_finite()),
                "nonfinite staged graph output"
            );
            bit_defects += replay
                .iter()
                .zip(&scores)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            bit_defects += graph_confidence
                .iter()
                .zip(&eager_confidence)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            bit_defects += usize::from(s.clone_dtoh(&tokens)? != actual_tokens);
            graph_cases += 1;
            if case_index == 0 {
                let rounded_base = s.clone_htod(
                    &base_values[..v]
                        .iter()
                        .copied()
                        .map(bf16::from_f32)
                        .collect::<Vec<_>>(),
                )?;
                let rounded_bias = s.clone_htod(
                    &bias_values[..v]
                        .iter()
                        .copied()
                        .map(bf16::from_f32)
                        .collect::<Vec<_>>(),
                )?;
                unsafe {
                    s.launch_builder(&old_sum)
                        .arg(&rounded_base)
                        .arg(&rounded_bias)
                        .arg(&mut combined.slice_mut(..v))
                        .arg(&mut tokens.slice_mut(..1))
                        .arg(&(v as i32))
                        .arg(&1i32)
                        .launch(grid(1, 1024))?;
                }
                control = s
                    .clone_dtoh(&combined.slice(..v))?
                    .iter()
                    .zip(&scores[..v])
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                ensure!(control > 0, "premature BF16 score control escaped");
            }
        }
        println!(
            "STAGED_NATIVE_HEADS cases={} projections={projections} graph_cases={graph_cases} full_vocabulary={v} min_argmax_margin={min_margin} premature_bf16_control={control} numerical_defects={defects} bit_defects={bit_defects}",
            cases.len()
        );
        ensure!(
            defects == 0 && bit_defects == 0,
            "staged native head DEFECT"
        );
        println!(
            "PASS STAGED_NATIVE_HEADS cases={} projections={projections} graph_cases={graph_cases} full_vocabulary={v} premature_bf16_detected=true",
            cases.len()
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a GPU window and native target/draft, T14 fixtures and frozen reference floors"]
    fn fixture_forward_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let target = PathBuf::from(std::env::var("FF_ARGMAX_MODEL")?);
        let draft_path = PathBuf::from(std::env::var("FF_ARGMAX_DRAFT")?);
        let reference = PathBuf::from(std::env::var("FF_MARKOV_FIXTURES")?);
        let floor_manifest = PathBuf::from(std::env::var("FF_MINICPM_TOLERANCES")?);
        let floor_root = floor_manifest
            .parent()
            .context("floor manifest parent missing")?;
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&floor_manifest)?)?;
        ensure!(
            manifest["candidate_executed"] == false,
            "reference floors were not frozen before the candidate"
        );
        let config = crate::model_spec::ModelSpec::read(&target)?.config();
        let c = DraftConfig::read(&draft_path, &config)?;
        ensure!(
            c.block_size == 15 && c.hidden_size == 2048 && c.head_dim == 256,
            "T14 native draft geometry required"
        );
        let names = [
            "math-block-0",
            "math-block-1",
            "code-block-0",
            "code-block-1",
            "general-block-0",
            "general-block-1",
        ];
        let mut context_max = 0;
        let mut confidence_limits = std::collections::BTreeMap::new();
        for name in names {
            let data = std::fs::read(reference.join(format!("{name}.safetensors")))?;
            let tensors = SafeTensors::deserialize(&data)?;
            let shape = tensors.tensor("target_taps")?.shape().to_vec();
            ensure!(
                shape.len() == 4
                    && shape[0] == 1
                    && shape[2] == c.target_layer_ids.len()
                    && shape[3] == c.hidden_size,
                "T14 tap layout mismatch"
            );
            context_max = context_max.max(shape[1]);
            for depth in 0..=c.num_hidden_layers {
                println!(
                    "T19_FROZEN_BOUND fixture={name} layers={depth} coefficient={} context={} block={}",
                    coefficient(&c, shape[1], depth, 0),
                    shape[1],
                    c.block_size
                );
            }
            let floor_data = std::fs::read(floor_root.join(format!("{name}-floor.safetensors")))?;
            let floor = SafeTensors::deserialize(&floor_data)?;
            let b = c.block_size;
            let h = c.hidden_size;
            let block_ids = ids(&tensors, "draft_input_ids", &[1, b])?;
            let proposals = ids(&tensors, "proposals_tensor", &[b])?;
            let parents: Vec<_> = std::iter::once(block_ids[0])
                .chain(proposals.into_iter().take(b - 1))
                .collect();
            let limits = confidence_bounds(
                &c,
                &draft_path,
                (
                    &values(&tensors, "draft_norm_hidden", &[1, b, h])?,
                    &values(&floor, "confidence_c_k", &[b])?,
                ),
                &parents,
                coefficient(&c, shape[1], c.num_hidden_layers, 2),
            )?;
            println!(
                "T19_CONFIDENCE_BOUND fixture={name} sigmoid_lipschitz=0.25 full_max={} isolated_max={}",
                limits.0.iter().copied().fold(0.0f32, f32::max),
                limits.1.iter().copied().fold(0.0f32, f32::max)
            );
            confidence_limits.insert(name, limits);
        }
        let capacity = context_max
            .checked_add(c.block_size)
            .context("draft fixture capacity overflow")?
            .next_multiple_of(32);
        let rows = context_max.max(c.block_size + 1);
        let runtime = PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?);
        std::fs::create_dir_all(&runtime)?;
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        let closure = Rc::new(super::super::closure::Closure::new(
            true,
            "T19 frozen native fixture calibration".into(),
        ));
        let mut blas = Blas::new(&device, &ops, &runtime, closure.clone())?;
        let mut draft = Dense::new(
            &draft_path,
            &device,
            &config,
            capacity,
            rows,
            &mut blas,
            super::super::markov::ScoreRounding::F32,
        )?;
        draft.grow(rows)?;
        let embedding = native(
            &device,
            &target,
            "model.language_model.embed_tokens.weight",
            &[c.vocab_size, c.hidden_size],
        )?;
        let head = native(
            &device,
            &target,
            "lm_head.weight",
            &[c.vocab_size, c.hidden_size],
        )?;
        let profile = super::super::closure::Profile::draft(
            &draft.config,
            capacity,
            rows,
            &target,
            &draft_path,
        )?;
        draft.prepare_linears(&head, &mut blas, &profile, true)?;
        draft.prepare_linears(&head, &mut blas, &profile, false)?;
        draft.setup_attention(&device, &ops, &config, &runtime, &closure)?;
        draft.setup_markov(&device, &ops, &runtime, &closure)?;
        let s = &device.stream;
        let mut defects = 0;
        let mut rejected_control = 0;
        let mut graph_replays = 0;
        let mut native_requests = 0;
        let mut reclassified_cells = 0;
        for name in names {
            let data = std::fs::read(reference.join(format!("{name}.safetensors")))?;
            let tensors = SafeTensors::deserialize(&data)?;
            let floor_data = std::fs::read(floor_root.join(format!("{name}-floor.safetensors")))?;
            let floors = SafeTensors::deserialize(&floor_data)?;
            let context = tensors.tensor("target_taps")?.shape()[1];
            let h = c.hidden_size;
            let b = c.block_size;
            let v = c.vocab_size;
            let block_ids = ids(&tensors, "draft_input_ids", &[1, b])?;
            let positions = ids(&tensors, "draft_position_ids", &[1, b])?;
            let ctx_positions = ids(&tensors, "ctx_position_ids", &[1, context])?;
            ensure!(
                positions
                    .iter()
                    .enumerate()
                    .all(|(i, p)| *p as usize == context + i)
                    && ctx_positions
                        .iter()
                        .enumerate()
                        .all(|(i, p)| *p as usize == i),
                "T14 frozen absolute positions changed"
            );
            ensure!(
                block_ids[1..].iter().all(|id| *id == c.mask_token_id),
                "T14 mask layout changed"
            );
            let expected_ids = ids(&tensors, "proposals_tensor", &[b])?;
            let expected_hidden = values(&tensors, "draft_norm_hidden", &[1, b, h])?;
            let expected_confidence = values(&tensors, "confidence_c_k", &[b])?;
            let final_coefficient = coefficient(&c, context, c.num_hidden_layers, 2);
            let (confidence_budget, isolated_budget) = confidence_limits
                .get(name)
                .context("confidence budget absent")?;
            let mut taps: Vec<_> = values(&tensors, "target_taps", &[1, context, 5, h])?
                .into_iter()
                .map(bf16::from_f32)
                .collect();
            taps.resize(rows * c.capture_width(), bf16::ZERO);
            let taps = s.clone_htod(&taps)?;
            draft.position_shift = 0;
            draft.reset_setup()?;
            let poison = vec![bf16::from_f32(-17.0); capacity * c.kv_dim()];
            for cache in &mut draft.kv {
                s.memcpy_htod(&poison, &mut cache.k)?;
                s.memcpy_htod(&poison, &mut cache.v)?;
            }
            let mut trace = Trace::default();
            let bound_plans = blas.fixed_plan_count();
            draft.inject(
                &taps,
                context,
                0,
                Compute {
                    ops: &ops,
                    blas: &mut blas,
                },
                Some((&mut trace, "context")),
            )?;
            ensure!(
                blas.fixed_plan_count() == bound_plans,
                "draft context injection created a plan after setup"
            );
            let proposals = draft.propose(
                block_ids[0],
                context,
                (&embedding, &head),
                Compute {
                    ops: &ops,
                    blas: &mut blas,
                },
                Some((&mut trace, "draft")),
                Some(&expected_ids),
            )?;
            let mut fixture_defects = 0;
            for (point, key, coef) in [
                (
                    "context.c_projected",
                    "h_ctx_pre_norm",
                    super::super::bf16_chain_bound(1, &[c.capture_width()]),
                ),
                (
                    "context.c_committed",
                    "h_ctx",
                    coefficient(&c, context, 0, 0),
                ),
            ] {
                fixture_defects += compare(
                    &format!("{name}.{point}"),
                    &trace.values(point)?,
                    (
                        &values(&tensors, key, &[1, context, h])?,
                        &values(&floors, key, &[1, context, h])?,
                    ),
                    coef,
                    h,
                )?;
            }
            for layer in 0..c.num_hidden_layers {
                let key = format!("draft_layer_{layer}");
                fixture_defects += compare(
                    &format!("{name}.{key}"),
                    &trace.values(&format!("draft.layer.{layer}"))?,
                    (
                        &values(&tensors, &key, &[1, b, h])?,
                        &values(&floors, &key, &[1, b, h])?,
                    ),
                    coefficient(&c, context, layer + 1, 0),
                    h,
                )?;
            }
            fixture_defects += compare(
                &format!("{name}.final_norm"),
                &trace.values("draft.H")?,
                (
                    &expected_hidden,
                    &values(&floors, "draft_norm_hidden", &[1, b, h])?,
                ),
                final_coefficient,
                h,
            )?;
            for row in [0, b - 1] {
                for (point, key, coef) in [
                    (
                        "draft.B",
                        format!("u_logits_k{row}"),
                        coefficient(&c, context, c.num_hidden_layers, 3),
                    ),
                    (
                        "draft.M",
                        format!("b_logits_k{row}"),
                        super::super::bf16_chain_bound(1, &[c.markov_rank]),
                    ),
                    (
                        "draft.s",
                        format!("combined_logits_k{row}"),
                        coefficient(&c, context, c.num_hidden_layers, 3)
                            + super::super::bf16_chain_bound(1, &[c.markov_rank])
                            + super::super::gamma(1, 2f64.powi(-24)),
                    ),
                ] {
                    let actual = trace.values(point)?;
                    fixture_defects += compare(
                        &format!("{name}.{key}"),
                        &actual[row * v..(row + 1) * v],
                        (
                            &values(&tensors, &key, &[v])?,
                            &values(&floors, &key, &[v])?,
                        ),
                        coef,
                        v,
                    )?;
                }
            }
            let confidence = trace.values("draft.conf")?;
            ensure!(
                confidence.iter().all(|p| (0.0..=1.0).contains(p)),
                "nonfinite/out-of-range confidence"
            );
            fixture_defects += compare(
                &format!("{name}.confidence"),
                &confidence,
                (&expected_confidence, confidence_budget),
                0.0,
                b,
            )?;
            let supplied: Vec<_> = expected_hidden
                .iter()
                .copied()
                .map(bf16::from_f32)
                .collect();
            s.memcpy_htod(&supplied, &mut draft.n)?;
            unsafe {
                s.launch_builder(&draft.concat_dot_sigmoid)
                    .arg(&draft.n)
                    .arg(&draft.weights.w1)
                    .arg(&draft.ids)
                    .arg(&draft.forced_tokens)
                    .arg(&draft.weights.confidence)
                    .arg(&draft.weights.confidence_bias)
                    .arg(&mut draft.confidence_values)
                    .arg(&(b as i32))
                    .arg(&(h as i32))
                    .arg(&(c.markov_rank as i32))
                    .arg(&(v as i32))
                    .launch(grid(b, 256))?;
            }
            fixture_defects += compare(
                &format!("{name}.confidence_isolated"),
                &s.clone_dtoh(&draft.confidence_values)?,
                (&expected_confidence, isolated_budget),
                0.0,
                b,
            )?;
            let top = values(&tensors, "combined_top32_vals", &[b, 32])?;
            let top_floor = values(&floors, "combined_top32_vals", &[b, 32])?;
            let top_ids = ids(&tensors, "combined_top32_ids", &[b, 32])?;
            let qualified: Vec<_> = (0..b)
                .map(|r| top[r * 32] - top[r * 32 + 1] > top_floor[r * 32] + top_floor[r * 32 + 1])
                .collect();
            let winner_valid = |row: usize, token: u32| -> Result<bool> {
                if qualified[row] {
                    return Ok(token == expected_ids[row]);
                }
                let lo = row * 32;
                let lower = top[lo] - top_floor[lo];
                ensure!(
                    top[lo + 31] + top_floor[lo + 31] < lower,
                    "ambiguity interval exceeds stored top32"
                );
                Ok(top_ids[lo..lo + 32]
                    .iter()
                    .position(|id| *id == token)
                    .is_some_and(|index| top[lo + index] + top_floor[lo + index] >= lower))
            };
            let mut teacher_id_defects = 0;
            for (row, actual) in proposals.iter().enumerate() {
                teacher_id_defects += usize::from(!winner_valid(row, *actual)?);
            }
            fixture_defects += teacher_id_defects;
            let scores = trace.values("draft.s")?;
            let mut actual_top = Vec::with_capacity(b * 32);
            for row in scores.chunks_exact(v) {
                let mut sorted = row.to_vec();
                sorted.select_nth_unstable_by(31, |a, b| b.total_cmp(a));
                sorted[..32].sort_unstable_by(|a, b| b.total_cmp(a));
                actual_top.extend_from_slice(&sorted[..32]);
            }
            fixture_defects += compare(
                &format!("{name}.combined_top32_vals"),
                &actual_top,
                (&top, &top_floor),
                coefficient(&c, context, c.num_hidden_layers, 3)
                    + super::super::bf16_chain_bound(1, &[c.markov_rank])
                    + super::super::gamma(1, 2f64.powi(-24)),
                32,
            )?;
            for (row, (actual, expected)) in proposals.iter().zip(&expected_ids).enumerate() {
                if actual != expected {
                    let reference_actual = top_ids[row * 32..(row + 1) * 32]
                        .iter()
                        .position(|id| id == actual)
                        .map(|index| top[row * 32 + index]);
                    println!(
                        "T19_PROPOSAL_RAW fixture={name} row={row} expected={expected} actual={actual} candidate_expected={} candidate_actual={} reference_expected={} reference_actual={reference_actual:?} reference_margin={} detection_threshold={} qualified={} winner_valid={}",
                        scores[row * v + *expected as usize],
                        scores[row * v + *actual as usize],
                        top[row * 32],
                        top[row * 32] - top[row * 32 + 1],
                        top_floor[row * 32] + top_floor[row * 32 + 1],
                        qualified[row],
                        winner_valid(row, *actual)?,
                    );
                }
            }
            let margin = (0..b)
                .map(|r| top[r * 32] - top[r * 32 + 1])
                .fold(f32::INFINITY, f32::min);
            println!(
                "T19_TEACHER fixture={name} proposal_mismatches={} min_top12_margin={margin} defects={fixture_defects}",
                proposals
                    .iter()
                    .zip(&expected_ids)
                    .filter(|(a, b)| a != b)
                    .count()
            );
            defects += fixture_defects;

            draft.reset_setup()?;
            draft.inject(
                &taps,
                context,
                0,
                Compute {
                    ops: &ops,
                    blas: &mut blas,
                },
                None,
            )?;
            let free = draft.propose(
                block_ids[0],
                context,
                (&embedding, &head),
                Compute {
                    ops: &ops,
                    blas: &mut blas,
                },
                None,
                None,
            )?;
            let first_ambiguity = qualified.iter().position(|valid| !valid);
            let mut free_id_defects = 0;
            for (row, actual) in free.iter().enumerate() {
                if first_ambiguity.is_none_or(|first| row <= first) {
                    free_id_defects += usize::from(!winner_valid(row, *actual)?);
                }
            }
            defects += free_id_defects;
            let mut expected_bits = Vec::new();
            draft.state_bits(context, b, &mut expected_bits)?;
            let anchor = s.clone_htod(&[block_ids[0]])?;
            let prefix = s.clone_htod(&[context as i32])?;
            s.begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )?;
            let launched = draft.propose_device(
                &anchor,
                &prefix,
                context,
                (&embedding, &head),
                Compute {
                    ops: &ops,
                    blas: &mut blas,
                },
            );
            let captured = s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
            launched?;
            let graph = captured?.context("empty draft graph")?;
            for _ in 0..2 {
                graph.launch()?;
                s.synchronize()?;
                let mut actual = Vec::new();
                draft.state_bits(context, b, &mut actual)?;
                defects += usize::from(actual != expected_bits);
                graph_replays += 1;
            }
            drop(graph);
            let requests: Vec<_> = (0..b)
                .map(|row| {
                    (
                        row as u8,
                        if row == 0 {
                            block_ids[0]
                        } else {
                            expected_ids[row - 1]
                        },
                    )
                })
                .collect();
            let first = draft.distributions_batch(&requests)?;
            let expanded: Vec<_> = requests.iter().copied().cycle().take(64).collect();
            let many = draft.distributions_batch(&expanded)?;
            for (i, actual) in many.iter().enumerate() {
                defects += usize::from(!actual.bits_equal(&first[i % b]));
            }
            for chunk in [1, 8] {
                let mut start = 0;
                for requests in requests.chunks(chunk) {
                    let actual = draft.distributions_batch(requests)?;
                    for (index, value) in actual.iter().enumerate() {
                        defects += usize::from(!value.bits_equal(&first[start + index]));
                    }
                    native_requests += actual.len();
                    start += actual.len();
                }
            }
            let mut native_exact_defects = 0;
            for (row, actual) in first.iter().enumerate() {
                native_exact_defects += usize::from(actual.tokens[0] != proposals[row]);
                native_exact_defects += usize::from(!winner_valid(row, actual.tokens[0])?);
            }
            defects += native_exact_defects;
            let raw_teacher = proposals
                .iter()
                .zip(&expected_ids)
                .filter(|(a, b)| a != b)
                .count();
            let raw_free = free
                .iter()
                .zip(&expected_ids)
                .filter(|(a, b)| a != b)
                .count();
            let raw_native = first
                .iter()
                .zip(&expected_ids)
                .filter(|(a, b)| a.tokens[0] != **b)
                .count();
            let reclassified = usize::from(raw_teacher > 0 && teacher_id_defects == 0)
                + usize::from(raw_free > 0 && free_id_defects == 0)
                + usize::from(raw_native > 0 && native_exact_defects == 0);
            reclassified_cells += reclassified;
            println!(
                "T19_ID_CORRECTION fixture={name} raw_teacher={raw_teacher} raw_free={raw_free} raw_native_reference={raw_native} qualified_rows={} ambiguous_rows={} first_ambiguity={first_ambiguity:?} teacher_id_defects={teacher_id_defects} free_id_defects={free_id_defects} native_same_input_exact_defects={native_exact_defects} reclassified_scored_cells={reclassified}",
                qualified.iter().filter(|v| **v).count(),
                qualified.iter().filter(|v| !**v).count()
            );
            native_requests += requests.len() + expanded.len();
            if name == names[0] {
                let maximum = s
                    .clone_dtoh(&draft.weights.w2)?
                    .iter()
                    .map(|v| v.to_f32().abs())
                    .fold(0.0f32, f32::max);
                let parent = block_ids[0] as usize;
                let embedding = s.clone_dtoh(
                    &draft
                        .weights
                        .w1
                        .slice(parent * c.markov_rank..(parent + 1) * c.markov_rank),
                )?;
                let magnitude: f32 =
                    embedding.iter().map(|v| v.to_f32().abs()).sum::<f32>() * maximum;
                let mut base = vec![bf16::ZERO; v];
                base[v - 1] = bf16::from_f32(2.0 * magnitude + 1.0);
                s.memcpy_htod(&base, &mut draft.base.slice_mut((b - 1) * v..b * v))?;
                let tail = draft.distributions_batch(&[((b - 1) as u8, block_ids[0])])?;
                defects += usize::from(tail[0].tokens[0] as usize != v - 1);
                native_requests += 1;
                println!(
                    "T19_VOCAB_TAIL vocab={v} row={} expected={} actual={} defects={defects}",
                    b - 1,
                    v - 1,
                    tail[0].tokens[0]
                );
            }

            draft.position_shift = 257;
            draft.reset_setup()?;
            draft.inject(
                &taps,
                context,
                0,
                Compute {
                    ops: &ops,
                    blas: &mut blas,
                },
                None,
            )?;
            let mut wrong = Trace::default();
            draft.propose(
                block_ids[0],
                context,
                (&embedding, &head),
                Compute {
                    ops: &ops,
                    blas: &mut blas,
                },
                Some((&mut wrong, "draft")),
                Some(&expected_ids),
            )?;
            let mut rejected = 0;
            for layer in 0..c.num_hidden_layers {
                let key = format!("draft_layer_{layer}");
                rejected += compare(
                    &format!("{name}.wrong_position.{key}"),
                    &wrong.values(&format!("draft.layer.{layer}"))?,
                    (
                        &values(&tensors, &key, &[1, b, h])?,
                        &values(&floors, &key, &[1, b, h])?,
                    ),
                    coefficient(&c, context, layer + 1, 0),
                    h,
                )?;
            }
            for row in [0, b - 1] {
                let key = format!("combined_logits_k{row}");
                let actual = wrong.values("draft.s")?;
                rejected += compare(
                    &format!("{name}.wrong_position.{key}"),
                    &actual[row * v..(row + 1) * v],
                    (
                        &values(&tensors, &key, &[v])?,
                        &values(&floors, &key, &[v])?,
                    ),
                    coefficient(&c, context, c.num_hidden_layers, 3)
                        + super::super::bf16_chain_bound(1, &[c.markov_rank]),
                    v,
                )?;
            }
            rejected_control += usize::from(rejected > 0);
            defects += usize::from(rejected == 0);
            println!(
                "T19_FIXTURE fixture={name} free_proposal_mismatches={} wrong_position_violations={rejected} defects={defects}",
                free.iter()
                    .zip(&expected_ids)
                    .filter(|(a, b)| a != b)
                    .count()
            );
        }
        println!(
            "T19_DRAFT fixtures=6 graph_replays={graph_replays} native_markov_requests={native_requests} wrong_position_rejected={rejected_control} reclassified_scored_cells={reclassified_cells} defects={defects} {}",
            if defects == 0 { "PASS" } else { "DEFECT" }
        );
        ensure!(defects == 0, "T19 draft has {defects} finite defects");
        Ok(())
    }
}
