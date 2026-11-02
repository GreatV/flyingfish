use super::Device;
use super::blas::Blas;
use super::contracts::{RowDomain, Scratch, SeqRef, Tensor, TensorMut, view};
use super::flash::{Flash, Query, Score, Shape};
use super::gdn::{GdnBlockBindings, GdnBlockMode, GdnBlockPlan};
use super::linear::LinearWeightRef;
use super::moe::prefill::{MoePrefillContribute, MoePrefillFinish, PrefillOps};
use super::moe::route::{BlockRouteBindings, BlockRoutePlan, RouteOps};
use super::ops::{
    CommonOpsRef, GatedRmsInput, GatedRmsMode, Ops, ResidualRmsInput, RmsMode, RmsOptions,
    RotaryPrefixKv, flat,
};
use super::weights::{CohortStorage, WeightCatalog};
use crate::backend::gpu::{CudaSlice, CudaStream, DevicePtr, PushKernelArg};
use crate::gdn_plan::GdnWorkspace;
use crate::model_spec::{ExpertFormat, RoutedWeightBoundary, SwigluRounding};
use crate::op_contract::{ExecutionPhase, GdnSpec};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::sync::Arc;

#[cfg(all(test, feature = "cuda"))]
#[path = "block_tests.rs"]
mod tests;

fn dense<T: crate::backend::gpu::DeviceRepr + Copy>(
    buffer: &CudaSlice<T>,
    stream: &Arc<CudaStream>,
    rows: u32,
    width: u32,
) -> Result<Tensor<T, 2>> {
    view(buffer, stream, 0, [rows, width], [u64::from(width), 1])
}

fn heads(
    buffer: &CudaSlice<bf16>,
    stream: &Arc<CudaStream>,
    rows: u32,
    heads: u32,
    dim: u32,
) -> Result<Tensor<bf16, 3>> {
    view(
        buffer,
        stream,
        0,
        [rows, heads, dim],
        [u64::from(heads) * u64::from(dim), u64::from(dim), 1],
    )
}

struct Projection {
    weight: Tensor<bf16, 2>,
    output: CudaSlice<bf16>,
    rows: u32,
}

impl Projection {
    fn prepare(device: &Device, blas: &Blas, weight: Tensor<bf16, 2>, rows: u32) -> Result<Self> {
        weight.validate()?;
        ensure!(
            weight.stride == [u64::from(weight.shape[1]), 1],
            "block projection weight layout"
        );
        blas.prepare_request_fixed(rows, weight.shape[0], weight.shape[1])?;
        Ok(Self {
            weight,
            output: device
                .stream
                .alloc_zeros(rows as usize * weight.shape[0] as usize)?,
            rows,
        })
    }

    fn run(&self, stream: &Arc<CudaStream>, blas: &Blas, input: Tensor<bf16, 2>) -> Result<()> {
        ensure!(
            input.shape[0] > 0 && input.shape[0] <= self.rows,
            "projection active rows exceed capacity"
        );
        blas.linear_typed(
            ExecutionPhase::Prefill,
            LinearWeightRef::Bf16(self.weight),
            input,
            &mut TensorMut(dense(
                &self.output,
                stream,
                input.shape[0],
                self.weight.shape[0],
            )?),
        )
    }

    fn tensor(&self, stream: &Arc<CudaStream>) -> Result<Tensor<bf16, 2>> {
        dense(&self.output, stream, self.rows, self.weight.shape[0])
    }
}

pub struct GdnBlockWeights {
    pub spec: GdnSpec,
    pub qkv: Tensor<bf16, 2>,
    pub a: Tensor<bf16, 2>,
    pub b: Tensor<bf16, 2>,
    pub z: Tensor<bf16, 2>,
    pub output: Tensor<bf16, 2>,
    pub conv: Tensor<bf16, 2>,
    pub right: Tensor<bf16, 2>,
    pub decay: [Tensor<bf16, 1>; 2],
    pub softmax: [Tensor<bf16, 1>; 3],
    pub norm: Tensor<bf16, 1>,
}

pub struct GdnBlockMixer {
    qkv: Projection,
    a: Projection,
    b: Projection,
    z: Projection,
    output: Projection,
    plans: [GdnBlockPlan; 2],
    tail: Option<GdnBlockPlan>,
    core: CudaSlice<bf16>,
    normalized: CudaSlice<bf16>,
    norm: Tensor<bf16, 1>,
    heads: u32,
    pub state: CudaSlice<f32>,
    pub history: CudaSlice<f32>,
    _scratch: CudaSlice<u8>,
    _valid: CudaSlice<i32>,
}

impl GdnBlockMixer {
    pub fn prepare(
        device: &Device,
        blas: &Blas,
        rows: u32,
        weights: GdnBlockWeights,
        state: CudaSlice<f32>,
        history: CudaSlice<f32>,
    ) -> Result<Self> {
        let s = &device.stream;
        let spec = weights.spec;
        let channels = spec.conv_dim()?;
        let width = spec.hv * 128;
        ensure!(
            weights.qkv.shape[0] == channels
                && weights.a.shape == [spec.hv, weights.qkv.shape[1]]
                && weights.b.shape == weights.a.shape
                && weights.z.shape == [width, weights.qkv.shape[1]]
                && weights.output.shape == [weights.qkv.shape[1], width],
            "GDN block projections differ from spec"
        );
        let qkv = Projection::prepare(device, blas, weights.qkv, rows)?;
        let a = Projection::prepare(device, blas, weights.a, rows)?;
        let b = Projection::prepare(device, blas, weights.b, rows)?;
        let z = Projection::prepare(device, blas, weights.z, rows)?;
        let output = Projection::prepare(device, blas, weights.output, rows)?;
        let core = s.alloc_zeros(rows as usize * width as usize)?;
        let normalized = s.alloc_zeros(rows as usize * width as usize)?;
        let valid = s.clone_htod(&vec![1i32; rows as usize])?;
        let workspace = GdnWorkspace::new(&spec, rows, false)?;
        let scratch = s.alloc_zeros::<u8>(workspace.bytes as usize)?;
        let state_view = view(&state, s, 0, [spec.hv, 128, 128], [16384, 128, 1])?;
        let history_view = dense(&history, s, channels, 3)?;
        let bindings = || -> Result<_> {
            Ok(GdnBlockBindings {
                qkv: qkv.tensor(s)?,
                a: a.tensor(s)?,
                conv_weight: weights.conv,
                decay: weights.decay,
                valid: view(&valid, s, 0, [rows], [1])?,
                output: TensorMut(heads(&core, s, rows, spec.hv, 128)?),
            })
        };
        let scratch_view = || Scratch {
            ptr: scratch.device_ptr(s).0,
            bytes: workspace.bytes,
        };
        let preview = GdnBlockPlan::prepare(
            &device.ctx,
            spec,
            bindings()?,
            GdnBlockMode::Preview {
                state: state_view,
                history: history_view,
                right: weights.right,
                softmax: weights.softmax,
            },
            scratch_view(),
        )?;
        let commit = GdnBlockPlan::prepare(
            &device.ctx,
            spec,
            bindings()?,
            GdnBlockMode::Commit {
                state: TensorMut(state_view),
                history: TensorMut(history_view),
                b: b.tensor(s)?,
            },
            scratch_view(),
        )?;
        Ok(Self {
            qkv,
            a,
            b,
            z,
            output,
            plans: [preview, commit],
            tail: None,
            core,
            normalized,
            norm: weights.norm,
            heads: spec.hv,
            state,
            history,
            _scratch: scratch,
            _valid: valid,
        })
    }

    fn run(&self, context: &BlockContext<'_>, input: Tensor<bf16, 2>, preview: bool) -> Result<()> {
        let BlockContext {
            stream,
            blas,
            common,
            domain,
            eps,
            ..
        } = context;
        self.qkv.run(stream, blas, input)?;
        self.a.run(stream, blas, input)?;
        if !preview {
            self.b.run(stream, blas, input)?;
        }
        self.z.run(stream, blas, input)?;
        let plan = if input.shape[0] == self.qkv.rows {
            &self.plans[usize::from(!preview)]
        } else {
            ensure!(!preview, "GDN preview requires the full block");
            self.tail
                .as_ref()
                .context("GDN causal tail is not prepared")?
        };
        plan.run(stream)?;
        common.norm.rms_gated(
            stream,
            *domain,
            GatedRmsInput {
                x: heads(&self.core, stream, domain.row_capacity, self.heads, 128)?,
                weight: self.norm,
                gate: heads(&self.z.output, stream, domain.row_capacity, self.heads, 128)?,
                mode: GatedRmsMode::SiluF32,
            },
            &mut TensorMut(heads(
                &self.normalized,
                stream,
                domain.row_capacity,
                self.heads,
                128,
            )?),
            *eps,
        )?;
        self.output.run(
            stream,
            blas,
            dense(
                &self.normalized,
                stream,
                domain.row_capacity,
                self.heads * 128,
            )?,
        )
    }
}

pub struct AttentionBlockWeights {
    pub q_gate: Tensor<bf16, 2>,
    pub k: Tensor<bf16, 2>,
    pub v: Tensor<bf16, 2>,
    pub output: Tensor<bf16, 2>,
    pub q_norm: Tensor<bf16, 1>,
    pub k_norm: Tensor<bf16, 1>,
    pub table: Tensor<bf16, 2>,
    pub heads: u32,
    pub kv_heads: u32,
    pub dim: u32,
}

pub struct AttentionBlockMixer {
    q_gate: Projection,
    key: Projection,
    value: Projection,
    output: Projection,
    q_norm: Tensor<bf16, 1>,
    k_norm: Tensor<bf16, 1>,
    table: Tensor<bf16, 2>,
    q: CudaSlice<bf16>,
    normalized_k: CudaSlice<bf16>,
    attention: CudaSlice<bf16>,
    gated: CudaSlice<bf16>,
    flash: Flash,
    heads: u32,
    kv_heads: u32,
    dim: u32,
    pub k: CudaSlice<bf16>,
    pub v: CudaSlice<bf16>,
}

impl AttentionBlockMixer {
    pub fn prepare(
        device: &Device,
        blas: &Blas,
        rows: u32,
        weights: AttentionBlockWeights,
        k: CudaSlice<bf16>,
        v: CudaSlice<bf16>,
    ) -> Result<Self> {
        let s = &device.stream;
        let hidden = weights.q_gate.shape[1];
        let (h, kv, d) = (weights.heads, weights.kv_heads, weights.dim);
        ensure!(
            h > 0
                && kv > 0
                && h.is_multiple_of(kv)
                && d == 256
                && weights.q_gate.shape == [2 * h * d, hidden]
                && weights.k.shape == [kv * d, hidden]
                && weights.v.shape == weights.k.shape
                && weights.output.shape == [hidden, h * d]
                && k.len() == kv as usize * weights.table.shape[0] as usize * d as usize
                && v.len() == k.len(),
            "attention block geometry/cache extent mismatch"
        );
        Ok(Self {
            q_gate: Projection::prepare(device, blas, weights.q_gate, rows)?,
            key: Projection::prepare(device, blas, weights.k, rows)?,
            value: Projection::prepare(device, blas, weights.v, rows)?,
            output: Projection::prepare(device, blas, weights.output, rows)?,
            q_norm: weights.q_norm,
            k_norm: weights.k_norm,
            table: weights.table,
            q: s.alloc_zeros(rows as usize * (h * d) as usize)?,
            normalized_k: s.alloc_zeros(rows as usize * (kv * d) as usize)?,
            attention: s.alloc_zeros(rows as usize * (h * d) as usize)?,
            gated: s.alloc_zeros(rows as usize * (h * d) as usize)?,
            flash: Flash::new_with_score(
                s.clone(),
                rows as usize,
                h as usize,
                kv as usize,
                d as usize,
                Score::from_tokens((d as f32).sqrt().recip(), None)?,
                weights.table.shape[0] as usize,
            )?,
            heads: h,
            kv_heads: kv,
            dim: d,
            k,
            v,
        })
    }

    fn run(
        &mut self,
        context: &BlockContext<'_>,
        input: Tensor<bf16, 2>,
        prefix: u32,
        preview: bool,
    ) -> Result<()> {
        let BlockContext {
            stream: s,
            blas,
            common,
            domain,
            eps,
            ops,
        } = context;
        let rows = domain.row_capacity;
        let (h, kv, d) = (self.heads, self.kv_heads, self.dim);
        self.q_gate.run(s, blas, input)?;
        self.key.run(s, blas, input)?;
        self.value.run(s, blas, input)?;
        let query = view(
            &self.q_gate.output,
            s,
            0,
            [rows, h, d],
            [u64::from(2 * h * d), u64::from(2 * d), 1],
        )?;
        for (input, weight, destination) in [
            (query, self.q_norm, heads(&self.q, s, rows, h, d)?),
            (
                heads(&self.key.output, s, rows, kv, d)?,
                self.k_norm,
                heads(&self.normalized_k, s, rows, kv, d)?,
            ),
        ] {
            common.norm.rms_with(
                s,
                *domain,
                input,
                weight,
                &mut TensorMut(destination),
                RmsOptions {
                    mode: RmsMode::ZeroCentered,
                    eps: *eps,
                },
            )?;
        }
        let capacity = self.table.shape[0];
        ops.rotary_prefix_kv(
            s,
            RotaryPrefixKv {
                q: TensorMut(heads(&self.q, s, rows, h, d)?),
                key: heads(&self.normalized_k, s, rows, kv, d)?,
                value: heads(&self.value.output, s, rows, kv, d)?,
                k: TensorMut(heads(&self.k, s, kv, capacity, d)?),
                v: TensorMut(heads(&self.v, s, kv, capacity, d)?),
                table: self.table,
                start: prefix,
            },
        )?;
        self.flash.run_masked(Query {
            q: &self.q,
            q_row_stride: (h * d) as usize,
            k: &self.k,
            v: &self.v,
            out: &mut self.attention,
            shape: Shape {
                rows: rows as usize,
                start: prefix as usize,
                capacity: capacity as usize,
                q_heads: h as usize,
                kv_heads: kv as usize,
                head_dim: d as usize,
                causal: !preview,
                window_left: None,
            },
            keys: (prefix + rows) as usize,
            prefix: None,
        })?;
        let gate = view(
            &self.q_gate.output,
            s,
            d as usize,
            [rows, h, d],
            [u64::from(2 * h * d), u64::from(2 * d), 1],
        )?;
        common.gate.sigmoid_mul(
            s,
            *domain,
            heads(&self.attention, s, rows, h, d)?,
            gate,
            &mut TensorMut(heads(&self.gated, s, rows, h, d)?),
        )?;
        self.output
            .run(s, blas, dense(&self.gated, s, rows, h * d)?)
    }
}

pub enum BlockMixer {
    Gdn(Box<GdnBlockMixer>),
    Attention(Box<AttentionBlockMixer>),
}

struct BlockContext<'a> {
    stream: &'a Arc<CudaStream>,
    blas: &'a Blas,
    common: &'a CommonOpsRef,
    ops: &'a Ops,
    domain: RowDomain,
    eps: f32,
}

pub struct PreNormBlock {
    weights: [Tensor<bf16, 1>; 2],
    normalized: CudaSlice<bf16>,
    moe_input: CudaSlice<bf16>,
    moe_output: CudaSlice<bf16>,
    metadata: CudaSlice<i32>,
    common: CommonOpsRef,
    pub mixer: BlockMixer,
    rows: u32,
    hidden: u32,
    eps: f32,
    pub prefix: u32,
}

impl PreNormBlock {
    pub fn model_residency(
        device: &Device,
        spec: &crate::model_spec::ModelSpec,
        manifest: &crate::checkpoint::TensorManifest,
        capacity: u32,
        device_budget_bytes: Option<u64>,
    ) -> Result<crate::experts::plan::BlockResidency> {
        use crate::experts::plan::{BlockResidency, BlockResources};
        use crate::gdn_plan::{add, mul};
        use crate::model_spec::{Architecture, Mixer, RoutedWeightBoundary};
        ensure!(
            spec.architecture == Architecture::Clover
                && capacity >= 32
                && capacity <= spec.max_position,
            "Clover block residency capacity/spec mismatch"
        );
        let rows = 32u64;
        let h = u64::from(spec.hidden);
        let q = u64::from(spec.q_heads) * u64::from(spec.head_dim);
        let kv = u64::from(spec.kv_heads) * u64::from(spec.head_dim);
        let gdn = GdnSpec::from_model(spec)?;
        let gdn_count = spec
            .layers
            .iter()
            .filter(|layer| layer.mixer == Mixer::Gdn)
            .count() as u64;
        let attention_count = spec.layers.len() as u64 - gdn_count;
        let gdn_row = add(
            add(u64::from(gdn.conv_dim()?), 2 * u64::from(gdn.hv))?,
            add(3 * u64::from(gdn.hv) * 128, h)?,
        )?;
        let attention_row = add(add(5 * q, 3 * kv)?, h)?;
        let mixer_buffers = add(
            mul(gdn_count, add(mul(rows * 2, gdn_row)?, rows * 4)?)?,
            mul(attention_count, mul(rows * 2, attention_row)?)?,
        )?;
        let block_buffers = mul(spec.layers.len() as u64, add(mul(rows * 6, h)?, 32)?)?;
        let states = add(
            mul(gdn_count, gdn.image_bytes()?)?,
            mul(attention_count, mul(u64::from(capacity), 4 * kv)?)?,
        )?;
        let mixer_scratch = add(
            mul(
                gdn_count,
                GdnWorkspace::new(&gdn, rows as u32, false)?.bytes,
            )?,
            mul(attention_count, rows * u64::from(spec.q_heads) * 4)?,
        )?;
        let moe = spec
            .moe
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("block MoE geometry missing"))?;
        let moe_scratch = super::moe::prefill::prefill_scratch_bytes_mixed_with_boundary(
            rows as usize,
            rows as usize,
            moe.experts_per_tok as usize,
            moe.experts as usize,
            spec.hidden as usize,
            (
                moe.expert_intermediate as usize,
                moe.shared_intermediate as usize,
            ),
            (None, RoutedWeightBoundary::BeforeContributionBf16),
        )?;
        let assignments = rows * u64::from(moe.experts_per_tok);
        let contributions = mul(assignments, 2 * h)?;
        let routes = add(
            mul(rows, u64::from(moe.experts) * 6)?,
            add(assignments * 24, u64::from(moe.experts) * 36 + rows * 4)?,
        )?;
        let sampler = add(
            mul(rows, u64::from(spec.vocab) * 6)?,
            super::sample::BlockSampler::work_bytes(rows as u32, spec.vocab, spec.eos.len())?,
        )?;
        let rotary = mul(u64::from(capacity), u64::from(spec.rotary_dim) * 2)?;
        let mut work = 4 << 20;
        for bytes in [
            mixer_buffers,
            block_buffers,
            states,
            mixer_scratch,
            moe_scratch,
            contributions,
            routes,
            sampler,
            rotary,
        ] {
            work = add(work, bytes)?;
        }
        let (free, total) = device.ctx.mem_get_info()?;
        ensure!(
            device_budget_bytes != Some(0),
            "block device budget is zero"
        );
        let available = device_budget_bytes.map_or(free as u64, |cap| cap.min(free as u64));
        let host = super::calibrate::host_memory(std::path::Path::new("/"))?;
        let copies = super::calibrate::copy_engines(device)?;
        #[cfg(feature = "cuda")]
        let device_reserve = total / 16;
        #[cfg(feature = "rocm")]
        let device_reserve = total.div_ceil(12);
        let resources = BlockResources {
            device_free_bytes: available,
            host_available_bytes: host,
            device_work_bytes: add(work, device_reserve as u64)?,
            host_work_bytes: add(host / 16, add(contributions, add(rows * h * 2, routes)?)?)?,
            compute_units: u32::try_from(device.info.sms)?,
            copy_engines: copies,
            rows: rows as u32,
            staging_slots_max: spec.moe.as_ref().context("block MoE missing")?.experts,
        };
        let plan = BlockResidency::new(spec, manifest, resources)?;
        println!(
            "BLOCK_RESIDENCY {}",
            serde_json::json!({"capacity":capacity,"rows":rows,
            "device_free":free,"device_total":total,"host_available":host,"compute_units":resources.compute_units,
            "device_budget":device_budget_bytes,"device_available":available,
            "copy_engines":copies,"work_bytes":work,"state_bytes":states,"mixer_buffers":mixer_buffers,
            "mixer_scratch":mixer_scratch,"moe_scratch":moe_scratch,"sampler_reserve":sampler,
            "device_fixed":plan.device_fixed_bytes,"host_fixed":plan.host_fixed_bytes,
            "global_dense":plan.global_dense_bytes,"dense":plan.dense,"expert_slot":plan.experts.slot_bytes(),
            "expert_slots":plan.experts.slots_per_layer,"host_blocks":plan.experts.host_blocks,
            "spare_blocks":plan.experts.spare_blocks,"disk_expert_bytes":plan.experts.disk_bytes,
            "minimum_cohort":plan.minimum_cohort,"arithmetic":"GPU","source":"setup counts and capacity"})
        );
        Ok(plan)
    }

    #[cfg(test)]
    pub(crate) fn boundaries(&self) -> [(&'static str, &CudaSlice<bf16>); 4] {
        let mixer = match &self.mixer {
            BlockMixer::Gdn(m) => &m.output.output,
            BlockMixer::Attention(m) => &m.output.output,
        };
        [
            ("input_norm", &self.normalized),
            ("mixer", mixer),
            ("moe_input", &self.moe_input),
            ("moe", &self.moe_output),
        ]
    }

    pub fn prepare(
        device: &Device,
        rows: u32,
        weights: [Tensor<bf16, 1>; 2],
        mixer: BlockMixer,
        eps: f32,
        prefix: u32,
    ) -> Result<Self> {
        let hidden = weights[0].shape[0];
        ensure!(
            (1..=32).contains(&rows)
                && hidden > 0
                && weights[1].shape == [hidden]
                && eps.is_finite()
                && eps > 0.0,
            "pre-norm block geometry/epsilon mismatch"
        );
        for weight in weights {
            weight.validate()?;
        }
        let s = &device.stream;
        Ok(Self {
            weights,
            normalized: s.alloc_zeros((rows * hidden) as usize)?,
            moe_input: s.alloc_zeros((rows * hidden) as usize)?,
            moe_output: s.alloc_zeros((rows * hidden) as usize)?,
            metadata: s.clone_htod(&[0i32, rows as i32, rows as i32, hidden as i32, 1, 0, 0, 0])?,
            common: CommonOpsRef::new(&device.ctx)?,
            mixer,
            rows,
            hidden,
            eps,
            prefix,
        })
    }

    pub fn run(
        &mut self,
        device: &Device,
        blas: &Blas,
        ops: &Ops,
        x: &mut CudaSlice<bf16>,
        mode: (u32, bool),
        mut ffn: impl FnMut(&CudaSlice<bf16>, &mut CudaSlice<bf16>) -> Result<()>,
    ) -> Result<()> {
        let (rows, preview) = mode;
        ensure!(
            rows > 0 && rows <= self.rows,
            "block active rows exceed capacity"
        );
        let s = &device.stream;
        ensure!(
            x.len() >= (rows * self.hidden) as usize,
            "pre-norm block input extent mismatch"
        );
        let domain = RowDomain {
            meta: SeqRef {
                ptr: self.metadata.device_ptr(s).0,
            },
            row_base: 0,
            row_capacity: rows,
        };
        let norm = |buffer| heads(buffer, s, rows, 1, self.hidden);
        self.common.norm.rms_with(
            s,
            domain,
            norm(x)?,
            self.weights[0],
            &mut TensorMut(norm(&self.normalized)?),
            RmsOptions {
                mode: RmsMode::ZeroCentered,
                eps: self.eps,
            },
        )?;
        let context = BlockContext {
            stream: s,
            blas,
            common: &self.common,
            ops,
            domain,
            eps: self.eps,
        };
        let input = dense(&self.normalized, s, rows, self.hidden)?;
        let output = match &mut self.mixer {
            BlockMixer::Gdn(mixer) => {
                mixer.run(&context, input, preview)?;
                &mixer.output.output
            }
            BlockMixer::Attention(mixer) => {
                mixer.run(&context, input, self.prefix, preview)?;
                &mixer.output.output
            }
        };
        self.common.norm.rms_add_with(
            s,
            domain,
            ResidualRmsInput {
                x: &mut TensorMut(norm(x)?),
                residual: norm(output)?,
                weight: self.weights[1],
            },
            &mut TensorMut(norm(&self.moe_input)?),
            RmsOptions {
                mode: RmsMode::ZeroCentered,
                eps: self.eps,
            },
        )?;
        ffn(&self.moe_input, &mut self.moe_output)?;
        unsafe {
            s.launch_builder(&ops.residual)
                .arg(&mut x.slice_mut(..))
                .arg(&self.moe_output)
                .arg(&((rows * self.hidden) as i32))
                .launch(flat((rows * self.hidden) as usize))?;
        }
        if !preview {
            self.prefix = self
                .prefix
                .checked_add(rows)
                .ok_or_else(|| anyhow::anyhow!("block prefix overflow"))?;
        }
        Ok(())
    }
}

struct BlockMoeWeights {
    router: Tensor<bf16, 2>,
    shared: [Tensor<bf16, 2>; 3],
    shared_gate: Tensor<bf16, 1>,
}

struct BlockMoe {
    ops: PrefillOps,
    router: RouteOps,
    plans: [BlockRoutePlan; 2],
    tail: Option<BlockRoutePlan>,
    logits: CudaSlice<bf16>,
    ids: CudaSlice<i32>,
    weights: CudaSlice<f32>,
    active: CudaSlice<u32>,
    done: CudaSlice<u32>,
    contributions: CudaSlice<bf16>,
    gu: CudaSlice<u64>,
    dn: CudaSlice<u64>,
    live: CudaSlice<u32>,
    generation: CudaSlice<u64>,
    gu_host: Vec<u64>,
    dn_host: Vec<u64>,
    live_host: Vec<u32>,
    epoch: u64,
    scratch: CudaSlice<u8>,
    _route_scratch: CudaSlice<u8>,
    _valid: CudaSlice<i32>,
    _allowed: CudaSlice<u32>,
    counts: [u64; 5],
}

impl BlockMoe {
    fn prepare(device: &Device) -> Result<Self> {
        let s = &device.stream;
        let logits = s.alloc_zeros(32 * 256)?;
        let ids = s.alloc_zeros(32 * 8)?;
        let weights = s.alloc_zeros(32 * 8)?;
        let allowed = s.alloc_zeros(256)?;
        let valid = s.clone_htod(&[1i32; 32])?;
        let route_scratch = s.alloc_zeros::<u8>(32 * 256 * 4)?;
        let bind = |top_p| {
            BlockRoutePlan::prepare(
                &device.ctx,
                BlockRouteBindings {
                    logits: dense(&logits, s, 32, 256)?,
                    ids: TensorMut(dense(&ids, s, 32, 8)?),
                    weights: TensorMut(dense(&weights, s, 32, 8)?),
                    valid: view(&valid, s, 0, [32], [1])?,
                    allowed: TensorMut(view(&allowed, s, 0, [256], [1])?),
                },
                Scratch {
                    ptr: route_scratch.device_ptr(s).0,
                    bytes: route_scratch.len() as u64,
                },
                top_p,
            )
        };
        let plans = [bind(Some(0.9))?, bind(None)?];
        let bytes = super::moe::prefill::prefill_scratch_bytes_mixed_with_boundary(
            32,
            32,
            8,
            256,
            2048,
            (512, 512),
            (None, RoutedWeightBoundary::BeforeContributionBf16),
        )?;
        Ok(Self {
            ops: PrefillOps::new_with_boundaries(
                &device.ctx,
                s,
                ExpertFormat::Bf16,
                SwigluRounding::Once,
                RoutedWeightBoundary::BeforeContributionBf16,
            )?,
            router: RouteOps::new(&device.ctx, s, 256, 32)?,
            plans,
            tail: None,
            logits,
            ids,
            weights,
            active: s.clone_htod(&[1u32; 256])?,
            done: s.alloc_zeros(256)?,
            contributions: s.alloc_zeros(256 * 2048)?,
            gu: s.alloc_zeros(256)?,
            dn: s.alloc_zeros(256)?,
            live: s.alloc_zeros(256)?,
            generation: s.alloc_zeros(256)?,
            gu_host: vec![0; 256],
            dn_host: vec![0; 256],
            live_host: vec![0; 256],
            epoch: 0,
            scratch: s.alloc_zeros(bytes as usize)?,
            _route_scratch: route_scratch,
            _valid: valid,
            _allowed: allowed,
            counts: [0; 5],
        })
    }

    fn contribute(
        &mut self,
        s: &Arc<CudaStream>,
        w: &BlockMoeWeights,
        x: Tensor<bf16, 2>,
    ) -> Result<()> {
        let rows = x.shape[0];
        self.epoch = self
            .epoch
            .checked_add(1)
            .context("cohort generation overflow")?;
        s.memcpy_htod(&self.gu_host, &mut self.gu)?;
        s.memcpy_htod(&self.dn_host, &mut self.dn)?;
        s.memcpy_htod(&self.live_host, &mut self.live)?;
        s.memcpy_htod(&[self.epoch; 256], &mut self.generation)?;
        self.ops.prefill_contribute(
            s,
            MoePrefillContribute {
                mask_from_slots: true,
                expert_inter: 512,
                wave: None,
                wait_copy: None,
                reader_event: None,
                cpu_arithmetic: &[true; 256],
                fp8_slot: None,
                x,
                ids: dense(&self.ids, s, rows, 8)?,
                weights: dense(&self.weights, s, rows, 8)?,
                active: dense(&self.active, s, rows, 8)?,
                assignment_gen: None,
                slots_gate_up: view(&self.gu, s, 0, [256], [1])?,
                slots_down: view(&self.dn, s, 0, [256], [1])?,
                slot_valid: view(&self.live, s, 0, [256], [1])?,
                slot_generation: view(&self.generation, s, 0, [256], [1])?,
                slots_gate_up_host: &self.gu_host,
                slots_down_host: &self.dn_host,
                shared_gate: w.shared[0],
                shared_up: w.shared[1],
                shared_down: w.shared[2],
                shared_expert_gate: Some(w.shared_gate),
                contributions: heads(&self.contributions, s, rows, 8, 2048)?,
                done: TensorMut(dense(&self.done, s, rows, 8)?),
            },
            &Scratch {
                ptr: self.scratch.device_ptr(s).0,
                bytes: self.scratch.len() as u64,
            },
        )?;
        self.counts[4] += 1;
        Ok(())
    }

    fn run(
        &mut self,
        device: &Device,
        w: &BlockMoeWeights,
        store: &mut CohortStorage,
        layer: u32,
        buffers: (&CudaSlice<bf16>, &mut CudaSlice<bf16>),
        mode: (u32, bool),
    ) -> Result<()> {
        use crate::experts::plan::Tier;
        let s = &device.stream;
        let (input, output) = buffers;
        let (rows, preview) = mode;
        let x = dense(input, s, rows, 2048)?;
        self.router.logits_bf16(
            x,
            w.router,
            &mut TensorMut(dense(&self.logits, s, rows, 256)?),
        )?;
        let plan = if rows == 32 {
            &self.plans[usize::from(!preview)]
        } else {
            ensure!(!preview, "MoE preview requires the full block");
            self.tail
                .as_ref()
                .context("MoE causal tail is not prepared")?
        };
        plan.run(s)?;
        let mut selected = [0i32; 256];
        let assignments = rows as usize * 8;
        s.memcpy_dtoh(&self.ids.slice(..assignments), &mut selected[..assignments])?;
        let selected = &mut selected[..assignments];
        ensure!(
            selected.iter().all(|&id| (0..256).contains(&id)),
            "cohort route id out of range"
        );
        selected.sort_unstable();
        self.counts = [0; 5];
        s.memset_zeros(&mut self.done)?;
        self.live_host.fill(0);
        let mut misses = [0u32; 256];
        let mut count = 0;
        let mut previous = -1;
        for &expert in selected.iter() {
            if expert == previous {
                continue;
            }
            previous = expert;
            self.counts[0] += 1;
            if store.tier(layer, expert as u32)? == Tier::Gpu {
                let (gu, dn) = store.resident(layer, expert as u32)?;
                self.gu_host[expert as usize] = gu;
                self.dn_host[expert as usize] = dn;
                self.live_host[expert as usize] = 1;
                self.counts[1] += 1;
            } else {
                misses[count] = expert as u32;
                count += 1;
            }
        }
        if self.counts[1] > 0 {
            self.contribute(s, w, x)?;
        }
        if count > 0 {
            ensure!(
                store.wave_capacity() > 0,
                "cohort misses without staging slots"
            );
            for wave in misses[..count].chunks(store.wave_capacity() as usize) {
                let bytes = store.stage(s, layer, wave)?;
                self.counts[2] += bytes[0];
                self.counts[3] += bytes[1];
                self.live_host.fill(0);
                for (slot, &expert) in wave.iter().enumerate() {
                    let (gu, dn) = store.staged(slot as u32)?;
                    self.gu_host[expert as usize] = gu;
                    self.dn_host[expert as usize] = dn;
                    self.live_host[expert as usize] = 1;
                }
                self.contribute(s, w, x)?;
            }
        }
        self.ops.prefill_finish(
            s,
            MoePrefillFinish {
                expert_inter: 512,
                x,
                ids: dense(&self.ids, s, rows, 8)?,
                final_active: dense(&self.active, s, rows, 8)?,
                contributions: heads(&self.contributions, s, rows, 8, 2048)?,
                done: dense(&self.done, s, rows, 8)?,
                experts: 256,
                shared_gate: w.shared[0],
                shared_up: w.shared[1],
                shared_down: w.shared[2],
                shared_expert_gate: Some(w.shared_gate),
                y: TensorMut(dense(output, s, rows, 2048)?),
            },
            &Scratch {
                ptr: self.scratch.device_ptr(s).0,
                bytes: self.scratch.len() as u64,
            },
        )
    }
}

pub struct BlockForward {
    // Expert storage synchronizes the context before the remaining fields drop.
    experts: CohortStorage,
    weights: WeightCatalog,
    layers: Vec<PreNormBlock>,
    moe_weights: Vec<BlockMoeWeights>,
    moe: BlockMoe,
    blas: Blas,
    ops: Ops,
    ids: CudaSlice<u32>,
    hidden: CudaSlice<bf16>,
    normalized: CudaSlice<bf16>,
    _table: CudaSlice<bf16>,
    pub logits: CudaSlice<bf16>,
    capacity: u32,
    vocab: u32,
    tail_rows: u32,
    #[cfg(test)]
    pub(crate) counts: [u64; 5],
}

impl BlockForward {
    #[cfg(test)]
    pub(crate) fn committed_image(
        &self,
        device: &Device,
    ) -> Result<Vec<std::collections::BTreeMap<String, Vec<f32>>>> {
        self.layers
            .iter()
            .map(|block| cache_values(block, &device.stream, block.prefix as usize))
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn preview_commit_control(&mut self, device: &Device) -> Result<bool> {
        let before = self.committed_image(device)?;
        let BlockMixer::Gdn(mixer) = &mut self.layers[0].mixer else {
            anyhow::bail!("Runner FAIL preview control requires GDN layer zero");
        };
        let state = device.stream.clone_dtoh(&mixer.state)?;
        let history = device.stream.clone_dtoh(&mixer.history)?;
        mixer.plans[1].run(&device.stream)?;
        let changed = device.stream.clone_dtoh(&mixer.state)? != state
            || device.stream.clone_dtoh(&mixer.history)? != history;
        device.stream.memcpy_htod(&state, &mut mixer.state)?;
        device.stream.memcpy_htod(&history, &mut mixer.history)?;
        let restored = self.committed_image(device)?;
        ensure!(
            restored
                .iter()
                .zip(&before)
                .all(|(a, b)| a.keys().eq(b.keys())
                    && a.values()
                        .flatten()
                        .map(|v| v.to_bits())
                        .eq(b.values().flatten().map(|v| v.to_bits()))),
            "Runner FAIL preview control restoration"
        );
        Ok(changed)
    }

    pub fn load(
        device: &Device,
        directory: &std::path::Path,
        runtime: &std::path::Path,
        capacity: u32,
        device_budget_bytes: Option<u64>,
    ) -> Result<Self> {
        use crate::checkpoint::{LoadPolicy, MappedCheckpoint};
        use crate::experts::catalogue::{ExpertCatalogue, ExpertGeometry};
        let spec = crate::model_spec::ModelSpec::read(directory)?;
        let manifest = MappedCheckpoint::open(directory)?.manifest(&spec, LoadPolicy::TextOnly)?;
        let plan =
            PreNormBlock::model_residency(device, &spec, &manifest, capacity, device_budget_bytes)?;
        ensure!(
            plan.dense.host_bytes == 0 && plan.dense.disk_bytes == 0,
            "block forward requires the derived dense-resident placement"
        );
        let catalogue = ExpertCatalogue::from_manifest(
            &manifest,
            ExpertGeometry::from_model(&spec)?,
            spec.expert_format.context("block expert format missing")?,
        )?;
        let ops = Ops::new_with_swiglu_rounding(&device.ctx, SwigluRounding::Once)?;
        let closure = std::rc::Rc::new(super::closure::Closure::new(false, "block_forward".into()));
        let blas = Blas::new(device, &ops, runtime, closure)?;
        let weights = WeightCatalog::load(directory, &spec, device)?;
        device.upload.synchronize()?;
        let s = &device.stream;
        let rope = crate::model_spec::RopeSpec {
            theta: spec.rope_theta,
            pairs: spec.rotary_dim / 2,
            pair_stride: spec.rotary_dim / 2,
            denominator: spec.rotary_dim,
        };
        let table = s.clone_htod(&rope.table(64, capacity as usize)?)?;
        let layers = (0..spec.layers.len())
            .map(|layer| Self::layer(device, &blas, &weights, &spec, layer, &table, capacity))
            .collect::<Result<Vec<_>>>()?;
        let moe_weights = (0..spec.layers.len())
            .map(|layer| {
                let prefix = format!("model.layers.{layer}.mlp.");
                let matrix = |name: &str| weights.matrix(s, &format!("{prefix}{name}"));
                Ok(BlockMoeWeights {
                    router: matrix("gate.weight")?,
                    shared: [
                        matrix("shared_expert.gate_proj.weight")?,
                        matrix("shared_expert.up_proj.weight")?,
                        matrix("shared_expert.down_proj.weight")?,
                    ],
                    shared_gate: weights
                        .vector(s, &format!("{prefix}shared_expert_gate.weight"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let moe = BlockMoe::prepare(device)?;
        blas.prepare_request_fixed(32, spec.vocab, spec.hidden)?;
        let hidden = s.alloc_zeros(32 * spec.hidden as usize)?;
        let normalized = s.alloc_zeros(32 * spec.hidden as usize)?;
        let logits = s.alloc_zeros(32 * spec.vocab as usize)?;
        let ids = s.alloc_zeros(32)?;
        let experts =
            CohortStorage::load(device, directory, catalogue, &plan.experts, plan.admission)?;
        Ok(Self {
            weights,
            layers,
            moe_weights,
            moe,
            experts,
            blas,
            ops,
            ids,
            hidden,
            normalized,
            _table: table,
            logits,
            capacity,
            vocab: spec.vocab,
            tail_rows: 0,
            #[cfg(test)]
            counts: [0; 5],
        })
    }

    fn layer(
        device: &Device,
        blas: &Blas,
        w: &WeightCatalog,
        spec: &crate::model_spec::ModelSpec,
        layer: usize,
        table: &CudaSlice<bf16>,
        capacity: u32,
    ) -> Result<PreNormBlock> {
        let s = &device.stream;
        let prefix = format!("model.layers.{layer}.");
        let matrix = |name: &str| w.matrix(s, &format!("{prefix}{name}"));
        let vector = |name: &str| w.vector(s, &format!("{prefix}{name}"));
        let mixer = if spec.layers[layer].mixer == crate::model_spec::Mixer::Gdn {
            let gdn = GdnSpec::from_model(spec)?;
            BlockMixer::Gdn(Box::new(GdnBlockMixer::prepare(
                device,
                blas,
                32,
                GdnBlockWeights {
                    spec: gdn,
                    qkv: matrix("linear_attn.in_proj_qkv.weight")?,
                    a: matrix("linear_attn.in_proj_a.weight")?,
                    b: matrix("linear_attn.in_proj_b.weight")?,
                    z: matrix("linear_attn.in_proj_z.weight")?,
                    output: matrix("linear_attn.out_proj.weight")?,
                    conv: matrix("linear_attn.conv1d.weight")?,
                    right: matrix("linear_attn.conv_right_weight")?,
                    decay: [vector("linear_attn.A_log")?, vector("linear_attn.dt_bias")?],
                    softmax: [
                        vector("linear_attn.bd_softmax_logit_scale")?,
                        vector("linear_attn.bd_softmax_history_bias")?,
                        vector("linear_attn.bd_softmax_history_c")?,
                    ],
                    norm: vector("linear_attn.norm.weight")?,
                },
                s.alloc_zeros(32 * 128 * 128)?,
                s.alloc_zeros(8192 * 3)?,
            )?))
        } else {
            BlockMixer::Attention(Box::new(AttentionBlockMixer::prepare(
                device,
                blas,
                32,
                AttentionBlockWeights {
                    q_gate: matrix("self_attn.q_proj.weight")?,
                    k: matrix("self_attn.k_proj.weight")?,
                    v: matrix("self_attn.v_proj.weight")?,
                    output: matrix("self_attn.o_proj.weight")?,
                    q_norm: vector("self_attn.q_norm.weight")?,
                    k_norm: vector("self_attn.k_norm.weight")?,
                    table: dense(table, s, capacity, 64)?,
                    heads: 16,
                    kv_heads: 2,
                    dim: 256,
                },
                s.alloc_zeros(2 * capacity as usize * 256)?,
                s.alloc_zeros(2 * capacity as usize * 256)?,
            )?))
        };
        PreNormBlock::prepare(
            device,
            32,
            [
                vector("input_layernorm.weight")?,
                vector("post_attention_layernorm.weight")?,
            ],
            mixer,
            spec.rms_eps,
            0,
        )
    }

    pub(crate) fn prepare_causal_rows(&mut self, device: &Device, rows: u32) -> Result<()> {
        ensure!((1..=32).contains(&rows), "causal setup rows must be 1..32");
        self.tail_rows = rows;
        if rows == 32 {
            return Ok(());
        }
        for block in &mut self.layers {
            let projections = match &mut block.mixer {
                BlockMixer::Gdn(m) => {
                    m.tail = Some(m.plans[1].causal_rows(&device.ctx, rows)?);
                    vec![&m.qkv, &m.a, &m.b, &m.z, &m.output]
                }
                BlockMixer::Attention(m) => vec![&m.q_gate, &m.key, &m.value, &m.output],
            };
            for projection in projections {
                self.blas.prepare_request_fixed(
                    rows,
                    projection.weight.shape[0],
                    projection.weight.shape[1],
                )?;
            }
        }
        self.moe.tail = Some(self.moe.plans[1].causal_rows(&device.ctx, rows)?);
        Ok(())
    }

    pub(crate) fn reset(&mut self, device: &Device) -> Result<()> {
        device.stream.synchronize()?;
        for block in &mut self.layers {
            block.prefix = 0;
            if let BlockMixer::Gdn(m) = &mut block.mixer {
                device.stream.memset_zeros(&mut m.state)?;
                device.stream.memset_zeros(&mut m.history)?;
            }
        }
        self.tail_rows = 0;
        Ok(())
    }

    fn embed_ids(&mut self, device: &Device, ids: u64, rows: u32) -> Result<()> {
        let s = &device.stream;
        unsafe {
            s.launch_builder(&self.ops.embed)
                .arg(self.weights.bf16("model.embed_tokens.weight")?)
                .arg(&ids)
                .arg(&mut self.hidden)
                .arg(&(rows as i32))
                .arg(&2048i32)
                .launch(flat(rows as usize * 2048))?;
        }
        Ok(())
    }

    pub fn run(&mut self, device: &Device, tokens: &[u32], preview: bool) -> Result<()> {
        self.forward_host(device, tokens, preview, |_, _, _, _| Ok(()))?;
        if preview {
            self.project_logits(device, 32)?;
        }
        Ok(())
    }

    pub(crate) fn preview(&mut self, device: &Device, tokens: &CudaSlice<i32>) -> Result<()> {
        ensure!(tokens.len() == 32, "preview requires 32 device token IDs");
        self.embed_ids(device, tokens.device_ptr(&device.stream).0, 32)?;
        self.execute(device, 32, true, |_, _, _, _| Ok(()))?;
        self.project_logits(device, 32)
    }

    pub(crate) fn causal_device(&mut self, device: &Device, tokens: &CudaSlice<i32>) -> Result<()> {
        ensure!(tokens.len() == 32, "causal block token extent mismatch");
        let (pointer, _) = tokens.device_ptr(&device.stream);
        self.embed_ids(device, pointer, 32)?;
        self.execute(device, 32, false, |_, _, _, _| Ok(()))
    }

    #[cfg(test)]
    fn forward(
        &mut self,
        device: &Device,
        tokens: &[u32],
        preview: bool,
        observe: impl FnMut(usize, &PreNormBlock, &BlockMoe, &CudaSlice<bf16>) -> Result<()>,
    ) -> Result<()> {
        self.forward_host(device, tokens, preview, observe)?;
        self.project_logits(device, tokens.len() as u32)
    }

    fn forward_host(
        &mut self,
        device: &Device,
        tokens: &[u32],
        preview: bool,
        observe: impl FnMut(usize, &PreNormBlock, &BlockMoe, &CudaSlice<bf16>) -> Result<()>,
    ) -> Result<()> {
        ensure!(
            (1..=32).contains(&tokens.len()) && tokens.iter().all(|&id| id < self.vocab),
            "block token extent/id mismatch"
        );
        let rows = tokens.len() as u32;
        device
            .stream
            .memcpy_htod(tokens, &mut self.ids.slice_mut(..tokens.len()))?;
        let pointer = self.ids.device_ptr(&device.stream).0;
        self.embed_ids(device, pointer, rows)?;
        self.execute(device, rows, preview, observe)
    }

    fn execute(
        &mut self,
        device: &Device,
        rows: u32,
        preview: bool,
        mut observe: impl FnMut(usize, &PreNormBlock, &BlockMoe, &CudaSlice<bf16>) -> Result<()>,
    ) -> Result<()> {
        ensure!(
            (1..=32).contains(&rows)
                && (!preview || rows == 32)
                && (rows == 32 || rows == self.tail_rows),
            "block active rows are not prepared"
        );
        let prefix = self.layers[0].prefix;
        #[cfg(test)]
        {
            self.counts = [0; 5];
        }
        ensure!(
            prefix
                .checked_add(rows)
                .is_some_and(|end| end <= self.capacity)
                && self.layers.iter().all(|layer| layer.prefix == prefix),
            "block prefix/capacity mismatch"
        );
        for (index, block) in self.layers.iter_mut().enumerate() {
            block.run(
                device,
                &self.blas,
                &self.ops,
                &mut self.hidden,
                (rows, preview),
                |x, y| {
                    self.moe.run(
                        device,
                        &self.moe_weights[index],
                        &mut self.experts,
                        index as u32,
                        (x, y),
                        (rows, preview),
                    )
                },
            )?;
            #[cfg(test)]
            for (total, count) in self.counts.iter_mut().zip(self.moe.counts) {
                *total += count;
            }
            observe(index, block, &self.moe, &self.hidden)?;
        }
        Ok(())
    }

    fn project_logits(&mut self, device: &Device, rows: u32) -> Result<()> {
        let s = &device.stream;
        let block = &self.layers[0];
        block.common.norm.rms_with(
            s,
            RowDomain {
                meta: SeqRef {
                    ptr: block.metadata.device_ptr(s).0,
                },
                row_base: 0,
                row_capacity: rows,
            },
            heads(&self.hidden, s, rows, 1, 2048)?,
            self.weights.vector(s, "model.norm.weight")?,
            &mut TensorMut(heads(&self.normalized, s, rows, 1, 2048)?),
            RmsOptions {
                mode: RmsMode::ZeroCentered,
                eps: block.eps,
            },
        )?;
        self.blas.linear_typed(
            ExecutionPhase::Prefill,
            LinearWeightRef::Bf16(self.weights.matrix(s, "lm_head.weight")?),
            dense(&self.normalized, s, rows, 2048)?,
            &mut TensorMut(dense(&self.logits, s, rows, self.vocab)?),
        )
    }
}

#[cfg(test)]
fn transpose_state(x: &[f32]) -> Vec<f32> {
    (0..x.len())
        .map(|i| x[i / 16384 * 16384 + i % 128 * 128 + i % 16384 / 128])
        .collect()
}

#[cfg(test)]
fn cache_values(
    block: &PreNormBlock,
    s: &Arc<CudaStream>,
    extent: usize,
) -> Result<std::collections::BTreeMap<String, Vec<f32>>> {
    Ok(match &block.mixer {
        BlockMixer::Gdn(m) => std::collections::BTreeMap::from([
            (
                "state_ssm".into(),
                transpose_state(&s.clone_dtoh(&m.state)?),
            ),
            ("state_conv".into(), s.clone_dtoh(&m.history)?),
        ]),
        BlockMixer::Attention(m) => {
            let mut result = std::collections::BTreeMap::new();
            for (name, buffer) in [("state_key", &m.k), ("state_value", &m.v)] {
                let full = s.clone_dtoh(buffer)?;
                let capacity = full.len() / (2 * 256);
                let mut values = Vec::new();
                for h in 0..2 {
                    values.extend(
                        full[h * capacity * 256..h * capacity * 256 + extent * 256]
                            .iter()
                            .map(|v| v.to_f32()),
                    );
                }
                result.insert(name.into(), values);
            }
            result
        }
    })
}
