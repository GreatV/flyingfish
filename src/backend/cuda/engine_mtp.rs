//! Single-Engine MTP component shared by generation and numerical probes.
use super::{
    Work,
    qwen::{heads, limits, matrix, scratch, vector, view},
};
#[cfg(test)]
use crate::backend::cuda::closure::Closure;
use crate::{
    backend::cuda::{
        Device,
        attention_d256::{
            AttentionBindings, FullAttentionOps, FullAttentionPlan, PositionBindings,
        },
        blas::Blas,
        contracts::{RowDomain, TensorMut},
        cubin,
        flash::{Flash, Shape},
        linear::LayerLinearPlan,
        ops::{CommonOpsRef, Ops, RmsMode, RmsOptions, flat},
        state::StateStore,
        weights::WeightCatalog,
    },
    checkpoint::LoadPolicy,
    linear_plan::LayerDequantPlan,
    model_spec::{LayerPlan, Mixer, ModelSpec},
    mtp::{HIDDEN, MtpHiddenTap, MtpPlan, MtpRecurrentTap, VOCAB},
    op_contract::ExecutionPhase,
};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, PushKernelArg};
use half::bf16;
#[cfg(test)]
use std::path::Path;
#[cfg(test)]
use std::rc::Rc;
use std::{collections::BTreeMap, sync::Arc};

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
enum Control {
    #[default]
    None,
    SwapConcat,
    ShiftNewPosition,
    StandardNorm,
}

struct MtpRuntime {
    plan: MtpPlan,
    work: Work,
    embedding: CudaSlice<bf16>,
    hidden: CudaSlice<bf16>,
    concat: CudaSlice<bf16>,
    logits: CudaSlice<bf16>,
    positions: CudaSlice<i32>,
    arena: CudaSlice<u8>,
    scratch: CudaSlice<u8>,
    mask: CudaSlice<u64>,
    linear: LayerLinearPlan,
    attention: FullAttentionPlan,
    common: CommonOpsRef,
    multiply: CudaFunction,
    flash: Flash,
}

struct Execution<'a> {
    device: &'a Device,
    ops: &'a Ops,
    blas: &'a Blas,
    catalog: &'a WeightCatalog,
    embedding: &'a CudaSlice<bf16>,
    head: &'a CudaSlice<bf16>,
}

impl MtpRuntime {
    fn allocation_bytes(&self) -> usize {
        let w = &self.work;
        let bf16_elements = [
            &w.x,
            &w.n,
            &w.qkv,
            &w.attn,
            &w.out,
            &w.gu,
            &w.act,
            &w.logits,
            &self.embedding,
            &self.hidden,
            &self.concat,
            &self.logits,
        ]
        .into_iter()
        .map(CudaSlice::len)
        .sum::<usize>()
            + w.mixer
                .as_ref()
                .map_or(0, |m| m.q.len() + m.gate.len() + m.ab.len());
        bf16_elements * 2
            + (w.ids.len()
                + w.position.len()
                + w.length.len()
                + w.token.len()
                + self.positions.len())
                * 4
            + self.mask.len() * 8
            + self.arena.len()
            + self.scratch.len()
            + self.plan.max_rows * 24 * size_of::<f32>() // Flash's setup-bound LSE.
    }
    fn new(
        device: &Device,
        spec: &ModelSpec,
        catalog: &WeightCatalog,
        state: &StateStore,
        blas: &Blas,
        plan: MtpPlan,
    ) -> Result<Self> {
        plan.validate(spec)?;
        let s = &device.stream;
        let rows = plan.max_rows;
        let work = Work::new(s, spec, rows)?;
        let embedding = s.alloc_zeros(rows * HIDDEN)?;
        let hidden = s.alloc_zeros(rows * HIDDEN)?;
        let concat = s.alloc_zeros(rows * HIDDEN * 2)?;
        let logits = s.alloc_zeros(rows * VOCAB)?;
        let positions = s.alloc_zeros(rows * 3)?;
        let dequant = LayerDequantPlan::mtp(&catalog.manifest)?;
        ensure!(
            dequant.bytes as usize == crate::mtp::ARENA_BYTES && dequant.slots.len() == 7,
            "unexpected MTP materialization layout"
        );
        let arena = s.alloc_zeros(dequant.bytes as usize)?;
        let scratch_buffer = s.alloc_zeros(plan.memory(spec)?.scratch)?;
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
        let mut linear = LayerLinearPlan::bind(&dequant, catalog, &scratch(&arena, s), s)?;
        linear.prepare(blas, rows as u32)?;
        blas.prepare_fixed(rows as u32, HIDDEN as u32, (2 * HIDDEN) as u32)?;
        blas.prepare_fixed(rows as u32, VOCAB as u32, HIDDEN as u32)?;
        linear.materialize(blas.fp8_ops()?, s)?;
        let common = CommonOpsRef::new(&device.ctx)?;
        let mixer = work.mixer.as_ref().context("MTP Q/gate buffers missing")?;
        let kv = &state.kv[0];
        let attention = FullAttentionOps::prepare(
            &device.ctx,
            &crate::op_contract::FullAttentionSpec::q38(),
            limits(rows, plan.capacity)?,
            AttentionBindings {
                q_gate: heads(&work.qkv, s, 0, rows, 24, 512)?,
                k: heads(&work.qkv, s, rows * 12288, rows, 4, 256)?,
                v: heads(&work.qkv, s, rows * (12288 + 1024), rows, 4, 256)?,
                q: TensorMut(heads(&mixer.q, s, 0, rows, 24, 256)?),
                gate: TensorMut(heads(&mixer.gate, s, 0, rows, 24, 256)?),
                k_cache: TensorMut(view(
                    &kv.k,
                    s,
                    0,
                    [4, plan.capacity as u32, 256],
                    [(plan.capacity * 256) as u64, 256, 1],
                )?),
                v_cache: TensorMut(view(
                    &kv.v,
                    s,
                    0,
                    [4, plan.capacity as u32, 256],
                    [(plan.capacity * 256) as u64, 256, 1],
                )?),
                core_out: TensorMut(heads(&work.attn, s, 0, rows, 24, 256)?),
                q_norm: vector(catalog.bf16("mtp.layers.0.self_attn.q_norm.weight")?, s)?,
                k_norm: vector(catalog.bf16("mtp.layers.0.self_attn.k_norm.weight")?, s)?,
                causal_mask: view(&mask, s, 0, [64], [1])?,
                positions: PositionBindings::Axes(view(
                    &positions,
                    s,
                    0,
                    [rows as u32, 3],
                    [3, 1],
                )?),
            },
            common.clone(),
        )?;
        let multiply = cubin::module(&device.ctx, "q38_common")?.load_function("q38_multiply")?;
        Ok(Self {
            plan,
            work,
            embedding,
            hidden,
            concat,
            logits,
            positions,
            arena,
            scratch: scratch_buffer,
            mask,
            linear,
            attention,
            common,
            multiply,
            flash: Flash::new(s.clone(), rows, 24)?,
        })
    }

    fn run(
        &mut self,
        execution: Execution<'_>,
        state: &mut StateStore,
        input: RunInput<'_>,
    ) -> Result<Dump> {
        let Execution {
            device,
            ops,
            blas,
            catalog,
            embedding,
            head,
        } = execution;
        let RunInput {
            prefix,
            ids,
            hidden,
            #[cfg(test)]
            control,
            detailed,
        } = input;
        #[cfg(test)]
        let (shift_position, standard_norm, swap_concat) = (
            matches!(control, Control::ShiftNewPosition),
            matches!(control, Control::StandardNorm),
            matches!(control, Control::SwapConcat),
        );
        #[cfg(not(test))]
        let (shift_position, standard_norm, swap_concat) = (false, false, false);
        let rows = ids.len();
        self.plan.check_rows(prefix, rows)?;
        ensure!(
            rows == self.plan.max_rows && hidden.len() == rows * HIDDEN,
            "MTP call does not match bound rows/hidden"
        );
        ensure!(
            ids.iter().all(|&t| (t as usize) < VOCAB) && hidden.iter().all(|h| h.is_finite()),
            "invalid MTP inputs"
        );
        let s = &device.stream;
        state.set_sequence(s, prefix as u32, rows as u32, self.plan.max_rows as u32)?;
        s.memcpy_htod(ids, &mut self.work.ids)?;
        s.memcpy_htod(hidden, &mut self.hidden)?;
        let axes: Vec<i32> = (0..rows)
            .flat_map(|r| {
                let pos = (prefix + r + 1) as i32 + i32::from(shift_position);
                [pos; 3]
            })
            .collect();
        s.memcpy_htod(&axes, &mut self.positions)?;
        let domain = RowDomain {
            meta: state.seq_ref(s)?,
            row_base: 0,
            row_capacity: rows as u32,
        };
        let options = RmsOptions {
            mode: if standard_norm {
                RmsMode::Standard
            } else {
                RmsMode::ZeroCentered
            },
            eps: 1e-6,
        };
        let mut dump = Dump::default();
        dump.i32("positions", vec![rows, 3], &axes)?;
        dump.u32("input_ids", vec![rows], ids)?;
        if detailed {
            dump.bf16("hidden_input", vec![rows, HIDDEN], hidden.to_vec())?;
        }
        unsafe {
            s.launch_builder(&ops.embed)
                .arg(embedding)
                .arg(&self.work.ids)
                .arg(&mut self.embedding)
                .arg(&(rows as i32))
                .arg(&(HIDDEN as i32))
                .launch(flat(rows * HIDDEN))?;
        }
        if detailed {
            dump.device("embedding", s, &self.embedding, 0, rows, HIDDEN)?;
        }
        for (is_hidden, name, buffer) in [
            (false, "pre_fc_norm_embedding", &self.embedding),
            (true, "pre_fc_norm_hidden", &self.hidden),
        ] {
            let half = usize::from(is_hidden ^ swap_concat);
            let output = view(
                &self.concat,
                s,
                half * HIDDEN,
                [rows as u32, 1, HIDDEN as u32],
                [(2 * HIDDEN) as u64, HIDDEN as u64, 1],
            )?;
            self.common.norm.rms_with(
                s,
                domain,
                heads(buffer, s, 0, rows, 1, HIDDEN as u32)?,
                vector(catalog.bf16(&format!("mtp.{name}.weight"))?, s)?,
                &mut TensorMut(output),
                options,
            )?;
        }
        if detailed {
            let cat = s.clone_dtoh(&self.concat)?;
            for (name, half) in [("pre_fc_norm_embedding", 0), ("pre_fc_norm_hidden", 1)] {
                let half = half ^ usize::from(swap_concat);
                dump.bf16(
                    name,
                    vec![rows, HIDDEN],
                    cat.as_chunks::<{ 2 * HIDDEN }>()
                        .0
                        .iter()
                        .flat_map(|row| row[half * HIDDEN..(half + 1) * HIDDEN].iter().copied())
                        .collect(),
                )?;
            }
            dump.bf16("concat", vec![rows, 2 * HIDDEN], cat)?;
        }
        blas.linear_typed(
            self.plan.phase,
            catalog.linear_ref("mtp.fc.weight", s)?,
            matrix(&self.concat, s, 0, rows, 2 * HIDDEN)?,
            &mut TensorMut(matrix(&self.work.x, s, 0, rows, HIDDEN)?),
        )?;
        if detailed {
            dump.device("fc", s, &self.work.x, 0, rows, HIDDEN)?;
        }
        let norm = |x: &CudaSlice<bf16>, y: &CudaSlice<bf16>, name: &str| {
            self.common.norm.rms_with(
                s,
                domain,
                heads(x, s, 0, rows, 1, HIDDEN as u32)?,
                vector(catalog.bf16(name)?, s)?,
                &mut TensorMut(heads(y, s, 0, rows, 1, HIDDEN as u32)?),
                RmsOptions {
                    mode: RmsMode::ZeroCentered,
                    eps: 1e-6,
                },
            )
        };
        norm(
            &self.work.x,
            &self.work.n,
            "mtp.layers.0.input_layernorm.weight",
        )?;
        if detailed {
            dump.device("norm1", s, &self.work.n, 0, rows, HIDDEN)?;
        }
        for (role, width, offset) in [
            ("self_attn.q_proj.weight", 12288, 0),
            ("self_attn.k_proj.weight", 1024, rows * 12288),
            ("self_attn.v_proj.weight", 1024, rows * (12288 + 1024)),
        ] {
            self.linear.run(
                blas,
                role,
                self.plan.phase,
                matrix(&self.work.n, s, 0, rows, HIDDEN)?,
                &mut TensorMut(matrix(&self.work.qkv, s, offset, rows, width)?),
            )?;
            if detailed {
                dump.device(role, s, &self.work.qkv, offset, rows, width)?;
            }
        }
        let mixer = self.work.mixer.as_ref().context("MTP mixer work missing")?;
        if detailed {
            self.common.norm.rms_with(
                s,
                domain,
                view(
                    &self.work.qkv,
                    s,
                    0,
                    [rows as u32, 24, 256],
                    [12288, 512, 1],
                )?,
                vector(catalog.bf16("mtp.layers.0.self_attn.q_norm.weight")?, s)?,
                &mut TensorMut(heads(&mixer.q, s, 0, rows, 24, 256)?),
                RmsOptions {
                    mode: RmsMode::ZeroCentered,
                    eps: 1e-6,
                },
            )?;
            dump.device("q_norm", s, &mixer.q, 0, rows, 6144)?;
            self.common.norm.rms_with(
                s,
                domain,
                heads(&self.work.qkv, s, rows * 12288, rows, 4, 256)?,
                vector(catalog.bf16("mtp.layers.0.self_attn.k_norm.weight")?, s)?,
                &mut TensorMut(heads(&self.work.n, s, 0, rows, 4, 256)?),
                RmsOptions {
                    mode: RmsMode::ZeroCentered,
                    eps: 1e-6,
                },
            )?;
            dump.device("k_norm", s, &self.work.n, 0, rows, 1024)?;
        }
        match self.plan.phase {
            ExecutionPhase::Prefill => {
                FullAttentionOps::preprocess(
                    &self.attention,
                    s,
                    domain.meta,
                    scratch(&self.scratch, s),
                )?;
                self.flash.run(
                    &mixer.q,
                    6144,
                    &state.kv[0].k,
                    &state.kv[0].v,
                    &mut self.work.attn,
                    Shape {
                        rows,
                        start: prefix,
                        q_heads: 24,
                        kv_heads: 4,
                        capacity: self.plan.capacity,
                        head_dim: 256,
                        causal: true,
                    },
                )?;
            }
            ExecutionPhase::Decode => {
                FullAttentionOps::run(&self.attention, s, domain.meta, scratch(&self.scratch, s))?
            }
        }
        if detailed {
            dump.device("q_rope", s, &mixer.q, 0, rows, 6144)?;
            dump.device("attention_gate", s, &mixer.gate, 0, rows, 6144)?;
            dump.device("attention_core", s, &self.work.attn, 0, rows, 6144)?;
            for (name, buf) in [("k_cache", &state.kv[0].k), ("v_cache", &state.kv[0].v)] {
                dump.bf16(name, vec![4, self.plan.capacity, 256], s.clone_dtoh(buf)?)?;
            }
        }
        self.common.gate.sigmoid_mul(
            s,
            domain,
            heads(&self.work.attn, s, 0, rows, 24, 256)?,
            heads(&mixer.gate, s, 0, rows, 24, 256)?,
            &mut TensorMut(heads(&self.work.attn, s, 0, rows, 24, 256)?),
        )?;
        if detailed {
            dump.device("attention_gated", s, &self.work.attn, 0, rows, 6144)?;
        }
        self.linear.run(
            blas,
            "self_attn.o_proj.weight",
            self.plan.phase,
            matrix(&self.work.attn, s, 0, rows, 6144)?,
            &mut TensorMut(matrix(&self.work.out, s, 0, rows, HIDDEN)?),
        )?;
        if detailed {
            dump.device("o_proj", s, &self.work.out, 0, rows, HIDDEN)?;
        }
        unsafe {
            s.launch_builder(&ops.residual)
                .arg(&mut self.work.x)
                .arg(&self.work.out)
                .arg(&((rows * HIDDEN) as i32))
                .launch(flat(rows * HIDDEN))?;
        }
        if detailed {
            dump.device("residual1", s, &self.work.x, 0, rows, HIDDEN)?;
        }
        norm(
            &self.work.x,
            &self.work.n,
            "mtp.layers.0.post_attention_layernorm.weight",
        )?;
        if detailed {
            dump.device("norm2", s, &self.work.n, 0, rows, HIDDEN)?;
        }
        for (role, offset) in [
            ("mlp.gate_proj.weight", 0),
            ("mlp.up_proj.weight", rows * 17408),
        ] {
            self.linear.run(
                blas,
                role,
                self.plan.phase,
                matrix(&self.work.n, s, 0, rows, HIDDEN)?,
                &mut TensorMut(matrix(&self.work.gu, s, offset, rows, 17408)?),
            )?;
            if detailed {
                dump.device(role, s, &self.work.gu, offset, rows, 17408)?;
            }
        }
        self.common.gate.silu(
            s,
            domain,
            heads(&self.work.gu, s, 0, rows, 1, 17408)?,
            &mut TensorMut(heads(&self.work.act, s, 0, rows, 1, 17408)?),
        )?;
        if detailed {
            dump.device("mlp_silu", s, &self.work.act, 0, rows, 17408)?;
        }
        let a = matrix(&self.work.act, s, 0, rows, 17408)?;
        let up = matrix(&self.work.gu, s, rows * 17408, rows, 17408)?;
        unsafe {
            s.launch_builder(&self.multiply)
                .arg(&a.ptr)
                .arg(&up.ptr)
                .arg(&a.ptr)
                .arg(&((rows * 17408) as u64))
                .launch(flat(rows * 17408))?;
        }
        if detailed {
            dump.device("mlp_activation", s, &self.work.act, 0, rows, 17408)?;
        }
        self.linear.run(
            blas,
            "mlp.down_proj.weight",
            self.plan.phase,
            a,
            &mut TensorMut(matrix(&self.work.out, s, 0, rows, HIDDEN)?),
        )?;
        if detailed {
            dump.device("mlp_down", s, &self.work.out, 0, rows, HIDDEN)?;
        }
        unsafe {
            s.launch_builder(&ops.residual)
                .arg(&mut self.work.x)
                .arg(&self.work.out)
                .arg(&((rows * HIDDEN) as i32))
                .launch(flat(rows * HIDDEN))?;
        }
        if detailed {
            dump.device("pre_mtp_norm", s, &self.work.x, 0, rows, HIDDEN)?;
        }
        norm(&self.work.x, &self.work.n, "mtp.norm.weight")?;
        if detailed {
            dump.device("post_mtp_norm", s, &self.work.n, 0, rows, HIDDEN)?;
        }
        blas.linear_typed(
            self.plan.phase,
            crate::backend::cuda::linear::LinearWeightRef::Bf16(matrix(head, s, 0, VOCAB, HIDDEN)?),
            matrix(&self.work.n, s, 0, rows, HIDDEN)?,
            &mut TensorMut(matrix(&self.logits, s, 0, rows, VOCAB)?),
        )?;
        dump.device("logits", s, &self.logits, 0, rows, VOCAB)?;
        Ok(dump)
    }
}

struct RunInput<'a> {
    prefix: usize,
    ids: &'a [u32],
    hidden: &'a [bf16],
    #[cfg(test)]
    control: Control,
    detailed: bool,
}

#[derive(Default)]
struct Dump {
    tensors: BTreeMap<String, (safetensors::Dtype, Vec<usize>, Vec<u8>)>,
}
impl Dump {
    fn insert(
        &mut self,
        name: &str,
        dtype: safetensors::Dtype,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    ) -> Result<()> {
        ensure!(
            self.tensors
                .insert(name.into(), (dtype, shape, bytes))
                .is_none(),
            "duplicate MTP dump {name}"
        );
        Ok(())
    }
    fn bf16(&mut self, name: &str, shape: Vec<usize>, data: Vec<bf16>) -> Result<()> {
        ensure!(
            data.iter().all(|x| x.is_finite()),
            "nonfinite MTP dump {name}"
        );
        self.insert(
            name,
            safetensors::Dtype::BF16,
            shape,
            data.into_iter()
                .flat_map(|x| x.to_bits().to_le_bytes())
                .collect(),
        )
    }
    fn i32(&mut self, name: &str, shape: Vec<usize>, data: &[i32]) -> Result<()> {
        self.insert(
            name,
            safetensors::Dtype::I32,
            shape,
            data.iter().flat_map(|x| x.to_le_bytes()).collect(),
        )
    }
    fn u32(&mut self, name: &str, shape: Vec<usize>, data: &[u32]) -> Result<()> {
        self.insert(
            name,
            safetensors::Dtype::U32,
            shape,
            data.iter().flat_map(|x| x.to_le_bytes()).collect(),
        )
    }
    fn device(
        &mut self,
        name: &str,
        s: &Arc<CudaStream>,
        buffer: &CudaSlice<bf16>,
        offset: usize,
        rows: usize,
        width: usize,
    ) -> Result<()> {
        self.bf16(
            name,
            vec![rows, width],
            s.clone_dtoh(&buffer.slice(offset..offset + rows * width))?,
        )
    }
    #[cfg(test)]
    fn save(&self, path: &Path, metadata: std::collections::HashMap<String, String>) -> Result<()> {
        let tensors = self
            .tensors
            .iter()
            .map(|(n, (dtype, shape, data))| {
                Ok((
                    n.as_str(),
                    safetensors::tensor::TensorView::new(*dtype, shape.clone(), data)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        safetensors::serialize_to_file(tensors, Some(metadata), path)?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "engine_mtp_tests.rs"]
mod tests;

#[path = "engine_mtp_chain.rs"]
mod chain;
pub(super) use chain::Chain;
