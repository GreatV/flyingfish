//! Single-Engine MTP component shared by generation and numerical probes.
use super::{
    Work,
    qwen::{heads, limits, matrix, scratch, vector, view},
};
#[cfg(test)]
use crate::backend::cuda::closure::Closure;
#[cfg(test)]
use crate::model_spec::{LayerPlan, Mixer};
use crate::{
    backend::cuda::{
        Device,
        attention_d256::{
            AttentionBindings, FullAttentionOps, FullAttentionPlan, PositionBindings,
        },
        blas::Blas,
        contracts::{RowDomain, Tensor, TensorMut},
        cubin,
        flash::{Flash, Shape},
        linear::LayerLinearPlan,
        ops::{CommonOpsRef, Ops, ResidualRmsInput, RmsMode, RmsOptions, flat},
        state::StateStore,
        weights::WeightCatalog,
    },
    checkpoint::LoadPolicy,
    linear_plan::LayerDequantPlan,
    model_spec::ModelSpec,
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
    prediction: CudaSlice<u32>,
    arena: CudaSlice<u8>,
    scratch: CudaSlice<u8>,
    mask: CudaSlice<u64>,
    linear: LayerLinearPlan,
    attention: BTreeMap<usize, FullAttentionPlan>,
    max_position: usize,
    head_rows: usize,
    copy_finite: CudaFunction,
    stage_inputs: CudaFunction,
    #[cfg(test)]
    head_runs: usize,
    common: CommonOpsRef,
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
                + self.positions.len()
                + self.prediction.len())
                * 4
            + self.mask.len() * 8
            + self.arena.len()
            + self.scratch.len()
            + self.plan.max_rows * 24 * size_of::<f32>() // Flash's setup-bound LSE.
    }
    #[cfg(test)]
    fn new(
        device: &Device,
        spec: &ModelSpec,
        catalog: &WeightCatalog,
        state: &StateStore,
        blas: &Blas,
        plan: MtpPlan,
    ) -> Result<Self> {
        Self::allocate(device, spec, catalog, state, blas, plan, plan.max_rows)
    }
    fn allocate(
        device: &Device,
        spec: &ModelSpec,
        catalog: &WeightCatalog,
        state: &StateStore,
        blas: &Blas,
        plan: MtpPlan,
        head_rows: usize,
    ) -> Result<Self> {
        plan.validate(spec)?;
        let s = &device.stream;
        let rows = plan.max_rows;
        let work = Work::new(s, spec, rows)?;
        let embedding = s.alloc_zeros(rows * HIDDEN)?;
        let hidden = s.alloc_zeros(rows * HIDDEN)?;
        let concat = s.alloc_zeros(rows * HIDDEN * 2)?;
        let logits = s.alloc_zeros(head_rows * VOCAB)?;
        let positions = s.alloc_zeros(rows * 3)?;
        let dequant = LayerDequantPlan::mtp(&catalog.manifest)?;
        ensure!(
            dequant.bytes as usize == crate::mtp::ARENA_BYTES && dequant.slots.len() == 7,
            "unexpected MTP materialization layout"
        );
        let arena = s.alloc_zeros(dequant.bytes as usize)?;
        let scratch_buffer = s.alloc_zeros(plan.memory_with_head_rows(spec, head_rows)?.scratch)?;
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
        linear.materialize(blas.fp8_ops()?, s)?;
        let common = CommonOpsRef::new(&device.ctx)?;
        let mut runtime = Self {
            plan,
            work,
            embedding,
            hidden,
            concat,
            logits,
            positions,
            prediction: s.alloc_zeros(2)?,
            arena,
            scratch: scratch_buffer,
            mask,
            linear,
            attention: BTreeMap::new(),
            max_position: spec.max_position as usize,
            head_rows,
            copy_finite: cubin::module(&device.ctx, "q38_common")?
                .load_function("q38_copy_finite_bf16")?,
            stage_inputs: cubin::module(&device.ctx, "q38_common")?
                .load_function("q38_stage_inputs")?,
            #[cfg(test)]
            head_runs: 0,
            common,
            flash: Flash::new(s.clone(), rows, 24)?,
        };
        runtime.prepare_rows(device, catalog, state, blas, rows)?;
        Ok(runtime)
    }
    fn prepare_rows(
        &mut self,
        device: &Device,
        catalog: &WeightCatalog,
        state: &StateStore,
        blas: &Blas,
        rows: usize,
    ) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.plan.max_rows,
            "MTP binding exceeds workspace"
        );
        if self.attention.contains_key(&rows) {
            return Ok(());
        }
        let s = &device.stream;
        self.linear.prepare(blas, rows as u32)?;
        blas.prepare_fixed(1, HIDDEN as u32, (2 * HIDDEN) as u32)?;
        blas.prepare_fixed(rows as u32, HIDDEN as u32, (2 * HIDDEN) as u32)?;
        if rows <= self.head_rows {
            blas.prepare_fixed(rows as u32, VOCAB as u32, HIDDEN as u32)?;
        }
        let mixer = self
            .work
            .mixer
            .as_ref()
            .context("MTP Q/gate buffers missing")?;
        let kv = &state.kv[0];
        let attention = FullAttentionOps::prepare_reusing(
            &device.ctx,
            &crate::op_contract::FullAttentionSpec::q38(),
            limits(rows, self.plan.capacity)?,
            AttentionBindings {
                q_gate: heads(&self.work.qkv, s, 0, rows, 24, 512)?,
                k: heads(&self.work.qkv, s, rows * 12288, rows, 4, 256)?,
                v: heads(&self.work.qkv, s, rows * (12288 + 1024), rows, 4, 256)?,
                q: TensorMut(heads(&mixer.q, s, 0, rows, 24, 256)?),
                gate: TensorMut(heads(&mixer.gate, s, 0, rows, 24, 256)?),
                k_cache: TensorMut(view(
                    &kv.k,
                    s,
                    0,
                    [4, self.plan.capacity as u32, 256],
                    [(self.plan.capacity * 256) as u64, 256, 1],
                )?),
                v_cache: TensorMut(view(
                    &kv.v,
                    s,
                    0,
                    [4, self.plan.capacity as u32, 256],
                    [(self.plan.capacity * 256) as u64, 256, 1],
                )?),
                core_out: TensorMut(heads(&self.work.attn, s, 0, rows, 24, 256)?),
                q_norm: vector(catalog.bf16("mtp.layers.0.self_attn.q_norm.weight")?, s)?,
                k_norm: vector(catalog.bf16("mtp.layers.0.self_attn.k_norm.weight")?, s)?,
                causal_mask: view(&self.mask, s, 0, [64], [1])?,
                positions: PositionBindings::Axes(view(
                    &self.positions,
                    s,
                    0,
                    [rows as u32, 3],
                    [3, 1],
                )?),
            },
            self.common.clone(),
            self.attention.values().next(),
        )?;
        self.attention.insert(rows, attention);
        Ok(())
    }

    #[cfg(test)]
    fn run(
        &mut self,
        execution: Execution<'_>,
        state: &mut StateStore,
        input: RunInput<'_>,
    ) -> Result<Dump> {
        self.run_inner(
            execution,
            state,
            input,
            None,
            self.plan.phase,
            OutputMode::Predict,
        )
    }
    #[cfg(test)]
    fn run_decode(
        &mut self,
        execution: Execution<'_>,
        state: &mut StateStore,
        input: RunInput<'_>,
        output: OutputMode,
    ) -> Result<Dump> {
        let output = if output == OutputMode::Predict && !input.detailed {
            OutputMode::PredictDevice
        } else {
            output
        };
        self.run_inner(
            execution,
            state,
            input,
            None,
            ExecutionPhase::Decode,
            output,
        )
    }
    fn run_prefill(
        &mut self,
        execution: Execution<'_>,
        state: &mut StateStore,
        prefix: usize,
        ids: &[u32],
        hidden: Tensor<bf16, 2>,
    ) -> Result<()> {
        self.run_inner(
            execution,
            state,
            RunInput {
                prefix,
                ids,
                device_ids: None,
                hidden: &[],
                detailed: false,
                #[cfg(test)]
                control: Control::None,
            },
            Some(hidden),
            ExecutionPhase::Prefill,
            OutputMode::StateOnly,
        )?;
        Ok(())
    }
    fn run_device(
        &mut self,
        execution: Execution<'_>,
        state: &mut StateStore,
        prefix: usize,
        input: ((Tensor<u32, 1>, usize), Tensor<bf16, 2>, Tensor<u32, 1>),
        phase: ExecutionPhase,
        output: OutputMode,
    ) -> Result<()> {
        self.run_inner(
            execution,
            state,
            RunInput {
                prefix,
                ids: &[],
                device_ids: Some((input.0.0, input.0.1, input.2)),
                hidden: &[],
                detailed: false,
                #[cfg(test)]
                control: Control::None,
            },
            Some(input.1),
            phase,
            output,
        )?;
        Ok(())
    }
    fn run_inner(
        &mut self,
        execution: Execution<'_>,
        state: &mut StateStore,
        input: RunInput<'_>,
        device_hidden: Option<Tensor<bf16, 2>>,
        phase: ExecutionPhase,
        output: OutputMode,
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
            device_ids,
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
        let (rows, device_source) = if let Some((source, offset, _)) = device_ids {
            let pointer = source.element_ptr(offset)?;
            (source.shape[0] as usize - offset, Some(pointer))
        } else {
            (ids.len(), None)
        };
        self.plan
            .check_absolute_rows(prefix, rows, self.max_position)?;
        ensure!(
            phase != ExecutionPhase::Decode || rows == 1,
            "MTP decode requires one row"
        );
        ensure!(
            output != OutputMode::Repair || rows <= 3,
            "canonical MTP repair requires one to three rows"
        );
        ensure!(
            matches!(output, OutputMode::StateOnly | OutputMode::Repair) || rows <= self.head_rows,
            "MTP head rows exceed binding"
        );
        let attention = self
            .attention
            .get(&rows)
            .context("MTP rows were not bound before Ready")?;
        ensure!(
            device_hidden.is_some() || hidden.len() == rows * HIDDEN,
            "MTP call does not match bound rows/hidden"
        );
        ensure!(
            ids.iter().all(|&t| (t as usize) < VOCAB) && hidden.iter().all(|h| h.is_finite()),
            "invalid MTP inputs"
        );
        let s = &device.stream;
        if let Some((source, _, flag)) = device_ids {
            source.validate()?;
            flag.validate()?;
            ensure!(
                source.stride == [1] && flag.stride == [1] && !detailed,
                "invalid device MTP IDs"
            );
            let destination = view(&self.work.ids, s, 0, [rows as u32], [1])?;
            let source_ptr = device_source.context("device MTP source missing")?;
            let flag_ptr = flag.element_ptr(flag.shape[0] as usize - 1)?;
            unsafe {
                s.launch_builder(&self.stage_inputs)
                    .arg(&source_ptr)
                    .arg(&destination.ptr)
                    .arg(&self.positions)
                    .arg(state.sequence.as_mut().context("MTP sequence missing")?)
                    .arg(&(rows as u32))
                    .arg(&(prefix as i32))
                    .arg(&(self.plan.capacity as i32))
                    .arg(&(VOCAB as u32))
                    .arg(&flag_ptr)
                    .launch(flat(rows))?;
            }
        } else {
            state.set_sequence(s, prefix as u32, rows as u32, self.plan.max_rows as u32)?;
            s.memcpy_htod(ids, &mut self.work.ids.slice_mut(..rows))?;
        }
        if let Some(source) = device_hidden {
            source.validate()?;
            ensure!(
                source.shape == [rows as u32, HIDDEN as u32]
                    && source.stride == [HIDDEN as u64, 1]
                    && !detailed,
                "invalid device MTP hidden view"
            );
            let destination = matrix(&self.hidden, s, 0, rows, HIDDEN)?;
            unsafe {
                if matches!(output, OutputMode::Queued | OutputMode::Repair) {
                    cudarc::driver::result::memcpy_dtod_async(
                        destination.ptr,
                        source.ptr,
                        rows * HIDDEN * size_of::<bf16>(),
                        s.cu_stream(),
                    )?;
                } else {
                    s.launch_builder(&self.copy_finite)
                        .arg(&source.ptr)
                        .arg(&destination.ptr)
                        .arg(&((rows * HIDDEN) as u64))
                        .launch(flat(rows * HIDDEN))?;
                }
            }
        } else {
            s.memcpy_htod(hidden, &mut self.hidden.slice_mut(..rows * HIDDEN))?;
        }
        let axes: Vec<i32> = (0..if device_ids.is_none() { rows } else { 0 })
            .flat_map(|r| {
                let pos = (prefix + r + 1) as i32 + i32::from(shift_position);
                [pos; 3]
            })
            .collect();
        if device_ids.is_none() {
            s.memcpy_htod(&axes, &mut self.positions.slice_mut(..rows * 3))?;
        }
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
        let dumping = detailed || output == OutputMode::Predict;
        if dumping {
            dump.i32("positions", vec![rows, 3], &axes)?;
            dump.u32("input_ids", vec![rows], ids)?;
        }
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
            let cat = s.clone_dtoh(&self.concat.slice(..rows * 2 * HIDDEN))?;
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
        let fc_weight = catalog.linear_ref("mtp.fc.weight", s)?;
        output.for_each_fc(rows, |row, count| {
            blas.linear_typed(
                phase,
                fc_weight,
                matrix(&self.concat, s, row * 2 * HIDDEN, count, 2 * HIDDEN)?,
                &mut TensorMut(matrix(&self.work.x, s, row * HIDDEN, count, HIDDEN)?),
            )
        })?;
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
                phase,
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
        match output.attention_phase(phase) {
            ExecutionPhase::Prefill => {
                FullAttentionOps::preprocess(attention, s, domain.meta, scratch(&self.scratch, s))?;
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
                FullAttentionOps::run(attention, s, domain.meta, scratch(&self.scratch, s))?
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
            phase,
            matrix(&self.work.attn, s, 0, rows, 6144)?,
            &mut TensorMut(matrix(&self.work.out, s, 0, rows, HIDDEN)?),
        )?;
        if detailed {
            dump.device("o_proj", s, &self.work.out, 0, rows, HIDDEN)?;
        }
        let add_norm =
            |x: &CudaSlice<bf16>, r: &CudaSlice<bf16>, y: &CudaSlice<bf16>, name: &str| {
                self.common.norm.rms_add_with(
                    s,
                    domain,
                    ResidualRmsInput {
                        x: &mut TensorMut(heads(x, s, 0, rows, 1, HIDDEN as u32)?),
                        residual: heads(r, s, 0, rows, 1, HIDDEN as u32)?,
                        weight: vector(catalog.bf16(name)?, s)?,
                    },
                    &mut TensorMut(heads(y, s, 0, rows, 1, HIDDEN as u32)?),
                    RmsOptions {
                        mode: RmsMode::ZeroCentered,
                        eps: 1e-6,
                    },
                )
            };
        add_norm(
            &self.work.x,
            &self.work.out,
            &self.work.n,
            "mtp.layers.0.post_attention_layernorm.weight",
        )?;
        if detailed {
            dump.device("residual1", s, &self.work.x, 0, rows, HIDDEN)?;
            dump.device("norm2", s, &self.work.n, 0, rows, HIDDEN)?;
        }
        for (role, offset) in [
            ("mlp.gate_proj.weight", 0),
            ("mlp.up_proj.weight", rows * 17408),
        ] {
            self.linear.run(
                blas,
                role,
                phase,
                matrix(&self.work.n, s, 0, rows, HIDDEN)?,
                &mut TensorMut(matrix(&self.work.gu, s, offset, rows, 17408)?),
            )?;
            if detailed {
                dump.device(role, s, &self.work.gu, offset, rows, 17408)?;
            }
        }
        let gate = heads(&self.work.gu, s, 0, rows, 1, 17408)?;
        let up = heads(&self.work.gu, s, rows * 17408, rows, 1, 17408)?;
        let mut activated = TensorMut(heads(&self.work.act, s, 0, rows, 1, 17408)?);
        self.common
            .gate
            .swiglu(s, domain, gate, up, &mut activated)?;
        let a = matrix(&self.work.act, s, 0, rows, 17408)?;
        if detailed {
            dump.device("mlp_activation", s, &self.work.act, 0, rows, 17408)?;
        }
        self.linear.run(
            blas,
            "mlp.down_proj.weight",
            phase,
            a,
            &mut TensorMut(matrix(&self.work.out, s, 0, rows, HIDDEN)?),
        )?;
        if detailed {
            dump.device("mlp_down", s, &self.work.out, 0, rows, HIDDEN)?;
        }
        add_norm(
            &self.work.x,
            &self.work.out,
            &self.work.n,
            "mtp.norm.weight",
        )?;
        if detailed {
            dump.device("pre_mtp_norm", s, &self.work.x, 0, rows, HIDDEN)?;
            dump.device("post_mtp_norm", s, &self.work.n, 0, rows, HIDDEN)?;
        }
        if output == OutputMode::Repair {
            return Ok(dump);
        }
        if output == OutputMode::StateOnly {
            if phase == ExecutionPhase::Decode {
                // Repair failures must remain recoverable by the transaction's
                // restore path. Check on the host instead of trapping the GPU.
                ensure!(
                    s.clone_dtoh(&self.work.n.slice(..rows * HIDDEN))?
                        .iter()
                        .all(|h| h.is_finite()),
                    "nonfinite MTP normalized hidden"
                );
            } else {
                let source = matrix(&self.work.n, s, 0, rows, HIDDEN)?;
                let destination = matrix(&self.hidden, s, 0, rows, HIDDEN)?;
                unsafe {
                    s.launch_builder(&self.copy_finite)
                        .arg(&source.ptr)
                        .arg(&destination.ptr)
                        .arg(&((rows * HIDDEN) as u64))
                        .launch(flat(rows * HIDDEN))?;
                }
            }
            return Ok(dump);
        }
        #[cfg(test)]
        {
            self.head_runs += 1;
        }
        blas.linear_typed(
            phase,
            crate::backend::cuda::linear::LinearWeightRef::Bf16(matrix(head, s, 0, VOCAB, HIDDEN)?),
            matrix(&self.work.n, s, 0, rows, HIDDEN)?,
            &mut TensorMut(matrix(&self.logits, s, 0, rows, VOCAB)?),
        )?;
        if matches!(output, OutputMode::PredictDevice | OutputMode::Queued) {
            ensure!(rows == 1, "device draft head requires one row");
            if output == OutputMode::PredictDevice {
                s.memset_zeros(&mut self.prediction)?;
            }
            unsafe {
                s.launch_builder(&ops.argmax)
                    .arg(&self.logits)
                    .arg(&mut self.prediction.slice_mut(..1))
                    .arg(&(VOCAB as i32))
                    .launch(crate::backend::cuda::ops::grid(1, 1024))?;
            }
            if output == OutputMode::PredictDevice {
                unsafe {
                    s.launch_builder(&ops.finite_bf16)
                        .arg(&self.logits)
                        .arg(&mut self.prediction.slice_mut(1..2))
                        .arg(&(VOCAB as u64))
                        .launch(flat(VOCAB))?;
                }
                let result = s.clone_dtoh(&self.prediction)?;
                ensure!(result[1] == 0, "nonfinite MTP dump logits");
                ensure!((result[0] as usize) < VOCAB, "invalid MTP argmax ID");
                dump.prediction = Some(result[0]);
            }
        } else {
            dump.device("logits", s, &self.logits, 0, rows, VOCAB)?;
        }
        Ok(dump)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputMode {
    Predict,
    PredictDevice,
    StateOnly,
    Queued,
    Repair,
}

impl OutputMode {
    fn for_each_fc(
        self,
        rows: usize,
        mut run: impl FnMut(usize, usize) -> Result<()>,
    ) -> Result<()> {
        if self == Self::Repair {
            for row in 0..rows {
                run(row, 1)?;
            }
            Ok(())
        } else {
            run(0, rows)
        }
    }
    fn attention_phase(self, phase: ExecutionPhase) -> ExecutionPhase {
        if self == Self::Repair {
            ExecutionPhase::Decode
        } else {
            phase
        }
    }
}

struct RunInput<'a> {
    prefix: usize,
    ids: &'a [u32],
    device_ids: Option<(Tensor<u32, 1>, usize, Tensor<u32, 1>)>,
    hidden: &'a [bf16],
    #[cfg(test)]
    control: Control,
    detailed: bool,
}

#[derive(Default)]
struct Dump {
    prediction: Option<u32>,
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
