//! Decode expert FFN contribute stage: top-k routed experts behind device
//! slot tables plus the shared expert, rows 1..8, writing per-(row, choice)
//! BF16 contributions, the ungated shared row and the per-assignment done
//! bits consumed by the shared combine op.
use crate::backend::cuda::contracts::{Scratch, Tensor, TensorMut, spans_disjoint};
use crate::backend::cuda::cubin;
#[cfg(test)]
use crate::backend::cuda::moe::combine::{CombineOps, MoeCombine};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use half::bf16;
use std::sync::Arc;

pub(crate) struct DecodeOps {
    gate_up: CudaFunction,
    down: CudaFunction,
    invariant: Option<(CudaFunction, CudaFunction)>,
    gate_up_w8a16: CudaFunction,
    down_w8a16: CudaFunction,
    #[cfg(test)]
    combine: CombineOps,
}

pub(crate) struct MoeDecode<'a> {
    pub x: Tensor<bf16, 2>,
    pub ids: Tensor<i32, 2>,
    pub active: Tensor<u32, 2>,
    pub weights: Tensor<bf16, 2>,
    pub slots_gate_up: Tensor<u64, 1>,
    pub slots_down: Tensor<u64, 1>,
    pub slot_valid: Tensor<u32, 1>,
    pub slot_generation: Tensor<u64, 1>,
    pub assignment_gen: Tensor<u64, 2>,
    pub shared_gate: Tensor<bf16, 2>,
    pub shared_up: Tensor<bf16, 2>,
    pub shared_down: Tensor<bf16, 2>,
    pub shared_expert_gate: Tensor<bf16, 1>,
    /// `[rows, top_k, hidden]` weighted BF16 contributions (dense).
    pub contributions: Tensor<bf16, 3>,
    pub shared_out: &'a mut TensorMut<bf16, 2>,
    pub done: &'a mut TensorMut<u32, 2>,
}

/// h `[rows*top_k, inter]` bf16 + h_shared `[rows, inter]` bf16, then the
/// `[rows]` bf16 scale of moe_shared_scale; inter % 8 == 0 keeps every region
/// 16-byte aligned.
pub(crate) fn decode_scratch_bytes(rows: usize, k: usize, inter: usize) -> u64 {
    rows as u64 * (k as u64 + 1) * inter as u64 * 2 + rows as u64 * 2
}

pub(super) struct DecodeDims {
    rows: u32,
    experts: u32,
    top_k: u32,
    hidden: u32,
    inter: u32,
}

pub(super) fn validate(input: &MoeDecode<'_>, scratch: &Scratch) -> Result<DecodeDims> {
    input.x.validate()?;
    let (rows, hidden) = (input.x.shape[0], input.x.shape[1]);
    ensure!(
        (1..=crate::linear_plan::MAX_SKINNY_ROWS as u32).contains(&rows),
        "decode rows {rows} exceeds skinny bound={} (x shape {:?})",
        crate::linear_plan::MAX_SKINNY_ROWS,
        input.x.shape
    );
    ensure!(
        input.x.stride == [u64::from(hidden), 1],
        "decode x must be row-contiguous [rows, hidden] (shape {:?}, stride {:?})",
        input.x.shape,
        input.x.stride
    );
    input.ids.validate()?;
    let top_k = input.ids.shape[1];
    ensure!(
        input.ids.shape == [rows, top_k] && input.ids.stride == [u64::from(top_k), 1],
        "decode ids must be row-contiguous [rows, top_k] (shape {:?}, stride {:?})",
        input.ids.shape,
        input.ids.stride
    );
    input.assignment_gen.validate()?;
    ensure!(
        input.assignment_gen.shape == [rows, top_k]
            && input.assignment_gen.stride == [u64::from(top_k), 1],
        "decode assignment_gen must be row-contiguous [rows, top_k] (shape {:?}, stride {:?})",
        input.assignment_gen.shape,
        input.assignment_gen.stride
    );
    input.active.validate()?;
    ensure!(
        input.active.shape == [rows, top_k] && input.active.stride == [u64::from(top_k), 1],
        "decode active must be row-contiguous [rows, top_k] (shape {:?}, stride {:?})",
        input.active.shape,
        input.active.stride
    );
    input.weights.validate()?;
    ensure!(
        input.weights.shape == [rows, top_k] && input.weights.stride == [u64::from(top_k), 1],
        "decode weights must be row-contiguous [rows, top_k] (shape {:?}, stride {:?})",
        input.weights.shape,
        input.weights.stride
    );
    input.contributions.validate()?;
    ensure!(
        input.contributions.shape == [rows, top_k, hidden]
            && input.contributions.stride
                == [u64::from(top_k) * u64::from(hidden), u64::from(hidden), 1],
        "decode contributions must be dense [rows, top_k, hidden] (shape {:?}, stride {:?}, hidden {hidden})",
        input.contributions.shape,
        input.contributions.stride
    );
    let shared_out = input.shared_out.read();
    shared_out.validate()?;
    ensure!(
        shared_out.shape == [rows, hidden] && shared_out.stride == [u64::from(hidden), 1],
        "decode shared_out must be row-contiguous [rows, hidden] (shape {:?}, stride {:?})",
        shared_out.shape,
        shared_out.stride
    );
    let done = input.done.read();
    done.validate()?;
    ensure!(
        done.shape == [rows, top_k] && done.stride == [u64::from(top_k), 1],
        "decode done must be row-contiguous [rows, top_k] (shape {:?}, stride {:?})",
        done.shape,
        done.stride
    );
    input.slots_gate_up.validate()?;
    let experts = input.slots_gate_up.shape[0];
    input.slots_down.validate()?;
    input.slot_valid.validate()?;
    input.slot_generation.validate()?;
    ensure!(
        input.slots_gate_up.stride == [1]
            && input.slots_down.stride == [1]
            && input.slot_valid.stride == [1]
            && input.slot_generation.stride == [1]
            && input.slots_down.shape == [experts]
            && input.slot_valid.shape == [experts]
            && input.slot_generation.shape == [experts],
        "decode slot tables must be contiguous [experts] (gate_up {}, down {}, valid {}, generation {}, experts {experts})",
        input.slots_gate_up.shape[0],
        input.slots_down.shape[0],
        input.slot_valid.shape[0],
        input.slot_generation.shape[0]
    );
    input.shared_gate.validate().context("decode shared_gate")?;
    input.shared_up.validate().context("decode shared_up")?;
    let inter = input.shared_gate.shape[0];
    for t in [&input.shared_gate, &input.shared_up] {
        ensure!(
            t.shape == [inter, hidden] && t.stride == [u64::from(hidden), 1],
            "decode shared gate/up must be row-contiguous [inter, hidden] (shape {:?}, stride {:?}, inter {inter}, hidden {hidden})",
            t.shape,
            t.stride
        );
    }
    input.shared_down.validate().context("decode shared_down")?;
    ensure!(
        input.shared_down.shape == [hidden, inter]
            && input.shared_down.stride == [u64::from(inter), 1],
        "decode shared_down must be row-contiguous [hidden, inter] (shape {:?}, stride {:?}, hidden {hidden}, inter {inter})",
        input.shared_down.shape,
        input.shared_down.stride
    );
    input
        .shared_expert_gate
        .validate()
        .context("decode shared_expert_gate")?;
    ensure!(
        input.shared_expert_gate.shape == [hidden] && input.shared_expert_gate.stride == [1],
        "decode shared expert gate must be a contiguous [hidden] vector (shape {:?}, stride {:?}, hidden {hidden})",
        input.shared_expert_gate.shape,
        input.shared_expert_gate.stride
    );
    ensure!(
        spans_disjoint(
            input.x.ptr,
            input.x.span_bytes,
            shared_out.ptr,
            shared_out.span_bytes
        )? && spans_disjoint(
            input.x.ptr,
            input.x.span_bytes,
            input.contributions.ptr,
            input.contributions.span_bytes
        )? && spans_disjoint(
            shared_out.ptr,
            shared_out.span_bytes,
            input.contributions.ptr,
            input.contributions.span_bytes
        )? && spans_disjoint(
            done.ptr,
            done.span_bytes,
            input.contributions.ptr,
            input.contributions.span_bytes
        )?,
        "decode contributions/shared_out overlap x/shared_out/done (contributions ptr {:#x} bytes {}, x ptr {:#x}, shared_out ptr {:#x}, done ptr {:#x})",
        input.contributions.ptr,
        input.contributions.span_bytes,
        input.x.ptr,
        shared_out.ptr,
        done.ptr
    );
    for (name, ptr, bytes) in [
        ("x", input.x.ptr, input.x.span_bytes),
        (
            "contributions",
            input.contributions.ptr,
            input.contributions.span_bytes,
        ),
        ("shared_out", shared_out.ptr, shared_out.span_bytes),
        ("done", done.ptr, done.span_bytes),
    ] {
        ensure!(
            spans_disjoint(scratch.ptr, scratch.bytes, ptr, bytes)?,
            "decode scratch overlaps {name} (scratch ptr {:#x} bytes {}, {name} ptr {ptr:#x} bytes {bytes})",
            scratch.ptr,
            scratch.bytes
        );
    }
    ensure!(
        (1..=32).contains(&top_k),
        "decode top_k {top_k} outside [1,32] (ids shape {:?})",
        input.ids.shape
    );
    ensure!(
        hidden.is_multiple_of(8),
        "decode hidden {hidden} must be a multiple of 8"
    );
    ensure!(
        inter.is_multiple_of(8),
        "decode inter {inter} must be a multiple of 8"
    );
    ensure!(
        u64::from(hidden) * 2 <= 48 * 1024,
        "decode x row of {hidden} exceeds the gate_up kernel shared bytes"
    );
    scratch.validate(decode_scratch_bytes(
        rows as usize,
        top_k as usize,
        inter as usize,
    ))?;
    Ok(DecodeDims {
        rows,
        experts,
        top_k,
        hidden,
        inter,
    })
}

pub(super) fn validate_w8a16(
    input: &MoeDecode<'_>,
    slot: &crate::experts::catalogue::Fp8Slot,
    scratch: &Scratch,
) -> Result<DecodeDims> {
    use crate::{
        experts::catalogue::{ExpertGeometry, ExpertLayout},
        model_spec::ExpertFormat,
    };
    let dims = validate(input, scratch)?;
    let geometry = ExpertGeometry {
        layers: 1,
        experts: dims.experts,
        hidden: dims.hidden,
        intermediate: dims.inter,
        top_k: dims.top_k,
    };
    ensure!(
        ExpertLayout::new(ExpertFormat::Fp8W8a16, &geometry)?.fp8_slot() == Some(slot),
        "W8A16 decode slot differs from geometry"
    );
    Ok(dims)
}

impl DecodeOps {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let module = cubin::module(ctx, "moe_ffn")?;
        Ok(Self {
            gate_up: module.load_function("moe_ffn_gate_up")?,
            down: module.load_function("moe_ffn_down")?,
            invariant: None,
            gate_up_w8a16: module.load_function("moe_ffn_gate_up_w8a16")?,
            down_w8a16: module.load_function("moe_ffn_down_w8a16")?,
            #[cfg(test)]
            combine: CombineOps::new(ctx)?,
        })
    }

    pub(crate) fn new_invariant(ctx: &Arc<CudaContext>) -> Result<Self> {
        let module = cubin::module(ctx, "moe_ffn")?;
        Ok(Self {
            gate_up: module.load_function("moe_ffn_gate_up_lane16")?,
            down: module.load_function("moe_ffn_down_lane16")?,
            gate_up_w8a16: module.load_function("moe_ffn_gate_up_w8a16_lane16")?,
            down_w8a16: module.load_function("moe_ffn_down_w8a16_lane16")?,
            invariant: Some((
                cubin::module(ctx, "gemm_skinny")?.load_function("gemm_skinny_bf16")?,
                cubin::module(ctx, "act")?.load_function("silu_mul_split")?,
            )),
            #[cfg(test)]
            combine: CombineOps::new(ctx)?,
        })
    }

    /// Contributes one cohort; inputs and slot tables stay immutable until the stream completes.
    pub fn contribute(
        &self,
        stream: &Arc<CudaStream>,
        input: &MoeDecode<'_>,
        scratch: &Scratch,
    ) -> Result<()> {
        let dims = validate(input, scratch)?;
        self.launch(stream, input, scratch, dims, (&self.gate_up, &self.down))
    }

    fn launch(
        &self,
        stream: &Arc<CudaStream>,
        input: &MoeDecode<'_>,
        scratch: &Scratch,
        dims: DecodeDims,
        functions: (&CudaFunction, &CudaFunction),
    ) -> Result<()> {
        let DecodeDims {
            rows,
            experts,
            top_k,
            hidden,
            inter,
        } = dims;
        ensure!(
            self.invariant.is_some() || rows <= 8,
            "legacy decode row bound=8, requested={rows}"
        );
        ensure!(
            self.invariant.is_none()
                || (u64::from(hidden) >= 2 * u64::from(inter)
                    && hidden.is_multiple_of(256)
                    && inter.is_multiple_of(256)),
            "invariant shared expert requires hidden>=2*inter and 256-aligned widths: hidden={hidden} inter={inter}"
        );
        let invariant = u32::from(self.invariant.is_some());
        let h_bytes = u64::from(rows) * u64::from(top_k) * u64::from(inter) * 2;
        let h_ptr = scratch.ptr;
        let hs_ptr = scratch.ptr + h_bytes;
        unsafe {
            stream
                .launch_builder(functions.0)
                .arg(&input.x.ptr)
                .arg(&input.ids.ptr)
                .arg(&input.active.ptr)
                .arg(&input.slots_gate_up.ptr)
                .arg(&input.slot_valid.ptr)
                .arg(&input.slot_generation.ptr)
                .arg(&input.assignment_gen.ptr)
                .arg(&input.shared_gate.ptr)
                .arg(&input.shared_up.ptr)
                .arg(&h_ptr)
                .arg(&hs_ptr)
                .arg(&(rows as i32))
                .arg(&(experts as i32))
                .arg(&(top_k as i32))
                .arg(&(hidden as i32))
                .arg(&(inter as i32))
                .launch(LaunchConfig {
                    grid_dim: (inter.div_ceil(32 >> invariant), rows, top_k + 1 - invariant),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: hidden * 2,
                })?;
            stream
                .launch_builder(functions.1)
                .arg(&h_ptr)
                .arg(&hs_ptr)
                .arg(&input.ids.ptr)
                .arg(&input.active.ptr)
                .arg(&input.weights.ptr)
                .arg(&input.slots_down.ptr)
                .arg(&input.slot_valid.ptr)
                .arg(&input.slot_generation.ptr)
                .arg(&input.assignment_gen.ptr)
                .arg(&input.shared_down.ptr)
                .arg(&input.contributions.ptr)
                .arg(&input.shared_out.read().ptr)
                .arg(&input.done.read().ptr)
                .arg(&(rows as i32))
                .arg(&(experts as i32))
                .arg(&(top_k as i32))
                .arg(&(hidden as i32))
                .arg(&(inter as i32))
                .launch(LaunchConfig {
                    grid_dim: (
                        hidden.div_ceil(16 >> invariant),
                        rows,
                        top_k + 1 - invariant,
                    ),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        if let Some((skinny, silu)) = &self.invariant {
            let shared = input.shared_out.read().ptr;
            let up = shared + u64::from(rows) * u64::from(inter) * 2;
            let gemm = |weight: u64, x: u64, y: u64, columns: u32, inner: u32| -> Result<()> {
                // SAFETY: validated dense regions cover every complete weight and activation tile.
                unsafe {
                    stream
                        .launch_builder(skinny)
                        .arg(&weight)
                        .arg(&x)
                        .arg(&y)
                        .arg(&(columns as i32))
                        .arg(&(inner as i32))
                        .arg(&i64::from(inner))
                        .arg(&(rows as i32))
                        .launch(crate::backend::cuda::blas::Blas::skinny_launch(
                            crate::backend::setup::LinearShape {
                                rows: rows as usize,
                                output: columns as usize,
                                input: inner as usize,
                            },
                        )?)?;
                }
                Ok(())
            };
            gemm(input.shared_gate.ptr, input.x.ptr, shared, inter, hidden)?;
            gemm(input.shared_up.ptr, input.x.ptr, up, inter, hidden)?;
            // SAFETY: shared_out holds two rows*inter regions; hs_ptr is disjoint scratch.
            unsafe {
                stream
                    .launch_builder(silu)
                    .arg(&shared)
                    .arg(&up)
                    .arg(&hs_ptr)
                    .arg(&((rows * inter) as i32))
                    .launch(crate::backend::cuda::ops::flat((rows * inter) as usize))?;
            }
            gemm(input.shared_down.ptr, hs_ptr, shared, hidden, inter)?;
        }
        Ok(())
    }

    /// Native expert contributions and the shared expert use one deterministic combine.
    #[cfg(test)]
    pub fn decode(
        &self,
        stream: &Arc<CudaStream>,
        input: &MoeDecode<'_>,
        y: &mut TensorMut<bf16, 2>,
        scratch: &Scratch,
        layout: Option<&crate::experts::catalogue::Fp8Slot>,
    ) -> Result<()> {
        if let Some(slot) = layout {
            self.contribute_w8a16(stream, input, slot, scratch)?;
        } else {
            self.contribute(stream, input, scratch)?;
        }
        let rows = input.x.shape[0];
        let top_k = input.ids.shape[1];
        let inter = input.shared_gate.shape[0];
        let scale_ptr = scratch.ptr + u64::from(rows) * u64::from(top_k + 1) * u64::from(inter) * 2;
        // The contiguous shared scale has shape [rows,1] and strides [1,1].
        let mut scale = unsafe {
            TensorMut::<bf16, 2>::new(scale_ptr, [rows, 1], [1, 1], u64::from(rows) * 2)?
        };
        self.combine
            .shared_scale(stream, input.x, input.shared_expert_gate, &mut scale)?;
        let combine_input = MoeCombine {
            ids: input.ids,
            final_active: input.active,
            done: input.done.read(),
            contributions: input.contributions,
            shared_out: input.shared_out.read(),
            x: input.x,
            shared_expert_gate: input.shared_expert_gate,
            scale,
            y: TensorMut(y.read()),
        };
        self.combine.combine(stream, &combine_input)
    }

    pub fn contribute_w8a16(
        &self,
        stream: &Arc<CudaStream>,
        input: &MoeDecode<'_>,
        slot: &crate::experts::catalogue::Fp8Slot,
        scratch: &Scratch,
    ) -> Result<()> {
        let dims = validate_w8a16(input, slot, scratch)?;
        self.launch(
            stream,
            input,
            scratch,
            dims,
            (&self.gate_up_w8a16, &self.down_w8a16),
        )
    }
}
