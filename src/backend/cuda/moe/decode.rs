//! Decode expert FFN contribute stage: top-k routed experts behind device
//! slot tables plus the shared expert, rows 1..8, writing per-(row, choice)
//! BF16 contributions, the ungated shared row and the per-assignment done
//! bits consumed by the shared combine op.
use crate::backend::cuda::contracts::{Scratch, Tensor, TensorMut};
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

/// W8A16 decode input: the MoeDecode fields with slots_gate_up and
/// slots_down carrying expert slot BASE pointers, one per expert, each
/// covering the whole 6-slice slot (3*I*H code bytes then (2*I + H)*2 BF16
/// scale bytes); the shared expert tensors stay BF16.
pub(crate) struct MoeDecodeW8a16<'a> {
    pub routed: MoeDecode<'a>,
    pub slot_bytes: u64,
}

/// One W8A16 expert slot: 3*I*H E4M3 code bytes plus (2*I + H) BF16 scale
/// bytes; 3,151,872 at I=512, H=2048.
pub(crate) fn slot_bytes_w8a16(hidden: u32, inter: u32) -> u64 {
    3 * u64::from(inter) * u64::from(hidden) + (2 * u64::from(inter) + u64::from(hidden)) * 2
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

fn spans_disjoint(a_ptr: u64, a_bytes: u64, b_ptr: u64, b_bytes: u64) -> bool {
    a_ptr + a_bytes <= b_ptr || b_ptr + b_bytes <= a_ptr
}

pub(super) fn validate(input: &MoeDecode<'_>, scratch: &Scratch) -> Result<DecodeDims> {
    input.x.validate()?;
    let (rows, hidden) = (input.x.shape[0], input.x.shape[1]);
    ensure!(
        (1..=8).contains(&rows),
        "decode rows {rows} outside [1,8] (x shape {:?})",
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
            input.contributions.ptr,
            input.contributions.span_bytes
        ) && spans_disjoint(
            shared_out.ptr,
            shared_out.span_bytes,
            input.contributions.ptr,
            input.contributions.span_bytes
        ) && spans_disjoint(
            done.ptr,
            done.span_bytes,
            input.contributions.ptr,
            input.contributions.span_bytes
        ),
        "decode contributions overlap x/shared_out/done (contributions ptr {:#x} bytes {}, x ptr {:#x}, shared_out ptr {:#x}, done ptr {:#x})",
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
            spans_disjoint(scratch.ptr, scratch.bytes, ptr, bytes),
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

pub(super) fn validate_w8a16(input: &MoeDecodeW8a16<'_>, scratch: &Scratch) -> Result<DecodeDims> {
    let dims = validate(&input.routed, scratch)?;
    let expected = slot_bytes_w8a16(dims.hidden, dims.inter);
    ensure!(
        input.slot_bytes == expected,
        "w8a16 decode slot_bytes {} does not match the dims-derived {} (hidden {}, inter {})",
        input.slot_bytes,
        expected,
        dims.hidden,
        dims.inter
    );
    Ok(dims)
}

impl DecodeOps {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let module = cubin::module(ctx, "moe_ffn")?;
        Ok(Self {
            gate_up: module.load_function("moe_ffn_gate_up")?,
            down: module.load_function("moe_ffn_down")?,
            gate_up_w8a16: module.load_function("moe_ffn_gate_up_w8a16")?,
            down_w8a16: module.load_function("moe_ffn_down_w8a16")?,
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
        let DecodeDims {
            rows,
            experts,
            top_k,
            hidden,
            inter,
        } = validate(input, scratch)?;
        let h_bytes = u64::from(rows) * u64::from(top_k) * u64::from(inter) * 2;
        let h_ptr = scratch.ptr;
        let hs_ptr = scratch.ptr + h_bytes;
        unsafe {
            stream
                .launch_builder(&self.gate_up)
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
                    grid_dim: (inter.div_ceil(32), rows, top_k + 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: hidden * 2,
                })?;
            stream
                .launch_builder(&self.down)
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
                    grid_dim: (hidden.div_ceil(16), rows, top_k + 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// One full decode MoE row block: contribute, shared sigmoid scale and
    /// the single deterministic combine, with final_active = active.
    #[cfg(test)]
    pub fn decode(
        &self,
        stream: &Arc<CudaStream>,
        input: &MoeDecode<'_>,
        y: &mut TensorMut<bf16, 2>,
        scratch: &Scratch,
    ) -> Result<()> {
        self.contribute(stream, input, scratch)?;
        let rows = input.x.shape[0];
        let top_k = input.ids.shape[1];
        let inter = input.shared_gate.shape[0];
        let scale_ptr = scratch.ptr + u64::from(rows) * u64::from(top_k + 1) * u64::from(inter) * 2;
        // moe_combine documents the scale as [rows,1] x stride [1,0];
        // contracts::Tensor::validate rejects zero strides, so this view of
        // the same [rows] words carries [1,1].
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

    /// W8A16 contribute: the same launch structure and guarantees as
    /// [`DecodeOps::contribute`], with routed experts read from their slot
    /// bases as E4M3 codes plus per-output-channel BF16 scales.
    pub fn contribute_w8a16(
        &self,
        stream: &Arc<CudaStream>,
        input: &MoeDecodeW8a16<'_>,
        scratch: &Scratch,
    ) -> Result<CudaEvent> {
        let DecodeDims {
            rows,
            experts,
            top_k,
            hidden,
            inter,
        } = validate_w8a16(input, scratch)?;
        let routed = &input.routed;
        let h_bytes = u64::from(rows) * u64::from(top_k) * u64::from(inter) * 2;
        let h_ptr = scratch.ptr;
        let hs_ptr = scratch.ptr + h_bytes;
        unsafe {
            stream
                .launch_builder(&self.gate_up_w8a16)
                .arg(&routed.x.ptr)
                .arg(&routed.ids.ptr)
                .arg(&routed.active.ptr)
                .arg(&routed.slots_gate_up.ptr)
                .arg(&routed.slot_valid.ptr)
                .arg(&routed.slot_generation.ptr)
                .arg(&routed.assignment_gen.ptr)
                .arg(&routed.shared_gate.ptr)
                .arg(&routed.shared_up.ptr)
                .arg(&h_ptr)
                .arg(&hs_ptr)
                .arg(&(rows as i32))
                .arg(&(experts as i32))
                .arg(&(top_k as i32))
                .arg(&(hidden as i32))
                .arg(&(inter as i32))
                .launch(LaunchConfig {
                    grid_dim: (inter.div_ceil(32), rows, top_k + 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: hidden * 2,
                })?;
            stream
                .launch_builder(&self.down_w8a16)
                .arg(&h_ptr)
                .arg(&hs_ptr)
                .arg(&routed.ids.ptr)
                .arg(&routed.active.ptr)
                .arg(&routed.weights.ptr)
                .arg(&routed.slots_down.ptr)
                .arg(&routed.slot_valid.ptr)
                .arg(&routed.slot_generation.ptr)
                .arg(&routed.assignment_gen.ptr)
                .arg(&routed.shared_down.ptr)
                .arg(&routed.contributions.ptr)
                .arg(&routed.shared_out.read().ptr)
                .arg(&routed.done.read().ptr)
                .arg(&(rows as i32))
                .arg(&(experts as i32))
                .arg(&(top_k as i32))
                .arg(&(hidden as i32))
                .arg(&(inter as i32))
                .launch(LaunchConfig {
                    grid_dim: (hidden.div_ceil(16), rows, top_k + 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(stream.record_event(None)?)
    }

    /// One full W8A16 decode MoE row block: contribute_w8a16, shared sigmoid
    /// scale and the single deterministic combine, with final_active =
    /// active.
    pub fn decode_w8a16(
        &self,
        stream: &Arc<CudaStream>,
        input: &MoeDecodeW8a16<'_>,
        y: &mut TensorMut<bf16, 2>,
        scratch: &Scratch,
    ) -> Result<CudaEvent> {
        self.contribute_w8a16(stream, input, scratch)?;
        let routed = &input.routed;
        let rows = routed.x.shape[0];
        let top_k = routed.ids.shape[1];
        let inter = routed.shared_gate.shape[0];
        let scale_ptr = scratch.ptr + u64::from(rows) * u64::from(top_k + 1) * u64::from(inter) * 2;
        // moe_combine documents the scale as [rows,1] x stride [1,0];
        // contracts::Tensor::validate rejects zero strides, so this view of
        // the same [rows] words carries [1,1].
        let mut scale = unsafe {
            TensorMut::<bf16, 2>::new(scale_ptr, [rows, 1], [1, 1], u64::from(rows) * 2)?
        };
        self.combine
            .shared_scale(stream, routed.x, routed.shared_expert_gate, &mut scale)?;
        let combine_input = MoeCombine {
            ids: routed.ids,
            final_active: routed.active,
            done: routed.done.read(),
            contributions: routed.contributions,
            shared_out: routed.shared_out.read(),
            x: routed.x,
            shared_expert_gate: routed.shared_expert_gate,
            scale,
            y: TensorMut(y.read()),
        };
        self.combine.combine(stream, &combine_input)
    }
}
