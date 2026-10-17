//! Decode expert FFN op wrappers: top-k routed experts behind device slot
//! tables plus the shared expert, rows 1..8.
use crate::backend::cuda::contracts::{Scratch, Tensor, TensorMut};
use crate::backend::cuda::cubin;
use anyhow::{Result, ensure};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use half::bf16;
use std::sync::Arc;

pub(crate) struct DecodeOps {
    gate_up: CudaFunction,
    down: CudaFunction,
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
    pub y: &'a mut TensorMut<bf16, 2>,
}

/// h `[rows, k+1, inter]` bf16 followed by scale `[rows]` f32; inter % 8 == 0
/// keeps the scale base 16-byte aligned.
pub(crate) fn decode_scratch_bytes(rows: usize, k: usize, inter: usize) -> u64 {
    rows as u64 * (k as u64 + 1) * inter as u64 * 2 + rows as u64 * 4
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
    ensure!((1..=8).contains(&rows), "decode rows outside [1,8]");
    ensure!(
        input.x.stride == [u64::from(hidden), 1],
        "decode x must be row-contiguous [rows, hidden]"
    );
    input.ids.validate()?;
    let top_k = input.ids.shape[1];
    ensure!(
        input.ids.shape == [rows, top_k] && input.ids.stride == [u64::from(top_k), 1],
        "decode ids must be row-contiguous [rows, top_k]"
    );
    input.assignment_gen.validate()?;
    ensure!(
        input.assignment_gen.shape == [rows, top_k]
            && input.assignment_gen.stride == [u64::from(top_k), 1],
        "decode assignment_gen must be row-contiguous [rows, top_k]"
    );
    input.active.validate()?;
    ensure!(
        input.active.shape == [rows, top_k] && input.active.stride == [u64::from(top_k), 1],
        "decode active must be row-contiguous [rows, top_k]"
    );
    input.weights.validate()?;
    ensure!(
        input.weights.shape == [rows, top_k] && input.weights.stride == [u64::from(top_k), 1],
        "decode router weights must be row-contiguous [rows, top_k]"
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
        "decode slot tables must be contiguous [experts]"
    );
    input.shared_gate.validate()?;
    input.shared_up.validate()?;
    let inter = input.shared_gate.shape[0];
    for t in [&input.shared_gate, &input.shared_up] {
        ensure!(
            t.shape == [inter, hidden] && t.stride == [u64::from(hidden), 1],
            "decode shared gate/up must be row-contiguous [inter, hidden]"
        );
    }
    input.shared_down.validate()?;
    ensure!(
        input.shared_down.shape == [hidden, inter]
            && input.shared_down.stride == [u64::from(inter), 1],
        "decode shared_down must be row-contiguous [hidden, inter]"
    );
    input.shared_expert_gate.validate()?;
    ensure!(
        input.shared_expert_gate.shape == [hidden] && input.shared_expert_gate.stride == [1],
        "decode shared expert gate must be a contiguous [hidden] vector"
    );
    let y = input.y.read();
    y.validate()?;
    ensure!(
        y.shape == [rows, hidden] && y.stride == [u64::from(hidden), 1],
        "decode output must be row-contiguous [rows, hidden]"
    );
    ensure!(
        spans_disjoint(input.x.ptr, input.x.span_bytes, y.ptr, y.span_bytes),
        "decode output overlaps x"
    );
    ensure!(
        spans_disjoint(scratch.ptr, scratch.bytes, y.ptr, y.span_bytes)
            && spans_disjoint(scratch.ptr, scratch.bytes, input.x.ptr, input.x.span_bytes),
        "decode scratch overlaps x or the output"
    );
    ensure!(
        (1..64).contains(&top_k),
        "decode top_k outside the gate_up grid limit"
    );
    ensure!(
        hidden.is_multiple_of(8),
        "decode hidden must be a multiple of 8"
    );
    ensure!(
        inter.is_multiple_of(8),
        "decode inter must be a multiple of 8"
    );
    ensure!(
        u64::from(hidden) * 2 <= 48 * 1024,
        "decode x row exceeds the gate_up kernel shared bytes"
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

impl DecodeOps {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let module = cubin::module(ctx, "moe_ffn")?;
        Ok(Self {
            gate_up: module.load_function("moe_ffn_gate_up")?,
            down: module.load_function("moe_ffn_down")?,
        })
    }

    pub fn decode(
        &self,
        stream: &Arc<CudaStream>,
        input: MoeDecode<'_>,
        scratch: &Scratch,
    ) -> Result<()> {
        let DecodeDims {
            rows,
            experts,
            top_k,
            hidden,
            inter,
        } = validate(&input, scratch)?;
        let h_bytes = u64::from(rows) * u64::from(top_k + 1) * u64::from(inter) * 2;
        let h_ptr = scratch.ptr;
        let scale_ptr = scratch.ptr + h_bytes;
        let y = input.y.read();
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
                .arg(&input.shared_expert_gate.ptr)
                .arg(&h_ptr)
                .arg(&scale_ptr)
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
                .arg(&input.ids.ptr)
                .arg(&input.active.ptr)
                .arg(&input.weights.ptr)
                .arg(&input.slots_down.ptr)
                .arg(&input.slot_valid.ptr)
                .arg(&input.slot_generation.ptr)
                .arg(&input.assignment_gen.ptr)
                .arg(&input.shared_down.ptr)
                .arg(&scale_ptr)
                .arg(&y.ptr)
                .arg(&(experts as i32))
                .arg(&(top_k as i32))
                .arg(&(hidden as i32))
                .arg(&(inter as i32))
                .launch(LaunchConfig {
                    grid_dim: (hidden.div_ceil(16), rows, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }
}
