//! Decode expert FFN contribute stage: top-k routed experts behind device
//! slot tables plus the shared expert, rows 1..8, writing per-(row, choice)
//! BF16 contributions, the ungated shared row and the per-assignment done
//! bits consumed by the shared combine op.
use crate::backend::cuda::contracts::{Scratch, Tensor, TensorMut, spans_disjoint};
use crate::backend::cuda::cubin;
#[cfg(test)]
use crate::backend::cuda::moe::combine::{CombineOps, MoeCombine};
use crate::backend::gpu::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::sync::Arc;

pub(crate) struct DecodeOps {
    gate_up: CudaFunction,
    down: CudaFunction,
    invariant: Option<(CudaFunction, CudaFunction)>,
    gate_up_w8a16: CudaFunction,
    down_w8a16: CudaFunction,
    swiglu_rounding: crate::model_spec::SwigluRounding,
    routed_weight: crate::model_spec::RoutedWeightBoundary,
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
    pub shared_expert_gate: Option<Tensor<bf16, 1>>,
    /// `[rows, top_k, hidden]` weighted BF16 contributions (dense).
    pub contributions: Tensor<bf16, 3>,
    pub shared_out: &'a mut TensorMut<bf16, 2>,
    pub done: &'a mut TensorMut<u32, 2>,
}

/// h `[rows*top_k, expert_inter]` bf16 + h_shared `[rows, shared_inter]` bf16,
/// then the `[rows]` bf16 scale of moe_shared_scale.
pub(crate) fn decode_scratch_bytes(rows: usize, k: usize, inter: usize) -> u64 {
    decode_scratch_bytes_mixed(rows, k, inter, inter)
}

pub(crate) fn decode_scratch_bytes_mixed(
    rows: usize,
    k: usize,
    inter: usize,
    shared_inter: usize,
) -> u64 {
    (rows as u64 * (k as u64 * inter as u64 + shared_inter as u64) + rows as u64) * 2
}

pub(super) struct DecodeDims {
    rows: u32,
    experts: u32,
    top_k: u32,
    hidden: u32,
    inter: u32,
    shared_inter: u32,
}

pub(super) fn validate(
    input: &MoeDecode<'_>,
    scratch: &Scratch,
    expert_inter: u32,
) -> Result<DecodeDims> {
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
    let shared_inter = input.shared_gate.shape[0];
    for t in [&input.shared_gate, &input.shared_up] {
        ensure!(
            t.shape == [shared_inter, hidden] && t.stride == [u64::from(hidden), 1],
            "decode shared gate/up must be row-contiguous [shared_inter, hidden] (shape {:?}, stride {:?}, shared_inter {shared_inter}, hidden {hidden})",
            t.shape,
            t.stride
        );
    }
    input.shared_down.validate().context("decode shared_down")?;
    ensure!(
        input.shared_down.shape == [hidden, shared_inter]
            && input.shared_down.stride == [u64::from(shared_inter), 1],
        "decode shared_down must be row-contiguous [hidden, shared_inter] (shape {:?}, stride {:?}, hidden {hidden}, shared_inter {shared_inter})",
        input.shared_down.shape,
        input.shared_down.stride
    );
    if let Some(gate) = &input.shared_expert_gate {
        gate.validate().context("decode shared_expert_gate")?;
        ensure!(
            gate.shape == [hidden] && gate.stride == [1],
            "decode shared expert gate must be a contiguous [hidden] vector (shape {:?}, stride {:?}, hidden {hidden})",
            gate.shape,
            gate.stride
        );
    }
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
        expert_inter.is_multiple_of(8),
        "decode expert inter {expert_inter} must be a multiple of 8"
    );
    ensure!(
        shared_inter.is_multiple_of(8),
        "decode shared inter {shared_inter} must be a multiple of 8"
    );
    ensure!(
        u64::from(hidden) * 2 <= 48 * 1024,
        "decode x row of {hidden} exceeds the gate_up kernel shared bytes"
    );
    scratch
        .validate(decode_scratch_bytes_mixed(
            rows as usize,
            top_k as usize,
            expert_inter as usize,
            shared_inter as usize,
        ))
        .with_context(|| {
            format!(
                "decode scratch rows={rows} top_k={top_k} expert_inter={expert_inter} shared_inter={shared_inter}"
            )
        })?;
    Ok(DecodeDims {
        rows,
        experts,
        top_k,
        hidden,
        inter: expert_inter,
        shared_inter,
    })
}

pub(super) fn validate_w8a16(
    input: &MoeDecode<'_>,
    slot: &crate::experts::catalogue::Fp8Slot,
    scratch: &Scratch,
    expert_inter: u32,
) -> Result<DecodeDims> {
    use crate::{
        experts::catalogue::{ExpertGeometry, ExpertLayout},
        model_spec::ExpertFormat,
    };
    let dims = validate(input, scratch, expert_inter)?;
    let geometry = ExpertGeometry {
        first_layer: 0,
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum LaunchMode {
    All,
    Routed,
    Shared,
}

impl DecodeOps {
    #[cfg(test)]
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        Self::new_with_boundaries(
            ctx,
            crate::model_spec::SwigluRounding::Twice,
            crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
        )
    }

    pub fn new_with_boundaries(
        ctx: &Arc<CudaContext>,
        swiglu_rounding: crate::model_spec::SwigluRounding,
        routed_weight: crate::model_spec::RoutedWeightBoundary,
    ) -> Result<Self> {
        let module = cubin::module(ctx, "moe_ffn")?;
        Ok(Self {
            gate_up: module.load_function("moe_ffn_gate_up")?,
            down: module.load_function("moe_ffn_down")?,
            invariant: None,
            gate_up_w8a16: module.load_function("moe_ffn_gate_up_w8a16")?,
            down_w8a16: module.load_function("moe_ffn_down_w8a16")?,
            swiglu_rounding,
            routed_weight,
            #[cfg(test)]
            combine: CombineOps::new(ctx)?,
        })
    }

    #[cfg(test)]
    pub(crate) fn new_invariant(ctx: &Arc<CudaContext>) -> Result<Self> {
        Self::new_invariant_with_boundaries(
            ctx,
            crate::model_spec::SwigluRounding::Twice,
            crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
        )
    }

    pub(crate) fn new_invariant_with_boundaries(
        ctx: &Arc<CudaContext>,
        swiglu_rounding: crate::model_spec::SwigluRounding,
        routed_weight: crate::model_spec::RoutedWeightBoundary,
    ) -> Result<Self> {
        #[cfg(feature = "rocm")]
        ctx.require_wmma()?;
        let module = cubin::module(ctx, "moe_ffn")?;
        Ok(Self {
            gate_up: module.load_function("moe_ffn_gate_up_lane16")?,
            down: module.load_function("moe_ffn_down_lane16")?,
            gate_up_w8a16: module.load_function("moe_ffn_gate_up_w8a16_lane16")?,
            down_w8a16: module.load_function("moe_ffn_down_w8a16_lane16")?,
            swiglu_rounding,
            routed_weight,
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
        expert_inter: u32,
    ) -> Result<()> {
        let dims = validate(input, scratch, expert_inter)?;
        self.launch(
            stream,
            input,
            scratch,
            dims,
            (&self.gate_up, &self.down),
            LaunchMode::All,
        )
    }

    pub fn contribute_routed(
        &self,
        stream: &Arc<CudaStream>,
        input: &MoeDecode<'_>,
        scratch: &Scratch,
        expert_inter: u32,
    ) -> Result<()> {
        let dims = validate(input, scratch, expert_inter)?;
        self.launch(
            stream,
            input,
            scratch,
            dims,
            (&self.gate_up, &self.down),
            LaunchMode::Routed,
        )
    }

    pub fn contribute_shared(
        &self,
        stream: &Arc<CudaStream>,
        input: &MoeDecode<'_>,
        scratch: &Scratch,
        expert_inter: u32,
    ) -> Result<()> {
        let dims = validate(input, scratch, expert_inter)?;
        self.launch(
            stream,
            input,
            scratch,
            dims,
            (&self.gate_up, &self.down),
            LaunchMode::Shared,
        )
    }

    fn launch(
        &self,
        stream: &Arc<CudaStream>,
        input: &MoeDecode<'_>,
        scratch: &Scratch,
        dims: DecodeDims,
        functions: (&CudaFunction, &CudaFunction),
        mode: LaunchMode,
    ) -> Result<()> {
        let DecodeDims {
            rows,
            experts,
            top_k,
            hidden,
            inter,
            shared_inter,
        } = dims;
        ensure!(
            self.invariant.is_some() || rows <= 8,
            "legacy decode row bound=8, requested={rows}"
        );
        ensure!(
            self.invariant.is_none()
                || (u64::from(hidden) >= 2 * u64::from(inter)
                    && hidden.is_multiple_of(256)
                    && inter.is_multiple_of(256)
                    && shared_inter.is_multiple_of(256)),
            "invariant shared expert requires hidden>=2*expert_inter and 256-aligned widths: hidden={hidden} inter={inter} shared_inter={shared_inter}"
        );
        let invariant = u32::from(self.invariant.is_some());
        let gate_inter = if invariant == 1 {
            inter
        } else {
            inter.max(shared_inter)
        };
        let h_bytes = u64::from(rows) * u64::from(top_k) * u64::from(inter) * 2;
        let h_ptr = scratch.ptr;
        let hs_ptr = scratch.ptr + h_bytes;
        if mode != LaunchMode::Shared || self.invariant.is_none() {
            let (top_k_arg, grid_z, mode_gate_inter) = match mode {
                LaunchMode::All => (top_k as i32, top_k + 1 - invariant, gate_inter),
                LaunchMode::Routed => (top_k as i32, top_k, inter),
                LaunchMode::Shared => (0i32, 1, shared_inter),
            };
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
                    .arg(&top_k_arg)
                    .arg(&(hidden as i32))
                    .arg(&(inter as i32))
                    .arg(&(shared_inter as i32))
                    .arg(&(self.swiglu_rounding as i32))
                    .launch(LaunchConfig {
                        grid_dim: (mode_gate_inter.div_ceil(32 >> invariant), rows, grid_z),
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
                    .arg(&top_k_arg)
                    .arg(&(hidden as i32))
                    .arg(&(inter as i32))
                    .arg(&(shared_inter as i32))
                    .arg(&(self.routed_weight as i32))
                    .launch(LaunchConfig {
                        grid_dim: (hidden.div_ceil(16 >> invariant), rows, grid_z),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
        }
        if mode != LaunchMode::Routed
            && let Some((skinny, silu)) = &self.invariant
        {
            let shared = input.shared_out.read().ptr;
            let up = shared + u64::from(rows) * u64::from(shared_inter) * 2;
            let gemm = |weight: u64, x: u64, y: u64, columns: u32, inner: u32| -> Result<()> {
                // SAFETY: validated dense regions cover every complete weight and activation tile.
                #[cfg(feature = "cuda")]
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
                        .arg(&0i32)
                        .launch(crate::backend::cuda::blas::Blas::skinny_launch(
                            crate::backend::setup::LinearShape {
                                rows: rows as usize,
                                output: columns as usize,
                                input: inner as usize,
                            },
                            false,
                        )?)?;
                }
                #[cfg(feature = "rocm")]
                unsafe {
                    stream
                        .launch_builder(skinny)
                        .arg(&weight)
                        .arg(&x)
                        .arg(&y)
                        .arg(&0u64)
                        .arg(&(columns as i32))
                        .arg(&(inner as i32))
                        .arg(&u64::from(inner))
                        .arg(&(rows as i32))
                        .launch(crate::backend::cuda::ops::grid(
                            columns.div_ceil(16) as usize,
                            128,
                        ))?;
                }
                Ok(())
            };
            gemm(
                input.shared_gate.ptr,
                input.x.ptr,
                shared,
                shared_inter,
                hidden,
            )?;
            gemm(input.shared_up.ptr, input.x.ptr, up, shared_inter, hidden)?;
            // SAFETY: shared_out holds two rows*inter regions; hs_ptr is disjoint scratch.
            unsafe {
                stream
                    .launch_builder(silu)
                    .arg(&shared)
                    .arg(&up)
                    .arg(&hs_ptr)
                    .arg(&((rows * shared_inter) as i32))
                    .arg(&(self.swiglu_rounding as i32))
                    .launch(crate::backend::cuda::ops::flat(
                        (rows * shared_inter) as usize,
                    ))?;
            }
            gemm(input.shared_down.ptr, hs_ptr, shared, hidden, shared_inter)?;
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
        expert_inter: u32,
    ) -> Result<()> {
        if let Some(slot) = layout {
            self.contribute_w8a16(stream, input, slot, scratch, expert_inter)?;
        } else {
            self.contribute(stream, input, scratch, expert_inter)?;
        }
        let rows = input.x.shape[0];
        let top_k = input.ids.shape[1];
        let scale_ptr = scratch.ptr
            + (u64::from(rows) * u64::from(top_k) * u64::from(expert_inter)
                + u64::from(rows) * u64::from(input.shared_gate.shape[0]))
                * 2;
        // The contiguous shared scale has shape [rows,1] and strides [1,1].
        let mut gate = None;
        let mut scale = None;
        if let Some(shared_gate) = input.shared_expert_gate {
            let mut written = unsafe {
                TensorMut::<bf16, 2>::new(scale_ptr, [rows, 1], [1, 1], u64::from(rows) * 2)?
            };
            self.combine
                .shared_scale(stream, input.x, shared_gate, &mut written)?;
            gate = Some(shared_gate);
            scale = Some(written);
        }
        let combine_input = MoeCombine {
            ids: input.ids,
            final_active: input.active,
            done: input.done.read(),
            contributions: input.contributions,
            shared_out: input.shared_out.read(),
            x: input.x,
            shared_expert_gate: gate,
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
        expert_inter: u32,
    ) -> Result<()> {
        let dims = validate_w8a16(input, slot, scratch, expert_inter)?;
        self.launch(
            stream,
            input,
            scratch,
            dims,
            (&self.gate_up_w8a16, &self.down_w8a16),
            LaunchMode::All,
        )
    }

    pub fn contribute_routed_w8a16(
        &self,
        stream: &Arc<CudaStream>,
        input: &MoeDecode<'_>,
        slot: &crate::experts::catalogue::Fp8Slot,
        scratch: &Scratch,
        expert_inter: u32,
    ) -> Result<()> {
        let dims = validate_w8a16(input, slot, scratch, expert_inter)?;
        self.launch(
            stream,
            input,
            scratch,
            dims,
            (&self.gate_up_w8a16, &self.down_w8a16),
            LaunchMode::Routed,
        )
    }
}

mod native {
    use super::super::{combine, route};
    use super::*;
    pub(crate) struct NativeMoeGeometry {
        pub hidden: u32,
        pub intermediate: u32,
        pub experts: u32,
        pub top_k: u32,
        pub scaling: f32,
        pub limit: f32,
    }

    pub(crate) struct NativeMoeWeights {
        pub router: Tensor<bf16, 2>,
        pub bias: Tensor<f32, 2>,
        pub slots: Tensor<u64, 2>,
        pub valid: Tensor<u32, 1>,
        pub shared: [super::super::super::linear::LinearWeightRef; 3],
    }

    pub(crate) struct NativeMoe {
        geometry: NativeMoeGeometry,
        weights: NativeMoeWeights,
        rows: u32,
        route: route::RouteOps,
        fp8: super::super::super::linear::Fp8Ops,
        shared: [super::super::super::linear::NativeProjection; 3],
        common: super::super::super::ops::CommonOpsRef,
        combine: combine::CombineOps,
        dispatch: CudaFunction,
        quant: CudaFunction,
        weight: CudaFunction,
        _buffers: NativeMoeBuffers,
        views: NativeMoeViews,
    }

    /// Allocation keepalive for the setup-bound `NativeMoeViews`.
    struct NativeMoeBuffers {
        _logits: crate::backend::gpu::CudaSlice<f32>,
        _ids: crate::backend::gpu::CudaSlice<i32>,
        _route_weights: crate::backend::gpu::CudaSlice<f32>,
        _codes: crate::backend::gpu::CudaSlice<u8>,
        _scales: crate::backend::gpu::CudaSlice<f32>,
        _gate: crate::backend::gpu::CudaSlice<bf16>,
        _up: crate::backend::gpu::CudaSlice<bf16>,
        _activation: crate::backend::gpu::CudaSlice<bf16>,
        _down_codes: crate::backend::gpu::CudaSlice<u8>,
        _down_scales: crate::backend::gpu::CudaSlice<f32>,
        _down: crate::backend::gpu::CudaSlice<bf16>,
        _contributions: crate::backend::gpu::CudaSlice<bf16>,
        _shared_gate: crate::backend::gpu::CudaSlice<bf16>,
        _shared_up: crate::backend::gpu::CudaSlice<bf16>,
        _shared_activation: crate::backend::gpu::CudaSlice<bf16>,
        _shared_out: crate::backend::gpu::CudaSlice<bf16>,
        _done: crate::backend::gpu::CudaSlice<u32>,
        _active: crate::backend::gpu::CudaSlice<u32>,
        _routed_meta: crate::backend::gpu::CudaSlice<i32>,
        _shared_meta: crate::backend::gpu::CudaSlice<i32>,
    }

    /// Setup-bound views of the internal buffers; the raw pointers of every
    /// `alloc_zeros` slice carry a pending-write event, so per-call views would
    /// make captured graphs wait on uncaptured work.
    struct NativeMoeViews {
        ids_ptr: u64,
        codes_ptr: u64,
        scales_ptr: u64,
        gate_ptr: u64,
        up_ptr: u64,
        logits: Tensor<f32, 2>,
        ids: Tensor<i32, 2>,
        route_weights: Tensor<f32, 2>,
        gate: Tensor<bf16, 3>,
        up: Tensor<bf16, 3>,
        activation3: Tensor<bf16, 3>,
        activation2: Tensor<bf16, 2>,
        down_codes_ptr: u64,
        down_scales_ptr: u64,
        down_ptr: u64,
        done: Tensor<u32, 2>,
        contributions: Tensor<bf16, 3>,
        active: Tensor<u32, 2>,
        routed_meta_ptr: u64,
        shared_gate: Tensor<bf16, 2>,
        shared_up: Tensor<bf16, 2>,
        shared_activation2: Tensor<bf16, 2>,
        shared_out: Tensor<bf16, 2>,
        shared_meta_ptr: u64,
    }

    impl NativeMoe {
        pub(crate) fn new(
            ctx: &Arc<CudaContext>,
            stream: &Arc<CudaStream>,
            geometry: NativeMoeGeometry,
            weights: NativeMoeWeights,
            rows: u32,
        ) -> Result<Self> {
            use super::super::super::linear::{Fp8Ops, NativeProjection};
            let g = &geometry;
            ensure!(
                rows > 0
                    && rows <= 16
                    && g.top_k > 0
                    && g.top_k <= 32
                    && g.experts >= g.top_k
                    && g.experts <= 1024
                    && g.hidden > 0
                    && g.intermediate > 0
                    && g.hidden.is_multiple_of(128)
                    && g.intermediate.is_multiple_of(128),
                "native MoE geometry is outside admitted extents"
            );
            ensure!(
                g.scaling.is_finite() && g.scaling > 0.0 && g.limit.is_finite() && g.limit > 0.0,
                "native MoE scaling and clamp must be positive and finite"
            );
            for (shape, stride, expected) in [
                (
                    weights.router.shape,
                    weights.router.stride,
                    [g.experts, g.hidden],
                ),
                (weights.bias.shape, weights.bias.stride, [rows, g.experts]),
                (weights.slots.shape, weights.slots.stride, [g.experts, 6]),
            ] {
                ensure!(
                    shape == expected && stride == [u64::from(expected[1]), 1],
                    "native MoE weight/table layout mismatch"
                );
            }
            weights.router.validate()?;
            weights.bias.validate()?;
            weights.slots.validate()?;
            weights.valid.validate()?;
            ensure!(
                weights.valid.shape == [g.experts] && weights.valid.stride == [1],
                "native MoE residency layout mismatch"
            );
            for (index, weight) in weights.shared.iter().enumerate() {
                let super::super::super::linear::LinearWeightRef::Fp8 { codes, .. } = weight else {
                    anyhow::bail!("native shared expert requires FP8 weights");
                };
                let expected = if index == 2 {
                    [g.hidden, g.intermediate]
                } else {
                    [g.intermediate, g.hidden]
                };
                ensure!(
                    codes.shape == expected,
                    "native shared projection extent mismatch"
                );
            }
            let fp8 = Fp8Ops::new(ctx)?;
            let shared = weights
                .shared
                .map(|w| NativeProjection::new(&fp8, stream, w, rows));
            let [gate, up, down] = shared;
            let shared = [gate?, up?, down?];
            let m = rows as usize;
            let assignments = m * g.top_k as usize;
            let h = g.hidden as usize;
            let i = g.intermediate as usize;
            let module = cubin::module(ctx, "moe_ffn")
                .map_err(|e| e.context("NativeMoe::new: loading moe_ffn cubin"))?;
            let logits_buf = stream.alloc_zeros(m * g.experts as usize)?;
            let ids_buf = stream.alloc_zeros(assignments)?;
            let route_weights_buf = stream.alloc_zeros(assignments)?;
            let codes_buf = stream.alloc_zeros(m * h)?;
            let scales_buf = stream.alloc_zeros(m * h / 128)?;
            let gate_buf = stream.alloc_zeros(assignments * i)?;
            let up_buf = stream.alloc_zeros(assignments * i)?;
            let activation_buf = stream.alloc_zeros(assignments * i)?;
            let down_codes_buf = stream.alloc_zeros(assignments * i)?;
            let down_scales_buf = stream.alloc_zeros(assignments * i / 128)?;
            let down_buf = stream.alloc_zeros(assignments * h)?;
            let contributions_buf = stream.alloc_zeros(assignments * h)?;
            let shared_gate_buf = stream.alloc_zeros(m * i)?;
            let shared_up_buf = stream.alloc_zeros(m * i)?;
            let shared_activation_buf = stream.alloc_zeros(m * i)?;
            let shared_out_buf = stream.alloc_zeros(m * h)?;
            let done_buf = stream.alloc_zeros(assignments)?;
            let active_buf = stream.clone_htod(&vec![1u32; assignments])?;
            let routed_meta_buf = stream.clone_htod(&[
                0,
                assignments as i32,
                assignments as i32,
                assignments as i32,
                1,
                0,
                0,
                0,
            ])?;
            let shared_meta_buf =
                stream.clone_htod(&[0, rows as i32, rows as i32, rows as i32, 1, 0, 0, 0])?;
            let views = {
                use super::super::super::contracts::{dense_layout, view};
                use crate::backend::gpu::DevicePtr;
                let (ids_stride, _) = dense_layout([rows, g.top_k])?;
                let logits_view = view(
                    &logits_buf,
                    stream,
                    0,
                    [rows, g.experts],
                    [u64::from(g.experts), 1],
                )?;
                let ids_view = view(&ids_buf, stream, 0, [rows, g.top_k], ids_stride)?;
                let route_weights_view =
                    view(&route_weights_buf, stream, 0, [rows, g.top_k], ids_stride)?;
                let gate_view = view(
                    &gate_buf,
                    stream,
                    0,
                    [rows * g.top_k, 1, g.intermediate],
                    [u64::from(g.intermediate), u64::from(g.intermediate), 1],
                )?;
                let up_view = view(
                    &up_buf,
                    stream,
                    0,
                    [rows * g.top_k, 1, g.intermediate],
                    [u64::from(g.intermediate), u64::from(g.intermediate), 1],
                )?;
                let activation3_view = view(
                    &activation_buf,
                    stream,
                    0,
                    [rows * g.top_k, 1, g.intermediate],
                    [u64::from(g.intermediate), u64::from(g.intermediate), 1],
                )?;
                let activation2_view = view(
                    &activation_buf,
                    stream,
                    0,
                    [rows * g.top_k, g.intermediate],
                    [u64::from(g.intermediate), 1],
                )?;
                let done_view = view(&done_buf, stream, 0, [rows, g.top_k], ids_stride)?;
                let contributions_view = view(
                    &contributions_buf,
                    stream,
                    0,
                    [rows, g.top_k, g.hidden],
                    [
                        u64::from(g.top_k) * u64::from(g.hidden),
                        u64::from(g.hidden),
                        1,
                    ],
                )?;
                let active_view = view(&active_buf, stream, 0, [rows, g.top_k], ids_stride)?;
                let shared_gate_view = view(
                    &shared_gate_buf,
                    stream,
                    0,
                    [rows, g.intermediate],
                    [u64::from(g.intermediate), 1],
                )?;
                let shared_up_view = view(
                    &shared_up_buf,
                    stream,
                    0,
                    [rows, g.intermediate],
                    [u64::from(g.intermediate), 1],
                )?;
                let shared_activation2_view = view(
                    &shared_activation_buf,
                    stream,
                    0,
                    [rows, g.intermediate],
                    [u64::from(g.intermediate), 1],
                )?;
                let shared_out_view = view(
                    &shared_out_buf,
                    stream,
                    0,
                    [rows, g.hidden],
                    [u64::from(g.hidden), 1],
                )?;
                let (codes_ptr, scales_ptr, gate_ptr, up_ptr, ids_ptr) = (
                    codes_buf.device_ptr(stream).0,
                    scales_buf.device_ptr(stream).0,
                    gate_buf.device_ptr(stream).0,
                    up_buf.device_ptr(stream).0,
                    ids_buf.device_ptr(stream).0,
                );
                NativeMoeViews {
                    ids_ptr,
                    codes_ptr,
                    scales_ptr,
                    gate_ptr,
                    up_ptr,
                    logits: logits_view,
                    ids: ids_view,
                    route_weights: route_weights_view,
                    gate: gate_view,
                    up: up_view,
                    activation3: activation3_view,
                    activation2: activation2_view,
                    down_codes_ptr: down_codes_buf.device_ptr(stream).0,
                    down_scales_ptr: down_scales_buf.device_ptr(stream).0,
                    down_ptr: down_buf.device_ptr(stream).0,
                    done: done_view,
                    contributions: contributions_view,
                    active: active_view,
                    routed_meta_ptr: routed_meta_buf.device_ptr(stream).0,
                    shared_gate: shared_gate_view,
                    shared_up: shared_up_view,
                    shared_activation2: shared_activation2_view,
                    shared_out: shared_out_view,
                    shared_meta_ptr: shared_meta_buf.device_ptr(stream).0,
                }
            };
            let route = route::RouteOps::new(ctx, stream, g.experts, m)?;
            #[cfg(feature = "cuda")]
            let route = {
                let mut route = route;
                route.prepare_logits_f32(rows, g.hidden)?;
                route
            };
            Ok(Self {
                rows,
                route,
                fp8,
                shared,
                common: super::super::super::ops::CommonOpsRef::new_with_swiglu_rounding(
                    ctx,
                    crate::model_spec::SwigluRounding::ClampedTwice,
                    g.limit,
                )?,
                combine: combine::CombineOps::new(ctx)?,
                dispatch: module
                    .load_function("moe_ffn_dispatch_e4m3_f32_128")
                    .map_err(|e| anyhow::anyhow!("moe_ffn: moe_ffn_dispatch_e4m3_f32_128: {e}"))?,
                quant: cubin::module(ctx, "fp8_linear")?.load_function("fp8_quant_f32_128")?,
                weight: module
                    .load_function("moe_ffn_weight_f32")
                    .map_err(|e| anyhow::anyhow!("moe_ffn: moe_ffn_weight_f32: {e}"))?,
                _buffers: NativeMoeBuffers {
                    _logits: logits_buf,
                    _ids: ids_buf,
                    _route_weights: route_weights_buf,
                    _codes: codes_buf,
                    _scales: scales_buf,
                    _gate: gate_buf,
                    _up: up_buf,
                    _activation: activation_buf,
                    _down_codes: down_codes_buf,
                    _down_scales: down_scales_buf,
                    _down: down_buf,
                    _contributions: contributions_buf,
                    _shared_gate: shared_gate_buf,
                    _shared_up: shared_up_buf,
                    _shared_activation: shared_activation_buf,
                    _shared_out: shared_out_buf,
                    _done: done_buf,
                    _active: active_buf,
                    _routed_meta: routed_meta_buf,
                    _shared_meta: shared_meta_buf,
                },
                views,
                geometry,
                weights,
            })
        }

        fn project(&self, stream: &Arc<CudaStream>, projection: i32) -> Result<()> {
            let g = &self.geometry;
            let (n, k) = if projection == 2 {
                (g.hidden, g.intermediate)
            } else {
                (g.intermediate, g.hidden)
            };
            unsafe {
                stream
                    .launch_builder(&self.dispatch)
                    .arg(&if projection == 2 {
                        self.views.down_codes_ptr
                    } else {
                        self.views.codes_ptr
                    })
                    .arg(&if projection == 2 {
                        self.views.down_scales_ptr
                    } else {
                        self.views.scales_ptr
                    })
                    .arg(&self.weights.slots.ptr)
                    .arg(&self.weights.valid.ptr)
                    .arg(&self.views.ids_ptr)
                    .arg(&if projection == 2 {
                        self.views.down_ptr
                    } else if projection == 0 {
                        self.views.gate_ptr
                    } else {
                        self.views.up_ptr
                    })
                    .arg(&self.rows)
                    .arg(&g.top_k)
                    .arg(&g.experts)
                    .arg(&n)
                    .arg(&k)
                    .arg(&projection)
                    .launch(LaunchConfig {
                        grid_dim: (n.div_ceil(8), self.rows, g.top_k),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            Ok(())
        }

        fn quantize(
            &self,
            stream: &Arc<CudaStream>,
            input: Tensor<bf16, 2>,
            codes: u64,
            scales: u64,
        ) -> Result<()> {
            let [m, k] = input.shape;
            unsafe {
                stream
                    .launch_builder(&self.quant)
                    .arg(&input.ptr)
                    .arg(&codes)
                    .arg(&scales)
                    .arg(&m)
                    .arg(&k)
                    .launch(super::super::super::ops::grid(
                        (m * (k / 128)).div_ceil(4) as usize,
                        128,
                    ))?;
            }
            Ok(())
        }

        pub(crate) fn route(
            &self,
            stream: &Arc<CudaStream>,
            x: Tensor<bf16, 2>,
        ) -> Result<(Vec<i32>, Vec<f32>)> {
            let g = &self.geometry;
            let logits = self.views.logits;
            let ids = self.views.ids;
            let weights = self.views.route_weights;
            #[cfg(feature = "cuda")]
            self.route
                .logits_f32_lt(stream, x, self.weights.router, &mut TensorMut(logits))?;
            #[cfg(feature = "rocm")]
            self.route
                .logits_f32(stream, x, self.weights.router, &mut TensorMut(logits))?;
            self.route.topk_noaux_f32(
                logits,
                Some(self.weights.bias),
                &mut TensorMut(ids),
                &mut TensorMut(weights),
                crate::model_spec::RouterScore::Sigmoid,
                g.scaling,
            )?;

            Ok((
                stream.clone_dtoh(&self._buffers._ids)?,
                stream.clone_dtoh(&self._buffers._route_weights)?,
            ))
        }

        pub(crate) fn run(
            &self,
            stream: &Arc<CudaStream>,
            x: Tensor<bf16, 2>,
            output: &TensorMut<bf16, 2>,
        ) -> Result<()> {
            use super::super::super::contracts::{RowDomain, SeqRef, no_overlap};
            let g = &self.geometry;
            let (m, h, k) = (self.rows, g.hidden, g.top_k);
            x.validate()?;
            output.0.validate()?;
            ensure!(
                x.shape == [m, h]
                    && output.0.shape == [m, h]
                    && x.stride == [u64::from(h), 1]
                    && output.0.stride == x.stride,
                "native MoE input/output layout mismatch"
            );
            no_overlap(&x, &output.0, "native MoE input", "output")?;
            no_overlap(&self.weights.router, &output.0, "router", "output")?;
            no_overlap(&self.weights.bias, &output.0, "bias", "output")?;
            no_overlap(&self.weights.slots, &output.0, "slots", "output")?;
            no_overlap(&self.weights.valid, &output.0, "valid", "output")?;
            for weight in &self.weights.shared {
                let super::super::super::linear::LinearWeightRef::Fp8 { codes, scale, .. } = weight
                else {
                    unreachable!()
                };
                no_overlap(codes, &output.0, "shared codes", "output")?;
                let super::super::super::linear::Fp8ScaleRef::F32(scale) = scale else {
                    unreachable!()
                };
                no_overlap(scale, &output.0, "shared scales", "output")?;
            }

            let ids = self.views.ids;
            let weights = self.views.route_weights;
            self.quantize(stream, x, self.views.codes_ptr, self.views.scales_ptr)?;
            self.project(stream, 0)?;
            self.project(stream, 1)?;
            let routed = RowDomain {
                meta: SeqRef {
                    ptr: self.views.routed_meta_ptr,
                },
                row_base: 0,
                row_capacity: m * k,
            };
            self.common.gate.swiglu(
                stream,
                routed,
                self.views.gate,
                self.views.up,
                &mut TensorMut(self.views.activation3),
            )?;
            self.quantize(
                stream,
                self.views.activation2,
                self.views.down_codes_ptr,
                self.views.down_scales_ptr,
            )?;
            self.project(stream, 2)?;
            unsafe {
                stream
                    .launch_builder(&self.weight)
                    .arg(&self.views.down_ptr)
                    .arg(&weights.ptr)
                    .arg(&self.views.contributions.ptr)
                    .arg(&self.views.done.ptr)
                    .arg(&(m * k))
                    .arg(&h)
                    .launch(LaunchConfig {
                        grid_dim: (h.div_ceil(256), m * k, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            super::super::w8a8::MoeSharedW8A8::run_block(
                stream,
                super::super::w8a8::W8A8BlockCall {
                    projections: &self.shared,
                    fp8: &self.fp8,
                    common: &self.common,
                    rows: RowDomain {
                        meta: SeqRef {
                            ptr: self.views.shared_meta_ptr,
                        },
                        row_base: 0,
                        row_capacity: m,
                    },
                    x,
                    gate: self.views.shared_gate,
                    up: self.views.shared_up,
                    activation: self.views.shared_activation2,
                    out: self.views.shared_out,
                },
            )?;
            self.combine.combine_bf16(
                stream,
                &combine::MoeCombine {
                    ids,
                    final_active: self.views.active,
                    done: self.views.done,
                    contributions: self.views.contributions,
                    shared_out: self.views.shared_out,
                    x,
                    shared_expert_gate: None,
                    scale: None,
                    y: TensorMut(output.0),
                },
            )
        }
    }
}
pub(crate) use native::{NativeMoe, NativeMoeGeometry, NativeMoeWeights};
