//! HD256 full attention for Qwen3.8-27B (ABI v1 §4).
//!
//! Shape is fixed by [`FullAttentionSpec::q38`]: 24 Q heads, 4 KV heads, GQA 6,
//! head_dim 256, scale 1/16. The CUDA side is `kernels/tcmqa_d256.cu`: split
//! Q/gate, partial RoPE on Q, RoPE+scatter on K, scatter on V, then the
//! narrow-tile (64-row, KT=32, 128-thread) phase 1 and its phase-2 merge.
//!
//! `run` is allocation-free and same-stream, in the ABI §4.2 order: split
//! Q/gate -> common Q/K norm -> partial RoPE and KV scatter -> segmented
//! attention. The Q/K norms are the common [`NormOps`]; this module does not
//! duplicate them, and it does not apply the output sigmoid gate (the caller's
//! `GateOps::sigmoid_mul` produces `gated_out`, and the o_projection consumes
//! the flattened 6144 width).
//!
//! Prefill rows are processed in segments of at most `SEG_MAX` rows. The segment
//! count is `ceil(max_rows / segment_rows)` and depends only on the plan, never
//! on the device `valid_rows`, so a captured Graph replays an identical launch
//! sequence as the position advances. Every kernel reads `prefix` and
//! `valid_rows` from the device `SequenceMeta`, and a segment with no live rows
//! writes nothing.

use super::contracts::{RowDomain, Scratch, SeqRef, Tensor, TensorMut};
use super::cubin;
use super::ops::{CommonOpsRef, RmsMode, RmsOptions};
use crate::op_contract::{AttentionLimits, AttentionWorkspace, FullAttentionSpec, WorkspaceReq};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use half::bf16;
use std::mem::size_of;
use std::sync::Arc;

pub(crate) const HQ: u32 = 24;
pub(crate) const HKV: u32 = 4;
pub(crate) const GQA: u32 = HQ / HKV;
pub(crate) const D: u32 = 256;

/// Narrow KT of the phase-1 kernel; `key_chunk` must be a positive multiple.
const KT: u32 = 32;
/// Maximum query rows in one phase-1/phase-2 segment (the 64-row pair tile).
const SEG_MAX: u32 = 64;
const PHASE1_THREADS: u32 = 128;
const PHASE2_THREADS: u32 = D;
const PREP_THREADS: u32 = 256;
/// `(MT + 2*KT) * (D + 8) * 2 + 64*8` from `kernels/tcmqa_d256.cu`.
const PHASE1_SMEM: u32 = 68096;
const MASK_LEN: u32 = 64;
const RMS_EPS: f32 = 1e-6;

/// The ABI §4.1 views. Every field carries an explicit stride and span; the
/// Where the RoPE coordinates come from. `Linear` uses the physical position
/// `prefix + row` from the device SequenceMeta — the only source before the
/// vision path existed. `Axes` reads each query row's (T, H, W) coordinate from
/// `table` (`[max_rows, 3]` i32, row-major): a multimodal prompt rotates a row
/// by its own axis coordinate while the KV cache slot and the causal key count
/// stay the physical position. With T = H = W the two branches must agree
/// bitwise; that degenerate case is the three-axis gate.
#[derive(Clone, Copy, Debug)]
pub(crate) enum PositionBindings {
    Linear,
    Axes(Tensor<i32, 2>),
}

/// binder validates them against the registered layout rather than assuming a
/// row width from `hidden`.
#[derive(Debug)]
pub(crate) struct AttentionBindings {
    /// Projection `[max_rows, 24, 512]`: Q at head offset 0, gate at +256.
    pub q_gate: Tensor<bf16, 3>,
    /// Projection K and V, `[max_rows, 4, 256]`, read-only.
    pub k: Tensor<bf16, 3>,
    pub v: Tensor<bf16, 3>,
    /// Independent Q and gate `[max_rows, 24, 256]`.
    pub q: TensorMut<bf16, 3>,
    pub gate: TensorMut<bf16, 3>,
    /// Head-major caches `[4, capacity, 256]`, two separate buffers.
    pub k_cache: TensorMut<bf16, 3>,
    pub v_cache: TensorMut<bf16, 3>,
    /// Ungated attention output `[max_rows, 24, 256]`.
    pub core_out: TensorMut<bf16, 3>,
    /// Zero-centred Q/K norm weights, `[256]`.
    pub q_norm: Tensor<bf16, 1>,
    pub k_norm: Tensor<bf16, 1>,
    /// Caller-prebuilt causal triangle, `u64[64]`, `mask[63] == u64::MAX`.
    pub causal_mask: Tensor<u64, 1>,
    /// Coordinate source for the RoPE stage.
    pub positions: PositionBindings,
}

struct Kernels {
    qg_split: CudaFunction,
    rope_q: CudaFunction,
    rope_k_scatter: CudaFunction,
    v_scatter: CudaFunction,
    phase1: CudaFunction,
    phase2: CudaFunction,
}

pub(crate) struct FullAttentionPlan {
    kernels: Kernels,
    bindings: AttentionBindings,
    common: CommonOpsRef,
    limits: AttentionLimits,
    ws: AttentionWorkspace,
    segment_rows: u32,
    max_parts: u32,
    segments: u32,
    split_cfg: LaunchConfig,
    scatter_cfg: LaunchConfig,
    phase1_cfg: LaunchConfig,
    phase2_cfg: LaunchConfig,
}

pub(crate) struct FullAttentionOps;

fn check_limits(limits: AttentionLimits) -> Result<()> {
    ensure!(limits.causal == 1, "full attention requires causal=1");
    ensure!(
        limits.capacity > 0 && limits.capacity <= i32::MAX as u32,
        "invalid attention capacity"
    );
    ensure!(
        limits.max_rows > 0 && limits.max_rows <= limits.capacity,
        "max_rows must be positive and within capacity"
    );
    ensure!(
        limits.key_chunk >= KT && limits.key_chunk.is_multiple_of(KT),
        "key_chunk must be a positive multiple of the narrow KT=32"
    );
    ensure!(
        limits.key_chunk <= limits.capacity,
        "key_chunk exceeds capacity"
    );
    // The KV cache head stride reaches the kernels as a 32-bit element count,
    // and the key-partition start is computed in 32-bit.
    ensure!(
        u64::from(limits.capacity) * u64::from(D) <= i32::MAX as u64,
        "KV cache head stride exceeds the kernel's 32-bit stride"
    );
    Ok(())
}

fn want<T: Copy, const R: usize>(view: &Tensor<T, R>, shape: [u32; R], name: &str) -> Result<()> {
    view.validate()?;
    ensure!(
        view.shape == shape,
        "{name} must be {shape:?}, got {:?}",
        view.shape
    );
    Ok(())
}

fn check_bindings(b: &AttentionBindings, limits: AttentionLimits) -> Result<()> {
    let rows = limits.max_rows;
    let cap = limits.capacity;
    want(&b.q_gate, [rows, HQ, 2 * D], "q_gate")?;
    ensure!(
        b.q_gate.stride == [u64::from(HQ) * 2 * u64::from(D), u64::from(2 * D), 1],
        "q_gate must be the packed [M,24,512] projection"
    );
    for (view, name) in [(&b.q, "q"), (&b.gate, "gate"), (&b.core_out, "core_out")] {
        let t = view.read();
        want(&t, [rows, HQ, D], name)?;
        ensure!(
            t.stride == [u64::from(HQ) * u64::from(D), u64::from(D), 1],
            "{name} must be a contiguous [M,24,256] view"
        );
    }
    for (view, name) in [(&b.k, "k"), (&b.v, "v")] {
        want(view, [rows, HKV, D], name)?;
        ensure!(
            view.stride == [u64::from(HKV) * u64::from(D), u64::from(D), 1],
            "{name} must be a contiguous [M,4,256] projection"
        );
    }
    for (view, name) in [(&b.k_cache, "k_cache"), (&b.v_cache, "v_cache")] {
        let t = view.read();
        want(&t, [HKV, cap, D], name)?;
        ensure!(
            t.stride == [u64::from(cap) * u64::from(D), u64::from(D), 1],
            "{name} must be head-major [4,capacity,256]"
        );
    }
    for (view, name) in [(&b.q_norm, "q_norm"), (&b.k_norm, "k_norm")] {
        want(view, [D], name)?;
        ensure!(view.stride == [1], "{name} must be contiguous");
    }
    want(&b.causal_mask, [MASK_LEN], "causal_mask")?;
    ensure!(
        b.causal_mask.stride == [1],
        "causal_mask must be contiguous"
    );
    if let PositionBindings::Axes(table) = &b.positions {
        want(table, [rows, 3], "positions")?;
        ensure!(
            table.stride == [3, 1],
            "positions must be a dense [max_rows, 3] (T, H, W) table"
        );
    }
    Ok(())
}

impl FullAttentionOps {
    /// CPU-only checked sizing: reads no GPU state and allocates nothing.
    pub(crate) fn workspace(
        spec: &FullAttentionSpec,
        limits: AttentionLimits,
    ) -> Result<WorkspaceReq> {
        spec.validate()?;
        check_limits(limits)?;
        let ws = AttentionWorkspace::q38(limits.max_rows, limits.capacity, limits.key_chunk)?;
        Ok(WorkspaceReq {
            bytes: ws.bytes,
            alignment: ws.alignment,
        })
    }

    /// Setup binder: loads its own kernels, validates the registered layout,
    /// opts in the phase-1 shared memory and binds stable addresses.
    pub(crate) fn prepare(
        ctx: &Arc<CudaContext>,
        spec: &FullAttentionSpec,
        limits: AttentionLimits,
        bindings: AttentionBindings,
        common: CommonOpsRef,
    ) -> Result<FullAttentionPlan> {
        spec.validate()?;
        check_limits(limits)?;
        check_bindings(&bindings, limits)?;
        let ws = AttentionWorkspace::q38(limits.max_rows, limits.capacity, limits.key_chunk)?;
        let segment_rows = u32::try_from(ws.segment_rows)?;
        let max_parts = u32::try_from(ws.max_parts)?;
        ensure!(segment_rows <= SEG_MAX, "segment rows exceed the pair tile");

        // Phase 2 indexes `sm[0 .. 2*Pactive + blockDim.x)`, and Pactive <= max_parts,
        // so the launch is sized for max_parts. Launching it with no dynamic shared
        // memory faults on the first write to the reduction slot.
        let phase2_smem = u32::try_from(
            (2 * u64::from(max_parts) + u64::from(PHASE2_THREADS))
                .checked_mul(4)
                .context("phase-2 shared memory overflow")?,
        )
        .context("phase-2 shared memory exceeds u32")?;
        use cudarc::driver::sys::CUdevice_attribute as Attribute;
        let optin =
            ctx.attribute(Attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)? as u64;
        for (name, bytes) in [
            ("phase 1", u64::from(PHASE1_SMEM)),
            ("phase 2", u64::from(phase2_smem)),
        ] {
            ensure!(
                bytes <= optin,
                "{name} dynamic shared memory {bytes} B exceeds the device opt-in limit {optin} B"
            );
        }

        let module = cubin::module(ctx, "tcmqa_d256")?;
        let phase1 = module.load_function("tcmqa_d256_phase1")?;
        let phase2 = module.load_function("tcmqa_d256_phase2")?;
        for (function, bytes) in [(&phase1, PHASE1_SMEM), (&phase2, phase2_smem)] {
            if bytes > 48 * 1024 {
                function.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    bytes as i32,
                )?;
            }
        }
        // The coordinate source picks the RoPE entries; the plan stores which
        // one it bound, so `run` never re-decides.
        let (rope_q_entry, rope_k_entry) = match &bindings.positions {
            PositionBindings::Linear => ("tcmqa_d256_rope_q", "tcmqa_d256_rope_k_scatter"),
            PositionBindings::Axes(_) => {
                ("tcmqa_d256_rope_q_axes", "tcmqa_d256_rope_k_scatter_axes")
            }
        };
        let kernels = Kernels {
            qg_split: module.load_function("tcmqa_d256_qg_split")?,
            rope_q: module.load_function(rope_q_entry)?,
            rope_k_scatter: module.load_function(rope_k_entry)?,
            v_scatter: module.load_function("tcmqa_d256_v_scatter")?,
            phase1,
            phase2,
        };

        let rows = limits.max_rows;
        let flat = |elements: u32| -> Result<LaunchConfig> {
            Ok(LaunchConfig {
                grid_dim: (elements.div_ceil(PREP_THREADS), 1, 1),
                block_dim: (PREP_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        };
        // Grids are sized for max_rows; the device valid_rows bounds the work.
        let split_cfg = flat(
            rows.checked_mul(HQ * D)
                .context("Q/gate split grid overflow")?,
        )?;
        let scatter_cfg = flat(
            rows.checked_mul(HKV * D)
                .context("KV scatter grid overflow")?,
        )?;
        let phase1_cfg = LaunchConfig {
            grid_dim: (max_parts, HKV, (GQA * segment_rows).div_ceil(SEG_MAX)),
            block_dim: (PHASE1_THREADS, 1, 1),
            shared_mem_bytes: PHASE1_SMEM,
        };
        let phase2_cfg = LaunchConfig {
            grid_dim: (HQ * segment_rows, 1, 1),
            block_dim: (PHASE2_THREADS, 1, 1),
            shared_mem_bytes: phase2_smem,
        };
        Ok(FullAttentionPlan {
            kernels,
            bindings,
            common,
            limits,
            ws,
            segment_rows,
            max_parts,
            segments: rows.div_ceil(segment_rows),
            split_cfg,
            scatter_cfg,
            phase1_cfg,
            phase2_cfg,
        })
    }

    /// Scratch bytes the preprocess chain alone needs: the out-of-place K-norm
    /// temp, which sits at the end of the full workspace at the same offset, so
    /// a caller that runs no TCMQA phases can size an arena from this.
    pub(crate) fn preprocess_workspace(limits: AttentionLimits) -> Result<WorkspaceReq> {
        check_limits(limits)?;
        let ws = AttentionWorkspace::q38(limits.max_rows, limits.capacity, limits.key_chunk)?;
        Ok(WorkspaceReq {
            bytes: ws.bytes - ws.k_norm,
            alignment: ws.alignment,
        })
    }

    /// Allocation-free, same-stream preprocess over the device `valid_rows`
    /// rows: Q/gate split, the common Q and K zero-centred norms, partial RoPE
    /// on Q, and RoPE + scatter of K and V into the head-major caches. On return
    /// `q`, `gate` and both caches hold this chunk, and `run` can continue with
    /// the TCMQA phases or another attention. `scratch` holds only the K-norm
    /// temp (see `preprocess_workspace`).
    pub(crate) fn preprocess(
        plan: &FullAttentionPlan,
        stream: &Arc<CudaStream>,
        meta: SeqRef,
        scratch: Scratch,
    ) -> Result<()> {
        meta.validate()?;
        scratch.validate(plan.ws.bytes - plan.ws.k_norm)?;
        let b = &plan.bindings;
        let limits = plan.limits;
        let rows = limits.max_rows;
        let q_tok = b.q.0.stride[0] as i32;
        let k_tok = b.k.stride[0] as i32;
        let v_tok = b.v.stride[0] as i32;
        let head_stride = (u64::from(limits.capacity) * u64::from(D)) as i32;
        let pos_stride = D as i32;
        let elem = size_of::<bf16>() as u64;
        let k_tmp = scratch.ptr;
        let k_tmp_span = u64::from(rows) * u64::from(HKV) * u64::from(D) * elem;

        // 1. split the packed Q/gate projection into independent Q and gate views
        unsafe {
            stream
                .launch_builder(&plan.kernels.qg_split)
                .arg(&b.q_gate.ptr)
                .arg(&b.q.0.ptr)
                .arg(&b.gate.0.ptr)
                .arg(&meta.ptr)
                .arg(&(b.q_gate.stride[0] as i32))
                .arg(&q_tok)
                .arg(&(b.gate.0.stride[0] as i32))
                .launch(plan.split_cfg)?;
        }

        // 2. common Q norm in place, common K norm out of place into the
        //    workspace temp. RowDomain carries the device meta, so the common
        //    kernels bound their work by valid_rows rather than by max_rows.
        let domain = RowDomain {
            meta,
            row_base: 0,
            row_capacity: rows,
        };
        let mut q_out = TensorMut(b.q.read());
        plan.common.norm.rms_with(
            stream,
            domain,
            b.q.read(),
            b.q_norm,
            &mut q_out,
            RmsOptions {
                mode: RmsMode::ZeroCentered,
                eps: RMS_EPS,
            },
        )?;
        let mut k_out = unsafe {
            TensorMut::new(
                k_tmp,
                [rows, HKV, D],
                [u64::from(HKV) * u64::from(D), u64::from(D), 1],
                k_tmp_span,
            )?
        };
        plan.common.norm.rms_with(
            stream,
            domain,
            b.k,
            b.k_norm,
            &mut k_out,
            RmsOptions {
                mode: RmsMode::ZeroCentered,
                eps: RMS_EPS,
            },
        )?;

        // 3. partial RoPE on Q; RoPE + scatter on the normalised K; scatter V.
        //    The linear branch rotates by the physical position; the axes branch
        //    reads (T, H, W) per row. Both keep the cache slot at `prefix + r`.
        match &b.positions {
            PositionBindings::Linear => unsafe {
                stream
                    .launch_builder(&plan.kernels.rope_q)
                    .arg(&b.q.0.ptr)
                    .arg(&meta.ptr)
                    .arg(&q_tok)
                    .launch(plan.scatter_cfg)?;
                stream
                    .launch_builder(&plan.kernels.rope_k_scatter)
                    .arg(&k_tmp)
                    .arg(&b.k_cache.0.ptr)
                    .arg(&meta.ptr)
                    .arg(&k_tok)
                    .arg(&head_stride)
                    .arg(&pos_stride)
                    .launch(plan.scatter_cfg)?;
            },
            PositionBindings::Axes(table) => unsafe {
                stream
                    .launch_builder(&plan.kernels.rope_q)
                    .arg(&b.q.0.ptr)
                    .arg(&meta.ptr)
                    .arg(&table.ptr)
                    .arg(&q_tok)
                    .launch(plan.scatter_cfg)?;
                stream
                    .launch_builder(&plan.kernels.rope_k_scatter)
                    .arg(&k_tmp)
                    .arg(&b.k_cache.0.ptr)
                    .arg(&meta.ptr)
                    .arg(&table.ptr)
                    .arg(&k_tok)
                    .arg(&head_stride)
                    .arg(&pos_stride)
                    .launch(plan.scatter_cfg)?;
            },
        }
        unsafe {
            stream
                .launch_builder(&plan.kernels.v_scatter)
                .arg(&b.v.ptr)
                .arg(&b.v_cache.0.ptr)
                .arg(&meta.ptr)
                .arg(&v_tok)
                .arg(&head_stride)
                .arg(&pos_stride)
                .launch(plan.scatter_cfg)?;
        }

        Ok(())
    }

    /// Allocation-free, same-stream forward over the device `valid_rows` rows:
    /// the preprocess chain, then the segmented TCMQA attention.
    pub(crate) fn run(
        plan: &FullAttentionPlan,
        stream: &Arc<CudaStream>,
        meta: SeqRef,
        scratch: Scratch,
    ) -> Result<()> {
        meta.validate()?;
        scratch.validate(plan.ws.bytes)?;
        let b = &plan.bindings;
        let limits = plan.limits;
        let seg = plan.segment_rows;
        let key_chunk = limits.key_chunk as i32;
        let pmax = plan.max_parts as i32;
        let seg_rows = seg as i32;
        let q_tok = b.q.0.stride[0] as i32;
        let head_stride = (u64::from(limits.capacity) * u64::from(D)) as i32;
        let pos_stride = D as i32;
        let out_stride = b.core_out.0.stride[0] as i32;
        let elem = size_of::<bf16>() as u64;
        let p_max = scratch.ptr + plan.ws.p_max;
        let p_sum = scratch.ptr + plan.ws.p_sum;
        let p_out = scratch.ptr + plan.ws.p_out;
        // The K-norm temp sits at the end of the op's workspace; preprocess gets
        // a scratch whose base is that offset.
        Self::preprocess(
            plan,
            stream,
            meta,
            Scratch {
                ptr: scratch.ptr + plan.ws.k_norm,
                bytes: scratch.bytes - plan.ws.k_norm,
            },
        )?;

        // 4. segmented attention. Segments run in order on one stream and reuse
        //    the single partials region; the launch count is fixed by the plan,
        //    and a segment past the device valid_rows exits without writing.
        let q_row_bytes = b.q.0.stride[0] * elem;
        let out_row_bytes = b.core_out.0.stride[0] * elem;
        for segment in 0..plan.segments {
            let row_base = segment * seg;
            let base = row_base as i32;
            let q_ptr = b.q.0.ptr + u64::from(row_base) * q_row_bytes;
            let out_ptr = b.core_out.0.ptr + u64::from(row_base) * out_row_bytes;
            unsafe {
                stream
                    .launch_builder(&plan.kernels.phase1)
                    .arg(&q_ptr)
                    .arg(&b.k_cache.0.ptr)
                    .arg(&b.v_cache.0.ptr)
                    .arg(&p_max)
                    .arg(&p_sum)
                    .arg(&p_out)
                    .arg(&b.causal_mask.ptr)
                    .arg(&meta.ptr)
                    .arg(&q_tok)
                    .arg(&head_stride)
                    .arg(&pos_stride)
                    .arg(&head_stride)
                    .arg(&pos_stride)
                    .arg(&key_chunk)
                    .arg(&pmax)
                    .arg(&seg_rows)
                    .arg(&base)
                    .launch(plan.phase1_cfg)?;
                stream
                    .launch_builder(&plan.kernels.phase2)
                    .arg(&p_max)
                    .arg(&p_sum)
                    .arg(&p_out)
                    .arg(&out_ptr)
                    .arg(&meta.ptr)
                    .arg(&out_stride)
                    .arg(&key_chunk)
                    .arg(&pmax)
                    .arg(&seg_rows)
                    .arg(&base)
                    .launch(plan.phase2_cfg)?;
            }
        }
        Ok(())
    }
}

impl FullAttentionPlan {
    /// Rows per phase-1/phase-2 segment: `min(max_rows, 64)`.
    pub(crate) fn segment_rows(&self) -> u32 {
        self.segment_rows
    }

    /// Key partitions the fixed-`Pmax` partial layout is sized for.
    pub(crate) fn max_parts(&self) -> u32 {
        self.max_parts
    }

    /// Segments launched per `run`, fixed by the plan.
    pub(crate) fn segments(&self) -> u32 {
        self.segments
    }

    /// Phase-1 dynamic shared memory per CTA.
    pub(crate) fn phase1_smem_bytes(&self) -> u32 {
        PHASE1_SMEM
    }

    /// Phase-2 dynamic shared memory per CTA: `(2*max_parts + 256) * 4`.
    pub(crate) fn phase2_smem_bytes(&self) -> u32 {
        self.phase2_cfg.shared_mem_bytes
    }

    pub(crate) fn workspace(&self) -> AttentionWorkspace {
        self.ws
    }
}

#[cfg(test)]
mod tests {
    //! GPU probe for the binder itself: `workspace` -> `prepare` -> `run` through
    //! the real launch path, including the common `NormOps`.
    //!
    //! The kernel numerics are covered by the G4 harness. What this adds is the
    //! plumbing: cudarc marshals launch arguments at run time from the kernel's
    //! signature, so a wrong argument order or width yields a wrong answer or a
    //! fault, and nothing else in the tree would notice. This is also the only
    //! place the Q/K norm is exercised through this module.
    //!
    //! `#[ignore]`-gated because it initialises a device:
    //!   FF_Q38_GPU_WINDOW=granted cargo test --features cuda --lib \
    //!     attention_d256 -- --ignored --nocapture
    use super::*;
    use crate::backend::cuda::Device;
    use crate::op_contract::SequenceMeta;
    use cudarc::driver::{CudaSlice, DevicePtr};

    /// The model's RoPE constants live in the kernel, so the reference states
    /// them again rather than importing them.
    const ROTARY: usize = 64;
    const THETA: f32 = 1.0e7;
    /// 1/sqrt(256), the scale the kernel applies to the QK scores.
    const SCALE: f32 = 0.0625;

    /// One bf16 ulp, relative to a value's own magnitude.
    ///
    /// This is the floor for any comparison where *both* sides are
    /// bf16-representable, which is the case here: the reference applies the
    /// same visible casts the kernels do, so its residual against its own bf16
    /// rounding is zero by construction and cannot serve as a floor. One ulp can
    /// genuinely separate two correct results, because the common norm ends in
    /// `rsqrtf` (~2 ulp) where the reference takes a `sqrt`, and a value sitting
    /// near a rounding boundary then lands one step away. Measured, not assumed:
    /// the Q path lands at 3.1e-3, i.e. 0.8 ulp, against the 3.125e-2 bound.
    /// (The G4 harness's floor is half an ulp because its reference is pure fp32.)
    const BF16_ULP: f32 = 1.0 / 256.0;
    const BINDER_BOUND: f32 = 8.0 * BF16_ULP;

    /// Deterministic xorshift, so both shapes and any rerun see the same bytes.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f64 / 9007199254740992.0 * 2.0 - 1.0) as f32
        }
        fn bf(&mut self) -> bf16 {
            bf16::from_f32(self.next() * 1.5)
        }
    }

    fn view<T: cudarc::driver::DeviceRepr + Copy, const R: usize>(
        buffer: &CudaSlice<T>,
        stream: &Arc<CudaStream>,
        shape: [u32; R],
        stride: [u64; R],
    ) -> Result<Tensor<T, R>> {
        let (ptr, _hold) = buffer.device_ptr(stream);
        unsafe { Tensor::new(ptr, shape, stride, (buffer.len() * size_of::<T>()) as u64) }
    }

    fn meta_device(
        stream: &Arc<CudaStream>,
        prefix: i32,
        valid_rows: i32,
        capacity: i32,
    ) -> Result<(CudaSlice<i32>, SeqRef)> {
        let m = SequenceMeta {
            prefix,
            valid_rows,
            key_count: prefix + valid_rows,
            capacity,
            causal: 1,
            reserved: [0; 3],
        };
        m.validate(capacity as u32, valid_rows as u32)?;
        let buffer = stream.clone_htod(&[
            m.prefix,
            m.valid_rows,
            m.key_count,
            m.capacity,
            m.causal,
            0,
            0,
            0,
        ])?;
        let (ptr, hold) = buffer.device_ptr(stream);
        drop(hold);
        Ok((buffer, SeqRef { ptr }))
    }

    struct Inputs<'a> {
        prefix: u32,
        valid_rows: u32,
        capacity: u32,
        q_gate: &'a [bf16],
        k_proj: &'a [bf16],
        v_proj: &'a [bf16],
        q_norm_w: &'a [bf16],
        k_norm_w: &'a [bf16],
        prefix_k: &'a [bf16],
        prefix_v: &'a [bf16],
    }

    /// The whole ABI §4.2 run order in fp32, with the model's visible bf16 casts
    /// applied where the kernels apply them: after the zero-centred norm and
    /// after RoPE. P is rounded to bf16 before it multiplies V, as the MMA does.
    /// The QK reduction and the softmax stay fp32, so this is the same reference
    /// the G4 harness uses, just on the test's own inputs.
    struct Reference {
        prefix: usize,
        valid_rows: usize,
        capacity: usize,
        cache_k: Vec<f32>,
        cache_v: Vec<f32>,
        q: Vec<f32>,
    }

    fn norm(x: &[bf16], w: &[bf16]) -> Vec<f32> {
        let values: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
        let mean = values.iter().map(|v| v * v).sum::<f32>() / values.len() as f32;
        let inv = 1.0 / (mean + RMS_EPS).sqrt();
        values
            .iter()
            .zip(w)
            .map(|(v, w)| bf16::from_f32(v * inv * (1.0 + w.to_f32())).to_f32())
            .collect()
    }

    fn rope(x: &mut [f32], pos: u32) {
        let half = ROTARY / 2;
        let (a, b): (Vec<f32>, Vec<f32>) = (x[..half].to_vec(), x[half..ROTARY].to_vec());
        for j in 0..half {
            let inv = 1.0 / THETA.powf(2.0 * j as f32 / ROTARY as f32);
            let angle = pos as f32 * inv;
            let (sin, cos) = angle.sin_cos();
            x[j] = bf16::from_f32(a[j] * cos - b[j] * sin).to_f32();
            x[j + half] = bf16::from_f32(b[j] * cos + a[j] * sin).to_f32();
        }
    }

    impl Reference {
        fn new(i: &Inputs<'_>) -> Self {
            let (d, hq, hkv) = (D as usize, HQ as usize, HKV as usize);
            let mut q = vec![0f32; i.valid_rows as usize * hq * d];
            for row in 0..i.valid_rows as usize {
                for head in 0..hq {
                    let base = (row * hq + head) * 2 * d;
                    let mut values = norm(&i.q_gate[base..base + d], i.q_norm_w);
                    rope(&mut values, i.prefix + row as u32);
                    let out = (row * hq + head) * d;
                    q[out..out + d].copy_from_slice(&values);
                }
            }
            let span = i.capacity as usize * d;
            let mut cache_k = vec![0f32; hkv * span];
            let mut cache_v = vec![0f32; hkv * span];
            for kv in 0..hkv {
                for pos in 0..i.prefix as usize {
                    let src = (kv * i.prefix as usize + pos) * d;
                    let dst = kv * span + pos * d;
                    for j in 0..d {
                        cache_k[dst + j] = i.prefix_k[src + j].to_f32();
                        cache_v[dst + j] = i.prefix_v[src + j].to_f32();
                    }
                }
                for row in 0..i.valid_rows as usize {
                    let src = (row * hkv + kv) * d;
                    let mut values = norm(&i.k_proj[src..src + d], i.k_norm_w);
                    rope(&mut values, i.prefix + row as u32);
                    let dst = kv * span + (i.prefix as usize + row) * d;
                    cache_k[dst..dst + d].copy_from_slice(&values);
                    for j in 0..d {
                        cache_v[dst + j] = i.v_proj[src + j].to_f32();
                    }
                }
            }
            Self {
                prefix: i.prefix as usize,
                valid_rows: i.valid_rows as usize,
                capacity: i.capacity as usize,
                cache_k,
                cache_v,
                q,
            }
        }

        fn core_out(&self) -> Vec<f32> {
            let (d, hq) = (D as usize, HQ as usize);
            let span = self.capacity * d;
            let mut out = vec![0f32; self.valid_rows * hq * d];
            for row in 0..self.valid_rows {
                let limit = self.prefix + row + 1;
                for head in 0..hq {
                    let kv = head / GQA as usize;
                    let q = &self.q[(row * hq + head) * d..][..d];
                    let mut scores = vec![0f32; limit];
                    for (pos, score) in scores.iter_mut().enumerate() {
                        let k = &self.cache_k[kv * span + pos * d..][..d];
                        *score = k.iter().zip(q).map(|(a, b)| a * b).sum::<f32>() * SCALE;
                    }
                    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let p: Vec<f32> = scores.iter().map(|v| (v - max).exp()).collect();
                    let den: f32 = p.iter().sum();
                    let mut num = vec![0f32; d];
                    for (pos, pv) in p.iter().enumerate() {
                        let pb = bf16::from_f32(*pv).to_f32();
                        let v = &self.cache_v[kv * span + pos * d..][..d];
                        for (dst, src) in num.iter_mut().zip(v) {
                            *dst = pb.mul_add(*src, *dst);
                        }
                    }
                    let base = (row * hq + head) * d;
                    for j in 0..d {
                        out[base + j] = bf16::from_f32(num[j] / den).to_f32();
                    }
                }
            }
            out
        }
    }

    /// What one `run_shape` call runs: the geometry, the optional Axes table
    /// (`None` is the linear physical-position path) and whether the fp32
    /// reference — which models physical positions — is a valid oracle for it.
    struct Case<'a> {
        prefix: u32,
        valid_rows: u32,
        max_rows: u32,
        capacity: u32,
        key_chunk: u32,
        axes: Option<&'a CudaSlice<i32>>,
        oracle: bool,
    }

    /// Everything one run leaves behind for bitwise cross-run comparison.
    struct RunDump {
        gate: Vec<bf16>,
        q: Vec<bf16>,
        core_out: Vec<bf16>,
        k_cache: Vec<bf16>,
        v_cache: Vec<bf16>,
    }

    /// One full `workspace` -> `prepare` -> `run` cycle on one shape, checked
    /// against the reference under the same derived bound the G4 harness uses:
    /// 8x the bf16 rounding floor of this reference, relative to its own scale.
    /// Nothing is tuned here -- the floor is measured from the reference.
    fn run_shape(device: &Device, case: Case<'_>) -> Result<RunDump> {
        let Case {
            prefix,
            valid_rows,
            max_rows,
            capacity,
            key_chunk,
            axes,
            oracle,
        } = case;
        let stream = &device.stream;
        let spec = FullAttentionSpec::q38();
        let limits = AttentionLimits {
            max_rows,
            capacity,
            key_chunk,
            causal: 1,
        };
        let request = FullAttentionOps::workspace(&spec, limits)?;
        ensure!(request.alignment == 256, "workspace alignment changed");
        let scratch_bytes = usize::try_from(request.bytes)?;

        // ---- host inputs: both halves of Q, K and V carry signal, so a wrong
        // head mapping or RoPE width cannot hide behind zeros.
        let mut rng = Rng(0x243F_6A88_5A30_8D3D ^ (u64::from(prefix) << 8) ^ u64::from(valid_rows));
        let (d, hq, hkv) = (D as usize, HQ as usize, HKV as usize);
        let q_gate: Vec<bf16> = (0..max_rows as usize * hq * 2 * d)
            .map(|_| rng.bf())
            .collect();
        let k_proj: Vec<bf16> = (0..max_rows as usize * hkv * d).map(|_| rng.bf()).collect();
        let v_proj: Vec<bf16> = (0..max_rows as usize * hkv * d).map(|_| rng.bf()).collect();
        let q_norm_w: Vec<bf16> = (0..d).map(|_| bf16::from_f32(rng.next() * 0.25)).collect();
        let k_norm_w: Vec<bf16> = (0..d).map(|_| bf16::from_f32(rng.next() * 0.25)).collect();
        let prefix_k: Vec<bf16> = (0..hkv * prefix as usize * d).map(|_| rng.bf()).collect();
        let prefix_v: Vec<bf16> = (0..hkv * prefix as usize * d).map(|_| rng.bf()).collect();

        // Caches are NaN past the visible extent: a read beyond it turns the
        // output NaN rather than into a plausible small error, and rows at or
        // past valid_rows must stay untouched.
        let nan = bf16::from_bits(0x7fc0);
        let mut k_cache = vec![nan; hkv * capacity as usize * d];
        let mut v_cache = vec![nan; hkv * capacity as usize * d];
        for kv in 0..hkv {
            for pos in 0..prefix as usize {
                let src = (kv * prefix as usize + pos) * d;
                let dst = kv * capacity as usize * d + pos * d;
                k_cache[dst..dst + d].copy_from_slice(&prefix_k[src..src + d]);
                v_cache[dst..dst + d].copy_from_slice(&prefix_v[src..src + d]);
            }
        }
        let mask: Vec<u64> = (0..MASK_LEN)
            .map(|r| {
                if r < MASK_LEN - 1 {
                    (1u64 << (r + 1)) - 1
                } else {
                    u64::MAX
                }
            })
            .collect();

        // ---- device buffers ----------------------------------------------
        let dev_q_gate = stream.clone_htod(&q_gate)?;
        let dev_k = stream.clone_htod(&k_proj)?;
        let dev_v = stream.clone_htod(&v_proj)?;
        let dev_q = stream.alloc_zeros::<bf16>(max_rows as usize * hq * d)?;
        let dev_gate = stream.alloc_zeros::<bf16>(max_rows as usize * hq * d)?;
        let dev_out = stream.clone_htod(&vec![nan; max_rows as usize * hq * d])?;
        let dev_k_cache = stream.clone_htod(&k_cache)?;
        let dev_v_cache = stream.clone_htod(&v_cache)?;
        let dev_q_norm = stream.clone_htod(&q_norm_w)?;
        let dev_k_norm = stream.clone_htod(&k_norm_w)?;
        let dev_mask = stream.clone_htod(&mask)?;
        let (_metadata, meta_ref) =
            meta_device(stream, prefix as i32, valid_rows as i32, capacity as i32)?;
        let scratch = stream.alloc_zeros::<u8>(scratch_bytes)?;
        let (scratch_ptr, _hold) = scratch.device_ptr(stream);

        let bindings = AttentionBindings {
            q_gate: view(
                &dev_q_gate,
                stream,
                [max_rows, HQ, 2 * D],
                [(HQ * 2 * D) as u64, (2 * D) as u64, 1],
            )?,
            k: view(
                &dev_k,
                stream,
                [max_rows, HKV, D],
                [(HKV * D) as u64, D as u64, 1],
            )?,
            v: view(
                &dev_v,
                stream,
                [max_rows, HKV, D],
                [(HKV * D) as u64, D as u64, 1],
            )?,
            q: TensorMut(view(
                &dev_q,
                stream,
                [max_rows, HQ, D],
                [(HQ * D) as u64, D as u64, 1],
            )?),
            gate: TensorMut(view(
                &dev_gate,
                stream,
                [max_rows, HQ, D],
                [(HQ * D) as u64, D as u64, 1],
            )?),
            k_cache: TensorMut(view(
                &dev_k_cache,
                stream,
                [HKV, capacity, D],
                [(capacity * D) as u64, D as u64, 1],
            )?),
            v_cache: TensorMut(view(
                &dev_v_cache,
                stream,
                [HKV, capacity, D],
                [(capacity * D) as u64, D as u64, 1],
            )?),
            core_out: TensorMut(view(
                &dev_out,
                stream,
                [max_rows, HQ, D],
                [(HQ * D) as u64, D as u64, 1],
            )?),
            q_norm: view(&dev_q_norm, stream, [D], [1])?,
            k_norm: view(&dev_k_norm, stream, [D], [1])?,
            causal_mask: view(&dev_mask, stream, [MASK_LEN], [1])?,
            positions: match axes {
                None => PositionBindings::Linear,
                Some(table) => PositionBindings::Axes(view(table, stream, [max_rows, 3], [3, 1])?),
            },
        };
        let common = CommonOpsRef::new(&device.ctx)?;
        let plan = FullAttentionOps::prepare(&device.ctx, &spec, limits, bindings, common)?;
        FullAttentionOps::run(
            &plan,
            stream,
            meta_ref,
            Scratch {
                ptr: scratch_ptr,
                bytes: request.bytes,
            },
        )?;
        stream.synchronize()?;

        // ---- compare -----------------------------------------------------
        // The split is pure data movement, so it is checked bitwise; the Q path
        // is not, because the common norm ends in `rsqrtf` (~2 ulp) where the
        // reference takes a `sqrt`, and a 2-ulp difference in the scale can flip
        // a bf16 rounding on a value sitting on the boundary. Checking it under
        // its own derived floor still isolates the stage without failing a
        // kernel that is merely *differently* rounded.
        let gate = stream.clone_dtoh(&dev_gate)?;
        let q_roped = stream.clone_dtoh(&dev_q)?;
        for row in 0..valid_rows as usize {
            for head in 0..hq {
                let src = (row * hq + head) * 2 * d + d;
                let dst = (row * hq + head) * d;
                ensure!(
                    gate[dst..dst + d]
                        .iter()
                        .zip(&q_gate[src..src + d])
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "gate view mismatch at row {row} head {head}"
                );
            }
        }
        let got = stream.clone_dtoh(&dev_out)?;
        let live = valid_rows as usize * hq * d;
        ensure!(
            got[live..].iter().all(|v| v.is_nan()),
            "rows at or past valid_rows were written (prefix={prefix} valid_rows={valid_rows})"
        );
        if oracle {
            let reference = Reference::new(&Inputs {
                prefix,
                valid_rows,
                capacity,
                q_gate: &q_gate,
                k_proj: &k_proj,
                v_proj: &v_proj,
                q_norm_w: &q_norm_w,
                k_norm_w: &k_norm_w,
                prefix_k: &prefix_k,
                prefix_v: &prefix_v,
            });
            let want = reference.core_out();
            let q_scale = reference
                .q
                .iter()
                .map(|v| v.abs())
                .fold(0f32, f32::max)
                .max(1e-12);
            let q_rel = q_roped
                .iter()
                .zip(&reference.q)
                .map(|(a, b)| (a.to_f32() - b).abs())
                .fold(0f32, f32::max)
                / q_scale;
            eprintln!("P1D_BINDER_Q rel={q_rel:e} bound={BINDER_BOUND:e}");
            ensure!(
                q_rel <= BINDER_BOUND,
                "Q norm + partial RoPE differs: rel={q_rel:e} bound={BINDER_BOUND:e}"
            );
            let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max).max(1e-12);
            let bound = BINDER_BOUND;
            let worst = want
                .iter()
                .enumerate()
                .map(|(i, &w)| (got[i].to_f32() - w).abs())
                .fold(0f32, f32::max);
            let rel = worst / scale;
            eprintln!(
                "P1D_BINDER prefix={prefix} valid_rows={valid_rows} max_rows={max_rows} \
                 capacity={capacity} key_chunk={key_chunk} segments={} max_parts={} \
                 phase2_smem={} rel={rel:e} bound={bound:e}",
                plan.segments(),
                plan.max_parts(),
                plan.phase2_smem_bytes()
            );
            ensure!(
                rel <= bound,
                "binder output differs from the fp32 reference: rel={rel:e} bound={bound:e} \
                 (prefix={prefix} valid_rows={valid_rows})"
            );
        }
        ensure!(
            plan.phase2_smem_bytes() == (2 * plan.max_parts() + PHASE2_THREADS) * 4,
            "phase-2 shared memory sizing changed"
        );
        Ok(RunDump {
            gate,
            q: q_roped,
            core_out: got,
            k_cache: stream.clone_dtoh(&dev_k_cache)?,
            v_cache: stream.clone_dtoh(&dev_v_cache)?,
        })
    }

    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_Q38_GPU_WINDOW=granted"]
    fn binder_matches_the_fp32_reference() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        // The environment selects the device ordinal for the GPU probe.
        let ordinal: usize = std::env::var("FF_Q38_GPU_ORDINAL")
            .unwrap_or_else(|_| "0".into())
            .parse()?;
        let device = Device::new(ordinal)?;
        // m65: prefix 33, two segments -- both partials and the merge exercised.
        run_shape(
            &device,
            Case {
                prefix: 33,
                valid_rows: 65,
                max_rows: 65,
                capacity: 256,
                key_chunk: 32,
                axes: None,
                oracle: true,
            },
        )?;
        // replay17: max_rows 64 with only 17 live rows -- the graph-replay shape.
        run_shape(
            &device,
            Case {
                prefix: 0,
                valid_rows: 17,
                max_rows: 64,
                capacity: 1024,
                key_chunk: 64,
                axes: None,
                oracle: true,
            },
        )?;

        // ---- three-axis gate (VISION-WIRE item 2) -------------------------
        // The degenerate Axes table (T = H = W = physical position) must be
        // BIT-identical to the Linear path on the same shape: a wrong axis
        // pointer, dispatch or launch would otherwise hide inside the reference
        // bound. Then a non-degenerate table must move the output -- otherwise
        // this would pass a build whose axes entries quietly ignore the table.
        fn case<'a>(axes: Option<&'a CudaSlice<i32>>, oracle: bool) -> Case<'a> {
            Case {
                prefix: 33,
                valid_rows: 65,
                max_rows: 65,
                capacity: 256,
                key_chunk: 32,
                axes,
                oracle,
            }
        }
        let linear = run_shape(&device, case(None, true))?;
        let degenerate: Vec<i32> = (0..65).flat_map(|r| [33 + r, 33 + r, 33 + r]).collect();
        let degenerate = device.stream.clone_htod(&degenerate)?;
        let axes = run_shape(&device, case(Some(&degenerate), true))?;
        for (name, a, b) in [
            ("q", &linear.q, &axes.q),
            ("core_out", &linear.core_out, &axes.core_out),
            ("k_cache", &linear.k_cache, &axes.k_cache),
            ("v_cache", &linear.v_cache, &axes.v_cache),
            ("gate", &linear.gate, &axes.gate),
        ] {
            ensure!(
                a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "degenerate axes changed {name}: the pure-text path must be bit-identical"
            );
        }
        eprintln!(
            "P1D_BINDER_AXES degenerate=bitwise_identical over q/core_out/k_cache/v_cache/gate"
        );

        // W = physical + 7 is not the pure-text coordinate, so the oracle does
        // not apply; the claim is only that the table is really read.
        let shifted: Vec<i32> = (0..65).flat_map(|r| [33 + r, 33 + r, 40 + r]).collect();
        let shifted = device.stream.clone_htod(&shifted)?;
        let moved = run_shape(&device, case(Some(&shifted), false))?;
        ensure!(
            moved
                .core_out
                .iter()
                .zip(&linear.core_out)
                .any(|(x, y)| x.to_bits() != y.to_bits()),
            "a non-degenerate axes table changed nothing: the table is not read"
        );
        eprintln!("P1D_BINDER_AXES non-degenerate=differs (control)");
        eprintln!(
            "P1D_BINDER_PROBE ok=true shapes=2 axes=degenerate+control norm=common_rms ordinal={ordinal}"
        );
        Ok(())
    }
}
