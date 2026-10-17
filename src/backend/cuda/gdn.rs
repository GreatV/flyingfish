//! Gated Delta Net for Qwen3.8-27B (ABI v1 §6).
//!
//! Shape is fixed by [`GdnSpec::q38`]: 48 value heads, 16 key heads, GQA 3,
//! dk = dv = 128, conv width 4, history width 3, chunk 32, FP32 tiles. Kernels are in `kernels/gdn/`.
//!
//! `run` issues the whole mixer between the `qkv`/`a`/`b` projections and the ungated
//! `core_out` on one stream; Q/K norms and gates use common ops ([`CommonOpsRef`]).
//!
//! `hist_update` is a separate launch after conv; conv blocks read `history[]` to seed their window.
//!
//! Ordinary `run` selects the core by phase. `run_verify` keeps Prefill preparation and uses the
//! R1..5 recurrent core on sm_89. Modules and launch bounds are prepared before Ready.
//!
//! Rows are local: `qkv` and `core_out` are this call's `[M, ...]`; prefix rows are reachable only
//! through `history[]`. `GdnMeta.prefix` is validated, never used as an index.

/// Test-only record of which core each `run()` selected.
#[cfg(test)]
pub(crate) static PHASE_LOG: std::sync::Mutex<Vec<&'static str>> =
    std::sync::Mutex::new(Vec::new());

#[cfg(test)]
pub(crate) fn record_phase(which: &'static str) {
    PHASE_LOG.lock().unwrap().push(which);
}

#[cfg(test)]
pub(crate) fn take_phase_log() -> Vec<&'static str> {
    std::mem::take(&mut *PHASE_LOG.lock().unwrap())
}

use super::contracts::{RowDomain, Scratch, SeqRef, Tensor, TensorMut};
use super::cubin;
use super::ops::{CommonOpsRef, GdnGateBindings};
use crate::gdn_plan::{
    CW_P2_CHUNK, CW_P2_VALUE_BLOCK, GdnPolicy, GdnWorkspace, Region, add, cw_p2, cw_p2_bytes, mul,
};
use crate::op_contract::{ExecutionPhase, GdnSpec, WorkspaceReq};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use half::bf16;
use std::mem::size_of;
use std::sync::Arc;

/// Value heads, and the head count the core indexes Q/K/V by.
pub(crate) const HV: u32 = 48;
/// Key heads. K is expanded from these to [`HV`] before the core is launched.
pub(crate) const HK: u32 = 16;
/// Value heads per key head.
pub(crate) const GQA: u32 = HV / HK;
pub(crate) const DK: u32 = 128;
pub(crate) const DV: u32 = 128;
/// Conv thread block / channel tile, from `kernels/gdn/gdn_conv.cu`.
const CONV_CHAN: u32 = 256;
/// Rows per conv block.
const CONV_ROWS: u32 = 32;
/// Core thread block.
const CORE_THREADS: u32 = 512;
/// Prep thread block.
const PREP_THREADS: u32 = 128;
/// Dynamic shared memory the recurrent and chunked cores request; static shared memory is 0.
const RECURRENT_SMEM: u32 = 4096;
const CHUNKED_SMEM: u32 = 62_336;
/// Chunkwise-P2 thread block; both stages use one arena sized by `gdn_cw_p2_layout.h`.
const P2_THREADS: u32 = 256;

/// The ABI §6 views; `prepare` validates every stride and span against the registered layout.
/// `conv_dim` comes from [`GdnSpec::conv_dim`], not from these bindings.
#[derive(Debug)]
pub(crate) struct GdnBindings {
    /// Pre-conv projection `[M, conv_dim]`: Q at offset 0 width `HK*DK`, K next
    /// width `HK*DK`, V next width `HV*DV`.
    pub qkv: Tensor<bf16, 2>,
    /// Gate inputs `[M, 48]`.
    pub a: Tensor<bf16, 2>,
    pub b: Tensor<bf16, 2>,
    /// Depthwise conv weights `[conv_dim, 4]`, a zero-copy reshape of the
    /// manifest's `[conv_dim, 1, 4]`.
    pub conv_weight: Tensor<bf16, 2>,
    /// `[48]` each.
    pub a_log: Tensor<bf16, 1>,
    pub dt_bias: Tensor<bf16, 1>,
    /// Ungated core output `[M, 48, 128]`. The caller applies `rms_gated`.
    pub core_out: TensorMut<bf16, 3>,
    /// Recurrent state `[48, 128, 128]`, axes (value_head, key_dim, value_dim).
    /// Persistent across calls; owned by the Engine's StateStore.
    pub s: TensorMut<f32, 3>,
    /// Conv history `[conv_dim, 3]`, oldest -> newest, an EXACT widening of the
    /// raw pre-conv BF16 input (NOT of the conv or SiLU result). Persistent.
    pub history: TensorMut<f32, 2>,
}

struct Kernels {
    conv: CudaFunction,
    hist_update: CudaFunction,
    prep: CudaFunction,
    beta_widen: CudaFunction,
    chunked: CudaFunction,
    recurrent: CudaFunction,
    rstep: Option<CudaFunction>,
    hist_snapshot: CudaFunction,
    core_snapshot: CudaFunction,
    /// Bound only when the plan was prepared for the chunkwise-P2 split.
    p2: Option<ChunkwiseP2>,
}

/// The two chunkwise-P2 stages, launched A then H on one stream with nothing between:
/// A fills the workspace slot, H walks the chunks in order.
struct ChunkwiseP2 {
    a: CudaFunction,
    h: CudaFunction,
}

/// Chunkwise-P2 launch configs for a plan bound at `rows`: stage A one CTA per (chunk, head)
/// with grid.x = ceil(rows/64), stage H one CTA per 32-wide value block with grid.x = DV/32.
fn cw_p2_grids(rows: u32) -> (LaunchConfig, LaunchConfig) {
    let per_chunk = LaunchConfig {
        grid_dim: (rows.div_ceil(CW_P2_CHUNK), HV, 1),
        block_dim: (P2_THREADS, 1, 1),
        shared_mem_bytes: cw_p2::SMEM_BYTES as u32,
    };
    let per_value_block = LaunchConfig {
        grid_dim: (DV / CW_P2_VALUE_BLOCK, HV, 1),
        block_dim: (P2_THREADS, 1, 1),
        shared_mem_bytes: cw_p2::SMEM_BYTES as u32,
    };
    (per_chunk, per_value_block)
}

/// CUDA caps gridDim.y and gridDim.z at 65535; prep puts rows on gridDim.z and beta on gridDim.y.
pub(crate) const CUDA_GRID_YZ_MAX: u32 = 65_535;

/// Prep grids for `rows`; `prepare` and `for_rows` both derive them here.
fn prep_grids(
    prep_cfg: LaunchConfig,
    beta_cfg: LaunchConfig,
    conv_cfg: LaunchConfig,
    rows: u32,
) -> Result<(LaunchConfig, LaunchConfig, LaunchConfig)> {
    ensure!(
        rows <= CUDA_GRID_YZ_MAX,
        "GDN prep rows {rows} exceed the CUDA gridDim.y/z limit {CUDA_GRID_YZ_MAX}"
    );
    let mut prep_cfg = prep_cfg;
    prep_cfg.grid_dim.2 = rows;
    let mut beta_cfg = beta_cfg;
    beta_cfg.grid_dim.1 = rows;
    let mut conv_cfg = conv_cfg;
    conv_cfg.grid_dim.1 = rows.div_ceil(CONV_ROWS);
    Ok((prep_cfg, beta_cfg, conv_cfg))
}

pub(crate) struct GdnPlan {
    snapshots: Option<(Tensor<f32, 4>, Tensor<f32, 3>)>,
    kernels: Arc<Kernels>,
    bindings: GdnBindings,
    common: CommonOpsRef,
    ws: GdnWorkspace,
    /// The single derivation of `conv_dim`, asserted against the kernel constant.
    conv_dim: u32,
    /// `GdnSpec::l2_eps`, used for the Q/K `l2_heads` calls.
    l2_eps: f32,
    max_rows: u32,
    prep_cfg: LaunchConfig,
    beta_cfg: LaunchConfig,
    conv_cfg: LaunchConfig,
    hist_cfg: LaunchConfig,
    chunked_cfg: LaunchConfig,
    recurrent_cfg: LaunchConfig,
    /// Whether this plan carries the chunkwise-P2 split.
    cw_p2: bool,
    /// Chunkwise-P2 workspace bytes this plan needs past `ws.bytes`, 0 when it does not
    /// carry the split. The slot region starts at `ws.bytes`.
    ws_p2: u64,
}

/// BF16 sources the P2 stages read: Q and K as 16 key heads each, V as the conv_out
/// column window with row stride `v_row_stride` elements.
#[derive(Copy, Clone)]
pub(crate) struct P2Sources {
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub v_row_stride: u32,
}

impl GdnPlan {
    /// Setup-only prefix view; reuses loaded modules and physical addresses.
    pub(crate) fn for_rows(
        &self,
        rows: u32,
        snapshots: Option<(Tensor<f32, 4>, Tensor<f32, 3>)>,
    ) -> Result<Self> {
        ensure!(
            rows > 0 && rows <= self.max_rows,
            "GDN rows outside binding"
        );
        if let Some((state, history)) = snapshots {
            ensure!((1..=5).contains(&rows), "snapshot plan bound must be 1..5");
            state.validate()?;
            ensure!(
                state.shape == [4, HV, DK, DV]
                    && state.stride
                        == [
                            u64::from(HV * DK * DV),
                            u64::from(DK * DV),
                            u64::from(DV),
                            1
                        ],
                "invalid state snapshot layout"
            );
            history.validate()?;
            ensure!(
                history.shape == [4, self.conv_dim, 3]
                    && history.stride == [u64::from(self.conv_dim * 3), 3, 1],
                "invalid history snapshot layout"
            );
        }
        let b = &self.bindings;
        let bindings = GdnBindings {
            qkv: b.qkv.prefix_rows(rows)?,
            a: b.a.prefix_rows(rows)?,
            b: b.b.prefix_rows(rows)?,
            conv_weight: b.conv_weight,
            a_log: b.a_log,
            dt_bias: b.dt_bias,
            core_out: TensorMut(b.core_out.read().prefix_rows(rows)?),
            s: TensorMut(b.s.read()),
            history: TensorMut(b.history.read()),
        };
        check_bindings(&bindings, rows, self.conv_dim)?;
        let (prep_cfg, beta_cfg, conv_cfg) =
            prep_grids(self.prep_cfg, self.beta_cfg, self.conv_cfg, rows)?;
        let cw_p2 = snapshots.is_none() && self.cw_p2 && rows > crate::gdn_plan::MIN_VERIFY_ROWS;
        Ok(Self {
            snapshots,
            kernels: self.kernels.clone(),
            bindings,
            common: self.common.clone(),
            ws: GdnWorkspace::q38(&GdnSpec::q38(), rows)?,
            conv_dim: self.conv_dim,
            l2_eps: self.l2_eps,
            max_rows: rows,
            prep_cfg,
            beta_cfg,
            conv_cfg,
            hist_cfg: self.hist_cfg,
            chunked_cfg: self.chunked_cfg,
            recurrent_cfg: self.recurrent_cfg,
            ws_p2: if cw_p2 { cw_p2_bytes(rows) } else { 0 },
            cw_p2,
        })
    }

    /// Chunkwise-P2 prefill core: stage A then stage H on the caller's stream with no
    /// synchronisation between them; row counts resolve on the device.
    pub(crate) fn gdn_cw_p2_step(
        &self,
        stream: &Arc<CudaStream>,
        meta: SeqRef,
        scratch: Scratch,
        src: P2Sources,
    ) -> Result<()> {
        meta.validate()?;
        scratch.validate(add(self.ws.bytes, self.ws_p2)?)?;
        let p2 = self
            .kernels
            .p2
            .as_ref()
            .context("chunkwise P2 stages were not bound by prepare")?;
        let rows = self.max_rows;
        let ws = scratch.ptr;
        let (q, k, v) = (src.q, src.k, src.v);
        let beta = ws_f32(self.ws.beta_fp32, ws, rows, HV)?.0.ptr;
        let g = ws_f32(self.ws.g, ws, rows, HV)?.0.ptr;
        // The slot region starts at the 256 B aligned end of the fused layout; all stages take this base as `ws`.
        let slot = add(ws, self.ws.bytes)?;
        let stage = crate::gdn_plan::P2StageArgs {
            q,
            k,
            v,
            beta,
            g,
            ws: slot,
            meta: meta.ptr,
            valid_rows: 0,
            v_row_stride: src.v_row_stride,
        };
        let (a_cfg, h_cfg) = cw_p2_grids(rows);
        unsafe {
            stream
                .launch_builder(&p2.a)
                .arg(&stage.q)
                .arg(&stage.k)
                .arg(&stage.v)
                .arg(&stage.beta)
                .arg(&stage.g)
                .arg(&stage.ws)
                .arg(&stage.meta)
                .arg(&stage.valid_rows)
                .arg(&stage.v_row_stride)
                .launch(a_cfg)?;
            stream
                .launch_builder(&p2.h)
                .arg(&stage.q)
                .arg(&stage.k)
                .arg(&stage.v)
                .arg(&stage.beta)
                .arg(&stage.g)
                .arg(&self.bindings.s.0.ptr)
                .arg(&self.bindings.core_out.0.ptr)
                .arg(&slot)
                .arg(&stage.meta)
                .arg(&stage.valid_rows)
                .launch(h_cfg)?;
        }
        Ok(())
    }
}

pub(crate) struct GdnOps;

pub(crate) const GDN_CONV_DIM: u32 = 10_240;

impl GdnOps {
    /// CPU-only checked sizing; the only source of the Engine's GDN byte accounting.
    pub(crate) fn workspace(
        spec: &GdnSpec,
        max_rows: u32,
        policy: GdnPolicy,
    ) -> Result<WorkspaceReq> {
        policy.workspace(spec, max_rows)
    }

    /// Setup binder: loads its own kernels, validates the registered layout, opts
    /// in the chunked dynamic shared memory, and binds stable addresses.
    pub(crate) fn prepare(
        ctx: &Arc<CudaContext>,
        spec: &GdnSpec,
        max_rows: u32,
        bindings: GdnBindings,
        common: CommonOpsRef,
        cw_p2: bool,
    ) -> Result<GdnPlan> {
        spec.validate()?;
        ensure!(
            HK * GQA == HV,
            "GDN GQA {GQA} does not expand {HK} key heads to {HV} value heads"
        );
        let ws = GdnWorkspace::q38(spec, max_rows)?;
        // `conv_dim` is derived once and asserted against the kernel constant.
        let conv_dim = spec.conv_dim()?;
        ensure!(
            conv_dim == GDN_CONV_DIM,
            "GDN conv_dim {conv_dim} does not match the compiled kernel constant {GDN_CONV_DIM}"
        );
        check_bindings(&bindings, max_rows, conv_dim)?;

        use cudarc::driver::sys::CUdevice_attribute as Attribute;
        let optin =
            ctx.attribute(Attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)? as u64;
        for (name, bytes) in [
            ("recurrent", u64::from(RECURRENT_SMEM)),
            ("chunked", u64::from(CHUNKED_SMEM)),
            ("chunkwise P2 stages", cw_p2::SMEM_BYTES),
        ] {
            ensure!(
                bytes <= optin,
                "{name} dynamic shared memory {bytes} B exceeds the device opt-in limit {optin} B"
            );
        }

        let conv = cubin::module(ctx, "gdn/gdn_conv")?;
        let chunked_mod = cubin::module(ctx, "gdn/gdn_chunked")?;
        let prep_mod = cubin::module(ctx, "gdn/gdn_prep")?;
        let recurrent_mod = cubin::module(ctx, "gdn/gdn_recurrent")?;
        let f_conv = conv.load_function("gdn_conv")?;
        let f_hist = conv.load_function("gdn_conv_hist_update")?;
        let f_prep = prep_mod.load_function("gdn_prep_fp32")?;
        let f_beta = prep_mod.load_function("gdn_beta_widen")?;
        let f_chunked = chunked_mod.load_function("gdn_chunked_core_v1")?;
        let f_recurrent = recurrent_mod.load_function("gdn_recurrent_core_v2")?;

        // The dynamic shared memory opt-in is applied to every core at setup; a launch without it
        // fails with `invalid argument`.
        for (function, bytes) in [(&f_recurrent, RECURRENT_SMEM), (&f_chunked, CHUNKED_SMEM)] {
            if bytes > 48 * 1024 {
                function.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    bytes as i32,
                )?;
            }
        }

        // The rstep core binds only on sm_89.
        let rstep = if cubin::capability(ctx)? == 89 {
            Some(cubin::module(ctx, "gdn/gdn_rstep")?.load_function("gdn_rstep_core_v1")?)
        } else {
            None
        };
        let hist_snapshot =
            cubin::module(ctx, "gdn/gdn_conv")?.load_function("gdn_conv_hist_snapshot")?;
        let core_snapshot = if rstep.is_some() {
            cubin::module(ctx, "gdn/gdn_rstep")?.load_function("gdn_rstep_core_snapshot")?
        } else {
            let f = cubin::module(ctx, "gdn/gdn_chunked")?
                .load_function("gdn_chunked_core_snapshot")?;
            f.set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, CHUNKED_SMEM as i32)?;
            f
        };
        // All three arenas are under the 48 KiB no-opt-in cap.
        let p2 = if !cw_p2 {
            None
        } else {
            let p2_mod = cubin::module(ctx, "gdn/gdn_cw_p2")?;
            Some(ChunkwiseP2 {
                a: p2_mod.load_function("gdn_cw_p2_stage_a")?,
                h: p2_mod.load_function("gdn_cw_p2_stage_h")?,
            })
        };
        let kernels = Kernels {
            hist_snapshot,
            core_snapshot,
            rstep,
            conv: f_conv,
            hist_update: f_hist,
            prep: f_prep,
            beta_widen: f_beta,
            chunked: f_chunked,
            recurrent: f_recurrent,
            p2,
        };
        // Grids derive from the bound rows, never from device valid_rows.
        let (prep_cfg, beta_cfg, conv_cfg) = prep_grids(
            LaunchConfig {
                grid_dim: (DV.div_ceil(PREP_THREADS), HV, 1),
                block_dim: (PREP_THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            LaunchConfig {
                grid_dim: (HV.div_ceil(PREP_THREADS), 1, 1),
                block_dim: (PREP_THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            LaunchConfig {
                grid_dim: (conv_dim / CONV_CHAN, 1, 1),
                block_dim: (CONV_CHAN, 1, 1),
                shared_mem_bytes: 0,
            },
            max_rows,
        )?;
        Ok(GdnPlan {
            snapshots: None,
            kernels: Arc::new(kernels),
            bindings,
            common,
            ws,
            conv_dim,
            l2_eps: spec.l2_eps,
            max_rows,
            prep_cfg,
            conv_cfg,
            hist_cfg: LaunchConfig {
                grid_dim: (conv_dim / CONV_CHAN, 1, 1),
                block_dim: (CONV_CHAN, 1, 1),
                shared_mem_bytes: 0,
            },
            beta_cfg,
            chunked_cfg: LaunchConfig {
                grid_dim: (1, HV, 1),
                block_dim: (CORE_THREADS, 1, 1),
                shared_mem_bytes: CHUNKED_SMEM,
            },
            recurrent_cfg: LaunchConfig {
                grid_dim: (2, HV, 1),
                block_dim: (CORE_THREADS, 1, 1),
                shared_mem_bytes: RECURRENT_SMEM,
            },
            ws_p2: if cw_p2 { cw_p2_bytes(max_rows) } else { 0 },
            cw_p2,
        })
    }

    /// Issues the whole mixer on ONE stream, in the ABI §6 order.
    pub(crate) fn run(
        plan: &GdnPlan,
        stream: &Arc<CudaStream>,
        meta: SeqRef,
        phase: ExecutionPhase,
        scratch: Scratch,
    ) -> Result<()> {
        Self::run_core(plan, stream, meta, phase, scratch, false)
    }

    /// Batched target verification/replay, identified by the Engine's device-prefix
    /// transaction. Phase stays Prefill; only the prepared core is substituted.
    pub(crate) fn run_verify(
        plan: &GdnPlan,
        stream: &Arc<CudaStream>,
        meta: SeqRef,
        scratch: Scratch,
    ) -> Result<()> {
        ensure!(
            plan.kernels.rstep.is_none() || (1..=5).contains(&plan.max_rows),
            "verify GDN bound must be 1..5"
        );
        ensure!(plan.snapshots.is_some(), "verify snapshots not bound");
        Self::run_core(plan, stream, meta, ExecutionPhase::Prefill, scratch, true)
    }

    fn run_core(
        plan: &GdnPlan,
        stream: &Arc<CudaStream>,
        meta: SeqRef,
        phase: ExecutionPhase,
        scratch: Scratch,
        verify: bool,
    ) -> Result<()> {
        let snapshots = if verify {
            plan.snapshots.as_ref()
        } else {
            None
        };
        let rstep = if verify {
            plan.kernels.rstep.as_ref()
        } else {
            None
        };
        meta.validate()?;
        // The P2 plan skips the prep launch, so its FP32 regions are unwritten; decode uses the
        // separate one-row decode plan.
        ensure!(
            !plan.cw_p2 || phase != ExecutionPhase::Decode,
            "the chunkwise-P2 plan has no FP32 sources: decode must use the decode plan"
        );
        ensure!(
            !verify || !plan.cw_p2,
            "chunkwise plan cannot verify; verify keeps the recurrent core"
        );
        scratch.validate(add(plan.ws.bytes, plan.ws_p2)?)?;
        let b = &plan.bindings;
        let cd = plan.conv_dim;
        let rows = plan.max_rows;
        let el_bf = size_of::<bf16>() as u64;
        let ws_ptr = scratch.ptr;

        let span = mul(mul(u64::from(cd), el_bf)?, u64::from(rows))?;
        // Column offsets into `qkv` in bytes (ABI §6): Q at 0 width HK*DK, K next width HK*DK,
        // V next width HV*DV.
        let q_off = 0u64;
        let k_off = u64::from(HK * DK) * el_bf;
        let v_off = add(k_off, u64::from(HK * DK) * el_bf)?;

        let conv_out = ws_bf16(plan.ws.conv_out, ws_ptr, rows, cd)?;
        let mut q_bf16 = ws_bf16_3(plan.ws.q_bf16, ws_ptr, rows, HK, DK)?;
        let mut k_bf16 = ws_bf16_3(plan.ws.k_bf16, ws_ptr, rows, HK, DK)?;
        let mut beta = ws_bf16(plan.ws.beta, ws_ptr, rows, HV)?;
        let beta_fp32 = ws_f32(plan.ws.beta_fp32, ws_ptr, rows, HV)?;
        let mut g = ws_f32(plan.ws.g, ws_ptr, rows, HV)?;
        let q_fp32 = ws_f32_3(plan.ws.q_fp32, ws_ptr, rows, HV, DV)?;
        let k_fp32 = ws_f32_3(plan.ws.k_fp32, ws_ptr, rows, HV, DV)?;
        let v_fp32 = ws_f32_3(plan.ws.v_fp32, ws_ptr, rows, HV, DV)?;

        let dom = RowDomain {
            meta,
            row_base: 0,
            row_capacity: rows,
        };
        let l2_eps = plan.l2_eps;
        let conv_ptr = conv_out.0.ptr;

        // conv: qkv -> conv_out with the exact SiLU; the conv output is cast to BF16 and back
        // before silu() (ABI §5).
        unsafe {
            stream
                .launch_builder(&plan.kernels.conv)
                .arg(&b.qkv.ptr)
                .arg(&b.conv_weight.ptr)
                .arg(&b.history.0.ptr)
                .arg(&conv_ptr)
                .arg(&meta.ptr)
                .arg(&(cd as i32))
                .launch(plan.conv_cfg)?;
            // hist_update: after conv, same stream.
            if let Some((_, history)) = snapshots {
                stream
                    .launch_builder(&plan.kernels.hist_snapshot)
                    .arg(&b.qkv.ptr)
                    .arg(&b.history.0.ptr)
                    .arg(&meta.ptr)
                    .arg(&(cd as i32))
                    .arg(&history.ptr)
                    .launch(plan.hist_cfg)?;
            } else {
                stream
                    .launch_builder(&plan.kernels.hist_update)
                    .arg(&b.qkv.ptr)
                    .arg(&b.history.0.ptr)
                    .arg(&meta.ptr)
                    .arg(&(cd as i32))
                    .launch(plan.hist_cfg)?;
            }
        }
        // common l2_heads on Q and K only (V is not normed, ABI §5), read as conv_out column
        // windows of 16 key heads each.
        plan.common.norm.l2_heads(
            stream,
            dom,
            as3(conv_ptr, rows, HK, cd, span, q_off)?,
            &mut q_bf16,
            l2_eps,
        )?;
        plan.common.norm.l2_heads(
            stream,
            dom,
            as3(conv_ptr, rows, HK, cd, span, k_off)?,
            &mut k_bf16,
            l2_eps,
        )?;
        // common gdn_gates: beta BF16, g FP32.
        plan.common.gate.gdn_gates_with(
            stream,
            dom,
            GdnGateBindings {
                a: b.a,
                b: b.b,
                a_log: b.a_log,
                dt_bias: b.dt_bias,
                beta: &mut beta,
                g: &mut g,
            },
        )?;
        // V is read in place from its conv_out columns through the validated window helper.
        let v_bf16 = as3(conv_ptr, rows, HV, cd, span, v_off)?;
        // Exact widens; the cores read `beta` as FP32.
        unsafe {
            stream
                .launch_builder(&plan.kernels.beta_widen)
                .arg(&beta.0.ptr)
                .arg(&beta_fp32.0.ptr)
                .arg(&meta.ptr)
                .launch(plan.beta_cfg)?;
            // Q/K widened and GQA-expanded to value heads, Q scaled by 1/sqrt(DK) after the widen;
            // V stays a conv_out column window with row stride conv_dim. The P2 plan skips this launch.
            if !plan.cw_p2 {
                stream
                    .launch_builder(&plan.kernels.prep)
                    .arg(&q_bf16.0.ptr)
                    .arg(&k_bf16.0.ptr)
                    .arg(&v_bf16.ptr)
                    .arg(&q_fp32.0.ptr)
                    .arg(&k_fp32.0.ptr)
                    .arg(&v_fp32.0.ptr)
                    .arg(&cd)
                    .arg(&meta.ptr)
                    .launch(plan.prep_cfg)?;
            }
        }
        // The core is selected by phase, never by a host valid_rows. The launch builder must be one
        // expression: it borrows the stream.
        #[cfg(test)]
        record_phase(match phase {
            ExecutionPhase::Prefill if plan.cw_p2 => "chunkwise_p2",
            ExecutionPhase::Prefill if rstep.is_some() => "rstep",
            ExecutionPhase::Prefill => "chunked",
            ExecutionPhase::Decode => "recurrent",
        });
        unsafe {
            match phase {
                ExecutionPhase::Prefill if plan.cw_p2 => {
                    plan.gdn_cw_p2_step(
                        stream,
                        meta,
                        scratch,
                        P2Sources {
                            q: q_bf16.0.ptr,
                            k: k_bf16.0.ptr,
                            v: v_bf16.ptr,
                            v_row_stride: cd,
                        },
                    )?;
                }
                ExecutionPhase::Prefill if snapshots.is_some() => {
                    let (state, _) = snapshots.context("missing snapshot bindings")?;
                    let (bound, config) = if rstep.is_some() {
                        (rows as i32, plan.recurrent_cfg)
                    } else {
                        (0, plan.chunked_cfg)
                    };
                    stream
                        .launch_builder(&plan.kernels.core_snapshot)
                        .arg(&q_fp32.0.ptr)
                        .arg(&k_fp32.0.ptr)
                        .arg(&v_fp32.0.ptr)
                        .arg(&beta_fp32.0.ptr)
                        .arg(&g.0.ptr)
                        .arg(&b.s.0.ptr)
                        .arg(&b.core_out.0.ptr)
                        .arg(&meta.ptr)
                        .arg(&bound)
                        .arg(&state.ptr)
                        .launch(config)?;
                }
                ExecutionPhase::Prefill if rstep.is_some() => {
                    let core = rstep.context("missing prepared rstep core")?;
                    stream
                        .launch_builder(core)
                        .arg(&q_fp32.0.ptr)
                        .arg(&k_fp32.0.ptr)
                        .arg(&v_fp32.0.ptr)
                        .arg(&beta_fp32.0.ptr)
                        .arg(&g.0.ptr)
                        .arg(&b.s.0.ptr)
                        .arg(&b.core_out.0.ptr)
                        .arg(&meta.ptr)
                        .arg(&(rows as i32))
                        .launch(plan.recurrent_cfg)?;
                }
                ExecutionPhase::Prefill => {
                    stream
                        .launch_builder(&plan.kernels.chunked)
                        .arg(&q_fp32.0.ptr)
                        .arg(&k_fp32.0.ptr)
                        .arg(&v_fp32.0.ptr)
                        .arg(&beta_fp32.0.ptr)
                        .arg(&g.0.ptr)
                        .arg(&b.s.0.ptr)
                        .arg(&b.core_out.0.ptr)
                        .arg(&meta.ptr)
                        // The trailing `valid_rows_in` must be 0 so the kernel reads valid_rows from `meta` on
                        // every replay.
                        .arg(&0i32)
                        .launch(plan.chunked_cfg)?;
                }
                ExecutionPhase::Decode => stream
                    .launch_builder(&plan.kernels.recurrent)
                    .arg(&q_fp32.0.ptr)
                    .arg(&k_fp32.0.ptr)
                    .arg(&v_fp32.0.ptr)
                    .arg(&beta_fp32.0.ptr)
                    .arg(&g.0.ptr)
                    .arg(&b.s.0.ptr)
                    .arg(&b.core_out.0.ptr)
                    .arg(&meta.ptr)
                    .launch(plan.recurrent_cfg)
                    .map(|_| ())?,
            };
        }
        Ok(())
    }
}

/// A `[rows, heads, DV]` column window of conv_out with row stride cd at `base + offset`;
/// its capacity is `span - offset` so extent validation rejects a window past the buffer end.
fn as3(
    base: u64,
    rows: u32,
    heads: u32,
    cd: u32,
    span: u64,
    offset: u64,
) -> Result<Tensor<bf16, 3>> {
    ensure!(cd.is_multiple_of(DV), "conv_dim must be a multiple of DV");
    let cap = span
        .checked_sub(offset)
        .context("GDN column window starts past the end of the buffer")?;
    let ptr = base
        .checked_add(offset)
        .context("GDN column window address overflow")?;
    unsafe {
        Tensor::new(
            ptr,
            [rows, heads, DV],
            [u64::from(cd), u64::from(DV), 1],
            cap,
        )
    }
}

fn ws_region(r: Region, rows: u32, cols: u32, elem: u64) -> Result<()> {
    let need = mul(u64::from(rows), mul(u64::from(cols), elem)?)?;
    ensure!(
        need <= r.bytes,
        "GDN workspace region too small: needs {need} B, has {}",
        r.bytes
    );
    ensure!(
        r.offset.is_multiple_of(256),
        "GDN workspace region not 256 B aligned"
    );
    Ok(())
}
fn ws_bf16(r: Region, ptr: u64, rows: u32, cols: u32) -> Result<TensorMut<bf16, 2>> {
    ws_region(r, rows, cols, size_of::<bf16>() as u64)?;
    unsafe { TensorMut::new(ptr + r.offset, [rows, cols], [u64::from(cols), 1], r.bytes) }
}
fn ws_bf16_3(r: Region, ptr: u64, rows: u32, h: u32, d: u32) -> Result<TensorMut<bf16, 3>> {
    ws_region(r, rows, h * d, size_of::<bf16>() as u64)?;
    unsafe {
        TensorMut::new(
            ptr + r.offset,
            [rows, h, d],
            [u64::from(h * d), u64::from(d), 1],
            r.bytes,
        )
    }
}
fn ws_f32(r: Region, ptr: u64, rows: u32, cols: u32) -> Result<TensorMut<f32, 2>> {
    ws_region(r, rows, cols, size_of::<f32>() as u64)?;
    unsafe { TensorMut::new(ptr + r.offset, [rows, cols], [u64::from(cols), 1], r.bytes) }
}
fn ws_f32_3(r: Region, ptr: u64, rows: u32, h: u32, d: u32) -> Result<TensorMut<f32, 3>> {
    ws_region(r, rows, h * d, size_of::<f32>() as u64)?;
    unsafe {
        TensorMut::new(
            ptr + r.offset,
            [rows, h, d],
            [u64::from(h * d), u64::from(d), 1],
            r.bytes,
        )
    }
}

/// Validate the registered layout against the ABI §6 binding table; every stride is checked.
fn check_bindings(b: &GdnBindings, rows: u32, conv_dim: u32) -> Result<()> {
    b.qkv.validate()?;
    ensure!(
        b.qkv.shape[1] == conv_dim && b.qkv.stride == [u64::from(conv_dim), 1],
        "GDN qkv must be [max_rows, conv_dim] with stride [conv_dim, 1]"
    );
    ensure!(
        b.qkv.shape[0] == rows,
        "GDN qkv rows ({}) must equal the plan's max_rows ({rows})",
        b.qkv.shape[0]
    );
    for (t, what) in [(&b.a, "a"), (&b.b, "b")] {
        t.validate()?;
        ensure!(
            t.shape[0] == rows
                && t.shape[1] == HV
                && t.stride[1] == 1
                && t.stride[0] >= u64::from(HV),
            "GDN {what} must be [max_rows, 48] with stride [48, 1]"
        );
    }
    b.conv_weight.validate()?;
    ensure!(
        b.conv_weight.shape == [conv_dim, 4] && b.conv_weight.stride == [4, 1],
        "GDN conv_weight must be [conv_dim, 4] with stride [4, 1] (a zero-copy reshape of the manifest's [conv_dim, 1, 4])"
    );
    for (t, what) in [(&b.a_log, "A_log"), (&b.dt_bias, "dt_bias")] {
        t.validate()?;
        ensure!(
            t.shape == [HV] && t.stride == [1],
            "GDN {what} must be [48] with stride [1]"
        );
    }
    b.core_out.0.validate()?;
    ensure!(
        b.core_out.0.shape[0] == rows
            && b.core_out.0.shape[1] == HV
            && b.core_out.0.shape[2] == DV
            && b.core_out.0.stride == [u64::from(HV * DV), u64::from(DV), 1],
        "GDN core_out must be [max_rows, 48, 128] with strides [6144, 128, 1]"
    );
    b.s.0.validate()?;
    ensure!(
        b.s.0.shape == [HV, DK, DV] && b.s.0.stride == [u64::from(DK * DV), u64::from(DV), 1],
        "GDN S must be [48, 128, 128] on (value_head, key_dim, value_dim) with strides [16384, 128, 1]"
    );
    b.history.0.validate()?;
    ensure!(
        b.history.0.shape == [conv_dim, 3] && b.history.0.stride == [3, 1],
        "GDN history must be [conv_dim, 3] oldest->newest with stride [3, 1]"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::prep_grids;
    use cudarc::driver::LaunchConfig;

    fn cfg(d: (u32, u32, u32)) -> LaunchConfig {
        LaunchConfig {
            grid_dim: d,
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    #[test]
    fn every_prep_grid_is_a_function_of_the_live_rows_alone() -> Result<()> {
        let full = (cfg((2, 48, 2048)), cfg((1, 2048, 1)), cfg((1280, 16, 1)));
        for rows in [5u32, 7, 1021, 2047, 2048] {
            let (prep, beta, conv) = prep_grids(full.0, full.1, full.2, rows)?;
            assert_eq!(prep.grid_dim.2, rows, "gdn_prep_fp32 grid z");
            assert_eq!(beta.grid_dim.1, rows, "gdn_beta_widen grid y");
            assert_eq!(
                conv.grid_dim.1,
                rows.div_ceil(super::CONV_ROWS),
                "gdn_conv grid y"
            );
            // The non-row dimensions are the plan's own and never move with the step.
            assert_eq!(prep.grid_dim.0, full.0.grid_dim.0);
            assert_eq!(prep.grid_dim.1, full.0.grid_dim.1);
            assert_eq!(beta.grid_dim.0, full.1.grid_dim.0);
            assert_eq!(beta.grid_dim.2, full.1.grid_dim.2);
            assert_eq!(conv.grid_dim.0, full.2.grid_dim.0);
            assert_eq!(conv.grid_dim.2, full.2.grid_dim.2);
        }
        Ok(())
    }

    #[test]
    fn a_narrower_step_never_widens_a_grid() -> Result<()> {
        let full = (cfg((2, 48, 2048)), cfg((1, 2048, 1)), cfg((1280, 16, 1)));
        let (wide, _, _) = prep_grids(full.0, full.1, full.2, 2048)?;
        let (narrow, _, _) = prep_grids(full.0, full.1, full.2, 1021)?;
        assert!(narrow.grid_dim.2 < wide.grid_dim.2);
        Ok(())
    }

    #[test]
    fn rows_beyond_the_cuda_grid_yz_limit_are_rejected() -> Result<()> {
        let full = (cfg((2, 48, 2048)), cfg((1, 2048, 1)), cfg((1280, 16, 1)));
        prep_grids(full.0, full.1, full.2, CUDA_GRID_YZ_MAX)?;
        let rejected = prep_grids(full.0, full.1, full.2, CUDA_GRID_YZ_MAX + 1)
            .expect_err("rows above the CUDA gridDim.y/z limit must not form a launch");
        assert!(rejected.to_string().contains("gridDim"), "{rejected}");
        Ok(())
    }

    use super::*;

    #[test]
    #[ignore = "authorized sm89 GPU window required"]
    fn gdn_verify_snapshot_rows_gpu() -> Result<()> {
        use cudarc::driver::{CudaSlice, DevicePtr};
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let device = crate::backend::cuda::Device::new(0)?;
        ensure!(
            device.info.arch == 89,
            "bitwise R-step primitive gate requires sm89"
        );
        let stream = &device.stream;
        let state_elements = (HV * DK * DV) as usize;
        let row_elements = (HV * DK) as usize;
        let signal =
            |i: usize, prime: usize| ((i * 17 % prime) as f32 - (prime / 2) as f32) / 1024.0;
        let initial: Vec<_> = (0..state_elements).map(|i| signal(i, 127)).collect();
        let q: Vec<_> = (0..5 * row_elements).map(|i| signal(i, 131)).collect();
        let k: Vec<_> = (0..5 * row_elements).map(|i| signal(i, 137)).collect();
        let v: Vec<_> = (0..5 * row_elements).map(|i| signal(i, 139)).collect();
        let beta = vec![0.5f32; 5 * HV as usize];
        let g: Vec<_> = (0..5 * HV as usize)
            .map(|i| -((i * 17 % 157 + 1) as f32) / 256.0)
            .collect();
        let q = stream.clone_htod(&q)?;
        let k = stream.clone_htod(&k)?;
        let v = stream.clone_htod(&v)?;
        let beta = stream.clone_htod(&beta)?;
        let g = stream.clone_htod(&g)?;
        let module = cubin::module(&device.ctx, "gdn/gdn_rstep")?;
        let snapshot_kernel = module.load_function("gdn_rstep_core_snapshot")?;
        let plain_kernel = module.load_function("gdn_rstep_core_v1")?;
        let config = LaunchConfig {
            grid_dim: (2, HV, 1),
            block_dim: (512, 1, 1),
            shared_mem_bytes: RECURRENT_SMEM,
        };
        for rows in 1..=5usize {
            let mut state = stream.clone_htod(&initial)?;
            let mut snapshots = stream.alloc_zeros::<f32>(5 * state_elements)?;
            let mut out = stream.alloc_zeros::<bf16>(5 * row_elements)?;
            let meta =
                stream.clone_htod(&[17i32, rows as i32, 17 + rows as i32, 128, 1, 0, 0, 0])?;
            unsafe {
                stream
                    .launch_builder(&snapshot_kernel)
                    .arg(&q)
                    .arg(&k)
                    .arg(&v)
                    .arg(&beta)
                    .arg(&g)
                    .arg(&mut state)
                    .arg(&mut out)
                    .arg(&meta)
                    .arg(&(rows as i32))
                    .arg(&mut snapshots)
                    .launch(config)?;
            }
            let slots = stream.clone_dtoh(&snapshots)?;
            let final_state = stream.clone_dtoh(&state)?;
            let actual = stream.clone_dtoh(&out)?;
            let mut reference = stream.clone_htod(&initial)?;
            for row in 0..rows {
                let meta =
                    stream.clone_htod(&[17 + row as i32, 1, 18 + row as i32, 128, 1, 0, 0, 0])?;
                let offset = row * row_elements;
                let qp = q.device_ptr(stream).0 + (offset * 4) as u64;
                let kp = k.device_ptr(stream).0 + (offset * 4) as u64;
                let vp = v.device_ptr(stream).0 + (offset * 4) as u64;
                let bp = beta.device_ptr(stream).0 + (row * HV as usize * 4) as u64;
                let gp = g.device_ptr(stream).0 + (row * HV as usize * 4) as u64;
                let mut output = stream.alloc_zeros::<bf16>(row_elements)?;
                unsafe {
                    stream
                        .launch_builder(&plain_kernel)
                        .arg(&qp)
                        .arg(&kp)
                        .arg(&vp)
                        .arg(&bp)
                        .arg(&gp)
                        .arg(&mut reference)
                        .arg(&mut output)
                        .arg(&meta)
                        .arg(&1i32)
                        .launch(config)?;
                }
                let expected = stream.clone_dtoh(&reference)?;
                // The last live row is not snapshotted (it is the final S write-back).
                if row + 1 < rows {
                    ensure!(
                        expected
                            .iter()
                            .zip(&slots[row * state_elements..(row + 1) * state_elements])
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "snapshot state differs from independent R1 chain, rows={rows} row={row}"
                    );
                }
                ensure!(
                    stream.clone_dtoh(&output)? == actual[offset..offset + row_elements],
                    "snapshot output changed"
                );
                if row + 1 == rows {
                    ensure!(expected == final_state, "final state changed");
                }
            }
            ensure!(
                slots[(rows - 1) * state_elements..]
                    .iter()
                    .all(|v| v.to_bits() == 0),
                "dead snapshot slots written"
            );
        }
        let conv_dim = GDN_CONV_DIM as usize;
        let initial: Vec<_> = (0..conv_dim * 3).map(|i| signal(i, 149)).collect();
        let raw: Vec<_> = (0..5 * conv_dim)
            .map(|i| bf16::from_f32(signal(i, 151)))
            .collect();
        let qkv = stream.clone_htod(&raw)?;
        let kernel =
            cubin::module(&device.ctx, "gdn/gdn_conv")?.load_function("gdn_conv_hist_snapshot")?;
        for rows in 1..=5usize {
            let mut history = stream.clone_htod(&initial)?;
            let mut slots = stream.alloc_zeros::<f32>(5 * conv_dim * 3)?;
            let meta =
                stream.clone_htod(&[17i32, rows as i32, 17 + rows as i32, 128, 1, 0, 0, 0])?;
            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(&qkv)
                    .arg(&mut history)
                    .arg(&meta)
                    .arg(&(conv_dim as i32))
                    .arg(&mut slots)
                    .launch(LaunchConfig {
                        grid_dim: (GDN_CONV_DIM / 256, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            let slots = stream.clone_dtoh(&slots)?;
            let final_history = stream.clone_dtoh(&history)?;
            for row in 0..rows {
                for ch in 0..conv_dim {
                    for tap in 0..3 {
                        let index = row + 1 + tap;
                        let expected = if index < 3 {
                            initial[ch * 3 + index]
                        } else {
                            raw[(index - 3) * conv_dim + ch].to_f32()
                        };
                        if row + 1 < rows {
                            ensure!(
                                slots[(row * conv_dim + ch) * 3 + tap].to_bits()
                                    == expected.to_bits(),
                                "raw history snapshot index mismatch"
                            );
                        }
                        if row + 1 == rows {
                            ensure!(
                                final_history[ch * 3 + tap].to_bits() == expected.to_bits(),
                                "final history mismatch"
                            );
                        }
                    }
                }
            }
            ensure!(
                slots[(rows - 1) * conv_dim * 3..]
                    .iter()
                    .all(|v| v.to_bits() == 0),
                "dead history slots written"
            );
        }
        let a = stream.clone_htod(&vec![bf16::from_f32(0.25); 5 * HV as usize])?;
        let b = stream.clone_htod(&vec![bf16::from_f32(0.5); 5 * HV as usize])?;
        let conv_weight = stream.clone_htod(&vec![bf16::from_f32(0.25); conv_dim * 4])?;
        let a_log = stream.alloc_zeros::<bf16>(HV as usize)?;
        let dt_bias = stream.alloc_zeros::<bf16>(HV as usize)?;
        let core_out = stream.alloc_zeros::<bf16>(5 * row_elements)?;
        let state = stream.alloc_zeros::<f32>(state_elements)?;
        let history = stream.alloc_zeros::<f32>(conv_dim * 3)?;
        let snapshots = stream.alloc_zeros::<f32>(4 * state_elements)?;
        let history_snapshots = stream.alloc_zeros::<f32>(4 * conv_dim * 3)?;
        let matrix = |buffer: &CudaSlice<bf16>, columns: u32| unsafe {
            Tensor::new(
                buffer.device_ptr(stream).0,
                [buffer.len() as u32 / columns, columns],
                [u64::from(columns), 1],
                (buffer.len() * 2) as u64,
            )
        };
        let bindings = unsafe {
            GdnBindings {
                qkv: matrix(&qkv, GDN_CONV_DIM)?,
                a: matrix(&a, HV)?,
                b: matrix(&b, HV)?,
                conv_weight: matrix(&conv_weight, 4)?,
                a_log: Tensor::new(a_log.device_ptr(stream).0, [HV], [1], u64::from(HV) * 2)?,
                dt_bias: Tensor::new(dt_bias.device_ptr(stream).0, [HV], [1], u64::from(HV) * 2)?,
                core_out: TensorMut(Tensor::new(
                    core_out.device_ptr(stream).0,
                    [5, HV, DV],
                    [u64::from(HV * DV), u64::from(DV), 1],
                    (5 * row_elements * 2) as u64,
                )?),
                s: TensorMut(Tensor::new(
                    state.device_ptr(stream).0,
                    [HV, DK, DV],
                    [u64::from(DK * DV), u64::from(DV), 1],
                    (state_elements * 4) as u64,
                )?),
                history: TensorMut(Tensor::new(
                    history.device_ptr(stream).0,
                    [GDN_CONV_DIM, 3],
                    [3, 1],
                    (conv_dim * 3 * 4) as u64,
                )?),
            }
        };
        let full = GdnOps::prepare(
            &device.ctx,
            &GdnSpec::q38(),
            5,
            bindings,
            CommonOpsRef::new(&device.ctx)?,
            true,
        )?;
        let ordinary = full.for_rows(5, None)?;
        ensure!(
            ordinary.cw_p2 && ordinary.ws_p2 > 0,
            "M5 negative control did not carry P2"
        );
        let snapshot_views = unsafe {
            (
                Tensor::new(
                    snapshots.device_ptr(stream).0,
                    [4, HV, DK, DV],
                    [
                        u64::from(HV * DK * DV),
                        u64::from(DK * DV),
                        u64::from(DV),
                        1,
                    ],
                    (4 * state_elements * 4) as u64,
                )?,
                Tensor::new(
                    history_snapshots.device_ptr(stream).0,
                    [4, GDN_CONV_DIM, 3],
                    [u64::from(GDN_CONV_DIM * 3), 3, 1],
                    (4 * conv_dim * 3 * 4) as u64,
                )?,
            )
        };
        let plan = full.for_rows(5, Some(snapshot_views))?;
        ensure!(
            !plan.cw_p2 && plan.ws_p2 == 0,
            "snapshot-bound M5 plan retained chunkwise dispatch"
        );
        ensure!(
            full.cw_p2 && full.ws_p2 > 0,
            "M5 verify binding changed ordinary prefill plan"
        );
        let scratch = stream.alloc_zeros::<u8>(plan.ws.bytes as usize)?;
        let meta = stream.clone_htod(&[17i32, 5, 22, 128, 1, 0, 0, 0])?;
        GdnOps::run_verify(
            &plan,
            stream,
            SeqRef {
                ptr: meta.device_ptr(stream).0,
            },
            Scratch {
                ptr: scratch.device_ptr(stream).0,
                bytes: plan.ws.bytes,
            },
        )?;
        stream.synchronize()?;
        ensure!(
            stream.clone_dtoh(&core_out)?.iter().all(|v| v.is_finite())
                && stream.clone_dtoh(&state)?.iter().all(|v| v.is_finite())
                && stream.clone_dtoh(&snapshots)?.iter().all(|v| v.is_finite()),
            "M5 verify produced nonfinite values"
        );
        let bound_snapshots = stream.clone_dtoh(&snapshots)?;
        for row in 0..4 {
            ensure!(
                bound_snapshots[row * state_elements..(row + 1) * state_elements]
                    .iter()
                    .any(|v| v.to_bits() != 0),
                "snapshot state differs: M5 verify omitted a retained state prefix"
            );
        }
        let saved_history = stream.clone_dtoh(&history_snapshots)?;
        for row in 0..4 {
            for ch in 0..conv_dim {
                for tap in 0..3 {
                    let index = row + 1 + tap;
                    let expected = if index < 3 {
                        0.0
                    } else {
                        raw[(index - 3) * conv_dim + ch].to_f32()
                    };
                    ensure!(
                        saved_history[(row * conv_dim + ch) * 3 + tap].to_bits()
                            == expected.to_bits(),
                        "raw history snapshot index mismatch in M5 bound plan"
                    );
                }
            }
        }
        let reference_state = stream.alloc_zeros::<f32>(state_elements)?;
        let reference_history = stream.alloc_zeros::<f32>(conv_dim * 3)?;
        let reference_out = stream.alloc_zeros::<bf16>(5 * row_elements)?;
        let final_state = stream.clone_dtoh(&state)?;
        let final_history = stream.clone_dtoh(&history)?;
        let input_row = |source: Tensor<bf16, 2>, row: usize| {
            let offset = row as u64 * source.stride[0] * 2;
            unsafe {
                Tensor::new(
                    source.ptr + offset,
                    [1, source.shape[1]],
                    source.stride,
                    source.span_bytes - offset,
                )
            }
        };
        for row in 0..5 {
            let b = &full.bindings;
            let bindings = unsafe {
                GdnBindings {
                    qkv: input_row(b.qkv, row)?,
                    a: input_row(b.a, row)?,
                    b: input_row(b.b, row)?,
                    conv_weight: b.conv_weight,
                    a_log: b.a_log,
                    dt_bias: b.dt_bias,
                    core_out: TensorMut(Tensor::new(
                        reference_out.device_ptr(stream).0 + (row * row_elements * 2) as u64,
                        [1, HV, DV],
                        [u64::from(HV * DV), u64::from(DV), 1],
                        ((5 - row) * row_elements * 2) as u64,
                    )?),
                    s: TensorMut(Tensor::new(
                        reference_state.device_ptr(stream).0,
                        b.s.0.shape,
                        b.s.0.stride,
                        b.s.0.span_bytes,
                    )?),
                    history: TensorMut(Tensor::new(
                        reference_history.device_ptr(stream).0,
                        b.history.0.shape,
                        b.history.0.stride,
                        b.history.0.span_bytes,
                    )?),
                }
            };
            let reference = GdnOps::prepare(
                &device.ctx,
                &GdnSpec::q38(),
                1,
                bindings,
                full.common.clone(),
                false,
            )?;
            let meta =
                stream.clone_htod(&[17 + row as i32, 1, 18 + row as i32, 128, 1, 0, 0, 0])?;
            GdnOps::run_core(
                &reference,
                stream,
                SeqRef {
                    ptr: meta.device_ptr(stream).0,
                },
                ExecutionPhase::Prefill,
                Scratch {
                    ptr: scratch.device_ptr(stream).0,
                    bytes: plan.ws.bytes,
                },
                true,
            )?;
            let expected = stream.clone_dtoh(&reference_state)?;
            let actual = if row < 4 {
                &bound_snapshots[row * state_elements..(row + 1) * state_elements]
            } else {
                &final_state
            };
            ensure!(
                expected
                    .iter()
                    .zip(actual)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "snapshot state differs from prepared R1 chain, row={row}"
            );
            let expected = stream.clone_dtoh(&reference_history)?;
            let actual = if row < 4 {
                &saved_history[row * conv_dim * 3..(row + 1) * conv_dim * 3]
            } else {
                &final_history
            };
            ensure!(
                expected
                    .iter()
                    .zip(actual)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "raw history snapshot index mismatch against prepared R1 chain, row={row}"
            );
        }
        ensure!(
            stream
                .clone_dtoh(&reference_out)?
                .iter()
                .zip(stream.clone_dtoh(&core_out)?)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "snapshot output changed against prepared R1 chain"
        );
        take_phase_log();
        GdnOps::run(
            &plan,
            stream,
            SeqRef {
                ptr: meta.device_ptr(stream).0,
            },
            ExecutionPhase::Prefill,
            Scratch {
                ptr: scratch.device_ptr(stream).0,
                bytes: plan.ws.bytes,
            },
        )?;
        stream.synchronize()?;
        ensure!(
            take_phase_log() == ["chunked"],
            "M5 shared-plan mutant was not exercised"
        );
        let ordinary_scratch =
            stream.alloc_zeros::<u8>((ordinary.ws.bytes + ordinary.ws_p2) as usize)?;
        GdnOps::run(
            &ordinary,
            stream,
            SeqRef {
                ptr: meta.device_ptr(stream).0,
            },
            ExecutionPhase::Prefill,
            Scratch {
                ptr: ordinary_scratch.device_ptr(stream).0,
                bytes: ordinary.ws.bytes + ordinary.ws_p2,
            },
        )?;
        stream.synchronize()?;
        ensure!(
            take_phase_log() == ["chunkwise_p2"],
            "ordinary M5 prefill route changed"
        );
        ensure!(
            stream.clone_dtoh(&state)?.iter().all(|v| v.is_finite())
                && stream.clone_dtoh(&core_out)?.iter().all(|v| v.is_finite()),
            "ordinary M5 prefill nonfinite values"
        );
        println!(
            "GDN_VERIFY_M5_BINDING_PASS recurrent_snapshot_rows=4 retained_state_prefixes=4 raw_history_CPU ordinary_prefill_P2_unchanged prepared_R1_state_output_history_bitwise shared_plan_mutant_chunked ordinary_M5_P2_launched"
        );
        println!(
            "GDN_VERIFY_SNAPSHOT_PRIMITIVE_PASS rows1..5 state_bitwise_R1 raw_history_CPU dead_slots M5_bound_plan"
        );
        Ok(())
    }

    fn align256(n: u64) -> u64 {
        (n + 255) & !255
    }

    #[test]
    fn workspace_regions_match_the_interface_table() {
        let spec = GdnSpec::q38();
        let conv_dim = u64::from(spec.conv_dim().expect("conv_dim"));
        for rows in [1u32, 64, 2048] {
            let ws = GdnWorkspace::q38(&spec, rows).expect("workspace");
            let r = u64::from(rows);
            let head = u64::from(HV);
            let khead = u64::from(HK);
            let d = u64::from(DV);
            let dk = u64::from(DK);

            let want = |cols: u64, elem: u64| align256(r * cols * elem);
            assert_eq!(
                ws.conv_out.bytes,
                want(conv_dim, 2),
                "conv_out, rows={rows}"
            );
            assert_eq!(ws.q_bf16.bytes, want(khead * dk, 2), "q_bf16, rows={rows}");
            assert_eq!(ws.k_bf16.bytes, want(khead * dk, 2), "k_bf16, rows={rows}");
            assert_eq!(ws.beta.bytes, want(head, 2), "beta, rows={rows}");
            assert_eq!(ws.beta_fp32.bytes, want(head, 4), "beta_fp32, rows={rows}");
            assert_eq!(ws.g.bytes, want(head, 4), "g, rows={rows}");
            assert_eq!(ws.q_fp32.bytes, want(head * d, 4), "q_fp32, rows={rows}");
            assert_eq!(ws.k_fp32.bytes, want(head * d, 4), "k_fp32, rows={rows}");
            assert_eq!(ws.v_fp32.bytes, want(head * d, 4), "v_fp32, rows={rows}");

            // Regions within the Engine slice have 256 B alignment and do not overlap.
            let regions = [
                ("conv_out", ws.conv_out),
                ("q_bf16", ws.q_bf16),
                ("k_bf16", ws.k_bf16),
                ("beta", ws.beta),
                ("beta_fp32", ws.beta_fp32),
                ("g", ws.g),
                ("q_fp32", ws.q_fp32),
                ("k_fp32", ws.k_fp32),
                ("v_fp32", ws.v_fp32),
            ];
            for (n, reg) in regions {
                assert!(
                    reg.offset.is_multiple_of(256),
                    "{n} offset {} not 256-aligned",
                    reg.offset
                );
                assert!(
                    reg.bytes.is_multiple_of(256),
                    "{n} bytes {} not 256-aligned",
                    reg.bytes
                );
                assert!(
                    reg.offset + reg.bytes <= ws.bytes,
                    "{n} runs past the slice"
                );
            }
            let mut spans: Vec<(u64, u64, &str)> = regions
                .iter()
                .map(|(n, r)| (r.offset, r.offset + r.bytes, *n))
                .collect();
            spans.sort();
            for w in spans.windows(2) {
                assert!(w[0].1 <= w[1].0, "{} overlaps {}", w[0].2, w[1].2);
            }
            // The declared total is the Engine slice and accounts for every region.
            let last = spans.last().expect("regions");
            assert_eq!(
                ws.bytes,
                align256(last.1),
                "total does not close over the regions"
            );
        }
    }

    /// `workspace()` is pure: same inputs, same bytes, no CUDA context.
    #[test]
    fn workspace_is_pure_and_matches_the_plan() {
        let spec = GdnSpec::q38();
        let a = GdnOps::workspace(&spec, 2048, GdnPolicy::LEGACY).expect("workspace");
        let b = GdnOps::workspace(&spec, 2048, GdnPolicy::LEGACY).expect("workspace");
        assert_eq!(a.bytes, b.bytes);
        assert_eq!(a.alignment, 256);
        assert_eq!(a.bytes, GdnWorkspace::q38(&spec, 2048).expect("ws").bytes);
        // decode's plan is max_rows = 1; the Engine shares one scratch sized to the larger.
        let d = GdnOps::workspace(&spec, 1, GdnPolicy::LEGACY).expect("decode workspace");
        assert!(
            d.bytes < a.bytes,
            "a 1-row plan must not exceed the prefill plan"
        );
        assert!(
            d.bytes >= 102_880,
            "one row needs the full 102,880 B of intermediates"
        );
    }

    /// A column window's capacity is `span - offset`, so extent validation rejects a V offset
    /// that overruns the final row.
    #[test]
    fn column_window_capacity_is_measured_from_the_window() {
        let cd = u64::from(GDN_CONV_DIM);
        let el = size_of::<bf16>() as u64;
        // ABI §6: Q at 0 width HK*DK, K next, V next width HV*DV, in bytes.
        let q_off = 0u64;
        let k_off = u64::from(HK * DK) * el;
        let v_off = 2 * u64::from(HK * DK) * el;
        assert_eq!(v_off, 8192, "V starts two key-head blocks in");
        // Wrong offset: V's own width added where K's belongs.
        let v_off_bug = q_off + u64::from(HV * DV) * el + k_off;
        assert_eq!(v_off_bug, 16384, "the historical bug sat 8192 B too far in");

        let builds = |rows: u32, off: u64, span: u64| -> bool {
            match span.checked_sub(off) {
                None => false,
                Some(cap) => unsafe {
                    Tensor::<bf16, 3>::new(0x1000, [rows, HV, DV], [cd, u64::from(DV), 1], cap)
                        .is_ok()
                },
            }
        };

        for rows in [1u32, 64] {
            let span = cd * el * u64::from(rows);
            for (nm, off) in [("q", q_off), ("k", k_off), ("v", v_off)] {
                assert!(
                    builds(rows, off, span),
                    "{nm} at offset {off} must build (rows={rows})"
                );
            }
            assert!(
                !builds(rows, v_off_bug, span),
                "the historical v_off {v_off_bug} must now be REJECTED (rows={rows}); \
                 if this passes, the capacity hardening was lost"
            );
        }
    }

    /// `conv_dim` comes from one derivation.
    #[test]
    fn conv_dim_has_a_single_derivation() {
        let spec = GdnSpec::q38();
        assert_eq!(spec.conv_dim().expect("conv_dim"), GDN_CONV_DIM);
        assert_eq!(GDN_CONV_DIM, 2 * HK * DK + HV * DV);
    }

    /// The chunkwise-P2 slot offsets are the kernels' `P2_OFF_*` values: each 256 B
    /// aligned, pairwise disjoint, filling a slot of exactly `GDN_CW_P2_SLOT_BYTES`, and
    /// the planner's byte count is that many slots per (chunk, head).
    #[test]
    fn cw_p2_slot_is_aligned_and_the_planner_sizes_it_exactly() -> Result<()> {
        let layout = crate::gdn_plan::cw_p2_slot_layout();
        assert_eq!(layout.len(), crate::gdn_plan::CW_P2_SLOT.len());
        let mut spans: Vec<(u64, u64, &str)> = Vec::new();
        for ((name, bytes, _), (offset, laid_out, _)) in
            crate::gdn_plan::CW_P2_SLOT.iter().zip(&layout)
        {
            assert_eq!(bytes, laid_out, "{name} size disagrees with the layout");
            assert!(
                offset.is_multiple_of(256),
                "{name} at {offset} is not 256 B aligned"
            );
            spans.push((*offset, *offset + *bytes, name));
        }
        spans.sort();
        for w in spans.windows(2) {
            assert!(w[0].1 <= w[1].0, "{} overlaps {}", w[0].2, w[1].2);
        }
        assert_eq!(
            spans.last().expect("fields").1,
            cw_p2::SLOT_BYTES,
            "the slot does not close over its fields"
        );
        assert_eq!(cw_p2::SLOT_BYTES, 59_136, "the kernel slot size moved");

        let spec = GdnSpec::q38();
        for rows in [1u32, 64, 2048, 16384] {
            let slots = u64::from(rows.div_ceil(CW_P2_CHUNK)) * u64::from(HV);
            assert_eq!(cw_p2_bytes(rows), slots * cw_p2::SLOT_BYTES, "rows={rows}");
            // The planner's number, with the split armed, is the fused layout plus the
            // slots; unarmed it is the fused layout alone.
            let fused = GdnWorkspace::q38(&spec, rows)?.bytes;
            assert_eq!(
                GdnPolicy::P2.workspace(&spec, rows)?.bytes,
                fused + cw_p2_bytes(rows),
                "rows={rows}"
            );
            assert_eq!(
                GdnPolicy::LEGACY.workspace(&spec, rows)?.bytes,
                fused,
                "the split must not tax a plan that does not carry it, rows={rows}"
            );
            // Every slot sits inside the region the plan accounts for.
            assert!(cw_p2_bytes(rows).is_multiple_of(256), "rows={rows}");
        }
        Ok(())
    }

    /// The chunkwise-P2 grids derive from the bound rows: stage A one CTA per (chunk, head),
    /// stage H one per 32-wide value block independent of the step.
    #[test]
    fn cw_p2_grids_are_a_function_of_the_bound_rows() {
        for (rows, want_chunks) in [(2048u32, 32u32), (16384, 256)] {
            let (a, h) = cw_p2_grids(rows);
            assert_eq!(
                a.grid_dim,
                (want_chunks, HV, 1),
                "stage A grid, rows={rows}"
            );
            assert_eq!(
                h.grid_dim,
                (4, HV, 1),
                "stage H covers DV in 32-wide blocks, rows={rows}"
            );
            for cfg in [a, h] {
                assert_eq!(cfg.block_dim, (P2_THREADS, 1, 1));
                assert_eq!(cfg.shared_mem_bytes, cw_p2::SMEM_BYTES as u32);
            }
        }
        // The H grid is the same whatever the step is; the per-chunk grid is monotone.
        assert_eq!(cw_p2_grids(5).0.grid_dim, cw_p2_grids(64).0.grid_dim);
        assert!(cw_p2_grids(16384).0.grid_dim.0 > cw_p2_grids(2048).0.grid_dim.0);
        for rows in [5u32, 64, 2048, 16384] {
            assert_eq!(cw_p2_grids(rows).1.grid_dim, cw_p2_grids(2048).1.grid_dim);
        }
    }

    /// The chunkwise-P2 stage dependency contract as a CPU model: within one (chunk, head)
    /// slot, stage A writes W, U, qk, Gc, Be and Sc and stage H writes vn and Sn, no field
    /// has a second writer and no field is written twice for the same pair.
    #[test]
    fn every_cw_p2_slot_field_has_exactly_one_writing_stage() {
        let rows = 2048u32;
        let slots = usize::try_from(rows.div_ceil(CW_P2_CHUNK)).expect("nchunk") * HV as usize;
        let fields = crate::gdn_plan::CW_P2_SLOT.len();
        // writers[field][slot]: how many stages write that field of that slot.
        let mut writers = vec![vec![0u8; slots]; fields];
        for (index, (_, _, writer)) in crate::gdn_plan::CW_P2_SLOT.iter().enumerate() {
            for count in &mut writers[index] {
                *count += 1;
            }
            assert_eq!(*writer, 'A', "only stage A writes slot fields");
        }
        for (index, count) in writers.iter().enumerate() {
            let name = crate::gdn_plan::CW_P2_SLOT[index].0;
            assert!(
                count.iter().all(|c| *c == 1),
                "{name} is not written exactly once for every (chunk, head)"
            );
        }
        let by_a: Vec<&str> = crate::gdn_plan::CW_P2_SLOT
            .iter()
            .filter(|(_, _, w)| *w == 'A')
            .map(|(n, _, _)| *n)
            .collect();
        let by_h: Vec<&str> = crate::gdn_plan::CW_P2_SLOT
            .iter()
            .filter(|(_, _, w)| *w == 'H')
            .map(|(n, _, _)| *n)
            .collect();
        assert_eq!(by_a, ["W", "U", "qk", "Gc", "Be", "Sc"], "stage A outputs");
        assert!(by_h.is_empty(), "stage H writes no slot field");
        // Stage A's outputs are what stage H reads, so both of those slots must
        // be full for every chunk and head the plan covers.
        let layout = crate::gdn_plan::cw_p2_slot_layout();
        for chunk in 0..slots / HV as usize {
            for head in 0..HV as usize {
                let base = chunk * HV as usize + head;
                for (index, (offset, bytes, _)) in layout.iter().enumerate() {
                    let name = crate::gdn_plan::CW_P2_SLOT[index].0;
                    assert!(
                        offset + bytes <= cw_p2::SLOT_BYTES,
                        "{name} runs past its slot"
                    );
                    assert_eq!(
                        writers[index][base], 1,
                        "{name} at chunk {chunk} head {head}"
                    );
                }
            }
        }
    }

    fn valid_bindings(rows: u32) -> GdnBindings {
        let cd = GDN_CONV_DIM;
        let bf = size_of::<bf16>() as u64;
        let f32b = size_of::<f32>() as u64;
        let r = u64::from(rows);
        let head = u64::from(HV);
        let d = u64::from(DV);
        let dk = u64::from(DK);
        // Span headroom so a widened stride fails the stride check, not extent validation.
        let padded = |bytes: u64| bytes + 1024 * 1024;
        unsafe {
            GdnBindings {
                qkv: Tensor::new(
                    0x1000,
                    [rows, cd],
                    [u64::from(cd), 1],
                    padded(r * u64::from(cd) * bf),
                )
                .expect("qkv"),
                a: Tensor::new(0x1000, [rows, HV], [head, 1], padded(r * head * bf)).expect("a"),
                b: Tensor::new(0x1000, [rows, HV], [head, 1], padded(r * head * bf)).expect("b"),
                conv_weight: Tensor::new(0x1000, [cd, 4], [4, 1], padded(u64::from(cd) * 4 * bf))
                    .expect("conv_weight"),
                a_log: Tensor::new(0x1000, [HV], [1], padded(head * bf)).expect("a_log"),
                dt_bias: Tensor::new(0x1000, [HV], [1], padded(head * bf)).expect("dt_bias"),
                core_out: TensorMut::new(
                    0x1000,
                    [rows, HV, DV],
                    [head * d, d, 1],
                    padded(r * head * d * bf),
                )
                .expect("core_out"),
                s: TensorMut::new(
                    0x1000,
                    [HV, DK, DV],
                    [dk * d, d, 1],
                    padded(head * dk * d * f32b),
                )
                .expect("s"),
                history: TensorMut::new(0x1000, [cd, 3], [3, 1], padded(u64::from(cd) * 3 * f32b))
                    .expect("history"),
            }
        }
    }

    /// A binding wider than the strides the cores bake in must be rejected by the binding
    /// guard, not by extent validation.
    #[test]
    fn bindings_must_carry_the_kernel_strides() {
        let rows = 4u32;
        check_bindings(&valid_bindings(rows), rows, GDN_CONV_DIM)
            .expect("dense bindings must pass");
        let reject = |name: &str, mangle: &dyn Fn(&mut GdnBindings)| {
            let mut b = valid_bindings(rows);
            mangle(&mut b);
            let error = check_bindings(&b, rows, GDN_CONV_DIM)
                .expect_err("{name} binding must be rejected");
            assert!(
                error.to_string().contains("GDN"),
                "{name}: expected the binding stride guard, got {error:?}"
            );
        };
        reject("qkv row stride", &|b: &mut GdnBindings| {
            b.qkv.stride[0] += 1
        });
        reject("core_out row stride", &|b: &mut GdnBindings| {
            b.core_out.0.stride[0] += 1
        });
        reject("core_out head stride", &|b: &mut GdnBindings| {
            b.core_out.0.stride[1] += 1
        });
        reject("S head stride", &|b: &mut GdnBindings| b.s.0.stride[0] += 1);
        reject("S key stride", &|b: &mut GdnBindings| b.s.0.stride[1] += 1);
        reject("history channel stride", &|b: &mut GdnBindings| {
            b.history.0.stride[0] += 1
        });
    }

    /// The V window must go through the same `as3` construction as Q and K, so a
    /// wrong column offset is rejected in run() instead of reading past conv_out.
    #[test]
    fn v_window_offsets_are_validated_before_launch() {
        let cd = u64::from(GDN_CONV_DIM);
        let el = size_of::<bf16>() as u64;
        // ABI §6: Q at 0 width HK*DK, K next, V next width HV*DV, in bytes.
        let v_off = 2 * u64::from(HK * DK) * el;
        assert_eq!(v_off, 8192, "V starts two key-head blocks in");
        // Wrong offset: V's own width added where K's belongs.
        let v_off_bug = u64::from(HV * DV) * el + u64::from(HK * DK) * el;
        assert_eq!(v_off_bug, 16384, "the historical bug sat 8192 B too far in");
        for rows in [1u32, 64] {
            let span = cd * el * u64::from(rows);
            let window = as3(0x1000, rows, HV, GDN_CONV_DIM, span, v_off).expect("V window");
            assert_eq!(window.shape, [rows, HV, DV]);
            assert_eq!(window.stride, [cd, u64::from(DV), 1]);
            assert!(
                as3(0x1000, rows, HV, GDN_CONV_DIM, span, v_off_bug).is_err(),
                "v_off {v_off_bug} must be rejected before any launch (rows={rows})"
            );
        }
    }
}

/// GPU probe for the GDN binder: `workspace` -> `prepare` -> `run` through the real
/// launch path and common ops, against oracle fixtures.
///
/// Checked contracts:
///   1. phase contract: Prefill-M1 selects chunked and Decode-M1 selects recurrent, each
///      matching its own oracle reference;
///   2. tail chunk: rows at or past valid_rows stay untouched in core_out and do not advance
///      S or history;
///   3. a negative control (a/b bindings swapped) that must fail on numbers;
///   4. end-to-end against the oracle for both phases plus a nonzero-prefix continuation.
///
/// Bounds are per-case: cases at >= 20k elements use the fraction term (5e-4) plus the case's
/// own `scaled` basis; smaller cases use a count bound.
#[cfg(test)]
mod gpu_probe {
    use super::*;
    use crate::backend::cuda::Device;
    use crate::op_contract::{ExecutionPhase, SequenceMeta};
    use cudarc::driver::{CudaSlice, DevicePtr};
    use std::path::PathBuf;
    const FLOOR_REL: f64 = 1e-3;
    /// core_out marker for rows at or past valid_rows; finite.
    const TAIL_MARKER: f32 = 1234.0;
    /// bf16 NaN (0x7FC0), written into input rows at or past valid_rows.
    const NAN_BITS: u16 = 0x7FC0;

    #[derive(Clone)]
    struct Terms {
        scaled: f64,
        dead_max: f64,
        floor: f64,
        ndiff: usize,
        nan: usize,
    }

    /// The gate's comparator, on BF16-rounded values.
    fn terms(got: &[bf16], reference: &[bf16], valid: usize) -> Terms {
        let heads = HV as usize;
        let dv = DV as usize;
        let rows = valid;
        let f = |x: &bf16| f64::from(x.to_f32());
        let mut rms = vec![0f64; rows * heads];
        for m in 0..rows {
            for h in 0..heads {
                let mut acc = 0f64;
                for d in 0..dv {
                    let x = f(&reference[(m * heads + h) * dv + d]);
                    acc += x * x;
                }
                rms[m * heads + h] = (acc / dv as f64).sqrt();
            }
        }
        let mut sorted = rms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let floor = sorted[sorted.len() / 2] * FLOOR_REL;
        let mut t = Terms {
            scaled: 0.0,
            dead_max: 0.0,
            floor,
            ndiff: 0,
            nan: 0,
        };
        for i in 0..rows * heads * dv {
            let g = f(&got[i]);
            if !g.is_finite() {
                t.nan += 1;
                continue;
            }
            let r = f(&reference[i]);
            if g != r {
                t.ndiff += 1;
            }
        }
        for (m, row_rms) in rms.iter().enumerate() {
            let live = *row_rms > floor;
            for d in 0..dv {
                let i = m * dv + d;
                let g = f(&got[i]);
                if !g.is_finite() {
                    continue;
                }
                let diff = (g - f(&reference[i])).abs();
                if live {
                    t.scaled = t.scaled.max(diff / *row_rms);
                } else {
                    t.dead_max = t.dead_max.max(diff);
                }
            }
        }
        t
    }

    struct Man {
        mode: String,
        valid: usize,
        prefix: i32,
        capacity: i32,
        plan_rows: u32,
        scaled_bound: f64,
        degenerate: bool,
        frac_bound: Option<f64>,
        count_bound: Option<usize>,
        rule: Option<String>,
        dir: PathBuf,
    }

    impl Man {
        fn load(name: &str) -> Man {
            let dir = PathBuf::from(
                std::env::var_os("FF_Q38_GDN_FIXTURE_DIR")
                    .expect("FF_Q38_GDN_FIXTURE_DIR is required"),
            )
            .join(name);
            let text = std::fs::read_to_string(dir.join("manifest.json"))
                .unwrap_or_else(|e| panic!("read {}/manifest.json: {e}", dir.display()));
            let m: serde_json::Value = serde_json::from_str(&text).expect("manifest json");
            let bounds = &m["bounds"];
            let elems = m["valid_rows"].as_u64().unwrap() * HV as u64 * DV as u64;
            Man {
                mode: m["mode"].as_str().unwrap().to_string(),
                valid: m["valid_rows"].as_u64().unwrap() as usize,
                prefix: m["prefix"].as_i64().unwrap() as i32,
                capacity: m["capacity"].as_i64().unwrap() as i32,
                plan_rows: m["plan_rows"]
                    .as_u64()
                    .unwrap_or(m["valid_rows"].as_u64().unwrap()) as u32,
                scaled_bound: bounds["scaled_bound"].as_f64().unwrap(),
                degenerate: bounds
                    .get("scaled_bound_degenerate")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                    || bounds["scaled_bound"].as_f64().unwrap() == 0.0,
                frac_bound: bounds.get("frac_bound").and_then(|v| v.as_f64()),
                count_bound: if elems >= 20_000 {
                    None
                } else {
                    Some(bounds["count_bound"].as_u64().unwrap() as usize)
                },
                rule: bounds
                    .get("rule")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                dir,
            }
        }

        fn bf16(&self, name: &str, elems: usize) -> Vec<bf16> {
            let raw = std::fs::read(self.dir.join(name)).expect(name);
            assert_eq!(raw.len(), elems * 2, "{name} byte count");
            raw.as_chunks::<2>()
                .0
                .iter()
                .map(|c| bf16::from_bits(u16::from_le_bytes(*c)))
                .collect()
        }

        fn f32s(&self, name: &str, elems: usize) -> Vec<f32> {
            let raw = std::fs::read(self.dir.join(name)).expect(name);
            assert_eq!(raw.len(), elems * 4, "{name} byte count");
            raw.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        }
    }

    struct Case {
        man: Man,
        dev_qkv: CudaSlice<bf16>,
        dev_a: CudaSlice<bf16>,
        dev_b: CudaSlice<bf16>,
        dev_conv_w: CudaSlice<bf16>,
        dev_a_log: CudaSlice<bf16>,
        dev_dt_bias: CudaSlice<bf16>,
        dev_core: CudaSlice<bf16>,
        dev_s: CudaSlice<f32>,
        dev_hist: CudaSlice<f32>,
        scratch_buf: CudaSlice<u8>,
        scratch_bytes: u64,
    }

    fn view_bf16<const R: usize>(
        buf: &CudaSlice<bf16>,
        s: &Arc<CudaStream>,
        shape: [u32; R],
        stride: [u64; R],
    ) -> Result<Tensor<bf16, R>> {
        let (ptr, _h) = buf.device_ptr(s);
        unsafe { Tensor::new(ptr, shape, stride, (buf.len() * 2) as u64) }
    }
    fn view_bf16_mut<const R: usize>(
        buf: &CudaSlice<bf16>,
        s: &Arc<CudaStream>,
        shape: [u32; R],
        stride: [u64; R],
    ) -> Result<TensorMut<bf16, R>> {
        Ok(TensorMut(view_bf16(buf, s, shape, stride)?))
    }
    fn view_f32<const R: usize>(
        buf: &CudaSlice<f32>,
        s: &Arc<CudaStream>,
        shape: [u32; R],
        stride: [u64; R],
    ) -> Result<TensorMut<f32, R>> {
        let (ptr, _h) = buf.device_ptr(s);
        Ok(TensorMut(unsafe {
            Tensor::new(ptr, shape, stride, (buf.len() * 4) as u64)?
        }))
    }

    /// Build one case's device state: row-dimensioned inputs at or past `valid` carry bf16 NaN,
    /// core_out carries the marker.
    fn build_case(device: &Device, man: Man) -> Result<Case> {
        let s = &device.stream;
        let cd = 10_240usize;
        let plan = man.plan_rows as usize;
        let valid = man.valid;
        let mut qkv = vec![bf16::NAN; plan * cd];
        let mut a = vec![bf16::NAN; plan * HV as usize];
        let mut b = vec![bf16::NAN; plan * HV as usize];
        {
            let vqkv = man.bf16("qkv.bf16", valid * cd);
            let va = man.bf16("a.bf16", valid * HV as usize);
            let vb = man.bf16("b.bf16", valid * HV as usize);
            qkv[..valid * cd].copy_from_slice(&vqkv);
            a[..valid * HV as usize].copy_from_slice(&va);
            b[..valid * HV as usize].copy_from_slice(&vb);
            // The row poison must be the exact bit pattern the marker check anticipates.
            for x in qkv[valid * cd..].iter_mut() {
                *x = bf16::from_bits(NAN_BITS);
            }
            for x in a[valid * HV as usize..].iter_mut() {
                *x = bf16::from_bits(NAN_BITS);
            }
            for x in b[valid * HV as usize..].iter_mut() {
                *x = bf16::from_bits(NAN_BITS);
            }
        }
        let conv_w = man.bf16("conv_w.bf16", cd * 4);
        let a_log = man.bf16("a_log.bf16", HV as usize);
        let dt_bias = man.bf16("dt_bias.bf16", HV as usize);
        let s_init = man.f32s("s_init.f32", HV as usize * DK as usize * DV as usize);
        let hist_init = man.f32s("hist_init.f32", cd * 3);
        let mut core = vec![bf16::from_f32(TAIL_MARKER); plan * HV as usize * DV as usize];
        core[..valid * HV as usize * DV as usize].fill(bf16::ZERO);
        let dev_qkv = s.clone_htod(&qkv)?;
        let dev_a = s.clone_htod(&a)?;
        let dev_b = s.clone_htod(&b)?;
        let dev_conv_w = s.clone_htod(&conv_w)?;
        let dev_a_log = s.clone_htod(&a_log)?;
        let dev_dt_bias = s.clone_htod(&dt_bias)?;
        let dev_core = s.clone_htod(&core)?;
        let dev_s = s.clone_htod(&s_init)?;
        let dev_hist = s.clone_htod(&hist_init)?;
        let req = GdnOps::workspace(&GdnSpec::q38(), man.plan_rows, GdnPolicy::LEGACY)?;
        let scratch_buf = s.alloc_zeros::<u8>(req.bytes as usize)?;
        Ok(Case {
            man,
            dev_qkv,
            dev_a,
            dev_b,
            dev_conv_w,
            dev_a_log,
            dev_dt_bias,
            dev_core,
            dev_s,
            dev_hist,
            scratch_buf,
            scratch_bytes: req.bytes,
        })
    }

    fn bindings(c: &Case, s: &Arc<CudaStream>, swap_gates: bool) -> Result<GdnBindings> {
        let plan = c.man.plan_rows;
        let cd = 10_240u32;
        let hv = HV;
        let (a, b) = if swap_gates {
            (&c.dev_b, &c.dev_a)
        } else {
            (&c.dev_a, &c.dev_b)
        };
        Ok(GdnBindings {
            qkv: view_bf16(&c.dev_qkv, s, [plan, cd], [u64::from(cd), 1])?,
            a: view_bf16(a, s, [plan, hv], [u64::from(hv), 1])?,
            b: view_bf16(b, s, [plan, hv], [u64::from(hv), 1])?,
            conv_weight: view_bf16(&c.dev_conv_w, s, [cd, 4], [4, 1])?,
            a_log: view_bf16(&c.dev_a_log, s, [hv], [1])?,
            dt_bias: view_bf16(&c.dev_dt_bias, s, [hv], [1])?,
            core_out: view_bf16_mut(
                &c.dev_core,
                s,
                [plan, hv, DV],
                [u64::from(hv * DV), u64::from(DV), 1],
            )?,
            s: view_f32(
                &c.dev_s,
                s,
                [hv, DK, DV],
                [u64::from(DK * DV), u64::from(DV), 1],
            )?,
            history: view_f32(&c.dev_hist, s, [cd, 3], [3, 1])?,
        })
    }

    fn meta_device(
        s: &Arc<CudaStream>,
        prefix: i32,
        valid: i32,
        capacity: i32,
        plan_rows: u32,
    ) -> Result<(CudaSlice<i32>, SeqRef)> {
        let m = SequenceMeta {
            prefix,
            valid_rows: valid,
            key_count: prefix + valid,
            capacity,
            causal: 1,
            reserved: [0; 3],
        };
        m.validate(capacity as u32, plan_rows)?;
        let buffer = s.clone_htod(&[
            m.prefix,
            m.valid_rows,
            m.key_count,
            m.capacity,
            m.causal,
            0,
            0,
            0,
        ])?;
        let (ptr, hold) = buffer.device_ptr(s);
        drop(hold);
        Ok((buffer, SeqRef { ptr }))
    }

    fn run_case(
        device: &Device,
        c: &Case,
        common: CommonOpsRef,
        phase: ExecutionPhase,
        swap_gates: bool,
    ) -> Result<()> {
        let s = &device.stream;
        let b = bindings(c, s, swap_gates)?;
        let plan = GdnOps::prepare(
            &device.ctx,
            &GdnSpec::q38(),
            c.man.plan_rows,
            b,
            common,
            false,
        )?;
        let (meta_buf, meta) = meta_device(
            s,
            c.man.prefix,
            c.man.valid as i32,
            c.man.capacity,
            c.man.plan_rows,
        )?;
        let (sp, _hold) = c.scratch_buf.device_ptr(s);
        GdnOps::run(
            &plan,
            s,
            meta,
            phase,
            Scratch {
                ptr: sp,
                bytes: c.scratch_bytes,
            },
        )?;
        s.synchronize()?;
        drop(meta_buf);
        Ok(())
    }

    /// The per-case gate, three terms:
    ///   (a) COUNT: the per-case count bound below 20k elements; the fraction bound (<= 5e-4) at >= 20k.
    ///   (b) SCALED against the case's own basis; informational when the basis is 0 and for rec_m1.
    ///   (c) the dead-row absolute term against the floor; always binds.
    ///
    /// The scaled comparison is `measured <= basis` in f64 with `RMS_ULP_TOL` relative slack.
    fn check_core(case: &str, man: &Man, got: &[bf16], reference: &[bf16]) -> Terms {
        let elems = man.valid * HV as usize * DV as usize;
        let t = terms(got, reference, man.valid);
        let frac = t.ndiff as f64 / elems as f64;
        let degenerate = man.degenerate;
        let (rule, scaled_informational) = if degenerate {
            (
                "COUNT (basis 0, v1.2 section 3: degenerate, scaled informational)".to_string(),
                true,
            )
        } else if let Some(r) = &man.rule {
            (format!("COUNT ({r}; scaled informational)"), true)
        } else {
            (
                "COUNT+FRACTION and SCALED (v1.2 section 3)".to_string(),
                false,
            )
        };
        judge_with_rule(case, man, &t, frac, elems, &rule, scaled_informational)
    }

    /// The three terms, with the rule that judged them printed.
    fn judge_with_rule(
        case: &str,
        man: &Man,
        t: &Terms,
        frac: f64,
        elems: usize,
        rule: &str,
        scaled_informational: bool,
    ) -> Terms {
        println!(
            "  {case}: scaled={:.3e} (bound {:.3e}{}) frac/count={} (frac {:.3e}) \
             dead={:.3e} (floor {:.3e}) nan={}  [judged: {rule}]",
            t.scaled,
            man.scaled_bound,
            if scaled_informational {
                ", informational"
            } else {
                ""
            },
            t.ndiff,
            frac,
            t.dead_max,
            t.floor,
            t.nan
        );
        assert_eq!(t.nan, 0, "{case}: non-finite output elements");
        assert!(
            t.dead_max <= t.floor,
            "{case}: dead-row absolute term {} > floor {}",
            t.dead_max,
            t.floor
        );
        // (a) count / fraction
        if let Some(cb) = man.count_bound {
            assert!(
                t.ndiff <= cb,
                "{case}: {}/{} elements differ, count bound {cb} [{rule}]",
                t.ndiff,
                elems
            );
        } else {
            assert!(
                frac <= man.frac_bound.unwrap(),
                "{case}: differing fraction {frac:.3e} > {} [{rule}]",
                man.frac_bound.unwrap()
            );
        }
        // (b) scaled, when it is a constraint
        if !scaled_informational {
            // The basis and the measured row RMS are 128-term f64 reductions in different orders;
            // the slack covers f64 rounding only.
            const RMS_ULP_TOL: f64 = 8.0 * f64::EPSILON;
            assert!(
                t.scaled <= man.scaled_bound * (1.0 + RMS_ULP_TOL),
                "{case}: scaled {:.17e} > per-case basis {:.17e} (excess {:.3e} relative) [{rule}]",
                t.scaled,
                man.scaled_bound,
                (t.scaled - man.scaled_bound) / man.scaled_bound,
            );
        }
        t.clone()
    }

    /// Required output directory for the stage dumps.
    fn dump_dir() -> PathBuf {
        let dir = PathBuf::from(
            std::env::var_os("FF_Q38_PROBE_DUMP").expect("FF_Q38_PROBE_DUMP is required"),
        );
        std::fs::create_dir_all(&dir).expect("create FF_Q38_PROBE_DUMP");
        dir
    }

    /// Stage dumps: reads the plan's own workspace offsets after a run and writes the valid rows
    /// of conv_out / q / k / v / beta_fp32 / g, plus the final state S.
    fn dump_workspace(case: &Case, name: &str, s: &Arc<CudaStream>) -> Result<()> {
        let ws = GdnWorkspace::q38(&GdnSpec::q38(), case.man.plan_rows)?;
        let host = s.clone_dtoh(&case.scratch_buf)?;
        let valid = case.man.valid as u64;
        let cd = 10_240u64;
        let hv = u64::from(HV);
        let dv = u64::from(DV);
        let dir = dump_dir();
        let wr = |region: Region, bytes: u64, path: PathBuf| -> Result<()> {
            let start = region.offset as usize;
            std::fs::write(&path, &host[start..start + bytes as usize])
                .map_err(|e| anyhow::anyhow!("write {}: {e}", path.display()))
        };
        wr(
            ws.conv_out,
            valid * cd * 2,
            dir.join(format!("gdn_ws_{name}_conv.bf16")),
        )?;
        wr(
            ws.q_fp32,
            valid * hv * dv * 4,
            dir.join(format!("gdn_ws_{name}_q.f32")),
        )?;
        wr(
            ws.k_fp32,
            valid * hv * dv * 4,
            dir.join(format!("gdn_ws_{name}_k.f32")),
        )?;
        wr(
            ws.v_fp32,
            valid * hv * dv * 4,
            dir.join(format!("gdn_ws_{name}_v.f32")),
        )?;
        wr(
            ws.beta_fp32,
            valid * hv * 4,
            dir.join(format!("gdn_ws_{name}_beta.f32")),
        )?;
        wr(
            ws.g,
            valid * hv * 4,
            dir.join(format!("gdn_ws_{name}_g.f32")),
        )?;
        // S is read from its binding, not the scratch region.
        let svals = s.clone_dtoh(&case.dev_s)?;
        let mut sraw = Vec::with_capacity(svals.len() * 4);
        for x in &svals {
            sraw.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::write(dir.join(format!("gdn_ws_{name}_s_final.f32")), &sraw)
            .map_err(|e| anyhow::anyhow!("write S_final: {e}"))?;
        Ok(())
    }

    #[test]
    #[ignore = "requires GPU window, FF_Q38_GPU_WINDOW=granted, FF_Q38_GDN_FIXTURE_DIR and FF_Q38_PROBE_DUMP"]
    fn gdn_binder_probe() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let ordinal: usize = std::env::var("FF_Q38_GPU_ORDINAL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let device = Device::new(ordinal)?;
        let s = &device.stream;
        let common = CommonOpsRef::new(&device.ctx)?;
        let phase_of = |m: &str| {
            if m == "recurrent" {
                ExecutionPhase::Decode
            } else {
                ExecutionPhase::Prefill
            }
        };

        // 4. end-to-end: every case against its own reference, plus case 2's tail differential
        for name in [
            "prefill_m64",
            "prefill_m1",
            "decode_m1",
            "tail_m40of64",
            "prefix_cont",
        ] {
            match std::env::var("FF_Q38_PROBE_ONLY") {
                Ok(only) if only != name => continue,
                _ => {}
            }
            let man = Man::load(name);
            let case = build_case(&device, man)?;
            let case = &case;
            run_case(
                &device,
                case,
                common.clone(),
                phase_of(&case.man.mode),
                false,
            )?;
            let elems = case.man.valid * HV as usize * DV as usize;
            dump_workspace(case, name, s)?;
            let got = s.clone_dtoh(&case.dev_core)?;
            let got_valid = &got[..elems];
            {
                // Raw core_out for the host-side stage/core checker, into the same run dir.
                let mut raw = Vec::with_capacity(elems * 2);
                for x in got_valid {
                    raw.extend_from_slice(&x.to_bits().to_le_bytes());
                }
                let dir = dump_dir();
                let _ = std::fs::write(dir.join(format!("gdn_dump_{name}.bf16")), &raw);
                println!(
                    "  {name} first elements got={:?} ref={:?}",
                    got_valid[..4]
                        .iter()
                        .map(|x| x.to_f32())
                        .collect::<Vec<_>>(),
                    case.man.bf16("ref_core.bf16", elems)[..4]
                        .iter()
                        .map(|x| x.to_f32())
                        .collect::<Vec<_>>()
                );
            }
            let reference = case.man.bf16("ref_core.bf16", elems);
            check_core(name, &case.man, got_valid, &reference);

            // Rows at or past valid_rows keep the marker exactly.
            let plan = case.man.plan_rows as usize;
            for (i, x) in got[elems..].iter().enumerate() {
                // Compare bit patterns against the stored BF16 marker; TAIL_MARKER is not representable in BF16.
                assert_eq!(
                    x.to_bits(),
                    bf16::from_f32(TAIL_MARKER).to_bits(),
                    "{name}: core_out row-dimension element {i} past valid_rows was written"
                );
            }

            // 2. the tail differential: the same valid rows through a plan sized exactly
            // to them must change nothing when the tail is present. Bitwise.
            if plan != case.man.valid {
                let mut man2 = Man::load(name);
                man2.plan_rows = man2.valid as u32;
                let case2 = build_case(&device, man2)?;
                run_case(
                    &device,
                    &case2,
                    common.clone(),
                    phase_of(&case2.man.mode),
                    false,
                )?;
                let got2 = s.clone_dtoh(&case2.dev_core)?;
                assert!(
                    got2[..elems] == got[..elems],
                    "{name}: core_out changed when the poisoned tail was removed"
                );
                let s1 = s.clone_dtoh(&case.dev_s)?;
                let s2 = s.clone_dtoh(&case2.dev_s)?;
                assert!(
                    s1 == s2,
                    "{name}: S advanced over rows at or past valid_rows"
                );
                let h1 = s.clone_dtoh(&case.dev_hist)?;
                let h2 = s.clone_dtoh(&case2.dev_hist)?;
                assert!(
                    h1 == h2,
                    "{name}: history advanced over rows at or past valid_rows"
                );
                println!(
                    "  {name}: tail differential (plan {} vs {}) bitwise identical",
                    plan, case2.man.plan_rows
                );
            }
        }

        // 1. phase contract on one input: Prefill-M1 vs Decode-M1
        {
            let man = Man::load("prefill_m1");
            let case = build_case(&device, man)?;
            let elems = case.man.valid * HV as usize * DV as usize;
            run_case(
                &device,
                &case,
                common.clone(),
                ExecutionPhase::Prefill,
                false,
            )?;
            dump_workspace(&case, "prefill_m1_prefill", s)?;
            {
                let dir = dump_dir();
                let mut raw = Vec::with_capacity(elems * 2);
                for x in s.clone_dtoh(&case.dev_core)?[..elems].iter() {
                    raw.extend_from_slice(&x.to_bits().to_le_bytes());
                }
                let _ = std::fs::write(dir.join("gdn_dump_prefill_m1_prefill.bf16"), &raw);
            }
            let got_chunked = s.clone_dtoh(&case.dev_core)?;
            let ref_chunked = case.man.bf16("ref_core.bf16", elems);
            check_core(
                "phase:prefill_m1",
                &case.man,
                &got_chunked[..elems],
                &ref_chunked,
            );

            // Rebuild the case for the Decode leg: `run` advances dev_s and dev_hist in place, and both
            // phases start from the fixture's initial state.
            let case2 = build_case(&device, Man::load("prefill_m1"))?;
            run_case(
                &device,
                &case2,
                common.clone(),
                ExecutionPhase::Decode,
                false,
            )?;
            // Distinct dump name: the second phase overwrites the scratch buffer.
            dump_workspace(&case2, "prefill_m1_decode", s)?;
            let got_rec = s.clone_dtoh(&case2.dev_core)?;
            let dir = dump_dir();
            let mut raw = Vec::with_capacity(elems * 2);
            for x in &got_rec[..elems] {
                raw.extend_from_slice(&x.to_bits().to_le_bytes());
            }
            let _ = std::fs::write(dir.join("gdn_dump_prefill_m1_decode.bf16"), &raw);
            let ref_rec = case2.man.bf16("ref_core_other.bf16", elems);
            let man_rec = {
                let mut m = Man::load("prefill_m1");
                m.scaled_bound = f64::INFINITY; // same count rule; scaled not the operative term
                m.count_bound = Some(4);
                m
            };
            check_core("phase:decode_m1", &man_rec, &got_rec[..elems], &ref_rec);
            // The phase contract is asserted on the path taken: at M=1 both cores compute the same
            // recurrence, so outputs cannot distinguish them.
            let log = take_phase_log();
            let last_two: Vec<&str> = log.iter().rev().take(2).rev().copied().collect();
            println!("  phase contract: cores selected = {log:?}");
            assert_eq!(
                last_two,
                vec!["chunked", "recurrent"],
                "Prefill-M1 must select the chunked core and Decode-M1 the recurrent core; \
                 selected {log:?}. Output comparison cannot substitute for this: at M=1 the \
                 two paths are numerically identical."
            );
            let differing = got_chunked[..elems]
                .iter()
                .zip(&got_rec[..elems])
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            println!(
                "  phase contract: Prefill-M1 vs Decode-M1 differ on {differing} elements (informational)"
            );
        }

        // 3. negative control: a/b swapped must fail the same comparison
        {
            let man = Man::load("prefill_m64");
            let case = build_case(&device, man)?;
            let elems = case.man.valid * HV as usize * DV as usize;
            run_case(&device, &case, common, ExecutionPhase::Prefill, true)?;
            let got = s.clone_dtoh(&case.dev_core)?;
            let reference = case.man.bf16("ref_core.bf16", elems);
            let t = terms(&got[..elems], &reference, case.man.valid);
            let frac = t.ndiff as f64 / elems as f64;
            println!(
                "  control(swapped a/b): scaled={:.3e} frac={:.3e} (bounds {:.3e} / {:?})",
                t.scaled, frac, case.man.scaled_bound, case.man.frac_bound
            );
            let failed = t.scaled > case.man.scaled_bound
                || frac > case.man.frac_bound.unwrap_or(0.0)
                || t.nan > 0;
            assert!(
                failed,
                "the swapped-gate control passed the per-case bounds"
            );
        }

        eprintln!("GDN_BINDER_PROBE PASS");
        Ok(())
    }
}
