//! Gated delta network with 16 key heads and 32 or 48 value heads.
//!
//! Geometry is bound from [`GdnSpec`] before launch: GQA 2 or 3,
//! dk = dv = 128, conv width 4, history width 3, chunk 32, FP32 tiles. Kernels are in `kernels/gdn/`.
//!
//! `run` issues the whole mixer between the `qkv`/`a`/`b` projections and the ungated
//! `core_out` on one stream; Q/K normalization uses FP32 before expansion.
//!
//! `hist_update` is a separate launch after conv; conv blocks read `history[]` to seed their window.
//!
//! Ordinary `run` selects the core by phase. `run_verify` keeps Prefill preparation and uses the
//! ordered h32 R1..16 and h48 R1..5 cores with resource-checked native bindings.
//! Plans bind before Ready.
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
use super::ops::GdnGateBindings;
use crate::gdn_plan::{
    CW_P2_CHUNK, CW_P2_MAX_GROUPS, CW_P2_VALUE_BLOCK, GdnPolicy, GdnWorkspace, Region, add, cw_p2,
    mul,
};
use crate::op_contract::{ExecutionPhase, GdnSpec, WorkspaceReq};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use half::bf16;
use std::mem::size_of;
use std::sync::Arc;

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
    /// Gate inputs `[M, hv]`.
    pub a: Tensor<bf16, 2>,
    pub b: Tensor<bf16, 2>,
    /// Depthwise conv weights `[conv_dim, 4]`, a zero-copy reshape of the
    /// manifest's `[conv_dim, 1, 4]`.
    pub conv_weight: Tensor<bf16, 2>,
    /// `[hv]` each.
    pub a_log: Tensor<bf16, 1>,
    pub dt_bias: Tensor<bf16, 1>,
    /// Ungated core output `[M, hv, 128]`. The caller applies `rms_gated`.
    pub core_out: TensorMut<bf16, 3>,
    /// Recurrent state `[hv, 128, 128]`, axes (value_head, key_dim, value_dim).
    /// Persistent across calls; owned by the Engine's StateStore.
    pub s: TensorMut<f32, 3>,
    /// Conv history `[conv_dim, 3]`, oldest -> newest, an EXACT widening of the
    /// raw pre-conv BF16 input (NOT of the conv or SiLU result). Persistent.
    pub history: TensorMut<f32, 2>,
}

struct Kernels {
    gate: CudaFunction,
    conv: CudaFunction,
    conv_l2: Option<[CudaFunction; 2]>,
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

fn bind_conv(
    kernels: &Kernels,
    spec: &GdnSpec,
    rows: u32,
    sms: u32,
    compact: bool,
) -> Result<(CudaFunction, u32)> {
    let qk_blocks = 2 * spec.hk * spec.dk / CONV_CHAN;
    let narrow = compact
        && qk_blocks * rows.div_ceil(CONV_ROWS) < sms
        && rows.div_ceil(8) > rows.div_ceil(CONV_ROWS);
    let function = if compact {
        &kernels
            .conv_l2
            .as_ref()
            .context("compact GDN convolution not bound")?[usize::from(narrow)]
    } else {
        &kernels.conv
    };
    Ok((function.clone(), if narrow { 8 } else { CONV_ROWS }))
}

/// Chunkwise-P2 launch configs for a plan bound at `rows`: stage A one CTA per (chunk, head)
/// with grid.x = ceil(rows/64), stage H one CTA per 32-wide value block with grid.x = DV/32.
fn cw_p2_grids(rows: u32, hv: u32) -> (LaunchConfig, LaunchConfig) {
    let per_chunk = LaunchConfig {
        grid_dim: (rows.div_ceil(CW_P2_CHUNK), hv, 1),
        block_dim: (P2_THREADS, 1, 1),
        shared_mem_bytes: cw_p2::SMEM_BYTES as u32,
    };
    let per_value_block = LaunchConfig {
        grid_dim: (DV / CW_P2_VALUE_BLOCK, hv, 1),
        block_dim: (P2_THREADS, 1, 1),
        shared_mem_bytes: cw_p2::SMEM_BYTES as u32,
    };
    (per_chunk, per_value_block)
}

fn cw_p2_groups(
    ws: &GdnWorkspace,
    rows: u32,
    hv: u32,
) -> Result<[Option<(i32, i32, LaunchConfig)>; CW_P2_MAX_GROUPS]> {
    let mut groups = [None; CW_P2_MAX_GROUPS];
    if ws.p2_group_rows == 0 {
        return Ok(groups);
    }
    ensure!(
        rows.div_ceil(ws.p2_group_rows) as usize <= groups.len(),
        "GDN P2 group count exceeds its slot-geometry bound"
    );
    for (index, base) in (0..rows).step_by(ws.p2_group_rows as usize).enumerate() {
        let count = ws.p2_group_rows.min(rows - base);
        groups[index] = Some((base as i32, count as i32, cw_p2_grids(count, hv).0));
    }
    Ok(groups)
}

/// CUDA caps gridDim.y and gridDim.z at 65535; prep puts rows on gridDim.z and beta on gridDim.y.
pub(crate) const CUDA_GRID_YZ_MAX: u32 = 65_535;

/// Prep grids for `rows`; `prepare` and `for_rows` both derive them here.
fn prep_grids(
    prep_cfg: LaunchConfig,
    beta_cfg: LaunchConfig,
    conv_cfg: LaunchConfig,
    rows: u32,
    conv_rows: u32,
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
    conv_cfg.grid_dim.1 = rows.div_ceil(conv_rows);
    Ok((prep_cfg, beta_cfg, conv_cfg))
}

pub(crate) struct GdnPlan {
    spec: GdnSpec,
    snapshots: Option<(Tensor<f32, 4>, Tensor<f32, 3>)>,
    kernels: Arc<Kernels>,
    conv: CudaFunction,
    sms: u32,
    bindings: GdnBindings,
    ws: GdnWorkspace,
    /// The single derivation of `conv_dim`, asserted against the kernel constant.
    conv_dim: u32,
    /// `GdnSpec::l2_eps`, used by FP32 Q/K normalization.
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
    /// Prebound row base, group extent and stage-A grid, in chunk order.
    p2_groups: [Option<(i32, i32, LaunchConfig)>; CW_P2_MAX_GROUPS],
    p2_h_cfg: LaunchConfig,
}

/// Q/K are expanded FP32; V is the BF16 conv_out column window at v_row_stride.
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
            let limit = self.spec.snapshot_limit()?;
            ensure!(
                (1..=limit).contains(&rows),
                "GDN hv={} snapshot rows={rows} exceeds bound={limit}",
                self.spec.hv
            );
            state.validate()?;
            let slots = state.shape[0];
            ensure!(
                slots < limit
                    && slots >= rows - 1
                    && state.shape == [slots, self.spec.hv, DK, DV]
                    && state.stride
                        == [
                            u64::from(self.spec.hv * DK * DV),
                            u64::from(DK * DV),
                            u64::from(DV),
                            1
                        ],
                "invalid state snapshot layout"
            );
            history.validate()?;
            ensure!(
                history.shape == [slots, self.conv_dim, 3]
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
        check_bindings(&bindings, rows, self.conv_dim, self.spec.hv)?;
        let cw_p2 = snapshots.is_none() && self.cw_p2 && rows > crate::gdn_plan::MIN_VERIFY_ROWS;
        let ws = GdnWorkspace::new(&self.spec, rows, cw_p2)?;
        let (conv, conv_rows) = bind_conv(
            &self.kernels,
            &self.spec,
            rows,
            self.sms,
            ws.q_bf16.bytes > 0,
        )?;
        let (prep_cfg, beta_cfg, conv_cfg) =
            prep_grids(self.prep_cfg, self.beta_cfg, self.conv_cfg, rows, conv_rows)?;
        Ok(Self {
            spec: self.spec,
            snapshots,
            kernels: self.kernels.clone(),
            conv,
            sms: self.sms,
            bindings,
            ws,
            conv_dim: self.conv_dim,
            l2_eps: self.l2_eps,
            max_rows: rows,
            prep_cfg,
            beta_cfg,
            conv_cfg,
            hist_cfg: self.hist_cfg,
            chunked_cfg: self.chunked_cfg,
            recurrent_cfg: self.recurrent_cfg,
            p2_groups: cw_p2_groups(&ws, rows, self.spec.hv)?,
            p2_h_cfg: cw_p2_grids(rows, self.spec.hv).1,
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
        scratch.validate(self.ws.bytes)?;
        ensure!(
            self.cw_p2 && self.p2_groups[0].is_some(),
            "chunkwise P2 slot groups were not bound"
        );
        let p2 = self
            .kernels
            .p2
            .as_ref()
            .context("chunkwise P2 stages were not bound by prepare")?;
        let rows = self.max_rows;
        let ws = scratch.ptr;
        let (q, k, v) = (src.q, src.k, src.v);
        let beta = ws_f32(self.ws.beta_fp32, ws, rows, self.spec.hv)?.0.ptr;
        let g = ws_f32(self.ws.g, ws, rows, self.spec.hv)?.0.ptr;
        let slot = add(ws, self.ws.p2.offset)?;
        let compact = self.ws.q_bf16.bytes > 0;
        let a_query = if compact {
            ws + self.ws.q_bf16.offset
        } else {
            q
        };
        let a_key = ws + self.ws.k_bf16.offset;
        for &(row_base, group_rows, a_cfg) in self.p2_groups.iter().flatten() {
            let stage = crate::gdn_plan::P2StageArgs {
                q,
                k,
                v,
                beta,
                g,
                ws: slot,
                meta: meta.ptr,
                max_rows: rows as i32,
                row_base,
                group_rows,
                v_row_stride: src.v_row_stride,
            };
            unsafe {
                let mut a = stream.launch_builder(&p2.a);
                a.arg(&a_query)
                    .arg(&stage.k)
                    .arg(&stage.v)
                    .arg(&stage.beta)
                    .arg(&stage.g)
                    .arg(&stage.ws)
                    .arg(&stage.meta)
                    .arg(&stage.max_rows)
                    .arg(&stage.row_base)
                    .arg(&stage.group_rows)
                    .arg(&stage.v_row_stride);
                if compact {
                    a.arg(&a_key);
                }
                a.launch(a_cfg)?;
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
                    .arg(&stage.max_rows)
                    .arg(&stage.row_base)
                    .arg(&stage.group_rows)
                    .launch(self.p2_h_cfg)?;
            }
        }
        Ok(())
    }
}

pub(crate) struct GdnOps;

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
        cw_p2: bool,
    ) -> Result<GdnPlan> {
        spec.validate()?;
        let ws = GdnWorkspace::new(spec, max_rows, cw_p2)?;
        let compact = ws.q_bf16.bytes > 0;
        let conv_dim = spec.conv_dim()?;
        let suffix = spec.kernel_suffix()?;
        let compiled_width = match suffix {
            "" => 10240,
            "_h32" => 8192,
            _ => anyhow::bail!("unsupported GDN kernel suffix={suffix}"),
        };
        ensure!(
            conv_dim == compiled_width,
            "GDN conv_dim={conv_dim} differs from compiled width={compiled_width} for hv={}",
            spec.hv
        );
        check_bindings(&bindings, max_rows, conv_dim, spec.hv)?;

        use cudarc::driver::sys::CUdevice_attribute as Attribute;
        let sms =
            u32::try_from(ctx.attribute(Attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?)?;
        ensure!(sms > 0, "GDN device has no multiprocessors");
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

        let module = |name| cubin::module(ctx, &format!("gdn/{name}{suffix}"));
        let conv = module("gdn_conv")?;
        let chunked_mod = module("gdn_chunked")?;
        let prep_mod = module("gdn_prep")?;
        let recurrent_mod = module("gdn_recurrent")?;
        let f_conv = conv.load_function("gdn_conv")?;
        let f_hist = conv.load_function("gdn_conv_hist_update")?;
        let f_prep = prep_mod.load_function("gdn_prep_l2_fp32")?;
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

        let rstep = Some(module("gdn_rstep")?.load_function("gdn_rstep_core_v1")?);
        let hist_snapshot = module("gdn_conv")?.load_function("gdn_conv_hist_snapshot")?;
        let core_snapshot = module("gdn_rstep")?.load_function("gdn_rstep_core_snapshot")?;
        // All three arenas are under the 48 KiB no-opt-in cap.
        let p2 = if !cw_p2 {
            None
        } else {
            let p2_mod = module(if compact {
                "gdn_cw_p2_compact"
            } else {
                "gdn_cw_p2"
            })?;
            Some(ChunkwiseP2 {
                a: p2_mod.load_function("gdn_cw_p2_stage_a")?,
                h: p2_mod.load_function("gdn_cw_p2_stage_h")?,
            })
        };
        let kernels = Kernels {
            gate: cubin::module(ctx, "q38_common")?.load_function(spec.gate_kernel()?)?,
            hist_snapshot,
            core_snapshot,
            rstep,
            conv: f_conv,
            conv_l2: if compact {
                Some([
                    conv.load_function("gdn_conv_l2_compact")?,
                    conv.load_function("gdn_conv_l2_compact_rows8")?,
                ])
            } else {
                None
            },
            hist_update: f_hist,
            prep: f_prep,
            beta_widen: f_beta,
            chunked: f_chunked,
            recurrent: f_recurrent,
            p2,
        };
        let max_threads =
            u32::try_from(ctx.attribute(Attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?)?;
        ensure!(
            ctx.attribute(Attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE)? == 32,
            "GDN requires 32-thread warps"
        );
        let core_functions = [
            (
                "recurrent",
                &kernels.recurrent,
                CORE_THREADS,
                RECURRENT_SMEM,
            ),
            ("chunked", &kernels.chunked, CORE_THREADS, CHUNKED_SMEM),
            (
                "rstep snapshot",
                &kernels.core_snapshot,
                CORE_THREADS,
                RECURRENT_SMEM,
            ),
        ];
        let rstep_function = kernels
            .rstep
            .as_ref()
            .map(|f| ("rstep", f, CORE_THREADS, RECURRENT_SMEM));
        let p2_functions = kernels.p2.iter().flat_map(|p2| {
            [
                ("P2 stage A", &p2.a, P2_THREADS, cw_p2::SMEM_BYTES as u32),
                ("P2 stage H", &p2.h, P2_THREADS, cw_p2::SMEM_BYTES as u32),
            ]
        });
        for (name, function, threads, shared) in core_functions
            .into_iter()
            .chain(rstep_function)
            .chain(p2_functions)
        {
            ensure!(
                threads <= max_threads && u64::from(shared) <= optin,
                "GDN {name} launch exceeds device limits: threads={threads}/{max_threads} shared={shared}/{optin}"
            );
            let resident = function
                .occupancy_max_active_blocks_per_multiprocessor(threads, shared as usize, None)
                .with_context(|| {
                    format!("GDN {name} occupancy threads={threads} shared={shared}")
                })?;
            ensure!(
                resident > 0,
                "GDN {name} has zero resident blocks: threads={threads} shared={shared}"
            );
        }
        let (conv, conv_rows) = bind_conv(&kernels, spec, max_rows, sms, compact)?;
        // Grids derive from the bound rows, never from device valid_rows.
        let (prep_cfg, beta_cfg, conv_cfg) = prep_grids(
            LaunchConfig {
                grid_dim: (DV.div_ceil(PREP_THREADS), spec.hk, 1),
                block_dim: (PREP_THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            LaunchConfig {
                grid_dim: (spec.hv.div_ceil(PREP_THREADS), 1, 1),
                block_dim: (PREP_THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            LaunchConfig {
                grid_dim: (conv_dim / CONV_CHAN, 1, 1),
                block_dim: (CONV_CHAN, 1, 1),
                shared_mem_bytes: 0,
            },
            max_rows,
            conv_rows,
        )?;
        Ok(GdnPlan {
            spec: *spec,
            snapshots: None,
            kernels: Arc::new(kernels),
            conv,
            sms,
            bindings,
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
                grid_dim: (1, spec.hv, 1),
                block_dim: (CORE_THREADS, 1, 1),
                shared_mem_bytes: CHUNKED_SMEM,
            },
            recurrent_cfg: LaunchConfig {
                grid_dim: (2, spec.hv, 1),
                block_dim: (CORE_THREADS, 1, 1),
                shared_mem_bytes: RECURRENT_SMEM,
            },
            p2_groups: cw_p2_groups(&ws, max_rows, spec.hv)?,
            p2_h_cfg: cw_p2_grids(max_rows, spec.hv).1,
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
            (1..=plan.spec.snapshot_limit()?).contains(&plan.max_rows),
            "GDN hv={} verify rows={} exceeds snapshot bound={}",
            plan.spec.hv,
            plan.max_rows,
            plan.spec.snapshot_limit()?
        );
        ensure!(
            plan.spec.hv != 32 || plan.kernels.rstep.is_some(),
            "GDN hv=32 verify requires the ordered recurrent core"
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
        // Decode uses the separate one-row plan.
        ensure!(
            !plan.cw_p2 || phase != ExecutionPhase::Decode,
            "chunkwise-P2 prefill plan cannot run phase={phase:?}; decode requires the one-row plan"
        );
        ensure!(
            !verify || !plan.cw_p2,
            "chunkwise plan cannot verify; verify keeps the recurrent core"
        );
        scratch.validate(plan.ws.bytes)?;
        let b = &plan.bindings;
        let cd = plan.conv_dim;
        let rows = plan.max_rows;
        let el_bf = size_of::<bf16>() as u64;
        let ws_ptr = scratch.ptr;

        let span = mul(mul(u64::from(cd), el_bf)?, u64::from(rows))?;
        // Packed Q/K each have hk * dk channels; V has hv * dv channels.
        let q_off = 0u64;
        let k_off = u64::from(plan.spec.hk * plan.spec.dk) * el_bf;
        let v_off = add(k_off, u64::from(plan.spec.hk * plan.spec.dk) * el_bf)?;

        let conv_out = ws_bf16(plan.ws.conv_out, ws_ptr, rows, cd)?;
        let mut beta = ws_bf16(plan.ws.beta, ws_ptr, rows, plan.spec.hv)?;
        let beta_fp32 = ws_f32(plan.ws.beta_fp32, ws_ptr, rows, plan.spec.hv)?;
        let mut g = ws_f32(plan.ws.g, ws_ptr, rows, plan.spec.hv)?;
        let q_fp32 = ws_f32_3(plan.ws.q_fp32, ws_ptr, rows, plan.ws.qk_heads, plan.spec.dk)?;
        let k_fp32 = ws_f32_3(plan.ws.k_fp32, ws_ptr, rows, plan.ws.qk_heads, plan.spec.dk)?;
        let v_fp32 = ws_f32_3(plan.ws.v_fp32, ws_ptr, rows, plan.spec.hv, plan.spec.dv)?;

        let dom = RowDomain {
            meta,
            row_base: 0,
            row_capacity: rows,
        };
        let l2_eps = plan.l2_eps;
        let conv_ptr = conv_out.0.ptr;
        let compact = plan.ws.q_bf16.bytes > 0;

        // conv: qkv -> conv_out with the exact SiLU; the conv output is cast to BF16 and back
        // before silu() (ABI §5).
        unsafe {
            let q_bf16 = ws_ptr + plan.ws.q_bf16.offset;
            let k_bf16 = ws_ptr + plan.ws.k_bf16.offset;
            if compact {
                ws_region(plan.ws.q_bf16, rows, plan.ws.qk_heads * DK, 2)?;
                ws_region(plan.ws.k_bf16, rows, plan.ws.qk_heads * DK, 2)?;
            }
            let cd_i32 = cd as i32;
            let mut conv = stream.launch_builder(&plan.conv);
            conv.arg(&b.qkv.ptr)
                .arg(&b.conv_weight.ptr)
                .arg(&b.history.0.ptr)
                .arg(&conv_ptr)
                .arg(&meta.ptr)
                .arg(&cd_i32);
            if compact {
                conv.arg(&q_fp32.0.ptr)
                    .arg(&k_fp32.0.ptr)
                    .arg(&q_bf16)
                    .arg(&k_bf16)
                    .arg(&l2_eps);
            }
            conv.launch(plan.conv_cfg)?;
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
        // common gdn_gates: beta BF16, g FP32.
        super::ops::GateOps::gdn_gates_with(
            &plan.kernels.gate,
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
        let v_bf16 = as3(conv_ptr, rows, plan.spec.hv, cd, span, v_off)?;
        let q = conv_ptr + q_off;
        let k = conv_ptr + k_off;
        // Exact widens; the cores read `beta` as FP32.
        unsafe {
            stream
                .launch_builder(&plan.kernels.beta_widen)
                .arg(&beta.0.ptr)
                .arg(&beta_fp32.0.ptr)
                .arg(&meta.ptr)
                .launch(plan.beta_cfg)?;
            // Q/K expand to value heads; Q is scaled by 1/sqrt(dk).
            let v = if plan.cw_p2 { 0 } else { v_fp32.0.ptr };
            if !compact {
                stream
                    .launch_builder(&plan.kernels.prep)
                    .arg(&q)
                    .arg(&k)
                    .arg(&v_bf16.ptr)
                    .arg(&q_fp32.0.ptr)
                    .arg(&k_fp32.0.ptr)
                    .arg(&v)
                    .arg(&cd)
                    .arg(&meta.ptr)
                    .arg(&l2_eps)
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
                            q: q_fp32.0.ptr,
                            k: k_fp32.0.ptr,
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
fn check_bindings(b: &GdnBindings, rows: u32, conv_dim: u32, hv: u32) -> Result<()> {
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
                && t.shape[1] == hv
                && t.stride[1] == 1
                && t.stride[0] >= u64::from(hv),
            "GDN {what} must be [max_rows, {hv}] with stride [{hv}, 1]"
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
            t.shape == [hv] && t.stride == [1],
            "GDN {what} must be [{hv}] with stride [1]"
        );
    }
    b.core_out.0.validate()?;
    ensure!(
        b.core_out.0.shape[0] == rows
            && b.core_out.0.shape[1] == hv
            && b.core_out.0.shape[2] == DV
            && b.core_out.0.stride == [u64::from(hv * DV), u64::from(DV), 1],
        "GDN core_out must be [max_rows, {hv}, 128] with strides [hv*128, 128, 1]"
    );
    b.s.0.validate()?;
    ensure!(
        b.s.0.shape == [hv, DK, DV] && b.s.0.stride == [u64::from(DK * DV), u64::from(DV), 1],
        "GDN S must be [{hv}, 128, 128] on (value_head, key_dim, value_dim) with strides [16384, 128, 1]"
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
    use cudarc::driver::{DevicePtr, LaunchConfig};

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
            let (prep, beta, conv) = prep_grids(full.0, full.1, full.2, rows, CONV_ROWS)?;
            assert_eq!(prep.grid_dim.2, rows, "GDN FP32 L2 prep grid z");
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
        let (wide, _, _) = prep_grids(full.0, full.1, full.2, 2048, CONV_ROWS)?;
        let (narrow, _, _) = prep_grids(full.0, full.1, full.2, 1021, CONV_ROWS)?;
        assert!(narrow.grid_dim.2 < wide.grid_dim.2);
        Ok(())
    }

    #[test]
    fn rows_beyond_the_cuda_grid_yz_limit_are_rejected() -> Result<()> {
        let full = (cfg((2, 48, 2048)), cfg((1, 2048, 1)), cfg((1280, 16, 1)));
        prep_grids(full.0, full.1, full.2, CUDA_GRID_YZ_MAX, CONV_ROWS)?;
        let rejected = prep_grids(full.0, full.1, full.2, CUDA_GRID_YZ_MAX + 1, CONV_ROWS)
            .expect_err("rows above the CUDA gridDim.y/z limit must not form a launch");
        assert!(rejected.to_string().contains("gridDim"), "{rejected}");
        Ok(())
    }

    use super::*;
    use crate::gdn_plan::cw_p2_bytes;

    struct GeometryBuffers {
        qkv: cudarc::driver::CudaSlice<bf16>,
        a: cudarc::driver::CudaSlice<bf16>,
        b: cudarc::driver::CudaSlice<bf16>,
        conv: cudarc::driver::CudaSlice<bf16>,
        a_log: cudarc::driver::CudaSlice<bf16>,
        dt: cudarc::driver::CudaSlice<bf16>,
        out: cudarc::driver::CudaSlice<bf16>,
        state: cudarc::driver::CudaSlice<f32>,
        history: cudarc::driver::CudaSlice<f32>,
        snapshots: Option<(
            cudarc::driver::CudaSlice<f32>,
            cudarc::driver::CudaSlice<f32>,
        )>,
    }

    fn geometry_state(heads: usize) -> Vec<f32> {
        (0..heads * 128 * 128)
            .map(|i| ((i * 19 % 251) as f32 - 125.0) / 1024.0)
            .collect()
    }

    fn geometry_view<T: cudarc::driver::DeviceRepr + Copy, const R: usize>(
        buffer: &cudarc::driver::CudaSlice<T>,
        stream: &Arc<CudaStream>,
        shape: [u32; R],
        stride: [u64; R],
        offset: usize,
    ) -> Result<Tensor<T, R>> {
        use cudarc::driver::DevicePtr;
        let bytes = size_of::<T>();
        let remaining = buffer
            .len()
            .checked_sub(offset)
            .context("fixture offset past buffer")?;
        let (ptr, _hold) = buffer.device_ptr(stream);
        unsafe {
            Tensor::new(
                ptr + (offset * bytes) as u64,
                shape,
                stride,
                (remaining * bytes) as u64,
            )
        }
    }

    impl GeometryBuffers {
        fn new(stream: &Arc<CudaStream>, spec: GdnSpec, rows: u32, valid: u32) -> Result<Self> {
            let hv = spec.hv as usize;
            let cd = spec.conv_dim()? as usize;
            let pattern = |i: usize, period: usize| {
                bf16::from_f32(((i % period) as f32 - (period / 2) as f32) / 512.0)
            };
            let period = |c: usize| if c < 2048 { 251 } else { 257 };
            let zero_head = |c: usize| c < 4096 && c % 2048 < 128;
            let qkv: Vec<_> = (0..rows as usize * cd)
                .map(|i| {
                    if i / cd < valid as usize {
                        let c = i % cd;
                        if zero_head(c) {
                            bf16::ZERO
                        } else {
                            pattern((i / cd) * 17 + c * 11, period(c))
                        }
                    } else {
                        bf16::NAN
                    }
                })
                .collect();
            let gates = |shift, prime| {
                (0..rows as usize * hv)
                    .map(|i| {
                        if i / hv < valid as usize {
                            pattern((i / hv) * 7 + (i % hv) * 13 + shift, prime)
                        } else {
                            bf16::NAN
                        }
                    })
                    .collect::<Vec<_>>()
            };
            let conv: Vec<_> = (0..cd * 4)
                .map(|i| pattern((i / 4) * 5 + i % 4, period(i / 4)))
                .collect();
            let a_log: Vec<_> = (0..hv)
                .map(|h| bf16::from_f32(-3.0 + h as f32 / 128.0))
                .collect();
            let dt: Vec<_> = (0..hv).map(|h| pattern(h * 3, 251)).collect();
            let history: Vec<_> = (0..cd * 3)
                .map(|i| {
                    if zero_head(i / 3) {
                        0.0
                    } else {
                        pattern((i / 3) * 23 + i % 3, period(i / 3)).to_f32()
                    }
                })
                .collect();
            Ok(Self {
                qkv: stream.clone_htod(&qkv)?,
                a: stream.clone_htod(&gates(0, 251))?,
                b: stream.clone_htod(&gates(19, 257))?,
                conv: stream.clone_htod(&conv)?,
                a_log: stream.clone_htod(&a_log)?,
                dt: stream.clone_htod(&dt)?,
                out: stream.clone_htod(&vec![bf16::from_bits(0x7fc5); rows as usize * hv * 128])?,
                state: stream.clone_htod(&geometry_state(hv))?,
                history: stream.clone_htod(&history)?,
                snapshots: None,
            })
        }

        fn bind_snapshots(
            &mut self,
            stream: &Arc<CudaStream>,
            spec: GdnSpec,
        ) -> Result<(Tensor<f32, 4>, Tensor<f32, 3>)> {
            ensure!(self.snapshots.is_none(), "geometry snapshots already bound");
            let cd = spec.conv_dim()?;
            let slots = spec.snapshot_limit()? - 1;
            let poison = f32::from_bits(0x7fc1_0001);
            self.snapshots = Some((
                stream.clone_htod(&vec![poison; slots as usize * spec.hv as usize * 128 * 128])?,
                stream.clone_htod(&vec![poison; slots as usize * cd as usize * 3])?,
            ));
            let (state, history) = self.snapshots.as_ref().context("snapshots not allocated")?;
            Ok((
                geometry_view(
                    state,
                    stream,
                    [slots, spec.hv, 128, 128],
                    [u64::from(spec.hv) * 128 * 128, 128 * 128, 128, 1],
                    0,
                )?,
                geometry_view(
                    history,
                    stream,
                    [slots, cd, 3],
                    [u64::from(cd) * 3, 3, 1],
                    0,
                )?,
            ))
        }

        fn bindings(
            &self,
            stream: &Arc<CudaStream>,
            spec: GdnSpec,
            rows: u32,
            offset: u32,
        ) -> Result<GdnBindings> {
            let hv = spec.hv;
            let cd = spec.conv_dim()?;
            let row = offset as usize;
            Ok(GdnBindings {
                qkv: geometry_view(
                    &self.qkv,
                    stream,
                    [rows, cd],
                    [u64::from(cd), 1],
                    row * cd as usize,
                )?,
                a: geometry_view(
                    &self.a,
                    stream,
                    [rows, hv],
                    [u64::from(hv), 1],
                    row * hv as usize,
                )?,
                b: geometry_view(
                    &self.b,
                    stream,
                    [rows, hv],
                    [u64::from(hv), 1],
                    row * hv as usize,
                )?,
                conv_weight: geometry_view(&self.conv, stream, [cd, 4], [4, 1], 0)?,
                a_log: geometry_view(&self.a_log, stream, [hv], [1], 0)?,
                dt_bias: geometry_view(&self.dt, stream, [hv], [1], 0)?,
                core_out: TensorMut(geometry_view(
                    &self.out,
                    stream,
                    [rows, hv, 128],
                    [u64::from(hv) * 128, 128, 1],
                    row * hv as usize * 128,
                )?),
                s: TensorMut(geometry_view(
                    &self.state,
                    stream,
                    [hv, 128, 128],
                    [128 * 128, 128, 1],
                    0,
                )?),
                history: TensorMut(geometry_view(&self.history, stream, [cd, 3], [3, 1], 0)?),
            })
        }
    }

    struct GeometryOutput {
        out: Vec<bf16>,
        state: Vec<f32>,
        history: Vec<f32>,
        prefix_states: Vec<f32>,
        prefix_histories: Vec<f32>,
        norm_ratio: f64,
        snapshot_mismatches: usize,
        p2_prepared: Vec<u8>,
        p2_slots: Vec<u8>,
    }

    fn geometry_meta(
        stream: &Arc<CudaStream>,
        prefix: u32,
        valid: u32,
    ) -> Result<cudarc::driver::CudaSlice<i32>> {
        stream
            .clone_htod(&[
                prefix as i32,
                valid as i32,
                (prefix + valid) as i32,
                8192,
                1,
                0,
                0,
                0,
            ])
            .map_err(Into::into)
    }

    #[derive(Clone, Copy, Debug)]
    enum GeometryCore {
        Recurrent,
        Selected,
        Verify,
        ChunkedSnapshot,
        RstepSnapshot,
        Chunked,
        P1(u32),
        P2,
        P2Whole,
        P2Expanded,
        WrongHeadStride,
    }

    fn geometry_plan(
        device: &crate::backend::cuda::Device,
        spec: GdnSpec,
        rows: u32,
        offset: u32,
        buffers: &mut GeometryBuffers,
        core: GeometryCore,
    ) -> Result<GdnPlan> {
        let stream = &device.stream;
        let compact = matches!(core, GeometryCore::P2 | GeometryCore::P2Whole)
            || matches!(core, GeometryCore::Selected)
                && GdnPolicy::for_device(&device.info).cw_p2_for_step(rows);
        let mut plan = GdnOps::prepare(
            &device.ctx,
            &spec,
            rows,
            buffers.bindings(stream, spec, rows, offset)?,
            compact,
        )?;
        if matches!(core, GeometryCore::P2Expanded) {
            let module = cubin::module(
                &device.ctx,
                &format!("gdn/gdn_cw_p2{}", spec.kernel_suffix()?),
            )?;
            Arc::get_mut(&mut plan.kernels)
                .context("expanded P2 kernels are shared")?
                .p2 = Some(ChunkwiseP2 {
                a: module.load_function("gdn_cw_p2_stage_a")?,
                h: module.load_function("gdn_cw_p2_stage_h")?,
            });
            plan.cw_p2 = true;
            let bounded = GdnWorkspace::new(&spec, rows, true)?;
            plan.ws.p2 = Region {
                offset: if bounded.p2.offset == bounded.v_fp32.offset {
                    plan.ws.v_fp32.offset
                } else {
                    plan.ws.bytes
                },
                bytes: bounded.p2.bytes,
            };
            plan.ws.p2_group_rows = bounded.p2_group_rows;
            plan.ws.bytes = plan.ws.bytes.max(add(plan.ws.p2.offset, plan.ws.p2.bytes)?);
            plan.p2_groups = cw_p2_groups(&plan.ws, rows, spec.hv)?;
        }
        if matches!(core, GeometryCore::P2Whole) {
            let base = add(plan.ws.k_bf16.offset, plan.ws.k_bf16.bytes)?;
            plan.ws.p2 = Region {
                offset: base,
                bytes: cw_p2_bytes(spec.hv, rows),
            };
            plan.ws.p2_group_rows = rows.div_ceil(CW_P2_CHUNK) * CW_P2_CHUNK;
            plan.ws.bytes = add(base, plan.ws.p2.bytes)?;
            plan.p2_groups = cw_p2_groups(&plan.ws, rows, spec.hv)?;
        }
        if matches!(
            core,
            GeometryCore::ChunkedSnapshot | GeometryCore::RstepSnapshot
        ) {
            let suffix = spec.kernel_suffix()?;
            let kernels = Arc::get_mut(&mut plan.kernels).context("snapshot kernels are shared")?;
            if matches!(core, GeometryCore::RstepSnapshot) {
                let module = cubin::module(&device.ctx, &format!("gdn/gdn_rstep{suffix}"))?;
                kernels.rstep = Some(module.load_function("gdn_rstep_core_v1")?);
                kernels.core_snapshot = module.load_function("gdn_rstep_core_snapshot")?;
            } else {
                ensure!(
                    rows <= 5,
                    "chunked snapshot kernel supports at most five rows"
                );
                kernels.rstep = None;
                kernels.core_snapshot =
                    cubin::module(&device.ctx, &format!("gdn/gdn_chunked{suffix}"))?
                        .load_function("gdn_chunked_core_snapshot")?;
                kernels.core_snapshot.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    CHUNKED_SMEM as i32,
                )?;
            }
        }
        if matches!(
            core,
            GeometryCore::Verify | GeometryCore::ChunkedSnapshot | GeometryCore::RstepSnapshot
        ) {
            plan = plan.for_rows(rows, Some(buffers.bind_snapshots(stream, spec)?))?;
        }
        if matches!(core, GeometryCore::WrongHeadStride) {
            let directory = std::path::PathBuf::from(
                std::env::var_os("FF_Q38_PROBE_DIR").context("GDN control directory missing")?,
            );
            let module = device
                .ctx
                .load_module(cudarc::nvrtc::Ptx::from_file(directory.join(format!(
                    "gdn_prep_h32_wrong_stride_sm{}.cubin",
                    device.info.arch
                ))))?;
            Arc::get_mut(&mut plan.kernels)
                .context("GDN control kernels are shared")?
                .prep = module.load_function("gdn_prep_l2_fp32")?;
        }
        Ok(plan)
    }

    fn geometry_run(
        device: &crate::backend::cuda::Device,
        spec: GdnSpec,
        rows: u32,
        valid: u32,
        core: GeometryCore,
        graph: bool,
    ) -> Result<GeometryOutput> {
        use cudarc::driver::{DevicePtr, sys};
        let label = std::ffi::CString::new(format!(
            "gdn hv={} rows={rows} valid={valid} core={core:?} graph={graph}",
            spec.hv
        ))?;
        let _range = super::super::profile::Range::new(true, &label);
        let stream = &device.stream;
        let mut buffers = GeometryBuffers::new(stream, spec, rows, valid)?;
        let sequential = matches!(core, GeometryCore::Recurrent);
        let mut plans = Vec::new();
        for row in 0..if sequential { valid } else { 1 } {
            let bound = if sequential { 1 } else { rows };
            let meta = geometry_meta(stream, 23 + row, if sequential { 1 } else { valid })?;
            let plan = geometry_plan(device, spec, bound, row, &mut buffers, core)?;
            plans.push((plan, meta));
        }
        let layout = plans.first().context("empty geometry plans")?.0.ws;
        let bytes = layout.bytes;
        let mut workspace = stream.alloc_zeros::<u8>(bytes as usize)?;
        let ptr = workspace.device_ptr(stream).0;
        if !sequential && valid < rows {
            for (region, element_bytes) in [
                (layout.q_fp32, 4),
                (layout.k_fp32, 4),
                (layout.q_bf16, 2),
                (layout.k_bf16, 2),
            ] {
                if region.bytes == 0 {
                    continue;
                }
                let live = valid as usize * layout.qk_heads as usize * 128 * element_bytes;
                let start = region.offset as usize + live;
                let end = (region.offset + region.bytes) as usize;
                stream.memcpy_htod(
                    &vec![0xa5u8; end - start],
                    &mut workspace.slice_mut(start..end),
                )?;
            }
        }
        let p1 = if let GeometryCore::P1(split) = core {
            let (plan, metadata) = &plans[0];
            let meta = SeqRef {
                ptr: metadata.device_ptr(stream).0,
            };
            GdnOps::run(
                plan,
                stream,
                meta,
                ExecutionPhase::Prefill,
                Scratch { ptr, bytes },
            )?;
            stream.memcpy_htod(&geometry_state(spec.hv as usize), &mut buffers.state)?;
            let module = cubin::module(
                &device.ctx,
                &format!("gdn/gdn_chunkwise_p1{}", spec.kernel_suffix()?),
            )?;
            let function = module.load_function(&format!("gdn_chunkwise_p1_core_s{split}"))?;
            let shared_mem_bytes = match split {
                4 => crate::gdn_plan::cw_p1::S4_BYTES,
                8 => crate::gdn_plan::cw_p1::S8_BYTES,
                16 => crate::gdn_plan::cw_p1::S16_BYTES,
                _ => anyhow::bail!("invalid fixture P1 split"),
            };
            Some((
                function,
                LaunchConfig {
                    grid_dim: (split, spec.hv, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes,
                },
            ))
        } else {
            None
        };
        device.finish_upload()?;
        let execute = || -> Result<()> {
            for (plan, metadata) in &plans {
                let meta = SeqRef {
                    ptr: metadata.device_ptr(stream).0,
                };
                if let Some((function, config)) = &p1 {
                    unsafe {
                        stream
                            .launch_builder(function)
                            .arg(&(ptr + plan.ws.q_fp32.offset))
                            .arg(&(ptr + plan.ws.k_fp32.offset))
                            .arg(&(ptr + plan.ws.v_fp32.offset))
                            .arg(&(ptr + plan.ws.beta_fp32.offset))
                            .arg(&(ptr + plan.ws.g.offset))
                            .arg(&plan.bindings.s.0.ptr)
                            .arg(&plan.bindings.core_out.0.ptr)
                            .arg(&meta.ptr)
                            .arg(&0i32)
                            .launch(*config)?;
                    }
                } else if matches!(core, GeometryCore::ChunkedSnapshot) {
                    GdnOps::run_core(
                        plan,
                        stream,
                        meta,
                        ExecutionPhase::Prefill,
                        Scratch { ptr, bytes },
                        true,
                    )?;
                } else if matches!(core, GeometryCore::Verify | GeometryCore::RstepSnapshot) {
                    GdnOps::run_verify(plan, stream, meta, Scratch { ptr, bytes })?;
                } else {
                    GdnOps::run(
                        plan,
                        stream,
                        meta,
                        if sequential {
                            ExecutionPhase::Decode
                        } else {
                            ExecutionPhase::Prefill
                        },
                        Scratch { ptr, bytes },
                    )?;
                }
            }
            Ok(())
        };
        if graph {
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let run = execute();
            let captured = stream.end_capture(sys::CUgraphInstantiate_flags(0));
            run?;
            captured?.context("empty geometry graph")?.launch()?;
        } else {
            execute()?;
        }
        stream.synchronize()?;
        let plan = &plans.last().context("empty geometry plans")?.0;
        let data = stream.clone_dtoh(&workspace.slice(..plan.ws.bytes as usize))?;
        let checked_rows = if sequential { 1 } else { valid };
        let norm_ratio = geometry_qk_norm(&data, plan, checked_rows)?;
        if !sequential && valid < rows {
            for (region, element_bytes) in [
                (plan.ws.q_fp32, 4),
                (plan.ws.k_fp32, 4),
                (plan.ws.q_bf16, 2),
                (plan.ws.k_bf16, 2),
            ] {
                if region.bytes == 0 {
                    continue;
                }
                let live = valid as usize * plan.ws.qk_heads as usize * 128 * element_bytes;
                let start = region.offset as usize + live;
                let end = (region.offset + region.bytes) as usize;
                ensure!(
                    data[start..end].iter().all(|byte| *byte == 0xa5),
                    "GDN Q/K prep wrote inactive rows"
                );
            }
        }
        let out = stream.clone_dtoh(&buffers.out)?;
        let state = stream.clone_dtoh(&buffers.state)?;
        let history = stream.clone_dtoh(&buffers.history)?;
        let (prefix_states, prefix_histories) =
            if let Some((states, histories)) = &buffers.snapshots {
                (stream.clone_dtoh(states)?, stream.clone_dtoh(histories)?)
            } else {
                (Vec::new(), Vec::new())
            };
        ensure!(
            out[..valid as usize * spec.hv as usize * 128]
                .iter()
                .all(|v| v.is_finite())
                && state.iter().chain(&history).all(|v| v.is_finite()),
            "nonfinite geometry fixture output"
        );
        ensure!(
            out[valid as usize * spec.hv as usize * 128..]
                .iter()
                .all(|v| v.to_bits() == 0x7fc5),
            "geometry fixture wrote an inactive row"
        );
        let mut snapshot_mismatches = 0;
        let (p2_prepared, p2_slots) = if matches!(core, GeometryCore::P2 | GeometryCore::P2Expanded)
        {
            let mut prepared = data[plan.ws.conv_out.offset as usize
                ..plan.ws.conv_out.offset as usize + valid as usize * plan.conv_dim as usize * 2]
                .to_vec();
            for row in 0..valid as usize {
                for head in 0..spec.hv as usize {
                    let physical = if plan.ws.qk_heads == spec.hk {
                        head / (spec.hv / spec.hk) as usize
                    } else {
                        head
                    };
                    for d in 0..DK as usize {
                        let index = (row * plan.ws.qk_heads as usize + physical) * DK as usize + d;
                        for (full, rounded) in [
                            (plan.ws.q_fp32, plan.ws.q_bf16),
                            (plan.ws.k_fp32, plan.ws.k_bf16),
                        ] {
                            let offset = full.offset as usize + index * 4;
                            let bits = &data[offset..offset + 4];
                            prepared.extend_from_slice(bits);
                            if rounded.bytes > 0 {
                                let offset = rounded.offset as usize + index * 2;
                                prepared.extend_from_slice(&data[offset..offset + 2]);
                            } else {
                                prepared.extend_from_slice(
                                    &bf16::from_f32(f32::from_ne_bytes(bits.try_into()?))
                                        .to_bits()
                                        .to_ne_bytes(),
                                );
                            }
                        }
                    }
                }
            }
            let slots = stream.clone_dtoh(&workspace.slice(
                plan.ws.p2.offset as usize..(plan.ws.p2.offset + plan.ws.p2.bytes) as usize,
            ))?;
            (prepared, slots)
        } else {
            (Vec::new(), Vec::new())
        };
        if let Some((saved_state, saved_history)) = &buffers.snapshots {
            let states = stream.clone_dtoh(saved_state)?;
            let histories = stream.clone_dtoh(saved_history)?;
            let state_width = spec.hv as usize * 128 * 128;
            let history_width = spec.conv_dim()? as usize * 3;
            for prefix in 0..(spec.snapshot_limit()? - 1) as usize {
                let state = &states[prefix * state_width..(prefix + 1) * state_width];
                let history = &histories[prefix * history_width..(prefix + 1) * history_width];
                if prefix + 1 < valid as usize {
                    let reference = geometry_run(
                        device,
                        spec,
                        rows,
                        prefix as u32 + 1,
                        if matches!(core, GeometryCore::ChunkedSnapshot) {
                            GeometryCore::Chunked
                        } else {
                            GeometryCore::Recurrent
                        },
                        false,
                    )?;
                    ensure!(
                        state.iter().chain(history).all(|value| value.is_finite()),
                        "nonfinite geometry snapshot"
                    );
                    snapshot_mismatches += state
                        .iter()
                        .zip(&reference.state)
                        .chain(history.iter().zip(&reference.history))
                        .filter(|(actual, reference)| actual.to_bits() != reference.to_bits())
                        .count();
                } else {
                    snapshot_mismatches += state
                        .iter()
                        .chain(history)
                        .filter(|value| value.to_bits() != 0x7fc1_0001)
                        .count();
                }
            }
        }
        Ok(GeometryOutput {
            out,
            state,
            history,
            prefix_states,
            prefix_histories,
            norm_ratio,
            snapshot_mismatches,
            p2_prepared,
            p2_slots,
        })
    }

    fn invariant_gdn(device: &crate::backend::cuda::Device, spec: GdnSpec) -> Result<usize> {
        let rows = spec.snapshot_limit()?;
        let reference = (1..=rows)
            .map(|valid| geometry_run(device, spec, rows, valid, GeometryCore::Recurrent, false))
            .collect::<Result<Vec<_>>>()?;
        let state_words = (spec.hv * DK * DV) as usize;
        let history_words = (spec.conv_dim()? * spec.history_width) as usize;
        let mut defects = 0;
        let mut controls = 0;
        for valid in 1..=rows {
            for graph in [false, true] {
                take_phase_log();
                let candidate =
                    geometry_run(device, spec, rows, valid, GeometryCore::Verify, graph)?;
                let phases = take_phase_log();
                let ordered = phases.first().copied() == Some("rstep");
                let expected = &reference[valid as usize - 1];
                let mismatches = geometry_mismatches(&candidate, expected);
                defects += usize::from(mismatches != 0 || !ordered);
                println!(
                    "{} INVARIANT_GDN hv={} bound={rows} valid={valid} graph={graph} output_state_history_bit_mismatches={mismatches} ordered_rstep={ordered}",
                    if mismatches == 0 && ordered {
                        "PASS"
                    } else {
                        "DEFECT"
                    },
                    spec.hv
                );
                for (stage, slots, width) in [
                    ("state", &candidate.prefix_states, state_words),
                    ("history", &candidate.prefix_histories, history_words),
                ] {
                    let used = (valid as usize - 1) * width;
                    ensure!(
                        slots[..used].iter().all(|v| v.is_finite()),
                        "nonfinite GDN {stage} prefix"
                    );
                    ensure!(
                        slots[used..].iter().all(|v| v.to_bits() == 0x7fc1_0001),
                        "GDN {stage} wrote a dead snapshot"
                    );
                    for prefix in 1..valid as usize {
                        let expected = if stage == "state" {
                            &reference[prefix - 1].state
                        } else {
                            &reference[prefix - 1].history
                        };
                        let got = &slots[(prefix - 1) * width..prefix * width];
                        let mismatches = got
                            .iter()
                            .zip(expected)
                            .filter(|(a, b)| a.to_bits() != b.to_bits())
                            .count();
                        defects += usize::from(mismatches != 0);
                        println!(
                            "{} INVARIANT_GDN_PREFIX hv={} valid={valid} prefix={prefix} stage={stage} graph={graph} bit_mismatches={mismatches}",
                            if mismatches == 0 { "PASS" } else { "DEFECT" },
                            spec.hv
                        );
                    }
                }
                if valid == rows && !graph {
                    let wrong = &candidate.prefix_states[state_words..2 * state_words];
                    let mismatches = wrong
                        .iter()
                        .zip(&reference[0].state)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    ensure!(mismatches > 0, "wrong snapshot row control escaped");
                    controls += 1;
                    println!(
                        "PASS INVARIANT_GDN_CONTROL wrong_snapshot_row=true hv={} bit_mismatches={mismatches}",
                        spec.hv
                    );
                }
            }
        }
        println!(
            "INVARIANT_GDN_SUMMARY hv={} rows=1..{rows} eager_graph=true prefix_control={controls} defects={defects}",
            spec.hv
        );
        ensure!(controls == 1, "invariant GDN control escaped");
        Ok(defects)
    }

    fn geometry_qk_norm(data: &[u8], plan: &GdnPlan, rows: u32) -> Result<f64> {
        let spec = plan.spec;
        let read_bf16 = |offset: usize| {
            f64::from(
                bf16::from_bits(u16::from_ne_bytes([data[offset], data[offset + 1]])).to_f32(),
            )
        };
        let read_f32 = |offset: usize| {
            f64::from(f32::from_ne_bytes(
                data[offset..offset + 4].try_into().unwrap(),
            ))
        };
        // DK reduction roundings, rsqrt's 2-ulp error, query scaling and output multiplies.
        let relative_bound = super::super::gamma(spec.dk as usize + 8, 2f64.powi(-24));
        let mut worst_ratio = 0f64;
        for row in 0..rows as usize {
            for kh in 0..spec.hk as usize {
                for (side, region) in [plan.ws.q_fp32, plan.ws.k_fp32].into_iter().enumerate() {
                    let offset = plan.ws.conv_out.offset as usize
                        + (row * plan.conv_dim as usize
                            + (side * spec.hk as usize + kh) * spec.dk as usize)
                            * 2;
                    let sum: f64 = (0..spec.dk as usize)
                        .map(|d| read_bf16(offset + d * 2).powi(2))
                        .sum();
                    ensure!(sum.is_finite(), "nonfinite GDN Q/K reference");
                    let inv = (sum + f64::from(plan.l2_eps)).sqrt().recip();
                    let scale = if side == 0 {
                        f64::from(spec.dk).sqrt().recip()
                    } else {
                        1.0
                    };
                    for child in 0..spec.hv / spec.hk {
                        let head = if plan.ws.qk_heads == spec.hk {
                            kh
                        } else {
                            kh * (spec.hv / spec.hk) as usize + child as usize
                        };
                        for d in 0..spec.dk as usize {
                            let expected = read_bf16(offset + d * 2) * inv * scale;
                            let index =
                                (row * plan.ws.qk_heads as usize + head) * spec.dk as usize + d;
                            let actual = read_f32(region.offset as usize + index * 4);
                            ensure!(actual.is_finite(), "nonfinite GDN Q/K preparation");
                            let error = (actual - expected).abs();
                            let bound = expected.abs() * relative_bound;
                            let ratio = if bound == 0.0 {
                                if error == 0.0 { 0.0 } else { f64::INFINITY }
                            } else {
                                error / bound
                            };
                            worst_ratio = worst_ratio.max(ratio);
                        }
                    }
                }
            }
        }
        println!(
            "GDN_QK_FP32 rows={rows} hv={} relative_bound={relative_bound} worst_ratio={worst_ratio} {}",
            spec.hv,
            if worst_ratio <= 1.0 { "PASS" } else { "DEFECT" }
        );
        Ok(worst_ratio)
    }

    fn geometry_mismatches(a: &GeometryOutput, b: &GeometryOutput) -> usize {
        a.out
            .iter()
            .zip(&b.out)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count()
            + a.state
                .iter()
                .zip(&b.state)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count()
            + a.history
                .iter()
                .zip(&b.history)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count()
            + a.snapshot_mismatches
            + b.snapshot_mismatches
    }

    fn geometry_follow(
        device: &crate::backend::cuda::Device,
        spec: GdnSpec,
        prefix: u32,
        prior: &GeometryOutput,
    ) -> Result<Vec<u32>> {
        use cudarc::driver::DevicePtr;
        let stream = &device.stream;
        let mut buffers = GeometryBuffers::new(stream, spec, 1, 1)?;
        stream.memcpy_htod(&prior.state, &mut buffers.state)?;
        stream.memcpy_htod(&prior.history, &mut buffers.history)?;
        let plan = geometry_plan(device, spec, 1, 0, &mut buffers, GeometryCore::Recurrent)?;
        let workspace = stream.alloc_zeros::<u8>(plan.ws.bytes as usize)?;
        let meta = geometry_meta(stream, prefix, 1)?;
        GdnOps::run(
            &plan,
            stream,
            SeqRef {
                ptr: meta.device_ptr(stream).0,
            },
            ExecutionPhase::Decode,
            Scratch {
                ptr: workspace.device_ptr(stream).0,
                bytes: plan.ws.bytes,
            },
        )?;
        stream.synchronize()?;
        let out = stream.clone_dtoh(&buffers.out)?;
        ensure!(
            out[..spec.hv as usize * 128].iter().all(|v| v.is_finite()),
            "Runner FAIL nonfinite following decode output"
        );
        let mut words = out
            .into_iter()
            .map(|v| u32::from(v.to_bits()))
            .collect::<Vec<_>>();
        for buffer in [&buffers.state, &buffers.history] {
            let values = stream.clone_dtoh(buffer)?;
            ensure!(
                values.iter().all(|v| v.is_finite()),
                "Runner FAIL nonfinite following decode state"
            );
            words.extend(values.into_iter().map(f32::to_bits));
        }
        Ok(words)
    }

    #[test]
    #[ignore = "exclusive CUDA; bounded P2 against the original whole-span core"]
    fn gdn_p2_groups_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let device = crate::backend::cuda::Device::new(0)?;
        ensure!(
            GdnPolicy::for_device(&device.info).cw_p2_for_step(4096),
            "Runner FAIL P2 resource qualification"
        );
        let mut defects = 0;
        let mut cells = 0;
        for config in [
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/minicpm-config.json"
            )),
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/q38-config.json"
            )),
        ] {
            let root: serde_json::Value = serde_json::from_str(config)?;
            let model = if root["model_type"] == "minicpmv4_7" {
                crate::model_spec::ModelSpec::minicpm(
                    &root,
                    &serde_json::json!({"eos_token_id":[248044,248046]}),
                )?
            } else {
                crate::model_spec::ModelSpec::qwen(&root, &serde_json::Value::Null)?
            };
            let spec = GdnSpec::from_model(&model)?;
            let boundary = GdnWorkspace::new(&spec, 4096, true)?.p2_group_rows;
            let cases = [
                (5, 5),
                (63, 63),
                (64, 64),
                (65, 65),
                (127, 127),
                (128, 128),
                (4096, boundary - 1),
                (4096, boundary),
                (4096, boundary + 1),
                (4096, 4096),
            ];
            for (rows, valid) in cases {
                let whole = geometry_run(&device, spec, rows, valid, GeometryCore::P2Whole, false)?;
                let next = geometry_follow(&device, spec, 23 + valid, &whole)?;
                for graph in [false, true] {
                    let bounded =
                        geometry_run(&device, spec, rows, valid, GeometryCore::Selected, graph)?;
                    let out_mismatches = bounded
                        .out
                        .iter()
                        .zip(&whole.out)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    let state_mismatches = bounded
                        .state
                        .iter()
                        .zip(&whole.state)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    let history_mismatches = bounded
                        .history
                        .iter()
                        .zip(&whole.history)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    let follow = geometry_follow(&device, spec, 23 + valid, &bounded)?;
                    ensure!(
                        follow.len() == next.len(),
                        "Runner FAIL following decode extent"
                    );
                    let follow_mismatches =
                        follow.iter().zip(&next).filter(|(a, b)| a != b).count();
                    let passed =
                        out_mismatches + state_mismatches + history_mismatches + follow_mismatches
                            == 0;
                    defects += usize::from(!passed);
                    cells += 1;
                    println!(
                        "{} GDN_P2_GROUP_CELL hv={} input=[{rows},{}] state=[{},128,128] valid={valid} graph={graph} native_group_rows={boundary} output_bits={out_mismatches} state_bits={state_mismatches} history_bits={history_mismatches} following_decode_bits={follow_mismatches} tails=unchanged",
                        if passed { "PASS" } else { "DEFECT" },
                        spec.hv,
                        spec.conv_dim()?,
                        spec.hv
                    );
                }
            }
        }
        println!(
            "{} GDN_P2_GROUPS cells={cells} defects={defects} output_state_history_bits=exact following_decode=true graph_eager=true",
            if defects == 0 { "PASS" } else { "DEFECT" }
        );
        ensure!(defects == 0, "DEFECT bounded P2 cells={defects}");
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive CUDA window"]
    fn mixer_smoke_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let device = crate::backend::cuda::Device::new(0)?;
        let root = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/minicpm-config.json"
        )))?;
        let model = crate::model_spec::ModelSpec::minicpm(
            &root,
            &serde_json::json!({"eos_token_id": [248044, 248046]}),
        )?;
        let qwen_root = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/q38-config.json"
        )))?;
        let qwen = crate::model_spec::ModelSpec::qwen(&qwen_root, &serde_json::Value::Null)?;
        let mut defects = 0;
        for spec in [GdnSpec::from_model(&model)?, GdnSpec::from_model(&qwen)?] {
            defects += invariant_gdn(&device, spec)?;
            for core in [
                GeometryCore::Recurrent,
                GeometryCore::Chunked,
                GeometryCore::P2,
            ] {
                for graph in [false, true] {
                    let reference_core = if matches!(core, GeometryCore::P2) {
                        GeometryCore::P1(4)
                    } else {
                        core
                    };
                    let reference = geometry_run(&device, spec, 2, 2, reference_core, false)?;
                    let candidate = geometry_run(&device, spec, 2, 2, core, graph)?;
                    defects += usize::from(
                        geometry_mismatches(&candidate, &reference) != 0
                            || candidate.norm_ratio > 1.0
                            || reference.norm_ratio > 1.0,
                    );
                }
            }
            for rows in [1, 4, 5, 64] {
                for graph in [false, true] {
                    let reference_core = if rows > crate::gdn_plan::MIN_VERIFY_ROWS {
                        GeometryCore::P1(4)
                    } else {
                        GeometryCore::Chunked
                    };
                    let expected_phase = if rows > crate::gdn_plan::MIN_VERIFY_ROWS {
                        "chunkwise_p2"
                    } else {
                        "chunked"
                    };
                    let reference = geometry_run(&device, spec, rows, rows, reference_core, false)?;
                    take_phase_log();
                    let candidate =
                        geometry_run(&device, spec, rows, rows, GeometryCore::Selected, graph)?;
                    let phases = take_phase_log();
                    let mismatches = geometry_mismatches(&candidate, &reference);
                    let passed = mismatches == 0
                        && phases == [expected_phase]
                        && candidate.norm_ratio <= 1.0
                        && reference.norm_ratio <= 1.0;
                    defects += usize::from(!passed);
                    println!(
                        "{} GDN_SELECTED hv={} rows={rows} graph={graph} phase={phases:?} expected={expected_phase} mismatches={mismatches}",
                        if passed { "PASS" } else { "DEFECT" },
                        spec.hv
                    );
                }
            }
        }
        println!(
            "MIXER_SMOKE geometries=2 prefix=23 selected_rows=1,4,5,64 graph_modes=2 defects={defects}"
        );
        ensure!(defects == 0, "mixer SMOKE DEFECT cases={defects}");
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA and FF_Q38_PROBE_DIR/gdn_prep_h32_wrong_stride_sm<arch>.cubin"]
    fn mixer_geometry_gdn_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let device = crate::backend::cuda::Device::new(0)?;
        let root: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/minicpm-config.json"
        )))?;
        let model = crate::model_spec::ModelSpec::minicpm(
            &root,
            &serde_json::json!({"eos_token_id": [248044, 248046]}),
        )?;
        let spec = GdnSpec::from_model(&model)?;
        let qwen_root = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/q38-config.json"
        )))?;
        let qwen = crate::model_spec::ModelSpec::qwen(&qwen_root, &serde_json::Value::Null)?;
        let mut defects = 0;
        let mut cases = 0;
        for spec in [spec, GdnSpec::from_model(&qwen)?] {
            for (rows, valid) in [
                (1, 1),
                (2, 2),
                (5, 5),
                (5, 2),
                (7, 7),
                (8, 8),
                (9, 9),
                (31, 31),
                (32, 32),
                (33, 33),
                (37, 37),
                (64, 37),
                (63, 63),
                (64, 64),
                (65, 65),
                (127, 127),
                (128, 128),
                (129, 129),
                (512, 512),
                (512, 509),
                (4096, 4096),
            ] {
                let verify_tail = rows == 5 && valid == 2;
                let mut cores = if verify_tail {
                    vec![GeometryCore::Verify]
                } else {
                    vec![
                        GeometryCore::Chunked,
                        GeometryCore::Selected,
                        GeometryCore::P1(4),
                        GeometryCore::P1(8),
                        GeometryCore::P1(16),
                        GeometryCore::P2,
                    ]
                };
                if rows <= 8 && !verify_tail {
                    cores.push(GeometryCore::Recurrent);
                }
                if rows <= 5 && !verify_tail {
                    cores.push(GeometryCore::Verify);
                }
                for core in cores {
                    for graph in [false, true] {
                        let reference_core = match core {
                            GeometryCore::P2 => GeometryCore::P1(4),
                            GeometryCore::Selected if rows > crate::gdn_plan::MIN_VERIFY_ROWS => {
                                GeometryCore::P1(4)
                            }
                            GeometryCore::Selected => GeometryCore::Chunked,
                            GeometryCore::Verify => GeometryCore::Recurrent,
                            _ => core,
                        };
                        let reference =
                            geometry_run(&device, spec, rows, valid, reference_core, false)?;
                        let candidate = geometry_run(&device, spec, rows, valid, core, graph)?;
                        let mismatches = geometry_mismatches(&candidate, &reference);
                        let compact_mismatches = if matches!(core, GeometryCore::P2)
                            && spec.hv == 48
                            && rows > crate::gdn_plan::MIN_VERIFY_ROWS
                        {
                            let expanded = geometry_run(
                                &device,
                                spec,
                                rows,
                                valid,
                                GeometryCore::P2Expanded,
                                false,
                            )?;
                            let mismatches = geometry_mismatches(&candidate, &expanded)
                                + candidate
                                    .p2_prepared
                                    .iter()
                                    .zip(&expanded.p2_prepared)
                                    .filter(|(a, b)| a != b)
                                    .count()
                                + candidate
                                    .p2_slots
                                    .iter()
                                    .zip(&expanded.p2_slots)
                                    .filter(|(a, b)| a != b)
                                    .count();
                            ensure!(
                                candidate.p2_prepared.len() == expanded.p2_prepared.len()
                                    && candidate.p2_slots.len() == expanded.p2_slots.len(),
                                "compact P2 comparison extent differs"
                            );
                            println!(
                                "{} GDN_COMPACT_BITS rows={rows} valid={valid} graph={graph} prepared_bytes={} slot_bytes={} mismatches={mismatches}",
                                if mismatches == 0 { "PASS" } else { "DEFECT" },
                                candidate.p2_prepared.len(),
                                candidate.p2_slots.len()
                            );
                            mismatches
                        } else {
                            0
                        };
                        let passed = mismatches == 0
                            && compact_mismatches == 0
                            && candidate.norm_ratio <= 1.0
                            && reference.norm_ratio <= 1.0;
                        println!(
                            "GDN_GEOMETRY hv={} core={core:?} rows={rows} valid={valid} prefix=23 graph={graph} mismatches={mismatches} {}",
                            spec.hv,
                            if passed { "PASS" } else { "DEFECT" }
                        );
                        defects += usize::from(!passed);
                        cases += 1;
                    }
                }
            }
        }
        println!("GDN_GEOMETRY_SUMMARY cases={cases} defects={defects}");
        let reference = geometry_run(&device, spec, 37, 37, GeometryCore::Chunked, false)?;
        let control = geometry_run(&device, spec, 37, 37, GeometryCore::WrongHeadStride, false)?;
        let mismatches = geometry_mismatches(&control, &reference);
        ensure!(
            mismatches > 0 || control.norm_ratio > 1.0,
            "wrong-head-stride GDN control escaped"
        );
        println!(
            "PASS GDN_GEOMETRY_CONTROL wrong_head_stride rows=37 mismatches={mismatches} norm_ratio={}",
            control.norm_ratio
        );
        ensure!(defects == 0, "GDN_GEOMETRY DEFECT cases={defects}");
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive CUDA window"]
    fn gdn_operation_timing_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let device = crate::backend::cuda::Device::new(0)?;
        let stream = &device.stream;
        let mut cells = Vec::new();
        for config in [
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/q38-config.json"
            )),
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/minicpm-config.json"
            )),
        ] {
            let root: serde_json::Value = serde_json::from_str(config)?;
            let model = if root["model_type"] == "minicpmv4_7" {
                crate::model_spec::ModelSpec::minicpm(
                    &root,
                    &serde_json::json!({"eos_token_id": [248044, 248046]}),
                )?
            } else {
                crate::model_spec::ModelSpec::qwen(&root, &serde_json::Value::Null)?
            };
            let spec = GdnSpec::from_model(&model)?;
            for rows in [512, 4096] {
                cells.push((spec, rows, [GeometryCore::Chunked, GeometryCore::P2]));
            }
            for rows in 1..=5 {
                cells.push((
                    spec,
                    rows,
                    [GeometryCore::ChunkedSnapshot, GeometryCore::RstepSnapshot],
                ));
            }
        }
        let mut defects = 0;
        for &(spec, rows, cores) in &cells {
            for core in cores {
                let reference_core = match core {
                    GeometryCore::P2 => GeometryCore::P1(4),
                    GeometryCore::RstepSnapshot => GeometryCore::Recurrent,
                    _ => core,
                };
                let reference = geometry_run(&device, spec, rows, rows, reference_core, false)?;
                for graph in [false, true] {
                    let actual = geometry_run(&device, spec, rows, rows, core, graph)?;
                    let mismatches = geometry_mismatches(&actual, &reference);
                    let passed =
                        mismatches == 0 && actual.norm_ratio <= 1.0 && reference.norm_ratio <= 1.0;
                    defects += usize::from(!passed);
                    println!(
                        "{} GDN_DECISION_CORRECT hv={} rows={rows} core={core:?} graph={graph} mismatches={mismatches}",
                        if passed { "PASS" } else { "DEFECT" },
                        spec.hv
                    );
                }
            }
        }
        ensure!(
            defects == 0,
            "GDN_DECISION correctness DEFECT cells={defects}"
        );
        for (spec, rows, cores) in cells {
            let metadata = geometry_meta(stream, 23, rows)?;
            let meta = SeqRef {
                ptr: metadata.device_ptr(stream).0,
            };
            let initial_state = geometry_state(spec.hv as usize);
            let mut arms = Vec::new();
            for core in cores {
                let mut buffers = GeometryBuffers::new(stream, spec, rows, rows)?;
                let initial_history = stream.clone_dtoh(&buffers.history)?;
                let plan = geometry_plan(&device, spec, rows, 0, &mut buffers, core)?;
                let bytes = plan.ws.bytes;
                let workspace = stream.alloc_zeros::<u8>(bytes as usize)?;
                arms.push((core, buffers, plan, workspace, initial_history));
            }
            device.finish_upload()?;
            let mut ratios = Vec::new();
            for block in 0..6 {
                let mut elapsed = [0.0; 4];
                for (position, arm) in [0usize, 1, 1, 0].into_iter().enumerate() {
                    let (core, buffers, plan, workspace, initial_history) = &mut arms[arm];
                    let ptr = workspace.device_ptr(stream).0;
                    let bytes = workspace.len() as u64;
                    let mean_ms = crate::backend::cuda::time_prepared_operation(
                        stream,
                        || {
                            stream.memcpy_htod(&initial_state, &mut buffers.state)?;
                            stream.memcpy_htod(initial_history, &mut buffers.history)?;
                            Ok(())
                        },
                        || {
                            GdnOps::run_core(
                                plan,
                                stream,
                                meta,
                                ExecutionPhase::Prefill,
                                Scratch { ptr, bytes },
                                matches!(
                                    core,
                                    GeometryCore::ChunkedSnapshot | GeometryCore::RstepSnapshot
                                ),
                            )
                        },
                    )?;
                    ensure!(
                        mean_ms.is_finite() && mean_ms > 0.0,
                        "invalid GDN decision timing"
                    );
                    elapsed[position] = mean_ms;
                    println!(
                        "GDN_DECISION_SAMPLE {}",
                        serde_json::json!({
                            "arch":device.info.arch,"hv":spec.hv,"rows":rows,"core":format!("{core:?}"),
                            "block":block + 1,"position":position + 1,"arm":arm,"mean_ms":mean_ms,
                            "warm":16,"calls":32,"graph":false,"synthetic":true,
                        })
                    );
                }
                if block != 0 {
                    ratios.push(
                        ((elapsed[1] / elapsed[0]).ln() + (elapsed[2] / elapsed[3]).ln()) / 2.0,
                    );
                }
            }
            let mean = ratios.iter().sum::<f64>() / 5.0;
            let variance = ratios
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / 4.0;
            let half = 2.7764451051977987 * (variance / 5.0).sqrt();
            println!(
                "GDN_DECISION_RESULT {}",
                serde_json::json!({
                    "arch":device.info.arch,"hv":spec.hv,"rows":rows,
                    "A":format!("{:?}",cores[0]),"B":format!("{:?}",cores[1]),
                    "ratio_B_over_A":mean.exp(),"ci95":[(mean-half).exp(),(mean+half).exp()],
                    "blocks":6,"discard":1,"df":4,"graph":false,"synthetic":true,
                })
            );
        }
        println!("PASS GDN_DECISION completed_cells=14 graph=false synthetic=true");
        Ok(())
    }

    fn gate_projection_isolation(device: &crate::backend::cuda::Device) -> Result<()> {
        use crate::backend::cuda::{
            blas::Blas, closure::Closure, linear::LinearWeightRef, ops::Ops,
        };
        use safetensors::{Dtype, SafeTensors};
        let model = std::path::PathBuf::from(std::env::var("FF_Q38_MODEL")?);
        let root = std::path::PathBuf::from(std::env::var("FF_Q38_PROBE_DIR")?);
        let runtime = std::path::PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?);
        let spec = crate::model_spec::ModelSpec::read(&model)?;
        let gdn = GdnSpec::from_model(&spec)?;
        ensure!(
            spec.hidden == 5120 && (gdn.hk, gdn.hv, gdn.dk, gdn.dv) == (16, 48, 128, 128),
            "gate isolation requires the published Qwen geometry"
        );
        let mapped = crate::checkpoint::MappedCheckpoint::open(&model)?;
        let views = mapped.views()?;
        let read = |role: &str, shape: &[usize]| -> Result<Vec<bf16>> {
            let name = format!("model.language_model.layers.0.linear_attn.{role}");
            let shard = mapped
                .index
                .weight_map
                .get(&name)
                .context("gate weight missing")?;
            let tensor = views
                .get(shard)
                .context("gate shard missing")?
                .tensor(&name)?;
            ensure!(
                tensor.dtype() == Dtype::BF16 && tensor.shape() == shape,
                "gate weight layout {name}"
            );
            Ok(tensor
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| bf16::from_bits(u16::from_le_bytes(*v)))
                .collect())
        };
        let stream = &device.stream;
        let a_weight = stream.clone_htod(&read("in_proj_a.weight", &[48, 5120])?)?;
        let b_weight = stream.clone_htod(&read("in_proj_b.weight", &[48, 5120])?)?;
        let a_log = stream.clone_htod(&read("A_log", &[48])?)?;
        let dt = stream.clone_htod(&read("dt_bias", &[48])?)?;
        let trace_bytes = std::fs::read(root.join("scalar-stages.safetensors"))?;
        let trace = SafeTensors::deserialize(&trace_bytes)?;
        let hidden = trace.tensor("target.0.debug.norm1")?;
        ensure!(
            hidden.dtype() == Dtype::F32 && hidden.shape() == [1, 1, 5120],
            "layer0 norm1 trace layout"
        );
        let hidden = hidden
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| {
                let value = f32::from_le_bytes(*v);
                let rounded = bf16::from_f32(value);
                ensure!(
                    value.is_finite() && rounded.to_f32().to_bits() == value.to_bits(),
                    "trace is not an exact native BF16 widen"
                );
                Ok(rounded)
            })
            .collect::<Result<Vec<_>>>()?;
        let input = stream.clone_htod(&hidden.repeat(5))?;
        let ops = Ops::new(&device.ctx)?;
        let blas = Blas::new(
            device,
            &ops,
            &runtime,
            std::rc::Rc::new(Closure::new(false, "gate projection isolation".into())),
        )?;
        blas.prepare_fixed(1, 48, 5120)?;
        blas.prepare_fixed(5, 48, 5120)?;
        let gate = cubin::module(&device.ctx, "q38_common")?.load_function("q38_gdn_gates")?;
        let mut observations = Vec::new();
        for rows in [1u32, 5] {
            let a = stream.alloc_zeros::<bf16>(rows as usize * 48)?;
            let b = stream.alloc_zeros::<bf16>(rows as usize * 48)?;
            let beta = stream.alloc_zeros::<bf16>(rows as usize * 48)?;
            let g = stream.alloc_zeros::<f32>(rows as usize * 48)?;
            for (weight, output) in [(&a_weight, &a), (&b_weight, &b)] {
                blas.linear_typed(
                    if rows == 1 {
                        ExecutionPhase::Decode
                    } else {
                        ExecutionPhase::Prefill
                    },
                    LinearWeightRef::Bf16(geometry_view(weight, stream, [48, 5120], [5120, 1], 0)?),
                    geometry_view(&input, stream, [rows, 5120], [5120, 1], 0)?,
                    &mut TensorMut(geometry_view(output, stream, [rows, 48], [48, 1], 0)?),
                )?;
            }
            let meta = geometry_meta(stream, if rows == 1 { 68 } else { 67 }, rows)?;
            super::super::ops::GateOps::gdn_gates_with(
                &gate,
                stream,
                RowDomain {
                    meta: SeqRef {
                        ptr: meta.device_ptr(stream).0,
                    },
                    row_base: 0,
                    row_capacity: rows,
                },
                GdnGateBindings {
                    a: geometry_view(&a, stream, [rows, 48], [48, 1], 0)?,
                    b: geometry_view(&b, stream, [rows, 48], [48, 1], 0)?,
                    a_log: geometry_view(&a_log, stream, [48], [1], 0)?,
                    dt_bias: geometry_view(&dt, stream, [48], [1], 0)?,
                    beta: &mut TensorMut(geometry_view(&beta, stream, [rows, 48], [48, 1], 0)?),
                    g: &mut TensorMut(geometry_view(&g, stream, [rows, 48], [48, 1], 0)?),
                },
            )?;
            observations.push((
                stream.clone_dtoh(&a)?,
                stream.clone_dtoh(&b)?,
                stream.clone_dtoh(&beta)?,
                stream.clone_dtoh(&g)?,
            ));
        }
        let single = &observations[0];
        let multiple = &observations[1];
        for head in 0..48 {
            let row = 48 + head;
            ensure!(
                [
                    single.0[head].to_f32(),
                    single.1[head].to_f32(),
                    single.2[head].to_f32(),
                    single.3[head],
                    multiple.0[row].to_f32(),
                    multiple.1[row].to_f32(),
                    multiple.2[row].to_f32(),
                    multiple.3[row]
                ]
                .iter()
                .all(|v| v.is_finite()),
                "nonfinite gate projection"
            );
            println!(
                "T24.13B_GATE_PROJECTION {}",
                serde_json::json!({"head":head,"selected_row":1,"verify_rows":5,
                "a_m1":single.0[head].to_f32(),"a_m5":multiple.0[row].to_f32(),"a_equal":single.0[head].to_bits()==multiple.0[row].to_bits(),
                "b_m1":single.1[head].to_f32(),"b_m5":multiple.1[row].to_f32(),"b_equal":single.1[head].to_bits()==multiple.1[row].to_bits(),
                "beta_m1":single.2[head].to_f32(),"beta_m5":multiple.2[row].to_f32(),"beta_equal":single.2[head].to_bits()==multiple.2[row].to_bits(),
                "g_m1":single.3[head],"g_m5":multiple.3[row],"g_equal":single.3[head].to_bits()==multiple.3[row].to_bits()})
            );
        }
        println!(
            "T24.13B_PROJECTION_PLAN {}",
            serde_json::json!({"stored_weight":[48,5120],"input":[5,5120],"input_rows":"same selected normalized row repeated five times","bound_plans":blas.bound_plan_receipt()?})
        );
        Ok(())
    }

    #[test]
    #[ignore = "exclusive CUDA, published Qwen gate weights and saved layer0 normalized row"]
    fn gdn_verify_snapshot_rows_gpu() -> Result<()> {
        let qwen = GdnSpec::q38();
        let qwen_conv_dim = qwen.conv_dim().expect("Qwen GDN geometry");
        use cudarc::driver::{CudaSlice, DevicePtr};
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let device = crate::backend::cuda::Device::new(0)?;
        gate_projection_isolation(&device)?;
        let stream = &device.stream;
        let state_elements = (qwen.hv * DK * DV) as usize;
        let row_elements = (qwen.hv * DK) as usize;
        let signal =
            |i: usize, prime: usize| ((i * 17 % prime) as f32 - (prime / 2) as f32) / 1024.0;
        let initial: Vec<_> = (0..state_elements).map(|i| signal(i, 127)).collect();
        let q: Vec<_> = (0..5 * row_elements).map(|i| signal(i, 131)).collect();
        let k: Vec<_> = (0..5 * row_elements).map(|i| signal(i, 137)).collect();
        let v: Vec<_> = (0..5 * row_elements).map(|i| signal(i, 139)).collect();
        let beta: Vec<_> = (0..5 * qwen.hv as usize)
            .map(|i| 0.25 + (i * 17 % 53) as f32 / 128.0)
            .collect();
        let g: Vec<_> = (0..5 * qwen.hv as usize)
            .map(|i| -((i * 17 % 157 + 1) as f32) / 256.0)
            .collect();
        let q = stream.clone_htod(&q)?;
        let k = stream.clone_htod(&k)?;
        let v = stream.clone_htod(&v)?;
        let beta = stream.clone_htod(&beta)?;
        let g = stream.clone_htod(&g)?;
        let module = cubin::module(&device.ctx, "gdn/gdn_rstep")?;
        let snapshot_kernel = module.load_function("gdn_rstep_core_snapshot")?;
        let plain_kernel = cubin::module(&device.ctx, "gdn/gdn_recurrent")?
            .load_function("gdn_recurrent_core_v2")?;
        let config = LaunchConfig {
            grid_dim: (2, qwen.hv, 1),
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
                let bp = beta.device_ptr(stream).0 + (row * qwen.hv as usize * 4) as u64;
                let gp = g.device_ptr(stream).0 + (row * qwen.hv as usize * 4) as u64;
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
                        .launch(config)?;
                }
                let expected = stream.clone_dtoh(&reference)?;
                let observed = if row + 1 < rows {
                    &slots[row * state_elements..(row + 1) * state_elements]
                } else {
                    &final_state
                };
                for head in 0..qwen.hv as usize {
                    let start = head * DK as usize * DV as usize;
                    let end = start + DK as usize * DV as usize;
                    let mismatches = expected[start..end]
                        .iter()
                        .zip(&observed[start..end])
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    println!(
                        "{} GDN_DECODE_VERIFY_HEAD rows={rows} row={row} head={head} state_elements=16384 bit_mismatches={mismatches}",
                        if mismatches == 0 { "PASS" } else { "DEFECT" }
                    );
                }
                if rows == 5 && row == 1 {
                    let width = DK as usize * DV as usize;
                    let mismatches = expected[43 * width..44 * width]
                        .iter()
                        .zip(&observed[42 * width..43 * width])
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    ensure!(mismatches > 0, "head42-for-head43 control escaped");
                    println!(
                        "PASS GDN_HEAD_CONTROL rows=5 row=1 expected_head=43 wrong_head=42 bit_mismatches={mismatches}"
                    );
                }
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
        let conv_dim = qwen_conv_dim as usize;
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
                        grid_dim: (qwen_conv_dim / 256, 1, 1),
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
        let a = stream.clone_htod(
            &(0..5 * qwen.hv as usize)
                .map(|i| bf16::from_f32(signal(i, 163)))
                .collect::<Vec<_>>(),
        )?;
        let b = stream.clone_htod(
            &(0..5 * qwen.hv as usize)
                .map(|i| bf16::from_f32(signal(i, 167)))
                .collect::<Vec<_>>(),
        )?;
        let conv_weight = stream.clone_htod(&vec![bf16::from_f32(0.25); conv_dim * 4])?;
        let a_log = stream.alloc_zeros::<bf16>(qwen.hv as usize)?;
        let dt_bias = stream.alloc_zeros::<bf16>(qwen.hv as usize)?;
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
                qkv: matrix(&qkv, qwen_conv_dim)?,
                a: matrix(&a, qwen.hv)?,
                b: matrix(&b, qwen.hv)?,
                conv_weight: matrix(&conv_weight, 4)?,
                a_log: Tensor::new(
                    a_log.device_ptr(stream).0,
                    [qwen.hv],
                    [1],
                    u64::from(qwen.hv) * 2,
                )?,
                dt_bias: Tensor::new(
                    dt_bias.device_ptr(stream).0,
                    [qwen.hv],
                    [1],
                    u64::from(qwen.hv) * 2,
                )?,
                core_out: TensorMut(Tensor::new(
                    core_out.device_ptr(stream).0,
                    [5, qwen.hv, DV],
                    [u64::from(qwen.hv * DV), u64::from(DV), 1],
                    (5 * row_elements * 2) as u64,
                )?),
                s: TensorMut(Tensor::new(
                    state.device_ptr(stream).0,
                    [qwen.hv, DK, DV],
                    [u64::from(DK * DV), u64::from(DV), 1],
                    (state_elements * 4) as u64,
                )?),
                history: TensorMut(Tensor::new(
                    history.device_ptr(stream).0,
                    [qwen_conv_dim, 3],
                    [3, 1],
                    (conv_dim * 3 * 4) as u64,
                )?),
            }
        };
        let full = GdnOps::prepare(&device.ctx, &GdnSpec::q38(), 5, bindings, true)?;
        let ordinary = full.for_rows(5, None)?;
        ensure!(
            ordinary.cw_p2 && ordinary.ws.p2.bytes > 0,
            "M5 negative control did not carry P2"
        );
        let snapshot_views = unsafe {
            (
                Tensor::new(
                    snapshots.device_ptr(stream).0,
                    [4, qwen.hv, DK, DV],
                    [
                        u64::from(qwen.hv * DK * DV),
                        u64::from(DK * DV),
                        u64::from(DV),
                        1,
                    ],
                    (4 * state_elements * 4) as u64,
                )?,
                Tensor::new(
                    history_snapshots.device_ptr(stream).0,
                    [4, qwen_conv_dim, 3],
                    [u64::from(qwen_conv_dim * 3), 3, 1],
                    (4 * conv_dim * 3 * 4) as u64,
                )?,
            )
        };
        let plan = full.for_rows(5, Some(snapshot_views))?;
        ensure!(
            !plan.cw_p2 && plan.ws.p2.bytes == 0,
            "snapshot-bound M5 plan retained chunkwise dispatch"
        );
        ensure!(
            full.cw_p2 && full.ws.p2.bytes > 0,
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
                        [1, qwen.hv, DV],
                        [u64::from(qwen.hv * DV), u64::from(DV), 1],
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
            let reference = GdnOps::prepare(&device.ctx, &GdnSpec::q38(), 1, bindings, false)?;
            let meta =
                stream.clone_htod(&[17 + row as i32, 1, 18 + row as i32, 128, 1, 0, 0, 0])?;
            GdnOps::run(
                &reference,
                stream,
                SeqRef {
                    ptr: meta.device_ptr(stream).0,
                },
                ExecutionPhase::Decode,
                Scratch {
                    ptr: scratch.device_ptr(stream).0,
                    bytes: plan.ws.bytes,
                },
            )?;
            let expected = stream.clone_dtoh(&reference_state)?;
            let actual = if row < 4 {
                &bound_snapshots[row * state_elements..(row + 1) * state_elements]
            } else {
                &final_state
            };
            if row == 1 {
                let width = DK as usize * DV as usize;
                for head in 0..qwen.hv as usize {
                    let start = head * width;
                    let mismatches = expected[start..start + width]
                        .iter()
                        .zip(&actual[start..start + width])
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    println!(
                        "{} GDN_BOUND_DECODE_VERIFY_HEAD row=1 head={head} state_elements=16384 bit_mismatches={mismatches}",
                        if mismatches == 0 { "PASS" } else { "DEFECT" }
                    );
                }
            }
            ensure!(
                expected
                    .iter()
                    .zip(actual)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "snapshot state differs from prepared decode chain, row={row}"
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
        let ordinary_scratch = stream.alloc_zeros::<u8>(ordinary.ws.bytes as usize)?;
        GdnOps::run(
            &ordinary,
            stream,
            SeqRef {
                ptr: meta.device_ptr(stream).0,
            },
            ExecutionPhase::Prefill,
            Scratch {
                ptr: ordinary_scratch.device_ptr(stream).0,
                bytes: ordinary.ws.bytes,
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
            "GDN_VERIFY_M5_BINDING_PASS recurrent_snapshot_rows=4 retained_state_prefixes=4 raw_history_CPU ordinary_prefill_P2_unchanged prepared_decode_state_output_history_bitwise shared_plan_mutant_chunked ordinary_M5_P2_launched"
        );
        println!(
            "GDN_VERIFY_SNAPSHOT_PRIMITIVE_PASS rows1..5 state_bitwise_decode raw_history_CPU dead_slots M5_bound_plan"
        );
        Ok(())
    }

    fn align256(n: u64) -> u64 {
        (n + 255) & !255
    }

    #[test]
    fn workspace_regions_match_the_interface_table() {
        for hv in [32, 48] {
            let spec = GdnSpec {
                hv,
                ..GdnSpec::q38()
            };
            let conv_dim = u64::from(spec.conv_dim().expect("conv_dim"));
            for rows in [1u32, 64, 2048] {
                let ws = GdnWorkspace::new(&spec, rows, false).expect("workspace");
                let r = u64::from(rows);
                let head = u64::from(spec.hv);
                let d = u64::from(DV);

                let want = |cols: u64, elem: u64| align256(r * cols * elem);
                assert_eq!(
                    ws.conv_out.bytes,
                    want(conv_dim, 2),
                    "conv_out, rows={rows}"
                );
                assert_eq!(ws.beta.bytes, want(head, 2), "beta, rows={rows}");
                assert_eq!(ws.beta_fp32.bytes, want(head, 4), "beta_fp32, rows={rows}");
                assert_eq!(ws.g.bytes, want(head, 4), "g, rows={rows}");
                assert_eq!(ws.q_fp32.bytes, want(head * d, 4), "q_fp32, rows={rows}");
                assert_eq!(ws.k_fp32.bytes, want(head * d, 4), "k_fp32, rows={rows}");
                assert_eq!(ws.v_fp32.bytes, want(head * d, 4), "v_fp32, rows={rows}");

                // Regions within the Engine slice have 256 B alignment and do not overlap.
                let regions = [
                    ("conv_out", ws.conv_out),
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
    }

    /// `workspace()` is pure: same inputs, same bytes, no CUDA context.
    #[test]
    fn workspace_is_pure_and_matches_the_plan() {
        let spec = GdnSpec::q38();
        let a = GdnWorkspace::new(&spec, 2048, false).expect("workspace");
        let b = GdnWorkspace::new(&spec, 2048, false).expect("workspace");
        assert_eq!(a.bytes, b.bytes);
        assert_eq!(a.alignment, 256);
        assert_eq!(
            a.bytes,
            GdnWorkspace::new(&spec, 2048, false).expect("ws").bytes
        );
        // decode's plan is max_rows = 1; the Engine shares one scratch sized to the larger.
        let d = GdnWorkspace::new(&spec, 1, false).expect("decode workspace");
        assert!(
            d.bytes < a.bytes,
            "a 1-row plan must not exceed the prefill plan"
        );
    }

    /// A column window's capacity is `span - offset`, so extent validation rejects a V offset
    /// that overruns the final row.
    #[test]
    fn column_window_capacity_is_measured_from_the_window() {
        let qwen = GdnSpec::q38();
        let qwen_conv_dim = qwen.conv_dim().expect("Qwen GDN geometry");
        let cd = u64::from(qwen_conv_dim);
        let el = size_of::<bf16>() as u64;
        // ABI §6: Q at 0 width HK*DK, K next, V next width HV*DV, in bytes.
        let q_off = 0u64;
        let k_off = u64::from(qwen.hk * DK) * el;
        let v_off = 2 * u64::from(qwen.hk * DK) * el;
        assert_eq!(v_off, 8192, "V starts two key-head blocks in");
        // Wrong offset: V's own width added where K's belongs.
        let v_off_bug = q_off + u64::from(qwen.hv * DV) * el + k_off;
        assert_eq!(v_off_bug, 16384, "the historical bug sat 8192 B too far in");

        let builds = |rows: u32, off: u64, span: u64| -> bool {
            match span.checked_sub(off) {
                None => false,
                Some(cap) => unsafe {
                    Tensor::<bf16, 3>::new(0x1000, [rows, qwen.hv, DV], [cd, u64::from(DV), 1], cap)
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
        for (hv, stored_channels) in [(32, 8192), (48, 10240)] {
            let spec = GdnSpec {
                hv,
                ..GdnSpec::q38()
            };
            assert_eq!(spec.conv_dim().expect("conv_dim"), stored_channels);
        }
    }

    /// The chunkwise-P2 slot offsets are the kernels' `P2_OFF_*` values: each 256 B
    /// aligned, pairwise disjoint, filling a slot of exactly `GDN_CW_P2_SLOT_BYTES`, and
    /// the planner's byte count is that many slots per (chunk, head).
    #[test]
    fn cw_p2_slot_is_aligned_and_the_planner_sizes_it_exactly() -> Result<()> {
        let qwen = GdnSpec::q38();
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
            let slots = u64::from(rows.div_ceil(CW_P2_CHUNK)) * u64::from(qwen.hv);
            assert_eq!(
                cw_p2_bytes(qwen.hv, rows),
                slots * cw_p2::SLOT_BYTES,
                "rows={rows}"
            );
            let fused = GdnWorkspace::new(&spec, rows, false)?.bytes;
            assert_eq!(
                GdnPolicy::P2.workspace(&spec, rows)?.bytes,
                GdnWorkspace::new(&spec, rows, GdnPolicy::P2.cw_p2_for_step(rows))?.bytes,
                "rows={rows}"
            );
            assert_eq!(
                GdnWorkspace::new(&spec, rows, false)?.bytes,
                fused,
                "the split must not tax a plan that does not carry it, rows={rows}"
            );
            // Every slot sits inside the region the plan accounts for.
            assert!(
                cw_p2_bytes(qwen.hv, rows).is_multiple_of(256),
                "rows={rows}"
            );
        }
        Ok(())
    }

    /// The chunkwise-P2 grids derive from the bound rows: stage A one CTA per (chunk, head),
    /// stage H one per 32-wide value block independent of the step.
    #[test]
    fn cw_p2_grids_are_a_function_of_the_bound_rows() {
        let qwen = GdnSpec::q38();
        for (rows, want_chunks) in [(2048u32, 32u32), (16384, 256)] {
            let (a, h) = cw_p2_grids(rows, qwen.hv);
            assert_eq!(
                a.grid_dim,
                (want_chunks, qwen.hv, 1),
                "stage A grid, rows={rows}"
            );
            assert_eq!(
                h.grid_dim,
                (4, qwen.hv, 1),
                "stage H covers DV in 32-wide blocks, rows={rows}"
            );
            for cfg in [a, h] {
                assert_eq!(cfg.block_dim, (P2_THREADS, 1, 1));
                assert_eq!(cfg.shared_mem_bytes, cw_p2::SMEM_BYTES as u32);
            }
        }
        // The H grid is the same whatever the step is; the per-chunk grid is monotone.
        assert_eq!(
            cw_p2_grids(5, qwen.hv).0.grid_dim,
            cw_p2_grids(64, qwen.hv).0.grid_dim
        );
        assert!(cw_p2_grids(16384, qwen.hv).0.grid_dim.0 > cw_p2_grids(2048, qwen.hv).0.grid_dim.0);
        for rows in [5u32, 64, 2048, 16384] {
            assert_eq!(
                cw_p2_grids(rows, qwen.hv).1.grid_dim,
                cw_p2_grids(2048, qwen.hv).1.grid_dim
            );
        }
    }

    /// The chunkwise-P2 stage dependency contract as a CPU model: within one (chunk, head)
    /// slot, stage A writes W, U, qk, Gc, Be and Sc and stage H writes vn and Sn, no field
    /// has a second writer and no field is written twice for the same pair.
    #[test]
    fn every_cw_p2_slot_field_has_exactly_one_writing_stage() {
        let qwen = GdnSpec::q38();
        let rows = 2048u32;
        let slots = usize::try_from(rows.div_ceil(CW_P2_CHUNK)).expect("nchunk") * qwen.hv as usize;
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
        for chunk in 0..slots / qwen.hv as usize {
            for head in 0..qwen.hv as usize {
                let base = chunk * qwen.hv as usize + head;
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
        let qwen = GdnSpec::q38();
        let qwen_conv_dim = qwen.conv_dim().expect("Qwen GDN geometry");
        let cd = qwen_conv_dim;
        let bf = size_of::<bf16>() as u64;
        let f32b = size_of::<f32>() as u64;
        let r = u64::from(rows);
        let head = u64::from(qwen.hv);
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
                a: Tensor::new(0x1000, [rows, qwen.hv], [head, 1], padded(r * head * bf))
                    .expect("a"),
                b: Tensor::new(0x1000, [rows, qwen.hv], [head, 1], padded(r * head * bf))
                    .expect("b"),
                conv_weight: Tensor::new(0x1000, [cd, 4], [4, 1], padded(u64::from(cd) * 4 * bf))
                    .expect("conv_weight"),
                a_log: Tensor::new(0x1000, [qwen.hv], [1], padded(head * bf)).expect("a_log"),
                dt_bias: Tensor::new(0x1000, [qwen.hv], [1], padded(head * bf)).expect("dt_bias"),
                core_out: TensorMut::new(
                    0x1000,
                    [rows, qwen.hv, DV],
                    [head * d, d, 1],
                    padded(r * head * d * bf),
                )
                .expect("core_out"),
                s: TensorMut::new(
                    0x1000,
                    [qwen.hv, DK, DV],
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
        let qwen = GdnSpec::q38();
        let qwen_conv_dim = qwen.conv_dim().expect("Qwen GDN geometry");
        let rows = 4u32;
        check_bindings(&valid_bindings(rows), rows, qwen_conv_dim, qwen.hv)
            .expect("dense bindings must pass");
        let reject = |name: &str, mangle: &dyn Fn(&mut GdnBindings)| {
            let mut b = valid_bindings(rows);
            mangle(&mut b);
            let error = check_bindings(&b, rows, qwen_conv_dim, qwen.hv)
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
        let qwen = GdnSpec::q38();
        let qwen_conv_dim = qwen.conv_dim().expect("Qwen GDN geometry");
        let cd = u64::from(qwen_conv_dim);
        let el = size_of::<bf16>() as u64;
        // ABI §6: Q at 0 width HK*DK, K next, V next width HV*DV, in bytes.
        let v_off = 2 * u64::from(qwen.hk * DK) * el;
        assert_eq!(v_off, 8192, "V starts two key-head blocks in");
        // Wrong offset: V's own width added where K's belongs.
        let v_off_bug = u64::from(qwen.hv * DV) * el + u64::from(qwen.hk * DK) * el;
        assert_eq!(v_off_bug, 16384, "the historical bug sat 8192 B too far in");
        for rows in [1u32, 64] {
            let span = cd * el * u64::from(rows);
            let window = as3(0x1000, rows, qwen.hv, qwen_conv_dim, span, v_off).expect("V window");
            assert_eq!(window.shape, [rows, qwen.hv, DV]);
            assert_eq!(window.stride, [cd, u64::from(DV), 1]);
            assert!(
                as3(0x1000, rows, qwen.hv, qwen_conv_dim, span, v_off_bug).is_err(),
                "v_off {v_off_bug} must be rejected before any launch (rows={rows})"
            );
        }
    }
}
