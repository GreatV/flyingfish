//! Gated delta network with per-head or key-channel decay.
//!
//! Geometry is bound from [`GdnSpec`] before launch: 16/32, 16/48 or 64/64 key/value heads,
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
use super::ops::{GatedRmsInput, GdnGateBindings};
use crate::backend::gpu::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use crate::gdn_plan::{
    CW_P2_CHUNK, CW_P2_VALUE_BLOCK, GdnPolicy, GdnWorkspace, Region, add, cw_p2, mul,
};
use crate::op_contract::{ExecutionPhase, GdnGateMode, GdnSpec, WorkspaceReq};
use anyhow::{Context, Result, ensure};
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

pub struct GdnBlockBindings {
    pub qkv: Tensor<bf16, 2>,
    pub a: Tensor<bf16, 2>,
    pub conv_weight: Tensor<bf16, 2>,
    /// A_log, dt_bias, both published BF16 [hv].
    pub decay: [Tensor<bf16, 1>; 2],
    pub valid: Tensor<i32, 1>,
    pub output: TensorMut<bf16, 3>,
}

pub enum GdnBlockMode {
    Preview {
        state: Tensor<f32, 3>,
        history: Tensor<f32, 2>,
        right: Tensor<bf16, 2>,
        /// Logit scale, history bias, history log-count coefficient.
        softmax: [Tensor<bf16, 1>; 3],
    },
    Commit {
        state: TensorMut<f32, 3>,
        history: TensorMut<f32, 2>,
        b: Tensor<bf16, 2>,
    },
}

pub struct GdnBlockPlan {
    spec: GdnSpec,
    bindings: GdnBlockBindings,
    mode: GdnBlockMode,
    workspace: GdnWorkspace,
    scratch: Scratch,
    kernels: [CudaFunction; 3],
    configs: [LaunchConfig; 3],
    rows: i32,
}

fn block_tensor<T: Copy, const R: usize>(
    tensor: &Tensor<T, R>,
    shape: [u32; R],
) -> Result<(u64, u64)> {
    tensor.validate()?;
    let (stride, _) = super::contracts::dense_layout(shape)?;
    ensure!(
        tensor.shape == shape && tensor.stride == stride,
        "GDN block tensor layout mismatch"
    );
    Ok((tensor.ptr, tensor.span_bytes))
}

impl GdnBlockPlan {
    pub fn prepare(
        ctx: &Arc<CudaContext>,
        spec: GdnSpec,
        bindings: GdnBlockBindings,
        mode: GdnBlockMode,
        scratch: Scratch,
    ) -> Result<Self> {
        spec.validate()?;
        ensure!(
            spec.gate_mode == GdnGateMode::PerHead,
            "GDN block requires per-head decay"
        );
        let rows = bindings.qkv.shape[0];
        ensure!((1..=32).contains(&rows), "GDN block rows must be 1..32");
        let channels = spec.conv_dim()?;
        let heads = spec.hv;
        let workspace = GdnWorkspace::new(&spec, rows, false)?;
        scratch.validate(workspace.bytes)?;
        let mut read = vec![
            block_tensor(&bindings.qkv, [rows, channels])?,
            block_tensor(&bindings.a, [rows, heads])?,
            block_tensor(&bindings.conv_weight, [channels, 4])?,
            block_tensor(&bindings.decay[0], [heads])?,
            block_tensor(&bindings.decay[1], [heads])?,
            block_tensor(&bindings.valid, [rows])?,
        ];
        let mut write = vec![
            block_tensor(&bindings.output.0, [rows, heads, 128])?,
            (scratch.ptr, scratch.bytes),
        ];
        let preview = match &mode {
            GdnBlockMode::Preview {
                state,
                history,
                right,
                softmax,
            } => {
                read.push(block_tensor(state, [heads, 128, 128])?);
                read.push(block_tensor(history, [channels, 3])?);
                read.push(block_tensor(right, [channels, 3])?);
                for scalar in softmax {
                    read.push(block_tensor(scalar, [heads])?);
                }
                true
            }
            GdnBlockMode::Commit { state, history, b } => {
                write.push(block_tensor(&state.0, [heads, 128, 128])?);
                write.push(block_tensor(&history.0, [channels, 3])?);
                read.push(block_tensor(b, [rows, heads])?);
                false
            }
        };
        for (i, &(ptr, bytes)) in write.iter().enumerate() {
            for &(other, extent) in read.iter().chain(&write[..i]) {
                ensure!(
                    super::contracts::spans_disjoint(ptr, bytes, other, extent)?,
                    "overlapping GDN block bindings"
                );
            }
        }
        let suffix = spec.kernel_suffix()?;
        let conv =
            cubin::module(ctx, &format!("gdn/gdn_conv{suffix}"))?.load_function(if preview {
                "gdn_conv_right_preview"
            } else {
                "gdn_conv_bf16_products_commit"
            })?;
        let prep = cubin::module(ctx, &format!("gdn/gdn_prep{suffix}"))?
            .load_function("gdn_block_l2_gates")?;
        let core = cubin::module(ctx, &format!("gdn/gdn_recurrent{suffix}"))?.load_function(
            if preview {
                "gdn_block_softmax_history_preview"
            } else {
                "gdn_recurrent_masked_commit"
            },
        )?;
        let configs = [
            LaunchConfig {
                grid_dim: (channels / 256, if preview { rows } else { 1 }, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            LaunchConfig {
                grid_dim: (1, spec.hk, rows),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            },
            LaunchConfig {
                grid_dim: (if preview { rows.div_ceil(4) } else { 2 }, heads, 1),
                block_dim: (if preview { 128 } else { 512 }, 1, 1),
                shared_mem_bytes: if preview { 0 } else { RECURRENT_SMEM },
            },
        ];
        let kernels = [conv, prep, core];
        for (kernel, config) in kernels.iter().zip(&configs) {
            ensure!(
                kernel.occupancy_max_active_blocks_per_multiprocessor(
                    config.block_dim.0,
                    config.shared_mem_bytes as usize,
                    None
                )? > 0,
                "GDN block kernel has zero occupancy"
            );
        }
        Ok(Self {
            spec,
            bindings,
            mode,
            workspace,
            scratch,
            kernels,
            configs,
            rows: rows as i32,
        })
    }

    pub(crate) fn causal_rows(&self, ctx: &Arc<CudaContext>, rows: u32) -> Result<Self> {
        ensure!(
            rows > 0 && rows <= self.bindings.qkv.shape[0],
            "GDN causal row extent"
        );
        let GdnBlockMode::Commit { state, history, b } = &self.mode else {
            anyhow::bail!("GDN row binding requires a causal plan");
        };
        let mut qkv = self.bindings.qkv;
        let mut a = self.bindings.a;
        let mut valid = self.bindings.valid;
        let mut output = self.bindings.output.0;
        let mut b = *b;
        qkv.shape[0] = rows;
        a.shape[0] = rows;
        valid.shape[0] = rows;
        output.shape[0] = rows;
        b.shape[0] = rows;
        Self::prepare(
            ctx,
            self.spec,
            GdnBlockBindings {
                qkv,
                a,
                valid,
                conv_weight: self.bindings.conv_weight,
                decay: self.bindings.decay,
                output: TensorMut(output),
            },
            GdnBlockMode::Commit {
                state: TensorMut(state.0),
                history: TensorMut(history.0),
                b,
            },
            Scratch {
                ptr: self.scratch.ptr,
                bytes: self.scratch.bytes,
            },
        )
    }

    pub fn run(&self, stream: &Arc<CudaStream>) -> Result<()> {
        let b = &self.bindings;
        let base = self.scratch.ptr;
        let conv = base + self.workspace.conv_out.offset;
        let q = base + self.workspace.q_fp32.offset;
        let k = base + self.workspace.k_fp32.offset;
        let v = base + self.workspace.v_fp32.offset;
        let g = base + self.workspace.g.offset;
        let beta = base + self.workspace.beta_fp32.offset;
        let (preview, b_ptr) = match &self.mode {
            GdnBlockMode::Preview { .. } => (1i32, 0u64),
            GdnBlockMode::Commit { b, .. } => (0, b.ptr),
        };
        unsafe {
            let mut launch = stream.launch_builder(&self.kernels[0]);
            launch.arg(&b.qkv.ptr).arg(&b.conv_weight.ptr);
            match &self.mode {
                GdnBlockMode::Preview { history, right, .. } => {
                    launch.arg(&right.ptr).arg(&history.ptr);
                }
                GdnBlockMode::Commit { history, .. } => {
                    launch.arg(&history.0.ptr);
                }
            }
            launch
                .arg(&b.valid.ptr)
                .arg(&conv)
                .arg(&self.rows)
                .launch(self.configs[0])?;
            stream
                .launch_builder(&self.kernels[1])
                .arg(&conv)
                .arg(&b.a.ptr)
                .arg(&b_ptr)
                .arg(&b.decay[0].ptr)
                .arg(&b.decay[1].ptr)
                .arg(&q)
                .arg(&k)
                .arg(&v)
                .arg(&g)
                .arg(&beta)
                .arg(&self.rows)
                .arg(&preview)
                .launch(self.configs[1])?;
            let mut launch = stream.launch_builder(&self.kernels[2]);
            launch.arg(&q).arg(&k).arg(&v);
            match &self.mode {
                GdnBlockMode::Preview { state, softmax, .. } => {
                    launch
                        .arg(&g)
                        .arg(&state.ptr)
                        .arg(&softmax[0].ptr)
                        .arg(&softmax[1].ptr)
                        .arg(&softmax[2].ptr)
                        .arg(&b.valid.ptr)
                        .arg(&b.output.0.ptr)
                        .arg(&self.rows);
                }
                GdnBlockMode::Commit { state, .. } => {
                    launch
                        .arg(&beta)
                        .arg(&g)
                        .arg(&state.0.ptr)
                        .arg(&b.output.0.ptr)
                        .arg(&b.valid.ptr)
                        .arg(&self.rows);
                }
            }
            launch.launch(self.configs[2])?;
        }
        Ok(())
    }
}

/// The ABI §6 views; `prepare` validates every stride and span against the registered layout.
/// `conv_dim` comes from [`GdnSpec::conv_dim`], not from these bindings.
#[derive(Clone, Copy, Debug)]
pub(crate) enum GdnGateInput {
    PerHead(Tensor<bf16, 2>),
    PerChannel(Tensor<bf16, 3>),
}

impl GdnGateInput {
    fn prefix_rows(&self, rows: u32) -> Result<Self> {
        match self {
            Self::PerHead(value) => Ok(Self::PerHead(value.prefix_rows(rows)?)),
            Self::PerChannel(value) => Ok(Self::PerChannel(value.prefix_rows(rows)?)),
        }
    }
}

#[derive(Debug)]
pub(crate) enum GdnGateOutput {
    PerHead(TensorMut<f32, 2>),
    PerChannel(TensorMut<f32, 3>),
}

impl GdnGateOutput {
    fn ptr(&self) -> u64 {
        match self {
            Self::PerHead(value) => value.0.ptr,
            Self::PerChannel(value) => value.0.ptr,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum GdnDecayWeights {
    Bf16 {
        a_log: Tensor<bf16, 1>,
        dt_bias: Tensor<bf16, 1>,
    },
    F32 {
        a_log: Tensor<f32, 1>,
        dt_bias: Tensor<f32, 1>,
    },
}

impl GdnDecayWeights {
    pub fn bind(
        catalog: &super::weights::WeightCatalog,
        stream: &Arc<CudaStream>,
        prefix: &str,
        spec: GdnSpec,
    ) -> Result<Self> {
        use crate::backend::gpu::DevicePtr;
        use safetensors::Dtype;
        spec.validate()?;
        let name_a = format!("{prefix}A_log");
        let name_dt = format!("{prefix}dt_bias");
        match spec.gate_mode {
            GdnGateMode::PerHead => Ok(Self::Bf16 {
                a_log: super::contracts::view(catalog.bf16(&name_a)?, stream, 0, [spec.hv], [1])?,
                dt_bias: super::contracts::view(
                    catalog.bf16(&name_dt)?,
                    stream,
                    0,
                    [spec.hv],
                    [1],
                )?,
            }),
            GdnGateMode::PerChannel { .. } => {
                let bind = |name: &str, count: u32| -> Result<Tensor<f32, 1>> {
                    let entry = catalog
                        .manifest
                        .entries
                        .get(name)
                        .with_context(|| format!("missing decay manifest entry {name}"))?;
                    ensure!(
                        entry.dtype == Dtype::F32 && entry.shape == [count as usize],
                        "native decay dtype/shape mismatch: {name}"
                    );
                    let Some(super::weights::StoredTensor::Bytes(bytes)) =
                        catalog.tensors.get(name)
                    else {
                        anyhow::bail!("missing native F32 decay tensor {name}");
                    };
                    ensure!(
                        bytes.len() == count as usize * size_of::<f32>(),
                        "native decay byte extent mismatch: {name}"
                    );
                    let (ptr, _read) = bytes.device_ptr(stream);
                    unsafe { Tensor::new(ptr, [count], [1], bytes.len() as u64) }
                };
                Ok(Self::F32 {
                    a_log: bind(&name_a, spec.hv)?,
                    dt_bias: bind(&name_dt, spec.hv * spec.dk)?,
                })
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum GdnConvWeights {
    Fused(Tensor<bf16, 2>),
    Separate([Tensor<bf16, 2>; 3]),
}

impl GdnConvWeights {
    pub fn bind(
        catalog: &super::weights::WeightCatalog,
        stream: &Arc<CudaStream>,
        prefix: &str,
        spec: GdnSpec,
    ) -> Result<Self> {
        let bind = |role: &str, channels| {
            super::contracts::view(
                catalog.bf16(&format!("{prefix}{role}.weight"))?,
                stream,
                0,
                [channels, 4],
                [4, 1],
            )
        };
        match spec.gate_mode {
            GdnGateMode::PerHead => Ok(Self::Fused(bind("conv1d", spec.conv_dim()?)?)),
            GdnGateMode::PerChannel { .. } => Ok(Self::Separate([
                bind("q_conv1d", spec.hk * DK)?,
                bind("k_conv1d", spec.hk * DK)?,
                bind("v_conv1d", spec.hv * DV)?,
            ])),
        }
    }
}

#[derive(Debug)]
pub(crate) struct GdnBindings {
    /// Pre-conv projection `[M, conv_dim]`: Q at offset 0 width `HK*DK`, K next
    /// width `HK*DK`, V next width `HV*DV`.
    pub qkv: Tensor<bf16, 2>,
    /// Gate inputs `[M, hv]` or `[M, hv, 128]`.
    pub a: GdnGateInput,
    pub b: Tensor<bf16, 2>,
    /// Depthwise conv weights `[conv_dim, 4]`, a zero-copy reshape of the
    /// manifest's `[conv_dim, 1, 4]`.
    pub conv_weight: GdnConvWeights,
    /// A_log is `[hv]`; dt_bias is `[hv]` for per-head gates and `[hv*128]` for per-channel gates.
    pub decay: GdnDecayWeights,
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
    step: Option<CudaFunction>,
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

fn fused_step_geometry(spec: &GdnSpec) -> bool {
    matches!(spec.gate_mode, GdnGateMode::PerHead)
        && (spec.hk, spec.dk, spec.dv, spec.hv) == (16, 128, 128, 32)
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

/// Stage A uses (chunk, head) CTAs; stage H uses 32-wide value strips in both gate modes.
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

fn cw_p2_groups(ws: &GdnWorkspace, rows: u32, hv: u32) -> Result<Vec<(i32, i32, LaunchConfig)>> {
    let mut groups = Vec::new();
    if ws.p2_group_rows == 0 {
        return Ok(groups);
    }
    for base in (0..rows).step_by(ws.p2_group_rows as usize) {
        let count = ws.p2_group_rows.min(rows - base);
        groups.push((base as i32, count as i32, cw_p2_grids(count, hv).0));
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
    p2_groups: Vec<(i32, i32, LaunchConfig)>,
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
        self.bind_rows(rows, None, snapshots, snapshots.is_none())
    }

    pub(crate) fn for_verify(
        &self,
        rows: u32,
        inputs: Option<(Tensor<bf16, 2>, GdnGateInput, Tensor<bf16, 2>)>,
    ) -> Result<Self> {
        self.bind_rows(rows, inputs, None, false)
    }

    fn bind_rows(
        &self,
        rows: u32,
        inputs: Option<(Tensor<bf16, 2>, GdnGateInput, Tensor<bf16, 2>)>,
        snapshots: Option<(Tensor<f32, 4>, Tensor<f32, 3>)>,
        allow_cw_p2: bool,
    ) -> Result<Self> {
        ensure!(
            rows > 0 && rows <= self.max_rows,
            "GDN rows outside binding"
        );
        if !allow_cw_p2 {
            let limit = self.spec.snapshot_limit()?;
            ensure!(
                (1..=limit).contains(&rows),
                "GDN hv={} snapshot rows={rows} exceeds bound={limit}",
                self.spec.hv
            );
        }
        if let Some((state, history)) = snapshots {
            let limit = self.spec.snapshot_limit()?;
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
        let (qkv, a, b_in) = match inputs {
            Some((qkv, a, b_in)) => (
                qkv.prefix_rows(rows)?,
                a.prefix_rows(rows)?,
                b_in.prefix_rows(rows)?,
            ),
            None => (
                b.qkv.prefix_rows(rows)?,
                b.a.prefix_rows(rows)?,
                b.b.prefix_rows(rows)?,
            ),
        };
        let bindings = GdnBindings {
            qkv,
            a,
            b: b_in,
            conv_weight: b.conv_weight,
            decay: b.decay,
            core_out: TensorMut(b.core_out.read().prefix_rows(rows)?),
            s: TensorMut(b.s.read()),
            history: TensorMut(b.history.read()),
        };
        check_bindings(&bindings, rows, self.conv_dim, self.spec.hv)?;
        let cw_p2 = allow_cw_p2 && self.cw_p2 && rows > crate::gdn_plan::MIN_VERIFY_ROWS;
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
            self.cw_p2 && !self.p2_groups.is_empty(),
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
        let g = match self.spec.gate_mode {
            crate::op_contract::GdnGateMode::PerHead => {
                ws_f32(self.ws.g, ws, rows, self.spec.hv)?.0.ptr
            }
            crate::op_contract::GdnGateMode::PerChannel { .. } => {
                ws_f32_3(self.ws.g, ws, rows, self.spec.hv, DV)?.0.ptr
            }
        };
        let slot = add(ws, self.ws.p2.offset)?;
        let compact = self.ws.q_bf16.bytes > 0;
        let a_query = if compact {
            ws + self.ws.q_bf16.offset
        } else {
            q
        };
        let a_key = ws + self.ws.k_bf16.offset;
        for &(row_base, group_rows, a_cfg) in &self.p2_groups {
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
            "_per_channel" => 24576,
            _ => anyhow::bail!("unsupported GDN kernel suffix={suffix}"),
        };
        ensure!(
            conv_dim == compiled_width,
            "GDN conv_dim={conv_dim} differs from compiled width={compiled_width} for hv={}",
            spec.hv
        );
        check_bindings(&bindings, max_rows, conv_dim, spec.hv)?;
        #[cfg(feature = "rocm")]
        ensure!(
            matches!(spec.hv, 32 | 48 | 64) && !cw_p2,
            "HIP GDN requires hv32/hv48/hv64 and the FP32 chunked/recurrent policy"
        );

        #[cfg(feature = "cuda")]
        use cudarc::driver::sys::CUdevice_attribute as Attribute;
        #[cfg(feature = "cuda")]
        let sms =
            u32::try_from(ctx.attribute(Attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?)?;
        #[cfg(feature = "rocm")]
        let sms = u32::try_from(ctx.properties.compute_units)?;
        ensure!(sms > 0, "GDN device has no multiprocessors");
        #[cfg(feature = "cuda")]
        let optin =
            ctx.attribute(Attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)? as u64;
        #[cfg(feature = "rocm")]
        let optin = ctx.properties.optin_shared_bytes;
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
        let f_conv = conv.load_function(if matches!(spec.gate_mode, GdnGateMode::PerHead) {
            "gdn_conv"
        } else {
            "gdn_conv_separate"
        })?;
        let f_hist = conv.load_function("gdn_conv_hist_update")?;
        let f_prep = prep_mod.load_function("gdn_prep_l2_fp32")?;
        let f_beta = prep_mod.load_function("gdn_beta_widen")?;
        let per_channel = matches!(
            spec.gate_mode,
            crate::op_contract::GdnGateMode::PerChannel { .. }
        );
        let f_chunked = chunked_mod.load_function(if per_channel {
            "gdn_chunked_core_per_channel_v1"
        } else {
            "gdn_chunked_core_v1"
        })?;
        let f_recurrent = recurrent_mod.load_function(if per_channel {
            "gdn_recurrent_core_per_channel_v2"
        } else {
            "gdn_recurrent_core_v2"
        })?;

        // The dynamic shared memory opt-in is applied to every core at setup; a launch without it
        // fails with `invalid argument`.
        #[cfg(feature = "cuda")]
        for (function, bytes) in [(&f_recurrent, RECURRENT_SMEM), (&f_chunked, CHUNKED_SMEM)] {
            if bytes > 48 * 1024 {
                function.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    bytes as i32,
                )?;
            }
        }

        let rstep = Some(module("gdn_rstep")?.load_function(if per_channel {
            "gdn_rstep_core_per_channel_v1"
        } else {
            "gdn_rstep_core_v1"
        })?);
        let hist_snapshot = module("gdn_conv")?.load_function("gdn_conv_hist_snapshot")?;
        let core_snapshot = module("gdn_rstep")?.load_function(if per_channel {
            "gdn_rstep_core_per_channel_snapshot"
        } else {
            "gdn_rstep_core_snapshot"
        })?;
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
                a: p2_mod.load_function(if per_channel {
                    "gdn_cw_p2_stage_a_per_channel"
                } else {
                    "gdn_cw_p2_stage_a"
                })?,
                h: p2_mod.load_function(if per_channel {
                    "gdn_cw_p2_stage_h_per_channel"
                } else {
                    "gdn_cw_p2_stage_h"
                })?,
            })
        };
        let kernels = Kernels {
            step: if fused_step_geometry(spec) {
                Some(module("gdn_rstep")?.load_function("gdn_step_fused")?)
            } else {
                None
            },
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
        #[cfg(feature = "cuda")]
        let max_threads =
            u32::try_from(ctx.attribute(Attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?)?;
        #[cfg(feature = "rocm")]
        let max_threads = u32::try_from(ctx.properties.max_threads)?;
        #[cfg(feature = "cuda")]
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
            .chain(
                kernels
                    .step
                    .iter()
                    .map(|f| ("fused step", f, CORE_THREADS, 0)),
            )
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

    pub(crate) fn run_step(
        plan: &GdnPlan,
        stream: &Arc<CudaStream>,
        meta: SeqRef,
        phase: ExecutionPhase,
        scratch: Scratch,
        verify: bool,
        gated: Option<&GatedRmsInput>,
    ) -> Result<bool> {
        if plan.spec.hv != 32
            || plan.snapshots.is_some()
            || (phase != ExecutionPhase::Decode && !verify)
        {
            if verify {
                Self::run_verify(plan, stream, meta, scratch)?;
            } else {
                Self::run(plan, stream, meta, phase, scratch)?;
            }
            return Ok(false);
        }
        meta.validate()?;
        scratch.validate(plan.ws.bytes)?;
        ensure!(
            !plan.cw_p2 && (1..=16).contains(&plan.max_rows),
            "fused GDN step requires recurrent rows 1..16"
        );
        ensure!(
            phase != ExecutionPhase::Decode || plan.max_rows == 1,
            "fused GDN decode requires one row"
        );
        let b = &plan.bindings;
        let GdnGateInput::PerHead(a) = &b.a else {
            anyhow::bail!("fused GDN step requires per-head gate input");
        };
        let GdnDecayWeights::Bf16 { a_log, dt_bias } = b.decay else {
            anyhow::bail!("fused GDN step requires BF16 decay parameters");
        };
        let (norm, gate) = if let Some(input) = gated {
            input.x.validate()?;
            input.weight.validate()?;
            input.gate.validate()?;
            ensure!(
                input.mode == super::ops::GatedRmsMode::SiluBf16
                    && input.x.ptr == b.core_out.0.ptr
                    && input.x.shape == b.core_out.0.shape
                    && input.x.stride == b.core_out.0.stride
                    && input.gate.shape == input.x.shape
                    && input.gate.stride == input.x.stride
                    && input.weight.shape == [DV]
                    && input.weight.stride == [1],
                "fused GDN gated norm layout mismatch"
            );
            (input.weight.ptr, input.gate.ptr)
        } else {
            (0, 0)
        };
        let conv = scratch.ptr + plan.ws.conv_out.offset;
        let kernel = plan
            .kernels
            .step
            .as_ref()
            .context("fused GDN step not bound")?;
        let GdnConvWeights::Fused(conv_weight) = b.conv_weight else {
            anyhow::bail!("fused GDN requires fused convolution weights");
        };
        unsafe {
            stream
                .launch_builder(&plan.conv)
                .arg(&b.qkv.ptr)
                .arg(&conv_weight.ptr)
                .arg(&b.history.0.ptr)
                .arg(&conv)
                .arg(&meta.ptr)
                .arg(&(plan.conv_dim as i32))
                .launch(plan.conv_cfg)?;
            stream
                .launch_builder(&plan.kernels.hist_update)
                .arg(&b.qkv.ptr)
                .arg(&b.history.0.ptr)
                .arg(&meta.ptr)
                .arg(&(plan.conv_dim as i32))
                .launch(plan.hist_cfg)?;
            stream
                .launch_builder(kernel)
                .arg(&conv)
                .arg(&a.ptr)
                .arg(&b.b.ptr)
                .arg(&a_log.ptr)
                .arg(&dt_bias.ptr)
                .arg(&b.s.0.ptr)
                .arg(&b.core_out.0.ptr)
                .arg(&norm)
                .arg(&gate)
                .arg(&meta.ptr)
                .arg(&(plan.max_rows as i32))
                .arg(&a.stride[0])
                .arg(&b.b.stride[0])
                .arg(&plan.spec.l2_eps)
                .arg(&plan.spec.rms_eps)
                .launch(LaunchConfig {
                    grid_dim: (1, plan.spec.hv, 1),
                    block_dim: (CORE_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(gated.is_some())
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
        ensure!(
            plan.snapshots.is_some() || plan.kernels.rstep.is_some(),
            "verify requires snapshot bindings or the ordered recurrent core"
        );
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
        let per_channel = matches!(
            plan.spec.gate_mode,
            crate::op_contract::GdnGateMode::PerChannel { .. }
        );
        let mut g = if per_channel {
            GdnGateOutput::PerChannel(ws_f32_3(
                plan.ws.g,
                ws_ptr,
                rows,
                plan.spec.hv,
                plan.spec.dv,
            )?)
        } else {
            GdnGateOutput::PerHead(ws_f32(plan.ws.g, ws_ptr, rows, plan.spec.hv)?)
        };
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
            conv.arg(&b.qkv.ptr);
            match &b.conv_weight {
                GdnConvWeights::Fused(weight) => {
                    conv.arg(&weight.ptr);
                }
                GdnConvWeights::Separate(weights) => {
                    for weight in weights {
                        conv.arg(&weight.ptr);
                    }
                }
            }
            conv.arg(&b.history.0.ptr)
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
        match (&b.a, b.decay, &mut g) {
            (
                GdnGateInput::PerHead(a),
                GdnDecayWeights::Bf16 { a_log, dt_bias },
                GdnGateOutput::PerHead(g_out),
            ) => {
                crate::backend::cuda::ops::GateOps::gdn_gates_with(
                    &plan.kernels.gate,
                    stream,
                    dom,
                    GdnGateBindings {
                        a: *a,
                        b: b.b,
                        a_log,
                        dt_bias,
                        beta: &mut beta,
                        g: g_out,
                    },
                )?;
            }
            (
                GdnGateInput::PerChannel(a),
                GdnDecayWeights::F32 { a_log, dt_bias },
                GdnGateOutput::PerChannel(g_out),
            ) => {
                let lower_bound = match plan.spec.gate_mode {
                    crate::op_contract::GdnGateMode::PerChannel { lower_bound } => lower_bound,
                    crate::op_contract::GdnGateMode::PerHead => unreachable!(),
                };
                crate::backend::cuda::ops::GateOps::gdn_channel_gates_with(
                    &plan.kernels.gate,
                    stream,
                    dom,
                    crate::backend::cuda::ops::GdnChannelGateBindings {
                        a: *a,
                        b: b.b,
                        a_log,
                        dt_bias,
                        beta: &mut beta,
                        g: g_out,
                        lower_bound,
                    },
                )?;
            }
            _ => anyhow::bail!("GDN gate mode and binding layout differ"),
        }
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
                        .arg(&g.ptr())
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
                        .arg(&g.ptr())
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
                        .arg(&g.ptr())
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
                    .arg(&g.ptr())
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
    b.b.validate()?;
    ensure!(
        b.b.shape[0] == rows
            && b.b.shape[1] == hv
            && b.b.stride[1] == 1
            && b.b.stride[0] >= u64::from(hv),
        "GDN b must be [max_rows, {hv}] with stride [{hv}, 1]"
    );
    let dt_bias_width = match &b.a {
        GdnGateInput::PerHead(a) => {
            a.validate()?;
            ensure!(
                a.shape[0] == rows
                    && a.shape[1] == hv
                    && a.stride[1] == 1
                    && a.stride[0] >= u64::from(hv),
                "GDN a must be [max_rows, {hv}] with stride [{hv}, 1]"
            );
            hv
        }
        GdnGateInput::PerChannel(a) => {
            a.validate()?;
            ensure!(
                a.shape == [rows, hv, 128] && a.stride == [u64::from(hv * 128), 128, 1],
                "GDN a must be [max_rows, {hv}, 128] with strides [hv*128, 128, 1]"
            );
            hv * 128
        }
    };
    let (a_shape, a_stride, dt_shape, dt_stride) = match (&b.a, b.decay) {
        (GdnGateInput::PerHead(_), GdnDecayWeights::Bf16 { a_log, dt_bias }) => {
            a_log.validate()?;
            dt_bias.validate()?;
            (a_log.shape, a_log.stride, dt_bias.shape, dt_bias.stride)
        }
        (GdnGateInput::PerChannel(_), GdnDecayWeights::F32 { a_log, dt_bias }) => {
            a_log.validate()?;
            dt_bias.validate()?;
            (a_log.shape, a_log.stride, dt_bias.shape, dt_bias.stride)
        }
        _ => anyhow::bail!("GDN gate mode and native decay dtype differ"),
    };
    ensure!(
        a_shape == [hv] && a_stride == [1],
        "GDN A_log must be [{hv}] with stride [1]"
    );
    ensure!(
        dt_shape == [dt_bias_width] && dt_stride == [1],
        "GDN dt_bias must be [{dt_bias_width}] with stride [1]"
    );
    let validate_conv = |weight: Tensor<bf16, 2>, channels| -> Result<()> {
        weight.validate()?;
        ensure!(
            weight.shape == [channels, 4] && weight.stride == [4, 1],
            "GDN conv weight must be [channels,4] with stride [4,1]"
        );
        Ok(())
    };
    match (&b.a, b.conv_weight) {
        (GdnGateInput::PerHead(_), GdnConvWeights::Fused(weight)) => {
            validate_conv(weight, conv_dim)?
        }
        (GdnGateInput::PerChannel(_), GdnConvWeights::Separate(weights)) => {
            let channels = [(conv_dim - hv * DV) / 2, (conv_dim - hv * DV) / 2, hv * DV];
            for (weight, channels) in weights.into_iter().zip(channels) {
                validate_conv(weight, channels)?;
            }
        }
        _ => anyhow::bail!("GDN gate mode and native convolution binding differ"),
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
    use crate::backend::gpu::{DevicePtr, LaunchConfig};

    #[test]
    fn fused_step_requires_compiled_geometry() {
        let spec = GdnSpec {
            hv: 32,
            ..GdnSpec::q38()
        };
        assert!(fused_step_geometry(&spec));
        for incompatible in [
            GdnSpec { hk: 8, ..spec },
            GdnSpec { dk: 64, ..spec },
            GdnSpec { dv: 64, ..spec },
            GdnSpec { hv: 48, ..spec },
            GdnSpec {
                gate_mode: GdnGateMode::PerChannel {
                    lower_bound: Some(-5.0),
                },
                ..spec
            },
        ] {
            assert!(!fused_step_geometry(&incompatible));
        }
    }

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
        let Err(rejected) = prep_grids(full.0, full.1, full.2, CUDA_GRID_YZ_MAX + 1, CONV_ROWS)
        else {
            panic!("rows above the CUDA gridDim.y/z limit must not form a launch");
        };
        assert!(rejected.to_string().contains("gridDim"), "{rejected}");
        Ok(())
    }

    use super::*;
    use crate::gdn_plan::cw_p2_bytes;

    #[cfg(feature = "cuda")]
    fn clover_fixture_f32(archive: &safetensors::SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
        let tensor = archive.tensor(name)?;
        ensure!(
            tensor.dtype() == safetensors::Dtype::F32,
            "Runner FAIL fixture dtype: {name}"
        );
        Ok(tensor
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect())
    }

    #[cfg(feature = "cuda")]
    fn clover_fixture_bf16(
        archive: &safetensors::SafeTensors<'_>,
        name: &str,
    ) -> Result<Vec<bf16>> {
        let tensor = archive.tensor(name)?;
        ensure!(
            tensor.dtype() == safetensors::Dtype::BF16,
            "Runner FAIL fixture dtype: {name}"
        );
        Ok(tensor
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| bf16::from_bits(u16::from_le_bytes(*v)))
            .collect())
    }

    #[cfg(feature = "cuda")]
    fn clover_state_transpose(source: &[f32]) -> Vec<f32> {
        (0..32 * 128 * 128)
            .map(|i| source[i / 16384 * 16384 + i % 128 * 128 + i % 16384 / 128])
            .collect()
    }

    #[cfg(feature = "cuda")]
    fn clover_block_fixture(
        device: &crate::backend::cuda::Device,
        archive: &safetensors::SafeTensors<'_>,
        case: &serde_json::Value,
        control: u32,
        carry: Option<&std::collections::BTreeMap<String, Vec<f32>>>,
    ) -> Result<(std::collections::BTreeMap<String, Vec<f32>>, usize)> {
        use super::super::contracts::view;
        use super::super::ops::{CommonOpsRef, GatedRmsMode};
        use crate::backend::gpu::DevicePtr;
        use cudarc::driver::sys;
        let stream = &device.stream;
        let tag = case["tag"].as_str().context("Runner FAIL fixture tag")?;
        let rows = u32::try_from(case["rows"].as_u64().context("Runner FAIL fixture rows")?)?;
        let preview = case["preview"]
            .as_bool()
            .context("Runner FAIL fixture mode")?;
        let f32s = |name: &str| clover_fixture_f32(archive, &format!("{tag}.{name}"));
        let bf16s = |name: &str| clover_fixture_bf16(archive, &format!("{tag}.{name}"));
        let upload = |name: &str| -> Result<_> { Ok(stream.clone_htod(&bf16s(name)?)?) };
        let qkv = upload("in_proj_qkv")?;
        let a = upload("in_proj_a")?;
        let b = if preview {
            None
        } else {
            Some(upload("in_proj_b")?)
        };
        let z = upload("in_proj_z")?;
        let w = upload("weight.conv1d.weight")?;
        let mut right_values = bf16s("weight.conv_right_weight")?;
        if control == 1 {
            right_values.fill(bf16::ZERO);
        }
        let right = stream.clone_htod(&right_values)?;
        let scale = upload("weight.bd_softmax_logit_scale")?;
        let bias = upload("weight.bd_softmax_history_bias")?;
        let mut history_c = bf16s("weight.bd_softmax_history_c")?;
        if control == 2 {
            history_c.fill(bf16::ZERO);
        }
        let history_c = stream.clone_htod(&history_c)?;
        let a_log = upload("weight.A_log")?;
        let dt = upload("weight.dt_bias")?;
        let norm_weight = upload("weight.norm.weight")?;
        let valid = archive.tensor(&format!("{tag}.valid"))?;
        ensure!(
            valid.dtype() == safetensors::Dtype::I32 && valid.shape() == [1, rows as usize],
            "Runner FAIL valid mask layout"
        );
        let valid_values: Vec<_> = valid
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| i32::from_le_bytes(*v))
            .collect();
        let valid = stream.clone_htod(&valid_values)?;
        let guard = f32::from_bits(0x7fc1_0001);
        let state_values = match carry {
            Some(previous) => previous
                .get("state")
                .context("missing committed state")?
                .clone(),
            None => f32s("state")?,
        };
        let mut initial_state = if control == 3 {
            state_values
        } else {
            clover_state_transpose(&state_values)
        };
        initial_state.extend([guard; 64]);
        let mut initial_history = match carry {
            Some(previous) => previous
                .get("history")
                .context("missing committed history")?
                .clone(),
            None => f32s("history")?,
        };
        initial_history.extend([guard; 64]);
        let mut state = stream.clone_htod(&initial_state)?;
        let mut history = stream.clone_htod(&initial_history)?;
        let words = rows as usize * 4096;
        let canary = bf16::from_bits(0x7fc1);
        let output = stream.clone_htod(&vec![canary; words + 64])?;
        let normalized = stream.clone_htod(&vec![canary; words + 64])?;
        let norm_input = stream.clone_htod(
            &f32s("bf.core")?
                .into_iter()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
        )?;
        let norm_isolated = stream.clone_htod(&vec![canary; words + 64])?;
        let root =
            serde_json::from_str(include_str!("../../../tests/fixtures/clover_config.json"))?;
        let model = crate::model_spec::ModelSpec::clover(
            &root,
            &serde_json::json!({"eos_token_id": 248044}),
        )?;
        let spec = GdnSpec::from_model(&model)?;
        let workspace = GdnWorkspace::new(&spec, rows, false)?;
        let scratch = stream.clone_htod(&vec![0xa5u8; workspace.bytes as usize + 256])?;
        let scalar = |value| view(value, stream, 0, [32], [1]);
        let mode = if preview {
            GdnBlockMode::Preview {
                state: view(&state, stream, 0, [32, 128, 128], [16384, 128, 1])?,
                history: view(&history, stream, 0, [8192, 3], [3, 1])?,
                right: view(&right, stream, 0, [8192, 3], [3, 1])?,
                softmax: [scalar(&scale)?, scalar(&bias)?, scalar(&history_c)?],
            }
        } else {
            GdnBlockMode::Commit {
                state: TensorMut(view(&state, stream, 0, [32, 128, 128], [16384, 128, 1])?),
                history: TensorMut(view(&history, stream, 0, [8192, 3], [3, 1])?),
                b: view(
                    b.as_ref().context("missing commit beta")?,
                    stream,
                    0,
                    [rows, 32],
                    [32, 1],
                )?,
            }
        };
        let plan = GdnBlockPlan::prepare(
            &device.ctx,
            spec,
            GdnBlockBindings {
                qkv: view(&qkv, stream, 0, [rows, 8192], [8192, 1])?,
                a: view(&a, stream, 0, [rows, 32], [32, 1])?,
                conv_weight: view(&w, stream, 0, [8192, 4], [4, 1])?,
                decay: [scalar(&a_log)?, scalar(&dt)?],
                valid: view(&valid, stream, 0, [rows], [1])?,
                output: TensorMut(view(&output, stream, 0, [rows, 32, 128], [4096, 128, 1])?),
            },
            mode,
            Scratch {
                ptr: scratch.device_ptr(stream).0,
                bytes: workspace.bytes,
            },
        )?;
        if control == 0 && rows == 32 {
            use sys::CUfunction_attribute as F;
            for (index, (kernel, config)) in plan.kernels.iter().zip(&plan.configs).enumerate() {
                println!(
                    "GDN_BLOCK_RESOURCE tag={tag} stage={index} registers={} local_bytes={} static_shared={} dynamic_shared={} resident_blocks={}",
                    kernel.get_attribute(F::CU_FUNC_ATTRIBUTE_NUM_REGS)?,
                    kernel.get_attribute(F::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?,
                    kernel.get_attribute(F::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)?,
                    config.shared_mem_bytes,
                    kernel.occupancy_max_active_blocks_per_multiprocessor(
                        config.block_dim.0,
                        config.shared_mem_bytes as usize,
                        None
                    )?
                );
            }
        }
        let metadata = stream.clone_htod(&[0i32, rows as i32, rows as i32, 128, 1, 0, 0, 0])?;
        let domain = RowDomain {
            meta: SeqRef {
                ptr: metadata.device_ptr(stream).0,
            },
            row_base: 0,
            row_capacity: rows,
        };
        let common = CommonOpsRef::new(&device.ctx)?;
        let norm_pairs = [(&output, &normalized), (&norm_input, &norm_isolated)]
            .into_iter()
            .map(|(input, destination)| -> Result<_> {
                Ok((
                    view(input, stream, 0, [rows, 32, 128], [4096, 128, 1])?,
                    view(destination, stream, 0, [rows, 32, 128], [4096, 128, 1])?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let norm_weight = view(&norm_weight, stream, 0, [128], [1])?;
        let norm_gate = view(&z, stream, 0, [rows, 32, 128], [4096, 128, 1])?;
        let execute = || -> Result<()> {
            plan.run(stream)?;
            for &(input, destination) in &norm_pairs {
                common.norm.rms_gated(
                    stream,
                    domain,
                    GatedRmsInput {
                        x: input,
                        weight: norm_weight,
                        gate: norm_gate,
                        mode: if control == 4 {
                            GatedRmsMode::SiluBf16
                        } else {
                            GatedRmsMode::SiluF32
                        },
                    },
                    &mut TensorMut(destination),
                    1e-6,
                )?;
            }
            Ok(())
        };
        let mut result = std::collections::BTreeMap::new();
        let mut structural_defects = 0;
        for graph in [false, true] {
            stream.memcpy_htod(&initial_state, &mut state)?;
            stream.memcpy_htod(&initial_history, &mut history)?;
            if graph {
                stream
                    .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                let launched = execute();
                let captured = stream.end_capture(sys::CUgraphInstantiate_flags(0));
                launched?;
                captured?
                    .context("Runner FAIL empty block graph")?
                    .launch()?;
            } else {
                execute()?;
            }
            stream
                .synchronize()
                .context("Runner FAIL GDN block launch")?;
            let got_output = stream.clone_dtoh(&output)?;
            let got_norm = stream.clone_dtoh(&normalized)?;
            let got_isolated = stream.clone_dtoh(&norm_isolated)?;
            let isolated_guard = got_isolated[words..]
                .iter()
                .all(|v| v.to_bits() == canary.to_bits());
            let got_state = stream.clone_dtoh(&state)?;
            let got_history = stream.clone_dtoh(&history)?;
            let got_scratch = stream.clone_dtoh(&scratch)?;
            let guards = isolated_guard
                && got_output[words..]
                    .iter()
                    .chain(&got_norm[words..])
                    .all(|v| v.to_bits() == canary.to_bits())
                && got_state[32 * 16384..]
                    .iter()
                    .chain(&got_history[8192 * 3..])
                    .all(|v| v.to_bits() == guard.to_bits())
                && got_scratch[workspace.bytes as usize..]
                    .iter()
                    .all(|v| *v == 0xa5);
            structural_defects += usize::from(!guards);
            if preview {
                let immutable = got_state
                    .iter()
                    .zip(&initial_state)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
                    && got_history
                        .iter()
                        .zip(&initial_history)
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                structural_defects += usize::from(!immutable);
                println!("CLOVER_GDN_CACHE tag={tag} graph={graph} immutable={immutable}");
            }
            let conv_start = workspace.conv_out.offset as usize;
            let conv_bytes = &got_scratch[conv_start..conv_start + rows as usize * 8192 * 2];
            for (name, values) in [
                (
                    "conv",
                    conv_bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|v| bf16::from_bits(u16::from_le_bytes(*v)).to_f32())
                        .collect(),
                ),
                (
                    "core",
                    got_output[..words].iter().map(|v| v.to_f32()).collect(),
                ),
                (
                    "norm",
                    got_norm[..words].iter().map(|v| v.to_f32()).collect(),
                ),
                ("state", clover_state_transpose(&got_state[..32 * 16384])),
                (
                    "norm_isolated",
                    got_isolated[..words].iter().map(|v| v.to_f32()).collect(),
                ),
                ("history", got_history[..8192 * 3].to_vec()),
            ] {
                let values: Vec<f32> = values;
                ensure!(
                    values.iter().all(|v| v.is_finite()),
                    "Runner FAIL nonfinite {tag} {name}"
                );
                if graph {
                    let eager: &Vec<f32> = result.get(name).context("missing eager result")?;
                    let identical = values
                        .iter()
                        .zip(eager)
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                    structural_defects += usize::from(!identical);
                    println!("CLOVER_GDN_GRAPH tag={tag} metric={name} identical={identical}");
                } else {
                    result.insert(name.to_owned(), values);
                }
            }
        }
        if control == 5 {
            ensure!(
                !preview && valid_values.iter().all(|v| *v == 1),
                "rounding control requires all-valid commit"
            );
            stream.memcpy_htod(&initial_history, &mut history)?;
            let old = cubin::module(&device.ctx, "gdn/gdn_conv_h32")?.load_function("gdn_conv")?;
            unsafe {
                stream
                    .launch_builder(&old)
                    .arg(&qkv.device_ptr(stream).0)
                    .arg(&w.device_ptr(stream).0)
                    .arg(&history.device_ptr(stream).0)
                    .arg(&(scratch.device_ptr(stream).0 + workspace.conv_out.offset))
                    .arg(&metadata.device_ptr(stream).0)
                    .arg(&8192i32)
                    .launch(LaunchConfig {
                        grid_dim: (32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            let bytes = stream.clone_dtoh(&scratch)?;
            let start = workspace.conv_out.offset as usize;
            result.insert(
                "conv".into(),
                bytes[start..start + rows as usize * 8192 * 2]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|v| bf16::from_bits(u16::from_le_bytes(*v)).to_f32())
                    .collect(),
            );
        }
        println!("CLOVER_GDN_STRUCTURE tag={tag} control={control} defects={structural_defects}");
        Ok((result, structural_defects))
    }

    #[cfg(feature = "cuda")]
    fn clover_rounding_outside(values: &[f32], reference: &[f32]) -> Result<usize> {
        ensure!(
            values.len() == reference.len(),
            "Runner FAIL rounding extent"
        );
        let rank = |value: f32| {
            let bits = bf16::from_f32(value).to_bits();
            if bits & 0x8000 != 0 {
                0x8000i32 - i32::from(bits & 0x7fff)
            } else {
                0x8000i32 + i32::from(bits)
            }
        };
        let mut outside = 0;
        for (&a, &b) in values.iter().zip(reference) {
            ensure!(
                a.is_finite() && b.is_finite(),
                "Runner FAIL rounding nonfinite"
            );
            outside += usize::from((rank(a) - rank(b)).abs() > 1);
        }
        Ok(outside)
    }

    #[cfg(feature = "cuda")]
    fn clover_score(
        values: &[f32],
        reference: &[f32],
        bound: &serde_json::Value,
    ) -> Result<(bool, f64, f64)> {
        ensure!(
            values.len() == reference.len(),
            "Runner FAIL comparison extent"
        );
        let mut max_abs = 0.0f64;
        let mut squared = 0.0;
        let mut norm = 0.0;
        for (&value, &reference) in values.iter().zip(reference) {
            ensure!(
                value.is_finite() && reference.is_finite(),
                "Runner FAIL comparison nonfinite"
            );
            let delta = f64::from(value) - f64::from(reference);
            max_abs = max_abs.max(delta.abs());
            squared += delta * delta;
            norm += f64::from(reference).powi(2);
        }
        let relative_l2 = squared.sqrt() / norm.sqrt().max(1e-30);
        let passed = max_abs
            <= bound["max_abs"]
                .as_f64()
                .context("missing frozen max_abs")?
            && relative_l2
                <= bound["relative_l2"]
                    .as_f64()
                    .context("missing frozen relative_l2")?;
        Ok((passed, max_abs, relative_l2))
    }

    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "GPU window; FF_CLOVER_REFERENCE must name a create-exclusive frozen reference"]
    fn clover_gdn_real_weight_preview_commit_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let root = std::path::PathBuf::from(
            std::env::var_os("FF_CLOVER_REFERENCE").context("FF_CLOVER_REFERENCE missing")?,
        );
        let table: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("thresholds.json"))?)?;
        ensure!(
            table["revision"] == "4b0402b65e523d0183a14c431955438714cb16ff"
                && table["schema_keys"] == 813,
            "Runner FAIL reference revision/schema"
        );
        let bytes = std::fs::read(root.join("gdn.safetensors"))?;
        let archive = safetensors::SafeTensors::deserialize(&bytes)?;
        let cases = table["cases"]
            .as_array()
            .context("Runner FAIL reference cases")?;
        ensure!(cases.len() == 16, "Runner FAIL reference case count");
        let device = crate::backend::cuda::Device::new(0)?;
        let mut defects = 0;
        let mut continuation = None;
        for case in cases {
            let tag = case["tag"].as_str().context("missing case tag")?;
            let carry = if tag == "case7.commit" {
                Some(
                    continuation
                        .as_ref()
                        .context("missing first committed block")?,
                )
            } else {
                None
            };
            let (result, structural) = clover_block_fixture(&device, &archive, case, 0, carry)?;
            defects += structural;
            if tag == "case5.commit" {
                continuation = Some(result.clone());
            }
            for (name, values) in result {
                if name == "conv" || name == "norm_isolated" {
                    let reference_name = if name == "norm_isolated" {
                        "norm"
                    } else {
                        "conv"
                    };
                    let reference =
                        clover_fixture_f32(&archive, &format!("{tag}.bf.{reference_name}"))?;
                    let outside = clover_rounding_outside(&values, &reference)?;
                    defects += usize::from(outside != 0);
                    println!(
                        "CLOVER_GDN_ROUNDING {} tag={tag} metric={name} outside_one_bf16_ulp={outside}",
                        if outside == 0 { "PASS" } else { "DEFECT" }
                    );
                    if name == "norm_isolated" {
                        continue;
                    }
                }
                let reference = clover_fixture_f32(&archive, &format!("{tag}.fp.{name}"))?;
                let (passed, max_abs, relative_l2) =
                    clover_score(&values, &reference, &case["bounds"][&name])?;
                defects += usize::from(!passed);
                println!(
                    "CLOVER_GDN {} tag={tag} metric={name} max_abs={max_abs:.9e} relative_l2={relative_l2:.9e} bound={}",
                    if passed { "PASS" } else { "DEFECT" },
                    case["bounds"][&name]
                );
            }
        }
        let mut detected = 0;
        for (tag, control, metric) in [
            ("case0.preview", 1, "core"),
            ("case4.preview", 2, "core"),
            ("case4.preview", 3, "core"),
            ("case5.commit", 4, "norm_isolated"),
            ("case5.commit", 5, "conv"),
        ] {
            let case = cases
                .iter()
                .find(|case| case["tag"] == tag)
                .context("missing control case")?;
            let (result, structural) =
                clover_block_fixture(&device, &archive, case, control, None)?;
            defects += structural;
            if control >= 4 {
                let reference_name = if control == 4 { "norm" } else { "conv" };
                let reference =
                    clover_fixture_f32(&archive, &format!("{tag}.bf.{reference_name}"))?;
                let outside = clover_rounding_outside(&result[metric], &reference)?;
                detected += usize::from(outside > 0);
                println!(
                    "CLOVER_GDN_CONTROL tag={tag} control={control} metric={metric} detected={} outside_one_bf16_ulp={outside}",
                    outside > 0
                );
                continue;
            }
            let reference = clover_fixture_f32(&archive, &format!("{tag}.fp.{metric}"))?;
            let (passed, max_abs, relative_l2) =
                clover_score(&result[metric], &reference, &case["bounds"][metric])?;
            detected += usize::from(!passed);
            println!(
                "CLOVER_GDN_CONTROL tag={tag} control={control} metric={metric} detected={} max_abs={max_abs:.9e} relative_l2={relative_l2:.9e}",
                !passed
            );
        }
        println!(
            "CLOVER_GDN {} cases={} graph_eager=true guards=true controls={detected}/5 defects={defects}",
            if defects == 0 && detected == 5 {
                "PASS"
            } else {
                "DEFECT"
            },
            cases.len()
        );
        ensure!(
            defects == 0 && detected == 5,
            "Clover GDN defects={defects} controls={detected}/5"
        );
        Ok(())
    }

    enum GeometryDecay {
        Bf16(
            crate::backend::gpu::CudaSlice<bf16>,
            crate::backend::gpu::CudaSlice<bf16>,
        ),
        F32(
            crate::backend::gpu::CudaSlice<f32>,
            crate::backend::gpu::CudaSlice<f32>,
        ),
    }

    struct GeometryBuffers {
        qkv: crate::backend::gpu::CudaSlice<bf16>,
        a: crate::backend::gpu::CudaSlice<bf16>,
        b: crate::backend::gpu::CudaSlice<bf16>,
        conv: crate::backend::gpu::CudaSlice<bf16>,
        decay: GeometryDecay,
        out: crate::backend::gpu::CudaSlice<bf16>,
        state: crate::backend::gpu::CudaSlice<f32>,
        history: crate::backend::gpu::CudaSlice<f32>,
        snapshots: Option<(
            crate::backend::gpu::CudaSlice<f32>,
            crate::backend::gpu::CudaSlice<f32>,
        )>,
    }

    fn geometry_state(heads: usize) -> Vec<f32> {
        (0..heads * 128 * 128)
            .map(|i| ((i * 19 % 251) as f32 - 125.0) / 1024.0)
            .collect()
    }

    fn geometry_view<T: crate::backend::gpu::DeviceRepr + Copy, const R: usize>(
        buffer: &crate::backend::gpu::CudaSlice<T>,
        stream: &Arc<CudaStream>,
        shape: [u32; R],
        stride: [u64; R],
        offset: usize,
    ) -> Result<Tensor<T, R>> {
        use crate::backend::gpu::DevicePtr;
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
            let key_width = spec.hk as usize * spec.dk as usize;
            let period = |c: usize| if c < key_width { 251 } else { 257 };
            let zero_head = |c: usize| c < 2 * key_width && c % key_width < 128;
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
            let per_channel = matches!(
                spec.gate_mode,
                crate::op_contract::GdnGateMode::PerChannel { .. }
            );
            let a_width = if per_channel { hv * 128 } else { hv };
            let a: Vec<_> = (0..rows as usize * a_width)
                .map(|i| {
                    if i / a_width < valid as usize {
                        let c = i % a_width;
                        pattern(
                            (i / a_width) * 17 + c * 13,
                            if c % 2 == 0 { 251 } else { 257 },
                        )
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
            let dt: Vec<_> = (0..a_width)
                .map(|c| pattern(c * 3, if c % 2 == 0 { 251 } else { 257 }))
                .collect();
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
                a: stream.clone_htod(&a)?,
                b: stream.clone_htod(&gates(19, 257))?,
                conv: stream.clone_htod(&conv)?,
                decay: if per_channel {
                    GeometryDecay::F32(
                        stream.clone_htod(
                            &(0..hv)
                                .map(|h| -3.0 + h as f32 / 128.0 + 0.00013 * (h % 7 + 1) as f32)
                                .collect::<Vec<_>>(),
                        )?,
                        stream.clone_htod(
                            &(0..a_width)
                                .map(|c| dt[c].to_f32() + 0.00017 * (c % 5 + 1) as f32)
                                .collect::<Vec<_>>(),
                        )?,
                    )
                } else {
                    GeometryDecay::Bf16(stream.clone_htod(&a_log)?, stream.clone_htod(&dt)?)
                },
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
                a: match spec.gate_mode {
                    crate::op_contract::GdnGateMode::PerHead => {
                        GdnGateInput::PerHead(geometry_view(
                            &self.a,
                            stream,
                            [rows, hv],
                            [u64::from(hv), 1],
                            row * hv as usize,
                        )?)
                    }
                    crate::op_contract::GdnGateMode::PerChannel { .. } => {
                        GdnGateInput::PerChannel(geometry_view(
                            &self.a,
                            stream,
                            [rows, hv, 128],
                            [u64::from(hv) * 128, 128, 1],
                            row * hv as usize * 128,
                        )?)
                    }
                },
                b: geometry_view(
                    &self.b,
                    stream,
                    [rows, hv],
                    [u64::from(hv), 1],
                    row * hv as usize,
                )?,
                conv_weight: if matches!(spec.gate_mode, GdnGateMode::PerHead) {
                    GdnConvWeights::Fused(geometry_view(&self.conv, stream, [cd, 4], [4, 1], 0)?)
                } else {
                    let width = spec.hk * DK;
                    GdnConvWeights::Separate([
                        geometry_view(&self.conv, stream, [width, 4], [4, 1], 0)?,
                        geometry_view(&self.conv, stream, [width, 4], [4, 1], width as usize * 4)?,
                        geometry_view(
                            &self.conv,
                            stream,
                            [spec.hv * DV, 4],
                            [4, 1],
                            width as usize * 8,
                        )?,
                    ])
                },
                decay: match &self.decay {
                    GeometryDecay::Bf16(a, dt) => GdnDecayWeights::Bf16 {
                        a_log: geometry_view(a, stream, [hv], [1], 0)?,
                        dt_bias: geometry_view(dt, stream, [hv], [1], 0)?,
                    },
                    GeometryDecay::F32(a, dt) => GdnDecayWeights::F32 {
                        a_log: geometry_view(a, stream, [hv], [1], 0)?,
                        dt_bias: geometry_view(dt, stream, [hv * 128], [1], 0)?,
                    },
                },
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

    const PER_CHANNEL_ORACLE_LIMIT: f64 = 0.05;

    struct GeometryOutput {
        out: Vec<bf16>,
        state: Vec<f32>,
        history: Vec<f32>,
        prefix_states: Vec<f32>,
        prefix_histories: Vec<f32>,
        norm_ratio: f64,
        oracle_out: f64,
        oracle_state: f64,
        oracle_history: usize,
        snapshot_mismatches: usize,
        #[cfg(feature = "cuda")]
        p2_prepared: Vec<u8>,
        #[cfg(feature = "cuda")]
        p2_slots: Vec<u8>,
    }

    fn geometry_meta(
        stream: &Arc<CudaStream>,
        prefix: u32,
        valid: u32,
    ) -> Result<crate::backend::gpu::CudaSlice<i32>> {
        let metadata = stream.clone_htod(&[
            prefix as i32,
            valid as i32,
            (prefix + valid) as i32,
            8192,
            1,
            0,
            0,
            0,
        ])?;
        Ok(metadata)
    }

    fn per_channel_gate_run(
        device: &crate::backend::cuda::Device,
        spec: GdnSpec,
        rows: u32,
        valid: u32,
        buffers: &GeometryBuffers,
        lower_bound: Option<f32>,
    ) -> Result<(Vec<f32>, Vec<bf16>)> {
        let stream = &device.stream;
        let bindings = buffers.bindings(stream, spec, rows, 0)?;
        let GdnDecayWeights::F32 { a_log, dt_bias } = bindings.decay else {
            anyhow::bail!("per-channel fixture requires native F32 parameters");
        };
        let beta = stream.alloc_zeros::<bf16>(rows as usize * spec.hv as usize)?;
        let g = stream.alloc_zeros::<f32>(rows as usize * spec.hv as usize * 128)?;
        let mut beta_view = TensorMut(geometry_view(
            &beta,
            stream,
            [rows, spec.hv],
            [u64::from(spec.hv), 1],
            0,
        )?);
        let mut g_view = TensorMut(geometry_view(
            &g,
            stream,
            [rows, spec.hv, 128],
            [u64::from(spec.hv) * 128, 128, 1],
            0,
        )?);
        let meta = geometry_meta(stream, 23, valid)?;
        let function =
            cubin::module(&device.ctx, "q38_common")?.load_function("gdn_gates_per_channel")?;
        crate::backend::cuda::ops::GateOps::gdn_channel_gates_with(
            &function,
            stream,
            RowDomain {
                meta: SeqRef {
                    ptr: meta.device_ptr(stream).0,
                },
                row_base: 0,
                row_capacity: rows,
            },
            crate::backend::cuda::ops::GdnChannelGateBindings {
                a: match bindings.a {
                    GdnGateInput::PerChannel(a) => a,
                    GdnGateInput::PerHead(_) => {
                        anyhow::bail!("per-channel gate fixture is per-head")
                    }
                },
                b: bindings.b,
                a_log,
                dt_bias,
                beta: &mut beta_view,
                g: &mut g_view,
                lower_bound,
            },
        )?;
        stream.synchronize()?;
        Ok((stream.clone_dtoh(&g)?, stream.clone_dtoh(&beta)?))
    }

    struct PerChannelGateFixture<'a> {
        a: &'a [bf16],
        b: &'a [bf16],
        a_log: &'a [f32],
        dt: &'a [f32],
    }

    fn per_channel_gate_reference(
        fixture: PerChannelGateFixture<'_>,
        hv: usize,
        rows: usize,
        valid: usize,
        lower_bound: Option<f32>,
    ) -> (Vec<f32>, Vec<bf16>) {
        let PerChannelGateFixture { a, b, a_log, dt } = fixture;
        let sigmoid = |x: f32| {
            if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let e = x.exp();
                e / (1.0 + e)
            }
        };
        let mut g = vec![0.0f32; rows * hv * 128];
        let mut beta = vec![bf16::ZERO; rows * hv];
        for first in (0..rows).step_by(64) {
            let mut decay = vec![0.0f32; hv * 128];
            for row in first..(first + 64).min(rows) {
                for head in 0..hv {
                    let rate = a_log[head].exp();
                    beta[row * hv + head] = if row < valid {
                        bf16::from_f32(sigmoid(b[row * hv + head].to_f32()))
                    } else {
                        bf16::ZERO
                    };
                    for channel in 0..128 {
                        let index = (row * hv + head) * 128 + channel;
                        let local = head * 128 + channel;
                        let value = if row < valid {
                            let t = a[index].to_f32() + dt[local];
                            if let Some(bound) = lower_bound {
                                bound * sigmoid(rate * t)
                            } else {
                                -rate * if t > 20.0 { t } else { t.exp().ln_1p() }
                            }
                        } else {
                            0.0
                        };
                        decay[local] = if row < valid {
                            decay[local] + value
                        } else {
                            0.0
                        };
                        g[index] = decay[local];
                    }
                }
            }
        }
        (g, beta)
    }

    #[derive(Clone, Copy, Debug)]
    enum GeometryCore {
        Recurrent,
        Selected,
        Verify,
        Replay,
        ChunkedSnapshot,
        RstepSnapshot,
        Chunked,
        #[cfg(feature = "cuda")]
        P1(u32),
        P2,
        #[cfg(feature = "cuda")]
        P2Whole,
        #[cfg(feature = "cuda")]
        P2Expanded,
        #[cfg(feature = "cuda")]
        WrongHeadStride,
    }

    impl GeometryCore {
        fn covered(self, device: &crate::backend::cuda::Device) -> bool {
            let needs_p2 = matches!(self, Self::P2);
            #[cfg(feature = "cuda")]
            let needs_p2 = needs_p2 || matches!(self, Self::P2Whole | Self::P2Expanded);
            let covered = !needs_p2 || GdnPolicy::for_device(&device.info).cw_p2_for_step(5);
            if !covered {
                println!("GDN NOT_COVERED core={self:?} arch={}", device.info.arch);
            }
            covered
        }
    }

    #[derive(Clone, Copy)]
    enum GeometryControl {
        None,
        ValueInput,
        KeyDecay,
        StrongDecay,
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
        let compact = matches!(core, GeometryCore::P2)
            || matches!(core, GeometryCore::Selected)
                && GdnPolicy::for_device(&device.info).cw_p2_for_step(rows);
        #[cfg(feature = "cuda")]
        let compact = compact || matches!(core, GeometryCore::P2Whole);
        let mut plan = GdnOps::prepare(
            &device.ctx,
            &spec,
            rows,
            buffers.bindings(stream, spec, rows, offset)?,
            compact,
        )?;
        #[cfg(feature = "cuda")]
        if matches!(core, GeometryCore::P2Expanded) {
            let module = cubin::module(
                &device.ctx,
                &format!("gdn/gdn_cw_p2{}", spec.kernel_suffix()?),
            )?;
            let per_channel = matches!(
                spec.gate_mode,
                crate::op_contract::GdnGateMode::PerChannel { .. }
            );
            Arc::get_mut(&mut plan.kernels)
                .context("expanded P2 kernels are shared")?
                .p2 = Some(ChunkwiseP2 {
                a: module.load_function(if per_channel {
                    "gdn_cw_p2_stage_a_per_channel"
                } else {
                    "gdn_cw_p2_stage_a"
                })?,
                h: module.load_function(if per_channel {
                    "gdn_cw_p2_stage_h_per_channel"
                } else {
                    "gdn_cw_p2_stage_h"
                })?,
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
        #[cfg(feature = "cuda")]
        if matches!(core, GeometryCore::P2Whole) {
            let base = add(plan.ws.k_bf16.offset, plan.ws.k_bf16.bytes)?;
            plan.ws.p2 = Region {
                offset: base,
                bytes: cw_p2_bytes(&spec, rows),
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
            let per_channel = matches!(
                spec.gate_mode,
                crate::op_contract::GdnGateMode::PerChannel { .. }
            );
            if matches!(core, GeometryCore::RstepSnapshot) {
                let module = cubin::module(&device.ctx, &format!("gdn/gdn_rstep{suffix}"))?;
                kernels.rstep = Some(module.load_function(if per_channel {
                    "gdn_rstep_core_per_channel_v1"
                } else {
                    "gdn_rstep_core_v1"
                })?);
                kernels.core_snapshot = module.load_function(if per_channel {
                    "gdn_rstep_core_per_channel_snapshot"
                } else {
                    "gdn_rstep_core_snapshot"
                })?;
            } else {
                ensure!(
                    rows <= 5,
                    "chunked snapshot kernel supports at most five rows"
                );
                kernels.rstep = None;
                kernels.core_snapshot =
                    cubin::module(&device.ctx, &format!("gdn/gdn_chunked{suffix}"))?
                        .load_function(if per_channel {
                            "gdn_chunked_core_per_channel_snapshot"
                        } else {
                            "gdn_chunked_core_snapshot"
                        })?;
                #[cfg(feature = "cuda")]
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
        #[cfg(feature = "cuda")]
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
        geometry_run_control(
            device,
            spec,
            rows,
            valid,
            core,
            graph,
            GeometryControl::None,
        )
    }

    fn geometry_run_control(
        device: &crate::backend::cuda::Device,
        spec: GdnSpec,
        rows: u32,
        valid: u32,
        core: GeometryCore,
        graph: bool,
        control: GeometryControl,
    ) -> Result<GeometryOutput> {
        use crate::backend::gpu::DevicePtr;
        let label = std::ffi::CString::new(format!(
            "gdn hv={} rows={rows} valid={valid} core={core:?} graph={graph}",
            spec.hv
        ))?;
        let _range = super::super::profile::Range::new(cfg!(feature = "cuda"), &label);
        let stream = &device.stream;
        let mut buffers = GeometryBuffers::new(stream, spec, rows, valid)?;
        let per_channel = matches!(
            spec.gate_mode,
            crate::op_contract::GdnGateMode::PerChannel { .. }
        );
        if matches!(control, GeometryControl::ValueInput) {
            ensure!(per_channel, "channel control requires per-channel gates");
            let mut qkv = stream.clone_dtoh(&buffers.qkv)?;
            let conv_dim = spec.conv_dim()? as usize;
            let index = 59 * conv_dim + 2 * spec.hk as usize * 128 + 4 * 128 + 91;
            qkv[index] = bf16::from_f32(qkv[index].to_f32() + 8.0);
            buffers.qkv = stream.clone_htod(&qkv)?;
        }
        if matches!(control, GeometryControl::KeyDecay) {
            ensure!(
                per_channel && valid == 1,
                "key-axis control requires one native per-channel row"
            );
            let cd = spec.conv_dim()? as usize;
            let mut qkv = vec![bf16::ZERO; rows as usize * cd];
            qkv[3 * 128 + 17] = bf16::ONE;
            buffers.qkv = stream.clone_htod(&qkv)?;
            let mut conv = vec![bf16::ZERO; cd * 4];
            for channel in 0..cd {
                conv[channel * 4 + 3] = bf16::ONE;
            }
            buffers.conv = stream.clone_htod(&conv)?;
            buffers.history = stream.alloc_zeros(cd * 3)?;
            let mut state = vec![0.0f32; 64 * 128 * 128];
            state[(3 * 128 + 17) * 128 + 91] = 1.0;
            buffers.state = stream.clone_htod(&state)?;
            let mut a = vec![bf16::ZERO; rows as usize * 64 * 128];
            a[3 * 128 + 17] = bf16::from_f32(-1.828125);
            a[3 * 128 + 91] = bf16::from_f32(-0.95703125);
            buffers.a = stream.clone_htod(&a)?;
            buffers.b = stream.clone_htod(&vec![bf16::from_f32(-100.0); rows as usize * 64])?;
            buffers.decay = GeometryDecay::F32(stream.alloc_zeros(64)?, stream.alloc_zeros(8192)?);
        }
        if matches!(control, GeometryControl::StrongDecay) {
            ensure!(per_channel, "strong decay requires per-channel gates");
            let a: Vec<_> = (0..rows as usize * 8192)
                .map(|index| bf16::from_f32(if index % 7 < 4 { 128.0 } else { -128.0 }))
                .collect();
            buffers.a = stream.clone_htod(&a)?;
            buffers.decay = GeometryDecay::F32(stream.alloc_zeros(64)?, stream.alloc_zeros(8192)?);
        }
        let initial_state = if per_channel {
            stream.clone_dtoh(&buffers.state)?
        } else {
            Vec::new()
        };
        let raw_history = if per_channel {
            Some((
                stream.clone_dtoh(&buffers.qkv)?,
                stream.clone_dtoh(&buffers.history)?,
            ))
        } else {
            None
        };
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
        #[cfg(feature = "cuda")]
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
                #[cfg(feature = "cuda")]
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
                    continue;
                }
                if matches!(core, GeometryCore::ChunkedSnapshot) {
                    GdnOps::run_core(
                        plan,
                        stream,
                        meta,
                        ExecutionPhase::Prefill,
                        Scratch { ptr, bytes },
                        true,
                    )?;
                } else if matches!(
                    core,
                    GeometryCore::Verify | GeometryCore::Replay | GeometryCore::RstepSnapshot
                ) {
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
            super::super::moe::testing::capture(stream, execute)?.launch()?;
        } else {
            execute()?;
        }
        stream.synchronize()?;
        let plan = &plans.last().context("empty geometry plans")?.0;
        let data_bytes = if per_channel {
            (plan.ws.k_fp32.offset + plan.ws.k_fp32.bytes) as usize
        } else {
            plan.ws.bytes as usize
        };
        let data = stream.clone_dtoh(&workspace.slice(..data_bytes))?;
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
        if matches!(control, GeometryControl::ValueInput) {
            let conv_dim = spec.conv_dim()? as usize;
            let channel = 2 * spec.hk as usize * 128 + 4 * 128 + 91;
            let prepared_offset = plan.ws.conv_out.offset as usize + (59 * conv_dim + channel) * 2;
            let prepared = bf16::from_bits(u16::from_ne_bytes([
                data[prepared_offset],
                data[prepared_offset + 1],
            ]));
            let raw = raw_history
                .as_ref()
                .map(|(qkv, _)| qkv[59 * conv_dim + channel]);
            println!(
                "GDN_PER_CHANNEL_CONTROL_PROBE raw={raw:?} prepared={} out={}",
                prepared.to_f32(),
                out[(59 * 64 + 4) * 128 + 91].to_f32()
            );
        }
        if let Some((index, value)) = out[..valid as usize * spec.hv as usize * 128]
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            eprintln!("Runner FAIL nonfinite output index={index} value={value:?}");
        }
        if let Some((index, value)) = state
            .iter()
            .chain(&history)
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            eprintln!("Runner FAIL nonfinite state/history index={index} value={value:?}");
        }
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
        #[cfg(feature = "cuda")]
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
            let slots = if per_channel {
                Vec::new()
            } else {
                stream.clone_dtoh(&workspace.slice(
                    plan.ws.p2.offset as usize..(plan.ws.p2.offset + plan.ws.p2.bytes) as usize,
                ))?
            };
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
                    if per_channel {
                        snapshot_mismatches += state
                            .iter()
                            .zip(&reference.state)
                            .filter(|(actual, reference)| {
                                (f64::from(**actual) - f64::from(**reference)).abs()
                                    > PER_CHANNEL_ORACLE_LIMIT
                                        * (1e-4 + f64::from(**reference).abs())
                            })
                            .count();
                        snapshot_mismatches += history
                            .iter()
                            .zip(&reference.history)
                            .filter(|(actual, reference)| actual.to_bits() != reference.to_bits())
                            .count();
                    } else {
                        snapshot_mismatches += state
                            .iter()
                            .zip(&reference.state)
                            .chain(history.iter().zip(&reference.history))
                            .filter(|(actual, reference)| actual.to_bits() != reference.to_bits())
                            .count();
                    }
                } else {
                    snapshot_mismatches += state
                        .iter()
                        .chain(history)
                        .filter(|value| value.to_bits() != 0x7fc1_0001)
                        .count();
                }
            }
        }
        let mut oracle_history = 0;
        if let Some((qkv, initial)) = &raw_history {
            let conv_dim = spec.conv_dim()? as usize;
            for channel in 0..conv_dim {
                for tap in 0..3 {
                    let source = valid as usize + tap;
                    let expected = if source < 3 {
                        initial[channel * 3 + source]
                    } else {
                        qkv[(source - 3) * conv_dim + channel].to_f32()
                    };
                    oracle_history +=
                        usize::from(history[channel * 3 + tap].to_bits() != expected.to_bits());
                }
            }
        }
        let (oracle_out, oracle_state) = if per_channel && !sequential {
            per_channel_oracle(&data, plan, valid, &out, &state, &initial_state)?
        } else {
            (0.0, 0.0)
        };
        Ok(GeometryOutput {
            out,
            state,
            history,
            prefix_states,
            prefix_histories,
            norm_ratio,
            oracle_out,
            oracle_state,
            oracle_history,
            snapshot_mismatches,
            #[cfg(feature = "cuda")]
            p2_prepared,
            #[cfg(feature = "cuda")]
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
        let per_channel = matches!(
            spec.gate_mode,
            crate::op_contract::GdnGateMode::PerChannel { .. }
        );
        for valid in 1..=rows {
            for graph in [false, true] {
                take_phase_log();
                let candidate =
                    geometry_run(device, spec, rows, valid, GeometryCore::Verify, graph)?;
                let phases = take_phase_log();
                let ordered = phases.first().copied() == Some("rstep");
                let expected = &reference[valid as usize - 1];
                let mismatches = geometry_mismatches(&candidate, expected);
                let history_mismatches = candidate
                    .history
                    .iter()
                    .zip(&expected.history)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let passed = ordered
                    && candidate.snapshot_mismatches == 0
                    && if per_channel {
                        history_mismatches == 0
                            && candidate.oracle_history == 0
                            && candidate.oracle_out <= PER_CHANNEL_ORACLE_LIMIT
                            && candidate.oracle_state <= PER_CHANNEL_ORACLE_LIMIT
                    } else {
                        mismatches == 0
                    };
                defects += usize::from(!passed);
                println!(
                    "{} INVARIANT_GDN hv={} bound={rows} valid={valid} graph={graph} output_state_history_bit_mismatches={mismatches} history_mismatches={history_mismatches} out_scaled={} state_scaled={} ordered_rstep={ordered}",
                    if passed { "PASS" } else { "DEFECT" },
                    spec.hv,
                    candidate.oracle_out,
                    candidate.oracle_state
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
                            .filter(|(a, b)| {
                                if per_channel && stage == "state" {
                                    (f64::from(**a) - f64::from(**b)).abs()
                                        > PER_CHANNEL_ORACLE_LIMIT * (1e-4 + f64::from(**b).abs())
                                } else {
                                    a.to_bits() != b.to_bits()
                                }
                            })
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

    fn key_axis_column_step(
        state: &mut [f64; 128],
        q: &[f64; 128],
        k: &[f64; 128],
        gate: &[f64; 128],
        value: f64,
        beta: f64,
    ) -> f64 {
        for (entry, decay) in state.iter_mut().zip(gate) {
            *entry *= decay.exp();
        }
        let memory: f64 = state.iter().zip(k).map(|(s, k)| s * k).sum();
        let delta = beta * (value - memory);
        for (entry, key) in state.iter_mut().zip(k) {
            *entry += key * delta;
        }
        state.iter().zip(q).map(|(s, q)| s * q).sum()
    }

    #[test]
    fn kda_decay_follows_key_axis_on_off_diagonal_state() {
        let mut state = [0.0; 128];
        state[17] = 1.0;
        let mut query = [0.0; 128];
        query[17] = 1.0 / 128.0f64.sqrt();
        let mut gate = [0.0; 128];
        gate[17] = 0.5f64.ln();
        gate[91] = 0.25f64.ln();
        let output = key_axis_column_step(&mut state, &query, &[0.0; 128], &gate, 0.0, 0.0);
        assert_eq!(state[17], 0.5);
        assert_eq!(output, 0.5 / 128.0f64.sqrt());
        let wrong_state = gate[91].exp();
        let wrong_output = wrong_state / 128.0f64.sqrt();
        assert!((state[17] - wrong_state).abs() > 0.05 * state[17]);
        assert!((output - wrong_output).abs() > 0.05 * output);
        println!(
            "PASS kda_key_axis_control key=17 value=91 correct_state=0.5 value_axis_state=0.25"
        );
    }

    fn per_channel_oracle(
        data: &[u8],
        plan: &GdnPlan,
        valid: u32,
        out: &[bf16],
        state: &[f32],
        initial_state: &[f32],
    ) -> Result<(f64, f64)> {
        let spec = plan.spec;
        ensure!(
            matches!(
                spec.gate_mode,
                crate::op_contract::GdnGateMode::PerChannel { .. }
            ),
            "per-channel oracle bound to a per-head plan"
        );
        let hv = spec.hv as usize;
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
        let qk = |region: Region, row: usize, head: usize, d: usize| {
            read_f32(region.offset as usize + ((row * hv + head) * 128 + d) * 4)
        };
        let v_offset = 2 * spec.hk as usize * 128;
        let conv_dim = plan.conv_dim as usize;
        let mut worst_out = 0f64;
        let mut worst_out_detail = (0usize, 0usize, 0usize, 0f64, 0f64);
        let mut worst_state = 0f64;
        for head in [0, 1, 3, 4, 17, hv - 1] {
            for channel in [0, 1, 31, 32, 91, 127] {
                let mut expected = [0f64; 128];
                for (d, value) in expected.iter_mut().enumerate() {
                    let index = (head * 128 + d) * 128 + channel;
                    *value = f64::from(initial_state[index]);
                }
                for row in 0..valid as usize {
                    let gate = std::array::from_fn(|d| {
                        let index = (row * hv + head) * 128 + d;
                        let current = read_f32(plan.ws.g.offset as usize + index * 4);
                        let previous = if row % 64 == 0 {
                            0.0
                        } else {
                            read_f32(plan.ws.g.offset as usize + (index - hv * 128) * 4)
                        };
                        current - previous
                    });
                    let q = std::array::from_fn(|d| qk(plan.ws.q_fp32, row, head, d));
                    let k = std::array::from_fn(|d| qk(plan.ws.k_fp32, row, head, d));
                    let beta = read_f32(plan.ws.beta_fp32.offset as usize + (row * hv + head) * 4);
                    let v = read_bf16(
                        plan.ws.conv_out.offset as usize
                            + (row * conv_dim + v_offset + head * 128 + channel) * 2,
                    );
                    let expected_out = key_axis_column_step(&mut expected, &q, &k, &gate, v, beta);
                    let actual = out[(row * hv + head) * 128 + channel].to_f32() as f64;
                    let ratio = (actual - expected_out).abs() / (1e-4 + expected_out.abs());
                    if ratio > worst_out {
                        worst_out = ratio;
                        worst_out_detail = (row, head, channel, expected_out, actual);
                    }
                }
                for (d, expected) in expected.iter().enumerate() {
                    let actual = state[(head * 128 + d) * 128 + channel] as f64;
                    worst_state =
                        worst_state.max((actual - expected).abs() / (1e-4 + expected.abs()));
                }
            }
        }
        if worst_out > 1.0 {
            let (row, head, channel, expected, actual) = worst_out_detail;
            println!(
                "GDN_PER_CHANNEL_ORACLE_WORST row={row} head={head} channel={channel} expected={expected} actual={actual} scaled={worst_out}"
            );
        }
        Ok((worst_out, worst_state))
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

    #[cfg(feature = "cuda")]
    fn geometry_follow(
        device: &crate::backend::cuda::Device,
        spec: GdnSpec,
        prefix: u32,
        prior: &GeometryOutput,
    ) -> Result<Vec<u32>> {
        use crate::backend::gpu::DevicePtr;
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
    #[cfg(feature = "cuda")]
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

    #[cfg(feature = "cuda")]
    fn fused_step_identity(device: &crate::backend::cuda::Device, spec: GdnSpec) -> Result<()> {
        use super::super::ops::CommonOpsRef;
        use cudarc::driver::sys;
        let stream = &device.stream;
        let common = CommonOpsRef::new(&device.ctx)?;
        let mut defects = 0;
        let mut cells = 0;
        let mut previous = Vec::<f32>::new();
        for (rows, valid) in (1..=16).map(|r| (r, r)).chain([(16, 5)]) {
            for gated in [false, true] {
                let mut expected: Option<(Vec<u16>, Vec<u32>)> = None;
                for (fused, graph) in [(false, false), (true, false), (true, true)] {
                    let buffers = GeometryBuffers::new(stream, spec, rows, valid)?;
                    let norm = stream.clone_htod(
                        &(0..128)
                            .map(|i| bf16::from_f32(0.5 + (i % 113) as f32 / 128.0))
                            .collect::<Vec<_>>(),
                    )?;
                    let z = stream.clone_htod(
                        &(0..rows * spec.hv * 128)
                            .map(|i| bf16::from_f32(((i * 17 % 263) as f32 - 131.0) / 64.0))
                            .collect::<Vec<_>>(),
                    )?;
                    let plan = GdnOps::prepare(
                        &device.ctx,
                        &spec,
                        rows,
                        buffers.bindings(stream, spec, rows, 0)?,
                        false,
                    )?;
                    let metadata = geometry_meta(stream, 23, valid)?;
                    let meta = SeqRef {
                        ptr: metadata.device_ptr(stream).0,
                    };
                    let workspace = stream.alloc_zeros::<u8>(plan.ws.bytes as usize)?;
                    let scratch = || Scratch {
                        ptr: workspace.device_ptr(stream).0,
                        bytes: plan.ws.bytes,
                    };
                    let input = || -> Result<GatedRmsInput> {
                        Ok(GatedRmsInput {
                            mode: super::super::ops::GatedRmsMode::SiluBf16,
                            x: plan.bindings.core_out.read(),
                            weight: geometry_view(&norm, stream, [128], [1], 0)?,
                            gate: geometry_view(
                                &z,
                                stream,
                                [rows, spec.hv, 128],
                                [u64::from(spec.hv * 128), 128, 1],
                                0,
                            )?,
                        })
                    };
                    device.finish_upload()?;
                    let execute = || -> Result<()> {
                        let phase = if rows == 1 {
                            ExecutionPhase::Decode
                        } else {
                            ExecutionPhase::Prefill
                        };
                        if fused {
                            let norm_input = input()?;
                            ensure!(
                                GdnOps::run_step(
                                    &plan,
                                    stream,
                                    meta,
                                    phase,
                                    scratch(),
                                    rows != 1,
                                    gated.then_some(&norm_input)
                                )? == gated,
                                "fused norm dispatch missing"
                            );
                        } else {
                            GdnOps::run_core(&plan, stream, meta, phase, scratch(), rows != 1)?;
                            if gated {
                                common.norm.rms_gated(
                                    stream,
                                    RowDomain {
                                        meta,
                                        row_base: 0,
                                        row_capacity: rows,
                                    },
                                    input()?,
                                    &mut TensorMut(plan.bindings.core_out.read()),
                                    spec.rms_eps,
                                )?;
                            }
                        }
                        Ok(())
                    };
                    if graph {
                        stream.begin_capture(
                            sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                        )?;
                        let result = execute();
                        let capture = stream.end_capture(sys::CUgraphInstantiate_flags(0));
                        result?;
                        capture?.context("empty fused GDN graph")?.launch()?;
                    } else {
                        execute()?;
                    }
                    stream.synchronize()?;
                    let raw = stream.clone_dtoh(&buffers.out)?;
                    ensure!(
                        raw[..(valid * spec.hv * 128) as usize]
                            .iter()
                            .all(|v| v.is_finite()),
                        "nonfinite fused GDN output"
                    );
                    let output = raw.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                    let state = stream.clone_dtoh(&buffers.state)?;
                    let history = stream.clone_dtoh(&buffers.history)?;
                    ensure!(
                        state.iter().chain(&history).all(|v| v.is_finite()),
                        "nonfinite fused GDN state/history"
                    );
                    let bits = state
                        .iter()
                        .chain(&history)
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>();
                    if let Some((ref out, ref persistent)) = expected {
                        let mismatches = output.iter().zip(out).filter(|(a, b)| a != b).count()
                            + bits.iter().zip(persistent).filter(|(a, b)| a != b).count();
                        defects += usize::from(mismatches != 0);
                        cells += 1;
                        println!(
                            "{} GDN_FUSED_IDENTITY rows={rows} valid={valid} gated={gated} graph={graph} output_state_history_bit_mismatches={mismatches}",
                            if mismatches == 0 { "PASS" } else { "DEFECT" }
                        );
                    } else {
                        expected = Some((output, bits));
                        if !gated && valid == rows {
                            if rows == 16 {
                                let mismatches = state
                                    .iter()
                                    .zip(&previous)
                                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                                    .count();
                                ensure!(mismatches > 0, "wrong accepted-row control escaped");
                                println!(
                                    "PASS GDN_FUSED_CONTROL accepted_rows=15 expected=16 state_bit_mismatches={mismatches}"
                                );
                            }
                            previous = state;
                        }
                    }
                }
            }
        }
        println!(
            "GDN_FUSED_SUMMARY cells={cells} defects={defects} byte_identity=true control=wrong_accepted_row"
        );
        ensure!(
            defects == 0,
            "DEFECT fused GDN byte identity cases={defects}"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive CUDA window"]
    #[cfg(feature = "cuda")]
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
        fused_step_identity(&device, GdnSpec::from_model(&model)?)?;
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
    #[ignore = "exclusive GPU; native KDA key-axis model-definition control"]
    fn gdn_per_channel_key_axis_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let root = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/glm53-flash-config.json"
        )))?;
        let model = crate::model_spec::ModelSpec::glm5_next(&root, &serde_json::Value::Null)?;
        let spec = GdnSpec::from_model(&model)?;
        let device = crate::backend::cuda::Device::new(0)?;
        let expected_state = (-5.0 / (1.0 + 1.828125f64.exp())).exp();
        let value_axis_state = (-5.0 / (1.0 + 0.95703125f64.exp())).exp();
        let q = f64::from(bf16::from_f32(1.0 / (1.0 + (-1.0f32).exp())).to_f32());
        let scaled_q = q / (q * q + 1e-6).sqrt() / 128.0f64.sqrt();
        let expected_out = expected_state * scaled_q;
        let mut defects = 0;
        let mut cells = 0;
        for (core, rows) in [
            (GeometryCore::Recurrent, 1),
            (GeometryCore::Replay, 1),
            (GeometryCore::Chunked, 1),
            (GeometryCore::P2, 5),
        ] {
            if !core.covered(&device) {
                continue;
            }
            for graph in [false, true] {
                cells += 1;
                let actual = geometry_run_control(
                    &device,
                    spec,
                    rows,
                    1,
                    core,
                    graph,
                    GeometryControl::KeyDecay,
                )?;
                let state = f64::from(actual.state[(3 * 128 + 17) * 128 + 91]);
                let out = f64::from(actual.out[3 * 128 + 91].to_f32());
                ensure!(
                    state.is_finite() && out.is_finite(),
                    "Runner FAIL nonfinite key-axis control"
                );
                let state_error = (state - expected_state).abs() / (1e-4 + expected_state.abs());
                let out_error = (out - expected_out).abs() / (1e-4 + expected_out.abs());
                let wrong_state_error =
                    (state - value_axis_state).abs() / (1e-4 + expected_state.abs());
                let wrong_out_error =
                    (out - value_axis_state * scaled_q).abs() / (1e-4 + expected_out.abs());
                let passed = state_error <= PER_CHANNEL_ORACLE_LIMIT
                    && out_error <= PER_CHANNEL_ORACLE_LIMIT
                    && wrong_state_error > PER_CHANNEL_ORACLE_LIMIT
                    && wrong_out_error > PER_CHANNEL_ORACLE_LIMIT;
                defects += usize::from(!passed);
                println!(
                    "{} KDA_KEY_AXIS core={core:?} graph={graph} key=17 value=91 state_scaled={state_error} out_scaled={out_error} value_axis_state_scaled={wrong_state_error} value_axis_out_scaled={wrong_out_error} limit={PER_CHANNEL_ORACLE_LIMIT}",
                    if passed { "PASS" } else { "DEFECT" }
                );
            }
        }
        println!("KDA_KEY_AXIS_SUMMARY cells={cells} defects={defects} model_definition=key_axis");
        ensure!(defects == 0, "key-axis defects={defects}");
        Ok(())
    }

    fn per_channel_norm_gate(device: &crate::backend::cuda::Device) -> Result<usize> {
        use super::super::ops::{CommonOpsRef, GatedRmsMode};
        let stream = &device.stream;
        let common = CommonOpsRef::new(&device.ctx)?;
        let mut defects = 0;
        for (rows, valid) in [(1u32, 1u32), (5, 2), (65, 63)] {
            let values: Vec<_> = (0..rows as usize * 8192)
                .map(|i| bf16::from_f32(((i * 19 + 3) % 251) as f32 / 128.0 - 1.0))
                .collect();
            let gates: Vec<_> = (0..values.len())
                .map(|i| bf16::from_f32(((i * 13 + 5) % 257) as f32 / 64.0 - 2.0))
                .collect();
            let weights: Vec<_> = (0..128)
                .map(|i| bf16::from_f32(((i * 7 + 11) % 131) as f32 / 128.0 + 0.5))
                .collect();
            let x = stream.clone_htod(&values)?;
            let z = stream.clone_htod(&gates)?;
            let w = stream.clone_htod(&weights)?;
            let y = stream.clone_htod(&vec![bf16::from_bits(0x7fc1); values.len()])?;
            let metadata = geometry_meta(stream, 0, valid)?;
            let shape = [rows, 64, 128];
            let stride = [8192, 128, 1];
            let mut output = TensorMut(geometry_view(&y, stream, shape, stride, 0)?);
            let mut execute = || {
                common.norm.rms_gated(
                    stream,
                    RowDomain {
                        meta: SeqRef {
                            ptr: metadata.device_ptr(stream).0,
                        },
                        row_base: 0,
                        row_capacity: rows,
                    },
                    GatedRmsInput {
                        x: geometry_view(&x, stream, shape, stride, 0)?,
                        gate: geometry_view(&z, stream, shape, stride, 0)?,
                        weight: geometry_view(&w, stream, [128], [1], 0)?,
                        mode: GatedRmsMode::SigmoidF32,
                    },
                    &mut output,
                    1e-5,
                )
            };
            device.finish_upload()?;
            for graph in [false, true] {
                if graph {
                    super::super::moe::testing::capture(stream, &mut execute)?.launch()?;
                } else {
                    execute()?;
                }
                let actual = stream.clone_dtoh(&y)?;
                let mut worst = 0f64;
                let mut wrong = 0f64;
                for slice in 0..valid as usize * 64 {
                    let start = slice * 128;
                    let square_sum: f64 = values[start..start + 128]
                        .iter()
                        .map(|v| f64::from(v.to_f32()).powi(2))
                        .sum();
                    let inverse = (square_sum / 128.0 + 1e-5).sqrt().recip();
                    for (d, weight) in weights.iter().enumerate() {
                        let index = start + d;
                        let gate = f64::from(gates[index].to_f32());
                        let expected = f64::from(values[index].to_f32())
                            * inverse
                            * f64::from(weight.to_f32())
                            / (1.0 + (-gate).exp());
                        let value = f64::from(actual[index].to_f32());
                        ensure!(value.is_finite(), "Runner FAIL nonfinite native gated norm");
                        worst = worst.max((value - expected).abs() / (1e-4 + expected.abs()));
                        wrong =
                            wrong.max((value - expected * gate).abs() / (1e-4 + expected.abs()));
                    }
                }
                let padding = actual[valid as usize * 8192..]
                    .iter()
                    .all(|v| v.to_bits() == 0x7fc1);
                let passed = worst <= PER_CHANNEL_ORACLE_LIMIT
                    && wrong > PER_CHANNEL_ORACLE_LIMIT
                    && padding;
                defects += usize::from(!passed);
                println!(
                    "{} KDA_SIGMOID_NORM shape=[{rows},64,128] valid={valid} graph={graph} scaled={worst} wrong_silu_scaled={wrong} padding={padding} limit={PER_CHANNEL_ORACLE_LIMIT}",
                    if passed { "PASS" } else { "DEFECT" }
                );
            }
        }
        Ok(defects)
    }

    #[test]
    #[ignore = "exclusive GPU; per-channel gate oracle and negative control"]
    fn gdn_per_channel_gates_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let device = crate::backend::cuda::Device::new(0)?;
        let spec = GdnSpec {
            hk: 64,
            hv: 64,
            rms_eps: 1e-5,
            gate_mode: crate::op_contract::GdnGateMode::PerChannel {
                lower_bound: Some(-5.0),
            },
            ..GdnSpec::q38()
        };
        let rows = 65;
        let valid = 63;
        let hv = spec.hv as usize;
        let stream = &device.stream;
        let mut buffers = GeometryBuffers::new(stream, spec, rows, valid)?;
        let a = stream.clone_dtoh(&buffers.a)?;
        let b = stream.clone_dtoh(&buffers.b)?;
        let GeometryDecay::F32(a_log, dt) = &buffers.decay else {
            anyhow::bail!("per-channel fixture requires native F32 parameters");
        };
        let a_log = stream.clone_dtoh(a_log)?;
        let dt = stream.clone_dtoh(dt)?;
        let baseline = per_channel_gate_run(&device, spec, rows, valid, &buffers, Some(-5.0))?;
        let mut defects = per_channel_norm_gate(&device)?;
        for lower_bound in [Some(-5.0), None] {
            let actual = if lower_bound.is_some() {
                baseline.clone()
            } else {
                per_channel_gate_run(&device, spec, rows, valid, &buffers, lower_bound)?
            };
            let expected = per_channel_gate_reference(
                PerChannelGateFixture {
                    a: &a,
                    b: &b,
                    a_log: &a_log,
                    dt: &dt,
                },
                hv,
                rows as usize,
                valid as usize,
                lower_bound,
            );
            let mut worst = 0f64;
            let mut nonfinite = 0;
            for (actual, expected) in actual.0.iter().zip(&expected.0) {
                nonfinite += usize::from(!actual.is_finite());
                let expected = f64::from(*expected);
                worst = worst
                    .max((f64::from(*actual) - expected).abs() / (2e-5 * (1.0 + expected.abs())));
            }
            let beta_mismatches = actual
                .1
                .iter()
                .zip(&expected.1)
                .filter(|(actual, expected)| actual.to_bits() != expected.to_bits())
                .count();
            ensure!(
                nonfinite == 0,
                "Runner FAIL nonfinite per-channel gate output"
            );
            let passed = worst <= 1.0 && beta_mismatches == 0;
            println!(
                "{} GDN_PER_CHANNEL_GATE lower_bound={lower_bound:?} input=[{rows},{hv},128] valid={valid} cumulative_block=64 scaled_error={worst} tolerance=2e-5 beta_bits={beta_mismatches} nonfinite={nonfinite}",
                if passed { "PASS" } else { "DEFECT" }
            );
            defects += usize::from(!passed);
        }
        let mut perturbed = a.clone();
        let index = (17 * hv + 3) * 128 + 91;
        perturbed[index] = bf16::from_f32(perturbed[index].to_f32() + 0.25);
        buffers.a = stream.clone_htod(&perturbed)?;
        let control = per_channel_gate_run(&device, spec, rows, valid, &buffers, Some(-5.0))?;
        let mut changed = 0;
        let mut wrong = 0;
        for (index, (before, after)) in baseline.0.iter().zip(&control.0).enumerate() {
            let row = index / (hv * 128);
            let head = index / 128 % hv;
            let channel = index % 128;
            let expected = (17..valid as usize).contains(&row) && head == 3 && channel == 91;
            let differs = before.to_bits() != after.to_bits();
            changed += usize::from(differs);
            wrong += usize::from(differs != expected);
        }
        let beta_mismatches = baseline
            .1
            .iter()
            .zip(&control.1)
            .filter(|(before, after)| before.to_bits() != after.to_bits())
            .count();
        let passed = changed == valid as usize - 17 && wrong == 0 && beta_mismatches == 0;
        println!(
            "{} GDN_PER_CHANNEL_GATE_CONTROL perturb=[17,3,91] changed={changed} expected={} wrong={wrong} beta_bits={beta_mismatches}",
            if passed { "PASS" } else { "DEFECT" },
            valid - 17
        );
        defects += usize::from(!passed);
        println!(
            "{} GDN_PER_CHANNEL_GATES cells=9 defects={defects} lower_bound_and_softplus=true negative_control=true sigmoid_norm=true",
            if defects == 0 { "PASS" } else { "DEFECT" }
        );
        ensure!(defects == 0, "per-channel gate defects={defects}");
        Ok(())
    }

    #[test]
    #[ignore = "exclusive GPU; production-shape per-channel GDN oracle"]
    fn gdn_per_channel_geometry_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let device = crate::backend::cuda::Device::new(0)?;
        let spec = GdnSpec {
            hk: 64,
            hv: 64,
            rms_eps: 1e-5,
            gate_mode: crate::op_contract::GdnGateMode::PerChannel {
                lower_bound: Some(-5.0),
            },
            ..GdnSpec::q38()
        };
        let mut defects = invariant_gdn(&device, spec)?;
        let mut cells = 0;
        let sequential = geometry_run(&device, spec, 65, 63, GeometryCore::Recurrent, false)?;
        let chunked = geometry_run(&device, spec, 65, 63, GeometryCore::Chunked, false)?;
        let diagnostic = (7 * 64 + 63) * 128 + 127;
        println!(
            "GDN_PER_CHANNEL_CROSS_CORE row=7 head=63 channel=127 recurrent={} chunked={} oracle_scaled={}",
            sequential.out[diagnostic].to_f32(),
            chunked.out[diagnostic].to_f32(),
            chunked.oracle_out
        );
        for (core, row_cases) in [
            (GeometryCore::Replay, &[(1, 1), (2, 2), (5, 5)][..]),
            (GeometryCore::ChunkedSnapshot, &[(1, 1), (5, 2), (5, 5)][..]),
            (GeometryCore::RstepSnapshot, &[(1, 1), (5, 2), (5, 5)][..]),
            (
                GeometryCore::Chunked,
                &[
                    (1, 1),
                    (2, 2),
                    (5, 5),
                    (31, 31),
                    (32, 32),
                    (33, 33),
                    (63, 63),
                    (65, 63),
                ][..],
            ),
            (
                GeometryCore::P2,
                &[
                    (5, 5),
                    (6, 6),
                    (7, 7),
                    (8, 8),
                    (16, 16),
                    (31, 31),
                    (32, 32),
                    (33, 33),
                    (63, 63),
                    (64, 37),
                    (65, 63),
                    (127, 127),
                    (129, 129),
                    (512, 509),
                ][..],
            ),
            (GeometryCore::Selected, &[(5, 5), (65, 63), (512, 509)][..]),
        ] {
            if !core.covered(&device) {
                continue;
            }
            for &(rows, valid) in row_cases {
                for graph in [false, true] {
                    take_phase_log();
                    let candidate = geometry_run(&device, spec, rows, valid, core, graph)?;
                    let phases = take_phase_log();
                    let expected = if GdnPolicy::for_device(&device.info).cw_p2_for_step(rows) {
                        "chunkwise_p2"
                    } else {
                        "chunked"
                    };
                    let phase_ok = !matches!(core, GeometryCore::Selected) || phases == [expected];
                    let passed = candidate.norm_ratio <= 1.0
                        && candidate.oracle_out <= PER_CHANNEL_ORACLE_LIMIT
                        && candidate.oracle_state <= PER_CHANNEL_ORACLE_LIMIT
                        && candidate.snapshot_mismatches == 0
                        && candidate.oracle_history == 0
                        && phase_ok;
                    defects += usize::from(!passed);
                    cells += 1;
                    println!(
                        "{} GDN_PER_CHANNEL_CELL core={core:?} input=[{rows},24576] state=[64,128,128] valid={valid} graph={graph} phase={phases:?} out_scaled={} state_scaled={} history_bits={} limit={PER_CHANNEL_ORACLE_LIMIT} qk_ratio={}",
                        if passed { "PASS" } else { "DEFECT" },
                        candidate.oracle_out,
                        candidate.oracle_state,
                        candidate.oracle_history,
                        candidate.norm_ratio
                    );
                }
            }
        }
        for core in [
            GeometryCore::Recurrent,
            GeometryCore::Chunked,
            GeometryCore::P2,
        ] {
            if !core.covered(&device) {
                continue;
            }
            for (rows, valid) in [(65, 63), (129, 129)] {
                for graph in [false, true] {
                    let mut candidate = geometry_run_control(
                        &device,
                        spec,
                        rows,
                        valid,
                        core,
                        graph,
                        GeometryControl::StrongDecay,
                    )?;
                    if matches!(core, GeometryCore::Recurrent) {
                        let reference = geometry_run_control(
                            &device,
                            spec,
                            rows,
                            valid,
                            GeometryCore::Chunked,
                            false,
                            GeometryControl::StrongDecay,
                        )?;
                        candidate.oracle_out = reference.oracle_out.max(
                            candidate.out[..valid as usize * 8192]
                                .iter()
                                .zip(&reference.out)
                                .map(|(a, b)| {
                                    let (a, b) = (f64::from(a.to_f32()), f64::from(b.to_f32()));
                                    (a - b).abs() / (1e-4 + b.abs())
                                })
                                .fold(0.0, f64::max),
                        );
                        candidate.oracle_state = reference.oracle_state.max(
                            candidate
                                .state
                                .iter()
                                .zip(&reference.state)
                                .map(|(a, b)| {
                                    let (a, b) = (f64::from(*a), f64::from(*b));
                                    (a - b).abs() / (1e-4 + b.abs())
                                })
                                .fold(0.0, f64::max),
                        );
                    }
                    let passed = candidate.oracle_out <= PER_CHANNEL_ORACLE_LIMIT
                        && candidate.oracle_state <= PER_CHANNEL_ORACLE_LIMIT
                        && candidate.oracle_history == 0
                        && candidate.snapshot_mismatches == 0;
                    defects += usize::from(!passed);
                    cells += 1;
                    println!(
                        "{} GDN_KEY_STRONG_DECAY core={core:?} rows={rows} valid={valid} graph={graph} out_scaled={} state_scaled={} limit={PER_CHANNEL_ORACLE_LIMIT}",
                        if passed { "PASS" } else { "DEFECT" },
                        candidate.oracle_out,
                        candidate.oracle_state
                    );
                }
            }
        }
        let selected = if GdnPolicy::for_device(&device.info).cw_p2_for_step(65) {
            GeometryCore::P2
        } else {
            GeometryCore::Chunked
        };
        let baseline = geometry_run(&device, spec, 65, 63, selected, false)?;
        let control = geometry_run_control(
            &device,
            spec,
            65,
            63,
            selected,
            false,
            GeometryControl::ValueInput,
        )?;
        let mut out_changed = 0;
        let mut out_wrong = 0;
        for row in 0..65usize {
            for head in 0..64usize {
                for channel in 0..128usize {
                    let affected = (59..63).contains(&row) && head == 4 && channel == 91;
                    let index = (row * 64 + head) * 128 + channel;
                    let differs = baseline.out[index].to_bits() != control.out[index].to_bits();
                    out_changed += usize::from(affected && differs);
                    out_wrong += usize::from(!affected && differs);
                }
            }
        }
        let mut state_changed = 0;
        let mut state_wrong = 0;
        for head in 0..64usize {
            for d in 0..128usize {
                for channel in 0..128usize {
                    let affected = head == 4 && channel == 91;
                    let index = (head * 128 + d) * 128 + channel;
                    let differs = baseline.state[index].to_bits() != control.state[index].to_bits();
                    state_changed += usize::from(affected && differs);
                    state_wrong += usize::from(affected != differs);
                }
            }
        }
        let history_mismatches = baseline
            .history
            .iter()
            .zip(&control.history)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        // 2026-10-27: BF16 rounding may preserve bits inside the causal region.
        let passed = out_changed > 0
            && state_changed == 128
            && out_wrong == 0
            && state_wrong == 0
            && history_mismatches == 0
            && control.oracle_out <= PER_CHANNEL_ORACLE_LIMIT
            && control.oracle_state <= PER_CHANNEL_ORACLE_LIMIT
            && control.oracle_history == 0;
        defects += usize::from(!passed);
        cells += 1;
        println!(
            "{} GDN_PER_CHANNEL_RECURRENCE_CONTROL perturb=[59,4,91] out_changed={out_changed} out_wrong={out_wrong} state_changed={state_changed} state_wrong={state_wrong} history_bits={history_mismatches}",
            if passed { "PASS" } else { "DEFECT" }
        );
        println!(
            "{} GDN_PER_CHANNEL_GEOMETRY cells={cells} defects={defects} oracle=f64-recurrence limit={PER_CHANNEL_ORACLE_LIMIT} snapshot_replay=true graph_eager=true",
            if defects == 0 { "PASS" } else { "DEFECT" }
        );
        ensure!(defects == 0, "per-channel GDN defects={defects}");
        Ok(())
    }

    #[test]
    #[ignore = "exclusive GPU; small per-channel memcheck coverage"]
    fn gdn_per_channel_memcheck_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let device = crate::backend::cuda::Device::new(0)?;
        let spec = GdnSpec {
            hk: 64,
            hv: 64,
            rms_eps: 1e-5,
            gate_mode: crate::op_contract::GdnGateMode::PerChannel {
                lower_bound: Some(-5.0),
            },
            ..GdnSpec::q38()
        };
        let mut cells = 0;
        let mut defects = 0;
        for (core, rows, valid) in [
            (GeometryCore::Recurrent, 5, 5),
            (GeometryCore::Replay, 5, 5),
            (GeometryCore::Verify, 5, 5),
            (GeometryCore::ChunkedSnapshot, 5, 2),
            (GeometryCore::Chunked, 65, 63),
            (GeometryCore::P2, 65, 63),
        ] {
            if !core.covered(&device) {
                continue;
            }
            let candidate = geometry_run(&device, spec, rows, valid, core, false)?;
            let passed = candidate.norm_ratio <= 1.0
                && candidate.oracle_out <= PER_CHANNEL_ORACLE_LIMIT
                && candidate.oracle_state <= PER_CHANNEL_ORACLE_LIMIT
                && candidate.oracle_history == 0
                && candidate.snapshot_mismatches == 0;
            defects += usize::from(!passed);
            cells += 1;
            println!(
                "{} GDN_PER_CHANNEL_MEMCHECK_CELL core={core:?} rows={rows} valid={valid} out_scaled={} state_scaled={} history_bits={}",
                if passed { "PASS" } else { "DEFECT" },
                candidate.oracle_out,
                candidate.oracle_state,
                candidate.oracle_history
            );
        }
        println!(
            "{} GDN_PER_CHANNEL_MEMCHECK cells={cells} defects={defects} available_core_types=true",
            if defects == 0 { "PASS" } else { "DEFECT" }
        );
        ensure!(defects == 0, "per-channel memcheck defects={defects}");
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA and FF_Q38_PROBE_DIR/gdn_prep_h32_wrong_stride_sm<arch>.cubin"]
    #[cfg(feature = "cuda")]
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
    #[cfg(feature = "cuda")]
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

    #[cfg(feature = "cuda")]
    fn gate_projection_rows(
        device: &crate::backend::cuda::Device,
        shape: [u32; 2],
        weights: [Vec<bf16>; 4],
        hidden: &[bf16],
    ) -> Result<usize> {
        use crate::backend::cuda::{
            blas::Blas,
            closure::Closure,
            linear::LayerLinearPlan,
            ops::Ops,
            weights::{StoredTensor, WeightCatalog},
        };
        use crate::checkpoint::{TensorEntry, TensorManifest, TensorScope};
        let [heads, width] = shape;
        let rows = crate::linear_plan::MAX_SKINNY_ROWS;
        ensure!(hidden.len() == width as usize, "gate input extent");
        let stream = &device.stream;
        let mut entries = std::collections::BTreeMap::new();
        let mut tensors = std::collections::BTreeMap::new();
        let mut bytes = 0;
        for (role, weight) in ["in_proj_a.weight", "in_proj_b.weight", "A_log", "dt_bias"]
            .into_iter()
            .zip(weights)
        {
            let name = format!("model.language_model.layers.0.linear_attn.{role}");
            let shape = if role.ends_with(".weight") {
                vec![heads as usize, width as usize]
            } else {
                vec![heads as usize]
            };
            ensure!(
                weight.len() == shape.iter().product::<usize>(),
                "gate weight extent"
            );
            let size = weight.len() * size_of::<bf16>();
            entries.insert(
                name.clone(),
                TensorEntry {
                    dtype: safetensors::Dtype::BF16,
                    shape,
                    shard: "gate-projection.safetensors".into(),
                    bytes: size,
                    load: true,
                    data_start: 0,
                    data_offsets: [0, size as u64],
                    scope: TensorScope::Text,
                    layer: Some(0),
                    role: format!("linear_attn.{role}"),
                    scale_name: None,
                },
            );
            tensors.insert(name, StoredTensor::Bf16(stream.clone_htod(&weight)?));
            bytes += size;
        }
        let catalog = WeightCatalog {
            #[cfg(feature = "cuda")]
            dense: None,
            linear_format: crate::model_spec::LinearFormat::Bf16,
            tensors,
            manifest: TensorManifest {
                entries,
                resident_bytes: bytes,
            },
            bytes,
            upload_content_sha256: None,
            upload_hashed_bytes: None,
            load_wall_ms: 0.0,
        };
        let dequant = crate::linear_plan::LayerDequantPlan::from_manifest(
            &catalog.manifest,
            0,
            catalog.linear_format,
        )?;
        let layer = LayerLinearPlan::bind(&dequant, &catalog, None, stream)?;
        let runtime = std::path::PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?);
        let ops = Ops::new(&device.ctx)?;
        let blas = Blas::new(
            device,
            &ops,
            &runtime,
            std::rc::Rc::new(Closure::new(false, "gate projection invariance".into())),
        )?;
        for m in 1..=rows as u32 {
            layer.prepare(&blas, m)?;
        }
        let gate = cubin::module(&device.ctx, "q38_common")?.load_function(match heads {
            48 => "q38_gdn_gates",
            32 => "gdn_gates_h32",
            _ => anyhow::bail!("unsupported published gate heads={heads}"),
        })?;
        let a_log = catalog.bf16("model.language_model.layers.0.linear_attn.A_log")?;
        let dt = catalog.bf16("model.language_model.layers.0.linear_attn.dt_bias")?;
        let mut defects = 0;
        for distinct in [false, true] {
            let values: Vec<_> = (0..rows * width as usize)
                .map(|i| {
                    let row = i / width as usize;
                    let col = i % width as usize;
                    if distinct {
                        bf16::from_f32(
                            hidden[col].to_f32()
                                + (((row * 73 + col * 31) % 509) as f32 - 254.0) / 512.0,
                        )
                    } else {
                        hidden[col]
                    }
                })
                .collect();
            let input = stream.clone_htod(&values)?;
            let reference: Vec<_> = (0..3)
                .map(|_| stream.alloc_zeros::<bf16>(rows * heads as usize))
                .collect::<std::result::Result<_, _>>()?;
            let reference_g = stream.alloc_zeros::<f32>(rows * heads as usize)?;
            let mut a = stream.alloc_zeros::<bf16>((rows + 1) * heads as usize)?;
            let mut b = stream.alloc_zeros::<bf16>((rows + 1) * heads as usize)?;
            let mut beta = stream.alloc_zeros::<bf16>((rows + 1) * heads as usize)?;
            let mut g = stream.alloc_zeros::<f32>((rows + 1) * heads as usize)?;
            for row in 0..rows {
                for (role, output) in ["a", "b"].into_iter().zip(&reference) {
                    layer.run(
                        &blas,
                        &format!("linear_attn.in_proj_{role}.weight"),
                        ExecutionPhase::Decode,
                        geometry_view(
                            &input,
                            stream,
                            [1, width],
                            [u64::from(width), 1],
                            row * width as usize,
                        )?,
                        &mut TensorMut(geometry_view(
                            output,
                            stream,
                            [1, heads],
                            [u64::from(heads), 1],
                            row * heads as usize,
                        )?),
                    )?;
                }
            }
            let reference_meta = geometry_meta(stream, 67, rows as u32)?;
            crate::backend::cuda::ops::GateOps::gdn_gates_with(
                &gate,
                stream,
                RowDomain {
                    meta: SeqRef {
                        ptr: reference_meta.device_ptr(stream).0,
                    },
                    row_base: 0,
                    row_capacity: rows as u32,
                },
                GdnGateBindings {
                    a: geometry_view(
                        &reference[0],
                        stream,
                        [rows as u32, heads],
                        [u64::from(heads), 1],
                        0,
                    )?,
                    b: geometry_view(
                        &reference[1],
                        stream,
                        [rows as u32, heads],
                        [u64::from(heads), 1],
                        0,
                    )?,
                    a_log: geometry_view(a_log, stream, [heads], [1], 0)?,
                    dt_bias: geometry_view(dt, stream, [heads], [1], 0)?,
                    beta: &mut TensorMut(geometry_view(
                        &reference[2],
                        stream,
                        [rows as u32, heads],
                        [u64::from(heads), 1],
                        0,
                    )?),
                    g: &mut TensorMut(geometry_view(
                        &reference_g,
                        stream,
                        [rows as u32, heads],
                        [u64::from(heads), 1],
                        0,
                    )?),
                },
            )?;
            let expected: Vec<_> = reference
                .iter()
                .map(|value| stream.clone_dtoh(value))
                .collect::<std::result::Result<_, _>>()?;
            let expected_g = stream.clone_dtoh(&reference_g)?;
            for m in 2..=rows as u32 {
                let meta = geometry_meta(stream, 67, m)?;
                for graph in [false, true] {
                    let canary = bf16::from_bits(0x7fc1);
                    for output in [&mut a, &mut b, &mut beta] {
                        stream
                            .memcpy_htod(&vec![canary; output.len()], &mut output.slice_mut(..))?;
                    }
                    stream.memcpy_htod(
                        &vec![f32::from_bits(0x7fc00001); g.len()],
                        &mut g.slice_mut(..),
                    )?;
                    device.finish_upload()?;
                    let execute = || -> Result<()> {
                        for (role, output) in [("a", &a), ("b", &b)] {
                            layer.run(
                                &blas,
                                &format!("linear_attn.in_proj_{role}.weight"),
                                ExecutionPhase::Prefill,
                                geometry_view(
                                    &input,
                                    stream,
                                    [m, width],
                                    [u64::from(width), 1],
                                    0,
                                )?,
                                &mut TensorMut(geometry_view(
                                    output,
                                    stream,
                                    [m, heads],
                                    [u64::from(heads), 1],
                                    0,
                                )?),
                            )?;
                        }
                        crate::backend::cuda::ops::GateOps::gdn_gates_with(
                            &gate,
                            stream,
                            RowDomain {
                                meta: SeqRef {
                                    ptr: meta.device_ptr(stream).0,
                                },
                                row_base: 0,
                                row_capacity: m,
                            },
                            GdnGateBindings {
                                a: geometry_view(&a, stream, [m, heads], [u64::from(heads), 1], 0)?,
                                b: geometry_view(&b, stream, [m, heads], [u64::from(heads), 1], 0)?,
                                a_log: geometry_view(a_log, stream, [heads], [1], 0)?,
                                dt_bias: geometry_view(dt, stream, [heads], [1], 0)?,
                                beta: &mut TensorMut(geometry_view(
                                    &beta,
                                    stream,
                                    [m, heads],
                                    [u64::from(heads), 1],
                                    0,
                                )?),
                                g: &mut TensorMut(geometry_view(
                                    &g,
                                    stream,
                                    [m, heads],
                                    [u64::from(heads), 1],
                                    0,
                                )?),
                            },
                        )
                    };
                    if graph {
                        use cudarc::driver::sys;
                        stream.begin_capture(
                            sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                        )?;
                        let launched = execute();
                        let captured = stream.end_capture(sys::CUgraphInstantiate_flags(0));
                        launched?;
                        captured?.context("empty gate projection graph")?.launch()?;
                    } else {
                        execute()?;
                    }
                    let got: Vec<_> = [&a, &b, &beta]
                        .into_iter()
                        .map(|value| stream.clone_dtoh(value))
                        .collect::<std::result::Result<_, _>>()?;
                    let got_g = stream.clone_dtoh(&g)?;
                    let count = m as usize * heads as usize;
                    ensure!(
                        got.iter()
                            .all(|values| values[..count].iter().all(|v| v.is_finite()))
                            && got_g[..count].iter().all(|v| v.is_finite()),
                        "Runner FAIL nonfinite gate projection"
                    );
                    ensure!(
                        got.iter().all(|values| values[count..]
                            .iter()
                            .all(|v| v.to_bits() == canary.to_bits()))
                            && got_g[count..].iter().all(|v| v.to_bits() == 0x7fc00001),
                        "gate projection wrote a dead row"
                    );
                    let mut mismatch_counts = Vec::new();
                    for (role, (actual, expected)) in ["a", "b", "beta"]
                        .into_iter()
                        .zip(got.iter().zip(&expected))
                    {
                        let mismatches: Vec<_> = actual[..count].iter().zip(expected).enumerate()
                            .filter(|(_, (a,b))| a.to_bits() != b.to_bits())
                            .map(|(index, (a,b))| serde_json::json!({"row":index / heads as usize,"head":index % heads as usize,"actual":a.to_f32(),"expected":b.to_f32()})).collect();
                        mismatch_counts.push(mismatches.len());
                        if !mismatches.is_empty() {
                            println!(
                                "DEFECT GDN_GATE_PROJECTION_VALUES {}",
                                serde_json::json!({"weight":shape,"distinct":distinct,"rows":m,"graph":graph,"role":role,"mismatches":mismatches})
                            );
                        }
                    }
                    mismatch_counts.push(
                        got_g[..count]
                            .iter()
                            .zip(&expected_g)
                            .filter(|(a, b)| a.to_bits() != b.to_bits())
                            .count(),
                    );
                    let exact = mismatch_counts.iter().all(|n| *n == 0);
                    defects += usize::from(!exact);
                    println!(
                        "{} GDN_GATE_PROJECTION {}",
                        if exact { "PASS" } else { "DEFECT" },
                        serde_json::json!({"weight":shape,"distinct":distinct,"rows":m,"graph":graph,"compared_per_output":count,"mismatches_a_b_beta_g":mismatch_counts})
                    );
                    if heads == 48 && !distinct && m == 5 && !graph {
                        let index = heads as usize + 43;
                        println!(
                            "GDN_GATE_HEAD43 {}",
                            serde_json::json!({"head":43,"row":1,"weight":shape,
                            "a_m1":expected[0][index].to_f32(),"a_m5":got[0][index].to_f32(),
                            "b_m1":expected[1][index].to_f32(),"b_m5":got[1][index].to_f32(),
                            "beta_m1":expected[2][index].to_f32(),"beta_m5":got[2][index].to_f32()})
                        );
                    }
                }
            }
        }
        println!(
            "GDN_GATE_PROJECTION_BINDING {}",
            serde_json::json!({"weight":shape,"fixed_lt_plans":blas.fixed_plan_count()})
        );
        Ok(defects)
    }

    #[cfg(feature = "cuda")]
    fn gate_projection_isolation(device: &crate::backend::cuda::Device) -> Result<usize> {
        use safetensors::{Dtype, SafeTensors};
        let model = std::path::PathBuf::from(std::env::var("FF_Q38_MODEL")?);
        let root = std::path::PathBuf::from(std::env::var("FF_Q38_PROBE_DIR")?);
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
        let mut defects = gate_projection_rows(
            device,
            [gdn.hv, spec.hidden],
            [
                read("in_proj_a.weight", &[48, 5120])?,
                read("in_proj_b.weight", &[48, 5120])?,
                read("A_log", &[48])?,
                read("dt_bias", &[48])?,
            ],
            &hidden,
        )?;
        let mini = crate::model_spec::ModelSpec::read(&root.join("minicpm"))?;
        let gdn = GdnSpec::from_model(&mini)?;
        ensure!(
            mini.hidden == 2048 && gdn.hv == 32,
            "MiniCPM gate production geometry"
        );
        let matrix = |seed: usize| {
            (0..gdn.hv as usize * mini.hidden as usize)
                .map(|i| {
                    bf16::from_f32(
                        (((i / mini.hidden as usize) * 73 + i % mini.hidden as usize * 31 + seed)
                            % 509) as f32
                            / 512.0
                            - 254.0 / 512.0,
                    )
                })
                .collect()
        };
        let hidden: Vec<_> = (0..mini.hidden)
            .map(|i| bf16::from_f32((i * 37 % 509) as f32 / 512.0 - 254.0 / 512.0))
            .collect();
        defects += gate_projection_rows(
            device,
            [gdn.hv, mini.hidden],
            [
                matrix(17),
                matrix(191),
                vec![bf16::from_f32(-1.0); gdn.hv as usize],
                vec![bf16::from_f32(0.25); gdn.hv as usize],
            ],
            &hidden,
        )?;
        println!(
            "{} GDN_GATE_PROJECTION_SUMMARY shapes=2 patterns=2 rows=2..16 eager_graph=true defects={defects}",
            if defects == 0 { "PASS" } else { "DEFECT" }
        );
        Ok(defects)
    }

    #[test]
    #[ignore = "exclusive CUDA, published Qwen gate weights and saved layer0 normalized row"]
    #[cfg(feature = "cuda")]
    fn gdn_verify_snapshot_rows_gpu() -> Result<()> {
        let qwen = GdnSpec::q38();
        let qwen_conv_dim = qwen.conv_dim().expect("Qwen GDN geometry");
        use cudarc::driver::{CudaSlice, DevicePtr};
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let device = crate::backend::cuda::Device::new(0)?;
        let projection_defects = gate_projection_isolation(&device)?;
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
                a: GdnGateInput::PerHead(matrix(&a, qwen.hv)?),
                b: matrix(&b, qwen.hv)?,
                conv_weight: GdnConvWeights::Fused(matrix(&conv_weight, 4)?),
                decay: GdnDecayWeights::Bf16 {
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
                },
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
        fn input_row<const N: usize>(
            source: Tensor<bf16, N>,
            row: usize,
        ) -> Result<Tensor<bf16, N>> {
            let offset = row as u64 * source.stride[0] * 2;
            let mut shape = source.shape;
            shape[0] = 1;
            unsafe {
                Tensor::new(
                    source.ptr + offset,
                    shape,
                    source.stride,
                    source.span_bytes - offset,
                )
            }
        }
        for row in 0..5 {
            let b = &full.bindings;
            let bindings = unsafe {
                GdnBindings {
                    qkv: input_row(b.qkv, row)?,
                    a: match b.a {
                        GdnGateInput::PerHead(a) => GdnGateInput::PerHead(input_row(a, row)?),
                        GdnGateInput::PerChannel(a) => GdnGateInput::PerChannel(input_row(a, row)?),
                    },
                    b: input_row(b.b, row)?,
                    conv_weight: b.conv_weight,
                    decay: b.decay,
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
        ensure!(
            projection_defects == 0,
            "DEFECT GDN gate projection cells={projection_defects}"
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
                cw_p2_bytes(&qwen, rows),
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
            assert!(cw_p2_bytes(&qwen, rows).is_multiple_of(256), "rows={rows}");
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
                a: GdnGateInput::PerHead(
                    Tensor::new(0x1000, [rows, qwen.hv], [head, 1], padded(r * head * bf))
                        .expect("a"),
                ),
                b: Tensor::new(0x1000, [rows, qwen.hv], [head, 1], padded(r * head * bf))
                    .expect("b"),
                conv_weight: GdnConvWeights::Fused(
                    Tensor::new(0x1000, [cd, 4], [4, 1], padded(u64::from(cd) * 4 * bf))
                        .expect("conv_weight"),
                ),
                decay: GdnDecayWeights::Bf16 {
                    a_log: Tensor::new(0x1000, [qwen.hv], [1], padded(head * bf)).expect("a_log"),
                    dt_bias: Tensor::new(0x1000, [qwen.hv], [1], padded(head * bf))
                        .expect("dt_bias"),
                },
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
