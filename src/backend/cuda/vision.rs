//! Vision binder: 2D RoPE + MHA attention (head_dim 72) and the merger (LayerNorm + two biased BF16 linears + exact GELU).
//!
//! Layouts, all head-major over rows:
//!
//! | view | shape | semantics |
//! |---|---|---|
//! | `q_in`, `k_in` | `[16, rows, 72]` | real rows only, pre-RoPE |
//! | `q_rot`, `k_rot` | `[16, rows, 72]` | scratch; rotated Q, rotated K real rows |
//! | `k_pad` | `[16, krows, 72]` | scratch; image-padded K, the attention tile source |
//! | `v` | `[16, krows, 72]` | image-padded V (no RoPE); caller-owned |
//! | `o` | `[rows, 16, 72]` | token-major — the following projection's `[rows, 1152]` |
//!
//! Image `b`'s keys occupy `[kb[b], kb[b] + n_b)` rounded up to a KT = 64 tile with `kb[b] % 64 == 0`;
//! padded rows of `k_pad` and `v` are never read from global memory.
//!
//! The merger requires merge-grouped packed patch order (2×2 blocks contiguous).

use super::Device;
use super::blas::Blas;
use super::contracts::{Scratch, Tensor, TensorMut};
use super::cubin;
use super::linear::LinearBiasMode;
use super::ops::{CommonOpsRef, GeluMode};
use crate::op_contract::WorkspaceReq;
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use half::bf16;
use std::sync::Arc;

pub(crate) const HQ: u32 = 16;
pub(crate) const D: u32 = 72;
/// Query rows per attention CTA and the key tile, from `kVisionTileSpec` in `kernels/vision_attn.cu`.
const MT: u32 = 64;
const KT: u32 = 64;
const ATTN_THREADS: u32 = 128;
const ROPE_THREADS: u32 = 256;
const MERGE: u32 = 2;
const HIDDEN: u32 = 1152;
const MERGER_FC1: u32 = HIDDEN * MERGE * MERGE;
const MERGER_FC2: u32 = 5120;
const LN_EPS: f32 = 1e-6;
/// Biased linears accumulate in FP32, add bias, then round once to BF16.
const BIAS_MODE: LinearBiasMode = LinearBiasMode::BeforeOutputRound;
const WS_ALIGN: u64 = 256;

/// The declared vision-tower shape; `validate` rejects any shape the compiled kernels do not implement.
#[derive(Clone, Copy, Debug)]
pub(crate) struct VisionSpec {
    pub hq: u32,
    pub d: u32,
    pub hidden: u32,
    pub fc1: u32,
    pub fc2: u32,
    pub ln_eps: f32,
}

impl VisionSpec {
    pub const fn q38() -> Self {
        Self {
            hq: 16,
            d: 72,
            hidden: HIDDEN,
            fc1: MERGER_FC1,
            fc2: MERGER_FC2,
            ln_eps: LN_EPS,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.hq == HQ
                && self.d == D
                && self.hidden == HIDDEN
                && self.fc1 == MERGER_FC1
                && self.fc2 == MERGER_FC2
                && self.ln_eps == LN_EPS,
            "unsupported vision specification"
        );
        Ok(())
    }
}

/// One image's patch grid, in patches (`image_grid_thw`'s `[h, w]`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PatchGrid {
    pub h: u32,
    pub w: u32,
}

/// Vision attention packing: image `b`'s real rows are `[cu_seqlens[b], cu_seqlens[b + 1])` in `grids` order;
/// its keys occupy `[k_bases[b], k_bases[b + 1])`, rounded up to a KT tile; every `k_bases[b]` is a multiple of KT.
#[derive(Clone, Debug)]
pub(crate) struct PatchPacking {
    grids: Vec<PatchGrid>,
    cu_seqlens: Vec<i32>,
    k_bases: Vec<i32>,
}

impl PatchPacking {
    pub(crate) fn build(grids: &[PatchGrid]) -> Result<Self> {
        ensure!(
            !grids.is_empty(),
            "vision attention requires at least one image"
        );
        let mut cu: Vec<i32> = Vec::with_capacity(grids.len() + 1);
        let mut kb: Vec<i32> = Vec::with_capacity(grids.len() + 1);
        cu.push(0);
        kb.push(0);
        let mut rows = 0u64;
        let mut padded = 0u64;
        for g in grids {
            ensure!(g.h > 0 && g.w > 0, "empty patch grid");
            ensure!(
                g.h.is_multiple_of(MERGE) && g.w.is_multiple_of(MERGE),
                "patch grid {}x{} is not merge-aligned",
                g.h,
                g.w
            );
            let n = u64::from(g.h) * u64::from(g.w);
            rows = rows.checked_add(n).context("patch rows overflow")?;
            padded = padded
                .checked_add(n.div_ceil(u64::from(KT)) * u64::from(KT))
                .context("padded key rows overflow")?;
            ensure!(
                rows <= i32::MAX as u64 && padded <= i32::MAX as u64,
                "patch rows exceed the kernel's 32-bit extents"
            );
            cu.push(i32::try_from(rows)?);
            kb.push(i32::try_from(padded)?);
        }
        ensure!(
            grids.len() <= u16::MAX as usize,
            "image count exceeds the grid z limit"
        );
        Ok(Self {
            grids: grids.to_vec(),
            cu_seqlens: cu,
            k_bases: kb,
        })
    }

    /// The images in packing order — the attention launch's z order.
    pub(crate) fn grids(&self) -> &[PatchGrid] {
        &self.grids
    }

    /// Real-row offsets, `images + 1` entries, `[0, real_rows]`.
    pub(crate) fn cu_seqlens(&self) -> &[i32] {
        &self.cu_seqlens
    }

    /// Padded key offsets, `images + 1` entries, `[0, padded_rows]`.
    pub(crate) fn k_bases(&self) -> &[i32] {
        &self.k_bases
    }

    /// Total real patch rows across all images.
    pub(crate) fn real_rows(&self) -> u32 {
        *self.cu_seqlens.last().expect("packing has entries") as u32
    }

    /// Total key rows after per-image padding to the KT tile.
    pub(crate) fn padded_rows(&self) -> u32 {
        *self.k_bases.last().expect("packing has entries") as u32
    }

    pub(crate) fn images(&self) -> usize {
        self.grids.len()
    }
}

/// The attention stage's views; every pointer is stable through capture.
#[derive(Debug)]
pub(crate) struct VisionAttnBindings {
    /// The qkv projection's Q slice, head-major real rows `[16, rows, 72]`.
    pub q_in: Tensor<bf16, 3>,
    /// The qkv projection's K slice, head-major real rows `[16, rows, 72]`.
    pub k_in: Tensor<bf16, 3>,
    /// V, head-major and image-padded `[16, krows, 72]`, no RoPE.
    pub v: Tensor<bf16, 3>,
    /// `[rows, 16, 72]` token-major.
    pub o: TensorMut<bf16, 3>,
}

/// Merger views: per-row LayerNorm(1152, bias), then Linear(4608→4608)+GELU(exact)+Linear(4608→5120) over 2×2 blocks.
#[derive(Debug)]
pub(crate) struct VisionMergerBindings {
    /// The last block's output, `[rows, 1152]`, all rows real.
    pub x: Tensor<bf16, 2>,
    pub norm_w: Tensor<bf16, 1>,
    pub norm_b: Tensor<bf16, 1>,
    pub fc1_w: Tensor<bf16, 2>,
    pub fc1_b: Tensor<bf16, 1>,
    pub fc2_w: Tensor<bf16, 2>,
    pub fc2_b: Tensor<bf16, 1>,
    /// `[rows/4, 5120]`, the image features the splice consumes.
    pub out: TensorMut<bf16, 2>,
}

pub(crate) struct VisionAttnPlan {
    rope: CudaFunction,
    attn: CudaFunction,
    bindings: VisionAttnBindings,
    cu: CudaSlice<i32>,
    kb: CudaSlice<i32>,
    grid_h: CudaSlice<i32>,
    grid_w: CudaSlice<i32>,
    rows: u32,
    nblocks: u32,
    /// (src offset in the k_rot region, dst offset in the k_pad region, bytes)
    copies: Vec<(u64, u64, u64)>,
    off_q_rot: u64,
    off_k_rot: u64,
    off_k_pad: u64,
    scratch_bytes: u64,
    rope_cfg: LaunchConfig,
    attn_cfg: LaunchConfig,
}

pub(crate) struct VisionMergerPlan {
    bindings: VisionMergerBindings,
    common: CommonOpsRef,
    rows: u32,
    tokens: u32,
    off_ln: u64,
    off_fc1: u64,
    scratch_bytes: u64,
}

pub(crate) struct VisionAttnOps;

pub(crate) struct VisionMergerOps;

fn align(n: u64) -> Result<u64> {
    Ok(n.checked_add(WS_ALIGN - 1)
        .context("workspace addition overflow")?
        & !(WS_ALIGN - 1))
}

/// Scratch layout of the attention stage: `(q_rot, k_rot, k_pad, bytes)`.
fn attn_layout(rows: u32, krows: u32) -> Result<(u64, u64, u64, u64)> {
    let row_bytes = u64::from(HQ) * u64::from(D) * 2;
    let rot_bytes = align(u64::from(rows) * row_bytes)?;
    let off_q_rot = 0u64;
    let off_k_rot = rot_bytes;
    let off_k_pad = off_k_rot
        .checked_add(rot_bytes)
        .context("workspace addition overflow")?;
    let bytes = off_k_pad
        .checked_add(u64::from(krows) * row_bytes)
        .context("workspace addition overflow")?;
    Ok((off_q_rot, off_k_rot, off_k_pad, bytes))
}

/// Scratch layout of the merger: `(ln, fc1, bytes)`; both buffers are `rows * 1152 * 2` bytes.
fn merger_layout(rows: u32) -> Result<(u64, u64, u64)> {
    let buf = align(u64::from(rows) * u64::from(HIDDEN) * 2)?;
    Ok((0, buf, buf.checked_mul(2).context("workspace overflow")?))
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

fn dense<T: Copy, const R: usize>(view: &Tensor<T, R>, name: &str) -> Result<()> {
    let mut stride = 1u64;
    for axis in (0..R).rev() {
        ensure!(view.stride[axis] == stride, "{name} must be contiguous");
        stride *= u64::from(view.shape[axis]);
    }
    Ok(())
}

impl VisionAttnOps {
    /// CPU-only checked sizing of the RoPE outputs and the padded K staging.
    pub(crate) fn workspace(spec: &VisionSpec, packing: &PatchPacking) -> Result<WorkspaceReq> {
        spec.validate()?;
        let (_, _, _, bytes) = attn_layout(packing.real_rows(), packing.padded_rows())?;
        Ok(WorkspaceReq {
            bytes,
            alignment: WS_ALIGN,
        })
    }

    /// Per-request setup binder; the image grid is part of the plan identity.
    pub(crate) fn prepare(
        device: &Device,
        spec: &VisionSpec,
        packing: &PatchPacking,
        bindings: VisionAttnBindings,
    ) -> Result<VisionAttnPlan> {
        spec.validate()?;
        let grids = packing.grids();
        let cu = packing.cu_seqlens();
        let kb = packing.k_bases();
        let rows = packing.real_rows();
        let krows = packing.padded_rows();
        let nblocks = packing.images() as u32;

        want(&bindings.q_in, [HQ, rows, D], "q_in")?;
        want(&bindings.k_in, [HQ, rows, D], "k_in")?;
        want(&bindings.v, [HQ, krows, D], "v")?;
        let o = bindings.o.read();
        want(&o, [rows, HQ, D], "o")?;
        for (view, name) in [
            (&bindings.q_in, "q_in"),
            (&bindings.k_in, "k_in"),
            (&bindings.v, "v"),
            (&o, "o"),
        ] {
            dense(view, name)?;
        }
        // The RoPE kernel bounds its flat thread index by `rows * HQ` in 32-bit.
        ensure!(
            u64::from(rows) * u64::from(HQ) <= i32::MAX as u64,
            "rows x heads exceed the RoPE kernel's 32-bit index"
        );
        let max_n = grids
            .iter()
            .map(|g| u64::from(g.h) * u64::from(g.w))
            .max()
            .unwrap_or(0);
        ensure!(
            max_n.div_ceil(u64::from(MT)) <= u64::from(u16::MAX),
            "query tiles exceed the grid x limit"
        );

        let (off_q_rot, off_k_rot, off_k_pad, scratch_bytes) = attn_layout(rows, krows)?;

        // Per-image copies of rotated K real rows to their padded offsets, fixed by the plan.
        let mut copies = Vec::with_capacity(grids.len() * HQ as usize);
        for h in 0..u64::from(HQ) {
            for (b, g) in grids.iter().enumerate() {
                let n = u64::from(g.h) * u64::from(g.w);
                let src = (h * u64::from(rows) + cu[b] as u64) * u64::from(D) * 2;
                let dst = (h * u64::from(krows) + kb[b] as u64) * u64::from(D) * 2;
                copies.push((src, dst, n * u64::from(D) * 2));
            }
        }

        let stream = &device.stream;
        let cu_dev = stream.clone_htod(cu)?;
        let kb_dev = stream.clone_htod(kb)?;
        let grid_h = stream.clone_htod(
            &grids
                .iter()
                .map(|g| i32::try_from(g.h))
                .collect::<Result<Vec<_>, _>>()?,
        )?;
        let grid_w = stream.clone_htod(
            &grids
                .iter()
                .map(|g| i32::try_from(g.w))
                .collect::<Result<Vec<_>, _>>()?,
        )?;

        let module = cubin::module(&device.ctx, "vision_attn")?;
        let attn = module.load_function("vision_attn_kernel")?;
        let rope =
            cubin::module(&device.ctx, "vision_rope")?.load_function("vision_rope_kernel")?;

        let rope_cfg = LaunchConfig {
            grid_dim: (
                (u64::from(rows) * u64::from(HQ)).div_ceil(u64::from(ROPE_THREADS)) as u32,
                1,
                1,
            ),
            block_dim: (ROPE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let attn_cfg = LaunchConfig {
            grid_dim: (max_n.div_ceil(u64::from(MT)) as u32, HQ, nblocks),
            block_dim: (ATTN_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        Ok(VisionAttnPlan {
            rope,
            attn,
            bindings,
            cu: cu_dev,
            kb: kb_dev,
            grid_h,
            grid_w,
            rows,
            nblocks,
            copies,
            off_q_rot,
            off_k_rot,
            off_k_pad,
            scratch_bytes,
            rope_cfg,
            attn_cfg,
        })
    }

    /// Allocation-free, same-stream forward: RoPE, K scatter into the padded layout, attention.
    pub(crate) fn run(
        plan: &VisionAttnPlan,
        stream: &Arc<CudaStream>,
        scratch: Scratch,
    ) -> Result<()> {
        scratch.validate(plan.scratch_bytes)?;
        let b = &plan.bindings;
        let (cu_ptr, _cu_hold) = plan.cu.device_ptr(stream);
        let (kb_ptr, _kb_hold) = plan.kb.device_ptr(stream);
        let (gh_ptr, _gh_hold) = plan.grid_h.device_ptr(stream);
        let (gw_ptr, _gw_hold) = plan.grid_w.device_ptr(stream);
        let q_rot = scratch.ptr + plan.off_q_rot;
        let k_rot = scratch.ptr + plan.off_k_rot;
        let k_pad = scratch.ptr + plan.off_k_pad;
        let rows = plan.rows as i32;
        let nblocks = plan.nblocks as i32;

        unsafe {
            stream
                .launch_builder(&plan.rope)
                .arg(&b.q_in.ptr)
                .arg(&b.k_in.ptr)
                .arg(&q_rot)
                .arg(&k_rot)
                .arg(&cu_ptr)
                .arg(&gh_ptr)
                .arg(&gw_ptr)
                .arg(&nblocks)
                .arg(&rows)
                .launch(plan.rope_cfg)?;
        }

        for &(src, dst, bytes) in &plan.copies {
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    (k_pad + dst) as _,
                    (k_rot + src) as _,
                    bytes as usize,
                    stream.cu_stream() as _,
                )?;
            }
        }

        unsafe {
            stream
                .launch_builder(&plan.attn)
                .arg(&q_rot)
                .arg(&k_pad)
                .arg(&b.v.ptr)
                .arg(&b.o.0.ptr)
                .arg(&cu_ptr)
                .arg(&kb_ptr)
                .arg(&nblocks)
                .launch(plan.attn_cfg)?;
        }
        Ok(())
    }
}

impl VisionMergerOps {
    /// CPU-only checked sizing of the LayerNorm output and the fc1 activation.
    pub(crate) fn workspace(spec: &VisionSpec, rows: u32) -> Result<WorkspaceReq> {
        spec.validate()?;
        ensure!(rows > 0, "merger needs at least one patch row");
        ensure!(
            rows.is_multiple_of(MERGE * MERGE),
            "merger rows must be whole 2x2 blocks"
        );
        let (_, _, bytes) = merger_layout(rows)?;
        Ok(WorkspaceReq {
            bytes,
            alignment: WS_ALIGN,
        })
    }

    /// Setup binder: validates the merger layout and binds the two biased linear plans in the shared [`Blas`].
    pub(crate) fn prepare(
        spec: &VisionSpec,
        bindings: VisionMergerBindings,
        common: CommonOpsRef,
        blas: &mut Blas,
    ) -> Result<VisionMergerPlan> {
        spec.validate()?;
        let rows = bindings.x.shape[0];
        let tokens = rows / (MERGE * MERGE);
        ensure!(
            rows > 0 && rows.is_multiple_of(MERGE * MERGE) && tokens > 0,
            "merger rows must be whole 2x2 blocks"
        );
        want(&bindings.x, [rows, HIDDEN], "x")?;
        dense(&bindings.x, "x")?;
        for (view, name) in [(&bindings.norm_w, "norm_w"), (&bindings.norm_b, "norm_b")] {
            want(view, [HIDDEN], name)?;
        }
        want(&bindings.fc1_w, [MERGER_FC1, MERGER_FC1], "fc1_w")?;
        want(&bindings.fc1_b, [MERGER_FC1], "fc1_b")?;
        want(&bindings.fc2_w, [MERGER_FC2, MERGER_FC1], "fc2_w")?;
        want(&bindings.fc2_b, [MERGER_FC2], "fc2_b")?;
        want(&bindings.out.read(), [tokens, MERGER_FC2], "out")?;
        dense(&bindings.fc1_w, "fc1_w")?;
        dense(&bindings.fc2_w, "fc2_w")?;

        let (off_ln, off_fc1, scratch_bytes) = merger_layout(rows)?;

        blas.prepare_biased(tokens, MERGER_FC1, MERGER_FC1, bindings.fc1_b, BIAS_MODE)?;
        blas.prepare_biased(tokens, MERGER_FC2, MERGER_FC1, bindings.fc2_b, BIAS_MODE)?;
        Ok(VisionMergerPlan {
            bindings,
            common,
            rows,
            tokens,
            off_ln,
            off_fc1,
            scratch_bytes,
        })
    }

    /// Allocation-free, same-stream forward: per-patch LayerNorm, then each 2×2 block as one 4608-wide row.
    pub(crate) fn run(
        plan: &VisionMergerPlan,
        stream: &Arc<CudaStream>,
        blas: &Blas,
        scratch: Scratch,
    ) -> Result<()> {
        scratch.validate(plan.scratch_bytes)?;
        let b = &plan.bindings;
        let elem = std::mem::size_of::<bf16>() as u64;
        let ln_row = u64::from(HIDDEN);
        let fc1_row = u64::from(MERGER_FC1);
        let ln_span = u64::from(plan.rows) * ln_row * elem;

        // The [rows, 1, 1152] LayerNorm output is the [tokens, 4608] fc1 input in place.
        let x3 = unsafe {
            Tensor::new(
                b.x.ptr,
                [plan.rows, 1, HIDDEN],
                [ln_row, ln_row, 1],
                b.x.span_bytes,
            )?
        };
        let mut ln3 = unsafe {
            TensorMut::new(
                scratch.ptr + plan.off_ln,
                [plan.rows, 1, HIDDEN],
                [ln_row, ln_row, 1],
                ln_span,
            )?
        };
        plan.common
            .norm
            .layer_norm(stream, x3, b.norm_w, b.norm_b, &mut ln3, LN_EPS)?;

        let ln2 = unsafe {
            Tensor::new(
                scratch.ptr + plan.off_ln,
                [plan.tokens, MERGER_FC1],
                [fc1_row, 1],
                ln_span,
            )?
        };
        let mut fc1 = unsafe {
            TensorMut::new(
                scratch.ptr + plan.off_fc1,
                [plan.tokens, MERGER_FC1],
                [fc1_row, 1],
                ln_span,
            )?
        };
        blas.linear_biased(b.fc1_w, b.fc1_b, ln2, &mut fc1, BIAS_MODE)?;
        let mut fc1_3 = unsafe {
            TensorMut::new(
                scratch.ptr + plan.off_fc1,
                [plan.tokens, 1, MERGER_FC1],
                [fc1_row, fc1_row, 1],
                ln_span,
            )?
        };
        plan.common
            .gate
            .gelu(stream, fc1_3.read(), &mut fc1_3, GeluMode::Exact)?;
        let mut out = TensorMut(b.out.read());
        blas.linear_biased(b.fc2_w, b.fc2_b, fc1.read(), &mut out, BIAS_MODE)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_pads_each_image_to_the_key_tile() {
        let packing = PatchPacking::build(&[
            PatchGrid { h: 64, w: 64 },
            PatchGrid { h: 10, w: 28 },
            PatchGrid { h: 16, w: 16 },
        ])
        .unwrap();
        // 4096 is tile-aligned; 280 pads to 320; 256 is tile-aligned.
        assert_eq!(packing.real_rows(), 4632);
        assert_eq!(packing.padded_rows(), 4672);
        assert_eq!(packing.images(), 3);
        assert_eq!(packing.grids().len(), 3);
        assert_eq!(packing.cu_seqlens(), [0, 4096, 4376, 4632]);
        assert_eq!(packing.k_bases(), [0, 4096, 4416, 4672]);
        for base in packing.k_bases() {
            assert_eq!(base % KT as i32, 0, "key base must be tile-aligned");
        }
    }

    #[test]
    fn packing_rejects_empty_or_unaligned_grids() {
        assert!(PatchPacking::build(&[]).is_err());
        assert!(PatchPacking::build(&[PatchGrid { h: 0, w: 4 }]).is_err());
        assert!(PatchPacking::build(&[PatchGrid { h: 3, w: 4 }]).is_err());
        assert!(PatchPacking::build(&[PatchGrid { h: 4, w: 5 }]).is_err());
    }

    #[test]
    fn merger_workspace_is_two_patch_row_buffers() {
        let req = VisionMergerOps::workspace(&VisionSpec::q38(), 4096).unwrap();
        assert_eq!(req.bytes, 2 * 4096 * 1152 * 2);
        assert!(VisionMergerOps::workspace(&VisionSpec::q38(), 4095).is_err());
    }
}

#[cfg(test)]
mod probe {
    //! GPU probe of `workspace` -> `prepare` -> `run` for the attention stage and the merger, with negative controls.
    use super::*;
    use crate::backend::cuda::Device;
    use crate::backend::cuda::closure::Closure;
    use crate::backend::cuda::ops::Ops;
    use cudarc::driver::{CudaSlice, DevicePtr};
    use std::rc::Rc;

    const BF16_ULP: f32 = 1.0 / 256.0;
    const BOUND: f32 = 8.0 * BF16_ULP;
    /// From `kernels/vision_rope.cu` / `kernels/vision_attn.cu` (private there).
    const THETA: f32 = 1.0e4;
    const SCALE: f32 = 0.11785113; // fp32 nearest to 72**-0.5, same bits as the kernel

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f64 / 9007199254740992.0 * 2.0 - 1.0) as f32
        }
    }

    fn view<T: cudarc::driver::DeviceRepr + Copy, const R: usize>(
        buffer: &CudaSlice<T>,
        stream: &Arc<CudaStream>,
        shape: [u32; R],
        stride: [u64; R],
    ) -> Result<Tensor<T, R>> {
        let (ptr, _hold) = buffer.device_ptr(stream);
        unsafe {
            Tensor::new(
                ptr,
                shape,
                stride,
                (buffer.len() * std::mem::size_of::<T>()) as u64,
            )
        }
    }

    /// One (row, head) of 72 dims: pairs (i, i+36), 18 frequencies per axis,
    /// row axis for i < 18 and col axis above, merge-grouped positions.
    fn rope_row(x: &mut [f32], pos_row: f32, pos_col: f32) {
        let half = (D / 2) as usize;
        let nfreq = half / 2;
        let (a, b): (Vec<f32>, Vec<f32>) = (x[..half].to_vec(), x[half..].to_vec());
        for i in 0..half {
            let pos = if i < nfreq { pos_row } else { pos_col };
            let inv = 1.0 / THETA.powf((i % nfreq) as f32 / nfreq as f32);
            let (sin, cos) = (pos * inv).sin_cos();
            x[i] = bf16::from_f32(a[i] * cos - b[i] * sin).to_f32();
            x[i + half] = bf16::from_f32(b[i] * cos + a[i] * sin).to_f32();
        }
    }

    /// Patch p's (row, col) inside grid (h, w) under the merge-grouped order.
    fn patch_pos(p: usize, _h: usize, w: usize) -> (f32, f32) {
        let nw = w / MERGE as usize;
        let m = MERGE as usize;
        let blk = p / (m * m);
        let inb = p % (m * m);
        (
            ((blk / nw) * m + inb / m) as f32,
            ((blk % nw) * m + inb % m) as f32,
        )
    }

    /// FP32 attention reference with bf16 casts after rotation, on P before PV and on the output; non-causal per image.
    fn reference_attn(
        grids: &[PatchGrid],
        q_in: &[bf16],
        k_in: &[bf16],
        v_pad: &[bf16],
        kb: &[i32],
    ) -> Vec<f32> {
        let (hq, d) = (HQ as usize, D as usize);
        let rows: usize = grids.iter().map(|g| (g.h * g.w) as usize).sum();
        let krows: usize = *kb.last().unwrap() as usize;
        let mut out = vec![0f32; rows * hq * d];
        let mut cu = 0usize;
        for (b, g) in grids.iter().enumerate() {
            let n = (g.h * g.w) as usize;
            let kbase = kb[b] as usize;
            for h in 0..hq {
                for r in 0..n {
                    let (pr, pc) = patch_pos(r, g.h as usize, g.w as usize);
                    let mut q = vec![0f32; d];
                    let qsrc = (h * rows + cu + r) * d;
                    for j in 0..d {
                        q[j] = q_in[qsrc + j].to_f32();
                    }
                    rope_row(&mut q, pr, pc);
                    let mut scores = vec![0f32; n];
                    for (kk, score) in scores.iter_mut().enumerate() {
                        let (kr, kc) = patch_pos(kk, g.h as usize, g.w as usize);
                        let mut k = vec![0f32; d];
                        let ksrc = (h * rows + cu + kk) * d;
                        for j in 0..d {
                            k[j] = k_in[ksrc + j].to_f32();
                        }
                        rope_row(&mut k, kr, kc);
                        *score = q.iter().zip(&k).map(|(a, b)| a * b).sum::<f32>() * SCALE;
                    }
                    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let p: Vec<f32> = scores.iter().map(|v| (v - max).exp()).collect();
                    let den: f32 = p.iter().sum();
                    let mut num = vec![0f32; d];
                    for kk in 0..n {
                        let pb = bf16::from_f32(p[kk]).to_f32();
                        for j in 0..d {
                            num[j] = pb
                                .mul_add(v_pad[(h * krows + kbase + kk) * d + j].to_f32(), num[j]);
                        }
                    }
                    let dst = (cu + r) * hq * d + h * d;
                    for j in 0..d {
                        out[dst + j] = bf16::from_f32(num[j] / den).to_f32();
                    }
                }
            }
            cu += n;
        }
        out
    }

    /// Abramowitz & Stegun 7.1.26, |err| <= 1.5e-7.
    fn erf(x: f32) -> f32 {
        let sign = if x < 0.0 { -1.0f32 } else { 1.0f32 };
        let x = x.abs();
        let t = 1.0 / (1.0 + 0.3275911 * x);
        let y = 1.0
            - ((((1.0614054 * t - 1.4531521) * t + 1.4214138) * t - 0.28449672) * t + 0.2548296)
                * t
                * (-x * x).exp();
        sign * y
    }

    struct MergerWeights<'a> {
        nw: &'a [bf16],
        nb: &'a [bf16],
        w1: &'a [bf16],
        b1: &'a [bf16],
        w2: &'a [bf16],
        b2: &'a [bf16],
    }

    /// FP32 merger reference; LayerNorm, each biased linear (bias before rounding) and exact GELU round once to bf16.
    fn reference_merger(
        x: &[bf16],
        rows: usize,
        w: &MergerWeights<'_>,
        postshuffle: bool,
    ) -> Vec<f32> {
        let (hidden, fc1, fc2) = (HIDDEN as usize, MERGER_FC1 as usize, MERGER_FC2 as usize);
        let tokens = rows / (MERGE * MERGE) as usize;
        let mut ln = vec![0f32; rows * hidden];
        if postshuffle {
            // `use_postshuffle_norm = True`: statistics over the whole 2x2 block (4608).
            for t in 0..tokens {
                let xr: Vec<f32> = x[t * fc1..(t + 1) * fc1]
                    .iter()
                    .map(|v| v.to_f32())
                    .collect();
                let mean = xr.iter().sum::<f32>() / fc1 as f32;
                let var = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / fc1 as f32;
                let inv = 1.0 / (var + LN_EPS).sqrt();
                for j in 0..fc1 {
                    ln[t * fc1 + j] = bf16::from_f32(
                        (xr[j] - mean) * inv * w.nw[j % hidden].to_f32()
                            + w.nb[j % hidden].to_f32(),
                    )
                    .to_f32();
                }
            }
        } else {
            for r in 0..rows {
                let xr: Vec<f32> = x[r * hidden..(r + 1) * hidden]
                    .iter()
                    .map(|v| v.to_f32())
                    .collect();
                let mean = xr.iter().sum::<f32>() / hidden as f32;
                let var = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hidden as f32;
                let inv = 1.0 / (var + LN_EPS).sqrt();
                for j in 0..hidden {
                    ln[r * hidden + j] =
                        bf16::from_f32((xr[j] - mean) * inv * w.nw[j].to_f32() + w.nb[j].to_f32())
                            .to_f32();
                }
            }
        }
        let mut fc1_out = vec![0f32; tokens * fc1];
        for t in 0..tokens {
            for o in 0..fc1 {
                let mut acc = w.b1[o].to_f32();
                for i in 0..fc1 {
                    acc += w.w1[o * fc1 + i].to_f32() * ln[t * fc1 + i];
                }
                fc1_out[t * fc1 + o] = bf16::from_f32(acc).to_f32();
            }
        }
        for v in fc1_out.iter_mut() {
            let x = *v;
            *v = bf16::from_f32(0.5 * x * (1.0 + erf(x / 2f32.sqrt()))).to_f32();
        }
        let mut out = vec![0f32; tokens * fc2];
        for t in 0..tokens {
            for o in 0..fc2 {
                let mut acc = w.b2[o].to_f32();
                for i in 0..fc1 {
                    acc += w.w2[o * fc1 + i].to_f32() * fc1_out[t * fc1 + i];
                }
                out[t * fc2 + o] = bf16::from_f32(acc).to_f32();
            }
        }
        out
    }

    fn max_rel(got: &[f32], reference: &[f32]) -> (f32, f32) {
        let peak = reference
            .iter()
            .map(|v| v.abs())
            .fold(0f32, f32::max)
            .max(1e-12);
        let err = got
            .iter()
            .zip(reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        (err, err / peak)
    }

    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_Q38_GPU_WINDOW=granted"]
    fn vision_binder_probe() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let device = Device::new(0)?;
        let s = &device.stream;
        let (hq, d) = (HQ as usize, D as usize);
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);

        // Attention stage: two images, grids 4x4 and 2x6, one tail tile.
        let grids = [PatchGrid { h: 4, w: 4 }, PatchGrid { h: 2, w: 6 }];
        let packing = PatchPacking::build(&grids)?;
        let kb: Vec<i32> = vec![0, 64, 128];
        let rows = 28usize;
        let krows = 128usize;
        assert_eq!(packing.cu_seqlens(), [0, 16, 28]);
        assert_eq!(packing.k_bases(), kb.as_slice());
        assert_eq!(packing.real_rows(), rows as u32);
        assert_eq!(packing.padded_rows(), krows as u32);
        let mut q_in = vec![bf16::ZERO; hq * rows * d];
        let mut k_in = vec![bf16::ZERO; hq * rows * d];
        for v in q_in.iter_mut().chain(k_in.iter_mut()) {
            *v = bf16::from_f32(rng.next());
        }
        let mut v_pad = vec![bf16::ZERO; hq * krows * d];
        for h in 0..hq {
            for (b, g) in grids.iter().enumerate() {
                let n = (g.h * g.w) as usize;
                for r in 0..n {
                    for j in 0..d {
                        v_pad[(h * krows + kb[b] as usize + r) * d + j] =
                            bf16::from_f32(0.9 * rng.next());
                    }
                }
            }
        }
        let dev_q = s.clone_htod(&q_in)?;
        let dev_k = s.clone_htod(&k_in)?;
        let dev_v = s.clone_htod(&v_pad)?;
        let dev_o = s.alloc_zeros::<bf16>(rows * hq * d)?;
        let bindings = VisionAttnBindings {
            q_in: view(
                &dev_q,
                s,
                [HQ, rows as u32, D],
                [rows as u64 * d as u64, d as u64, 1],
            )?,
            k_in: view(
                &dev_k,
                s,
                [HQ, rows as u32, D],
                [rows as u64 * d as u64, d as u64, 1],
            )?,
            v: view(
                &dev_v,
                s,
                [HQ, krows as u32, D],
                [krows as u64 * d as u64, d as u64, 1],
            )?,
            o: TensorMut(view(
                &dev_o,
                s,
                [rows as u32, HQ, D],
                [hq as u64 * d as u64, d as u64, 1],
            )?),
        };
        let plan = VisionAttnOps::prepare(&device, &VisionSpec::q38(), &packing, bindings)?;
        let req = VisionAttnOps::workspace(&VisionSpec::q38(), &packing)?;
        let scratch_buf = s.alloc_zeros::<u8>(req.bytes as usize)?;
        let (sp, _hold) = scratch_buf.device_ptr(s);
        VisionAttnOps::run(
            &plan,
            s,
            Scratch {
                ptr: sp,
                bytes: req.bytes,
            },
        )?;
        s.synchronize()?;
        let got: Vec<f32> = s.clone_dtoh(&dev_o)?.iter().map(|v| v.to_f32()).collect();
        let reference = reference_attn(&grids, &q_in, &k_in, &v_pad, &kb);
        let (err, rel) = max_rel(&got, &reference);
        eprintln!("Q38_VISION_ATTN_BINDER max|d|={err:.3e} rel={rel:.3e} bound={BOUND:.3e}");
        ensure!(
            rel <= BOUND,
            "attention binder output differs from the fp32 reference: rel={rel:.3e} bound={BOUND:.3e}"
        );

        // Negative control: Q rows swapped between the two images must exceed the bound.
        let mut q_swapped = q_in.clone();
        for h in 0..hq {
            for r in 0..4 {
                let a = (h * rows + r) * d;
                let b = (h * rows + 16 + r) * d;
                for j in 0..d {
                    q_swapped.swap(a + j, b + j);
                }
            }
        }
        let dev_q2 = s.clone_htod(&q_swapped)?;
        ensure!(
            dev_q2.len() == dev_q.len(),
            "internal control buffer mismatch"
        );
        let bindings2 = VisionAttnBindings {
            q_in: view(
                &dev_q2,
                s,
                [HQ, rows as u32, D],
                [rows as u64 * d as u64, d as u64, 1],
            )?,
            k_in: view(
                &dev_k,
                s,
                [HQ, rows as u32, D],
                [rows as u64 * d as u64, d as u64, 1],
            )?,
            v: view(
                &dev_v,
                s,
                [HQ, krows as u32, D],
                [krows as u64 * d as u64, d as u64, 1],
            )?,
            o: TensorMut(view(
                &dev_o,
                s,
                [rows as u32, HQ, D],
                [hq as u64 * d as u64, d as u64, 1],
            )?),
        };
        let plan2 = VisionAttnOps::prepare(&device, &VisionSpec::q38(), &packing, bindings2)?;
        VisionAttnOps::run(
            &plan2,
            s,
            Scratch {
                ptr: sp,
                bytes: req.bytes,
            },
        )?;
        s.synchronize()?;
        let got2: Vec<f32> = s.clone_dtoh(&dev_o)?.iter().map(|v| v.to_f32()).collect();
        let (_, ctl_rel) = max_rel(&got2, &reference);
        eprintln!("Q38_VISION_ATTN_CONTROL rel={ctl_rel:.3e} (must exceed {BOUND:.3e})");
        ensure!(
            ctl_rel > BOUND,
            "negative control was not rejected: rel={ctl_rel:.3e} bound={BOUND:.3e}"
        );

        // Merger stage: 8 rows, two merged tokens.
        let (mrows, hidden, fc1, fc2) = (
            8usize,
            HIDDEN as usize,
            MERGER_FC1 as usize,
            MERGER_FC2 as usize,
        );
        let tokens = 2usize;
        // Distinct per-row scales separate per-patch from block-wise LayerNorm statistics.
        let x: Vec<bf16> = (0..mrows * hidden)
            .map(|i| bf16::from_f32((1.0 + 0.35 * (i / hidden) as f32) * rng.next()))
            .collect();
        let nw: Vec<bf16> = (0..hidden)
            .map(|_| bf16::from_f32(1.0 + 0.05 * rng.next()))
            .collect();
        let nb: Vec<bf16> = (0..hidden)
            .map(|_| bf16::from_f32(0.05 * rng.next()))
            .collect();
        let w1: Vec<bf16> = (0..fc1 * fc1)
            .map(|_| bf16::from_f32(0.02 * rng.next()))
            .collect();
        let b1: Vec<bf16> = (0..fc1)
            .map(|_| bf16::from_f32(0.05 * rng.next()))
            .collect();
        let w2: Vec<bf16> = (0..fc2 * fc1)
            .map(|_| bf16::from_f32(0.02 * rng.next()))
            .collect();
        let b2: Vec<bf16> = (0..fc2)
            .map(|_| bf16::from_f32(0.05 * rng.next()))
            .collect();
        let dev_x = s.clone_htod(&x)?;
        let dev_nw = s.clone_htod(&nw)?;
        let dev_nb = s.clone_htod(&nb)?;
        let dev_w1 = s.clone_htod(&w1)?;
        let dev_b1 = s.clone_htod(&b1)?;
        let dev_w2 = s.clone_htod(&w2)?;
        let dev_b2 = s.clone_htod(&b2)?;
        let dev_out = s.alloc_zeros::<bf16>(tokens * fc2)?;
        let mk = |rows_: usize, w: usize| [rows_ as u32, w as u32];
        let merger_bindings = VisionMergerBindings {
            x: view(&dev_x, s, mk(mrows, hidden), [hidden as u64, 1])?,
            norm_w: view(&dev_nw, s, [HIDDEN], [1])?,
            norm_b: view(&dev_nb, s, [HIDDEN], [1])?,
            fc1_w: view(&dev_w1, s, mk(fc1, fc1), [fc1 as u64, 1])?,
            fc1_b: view(&dev_b1, s, [MERGER_FC1], [1])?,
            fc2_w: view(&dev_w2, s, mk(fc2, fc1), [fc1 as u64, 1])?,
            fc2_b: view(&dev_b2, s, [MERGER_FC2], [1])?,
            out: TensorMut(view(&dev_out, s, mk(tokens, fc2), [fc2 as u64, 1])?),
        };
        let common = CommonOpsRef::new(&device.ctx)?;
        let ops = Ops::new(&device.ctx)?;
        let closure = Rc::new(Closure::new(false, "vision binder probe".into()));
        let mut blas = Blas::new(&device, &ops, &std::env::temp_dir(), closure)?;
        let mplan =
            VisionMergerOps::prepare(&VisionSpec::q38(), merger_bindings, common, &mut blas)?;
        let mreq = VisionMergerOps::workspace(&VisionSpec::q38(), mrows as u32)?;
        let mscratch = s.alloc_zeros::<u8>(mreq.bytes as usize)?;
        let (mp, _mhold) = mscratch.device_ptr(s);
        VisionMergerOps::run(
            &mplan,
            s,
            &blas,
            Scratch {
                ptr: mp,
                bytes: mreq.bytes,
            },
        )?;
        s.synchronize()?;
        let mgot: Vec<f32> = s.clone_dtoh(&dev_out)?.iter().map(|v| v.to_f32()).collect();
        let mw = MergerWeights {
            nw: &nw,
            nb: &nb,
            w1: &w1,
            b1: &b1,
            w2: &w2,
            b2: &b2,
        };
        let mref = reference_merger(&x, mrows, &mw, false);
        let (merr, mrel) = max_rel(&mgot, &mref);
        eprintln!("Q38_VISION_MERGER_BINDER max|d|={merr:.3e} rel={mrel:.3e} bound={BOUND:.3e}");
        ensure!(
            mrel <= BOUND,
            "merger binder output differs from the fp32 reference: rel={mrel:.3e} bound={BOUND:.3e}"
        );

        // Negative control: block-wise (post-shuffle) LayerNorm statistics must exceed the bound.
        let mctl = reference_merger(&x, mrows, &mw, true);
        let (_, mctl_rel) = max_rel(&mgot, &mctl);
        eprintln!("Q38_VISION_MERGER_CONTROL rel={mctl_rel:.3e} (must exceed {BOUND:.3e})");
        ensure!(
            mctl_rel > BOUND,
            "post-shuffle-norm negative control was not rejected: rel={mctl_rel:.3e} bound={BOUND:.3e}"
        );

        eprintln!("Q38_VISION_BINDER_PROBE PASS");
        Ok(())
    }
}
