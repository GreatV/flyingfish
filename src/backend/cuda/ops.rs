use super::cubin;
use anyhow::Result;
use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig};
use std::sync::Arc;

pub struct Ops {
    pub embed: CudaFunction,
    pub embed_rmsnorm: CudaFunction,
    pub norm: CudaFunction,
    pub add_norm: CudaFunction,
    pub add_norm_capture: CudaFunction,
    pub rope_float: CudaFunction,
    pub kv_write: CudaFunction,
    pub rope_kv: CudaFunction,
    pub act: CudaFunction,
    pub argmax: CudaFunction,
    pub argmax_rows: CudaFunction,
    pub finite_bf16: CudaFunction,
    pub advance: CudaFunction,
    pub residual: CudaFunction,
}

impl Ops {
    pub fn finite(
        &self,
        stream: &Arc<CudaStream>,
        values: Tensor<bf16, 2>,
        flag: Tensor<u32, 1>,
    ) -> Result<()> {
        values.validate()?;
        flag.validate()?;
        ensure!(
            values.stride == [u64::from(values.shape[1]), 1] && flag.stride == [1],
            "finite check requires contiguous values and one status word"
        );
        let count = u64::from(values.shape[0]) * u64::from(values.shape[1]);
        let flag_ptr = flag.element_ptr(flag.shape[0] as usize - 1)?;
        unsafe {
            stream
                .launch_builder(&self.finite_bf16)
                .arg(&values.ptr)
                .arg(&flag_ptr)
                .arg(&count)
                .launch(flat(usize::try_from(count)?))?;
        }
        Ok(())
    }
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let embed = cubin::module(ctx, "embed")?;
        let embed_rmsnorm = cubin::module(ctx, "embed_rmsnorm")?;
        let norm = cubin::module(ctx, "norm")?;
        let add_norm_capture = cubin::module(ctx, "add_rmsnorm_capture")?;
        let rope = cubin::module(ctx, "rope")?;
        let rope_kv = cubin::module(ctx, "rope_kv")?;
        let act = cubin::module(ctx, "act")?;
        let sample = cubin::module(ctx, "sample")?;
        Ok(Self {
            embed: embed.load_function("embed")?,
            embed_rmsnorm: embed_rmsnorm.load_function("embed_rmsnorm")?,
            norm: norm.load_function("rms_norm")?,
            add_norm: norm.load_function("add_rmsnorm")?,
            add_norm_capture: add_norm_capture.load_function("add_rmsnorm_capture")?,
            rope_float: rope.load_function("rope_float")?,
            kv_write: rope.load_function("kv_write")?,
            rope_kv: rope_kv.load_function("rope_kv")?,
            act: act.load_function("silu_mul")?,
            argmax: sample.load_function("argmax")?,
            argmax_rows: sample.load_function("argmax_rows_bf16")?,
            finite_bf16: sample.load_function("finite_bf16")?,
            advance: sample.load_function("advance")?,
            residual: norm.load_function("residual")?,
        })
    }
}

pub fn grid(blocks: usize, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub fn flat(n: usize) -> LaunchConfig {
    grid(n.div_ceil(256), 256)
}

use super::contracts::{RowDomain, Tensor, TensorMut};
use anyhow::ensure;
use cudarc::driver::{CudaStream, PushKernelArg};
use half::bf16;

#[derive(Clone, Copy, Debug)]
#[repr(i32)]
pub(crate) enum RmsMode {
    Standard = 0,
    ZeroCentered = 1,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct RmsOptions {
    pub mode: RmsMode,
    pub eps: f32,
}
pub(crate) struct GdnGateBindings<'a> {
    pub a: Tensor<bf16, 2>,
    pub b: Tensor<bf16, 2>,
    pub a_log: Tensor<bf16, 1>,
    pub dt_bias: Tensor<bf16, 1>,
    pub beta: &'a mut TensorMut<bf16, 2>,
    pub g: &'a mut TensorMut<f32, 2>,
}
pub(crate) struct NormOps {
    norm: CudaFunction,
    layer_norm: CudaFunction,
}
pub(crate) struct GateOps {
    gate: CudaFunction,
    gdn: CudaFunction,
    gelu: CudaFunction,
}
#[derive(Clone)]
pub(crate) struct CommonOpsRef {
    pub norm: Arc<NormOps>,
    pub gate: Arc<GateOps>,
}
impl CommonOpsRef {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let module = cubin::module(ctx, "q38_common")?;
        Ok(Self {
            norm: Arc::new(NormOps {
                norm: module.load_function("q38_norm")?,
                layer_norm: module.load_function("q38_layer_norm")?,
            }),
            gate: Arc::new(GateOps {
                gate: module.load_function("q38_gate")?,
                gdn: module.load_function("q38_gdn_gates")?,
                gelu: module.load_function("q38_gelu")?,
            }),
        })
    }
}
fn no_overlap<T: Copy, U: Copy, const R: usize, const S: usize>(
    a: &Tensor<T, R>,
    b: &Tensor<U, S>,
) -> Result<()> {
    ensure!(
        a.ptr + a.span_bytes <= b.ptr || b.ptr + b.span_bytes <= a.ptr,
        "operator tensors overlap"
    );
    Ok(())
}
fn validate_io(x: &Tensor<bf16, 3>, y: &Tensor<bf16, 3>, rows: RowDomain) -> Result<()> {
    x.validate()?;
    y.validate()?;
    rows.validate(x.shape[0])?;
    ensure!(x.shape == y.shape, "operator output shape mismatch");
    // Exact in-place aliases are supported; partial aliases are not.
    if x.ptr == y.ptr {
        ensure!(
            x.stride == y.stride && x.span_bytes == y.span_bytes,
            "partial in-place alias"
        );
    } else {
        no_overlap(x, y)?;
    }
    for view in [x, y] {
        ensure!(
            view.stride[2] == 1
                && view.stride[1] >= u64::from(view.shape[2])
                && view.stride[0] >= u64::from(view.shape[1]) * view.stride[1],
            "overlapping/unsupported tensor strides"
        );
    }
    Ok(())
}
struct NormLaunch<'a> {
    rows: RowDomain,
    x: Tensor<bf16, 3>,
    weight: Option<Tensor<bf16, 1>>,
    z: Option<Tensor<bf16, 3>>,
    y: &'a mut TensorMut<bf16, 3>,
    mode: Option<RmsMode>,
    eps: f32,
}

pub(crate) struct GatedRmsInput {
    pub x: Tensor<bf16, 3>,
    pub weight: Tensor<bf16, 1>,
    pub gate: Tensor<bf16, 3>,
}

pub(crate) struct ResidualRmsInput<'a> {
    pub x: &'a mut TensorMut<bf16, 3>,
    pub residual: Tensor<bf16, 3>,
    pub weight: Tensor<bf16, 1>,
}

fn norm_slices_per_block(
    row_capacity: u32,
    x: &Tensor<bf16, 3>,
    wp: u64,
    zp: u64,
    zs: [u64; 3],
    y: &Tensor<bf16, 3>,
    mode: i32,
) -> u32 {
    let heads = x.shape[1];
    let width = x.shape[2];
    if x.stride[2] != 1 || y.stride[2] != 1 || (mode >= 3 && zs[2] != 1) {
        return 1;
    }
    let ptrs = x.ptr | y.ptr | wp | if mode >= 3 { zp } else { 0 };
    let strides = x.stride[0]
        | x.stride[1]
        | y.stride[0]
        | y.stride[1]
        | if mode >= 3 { zs[0] | zs[1] } else { 0 };
    if width == 128 && (ptrs & 7) == 0 && (strides & 3) == 0 {
        return 8;
    }
    if width == 256 && (ptrs & 15) == 0 && (strides & 7) == 0 {
        return 8;
    }
    if width == 5120
        && heads == 1
        && row_capacity > 16
        && (ptrs & 7) == 0
        && ((x.stride[0] | y.stride[0] | if mode >= 3 { zs[0] } else { 0 }) & 3) == 0
    {
        return 4;
    }
    1
}

impl NormOps {
    pub fn rms_with(
        &self,
        stream: &Arc<CudaStream>,
        rows: RowDomain,
        x: Tensor<bf16, 3>,
        weight: Tensor<bf16, 1>,
        y: &mut TensorMut<bf16, 3>,
        options: RmsOptions,
    ) -> Result<()> {
        let RmsOptions { mode, eps } = options;
        self.launch(
            stream,
            NormLaunch {
                rows,
                x,
                weight: Some(weight),
                z: None,
                y,
                mode: Some(mode),
                eps,
            },
        )
    }
    pub fn rms_add_with(
        &self,
        stream: &Arc<CudaStream>,
        rows: RowDomain,
        input: ResidualRmsInput<'_>,
        y: &mut TensorMut<bf16, 3>,
        options: RmsOptions,
    ) -> Result<()> {
        let ResidualRmsInput {
            x,
            residual: z,
            weight,
        } = input;
        let RmsOptions { mode, eps } = options;
        ensure!(
            matches!(mode, RmsMode::ZeroCentered),
            "residual norm requires zero-centered RMS semantics"
        );
        no_overlap(&x.0, &y.0)?;
        no_overlap(&z, &x.0)?;
        no_overlap(&weight, &x.0)?;
        self.launch(
            stream,
            NormLaunch {
                rows,
                x: x.0,
                weight: Some(weight),
                z: Some(z),
                y,
                mode: Some(mode),
                eps,
            },
        )
    }
    pub fn l2_heads(
        &self,
        stream: &Arc<CudaStream>,
        rows: RowDomain,
        x: Tensor<bf16, 3>,
        y: &mut TensorMut<bf16, 3>,
        eps: f32,
    ) -> Result<()> {
        self.launch(
            stream,
            NormLaunch {
                rows,
                x,
                weight: None,
                z: None,
                y,
                mode: None,
                eps,
            },
        )
    }
    pub fn rms_gated(
        &self,
        stream: &Arc<CudaStream>,
        rows: RowDomain,
        input: GatedRmsInput,
        y: &mut TensorMut<bf16, 3>,
        eps: f32,
    ) -> Result<()> {
        let GatedRmsInput { x, weight, gate: z } = input;
        self.launch(
            stream,
            NormLaunch {
                rows,
                x,
                weight: Some(weight),
                z: Some(z),
                y,
                mode: Some(RmsMode::Standard),
                eps,
            },
        )
    }
    fn launch(&self, stream: &Arc<CudaStream>, call: NormLaunch<'_>) -> Result<()> {
        let NormLaunch {
            rows,
            x,
            weight,
            z,
            y,
            mode,
            eps,
        } = call;
        let mode = match (mode, z.is_some()) {
            (Some(RmsMode::Standard), true) => 3,
            (Some(RmsMode::ZeroCentered), true) => 4,
            (Some(mode), false) => mode as i32,
            (None, false) => 2,
            _ => anyhow::bail!("gated norm requires standard RMS semantics"),
        };
        validate_io(&x, &y.0, rows)?;
        ensure!(eps.is_finite() && eps > 0.0, "invalid norm epsilon");
        let mut wp = 0u64;
        if let Some(w) = weight {
            w.validate()?;
            ensure!(
                w.shape == [x.shape[2]] && w.stride == [1],
                "norm weight layout mismatch"
            );
            no_overlap(&w, &y.0)?;
            wp = w.ptr;
        }
        let (zp, zs) = if let Some(z) = z {
            z.validate()?;
            ensure!(z.shape == x.shape, "norm gate shape mismatch");
            ensure!(
                z.stride[2] == 1
                    && z.stride[1] >= u64::from(z.shape[2])
                    && z.stride[0] >= u64::from(z.shape[1]) * z.stride[1],
                "overlapping/unsupported tensor strides"
            );
            no_overlap(&z, &y.0)?;
            (z.ptr, z.stride)
        } else {
            (0, [0; 3])
        };
        let slices = rows
            .row_capacity
            .checked_mul(x.shape[1])
            .filter(|&n| n <= i32::MAX as u32)
            .ok_or_else(|| anyhow::anyhow!("norm grid overflow"))?;
        let spb = norm_slices_per_block(rows.row_capacity, &x, wp, zp, zs, &y.0, mode);
        let blocks = slices.div_ceil(spb);
        unsafe {
            stream
                .launch_builder(&self.norm)
                .arg(&x.ptr)
                .arg(&wp)
                .arg(&zp)
                .arg(&y.0.ptr)
                .arg(&rows.meta.ptr)
                .arg(&rows.row_base)
                .arg(&rows.row_capacity)
                .arg(&x.shape[1])
                .arg(&x.shape[2])
                .arg(&x.stride[0])
                .arg(&x.stride[1])
                .arg(&x.stride[2])
                .arg(&y.0.stride[0])
                .arg(&y.0.stride[1])
                .arg(&y.0.stride[2])
                .arg(&zs[0])
                .arg(&zs[1])
                .arg(&zs[2])
                .arg(&mode)
                .arg(&eps)
                .launch(grid(blocks as usize, 256))?;
        }
        Ok(())
    }
}
impl GateOps {
    #[cfg(test)]
    pub fn silu(
        &self,
        stream: &Arc<CudaStream>,
        rows: RowDomain,
        x: Tensor<bf16, 3>,
        y: &mut TensorMut<bf16, 3>,
    ) -> Result<()> {
        self.launch(stream, rows, x, None, y)
    }
    pub fn sigmoid_mul(
        &self,
        stream: &Arc<CudaStream>,
        rows: RowDomain,
        x: Tensor<bf16, 3>,
        gate: Tensor<bf16, 3>,
        y: &mut TensorMut<bf16, 3>,
    ) -> Result<()> {
        self.launch(stream, rows, x, Some(gate), y)
    }
    pub fn swiglu(
        &self,
        stream: &Arc<CudaStream>,
        rows: RowDomain,
        x: Tensor<bf16, 3>,
        up: Tensor<bf16, 3>,
        y: &mut TensorMut<bf16, 3>,
    ) -> Result<()> {
        validate_io(&x, &y.0, rows)?;
        up.validate()?;
        ensure!(up.shape == x.shape, "SwiGLU shape mismatch");
        no_overlap(&x, &up)?;
        no_overlap(&up, &y.0)?;
        let width = x.shape[2];
        ensure!(
            x.shape[1] == 1
                && width.is_multiple_of(8)
                && x.stride == [u64::from(width), u64::from(width), 1]
                && up.stride == x.stride
                && y.0.stride == x.stride,
            "SwiGLU requires aligned contiguous rows"
        );
        ensure!(
            x.ptr.is_multiple_of(16) && up.ptr.is_multiple_of(16) && y.0.ptr.is_multiple_of(16),
            "SwiGLU tensors must be 16-byte aligned"
        );
        let vectors = width / 8;
        let blocks = vectors.div_ceil(256);
        ensure!(
            blocks <= i32::MAX as u32 && rows.row_capacity <= u16::MAX as u32,
            "SwiGLU grid outside CUDA limits"
        );
        unsafe {
            stream
                .launch_builder(&self.gate)
                .arg(&x.ptr)
                .arg(&up.ptr)
                .arg(&y.0.ptr)
                .arg(&rows.meta.ptr)
                .arg(&rows.row_base)
                .arg(&rows.row_capacity)
                .arg(&x.shape[1])
                .arg(&width)
                .arg(&x.stride[0])
                .arg(&x.stride[1])
                .arg(&x.stride[2])
                .arg(&up.stride[0])
                .arg(&up.stride[1])
                .arg(&up.stride[2])
                .arg(&y.0.stride[0])
                .arg(&y.0.stride[1])
                .arg(&y.0.stride[2])
                .arg(&2i32)
                .launch(LaunchConfig {
                    grid_dim: (blocks, rows.row_capacity, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }
    fn launch(
        &self,
        stream: &Arc<CudaStream>,
        rows: RowDomain,
        x: Tensor<bf16, 3>,
        gate: Option<Tensor<bf16, 3>>,
        y: &mut TensorMut<bf16, 3>,
    ) -> Result<()> {
        validate_io(&x, &y.0, rows)?;
        let mode = i32::from(gate.is_some());
        let (gp, gs) = if let Some(g) = gate {
            g.validate()?;
            ensure!(g.shape == x.shape, "gate shape mismatch");
            no_overlap(&g, &y.0)?;
            (g.ptr, g.stride)
        } else {
            (0, [0; 3])
        };
        let count = u64::from(rows.row_capacity) * u64::from(x.shape[1]) * u64::from(x.shape[2]);
        unsafe {
            stream
                .launch_builder(&self.gate)
                .arg(&x.ptr)
                .arg(&gp)
                .arg(&y.0.ptr)
                .arg(&rows.meta.ptr)
                .arg(&rows.row_base)
                .arg(&rows.row_capacity)
                .arg(&x.shape[1])
                .arg(&x.shape[2])
                .arg(&x.stride[0])
                .arg(&x.stride[1])
                .arg(&x.stride[2])
                .arg(&gs[0])
                .arg(&gs[1])
                .arg(&gs[2])
                .arg(&y.0.stride[0])
                .arg(&y.0.stride[1])
                .arg(&y.0.stride[2])
                .arg(&mode)
                .launch(grid(count.div_ceil(256).min(65535) as usize, 256))?;
        }
        Ok(())
    }
    pub fn gdn_gates_with(
        &self,
        stream: &Arc<CudaStream>,
        rows: RowDomain,
        bindings: GdnGateBindings<'_>,
    ) -> Result<()> {
        let GdnGateBindings {
            a,
            b,
            a_log,
            dt_bias,
            beta,
            g,
        } = bindings;
        rows.validate(a.shape[0])?;
        for t in [&a, &b, &beta.0] {
            t.validate()?;
            ensure!(
                t.shape == a.shape && t.shape[1] == 48 && t.stride[1] == 1 && t.stride[0] >= 48,
                "GDN gate layout mismatch"
            );
        }
        for t in [&a_log, &dt_bias] {
            t.validate()?;
            ensure!(
                t.shape == [48] && t.stride == [1],
                "GDN gate weight mismatch"
            );
            no_overlap(t, &beta.0)?;
            no_overlap(t, &g.0)?;
        }
        g.0.validate()?;
        ensure!(
            g.0.shape == a.shape && g.0.stride[1] == 1 && g.0.stride[0] >= 48,
            "GDN g layout mismatch"
        );
        for t in [&a, &b] {
            no_overlap(t, &beta.0)?;
            no_overlap(t, &g.0)?;
        }
        no_overlap(&beta.0, &g.0)?;
        let blocks = (u64::from(rows.row_capacity) * 48).div_ceil(256).min(65535);
        unsafe {
            stream
                .launch_builder(&self.gdn)
                .arg(&a.ptr)
                .arg(&b.ptr)
                .arg(&a_log.ptr)
                .arg(&dt_bias.ptr)
                .arg(&beta.0.ptr)
                .arg(&g.0.ptr)
                .arg(&rows.meta.ptr)
                .arg(&rows.row_base)
                .arg(&rows.row_capacity)
                .arg(&a.stride[0])
                .arg(&b.stride[0])
                .arg(&beta.0.stride[0])
                .arg(&g.0.stride[0])
                .launch(grid(blocks as usize, 256))?;
        }
        Ok(())
    }
}

/// Stateless activation choice; callers include this in their plan identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub(crate) enum GeluMode {
    Exact = 0,
    Tanh = 1,
}
fn validate_dense_io(x: &Tensor<bf16, 3>, y: &Tensor<bf16, 3>) -> Result<()> {
    x.validate()?;
    y.validate()?;
    ensure!(x.shape == y.shape, "dense operator shape mismatch");
    if x.ptr == y.ptr {
        ensure!(
            x.stride == y.stride && x.span_bytes == y.span_bytes,
            "partial dense alias"
        );
    } else {
        no_overlap(x, y)?;
    }
    for t in [x, y] {
        ensure!(
            t.stride[2] == 1
                && t.stride[1] >= u64::from(t.shape[2])
                && t.stride[0] >= u64::from(t.shape[1]) * t.stride[1],
            "overlapping dense strides"
        );
    }
    Ok(())
}
impl NormOps {
    /// Ordinary affine LayerNorm; all x.shape[0] rows are valid (no text SeqRef).
    pub fn layer_norm(
        &self,
        stream: &Arc<CudaStream>,
        x: Tensor<bf16, 3>,
        weight: Tensor<bf16, 1>,
        bias: Tensor<bf16, 1>,
        y: &mut TensorMut<bf16, 3>,
        eps: f32,
    ) -> Result<()> {
        validate_dense_io(&x, &y.0)?;
        ensure!(eps.is_finite() && eps > 0.0, "invalid LayerNorm epsilon");
        for t in [weight, bias] {
            t.validate()?;
            ensure!(
                t.shape == [x.shape[2]] && t.stride == [1],
                "LayerNorm affine shape mismatch"
            );
            no_overlap(&t, &y.0)?;
        }
        let blocks = x.shape[0]
            .checked_mul(x.shape[1])
            .filter(|&n| n <= i32::MAX as u32)
            .ok_or_else(|| anyhow::anyhow!("LayerNorm grid overflow"))?;
        unsafe {
            stream
                .launch_builder(&self.layer_norm)
                .arg(&x.ptr)
                .arg(&weight.ptr)
                .arg(&bias.ptr)
                .arg(&y.0.ptr)
                .arg(&x.shape[0])
                .arg(&x.shape[1])
                .arg(&x.shape[2])
                .arg(&x.stride[0])
                .arg(&x.stride[1])
                .arg(&x.stride[2])
                .arg(&y.0.stride[0])
                .arg(&y.0.stride[1])
                .arg(&y.0.stride[2])
                .arg(&eps)
                .launch(grid(blocks as usize, 256))?;
        }
        Ok(())
    }
}
impl GateOps {
    /// BF16 GELU with an explicit erf/tanh choice; all declared rows are valid.
    pub fn gelu(
        &self,
        stream: &Arc<CudaStream>,
        x: Tensor<bf16, 3>,
        y: &mut TensorMut<bf16, 3>,
        mode: GeluMode,
    ) -> Result<()> {
        validate_dense_io(&x, &y.0)?;
        let count = u64::from(x.shape[0])
            .checked_mul(u64::from(x.shape[1]))
            .and_then(|n| n.checked_mul(u64::from(x.shape[2])))
            .ok_or_else(|| anyhow::anyhow!("GELU element count overflow"))?;
        unsafe {
            stream
                .launch_builder(&self.gelu)
                .arg(&x.ptr)
                .arg(&y.0.ptr)
                .arg(&x.shape[0])
                .arg(&x.shape[1])
                .arg(&x.shape[2])
                .arg(&x.stride[0])
                .arg(&x.stride[1])
                .arg(&x.stride[2])
                .arg(&y.0.stride[0])
                .arg(&y.0.stride[1])
                .arg(&y.0.stride[2])
                .arg(&(mode as i32))
                .launch(grid(count.div_ceil(256).min(65535) as usize, 256))?;
        }
        Ok(())
    }
}
