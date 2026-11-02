use super::{contracts::no_overlap, cubin};
use crate::backend::gpu::{CudaContext, CudaFunction, CudaSlice, CudaView, LaunchConfig};
use anyhow::{Context, Result};
use std::sync::Arc;

pub struct Ops {
    pub embed: CudaFunction,
    pub embed_rmsnorm: CudaFunction,
    pub add_norm: CudaFunction,
    add_norm_unit: CudaFunction,
    pub add_norm_capture: CudaFunction,
    pub add_norm_fp32: CudaFunction,
    norm_capture: CudaFunction,
    pub kv_write: CudaFunction,
    pub rope_kv: CudaFunction,
    rotary_prefix_kv: CudaFunction,
    pub rope_pair_rows: CudaFunction,
    pub rope_pair_tail: CudaFunction,
    pub act: CudaFunction,
    pub argmax: CudaFunction,
    pub argmax_rows: CudaFunction,
    pub argmax_f32: CudaFunction,
    pub finite_bf16: CudaFunction,
    pub advance: CudaFunction,
    pub flint_decode_length: CudaFunction,
    pub lm_head_f32: CudaFunction,
    pub residual: CudaFunction,
    swiglu_rounding: crate::model_spec::SwigluRounding,
}

pub struct RotaryPrefixKv {
    pub q: TensorMut<bf16, 3>,
    pub key: Tensor<bf16, 3>,
    pub value: Tensor<bf16, 3>,
    pub k: TensorMut<bf16, 3>,
    pub v: TensorMut<bf16, 3>,
    pub table: Tensor<bf16, 2>,
    pub start: u32,
}

impl Ops {
    pub fn rotary_prefix_kv(&self, stream: &Arc<CudaStream>, b: RotaryPrefixKv) -> Result<()> {
        let [rows, q_heads, dim] = b.q.0.shape;
        let kv_heads = b.key.shape[1];
        let [capacity, rotary] = b.table.shape;
        ensure!(
            rows > 0
                && q_heads > 0
                && kv_heads > 0
                && q_heads.is_multiple_of(kv_heads)
                && rotary > 0
                && rotary.is_multiple_of(2)
                && rotary <= dim
                && b.start.checked_add(rows).is_some_and(|end| end <= capacity)
                && [rows, q_heads, kv_heads, dim, rotary, capacity]
                    .iter()
                    .all(|&n| n <= i32::MAX as u32)
                && b.key.shape == [rows, kv_heads, dim]
                && b.value.shape == b.key.shape
                && b.k.0.shape == [kv_heads, capacity, dim]
                && b.v.0.shape == b.k.0.shape,
            "rotary prefix KV geometry/extent mismatch"
        );
        for tensor in [&b.q.0, &b.key, &b.value, &b.k.0, &b.v.0] {
            tensor.validate()?;
            ensure!(
                tensor.stride == dense_layout(tensor.shape)?.0,
                "rotary prefix requires dense tensors"
            );
        }
        b.table.validate()?;
        ensure!(
            b.table.stride == dense_layout(b.table.shape)?.0,
            "rotary prefix table must be dense"
        );
        let spans = [
            (b.q.0.ptr, b.q.0.span_bytes),
            (b.k.0.ptr, b.k.0.span_bytes),
            (b.v.0.ptr, b.v.0.span_bytes),
            (b.key.ptr, b.key.span_bytes),
            (b.value.ptr, b.value.span_bytes),
            (b.table.ptr, b.table.span_bytes),
        ];
        for i in 0..3 {
            for &(ptr, bytes) in &spans[i + 1..] {
                ensure!(
                    super::contracts::spans_disjoint(spans[i].0, spans[i].1, ptr, bytes)?,
                    "rotary prefix KV bindings overlap"
                );
            }
        }
        let count =
            usize::try_from(u64::from(rows) * u64::from(q_heads + kv_heads) * u64::from(dim))?;
        ensure!(
            count <= i32::MAX as usize,
            "rotary prefix grid exceeds indexing range"
        );
        unsafe {
            stream
                .launch_builder(&self.rotary_prefix_kv)
                .arg(&b.q.0.ptr)
                .arg(&b.key.ptr)
                .arg(&b.value.ptr)
                .arg(&b.k.0.ptr)
                .arg(&b.v.0.ptr)
                .arg(&b.table.ptr)
                .arg(&(b.start as i32))
                .arg(&(rows as i32))
                .arg(&(q_heads as i32))
                .arg(&(kv_heads as i32))
                .arg(&(dim as i32))
                .arg(&(rotary as i32))
                .arg(&(capacity as i32))
                .launch(flat(count))?;
        }
        Ok(())
    }

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
        Self::new_with_swiglu_rounding(ctx, crate::model_spec::SwigluRounding::Twice)
    }

    pub fn swiglu_rounding(&self) -> crate::model_spec::SwigluRounding {
        self.swiglu_rounding
    }

    /// Interleaved (cos, sin) rows of one position range under the
    /// adjacent-pair layout, from per-layer inverse frequencies; `inverse`
    /// holds the conjugate the attention output needs.
    pub fn rope_pair_rows(
        &self,
        stream: &Arc<CudaStream>,
        inv_freq: Tensor<f32, 1>,
        start: i32,
        rows: usize,
        forward: &mut TensorMut<f32, 2>,
        inverse: &mut TensorMut<f32, 2>,
    ) -> Result<()> {
        let (stride, _) = dense_layout(inv_freq.shape)?;
        ensure!(
            inv_freq.stride == stride,
            "rope pair inv_freq must be dense: stride {:?} != {:?}",
            inv_freq.stride,
            stride
        );
        let (stride, _) = dense_layout(forward.0.shape)?;
        ensure!(
            forward.0.stride == stride,
            "rope pair forward must be dense: stride {:?} != {:?}",
            forward.0.stride,
            stride
        );
        let (stride, _) = dense_layout(inverse.0.shape)?;
        ensure!(
            inverse.0.stride == stride,
            "rope pair inverse must be dense: stride {:?} != {:?}",
            inverse.0.stride,
            stride
        );
        let half = inv_freq.shape[0] as usize;
        let tail = forward.0.shape[1] as usize;
        ensure!(
            half >= 1 && tail == 2 * half,
            "rope pair rows tail {tail} is not twice the inverse-frequency count {half}"
        );
        ensure!(rows >= 1, "rope pair rows {rows} is below 1");
        ensure!(start >= 0, "rope pair rows start {start} is negative");
        ensure!(
            forward.0.shape == [u32::try_from(rows)?, u32::try_from(tail)?],
            "rope pair forward shape {:?} != expected [{rows}, {tail}]",
            forward.0.shape
        );
        ensure!(
            inverse.0.shape == forward.0.shape,
            "rope pair inverse shape {:?} != forward {:?}",
            inverse.0.shape,
            forward.0.shape
        );
        unsafe {
            stream
                .launch_builder(&self.rope_pair_rows)
                .arg(&inv_freq.ptr)
                .arg(&start)
                .arg(&i32::try_from(rows)?)
                .arg(&i32::try_from(tail)?)
                .arg(&forward.0.ptr)
                .arg(&inverse.0.ptr)
                .launch(flat(rows * half))?;
        }
        Ok(())
    }

    /// Rotate the last `tail` elements of every vector of `x` in place by the
    /// row's interleaved (cos, sin) pair, adjacent pairs as complex numbers.
    pub fn rope_pair_tail(
        &self,
        stream: &Arc<CudaStream>,
        x: &mut TensorMut<bf16, 3>,
        rotary: Tensor<f32, 2>,
    ) -> Result<()> {
        let (stride, _) = dense_layout(x.0.shape)?;
        ensure!(
            x.0.stride == stride,
            "rope pair input must be dense: stride {:?} != {:?}",
            x.0.stride,
            stride
        );
        let (stride, _) = dense_layout(rotary.shape)?;
        ensure!(
            rotary.stride == stride,
            "rope pair rotary must be dense: stride {:?} != {:?}",
            rotary.stride,
            stride
        );
        let rows = x.0.shape[0] as usize;
        let vectors = x.0.shape[1] as usize;
        let width = x.0.shape[2] as usize;
        let tail = rotary.shape[1] as usize;
        ensure!(
            rows >= 1 && vectors >= 1,
            "rope pair tail rows {rows} or vectors {vectors} are below 1"
        );
        ensure!(
            tail >= 2 && tail.is_multiple_of(2) && tail <= width,
            "rope pair tail {tail} is odd, below 2, or exceeds the vector width {width}"
        );
        ensure!(
            rotary.shape == [x.0.shape[0], u32::try_from(tail)?],
            "rope pair rotary shape {:?} != expected [{}, {tail}]",
            rotary.shape,
            x.0.shape[0]
        );
        unsafe {
            stream
                .launch_builder(&self.rope_pair_tail)
                .arg(&x.0.ptr)
                .arg(&rotary.ptr)
                .arg(&i32::try_from(rows)?)
                .arg(&i32::try_from(vectors)?)
                .arg(&i32::try_from(width)?)
                .arg(&i32::try_from(tail)?)
                .launch(flat(rows * vectors * tail / 2))?;
        }
        Ok(())
    }

    pub fn new_with_swiglu_rounding(
        ctx: &Arc<CudaContext>,
        swiglu_rounding: crate::model_spec::SwigluRounding,
    ) -> Result<Self> {
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
            add_norm: norm.load_function("add_rmsnorm")?,
            add_norm_unit: norm.load_function("add_rmsnorm_unit")?,
            add_norm_capture: add_norm_capture.load_function("add_rmsnorm_capture")?,
            add_norm_fp32: norm.load_function("add_rmsnorm_fp32")?,
            norm_capture: cubin::module(ctx, "q38_common")?.load_function("q38_norm_capture")?,
            kv_write: rope.load_function("kv_write")?,
            rope_kv: rope_kv.load_function("rope_kv")?,
            rotary_prefix_kv: rope.load_function("rotary_prefix_kv")?,
            rope_pair_rows: rope.load_function("rope_pair_rows")?,
            rope_pair_tail: rope.load_function("rope_pair_tail")?,
            act: act.load_function("silu_mul")?,
            argmax: sample.load_function("argmax")?,
            argmax_rows: sample.load_function("argmax_rows_bf16")?,
            argmax_f32: sample.load_function("argmax_f32")?,
            finite_bf16: sample.load_function("finite_bf16")?,
            advance: sample.load_function("advance")?,
            flint_decode_length: sample.load_function("flint_decode_length")?,
            lm_head_f32: sample.load_function("lm_head_f32")?,
            residual: norm.load_function("residual")?,
            swiglu_rounding,
        })
    }

    pub fn flint_decode_length(
        &self,
        stream: &Arc<CudaStream>,
        position: &CudaSlice<i32>,
        length: &mut CudaSlice<i32>,
        ring: usize,
    ) -> Result<()> {
        ensure!(
            position.len() == 1 && length.len() == 1 && ring <= i32::MAX as usize,
            "Flint decode position/length/ring extent differs"
        );
        unsafe {
            stream
                .launch_builder(&self.flint_decode_length)
                .arg(position)
                .arg(&mut length.slice_mut(..))
                .arg(&(ring as i32))
                .launch(grid(1, 1))?;
        }
        Ok(())
    }

    pub fn lm_head_f32(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaView<'_, bf16>,
        weight: &CudaSlice<bf16>,
        logits: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let vocab = logits.len();
        let hidden = input.len();
        ensure!(
            weight.len()
                == vocab
                    .checked_mul(hidden)
                    .context("LM head extent overflow")?,
            "FP32 LM head weight extent differs"
        );
        unsafe {
            stream
                .launch_builder(&self.lm_head_f32)
                .arg(input)
                .arg(weight)
                .arg(&mut logits.slice_mut(..))
                .arg(&(vocab as i32))
                .arg(&(hidden as i32))
                .launch(grid(vocab, 256))?;
        }
        Ok(())
    }

    pub fn argmax_f32(
        &self,
        stream: &Arc<CudaStream>,
        logits: &CudaSlice<f32>,
        token: &mut CudaSlice<u32>,
    ) -> Result<()> {
        ensure!(token.len() == 1, "FP32 argmax token extent differs");
        unsafe {
            stream
                .launch_builder(&self.argmax_f32)
                .arg(logits)
                .arg(&mut token.slice_mut(..))
                .arg(&(logits.len() as i32))
                .launch(grid(1, 1024))?;
        }
        Ok(())
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

use super::contracts::{RowDomain, Tensor, TensorMut, dense_layout};
use crate::backend::gpu::{CudaStream, PushKernelArg};
use anyhow::ensure;
use half::bf16;

#[derive(Clone, Copy, Debug)]
#[repr(i32)]
pub enum RmsMode {
    Standard = 0,
    ZeroCentered = 1,
    /// F32 `x * rsqrt(mean(x^2) + eps) * w` with one BF16 rounding at the store: neither the
    /// normalized value nor the weight product is rounded before it (model definition
    /// `RMSNorm.forward`).
    Unrounded = 5,
}
#[derive(Clone, Copy, Debug)]
pub struct RmsOptions {
    pub mode: RmsMode,
    pub eps: f32,
}
pub struct GdnGateBindings<'a> {
    pub a: Tensor<bf16, 2>,
    pub b: Tensor<bf16, 2>,
    pub a_log: Tensor<bf16, 1>,
    pub dt_bias: Tensor<bf16, 1>,
    pub beta: &'a mut TensorMut<bf16, 2>,
    pub g: &'a mut TensorMut<f32, 2>,
}
pub struct GdnChannelGateBindings<'a> {
    pub a: Tensor<bf16, 3>,
    pub b: Tensor<bf16, 2>,
    pub a_log: Tensor<f32, 1>,
    pub dt_bias: Tensor<f32, 1>,
    pub beta: &'a mut TensorMut<bf16, 2>,
    pub g: &'a mut TensorMut<f32, 3>,
    pub lower_bound: Option<f32>,
}
pub use super::gdn::{GdnBlockBindings, GdnBlockMode, GdnBlockPlan};
pub use super::moe::route::{BlockRouteBindings, BlockRoutePlan};

pub struct NormOps {
    norm: CudaFunction,
    norm_sigmoid: CudaFunction,
    norm_silu_f32: CudaFunction,
    layer_norm: CudaFunction,
}
pub struct GateOps {
    gate: CudaFunction,
    gelu: CudaFunction,
    gelu_tanh_mul: CudaFunction,
    swiglu_clamped: CudaFunction,
    swiglu_rounding: crate::model_spec::SwigluRounding,
    swiglu_limit: f32,
}
#[derive(Clone)]
pub struct CommonOpsRef {
    pub norm: Arc<NormOps>,
    pub gate: Arc<GateOps>,
}
impl CommonOpsRef {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        Self::new_with_swiglu_rounding(ctx, crate::model_spec::SwigluRounding::Twice, 0.0)
    }

    pub fn new_with_swiglu_rounding(
        ctx: &Arc<CudaContext>,
        swiglu_rounding: crate::model_spec::SwigluRounding,
        swiglu_limit: f32,
    ) -> Result<Self> {
        let module = cubin::module(ctx, "q38_common")?;
        let act = cubin::module(ctx, "act")?;
        Ok(Self {
            norm: Arc::new(NormOps {
                norm: module.load_function("q38_norm")?,
                norm_sigmoid: module.load_function("q38_norm_sigmoid")?,
                norm_silu_f32: module.load_function("rms_silu_f32")?,
                layer_norm: module.load_function("q38_layer_norm")?,
            }),
            gate: Arc::new(GateOps {
                gate: module.load_function("q38_gate")?,
                gelu: module.load_function("q38_gelu")?,
                gelu_tanh_mul: module.load_function("gelu_tanh_mul")?,
                swiglu_clamped: act.load_function("silu_mul_split_clamped")?,
                swiglu_rounding,
                swiglu_limit,
            }),
        })
    }
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
        no_overlap(x, y, "operator input", "operator output")?;
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
    gate_mode: Option<GatedRmsMode>,
    eps: f32,
}

#[derive(Clone, Copy, PartialEq)]
pub enum GatedRmsMode {
    SiluBf16,
    SiluF32,
    SigmoidF32,
}

pub struct GatedRmsInput {
    pub x: Tensor<bf16, 3>,
    pub weight: Tensor<bf16, 1>,
    pub gate: Tensor<bf16, 3>,
    pub mode: GatedRmsMode,
}

pub struct ResidualRmsInput<'a> {
    pub x: &'a mut TensorMut<bf16, 3>,
    pub residual: Tensor<bf16, 3>,
    pub weight: Tensor<bf16, 1>,
}

pub struct AddRmsNorm {
    pub input: Tensor<bf16, 2>,
    pub residual: Tensor<bf16, 2>,
    pub weight: Option<Tensor<bf16, 1>>,
    pub output: TensorMut<bf16, 2>,
    pub sum: TensorMut<bf16, 2>,
    pub eps: f32,
}

impl Ops {
    pub fn add_rmsnorm(&self, stream: &Arc<CudaStream>, call: AddRmsNorm) -> Result<()> {
        let AddRmsNorm {
            input,
            residual,
            weight,
            output,
            sum,
            eps,
        } = call;
        for t in [input, residual, output.0, sum.0] {
            t.validate()?;
            ensure!(
                t.shape == input.shape && t.stride == [u64::from(t.shape[1]), 1],
                "add_rmsnorm requires matching contiguous rows"
            );
        }
        ensure!(eps.is_finite() && eps > 0.0, "invalid add_rmsnorm epsilon");
        for t in [input, residual, sum.0] {
            no_overlap(&t, &output.0, "residual input/sum", "norm output")?;
        }
        no_overlap(&input, &sum.0, "input", "sum")?;
        if residual.ptr != sum.0.ptr {
            no_overlap(&residual, &sum.0, "residual", "sum")?;
        }
        let rows = i32::try_from(input.shape[0])?;
        let dim = i32::try_from(input.shape[1])?;
        let function = if weight.is_some() {
            &self.add_norm
        } else {
            &self.add_norm_unit
        };
        let mut launch = stream.launch_builder(function);
        launch.arg(&input.ptr).arg(&residual.ptr);
        let wp = weight.map(|w| w.ptr);
        if let Some(w) = weight {
            w.validate()?;
            ensure!(
                w.shape == [input.shape[1]] && w.stride == [1],
                "add_rmsnorm weight extent"
            );
            no_overlap(&w, &output.0, "weight", "norm output")?;
            no_overlap(&w, &sum.0, "weight", "sum")?;
        }
        if let Some(ptr) = wp.as_ref() {
            launch.arg(ptr);
        }
        unsafe {
            launch
                .arg(&output.0.ptr)
                .arg(&sum.0.ptr)
                .arg(&rows)
                .arg(&dim)
                .arg(&eps)
                .launch(grid(rows as usize, 256))?;
        }
        Ok(())
    }
}

pub struct ResidualRmsCapture<'a> {
    pub input: ResidualRmsInput<'a>,
    pub capture: &'a mut TensorMut<bf16, 3>,
    pub slot: u32,
}

impl Ops {
    pub fn add_rmsnorm_capture(
        &self,
        stream: &Arc<CudaStream>,
        domain: Option<RowDomain>,
        input: ResidualRmsCapture<'_>,
        y: &mut TensorMut<bf16, 3>,
        options: RmsOptions,
    ) -> Result<()> {
        let ResidualRmsCapture {
            input,
            capture,
            slot,
        } = input;
        let ResidualRmsInput {
            x,
            residual,
            weight,
        } = input;
        for tensor in [x.0, residual, y.0] {
            tensor.validate()?;
            ensure!(
                tensor.shape == x.0.shape
                    && tensor.shape[1] == 1
                    && tensor.stride == [u64::from(tensor.shape[2]), u64::from(tensor.shape[2]), 1],
                "capture norm requires contiguous rows with one hidden vector"
            );
        }
        let hidden = x.0.shape[2];
        ensure!(
            [2048, 5120].contains(&hidden),
            "capture norm hidden width is unsupported"
        );
        weight.validate()?;
        capture.0.validate()?;
        let taps = capture.0.shape[1];
        ensure!(
            weight.shape == [hidden]
                && weight.stride == [1]
                && capture.0.shape[0] >= x.0.shape[0]
                && capture.0.shape[2] == hidden
                && slot < taps
                && capture.0.stride == [u64::from(taps) * u64::from(hidden), u64::from(hidden), 1],
            "capture norm weight or [rows,taps,hidden] layout mismatch"
        );
        ensure!(
            options.eps.is_finite() && options.eps > 0.0,
            "invalid capture norm epsilon"
        );
        for (a, b) in [
            (x.0, residual),
            (x.0, y.0),
            (x.0, capture.0),
            (residual, y.0),
            (residual, capture.0),
            (y.0, capture.0),
        ] {
            no_overlap(&a, &b, "operator input", "operator output")?;
        }
        for tensor in [x.0, residual, y.0, capture.0] {
            no_overlap(&weight, &tensor, "operator input", "operator output")?;
        }
        let (meta, base, rows) = if let Some(domain) = domain {
            domain.validate(x.0.shape[0])?;
            (domain.meta.ptr, domain.row_base, domain.row_capacity)
        } else {
            (0, 0, x.0.shape[0])
        };
        let pitch = i32::try_from(u64::from(taps) * u64::from(hidden))?;
        let column = i32::try_from(u64::from(slot) * u64::from(hidden))?;
        if matches!(options.mode, RmsMode::ZeroCentered) {
            let slices = norm_slices_per_block(
                rows,
                &x.0,
                weight.ptr,
                residual.ptr,
                residual.stride,
                &y.0,
                4,
            );
            unsafe {
                stream
                    .launch_builder(&self.norm_capture)
                    .arg(&x.0.ptr)
                    .arg(&weight.ptr)
                    .arg(&residual.ptr)
                    .arg(&y.0.ptr)
                    .arg(&meta)
                    .arg(&base)
                    .arg(&rows)
                    .arg(&1u32)
                    .arg(&hidden)
                    .arg(&x.0.stride[0])
                    .arg(&x.0.stride[1])
                    .arg(&x.0.stride[2])
                    .arg(&y.0.stride[0])
                    .arg(&y.0.stride[1])
                    .arg(&y.0.stride[2])
                    .arg(&residual.stride[0])
                    .arg(&residual.stride[1])
                    .arg(&residual.stride[2])
                    .arg(&4i32)
                    .arg(&options.eps)
                    .arg(&capture.0.ptr)
                    .arg(&u64::try_from(pitch)?)
                    .arg(&u64::try_from(column)?)
                    .launch(grid(rows.div_ceil(slices) as usize, 256))?;
            }
            return Ok(());
        }
        unsafe {
            stream
                .launch_builder(&self.add_norm_capture)
                .arg(&residual.ptr)
                .arg(&x.0.ptr)
                .arg(&weight.ptr)
                .arg(&y.0.ptr)
                .arg(&x.0.ptr)
                .arg(&capture.0.ptr)
                .arg(&i32::try_from(rows)?)
                .arg(&i32::try_from(hidden)?)
                .arg(&options.eps)
                .arg(&pitch)
                .arg(&column)
                .arg(&(options.mode as i32))
                .arg(&meta)
                .arg(&i32::try_from(base)?)
                .launch(grid(rows as usize, 256))?;
        }
        Ok(())
    }

    pub fn rms_add_fp32_with(
        &self,
        stream: &Arc<CudaStream>,
        rows: RowDomain,
        input: ResidualRmsInput<'_>,
        y: &mut TensorMut<bf16, 3>,
        eps: f32,
    ) -> Result<()> {
        let ResidualRmsInput {
            x,
            residual: z,
            weight,
        } = input;
        validate_io(&x.0, &y.0, rows)?;
        z.validate()?;
        weight.validate()?;
        ensure!(
            z.shape == x.0.shape
                && z.stride == x.0.stride
                && weight.shape == [x.0.shape[2]]
                && weight.stride == [1],
            "FP32 residual norm tensor layout differs"
        );
        no_overlap(&z, &x.0, "operator input", "operator output")?;
        no_overlap(&weight, &x.0, "operator input", "operator output")?;
        no_overlap(&x.0, &y.0, "operator input", "operator output")?;
        ensure!(eps.is_finite() && eps > 0.0, "invalid epsilon {eps}");
        let hidden = x.0.shape[2];
        ensure!(
            hidden != 0 && hidden.is_multiple_of(8) && hidden <= 2048,
            "FP32 residual norm hidden={hidden} outside the cached 8..=2048 extent"
        );
        unsafe {
            stream
                .launch_builder(&self.add_norm_fp32)
                .arg(&x.0.ptr)
                .arg(&z.ptr)
                .arg(&weight.ptr)
                .arg(&y.0.ptr)
                .arg(&x.0.ptr)
                .arg(&rows.row_capacity)
                .arg(&hidden)
                .arg(&eps)
                .launch(LaunchConfig {
                    grid_dim: (rows.row_capacity, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }
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
    if x.stride[2] != 1 || y.stride[2] != 1 || ((mode == 3 || mode == 4) && zs[2] != 1) {
        return 1;
    }
    let ptrs = x.ptr | y.ptr | wp | if mode == 3 || mode == 4 { zp } else { 0 };
    let strides = x.stride[0]
        | x.stride[1]
        | y.stride[0]
        | y.stride[1]
        | if mode == 3 || mode == 4 {
            zs[0] | zs[1]
        } else {
            0
        };
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
                gate_mode: None,
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
        no_overlap(&x.0, &y.0, "operator input", "operator output")?;
        no_overlap(&z, &x.0, "operator input", "operator output")?;
        no_overlap(&weight, &x.0, "operator input", "operator output")?;
        self.launch(
            stream,
            NormLaunch {
                rows,
                x: x.0,
                weight: Some(weight),
                z: Some(z),
                y,
                mode: Some(mode),
                gate_mode: None,
                eps,
            },
        )
    }
    #[cfg(test)]
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
                gate_mode: None,
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
        let GatedRmsInput {
            x,
            weight,
            gate: z,
            mode: gate_mode,
        } = input;
        self.launch(
            stream,
            NormLaunch {
                rows,
                x,
                weight: Some(weight),
                z: Some(z),
                y,
                mode: Some(RmsMode::Standard),
                gate_mode: Some(gate_mode),
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
            gate_mode,
            eps,
        } = call;
        let mode = match (mode, z.is_some(), gate_mode) {
            (
                Some(RmsMode::Standard),
                true,
                Some(GatedRmsMode::SiluBf16 | GatedRmsMode::SiluF32),
            ) => 3,
            (Some(RmsMode::Standard), true, Some(GatedRmsMode::SigmoidF32)) => 3,
            (Some(RmsMode::ZeroCentered), true, None) => 4,
            (Some(mode), false, None) => mode as i32,
            (None, false, None) => 2,
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
            no_overlap(&w, &y.0, "operator input", "operator output")?;
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
            no_overlap(&z, &y.0, "operator input", "operator output")?;
            (z.ptr, z.stride)
        } else {
            (0, [0; 3])
        };
        let slices = rows
            .row_capacity
            .checked_mul(x.shape[1])
            .filter(|&n| n <= i32::MAX as u32)
            .ok_or_else(|| anyhow::anyhow!("norm grid overflow"))?;
        let kernel = match gate_mode {
            Some(GatedRmsMode::SigmoidF32) => &self.norm_sigmoid,
            Some(GatedRmsMode::SiluF32) => &self.norm_silu_f32,
            _ => &self.norm,
        };
        let spb = norm_slices_per_block(rows.row_capacity, &x, wp, zp, zs, &y.0, mode);
        let blocks = slices.div_ceil(spb);
        unsafe {
            stream
                .launch_builder(kernel)
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
        no_overlap(&x, &up, "operator input", "operator output")?;
        no_overlap(&up, &y.0, "operator input", "operator output")?;
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
        if self.swiglu_rounding == crate::model_spec::SwigluRounding::ClampedTwice {
            ensure!(
                self.swiglu_limit.is_finite() && self.swiglu_limit > 0.0,
                "clamped SwiGLU limit {} must be positive and finite",
                self.swiglu_limit
            );
            unsafe {
                stream
                    .launch_builder(&self.swiglu_clamped)
                    .arg(&x.ptr)
                    .arg(&up.ptr)
                    .arg(&y.0.ptr)
                    .arg(&rows.meta.ptr)
                    .arg(&rows.row_base)
                    .arg(&rows.row_capacity)
                    .arg(&width)
                    .arg(&self.swiglu_limit)
                    .launch(LaunchConfig {
                        grid_dim: (blocks, rows.row_capacity, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            return Ok(());
        }
        let mode = match self.swiglu_rounding {
            crate::model_spec::SwigluRounding::Twice => 2,
            crate::model_spec::SwigluRounding::Once => 3,
            crate::model_spec::SwigluRounding::ClampedOnce => {
                anyhow::bail!("clamped-once SwiGLU execution is not bound")
            }
            crate::model_spec::SwigluRounding::ClampedTwice => {
                unreachable!("ClampedTwice launches the clamped entry above")
            }
        };
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
                .arg(&mode)
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
            no_overlap(&g, &y.0, "operator input", "operator output")?;
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
    pub fn gdn_channel_gates_with(
        function: &CudaFunction,
        stream: &Arc<CudaStream>,
        rows: RowDomain,
        bindings: GdnChannelGateBindings<'_>,
    ) -> Result<()> {
        let GdnChannelGateBindings {
            a,
            b,
            a_log,
            dt_bias,
            beta,
            g,
            lower_bound,
        } = bindings;
        rows.validate(a.shape[0])?;
        a.validate()?;
        ensure!(
            a.shape[1] == 64 && a.shape[2] == 128 && a.stride == [64 * 128, 128, 1],
            "GDN per-channel a must be [rows, 64, 128] with strides [8192, 128, 1]"
        );
        for t in [&b, &beta.0] {
            t.validate()?;
            ensure!(
                t.shape == a.shape[..2] && t.stride[1] == 1 && t.stride[0] >= 64,
                "GDN per-channel beta layout mismatch"
            );
        }
        a_log.validate()?;
        ensure!(
            a_log.shape == [64] && a_log.stride == [1],
            "GDN A_log layout mismatch"
        );
        dt_bias.validate()?;
        ensure!(
            dt_bias.shape == [64 * 128] && dt_bias.stride == [1],
            "GDN per-channel dt_bias layout mismatch"
        );
        g.0.validate()?;
        ensure!(
            g.0.shape == a.shape && g.0.stride == a.stride,
            "GDN per-channel g layout mismatch"
        );
        no_overlap(&a, &beta.0, "operator input", "operator output")?;
        no_overlap(&a, &g.0, "operator input", "operator output")?;
        no_overlap(&b, &beta.0, "operator input", "operator output")?;
        no_overlap(&b, &g.0, "operator input", "operator output")?;
        for t in [&a_log, &dt_bias] {
            no_overlap(t, &beta.0, "operator input", "operator output")?;
            no_overlap(t, &g.0, "operator input", "operator output")?;
        }
        no_overlap(&beta.0, &g.0, "operator input", "operator output")?;
        unsafe {
            stream
                .launch_builder(function)
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
                .arg(&(i32::from(lower_bound.is_some())))
                .arg(&lower_bound.unwrap_or(0.0))
                .launch(LaunchConfig {
                    grid_dim: (rows.row_capacity.div_ceil(64), 64, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    pub fn gdn_gates_with(
        function: &CudaFunction,
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
        let heads = a.shape[1];
        rows.validate(a.shape[0])?;
        for t in [&a, &b, &beta.0] {
            t.validate()?;
            ensure!(
                t.shape == a.shape
                    && t.shape[1] == heads
                    && t.stride[1] == 1
                    && t.stride[0] >= u64::from(heads),
                "GDN gate layout mismatch"
            );
        }
        for t in [&a_log, &dt_bias] {
            t.validate()?;
            ensure!(
                t.shape == [heads] && t.stride == [1],
                "GDN gate weight mismatch"
            );
            no_overlap(t, &beta.0, "operator input", "operator output")?;
            no_overlap(t, &g.0, "operator input", "operator output")?;
        }
        g.0.validate()?;
        ensure!(
            g.0.shape == a.shape && g.0.stride[1] == 1 && g.0.stride[0] >= u64::from(heads),
            "GDN g layout mismatch"
        );
        for t in [&a, &b] {
            no_overlap(t, &beta.0, "operator input", "operator output")?;
            no_overlap(t, &g.0, "operator input", "operator output")?;
        }
        no_overlap(&beta.0, &g.0, "operator input", "operator output")?;
        let blocks = (u64::from(rows.row_capacity) * u64::from(heads))
            .div_ceil(256)
            .min(65535);
        unsafe {
            stream
                .launch_builder(function)
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

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use anyhow::ensure;

    #[test]
    #[ignore = "GPU window; FF_Q38_GPU_WINDOW=granted required"]
    fn norm_unrounded_and_sigmoid_dispatch_gpu() -> Result<()> {
        use super::super::contracts::view;
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let device = super::super::Device::new(0)?;
        let s = &device.stream;
        let common = CommonOpsRef::new(&device.ctx)?;
        let mut cases = 0;
        for width in [512usize, 1280, 5120] {
            for rows in [1usize, 3] {
                let pattern = |count, phase| -> Vec<bf16> {
                    (0..count)
                        .map(|i| bf16::from_f32(((i * 37 + phase) % 251) as f32 / 64.0 - 2.0))
                        .collect()
                };
                let input = pattern(rows * width, 11);
                let weight = pattern(width, 17);
                let gate = pattern(rows * width, 83);
                let w = s.clone_htod(&weight)?;
                let z = s.clone_htod(&gate)?;
                let metadata = s.clone_htod(&[0i32, rows as i32, rows as i32, 16, 0, 0, 0, 0])?;
                let domain = RowDomain {
                    meta: super::super::contracts::SeqRef {
                        ptr: view(&metadata, s, 0, [8], [1])?.ptr,
                    },
                    row_base: 0,
                    row_capacity: rows as u32,
                };
                let shape = [rows as u32, 1, width as u32];
                let stride = [width as u64, width as u64, 1];
                for sigmoid in [false, true] {
                    for in_place in [false, true] {
                        let x = s.clone_htod(&input)?;
                        let output = s.alloc_zeros::<bf16>(rows * width)?;
                        let xv = view(&x, s, 0, shape, stride)?;
                        let mut yv = TensorMut(view(
                            if in_place { &x } else { &output },
                            s,
                            0,
                            shape,
                            stride,
                        )?);
                        let wv = view(&w, s, 0, [width as u32], [1])?;
                        if sigmoid {
                            common.norm.rms_gated(
                                s,
                                domain,
                                GatedRmsInput {
                                    x: xv,
                                    weight: wv,
                                    gate: view(&z, s, 0, shape, stride)?,
                                    mode: GatedRmsMode::SigmoidF32,
                                },
                                &mut yv,
                                1e-6,
                            )?;
                        } else {
                            common.norm.rms_with(
                                s,
                                domain,
                                xv,
                                wv,
                                &mut yv,
                                RmsOptions {
                                    mode: RmsMode::Unrounded,
                                    eps: 1e-6,
                                },
                            )?;
                        }
                        s.synchronize().context("norm dispatcher launch failed")?;
                        let got = s.clone_dtoh(if in_place { &x } else { &output })?;
                        let unit = f32::EPSILON / 2.0;
                        let operations = (2 * width + 16) as f32;
                        let gamma = operations * unit / (1.0 - operations * unit);
                        let mut outside = 0;
                        let mut gate_control = 0;
                        for row in 0..rows {
                            let values = &input[row * width..(row + 1) * width];
                            let sum: f32 = values.iter().map(|v| v.to_f32() * v.to_f32()).sum();
                            let inv = (sum / width as f32 + 1e-6).sqrt().recip();
                            for col in 0..width {
                                let index = row * width + col;
                                let ungated = (values[col].to_f32() * inv) * weight[col].to_f32();
                                let expected = if sigmoid {
                                    ungated / (1.0 + (-gate[index].to_f32()).exp())
                                } else {
                                    ungated
                                };
                                let bound =
                                    expected.abs() * (1.0 / 256.0 + gamma) + f32::MIN_POSITIVE;
                                outside += usize::from(
                                    !got[index].is_finite()
                                        || (got[index].to_f32() - expected).abs() > bound,
                                );
                                gate_control += usize::from((ungated - expected).abs() > bound);
                            }
                        }
                        ensure!(
                            outside == 0,
                            "norm dispatcher outside={outside} width={width} rows={rows} sigmoid={sigmoid} in_place={in_place}"
                        );
                        ensure!(!sigmoid || gate_control > 0, "ungated control escaped");
                        println!(
                            "PASS NORM_DISPATCH shape=[{rows},1,{width}] sigmoid={sigmoid} in_place={in_place} outside={outside} ungated_control={gate_control}"
                        );
                        cases += 1;
                    }
                }
            }
        }
        println!("PASS NORM_DISPATCH cases={cases}");
        Ok(())
    }

    #[test]
    #[ignore = "GPU window; FF_Q38_GPU_WINDOW=granted required"]
    fn lm_head_f32_matches_fp64_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let device = super::super::Device::new(0)?;
        let s = &device.stream;
        let ops = Ops::new(&device.ctx)?;
        for (vocab, hidden) in [(129_280usize, 5120usize), (150_144, 2048)] {
            let input: Vec<_> = (0..hidden)
                .map(|k| bf16::from_f32(((k * 11) % 127 + 1) as f32 / 128.0))
                .collect();
            let weights: Vec<_> = (0..vocab * hidden)
                .map(|i| {
                    let (row, k) = (i / hidden, i % hidden);
                    bf16::from_f32(((row * 17 + k * 29) % 127 + 1) as f32 / 128.0)
                })
                .collect();
            let exact: Vec<f64> = (0..127)
                .map(|row| {
                    input
                        .iter()
                        .enumerate()
                        .map(|(k, x)| {
                            f64::from(x.to_f32()) * f64::from(weights[row * hidden + k].to_f32())
                        })
                        .sum()
                })
                .collect();
            let unit = f64::from(f32::EPSILON) / 2.0;
            let gamma = hidden as f64 * unit / (1.0 - hidden as f64 * unit);
            let x = s.clone_htod(&input)?;
            let w = s.clone_htod(&weights)?;
            drop(weights);
            let mut logits = s.alloc_zeros::<f32>(vocab)?;
            device.finish_upload()?;
            ops.lm_head_f32(s, &x.as_view(), &w, &mut logits)?;
            let eager = s.clone_dtoh(&logits)?;
            let mut outside = 0;
            let mut rounded_control = 0;
            let mut max_error = 0.0f64;
            for (row, &got) in eager.iter().enumerate() {
                let want = exact[row % 127];
                let bound = gamma * want.abs();
                let error = (f64::from(got) - want).abs();
                outside += usize::from(!got.is_finite() || error > bound);
                max_error = max_error.max(error);
                rounded_control +=
                    usize::from((f64::from(bf16::from_f32(got).to_f32()) - want).abs() > bound);
            }
            ensure!(
                outside == 0,
                "FP32 head {outside}/{vocab} outside gamma_{hidden}"
            );
            ensure!(rounded_control > 0, "BF16 logit store control escaped");
            s.begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )?;
            ops.lm_head_f32(s, &x.as_view(), &w, &mut logits)?;
            let graph = s
                .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))?
                .context("empty FP32 head graph")?;
            graph.launch()?;
            let replay = s.clone_dtoh(&logits)?;
            let bits = eager
                .iter()
                .zip(&replay)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            ensure!(bits == 0, "FP32 head graph differs at {bits}/{vocab}");
            println!(
                "PASS LM_HEAD_F32 shape=[{vocab},{hidden}] fp64_outside={outside} max_error={max_error:.12e} gamma={gamma:.12e} graph_bits={bits} bf16_store_control={rounded_control}"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "GPU window; FF_Q38_GPU_WINDOW=granted required"]
    fn argmax_f32_covers_all_launch_warps_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let device = super::super::Device::new(0)?;
        let stream = &device.stream;
        let ops = Ops::new(&device.ctx)?;
        let vocab = 150_144;

        let mut unique = vec![0f32; vocab];
        unique[300] = 1.0;
        let logits = stream.clone_htod(&unique)?;
        let mut token = stream.alloc_zeros::<u32>(1)?;
        ops.argmax_f32(stream, &logits, &mut token)?;
        let actual = stream.clone_dtoh(&token)?[0];
        ensure!(actual == 300, "warp-9 argmax mismatch: {actual}");

        let mut tied = vec![0f32; vocab];
        tied[200] = 1.0;
        tied[1200] = 1.0;
        let logits = stream.clone_htod(&tied)?;
        let mut token = stream.alloc_zeros::<u32>(1)?;
        ops.argmax_f32(stream, &logits, &mut token)?;
        let actual = stream.clone_dtoh(&token)?[0];
        ensure!(actual == 200, "tied argmax mismatch: {actual}");

        println!("ARGMAX_F32 unique=300 tie=200 PASS");
        Ok(())
    }
}

/// Stateless activation choice; callers include this in their plan identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum GeluMode {
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
        no_overlap(x, y, "operator input", "operator output")?;
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
            no_overlap(&t, &y.0, "operator input", "operator output")?;
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

    /// y = bf16(bf16(tanh-GELU(x)) * up); all declared rows are valid.
    pub fn gelu_tanh_mul(
        &self,
        stream: &Arc<CudaStream>,
        x: Tensor<bf16, 3>,
        up: Tensor<bf16, 3>,
        y: &mut TensorMut<bf16, 3>,
    ) -> Result<()> {
        validate_dense_io(&x, &y.0)?;
        validate_dense_io(&up, &y.0)?;
        let count = u64::from(x.shape[0])
            .checked_mul(u64::from(x.shape[1]))
            .and_then(|n| n.checked_mul(u64::from(x.shape[2])))
            .ok_or_else(|| anyhow::anyhow!("gated GELU element count overflow"))?;
        unsafe {
            stream
                .launch_builder(&self.gelu_tanh_mul)
                .arg(&x.ptr)
                .arg(&up.ptr)
                .arg(&y.0.ptr)
                .arg(&x.shape[0])
                .arg(&x.shape[1])
                .arg(&x.shape[2])
                .arg(&x.stride[0])
                .arg(&x.stride[1])
                .arg(&x.stride[2])
                .arg(&up.stride[0])
                .arg(&up.stride[1])
                .arg(&up.stride[2])
                .arg(&y.0.stride[0])
                .arg(&y.0.stride[1])
                .arg(&y.0.stride[2])
                .launch(grid(count.div_ceil(256).min(65535) as usize, 256))?;
        }
        Ok(())
    }
}
