//! Shared flow, dense/sparse decoders and image-conditioned mesh refinement.
#[cfg(test)]
#[path = "pixal3d_tests.rs"]
pub(crate) mod tests;
use super::sparse::{BatchGroupNorm, SparseConv};
pub use super::sparse::{TriangleVoxels, adaptive_mesh_support, triangle_voxels};

use super::{
    Device,
    blas::Blas,
    closure::Closure,
    contracts::{Tensor, TensorMut, view},
    cubin,
    flash::Score,
    flash::{F32Attention, F32Mask, Flash, Shape},
    linear::LinearBiasMode,
    ops::{Ops, flat, grid},
};
use crate::{
    backend::gpu::{
        CudaFunction, CudaSlice, DevicePtr, DeviceRepr, PushKernelArg, ValidAsZeroBits,
    },
    pixal3d::{DecoderCheckpoint, DecoderKind, FlowCheckpoint, FlowSpec, Hi3DFlowCheckpoint},
    sparse::{SerializedWindows, SparseSites},
};
use anyhow::{Context, Result, ensure};
use half::{bf16, f16};
use safetensors::Dtype;
use std::{collections::BTreeMap, mem::size_of, path::Path, rc::Rc};

pub(crate) type FlowWeights<T> = BTreeMap<String, CudaSlice<T>>;

fn wait_input<T>(device: &Device, input: &CudaSlice<T>) -> Result<()> {
    ensure!(
        input.context() == &device.ctx,
        "input device context differs from setup"
    );
    if device.stream.as_ref() != input.stream().as_ref() {
        let ready = input.stream().record_event(Some(
            cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
        ))?;
        device.stream.wait(&ready)?;
    }
    Ok(())
}

fn finite_copy(device: &Device, input: &CudaSlice<f32>, output: &CudaSlice<f32>) -> Result<()> {
    wait_input(device, input)?;
    ensure!(
        output.context() == &device.ctx && input.len() == output.len() && !input.is_empty(),
        "finite copy extent/context mismatch"
    );
    let kernel = cubin::module(&device.ctx, "q38_common")?.load_function("copy_finite_f32")?;
    unsafe {
        device
            .stream
            .launch_builder(&kernel)
            .arg(input)
            .arg(output)
            .arg(&(input.len() as u64))
            .launch(flat(input.len()))?;
    }
    Ok(())
}

pub(crate) fn ptr<T: DeviceRepr>(x: &CudaSlice<T>, d: &Device) -> u64 {
    x.device_ptr(&d.stream).0
}
fn zero_attention_tail(d: &Device, values: [u64; 3], rows: [usize; 3], width: usize) -> Result<()> {
    let kernel = cubin::module(&d.ctx, "vision_ops")?.load_function("zero_head_tail_f16")?;
    for (data, rows) in values.into_iter().zip(rows) {
        unsafe {
            d.stream
                .launch_builder(&kernel)
                .arg(&data)
                .arg(&u32::try_from(rows * width)?)
                .arg(&96u32)
                .arg(&64u32)
                .launch(flat(rows * width))?;
        }
    }
    Ok(())
}
pub(crate) fn dense<T: DeviceRepr + Copy, const R: usize>(
    x: &CudaSlice<T>,
    d: &Device,
    shape: [u32; R],
) -> Result<Tensor<T, R>> {
    let (stride, _) = super::contracts::dense_layout(shape)?;
    view(x, &d.stream, 0, shape, stride)
}
pub(crate) fn matrix<T: Copy>(address: u64, rows: usize, width: usize) -> Result<Tensor<T, 2>> {
    let bytes = rows
        .checked_mul(width)
        .and_then(|n| n.checked_mul(size_of::<T>()))
        .context("flow matrix size overflow")?;
    unsafe {
        Tensor::new(
            address,
            [u32::try_from(rows)?, u32::try_from(width)?],
            [width as u64, 1],
            bytes as u64,
        )
    }
}

pub(crate) fn run_linear_f32(
    d: &Device,
    blas: &Blas,
    kernels: &Kernels,
    linear: Linear<f32>,
    rows: usize,
    buffers: [u64; 2],
    silu: bool,
) -> Result<()> {
    let [input, output] = buffers;
    blas.linear_f32(
        linear.weight,
        matrix(input, rows, linear.weight.shape[1] as usize)?,
        &mut TensorMut(matrix(output, rows, linear.weight.shape[0] as usize)?),
    )?;
    let width = linear.weight.shape[0];
    unsafe {
        let mut launch = d
            .stream
            .launch_builder(if silu { &kernels.silu } else { &kernels.bias });
        launch.arg(&output).arg(&linear.bias.ptr).arg(&output);
        launch
            .arg(
                &(if silu {
                    rows as u32 * width
                } else {
                    rows as u32
                }),
            )
            .arg(&width)
            .launch(flat(rows * width as usize))?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(crate) struct Linear<T> {
    pub(crate) weight: Tensor<T, 2>,
    pub(crate) bias: Tensor<T, 1>,
}
impl<T: DeviceRepr + Copy> Linear<T> {
    pub(crate) fn bind(
        weights: &FlowWeights<T>,
        d: &Device,
        name: &str,
        shape: [usize; 2],
    ) -> Result<Self> {
        let weight = weights
            .get(&format!("{name}.weight"))
            .with_context(|| format!("linear weight missing: {name}"))?;
        let bias = weights
            .get(&format!("{name}.bias"))
            .with_context(|| format!("linear bias missing: {name}"))?;
        Ok(Self {
            weight: dense(weight, d, [shape[0] as u32, shape[1] as u32])?,
            bias: dense(bias, d, [shape[0] as u32])?,
        })
    }
}
pub trait FlowElement: DeviceRepr + Copy + Default + ValidAsZeroBits {
    type Parameters: DeviceRepr + Copy + ValidAsZeroBits;
    const HALF: bool;
    fn from_f32(value: f32) -> Self;
    fn finite(value: Self) -> bool;
    fn prepare(
        blas: &mut Blas,
        rows: usize,
        weights: (Tensor<Self, 2>, Tensor<Self, 1>),
    ) -> Result<()>;
    fn linear(
        blas: &Blas,
        weights: (Tensor<Self, 2>, Tensor<Self, 1>),
        x: Tensor<Self, 2>,
        y: &mut TensorMut<Self, 2>,
        prefilled: bool,
    ) -> Result<()>;
    fn attention(
        op: &mut Flash,
        q: &CudaSlice<Self>,
        k: &CudaSlice<Self>,
        v: &CudaSlice<Self>,
        output: &mut CudaSlice<Self>,
        shape: Shape,
        offsets: [usize; 2],
    ) -> Result<()>;
}
impl FlowElement for bf16 {
    type Parameters = bf16;
    const HALF: bool = false;
    fn from_f32(value: f32) -> Self {
        bf16::from_f32(value)
    }
    fn finite(value: Self) -> bool {
        value.is_finite()
    }
    fn prepare(
        blas: &mut Blas,
        rows: usize,
        weights: (Tensor<Self, 2>, Tensor<Self, 1>),
    ) -> Result<()> {
        blas.prepare_biased(
            rows as u32,
            weights.0.shape[0],
            weights.0.shape[1],
            weights.1,
            LinearBiasMode::BeforeOutputRound,
        )
    }
    fn linear(
        blas: &Blas,
        weights: (Tensor<Self, 2>, Tensor<Self, 1>),
        x: Tensor<Self, 2>,
        y: &mut TensorMut<Self, 2>,
        prefilled: bool,
    ) -> Result<()> {
        ensure!(!prefilled, "BF16 prefilled flow linear is unsupported");
        blas.linear_biased(
            weights.0,
            weights.1,
            x,
            y,
            LinearBiasMode::BeforeOutputRound,
        )
    }
    fn attention(
        op: &mut Flash,
        q: &CudaSlice<Self>,
        k: &CudaSlice<Self>,
        v: &CudaSlice<Self>,
        output: &mut CudaSlice<Self>,
        shape: Shape,
        offsets: [usize; 2],
    ) -> Result<()> {
        ensure!(
            offsets == [0, 0],
            "BF16 attention cache offsets are unsupported"
        );
        op.run(q, shape.q_heads * shape.head_dim, k, v, output, shape)
    }
}
impl FlowElement for f16 {
    type Parameters = f32;
    const HALF: bool = true;
    fn from_f32(value: f32) -> Self {
        f16::from_f32(value)
    }
    fn finite(value: Self) -> bool {
        value.is_finite()
    }
    fn prepare(
        blas: &mut Blas,
        _rows: usize,
        _weights: (Tensor<Self, 2>, Tensor<Self, 1>),
    ) -> Result<()> {
        blas.prepare_f16_biased()
    }
    fn linear(
        blas: &Blas,
        weights: (Tensor<Self, 2>, Tensor<Self, 1>),
        x: Tensor<Self, 2>,
        y: &mut TensorMut<Self, 2>,
        prefilled: bool,
    ) -> Result<()> {
        if prefilled {
            blas.linear_f16_prefilled(weights.0, weights.1, x, y)
        } else {
            blas.linear_f16_biased(weights.0, weights.1, x, y)
        }
    }
    fn attention(
        op: &mut Flash,
        q: &CudaSlice<Self>,
        k: &CudaSlice<Self>,
        v: &CudaSlice<Self>,
        output: &mut CudaSlice<Self>,
        shape: Shape,
        offsets: [usize; 2],
    ) -> Result<()> {
        let count = shape.kv_heads * shape.capacity * shape.head_dim;
        op.run_f16(
            &q.slice(..),
            shape.q_heads * shape.head_dim,
            &k.slice(offsets[0]..offsets[0] + count),
            &v.slice(offsets[1]..offsets[1] + count),
            &mut output.slice_mut(..),
            (shape.rows, shape.capacity, shape.capacity),
        )
    }
}
impl<T: FlowElement> Linear<T> {
    pub(crate) fn prepare(self, blas: &mut Blas, rows: usize) -> Result<()> {
        T::prepare(blas, rows, (self.weight, self.bias))
    }
    pub(crate) fn run(self, blas: &Blas, rows: usize, input: u64, output: u64) -> Result<()> {
        T::linear(
            blas,
            (self.weight, self.bias),
            matrix(input, rows, self.weight.shape[1] as usize)?,
            &mut TensorMut(matrix(output, rows, self.weight.shape[0] as usize)?),
            false,
        )
    }
    fn run_prefilled(self, blas: &Blas, rows: usize, input: u64, output: u64) -> Result<()> {
        T::linear(
            blas,
            (self.weight, self.bias),
            matrix(input, rows, self.weight.shape[1] as usize)?,
            &mut TensorMut(matrix(output, rows, self.weight.shape[0] as usize)?),
            true,
        )
    }
}

pub(crate) struct Block<T: FlowElement> {
    pub(crate) qkv: Linear<T>,
    pub(crate) self_out: Linear<T>,
    pub(crate) cross_q: Linear<T>,
    pub(crate) cross_kv: Linear<T>,
    pub(crate) cross_out: Linear<T>,
    pub(crate) projection: Option<Linear<T>>,
    pub(crate) fc1: Linear<T>,
    pub(crate) fc2: Linear<T>,
    pub(crate) norm_weight: u64,
    pub(crate) norm_bias: u64,
    pub(crate) modulation: u64,
    pub(crate) gamma: [u64; 4],
}

struct ProducerBias {
    kernels: [CudaFunction; 4],
    grids: [crate::backend::gpu::LaunchConfig; 4],
    rows: u32,
    width: u32,
    output_widths: [u32; 4],
}

impl ProducerBias {
    fn new(d: &Device, rows: usize, width: usize) -> Result<Self> {
        ensure!(
            rows > 0 && width == 2112,
            "F16 producer bias requires P>0,H2112"
        );
        ensure!(
            rows <= i32::MAX as usize,
            "F16 producer bias row count exceeds GEMM limit"
        );
        let rows = u32::try_from(rows)?;
        let width = u32::try_from(width)?;
        let output_widths = [3 * width, width, 4 * width, width];
        let grids = output_widths.map(|output| {
            let bias_blocks = (u64::from(rows) * u64::from(output))
                .div_ceil(256)
                .min(65535);
            grid(u64::from(rows).max(bias_blocks) as usize, 256)
        });
        let mut grids = grids;
        grids[3] = grid(
            (u64::from(rows) * u64::from(4 * width))
                .div_ceil(256)
                .min(65535) as usize,
            256,
        );
        let module = cubin::module(&d.ctx, "q38_common")?;
        Ok(Self {
            kernels: [
                module.load_function("layer_norm_modulate_bias_f16")?,
                module.load_function("gated_add_layer_norm_bias_f16")?,
                module.load_function("residual_layer_norm_modulate_bias_f16")?,
                module.load_function("gelu_bias_fill_f16")?,
            ],
            grids,
            rows,
            width,
            output_widths,
        })
    }

    fn row(&self, d: &Device, inputs: [u64; 5], outputs: [u64; 3], site: usize) -> Result<()> {
        ensure!(site < 3, "F16 producer bias row site invalid");
        let [hidden, projected, modulation, weight, bias] = inputs;
        let [normalized, output, output_bias] = outputs;
        unsafe {
            d.stream
                .launch_builder(&self.kernels[site])
                .arg(&hidden)
                .arg(&projected)
                .arg(&modulation)
                .arg(&weight)
                .arg(&bias)
                .arg(&normalized)
                .arg(&self.rows)
                .arg(&self.width)
                .arg(&1e-6f32)
                .arg(&output)
                .arg(&output_bias)
                .arg(&self.output_widths[site])
                .launch(self.grids[site])?;
        }
        Ok(())
    }

    fn gelu(&self, d: &Device, values: u64, output: u64, bias: u64) -> Result<()> {
        unsafe {
            d.stream
                .launch_builder(&self.kernels[3])
                .arg(&values)
                .arg(&output)
                .arg(&bias)
                .arg(&self.rows)
                .arg(&(4 * self.width))
                .arg(&self.width)
                .launch(self.grids[3])?;
        }
        Ok(())
    }
}

pub(crate) struct Kernels {
    pub(crate) half: bool,
    pub(crate) heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) norm: CudaFunction,
    pub(crate) norm_f32: CudaFunction,
    norm_silu: Option<[CudaFunction; 2]>,
    #[cfg(test)]
    norm_silu_reference: bool,
    #[cfg(test)]
    norm_silu_separate: Option<[CudaFunction; 2]>,
    pub(crate) l2: CudaFunction,
    pub(crate) unpack: CudaFunction,
    qkv_prepare: Option<CudaFunction>,
    #[cfg(test)]
    qkv_reference: bool,
    row_norm: Option<[CudaFunction; 3]>,
    #[cfg(test)]
    row_norm_reference: bool,
    producer_bias: Option<ProducerBias>,
    #[cfg(test)]
    bias_reference: bool,
    pub(crate) unpack_cross: CudaFunction,
    #[cfg(test)]
    context_chains: std::cell::Cell<usize>,
    pub(crate) rope: CudaFunction,
    pub(crate) rope_hm: CudaFunction,
    pub(crate) phases: CudaFunction,
    pub(crate) modulate: CudaFunction,
    pub(crate) gated: CudaFunction,
    pub(crate) add_mod: CudaFunction,
    pub(crate) residual: CudaFunction,
    pub(crate) cast: CudaFunction,
    pub(crate) upcast: CudaFunction,
    pub(crate) bias: CudaFunction,
    pub(crate) silu: CudaFunction,
    pub(crate) sinusoid: CudaFunction,
    pub(crate) gelu: CudaFunction,
    pub(crate) transpose: CudaFunction,
    pub(crate) concat: CudaFunction,
}
enum CrossAttention {
    F32(F32Attention),
    F16(Flash),
}

struct ResidualIo {
    rows: usize,
    input_width: usize,
    output_width: usize,
    norm: [u64; 2],
    conv: [SparseConv; 2],
    conv_weights: [(Tensor<f16, 2>, Tensor<f16, 1>); 2],
    embedding: Linear<f16>,
    linear: Linear<f16>,
    skip: Option<Linear<f16>>,
    normalized: CudaSlice<f16>,
    first: CudaSlice<f16>,
    second: CudaSlice<f16>,
    result: CudaSlice<f16>,
    residual: CudaSlice<f16>,
    parameters: CudaSlice<f16>,
    silu: CudaFunction,
}
impl ResidualIo {
    fn new(
        d: &Device,
        sites: &SparseSites,
        weights: (&FlowWeights<f16>, &FlowWeights<f32>),
        name: &str,
        widths: [usize; 3],
        columns: Rc<CudaSlice<f16>>,
    ) -> Result<Self> {
        let [input, output, embedding] = widths;
        let hidden = output / 2;
        let rows = sites.coordinates().len();
        let (half, float) = weights;
        let bind = |suffix: &str, out, input| {
            Linear::bind(half, d, &format!("{name}.{suffix}"), [out, input])
        };
        let conv_weight = |suffix: &str, channels| -> Result<_> {
            let p = format!("{name}.{suffix}.conv");
            Ok((
                dense(
                    &half[&format!("{p}.weight")],
                    d,
                    [hidden as u32, (27 * channels) as u32],
                )?,
                dense(&half[&format!("{p}.bias")], d, [hidden as u32])?,
            ))
        };
        Ok(Self {
            rows,
            input_width: input,
            output_width: output,
            norm: [
                ptr(&float[&format!("{name}.norm1.weight")], d),
                ptr(&float[&format!("{name}.norm1.bias")], d),
            ],
            conv: [
                SparseConv::shared(d, sites, input, columns.clone())?,
                SparseConv::shared(d, sites, hidden, columns)?,
            ],
            conv_weights: [conv_weight("conv1", input)?, conv_weight("conv2", hidden)?],
            embedding: bind("emb_layers.1", 2 * hidden, embedding)?,
            linear: bind("linear", output, hidden)?,
            skip: if input == output {
                None
            } else {
                Some(bind("skip_connection", output, input)?)
            },
            normalized: d.stream.alloc_zeros(rows * input)?,
            first: d.stream.alloc_zeros(rows * hidden)?,
            second: d.stream.alloc_zeros(rows * hidden)?,
            result: d.stream.alloc_zeros(rows * output)?,
            residual: d
                .stream
                .alloc_zeros(if input == output { 0 } else { rows * output })?,
            parameters: d.stream.alloc_zeros(2 * hidden)?,
            silu: cubin::module(&d.ctx, "vision_ops")?.load_function("vision_silu_f16")?,
        })
    }
    fn run(
        &mut self,
        d: &Device,
        blas: &Blas,
        k: &Kernels,
        input: &CudaSlice<f16>,
        embedding: u64,
    ) -> Result<()> {
        let hidden = self.output_width / 2;
        let activate = |x: u64, out: u64, count: usize| -> Result<()> {
            unsafe {
                d.stream
                    .launch_builder(&self.silu)
                    .arg(&x)
                    .arg(&out)
                    .arg(&u32::try_from(count)?)
                    .launch(flat(count))?;
            }
            Ok(())
        };
        self.embedding
            .run(blas, 1, embedding, ptr(&self.parameters, d))?;
        k.norm_silu(
            d,
            ptr(input, d),
            ptr(&self.normalized, d),
            [self.rows, self.input_width],
            self.norm,
            0,
        )?;
        self.conv[0].run(
            blas,
            &self.normalized,
            self.conv_weights[0].0,
            self.conv_weights[0].1,
            &mut self.first,
        )?;
        k.norm_silu(
            d,
            ptr(&self.first, d),
            ptr(&self.first, d),
            [self.rows, hidden],
            [0, 0],
            ptr(&self.parameters, d),
        )?;
        self.conv[1].run(
            blas,
            &self.first,
            self.conv_weights[1].0,
            self.conv_weights[1].1,
            &mut self.second,
        )?;
        activate(
            ptr(&self.second, d),
            ptr(&self.second, d),
            self.second.len(),
        )?;
        self.linear
            .run(blas, self.rows, ptr(&self.second, d), ptr(&self.result, d))?;
        if let Some(skip) = self.skip {
            skip.run(blas, self.rows, ptr(input, d), ptr(&self.residual, d))?;
            k.residual(
                d,
                ptr(&self.result, d),
                ptr(&self.residual, d),
                self.result.len(),
            )?;
        } else {
            k.residual(d, ptr(&self.result, d), ptr(input, d), self.result.len())?;
        }
        Ok(())
    }
}

struct SparseFlowIo {
    rows: usize,
    inverse: CudaSlice<u32>,
    offsets: CudaSlice<u32>,
    children: CudaSlice<u32>,
    blocks: [ResidualIo; 4],
    projected: CudaSlice<f32>,
    input: CudaSlice<f16>,
    pooled: CudaSlice<f16>,
    concat: CudaSlice<f16>,
    expanded: CudaSlice<f16>,
    last_concat: CudaSlice<f16>,
    normalized: CudaSlice<f16>,
    output_float: CudaSlice<f32>,
    embedding: CudaSlice<f16>,
    pool: CudaFunction,
    index: CudaFunction,
    concatenate: CudaFunction,
    silu: CudaFunction,
}
impl SparseFlowIo {
    fn gather_partition(shape: [usize; 4], budget: usize) -> Result<[usize; 8]> {
        let [rows, parents, hidden, channels] = shape;
        let width = hidden
            .max(channels)
            .checked_mul(2)
            .context("sparse gather width overflow")?;
        ensure!(
            rows > 0
                && parents > 0
                && hidden >= 2
                && channels >= 2
                && width <= i32::MAX as usize / 27,
            "invalid sparse gather geometry"
        );
        let full = rows
            .checked_mul(27 * width)
            .context("shared sparse gather extent overflow")?;
        let elements = full.min(budget / size_of::<f16>());
        let shapes = [
            (rows, channels),
            (rows, channels / 2),
            (parents, channels),
            (parents, hidden / 2),
            (rows, 2 * hidden),
            (rows, channels / 2),
            (rows, 2 * channels),
            (rows, channels / 2),
        ];
        let mut partition = [0; 8];
        for ((rows, channels), tile) in shapes.into_iter().zip(&mut partition) {
            *tile = SparseConv::tile_rows(rows, channels, elements)?;
        }
        Ok(partition)
    }
    fn context_cache_fits(shape: [usize; 4], budget: usize, growth: usize) -> Result<bool> {
        let baseline = Self::gather_partition(shape, budget)?;
        ensure!(
            baseline.iter().all(|&rows| rows > 0),
            "sparse gather cannot fit one row"
        );
        let Some(remaining) = budget.checked_sub(growth) else {
            return Ok(false);
        };
        Ok(Self::gather_partition(shape, remaining)? == baseline)
    }
    fn new(
        d: &Device,
        sites: &SparseSites,
        parent: &SparseSites,
        inverse: &[u32],
        weights: (&FlowWeights<f16>, &FlowWeights<f32>),
        budget: usize,
    ) -> Result<Self> {
        let h = weights.1["t_embedder.mlp.2.bias"].len();
        let c = weights.1["input_layer.bias"].len();
        let rows = sites.coordinates().len();
        let parents = parent.coordinates().len();
        let mut lists = vec![Vec::new(); parents];
        for (child, &p) in inverse.iter().enumerate() {
            lists[p as usize].push(child as u32);
        }
        let mut offsets = vec![0u32];
        let mut children = Vec::with_capacity(rows);
        for list in lists {
            children.extend(list);
            offsets.push(children.len() as u32);
        }
        let full = rows
            .checked_mul(27 * (2 * h).max(2 * c))
            .context("shared sparse gather extent overflow")?;
        let columns = Rc::new(
            d.stream
                .alloc_zeros::<f16>(full.min(budget / size_of::<f16>()))?,
        );
        let v = cubin::module(&d.ctx, "vision_ops")?;
        Ok(Self {
            rows,
            inverse: d.stream.clone_htod(inverse)?,
            offsets: d.stream.clone_htod(&offsets)?,
            children: d.stream.clone_htod(&children)?,
            blocks: [
                ResidualIo::new(
                    d,
                    sites,
                    weights,
                    "input_blocks.0",
                    [c, c, h],
                    columns.clone(),
                )?,
                ResidualIo::new(
                    d,
                    parent,
                    weights,
                    "input_blocks.1",
                    [c, h, h],
                    columns.clone(),
                )?,
                ResidualIo::new(
                    d,
                    sites,
                    weights,
                    "out_blocks.0",
                    [(2 * h), c, h],
                    columns.clone(),
                )?,
                ResidualIo::new(
                    d,
                    sites,
                    weights,
                    "out_blocks.1",
                    [(2 * c), c, h],
                    columns.clone(),
                )?,
            ],
            projected: d.stream.alloc_zeros(rows * c)?,
            input: d.stream.alloc_zeros(rows * c)?,
            pooled: d.stream.alloc_zeros(parents * c)?,
            concat: d.stream.alloc_zeros(parents * (2 * h))?,
            expanded: d.stream.alloc_zeros(rows * (2 * h))?,
            last_concat: d.stream.alloc_zeros(rows * (2 * c))?,
            normalized: d.stream.alloc_zeros(rows * c)?,
            output_float: d.stream.alloc_zeros(rows * c)?,
            embedding: d.stream.alloc_zeros(h)?,
            pool: v.load_function("sparse_mean_seeded_f16")?,
            index: v.load_function("sparse_index_rows_f16")?,
            concatenate: v.load_function("concat_rows_f16")?,
            silu: v.load_function("vision_silu_f16")?,
        })
    }
    fn copy(
        &self,
        d: &Device,
        input: u64,
        index: u64,
        output: u64,
        rows: usize,
        width: usize,
    ) -> Result<()> {
        unsafe {
            d.stream
                .launch_builder(&self.index)
                .arg(&input)
                .arg(&index)
                .arg(&output)
                .arg(&(rows as u32))
                .arg(&(width as u32))
                .launch(flat(rows * width))?;
        }
        Ok(())
    }
    fn concat(
        &self,
        d: &Device,
        input: u64,
        skip: u64,
        output: u64,
        rows: usize,
        width: usize,
    ) -> Result<()> {
        unsafe {
            d.stream
                .launch_builder(&self.concatenate)
                .arg(&input)
                .arg(&skip)
                .arg(&output)
                .arg(&(rows as u32))
                .arg(&(width as u32))
                .arg(&(width as u32))
                .launch(flat(rows * width * 2))?;
        }
        Ok(())
    }
    fn encode(
        &mut self,
        d: &Device,
        blas: &Blas,
        k: &Kernels,
        input: Linear<f32>,
        buffers: [u64; 3],
    ) -> Result<()> {
        let [values, embedding, hidden] = buffers;
        let c = self.blocks[0].output_width;
        let h = self.blocks[1].output_width;
        run_linear_f32(
            d,
            blas,
            k,
            input,
            self.rows,
            [values, ptr(&self.projected, d)],
            false,
        )?;
        k.cast(
            d,
            ptr(&self.projected, d),
            ptr(&self.input, d),
            self.input.len(),
        )?;
        unsafe {
            d.stream
                .launch_builder(&self.silu)
                .arg(&embedding)
                .arg(&self.embedding)
                .arg(&(h as u32))
                .launch(flat(h))?;
        }
        self.blocks[0].run(d, blas, k, &self.input, ptr(&self.embedding, d))?;
        unsafe {
            d.stream
                .launch_builder(&self.pool)
                .arg(&self.blocks[0].result)
                .arg(&self.offsets)
                .arg(&self.children)
                .arg(&self.pooled)
                .arg(&((self.offsets.len() - 1) as u32))
                .arg(&(c as u32))
                .launch(flat(self.pooled.len()))?;
        }
        self.blocks[1].run(d, blas, k, &self.pooled, ptr(&self.embedding, d))?;
        self.copy(
            d,
            ptr(&self.blocks[1].result, d),
            0,
            hidden,
            self.offsets.len() - 1,
            h,
        )
    }
    fn decode(
        &mut self,
        d: &Device,
        blas: &Blas,
        k: &Kernels,
        output: Linear<f32>,
        hidden: u64,
        values: u64,
    ) -> Result<()> {
        let c = self.blocks[3].output_width;
        let h = self.blocks[1].output_width;
        self.concat(
            d,
            hidden,
            ptr(&self.blocks[1].result, d),
            ptr(&self.concat, d),
            self.offsets.len() - 1,
            h,
        )?;
        self.copy(
            d,
            ptr(&self.concat, d),
            ptr(&self.inverse, d),
            ptr(&self.expanded, d),
            self.rows,
            2 * h,
        )?;
        self.blocks[2].run(d, blas, k, &self.expanded, ptr(&self.embedding, d))?;
        self.concat(
            d,
            ptr(&self.blocks[2].result, d),
            ptr(&self.blocks[0].result, d),
            ptr(&self.last_concat, d),
            self.rows,
            c,
        )?;
        self.blocks[3].run(d, blas, k, &self.last_concat, ptr(&self.embedding, d))?;
        k.norm(
            d,
            ptr(&self.blocks[3].result, d),
            ptr(&self.normalized, d),
            [self.rows, c],
            [0, 0],
            (1e-5, false),
        )?;
        k.upcast(
            d,
            ptr(&self.normalized, d),
            ptr(&self.output_float, d),
            self.normalized.len(),
        )?;
        run_linear_f32(
            d,
            blas,
            k,
            output,
            self.rows,
            [ptr(&self.output_float, d), values],
            false,
        )
    }
}
impl Kernels {
    pub(crate) fn load<T: FlowElement>(d: &Device, spec: &FlowSpec) -> Result<Self> {
        let q = cubin::module(&d.ctx, "q38_common")?;
        let n = cubin::module(&d.ctx, "norm")?;
        let r = cubin::module(&d.ctx, "rope")?;
        let m = cubin::module(&d.ctx, "modulate")?;
        let v = cubin::module(&d.ctx, "vision_ops")?;
        Ok(Self {
            half: T::HALF,
            heads: spec.num_heads,
            head_dim: spec.model_channels / spec.num_heads,
            norm: q.load_function(if T::HALF {
                "q38_layer_norm_f16_fma"
            } else {
                "q38_layer_norm_bf16_affine_f32"
            })?,
            norm_f32: q.load_function("q38_layer_norm_f32")?,
            norm_silu: if T::HALF {
                Some([
                    q.load_function("layer_norm_silu_f16")?,
                    q.load_function("layer_norm_modulate_silu_f16")?,
                ])
            } else {
                None
            },
            #[cfg(test)]
            norm_silu_reference: false,
            #[cfg(test)]
            norm_silu_separate: if T::HALF {
                Some([
                    v.load_function("vision_silu_f16")?,
                    m.load_function("row_modulate_f16")?,
                ])
            } else {
                None
            },
            l2: n.load_function(if T::HALF {
                "l2_norm_scale_f16"
            } else {
                "l2_norm_scale_bf16"
            })?,
            unpack: v.load_function(if T::HALF {
                "qkv_unpack_hm_f16"
            } else {
                "qkv_unpack_hm_bf16"
            })?,
            qkv_prepare: if T::HALF {
                Some(n.load_function("qkv_rope_l2_d96_f16")?)
            } else {
                None
            },
            #[cfg(test)]
            qkv_reference: false,
            row_norm: if T::HALF {
                Some([
                    q.load_function("layer_norm_modulate_f16")?,
                    q.load_function("gated_add_layer_norm_f16")?,
                    q.load_function("residual_layer_norm_modulate_f16")?,
                ])
            } else {
                None
            },
            #[cfg(test)]
            row_norm_reference: false,
            producer_bias: None,
            #[cfg(test)]
            bias_reference: false,
            unpack_cross: v.load_function(if T::HALF {
                "kv_unpack_head_major_f16"
            } else {
                "qkv_unpack_hm_bf16"
            })?,
            #[cfg(test)]
            context_chains: std::cell::Cell::new(0),
            rope: r.load_function(if T::HALF {
                "rope_pair_tail_f16"
            } else {
                "rope_pair_tail_rn"
            })?,
            rope_hm: r.load_function("rope_pair_tail_hm")?,
            phases: r.load_function("rope_pair_rows_3d_direct")?,
            modulate: m.load_function(if T::HALF {
                "row_modulate_f16_f32"
            } else {
                "row_modulate"
            })?,
            gated: m.load_function(if T::HALF {
                "row_gated_add_f16_f32"
            } else {
                "row_gated_add"
            })?,
            add_mod: m.load_function(if T::HALF {
                "add_row_f16_f32"
            } else {
                "add_row_bf16_f32"
            })?,
            residual: n.load_function(if T::HALF { "residual_f16" } else { "residual" })?,
            cast: v.load_function(if T::HALF {
                "vision_downcast_f32_to_f16"
            } else {
                "vision_affine_cast_bf16"
            })?,
            upcast: v.load_function(if T::HALF {
                "vision_upcast_f16_to_f32"
            } else {
                "vision_upcast_f32"
            })?,
            bias: v.load_function("vision_bias_f32")?,
            silu: v.load_function("vision_add_row_silu_f32")?,
            sinusoid: v.load_function("sinusoidal_embed_f32")?,
            gelu: q.load_function(if T::HALF { "q38_gelu_f16" } else { "q38_gelu" })?,
            transpose: q.load_function("transpose_f32")?,
            concat: q.load_function("concat_rows_f32")?,
        })
    }
    fn prepare_qkv(
        &self,
        d: &Device,
        inputs: [u64; 4],
        outputs: [u64; 3],
        rows: usize,
    ) -> Result<()> {
        ensure!(
            self.half && self.head_dim == 96,
            "F16 QKV preparation requires D96"
        );
        let [packed, rotary, q_gamma, k_gamma] = inputs;
        let [q, k, v] = outputs;
        let count = rows
            .checked_mul(self.heads)
            .context("QKV head count overflow")?;
        let rows = i32::try_from(rows)?;
        let heads = i32::try_from(self.heads)?;
        i32::try_from(count)?;
        let kernel = self
            .qkv_prepare
            .as_ref()
            .context("F16 QKV preparation missing")?;
        unsafe {
            d.stream
                .launch_builder(kernel)
                .arg(&packed)
                .arg(&rotary)
                .arg(&q_gamma)
                .arg(&k_gamma)
                .arg(&q)
                .arg(&k)
                .arg(&v)
                .arg(&rows)
                .arg(&heads)
                .arg(&(96f32).sqrt())
                .arg(&1e-12f32)
                .launch(grid(count.div_ceil(4), 128))?;
        }
        Ok(())
    }
    fn prepare_qkv_separate(
        &self,
        d: &Device,
        inputs: [u64; 4],
        outputs: [u64; 4],
        rows: usize,
    ) -> Result<()> {
        let [packed, rotary, q_gamma, k_gamma] = inputs;
        let [q, key, kn, value] = outputs;
        self.unpack(d, packed, [q, key, value], rows, 3)?;
        if self.half {
            for data in [q, key] {
                unsafe {
                    d.stream
                        .launch_builder(&self.rope)
                        .arg(&data)
                        .arg(&rotary)
                        .arg(&(rows as i32))
                        .arg(&(self.heads as i32))
                        .arg(&(self.head_dim as i32))
                        .arg(&(self.head_dim as i32))
                        .arg(&1i32)
                        .launch(flat(rows * self.heads * self.head_dim / 2))?;
                }
            }
        }
        self.l2(d, q, q, q_gamma, rows, false)?;
        self.l2(d, key, kn, k_gamma, rows, true)
    }
    fn prepare_row_norm(
        &self,
        d: &Device,
        inputs: [u64; 5],
        output: u64,
        shape: [usize; 2],
        site: usize,
    ) -> Result<()> {
        let [rows, width] = shape;
        ensure!(self.half && width == 2112, "F16 row fusion requires H2112");
        let kernel = self
            .row_norm
            .as_ref()
            .and_then(|kernels| kernels.get(site))
            .context("F16 row fusion missing")?;
        let [hidden, projected, modulation, weight, bias] = inputs;
        unsafe {
            d.stream
                .launch_builder(kernel)
                .arg(&hidden)
                .arg(&projected)
                .arg(&modulation)
                .arg(&weight)
                .arg(&bias)
                .arg(&output)
                .arg(&u32::try_from(rows)?)
                .arg(&u32::try_from(width)?)
                .arg(&1e-6f32)
                .launch(grid(rows, 256))?;
        }
        Ok(())
    }
    fn prepare_cross_kv<T: FlowElement>(
        &self,
        d: &Device,
        blas: &Blas,
        linear: Linear<T>,
        input: u64,
        outputs: [u64; 3],
        rows: usize,
    ) -> Result<()> {
        ensure!(
            self.half && T::HALF,
            "context cache projection requires F16"
        );
        linear.run(blas, rows, input, outputs[0])?;
        self.unpack(d, outputs[0], [0, outputs[1], outputs[2]], rows, 2)?;
        #[cfg(test)]
        self.context_chains.set(self.context_chains.get() + 1);
        Ok(())
    }
    pub(crate) fn norm(
        &self,
        d: &Device,
        input: u64,
        output: u64,
        shape: [usize; 2],
        affine: [u64; 2],
        arithmetic: (f32, bool),
    ) -> Result<()> {
        let [rows, width] = shape;
        let (eps, f32_io) = arithmetic;
        unsafe {
            d.stream
                .launch_builder(if f32_io { &self.norm_f32 } else { &self.norm })
                .arg(&input)
                .arg(&affine[0])
                .arg(&affine[1])
                .arg(&output)
                .arg(&(rows as u32))
                .arg(&1u32)
                .arg(&(width as u32))
                .arg(&(width as u64))
                .arg(&0u64)
                .arg(&1u64)
                .arg(&(width as u64))
                .arg(&0u64)
                .arg(&1u64)
                .arg(&eps)
                .launch(grid(rows, 256))?;
        }
        Ok(())
    }
    fn norm_silu(
        &self,
        d: &Device,
        input: u64,
        output: u64,
        shape: [usize; 2],
        affine: [u64; 2],
        modulation: u64,
    ) -> Result<()> {
        let [rows, width] = shape;
        ensure!(
            self.half && rows > 0 && width > 0,
            "invalid F16 norm SiLU shape"
        );
        let rows32 = u32::try_from(rows)?;
        let width32 = u32::try_from(width)?;
        let functions = self
            .norm_silu
            .as_ref()
            .context("F16 norm SiLU kernels missing")?;
        #[cfg(test)]
        if self.norm_silu_reference {
            let separate = self
                .norm_silu_separate
                .as_ref()
                .context("norm SiLU reference missing")?;
            self.norm(d, input, output, shape, affine, (1e-6, false))?;
            unsafe {
                if modulation != 0 {
                    d.stream
                        .launch_builder(&separate[1])
                        .arg(&output)
                        .arg(&(modulation + 2 * width as u64))
                        .arg(&modulation)
                        .arg(&0u64)
                        .arg(&output)
                        .arg(&i32::try_from(rows)?)
                        .arg(&i32::try_from(width)?)
                        .arg(&(2 * width as u64))
                        .arg(&1u32)
                        .launch(crate::backend::gpu::LaunchConfig {
                            grid_dim: (rows32, width32.div_ceil(256), 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }
                d.stream
                    .launch_builder(&separate[0])
                    .arg(&output)
                    .arg(&output)
                    .arg(&u32::try_from(
                        rows.checked_mul(width)
                            .context("norm SiLU extent overflow")?,
                    )?)
                    .launch(flat(rows * width))?;
            }
            return Ok(());
        }
        unsafe {
            let mut launch = d
                .stream
                .launch_builder(&functions[usize::from(modulation != 0)]);
            launch
                .arg(&input)
                .arg(&affine[0])
                .arg(&affine[1])
                .arg(&output)
                .arg(&rows32)
                .arg(&width32)
                .arg(&1e-6f32);
            if modulation != 0 {
                launch.arg(&modulation);
            }
            launch.launch(grid(rows, 256))?;
        }
        Ok(())
    }
    pub(crate) fn cast(&self, d: &Device, input: u64, output: u64, count: usize) -> Result<()> {
        let count32 = u32::try_from(count)?;
        let count64 = count as u64;
        unsafe {
            let mut launch = d.stream.launch_builder(&self.cast);
            launch.arg(&input).arg(&output);
            if self.half {
                launch.arg(&count32);
            } else {
                launch.arg(&count64).arg(&0f32).arg(&1f32);
            }
            launch.launch(flat(count))?;
        }
        Ok(())
    }
    pub(crate) fn upcast(&self, d: &Device, input: u64, output: u64, count: usize) -> Result<()> {
        unsafe {
            d.stream
                .launch_builder(&self.upcast)
                .arg(&input)
                .arg(&output)
                .arg(&u32::try_from(count)?)
                .launch(flat(count))?;
        }
        Ok(())
    }
    pub(crate) fn l2(
        &self,
        d: &Device,
        input: u64,
        output: u64,
        gamma: u64,
        rows: usize,
        head_major: bool,
    ) -> Result<()> {
        unsafe {
            d.stream
                .launch_builder(&self.l2)
                .arg(&input)
                .arg(&gamma)
                .arg(&output)
                .arg(&(rows as i32))
                .arg(&(self.heads as i32))
                .arg(&(self.head_dim as i32))
                .arg(&(self.head_dim as f32).sqrt())
                .arg(&1e-12f32)
                .arg(&(if head_major { rows as i32 } else { 0 }))
                .launch(grid(rows * self.heads, 32))?;
        }
        Ok(())
    }
    pub(crate) fn unpack(
        &self,
        d: &Device,
        input: u64,
        outputs: [u64; 3],
        rows: usize,
        parts: i32,
    ) -> Result<()> {
        unsafe {
            d.stream
                .launch_builder(if parts == 2 {
                    &self.unpack_cross
                } else {
                    &self.unpack
                })
                .arg(&input)
                .arg(&outputs[0])
                .arg(&outputs[1])
                .arg(&outputs[2])
                .arg(&1u32)
                .arg(&(rows as u32))
                .arg(&(self.heads as u32))
                .arg(&(self.head_dim as u32))
                .arg(&parts)
                .launch(flat(rows * self.heads * self.head_dim))?;
        }
        Ok(())
    }
    pub(crate) fn residual(&self, d: &Device, x: u64, y: u64, count: usize) -> Result<()> {
        unsafe {
            d.stream
                .launch_builder(&self.residual)
                .arg(&x)
                .arg(&y)
                .arg(&i32::try_from(count)?)
                .launch(flat(count))?;
        }
        Ok(())
    }
}

fn spatial_phases(
    device: &Device,
    kernels: &Kernels,
    sites: &SparseSites,
) -> Result<CudaSlice<f32>> {
    let rows = sites.coordinates().len();
    let points: Vec<_> = sites
        .coordinates()
        .iter()
        .flat_map(|p| p[1..].iter().map(|&x| x as f32))
        .collect();
    let max = points.iter().copied().fold(0f32, f32::max) as usize;
    let frequencies: Vec<_> = (0..48)
        .map(|i| {
            let f = 1.0 / 10000f32.powf((i % 16) as f32 / 16.0);
            if max < 64 {
                f
            } else {
                let wavelength = f * 64.0 / (2.0 * std::f32::consts::PI);
                let stretch = 64.0 / ((64 + max) / 2) as f32;
                let ratio = (wavelength - 1.0) / 23.0;
                let ratio = if wavelength > 24.0 {
                    1.0
                } else if wavelength < 1.0 {
                    stretch
                } else {
                    ratio + (1.0 - ratio) * stretch
                };
                f * ratio
            }
        })
        .collect();
    let points = device.stream.clone_htod(&points)?;
    let frequencies = device.stream.clone_htod(&frequencies)?;
    let phases = device.stream.alloc_zeros(rows * 96)?;
    unsafe {
        device
            .stream
            .launch_builder(&kernels.phases)
            .arg(&points)
            .arg(&frequencies)
            .arg(&(rows as i32))
            .arg(&1i32)
            .arg(&96i32)
            .arg(&phases)
            .launch(flat(rows * 48))?;
    }
    Ok(phases)
}

struct Work<T: FlowElement> {
    row_index: CudaSlice<u32>,
    hidden: CudaSlice<T>,
    norm: CudaSlice<T>,
    projected: CudaSlice<T>,
    qkv: CudaSlice<T>,
    q: CudaSlice<T>,
    k: CudaSlice<T>,
    v: CudaSlice<T>,
    kn: CudaSlice<T>,
    attention: CudaSlice<T>,
    ff: CudaSlice<T>,
    cross_kv: CudaSlice<T>,
    cross_k: CudaSlice<T>,
    cross_kn: CudaSlice<T>,
    cross_v: CudaSlice<T>,
    cross_q32: CudaSlice<f32>,
    cross_k32: CudaSlice<f32>,
    cross_v32: CudaSlice<f32>,
    cross_out32: CudaSlice<f32>,
    outer: CudaSlice<f32>,
    input_rows: CudaSlice<f32>,
    output_rows: CudaSlice<f32>,
    block_mod: CudaSlice<T::Parameters>,
}
impl<T: FlowElement> Work<T> {
    fn planned_bytes(n: usize, spec: &FlowSpec, context: usize) -> Result<usize> {
        let h = spec.model_channels;
        let row = 4usize
            .checked_add(
                size_of::<T>()
                    .checked_mul(11 * h + spec.feed_forward_width())
                    .context("work row overflow")?,
            )
            .and_then(|v| {
                v.checked_add(if T::HALF {
                    0
                } else {
                    4 * (3 * h + spec.in_channels + spec.out_channels)
                })
            })
            .context("work row bytes overflow")?;
        let fixed = context
            .checked_mul(h)
            .and_then(|n| {
                n.checked_mul(
                    size_of::<T>() * (if T::HALF { 4 } else { 5 }) + if T::HALF { 0 } else { 8 },
                )
            })
            .and_then(|n| n.checked_add(6 * h * size_of::<T::Parameters>()))
            .context("work fixed bytes overflow")?;
        n.checked_mul(row)
            .and_then(|n| n.checked_add(fixed))
            .context("work bytes overflow")
    }
    fn bytes(&self) -> usize {
        [
            self.row_index.num_bytes(),
            self.hidden.num_bytes(),
            self.norm.num_bytes(),
            self.projected.num_bytes(),
            self.qkv.num_bytes(),
            self.q.num_bytes(),
            self.k.num_bytes(),
            self.v.num_bytes(),
            self.kn.num_bytes(),
            self.attention.num_bytes(),
            self.ff.num_bytes(),
            self.cross_kv.num_bytes(),
            self.cross_k.num_bytes(),
            self.cross_kn.num_bytes(),
            self.cross_v.num_bytes(),
            self.cross_q32.num_bytes(),
            self.cross_k32.num_bytes(),
            self.cross_v32.num_bytes(),
            self.cross_out32.num_bytes(),
            self.outer.num_bytes(),
            self.input_rows.num_bytes(),
            self.output_rows.num_bytes(),
            self.block_mod.num_bytes(),
        ]
        .into_iter()
        .sum()
    }
    fn new(d: &Device, n: usize, spec: &FlowSpec, context_rows: usize) -> Result<Self> {
        let s = &d.stream;
        let h = spec.model_channels;
        let planned = Self::planned_bytes(n, spec, context_rows)?;
        let output = Self {
            row_index: s.alloc_zeros(n)?,
            hidden: s.alloc_zeros(n * h)?,
            norm: s.alloc_zeros(n * h)?,
            projected: s.alloc_zeros(n * h)?,
            qkv: s.alloc_zeros(n * 3 * h)?,
            q: s.alloc_zeros(n * h)?,
            k: s.alloc_zeros(n * h)?,
            v: s.alloc_zeros(n * h)?,
            kn: s.alloc_zeros(n * h)?,
            attention: s.alloc_zeros(n * h)?,
            ff: s.alloc_zeros(n * spec.feed_forward_width())?,
            cross_kv: s.alloc_zeros(2 * context_rows * h)?,
            cross_k: s.alloc_zeros(if T::HALF { 0 } else { context_rows * h })?,
            cross_kn: s.alloc_zeros(context_rows * h)?,
            cross_v: s.alloc_zeros(context_rows * h)?,
            cross_q32: s.alloc_zeros(if T::HALF { 0 } else { n * h })?,
            cross_k32: s.alloc_zeros(if T::HALF { 0 } else { context_rows * h })?,
            cross_v32: s.alloc_zeros(if T::HALF { 0 } else { context_rows * h })?,
            cross_out32: s.alloc_zeros(if T::HALF { 0 } else { n * h })?,
            outer: s.alloc_zeros(if T::HALF { 0 } else { n * h })?,
            input_rows: s.alloc_zeros(if T::HALF { 0 } else { n * spec.in_channels })?,
            output_rows: s.alloc_zeros(if T::HALF { 0 } else { n * spec.out_channels })?,
            block_mod: s.alloc_zeros(6 * h)?,
        };
        ensure!(
            output.bytes() == planned,
            "flow work allocation plan mismatch"
        );
        Ok(output)
    }
}

fn validate_conditioning_origin(origin: &Rc<()>, flow: &Rc<()>) -> Result<()> {
    ensure!(
        Rc::ptr_eq(origin, flow),
        "conditioning originates from another flow"
    );
    Ok(())
}

fn conditioning_cache_slot(bindings: &[Option<Rc<()>>; 2], identity: &Rc<()>) -> Result<usize> {
    bindings
        .iter()
        .position(|binding| {
            binding
                .as_ref()
                .is_some_and(|key| Rc::ptr_eq(key, identity))
        })
        .context("conditioning has no prepared context cache binding")
}

fn context_cache_offsets(geometry: [usize; 3], slot: usize, block: usize) -> Result<[usize; 2]> {
    let [blocks, rows, channels] = geometry;
    ensure!(
        blocks > 0 && rows > 0 && channels > 0 && slot < 2 && block < blocks,
        "context cache index exceeds geometry"
    );
    let plane = rows
        .checked_mul(channels)
        .context("context cache plane overflow")?;
    let pair = plane
        .checked_mul(2)
        .context("context cache pair overflow")?;
    blocks
        .checked_mul(2)
        .and_then(|n| n.checked_mul(pair))
        .context("context cache capacity overflow")?;
    let first = (slot * blocks + block) * pair;
    Ok([first, first + plane])
}

pub(crate) fn validate_projection_binding(
    origins: [&Rc<()>; 2],
    coordinates: [&[[i32; 3]]; 2],
    shapes: [crate::backend::PixalConditioningShape; 2],
    views: [&[crate::backend::ProjectionCamera]; 2],
) -> Result<()> {
    validate_conditioning_origin(origins[0], origins[1])?;
    ensure!(
        coordinates[0] == coordinates[1],
        "projected coordinate order differs"
    );
    ensure!(shapes[0] == shapes[1], "projected stage/grid differs");
    ensure!(
        !views[0].is_empty() && views[0] == views[1],
        "projected camera/view mapping differs"
    );
    Ok(())
}

pub struct Conditioning<T: FlowElement = bf16> {
    origin: Rc<()>,
    identity: Rc<()>,
    global: CudaSlice<T>,
    projected: CudaSlice<T>,
    concat: Option<CudaSlice<f32>>,
}
pub struct Flow<T: FlowElement = bf16> {
    origin: Rc<()>,
    device: Device,
    spec: FlowSpec,
    rows: usize,
    coordinates: Vec<[i32; 3]>,
    projection_grid: Option<u32>,
    request: Option<Rc<()>>,
    steps: usize,
    _inner_weights: FlowWeights<T>,
    _f32: FlowWeights<f32>,
    blocks: Vec<Block<T>>,
    input: Linear<f32>,
    output: Linear<f32>,
    time: [Linear<f32>; 3],
    kernels: Kernels,
    blas: Blas,
    flash: Flash,
    cross: CrossAttention,
    work: Work<T>,
    context_cache: Option<CudaSlice<T>>,
    context_bindings: [Option<Rc<()>>; 2],
    #[cfg(test)]
    context_reference: bool,
    #[cfg(test)]
    context_allowance: usize,
    phases: CudaSlice<f32>,
    mods: CudaSlice<T>,
    time_count: usize,
    io: Option<SparseFlowIo>,
    time_embeddings: CudaSlice<T>,
    normalization: Option<[[Vec<f32>; 2]; 2]>,
}

impl Flow<bf16> {
    fn live_bytes(
        spec: &FlowSpec,
        rows: usize,
        steps: usize,
        time_width: usize,
    ) -> Result<[usize; 3]> {
        let h = spec.model_channels;
        let condition_bytes = (if spec.texture() { 2usize } else { 4usize })
            .checked_mul(5 * spec.cond_channels + rows * spec.projection_width())
            .and_then(|n| {
                n.checked_add(if spec.texture() {
                    4 * (rows + 2) * spec.out_channels
                } else {
                    0
                })
            })
            .context("flow conditioning bytes overflow")?;
        let sampler_bytes = rows
            .checked_mul(spec.out_channels)
            .and_then(|n| n.checked_mul((if spec.texture() { 3 } else { 8 }) * size_of::<f32>()))
            .and_then(|n| n.checked_add(8))
            .context("flow sampler bytes overflow")?;
        let time_bytes = size_of::<f32>() * (steps * (time_width + 8 * h + 1) + h + time_width / 2);
        Ok([condition_bytes, sampler_bytes, time_bytes])
    }
    fn planned_bytes(checkpoint: &FlowCheckpoint, rows: usize, steps: usize) -> Result<usize> {
        let spec = &checkpoint.spec;
        let h = spec.model_channels;
        let dim = h / spec.num_heads;
        let tensors = checkpoint.mapped.component_tensors()?;
        let time_width = tensors["t_embedder.mlp.0.weight"].shape[1];
        let live = Self::live_bytes(spec, rows, steps, time_width)?;
        [
            checkpoint.runtime_bytes,
            Work::<bf16>::planned_bytes(rows, spec, 5)?,
            if spec.sparse() { rows * dim * 4 } else { 0 },
            steps * 6 * h * size_of::<bf16>(),
            rows * spec.num_heads * 4,
            F32Attention::required_bytes([
                spec.num_heads as u32,
                u32::try_from(rows)?,
                5,
                dim as u32,
                dim as u32,
            ])?,
            live[0],
            live[1],
            live[2],
            (5 * spec.cond_channels + rows * spec.projection_width()) * 4,
        ]
        .into_iter()
        .try_fold(0usize, |n, v| {
            n.checked_add(v).context("flow peak bytes overflow")
        })
    }
    pub fn open(
        device: &Device,
        dir: &Path,
        stem: &str,
        runtime: &Path,
        geometry: (&[[i32; 3]], u32),
        request: &Rc<()>,
        budget: usize,
    ) -> Result<Self> {
        let (coordinates, grid_side) = geometry;
        let checkpoint = FlowCheckpoint::open(dir, stem)?;
        let spec = checkpoint.spec.clone();
        let pipeline = dir
            .parent()
            .context("checkpoint directory lacks model root")?
            .join(if stem.ends_with("_mv") {
                "pipeline_mv.json"
            } else {
                "pipeline.json"
            });
        let pipeline: serde_json::Value = serde_json::from_slice(&std::fs::read(pipeline)?)?;
        let sampler = &pipeline["args"][if spec.texture() {
            "tex_slat_sampler"
        } else if spec.sparse() {
            "shape_slat_sampler"
        } else {
            "sparse_structure_sampler"
        }];
        let parameters = &sampler["params"];
        let steps = usize::try_from(
            parameters["steps"]
                .as_u64()
                .context("sampler steps missing")?,
        )?;
        ensure!(
            sampler["name"] == "FlowEulerGuidanceIntervalSampler"
                && sampler["args"]["sigma_min"] == 1e-5
                && steps == 12
                && parameters["guidance_strength"] == if spec.texture() { 1.0 } else { 7.5 }
                && parameters["guidance_interval"]
                    == serde_json::json!([0.6, if spec.texture() { 0.9 } else { 1.0 }])
                && parameters["guidance_rescale"]
                    == if spec.texture() {
                        0.0
                    } else if spec.sparse() {
                        0.5
                    } else {
                        0.7
                    }
                && parameters["rescale_t"] == if spec.sparse() { 3.0 } else { 5.0 },
            "unsupported published flow sampler contract"
        );
        let normalization = if spec.texture() {
            let read = |name: &str| -> Result<[Vec<f32>; 2]> {
                let value = &pipeline["args"][name];
                let mean: Vec<f32> = serde_json::from_value(value["mean"].clone())?;
                let std: Vec<f32> = serde_json::from_value(value["std"].clone())?;
                ensure!(
                    mean.len() == spec.out_channels
                        && std.len() == mean.len()
                        && mean.iter().all(|x| x.is_finite())
                        && std.iter().all(|x| x.is_finite() && *x > 0.0),
                    "invalid latent normalization"
                );
                Ok([mean, std])
            };
            Some([
                read("shape_slat_normalization")?,
                read("tex_slat_normalization")?,
            ])
        } else {
            None
        };
        let rows = coordinates.len();
        ensure!(
            rows > 0 && rows <= i32::MAX as usize / (3 * spec.model_channels),
            "flow row extent overflow"
        );
        ensure!(
            matches!(grid_side, 16 | 32 | 64 | 72 | 80 | 88 | 96)
                && coordinates
                    .iter()
                    .flatten()
                    .all(|&v| v >= 0 && v < grid_side as i32),
            "flow coordinates outside projection grid"
        );
        ensure!(
            coordinates
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                == rows,
            "duplicate flow coordinates"
        );
        if !spec.sparse() {
            ensure!(
                grid_side == 16
                    && rows == 4096
                    && coordinates
                        .iter()
                        .enumerate()
                        .all(|(i, p)| *p
                            == [(i / 256) as i32, ((i / 16) % 16) as i32, (i % 16) as i32]),
                "SS coordinates must be the published 16^3 grid order"
            );
        }
        let device = device.clone();
        let (free, total) = device.ctx.mem_get_info()?;
        ensure!(
            budget > 0 && budget <= total,
            "invalid admitted flow device budget"
        );
        let kernels = Kernels::load::<bf16>(&device, &spec)?;
        let ops = Ops::new(&device.ctx)?;
        let mut blas = Blas::new(
            &device,
            &ops,
            runtime,
            Rc::new(Closure::new(true, "Pixal3D flow".into())),
        )?;
        let mut bf = BTreeMap::new();
        let mut fp = BTreeMap::new();
        let tensors = checkpoint.mapped.component_tensors()?;
        for (name, (_, use_bf16)) in spec.inventory() {
            let bytes = tensors[&name].data;
            if use_bf16 {
                let values: Vec<_> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| bf16::from_f32(f32::from_le_bytes(*b)))
                    .collect();
                ensure!(
                    values.iter().all(|v| v.is_finite()),
                    "non-finite weight: {name}"
                );
                bf.insert(name, device.stream.clone_htod(&values)?);
            } else {
                let values: Vec<_> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                ensure!(
                    values.iter().all(|v| v.is_finite()),
                    "non-finite weight: {name}"
                );
                fp.insert(name, device.stream.clone_htod(&values)?);
            }
        }
        let mut blocks = Vec::new();
        let h = spec.model_channels;
        for i in 0..spec.num_blocks {
            let p = format!("blocks.{i}");
            let linear = |name: &str, output, input| {
                Linear::bind(&bf, &device, &format!("{p}.{name}"), [output, input])
            };
            let weight = |name: &str| ptr(&fp[&format!("{p}.{name}")], &device);
            let b = Block {
                qkv: linear("self_attn.to_qkv", 3 * h, h)?,
                self_out: linear("self_attn.to_out", h, h)?,
                cross_q: linear("cross_attn.cross_attn_block.to_q", h, h)?,
                cross_kv: linear(
                    "cross_attn.cross_attn_block.to_kv",
                    2 * h,
                    spec.cond_channels,
                )?,
                cross_out: linear("cross_attn.cross_attn_block.to_out", h, h)?,
                projection: Some(linear(
                    "cross_attn.proj_linear",
                    h,
                    spec.projection_width(),
                )?),
                fc1: linear("mlp.mlp.0", spec.feed_forward_width(), h)?,
                fc2: linear("mlp.mlp.2", h, spec.feed_forward_width())?,
                norm_weight: weight("norm2.weight"),
                norm_bias: weight("norm2.bias"),
                modulation: weight("modulation"),
                gamma: [
                    weight("self_attn.q_rms_norm.gamma"),
                    weight("self_attn.k_rms_norm.gamma"),
                    weight("cross_attn.cross_attn_block.q_rms_norm.gamma"),
                    weight("cross_attn.cross_attn_block.k_rms_norm.gamma"),
                ],
            };
            for l in [
                b.qkv,
                b.self_out,
                b.cross_q,
                b.cross_out,
                b.projection.context("projection missing")?,
                b.fc1,
                b.fc2,
            ] {
                l.prepare(&mut blas, rows)?;
            }
            b.cross_kv.prepare(&mut blas, 5)?;
            blocks.push(b);
        }
        let input = Linear::bind(&fp, &device, "input_layer", [h, spec.in_channels])?;
        let output = Linear::bind(&fp, &device, "out_layer", [spec.out_channels, h])?;
        let time = [
            Linear::bind(
                &fp,
                &device,
                "t_embedder.mlp.0",
                [h, tensors["t_embedder.mlp.0.weight"].shape[1]],
            )?,
            Linear::bind(&fp, &device, "t_embedder.mlp.2", [h, h])?,
            Linear::bind(&fp, &device, "adaLN_modulation.1", [6 * h, h])?,
        ];
        let heads = spec.num_heads;
        let dim = h / heads;
        let phases = if spec.sparse() {
            let values: Vec<_> = coordinates
                .iter()
                .flat_map(|p| p.iter().map(|&x| x as f32))
                .collect();
            let points = device.stream.clone_htod(&values)?;
            let freqs: Vec<f32> = (0..dim / 2)
                .map(|i| {
                    if i >= 3 * (dim / 6) {
                        0.0
                    } else {
                        1.0 / 10000f32.powf((i % (dim / 6)) as f32 / (dim / 6) as f32)
                    }
                })
                .collect();
            let freqs = device.stream.clone_htod(&freqs)?;
            let phase = device.stream.alloc_zeros(rows * dim)?;
            unsafe {
                device
                    .stream
                    .launch_builder(&kernels.phases)
                    .arg(&points)
                    .arg(&freqs)
                    .arg(&(rows as i32))
                    .arg(&1i32)
                    .arg(&(dim as i32))
                    .arg(&phase)
                    .launch(flat(rows * dim / 2))?;
            }
            phase
        } else {
            fp.remove("rope_phases").context("SS phases missing")?
        };
        let work = Work::new(&device, rows, &spec, 5)?;
        let flash = Flash::new(device.stream.clone(), rows, heads, heads, dim)?;
        let mods = device.stream.alloc_zeros(steps * 6 * h)?;
        let [condition_bytes, sampler_bytes, time_bytes] =
            Self::live_bytes(&spec, rows, steps, time[0].weight.shape[1] as usize)?;
        let live_bytes = condition_bytes
            .checked_add(sampler_bytes)
            .and_then(|n| n.checked_add(time_bytes))
            .context("flow live bytes overflow")?;
        let cross_budget = device.workspace_budget(free, budget, live_bytes)?;
        let cross = F32Attention::new(
            device.stream.clone(),
            [heads as u32, rows as u32, 5, dim as u32, dim as u32],
            cross_budget,
        )?;
        device.workspace_budget(free, budget, live_bytes)?;
        if spec.texture() {
            eprintln!(
                "{}",
                serde_json::json!({"texture_setup": {"rows":rows,"runtime_weight_bytes":checkpoint.runtime_bytes,"work_bytes":work.bytes(),"phase_bytes":phases.num_bytes(),"modulation_bytes":mods.num_bytes(),"self_attention_lse_bytes":flash.lse.len()*size_of::<f32>(),"cross_attention_workspace_bytes":cross.workspace_bytes(),"quadratic_score_bytes":0,"conditioning_bound_bytes":condition_bytes,"sampler_bound_bytes":sampler_bytes,"time_bound_bytes":time_bytes,"free_before_bytes":free,"free_after_bytes":device.ctx.mem_get_info()?.0}})
            );
        }
        Ok(Self {
            origin: Rc::new(()),
            device: device.clone(),
            spec,
            rows,
            coordinates: coordinates.to_vec(),
            projection_grid: Some(grid_side),
            request: Some(request.clone()),
            steps,
            _inner_weights: bf,
            _f32: fp,
            blocks,
            input,
            output,
            time,
            kernels,
            blas,
            flash,
            cross: CrossAttention::F32(cross),
            work,
            context_cache: None,
            context_bindings: [None, None],
            #[cfg(test)]
            context_reference: false,
            #[cfg(test)]
            context_allowance: 0,
            phases,
            mods,
            time_count: 0,
            io: None,
            time_embeddings: device.stream.alloc_zeros(0)?,
            normalization,
        })
    }
}

impl Flow<f16> {
    pub fn sample_heun(
        &mut self,
        noise: &[f32],
        positive: &Conditioning<f16>,
        negative: &Conditioning<f16>,
    ) -> Result<Vec<f32>> {
        let rows = self.io.as_ref().context("sparse IO missing")?.rows;
        ensure!(
            noise.len() == rows * self.spec.in_channels,
            "Heun noise extent differs from setup"
        );
        validate_conditioning_origin(&positive.origin, &self.origin)?;
        validate_conditioning_origin(&negative.origin, &self.origin)?;
        let times = heun_schedule(rows, 25)?;
        let model_times: Vec<_> = times
            .iter()
            .flat_map(|&(time, next)| [time * 1000.0, next * 1000.0])
            .collect();
        self.set_times(&model_times)?;
        let mut state = Sampler::new(&self.device, noise, true)?;
        self.prepare_contexts(&[positive, negative])?;
        #[cfg(test)]
        let mut context_defects = 0;
        #[cfg(test)]
        let mut bias_defects = 0;
        #[cfg(test)]
        let mut attention_defects = 0;
        #[cfg(test)]
        let mut gather_defects = 0;
        #[cfg(test)]
        let mut norm_silu_defects = 0;
        let negative_values = state
            .negative
            .as_mut()
            .context("Heun negative buffer missing")?;
        let x0 = state.x0.as_mut().context("Heun predictor buffer missing")?;
        let positive_x0 = state
            .positive_x0
            .as_mut()
            .context("Heun corrector buffer missing")?;
        let update = cubin::module(&self.device.ctx, "flow")?.load_function("heun_update_f32")?;
        for (step, &(time, next)) in times.iter().enumerate() {
            for (stage, t) in [time, next].into_iter().enumerate() {
                let input = if stage == 0 { &state.state } else { &*x0 };
                self.forward(input, 2 * step + stage, positive, &mut state.positive)?;
                #[cfg(test)]
                if step == 0 && stage == 0 {
                    let compact = self.device.stream.clone_dtoh(&state.positive)?;
                    tests::flow_norm_silu_identity(
                        self,
                        input,
                        positive,
                        &mut state.positive,
                        &compact,
                        &mut norm_silu_defects,
                    )?;
                    tests::flow_gather_identity(
                        self,
                        input,
                        positive,
                        &mut state.positive,
                        &compact,
                        &mut gather_defects,
                    )?;
                    tests::flow_attention_identity(
                        self,
                        input,
                        positive,
                        &mut state.positive,
                        &compact,
                        &mut attention_defects,
                    )?;
                    tests::flow_context_identity(
                        self,
                        input,
                        positive,
                        &mut state.positive,
                        &compact,
                        &mut context_defects,
                    )?;
                    tests::flow_bias_identity(
                        self,
                        input,
                        positive,
                        &mut state.positive,
                        &compact,
                        &mut bias_defects,
                    )?;
                }
                let output = if stage == 0 {
                    &mut state.velocity
                } else {
                    &mut *positive_x0
                };
                let scale = if t >= 1.0 {
                    4.0
                } else if t >= 0.7 {
                    10.0 * (t * t)
                } else {
                    1.0
                };
                if scale != 1.0 {
                    self.forward(input, 2 * step + stage, negative, negative_values)?;
                    unsafe {
                        self.device
                            .stream
                            .launch_builder(&state.guided)
                            .arg(&0u64)
                            .arg(&state.positive)
                            .arg(&*negative_values)
                            .arg(&scale)
                            .arg(&0f32)
                            .arg(&(noise.len() as i64))
                            .arg(&0i32)
                            .arg(&*output)
                            .launch(flat(noise.len()))?;
                    }
                } else {
                    self.device.stream.memcpy_dtod(&state.positive, output)?;
                }
                let target = if stage == 0 {
                    ptr(x0, &self.device)
                } else {
                    ptr(&state.state, &self.device)
                };
                let second = if stage == 0 {
                    0
                } else {
                    ptr(positive_x0, &self.device)
                };
                unsafe {
                    self.device
                        .stream
                        .launch_builder(&update)
                        .arg(&state.state)
                        .arg(&state.velocity)
                        .arg(&second)
                        .arg(&target)
                        .arg(&(noise.len() as u64))
                        .arg(&(time - next))
                        .launch(flat(noise.len()))?;
                }
            }
        }
        let affine =
            cubin::module(&self.device.ctx, "q38_common")?.load_function("channel_affine_f32")?;
        unsafe {
            self.device
                .stream
                .launch_builder(&affine)
                .arg(&state.state)
                .arg(&self._f32["output_scale"])
                .arg(&self._f32["output_shift"])
                .arg(&state.state)
                .arg(&(rows as u32))
                .arg(&(self.spec.out_channels as u32))
                .launch(flat(noise.len()))?;
        }
        let values = self.device.stream.clone_dtoh(&state.state)?;
        ensure!(
            values.iter().all(|x| x.is_finite()),
            "non-finite Heun output"
        );
        #[cfg(test)]
        {
            println!(
                "{} F16_CONTEXT_SAMPLER_IDENTITY input_rows={rows} parent_rows={} enabled={} defects={context_defects} threshold_bits=0 timing=false",
                if context_defects == 0 {
                    "PASS"
                } else {
                    "DEFECT"
                },
                self.rows,
                self.context_cache.is_some()
            );
            ensure!(context_defects == 0, "sampler context cache defects");
            println!(
                "{} F16_PRODUCER_BIAS_SAMPLER_IDENTITY input_rows={rows} parent_rows={} enabled={} defects={bias_defects} threshold_bits=0 timing=false",
                if bias_defects == 0 { "PASS" } else { "DEFECT" },
                self.rows,
                self.context_cache.is_some()
            );
            ensure!(bias_defects == 0, "sampler producer bias defects");
            println!(
                "{} F16_Q_REUSE_SAMPLER_IDENTITY input_rows={rows} parent_rows={} enabled={} defects={attention_defects} threshold_bits=0 timing=false",
                if attention_defects == 0 {
                    "PASS"
                } else {
                    "DEFECT"
                },
                self.rows,
                self.context_cache.is_some()
            );
            ensure!(attention_defects == 0, "sampler Q reuse defects");
            println!(
                "{} F16_VECTOR_GATHER_SAMPLER_IDENTITY input_rows={rows} parent_rows={} enabled={} defects={gather_defects} threshold_bits=0 timing=false",
                if gather_defects == 0 {
                    "PASS"
                } else {
                    "DEFECT"
                },
                self.rows,
                self.context_cache.is_some()
            );
            ensure!(gather_defects == 0, "sampler vector gather defects");
            println!(
                "{} F16_NORM_SILU_SAMPLER_IDENTITY input_rows={rows} parent_rows={} defects={norm_silu_defects} threshold_bits=0 timing=false",
                if norm_silu_defects == 0 {
                    "PASS"
                } else {
                    "DEFECT"
                },
                self.rows
            );
            ensure!(norm_silu_defects == 0, "sampler norm SiLU defects");
        }
        Ok(values)
    }

    pub fn open_hi3d(
        device: &Device,
        root: &Path,
        component: &str,
        runtime: &Path,
        coordinates: &[[i32; 3]],
        resolution: usize,
        budget: usize,
    ) -> Result<Self> {
        let checkpoint = Hi3DFlowCheckpoint::open(root, component)?;
        checkpoint.spec.validate_grid(coordinates, resolution)?;
        let spec = checkpoint.spec.flow_spec();
        let sites = SparseSites::new(coordinates.iter().map(|p| [0, p[0], p[1], p[2]]).collect())?;
        let (parent, inverse) = sites.downsample(2)?;
        let rows = parent.coordinates().len();
        ensure!(
            rows <= i32::MAX as usize / (3 * spec.model_channels),
            "flow row extent overflow"
        );
        let device = device.clone();
        device.stream.synchronize()?;
        let (free, total) = device.ctx.mem_get_info()?;
        let h = spec.model_channels;
        let context_rows = 2917;
        let steps = 50;
        let io_rows = coordinates.len();
        let count = |factors: &[usize]| {
            factors.iter().try_fold(1usize, |a, &b| {
                a.checked_mul(b)
                    .context("flow placement byte count overflow")
            })
        };
        let work_bytes = count(&[rows, h, 15, 2])?
            .checked_add(count(&[context_rows, h, 4, 2])?)
            .and_then(|n| {
                n.checked_add(
                    rows * 4 + 6 * h * 4 + rows * 96 * 4 + 2 * rows * 22 * 4 + steps * 7 * h * 2,
                )
            })
            .context("flow work bytes overflow")?;
        let residual_bytes = [
            (io_rows, 256, 256),
            (rows, 256, h),
            (io_rows, 2 * h, 256),
            (io_rows, 512, 256),
        ]
        .into_iter()
        .try_fold(0usize, |sum, (n, input, output)| -> Result<usize> {
            let features = count(&[n, input + (2 + usize::from(input != output)) * output, 2])?;
            let maps = count(&[n, 2, 27, 4])?;
            sum.checked_add(features)
                .and_then(|s| s.checked_add(maps + 2 * output))
                .context("sparse residual bytes overflow")
        })?;
        let io_bytes = count(&[io_rows, 2 * 256, 4])?
            .checked_add(count(&[io_rows, 256 + 2 * h + 512 + 256, 2])?)
            .and_then(|n| {
                n.checked_add(
                    rows * (256 + 2 * h) * 2
                        + 2 * h
                        + (2 * io_rows + rows + 1) * 4
                        + residual_bytes,
                )
            })
            .context("sparse IO work bytes overflow")?;
        device.workspace_budget(
            free,
            budget,
            checkpoint
                .runtime_bytes
                .checked_add(work_bytes)
                .and_then(|n| n.checked_add(io_bytes))
                .context("flow placement bytes overflow")?,
        )?;
        let mut kernels = Kernels::load::<f16>(&device, &spec)?;
        kernels.producer_bias = Some(ProducerBias::new(&device, rows, h)?);
        let ops = Ops::new(&device.ctx)?;
        let mut blas = Blas::new(
            &device,
            &ops,
            runtime,
            Rc::new(Closure::new(true, "sparse flow".into())),
        )?;
        blas.prepare_f16_biased()?;
        let mut half = BTreeMap::new();
        let mut float = BTreeMap::new();
        for (name, tensor) in checkpoint.mapped.component_tensors()? {
            if tensor.dtype == "F16" {
                let values: Vec<_> = tensor
                    .data
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| f16::from_le_bytes(*b))
                    .collect();
                ensure!(
                    values.iter().all(|x| x.is_finite()),
                    "non-finite weight {name}"
                );
                half.insert(name, device.stream.clone_htod(&values)?);
            } else {
                let values: Vec<_> = tensor
                    .data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                ensure!(
                    values.iter().all(|x| x.is_finite()),
                    "non-finite weight {name}"
                );
                float.insert(name, device.stream.clone_htod(&values)?);
            }
        }
        float.insert(
            "output_scale".into(),
            device.stream.clone_htod(&checkpoint.spec.latent_std)?,
        );
        float.insert(
            "output_shift".into(),
            device.stream.clone_htod(&checkpoint.spec.latent_mean)?,
        );
        let mut blocks = Vec::with_capacity(spec.num_blocks);
        for i in 0..spec.num_blocks {
            let p = format!("blocks.{i}");
            let linear = |name: &str, out, input| {
                Linear::bind(&half, &device, &format!("{p}.{name}"), [out, input])
            };
            let parameter = |name: &str| ptr(&float[&format!("{p}.{name}")], &device);
            blocks.push(Block {
                qkv: linear("self_attn.to_qkv", 3 * h, h)?,
                self_out: linear("self_attn.to_out", h, h)?,
                cross_q: linear("cross_attn.to_q", h, h)?,
                cross_kv: linear("cross_attn.to_kv", 2 * h, spec.cond_channels)?,
                cross_out: linear("cross_attn.to_out", h, h)?,
                fc1: linear("mlp.mlp.0", 4 * h, h)?,
                fc2: linear("mlp.mlp.2", h, 4 * h)?,
                projection: None,
                norm_weight: parameter("norm2.weight"),
                norm_bias: parameter("norm2.bias"),
                modulation: parameter("modulation"),
                gamma: [
                    parameter("self_attn.q_rms_norm.gamma"),
                    parameter("self_attn.k_rms_norm.gamma"),
                    0,
                    0,
                ],
            });
        }
        let input = Linear::bind(&float, &device, "input_layer", [256, spec.in_channels])?;
        let output = Linear::bind(&float, &device, "out_layer", [spec.out_channels, 256])?;
        let time = [
            Linear::bind(&float, &device, "t_embedder.mlp.0", [h, 256])?,
            Linear::bind(&float, &device, "t_embedder.mlp.2", [h, h])?,
            Linear::bind(&float, &device, "adaLN_modulation.1", [6 * h, h])?,
        ];
        let phases = spatial_phases(&device, &kernels, &parent)?;
        let mut work = Work::new(&device, rows, &spec, context_rows)?;
        let flash = Flash::new_f16(
            device.stream.clone(),
            rows,
            22,
            22,
            96,
            Score {
                scale: 1.0 / 96f32.sqrt(),
                window_left: None,
                window_right: None,
            },
        )?;
        let cross = Flash::new_f16(
            device.stream.clone(),
            rows,
            22,
            22,
            96,
            Score {
                scale: 1.0 / 96f32.sqrt(),
                window_left: None,
                window_right: None,
            },
        )?;
        let mods = device.stream.alloc_zeros(steps * 6 * h)?;
        let time_embeddings = device.stream.alloc_zeros(steps * h)?;
        let sampler_bytes = count(&[io_rows, spec.in_channels, 6, 4])?
            .checked_add(8)
            .context("sampler bytes overflow")?;
        let conditioning_bytes = count(&[2, context_rows, spec.cond_channels, 2])?;
        let time_bytes = count(&[steps, 256 + 8 * h + 1, 4])?
            .checked_add((h + 128) * 4)
            .context("time workspace overflow")?;
        let live_bytes = io_bytes
            .checked_add(sampler_bytes)
            .and_then(|n| n.checked_add(conditioning_bytes))
            .and_then(|n| n.checked_add(time_bytes))
            .and_then(|n| n.checked_add(total / 16))
            .context("flow live bytes overflow")?;
        let baseline_remaining = device.workspace_budget(free, budget, live_bytes)?;
        let io_shape = [io_rows, rows, h, 256];
        let baseline_partition = SparseFlowIo::gather_partition(io_shape, baseline_remaining)?;
        let cache_elements = count(&[2, spec.num_blocks, 2, context_rows, h])?;
        let cache_bytes = count(&[cache_elements, size_of::<f16>()])?;
        let retired_bytes = work.cross_kn.num_bytes() + work.cross_v.num_bytes();
        let growth = cache_bytes
            .checked_sub(retired_bytes)
            .context("context cache extent differs")?;
        let zero_context_bytes = count(&[context_rows, spec.cond_channels, size_of::<f32>()])?;
        let cache_peak_growth = growth
            .checked_add(zero_context_bytes)
            .context("context cache peak overflow")?;
        let (context_cache, remaining) = if SparseFlowIo::context_cache_fits(
            io_shape,
            baseline_remaining,
            cache_peak_growth,
        )? {
            work.cross_kn = device.stream.alloc_zeros(0)?;
            work.cross_v = device.stream.alloc_zeros(0)?;
            let cache = device.stream.alloc_zeros(cache_elements).with_context(|| {
                format!("F16 context cache allocation failed: need {cache_bytes} bytes, budget {budget}, initial free {free}, measured legacy bank allowance {baseline_remaining}")
            })?;
            let cache_live_bytes = live_bytes
                .checked_add(zero_context_bytes)
                .context("context cache live bytes overflow")?;
            let remaining = device.workspace_budget(free, budget, cache_live_bytes)?;
            ensure!(
                SparseFlowIo::gather_partition(io_shape, remaining)? == baseline_partition,
                "context cache allocation changed sparse partitions; baseline allowance {baseline_remaining}, actual {remaining}, cache {cache_bytes}"
            );
            (Some(cache), remaining)
        } else {
            (None, baseline_remaining)
        };
        let io = SparseFlowIo::new(
            &device,
            &sites,
            &parent,
            &inverse,
            (&half, &float),
            remaining,
        )?;
        let actual_partition = std::array::from_fn(|i| io.blocks[i / 2].conv[i % 2].tile);
        ensure!(
            actual_partition == baseline_partition,
            "constructed sparse partitions differ from setup"
        );
        device.workspace_budget(free, budget, sampler_bytes + conditioning_bytes)?;
        #[cfg(test)]
        {
            let gather_bytes = io.blocks[2].conv[0].workspace_bytes() - io.rows * 27 * 4;
            let work_bytes = work_bytes
                - if context_cache.is_some() {
                    retired_bytes
                } else {
                    0
                };
            println!(
                "SPARSE_FLOW_PLACEMENT published_bytes={} config_affine_bytes={} work_bytes={work_bytes} io_bytes={io_bytes} shared_gather_bytes={gather_bytes} sampler_bytes={sampler_bytes} conditioning_bytes={conditioning_bytes} device_reserve_bytes={} training_grid={} execution_grid={resolution} input_rows={io_rows} parent_rows={rows}",
                checkpoint.runtime_bytes,
                2 * spec.out_channels * 4,
                total / 16,
                checkpoint.spec.resolution
            );
            println!(
                "F16_CONTEXT_CACHE_PLAN enabled={} cache_bytes={} retired_work_bytes={} fixed_growth_bytes={} zero_context_reserve_bytes={} baseline_bank_allowance={baseline_remaining} actual_bank_allowance={remaining} partitions={actual_partition:?} input_rows={io_rows} parent_rows={rows} timing=false",
                context_cache.is_some(),
                context_cache.as_ref().map_or(0, CudaSlice::num_bytes),
                if context_cache.is_some() {
                    retired_bytes
                } else {
                    0
                },
                if context_cache.is_some() { growth } else { 0 },
                if context_cache.is_some() {
                    zero_context_bytes
                } else {
                    0
                }
            );
        }
        Ok(Self {
            origin: Rc::new(()),
            device,
            spec,
            rows,
            coordinates: coordinates.to_vec(),
            projection_grid: None,
            request: None,
            steps,
            _inner_weights: half,
            _f32: float,
            blocks,
            input,
            output,
            time,
            kernels,
            blas,
            flash,
            cross: CrossAttention::F16(cross),
            work,
            context_cache,
            context_bindings: [None, None],
            #[cfg(test)]
            context_reference: false,
            #[cfg(test)]
            context_allowance: baseline_remaining,
            phases,
            mods,
            time_count: 0,
            io: Some(io),
            time_embeddings,
            normalization: None,
        })
    }
}

fn heun_schedule(rows: usize, steps: usize) -> Result<Vec<(f32, f32)>> {
    ensure!(rows > 0 && steps > 0, "invalid Heun schedule extent");
    let warp =
        ((19.0 * rows as f64 - 20000.0) / (3.0 * rows as f64 + 60000.0)).clamp(1.0, 5.0) as f32;
    let rational = |t: f32, scale: f32| scale * t / (1.0 + (scale - 1.0) * t);
    let delta = 1.0 / steps as f32;
    let warped: Vec<_> = (0..=steps)
        .map(|i| {
            let base = if i <= (steps / 2) {
                1.0 - i as f32 * delta
            } else {
                (steps - i) as f32 * delta
            };
            rational(rational(base, 2.0), warp)
        })
        .collect();
    let sigmas: Vec<_> = warped.iter().map(|&t| 0.002 + 0.998 * t).collect();
    Ok((0..steps)
        .map(|i| {
            (
                sigmas[i] * (1.0 + if warped[i] > 0.985 { 0.04 } else { 0.002 }),
                sigmas[i + 1],
            )
        })
        .collect())
}

impl<T: FlowElement> Flow<T> {
    fn linear_f32(
        &self,
        linear: Linear<f32>,
        rows: usize,
        input: u64,
        output: u64,
        silu: bool,
    ) -> Result<()> {
        run_linear_f32(
            &self.device,
            &self.blas,
            &self.kernels,
            linear,
            rows,
            [input, output],
            silu,
        )
    }
    fn set_times(&mut self, times: &[f32]) -> Result<()> {
        ensure!(
            !times.is_empty() && times.len() <= self.steps && times.iter().all(|x| x.is_finite()),
            "invalid flow time setup"
        );
        let d = &self.device;
        let n = times.len();
        let h = self.spec.model_channels;
        let half = self.time[0].weight.shape[1] as usize / 2;
        let t = d.stream.clone_htod(times)?;
        let freq: Vec<_> = (0..half)
            .map(|i| (-10000f64.ln() as f32 * i as f32 / half as f32).exp())
            .collect();
        let freq = d.stream.clone_htod(&freq)?;
        let embed = d.stream.alloc_zeros::<f32>(n * half * 2)?;
        let first = d.stream.alloc_zeros::<f32>(n * h)?;
        let second = d.stream.alloc_zeros::<f32>(n * h)?;
        let all = d.stream.alloc_zeros::<f32>(n * 6 * h)?;
        unsafe {
            d.stream
                .launch_builder(&self.kernels.sinusoid)
                .arg(&t)
                .arg(&freq)
                .arg(&embed)
                .arg(&(n as u32))
                .arg(&(half as u32))
                .arg(&0i32)
                .launch(flat(n * half))?;
        }
        self.linear_f32(self.time[0], n, ptr(&embed, d), ptr(&first, d), true)?;
        self.linear_f32(self.time[1], n, ptr(&first, d), ptr(&second, d), false)?;
        if T::HALF {
            self.kernels
                .cast(d, ptr(&second, d), ptr(&self.time_embeddings, d), n * h)?;
        }
        let zero = d.stream.alloc_zeros::<f32>(h)?;
        unsafe {
            d.stream
                .launch_builder(&self.kernels.silu)
                .arg(&second)
                .arg(&zero)
                .arg(&second)
                .arg(&((n * h) as u32))
                .arg(&(h as u32))
                .launch(flat(n * h))?;
        }
        self.linear_f32(self.time[2], n, ptr(&second, d), ptr(&all, d), false)?;
        self.kernels
            .cast(d, ptr(&all, d), ptr(&self.mods, d), n * 6 * h)?;
        self.time_count = n;
        Ok(())
    }
    fn prepare_contexts(&mut self, contexts: &[&Conditioning<T>]) -> Result<()> {
        self.context_bindings = [None, None];
        ensure!(
            !contexts.is_empty() && contexts.len() <= 2,
            "context cache requires one or two conditions"
        );
        ensure!(
            contexts.len() == 1 || !Rc::ptr_eq(&contexts[0].identity, &contexts[1].identity),
            "context cache conditions must have distinct identities"
        );
        for condition in contexts {
            validate_conditioning_origin(&condition.origin, &self.origin)?;
            ensure!(
                condition.global.context() == &self.device.ctx,
                "context cache device differs"
            );
            ensure!(
                condition.global.len()
                    == (if T::HALF { 2917 } else { 5 }) * self.spec.cond_channels,
                "context cache input extent differs"
            );
        }
        let Some(cache) = &self.context_cache else {
            return Ok(());
        };
        ensure!(T::HALF, "context cache requires F16");
        #[cfg(test)]
        let before = self.kernels.context_chains.get();
        let base = ptr(cache, &self.device);
        for (slot, condition) in contexts.iter().enumerate() {
            for (index, block) in self.blocks.iter().enumerate() {
                let offsets = context_cache_offsets(
                    [self.blocks.len(), 2917, self.spec.model_channels],
                    slot,
                    index,
                )?;
                self.kernels.prepare_cross_kv(
                    &self.device,
                    &self.blas,
                    block.cross_kv,
                    ptr(&condition.global, &self.device),
                    [
                        ptr(&self.work.cross_kv, &self.device),
                        base + (offsets[0] * size_of::<T>()) as u64,
                        base + (offsets[1] * size_of::<T>()) as u64,
                    ],
                    2917,
                )?;
            }
        }
        for (binding, condition) in self.context_bindings.iter_mut().zip(contexts) {
            *binding = Some(condition.identity.clone());
        }
        #[cfg(test)]
        println!(
            "F16_CONTEXT_CACHE_FILL contexts={} blocks={} chains={} setup_only=true",
            contexts.len(),
            self.blocks.len(),
            self.kernels.context_chains.get() - before
        );
        Ok(())
    }
    fn context_slot(&self, condition: &Conditioning<T>) -> Result<Option<usize>> {
        if self.context_cache.is_none() {
            return Ok(None);
        }
        #[cfg(test)]
        if self.context_reference {
            return Ok(None);
        }
        Ok(Some(conditioning_cache_slot(
            &self.context_bindings,
            &condition.identity,
        )?))
    }
    pub fn conditioning(
        &self,
        global: &[f32],
        projected: &[f32],
        coordinates: &[[i32; 3]],
    ) -> Result<Conditioning<T>> {
        ensure!(
            coordinates == self.coordinates,
            "conditioning coordinate rows differ from flow setup"
        );
        ensure!(
            global.len() == (if T::HALF { 2917 } else { 5 }) * self.spec.cond_channels
                && projected.len()
                    == if T::HALF {
                        0
                    } else {
                        self.rows * self.spec.projection_width()
                    },
            "flow conditioning shape mismatch"
        );
        ensure!(
            global.iter().chain(projected).all(|x| x.is_finite()),
            "non-finite conditioning"
        );
        let cast = |values: &[f32]| -> Result<_> {
            let values: Vec<_> = values.iter().map(|&v| T::from_f32(v)).collect();
            ensure!(
                values.iter().all(|x| T::finite(*x)),
                "conditioning overflows flow element type"
            );
            Ok(self.device.stream.clone_htod(&values)?)
        };
        Ok(Conditioning {
            origin: self.origin.clone(),
            identity: Rc::new(()),
            global: cast(global)?,
            projected: cast(projected)?,
            concat: None,
        })
    }
    pub fn conditioning_device(
        &self,
        global: &CudaSlice<f32>,
        coordinates: &[[i32; 3]],
    ) -> Result<Conditioning<T>> {
        ensure!(T::HALF, "device image conditioning requires F16 flow");
        ensure!(
            coordinates == self.coordinates,
            "conditioning coordinate rows differ from flow setup"
        );
        ensure!(
            global.context() == &self.device.ctx,
            "conditioning device context differs from flow setup"
        );
        ensure!(
            global.len() == 2917 * self.spec.cond_channels,
            "device conditioning shape mismatch"
        );
        Ok(Conditioning {
            origin: self.origin.clone(),
            identity: Rc::new(()),
            global: self.cast_conditioning(global)?,
            projected: self.device.stream.alloc_zeros(0)?,
            concat: None,
        })
    }
    pub fn conditioning_projected(
        &self,
        image: &crate::backend::ProjectedConditioning,
        views: &[crate::backend::ProjectionCamera],
    ) -> Result<Conditioning<T>> {
        let (global, projected) = image.binding(
            &self.coordinates,
            crate::backend::PixalConditioningShape {
                image_side: if self.spec.resolution <= 32 {
                    512
                } else {
                    1024
                },
                grid_side: self
                    .projection_grid
                    .context("projected conditioning requires grid setup")?,
                naf_side: self.spec.sparse().then_some(if self.spec.texture() {
                    1024
                } else {
                    512
                }),
            },
            self.request
                .as_ref()
                .context("projected conditioning requires request setup")?,
            views,
        )?;
        ensure!(
            global.len() == 5 * self.spec.cond_channels
                && projected.len() == self.rows * self.spec.projection_width(),
            "device projected conditioning shape mismatch"
        );
        Ok(Conditioning {
            origin: self.origin.clone(),
            identity: Rc::new(()),
            global: self.cast_conditioning(global)?,
            projected: self.cast_conditioning(projected)?,
            concat: None,
        })
    }
    fn cast_conditioning(&self, input: &CudaSlice<f32>) -> Result<CudaSlice<T>> {
        wait_input(&self.device, input)?;
        let mut output = self.device.stream.alloc_zeros(input.len())?;
        let cast = cubin::module(&self.device.ctx, "vision_ops")?.load_function(if T::HALF {
            "vision_downcast_f32_to_f16_checked"
        } else {
            "vision_downcast_f32_to_bf16_checked"
        })?;
        unsafe {
            self.device
                .stream
                .launch_builder(&cast)
                .arg(input)
                .arg(&mut output)
                .arg(&u32::try_from(input.len())?)
                .launch(flat(input.len()))?;
        }
        Ok(output)
    }
    pub fn condition_shape(
        &self,
        condition: &mut Conditioning<T>,
        physical_shape: &[f32],
        coordinates: &[[i32; 3]],
    ) -> Result<()> {
        validate_conditioning_origin(&condition.origin, &self.origin)?;
        ensure!(
            coordinates == self.coordinates,
            "shape condition coordinate rows differ"
        );
        ensure!(
            physical_shape.len() == self.rows * self.spec.out_channels
                && physical_shape.iter().all(|x| x.is_finite()),
            "invalid shape condition"
        );
        let values = self.device.stream.clone_htod(physical_shape)?;
        self.bind_shape(condition, values)
    }
    pub fn condition_shape_device(
        &self,
        condition: &mut Conditioning<T>,
        physical_shape: &CudaSlice<f32>,
        coordinates: &[[i32; 3]],
    ) -> Result<()> {
        validate_conditioning_origin(&condition.origin, &self.origin)?;
        ensure!(
            coordinates == self.coordinates
                && physical_shape.len() == self.rows * self.spec.out_channels,
            "shape condition extent/coordinate mismatch"
        );
        let values = self.device.stream.alloc_zeros(physical_shape.len())?;
        finite_copy(&self.device, physical_shape, &values)?;
        self.bind_shape(condition, values)
    }
    fn bind_shape(
        &self,
        condition: &mut Conditioning<T>,
        mut values: CudaSlice<f32>,
    ) -> Result<()> {
        let normalization = self
            .normalization
            .as_ref()
            .context("texture flow required")?;
        channel_transform(
            &self.device,
            &mut values,
            [&normalization[0][0], &normalization[0][1]],
            true,
        )?;
        condition.concat = Some(values);
        Ok(())
    }
    fn block<const CONTROL: bool>(
        &mut self,
        index: usize,
        modulation: u64,
        conditioning: &Conditioning<T>,
        cache_slot: Option<usize>,
    ) -> Result<()> {
        let d = &self.device;
        let n = self.rows;
        let h = self.spec.model_channels;
        let heads = self.spec.num_heads;
        let dim = h / heads;
        let ff = self.spec.feed_forward_width();
        let b = &self.blocks[index];
        let k = &self.kernels;
        let hidden = ptr(&self.work.hidden, d);
        let norm = ptr(&self.work.norm, d);
        let projected = ptr(&self.work.projected, d);
        let q = ptr(&self.work.q, d);
        let key = ptr(&self.work.k, d);
        let value = ptr(&self.work.v, d);
        let kn = ptr(&self.work.kn, d);
        let attention = ptr(&self.work.attention, d);
        let bm = ptr(&self.work.block_mod, d);
        unsafe {
            d.stream
                .launch_builder(&k.add_mod)
                .arg(&modulation)
                .arg(&b.modulation)
                .arg(&bm)
                .arg(&((6 * h) as u64))
                .arg(&((6 * h) as u32))
                .launch(flat(6 * h))?;
        }
        let row_fused = T::HALF;
        #[cfg(test)]
        let row_fused = row_fused && !k.row_norm_reference;
        let producer_bias = if T::HALF {
            Some(
                k.producer_bias
                    .as_ref()
                    .context("F16 producer bias not prepared")?,
            )
        } else {
            None
        };
        #[cfg(test)]
        let producer_bias = producer_bias.filter(|_| !k.bias_reference && row_fused);
        if let Some(preparation) = producer_bias {
            preparation.row(
                d,
                [hidden, 0, bm, 0, 0],
                [norm, ptr(&self.work.qkv, d), b.qkv.bias.ptr],
                0,
            )?;
        } else if row_fused {
            k.prepare_row_norm(d, [hidden, 0, bm, 0, 0], norm, [n, h], 0)?;
        } else {
            k.norm(d, hidden, norm, [n, h], [0, 0], (1e-6, false))?;
            self.modulate(norm, bm, 0)?;
        }
        if producer_bias.is_some() {
            b.qkv
                .run_prefilled(&self.blas, n, norm, ptr(&self.work.qkv, d))?;
        } else {
            b.qkv.run(&self.blas, n, norm, ptr(&self.work.qkv, d))?;
        }
        let fused = T::HALF;
        #[cfg(test)]
        let fused = fused && !k.qkv_reference;
        if fused {
            k.prepare_qkv(
                d,
                [
                    ptr(&self.work.qkv, d),
                    ptr(&self.phases, d),
                    b.gamma[0],
                    b.gamma[1],
                ],
                [q, kn, value],
                n,
            )?;
        } else {
            k.prepare_qkv_separate(
                d,
                [
                    ptr(&self.work.qkv, d),
                    ptr(&self.phases, d),
                    b.gamma[0],
                    b.gamma[1],
                ],
                [q, key, kn, value],
                n,
            )?;
        }
        if !T::HALF {
            for (kernel, data) in [(&k.rope, q), (&k.rope_hm, kn)] {
                unsafe {
                    d.stream
                        .launch_builder(kernel)
                        .arg(&data)
                        .arg(&self.phases)
                        .arg(&(n as i32))
                        .arg(&(heads as i32))
                        .arg(&(dim as i32))
                        .arg(&(dim as i32))
                        .launch(flat(n * heads * dim / 2))?;
                }
            }
        }
        if CONTROL {
            zero_attention_tail(d, [q, kn, value], [n, n, n], h)?;
        }
        T::attention(
            &mut self.flash,
            &self.work.q,
            &self.work.kn,
            &self.work.v,
            &mut self.work.attention,
            Shape {
                rows: n,
                start: 0,
                capacity: n,
                q_heads: heads,
                kv_heads: heads,
                head_dim: dim,
                causal: false,
                window_left: None,
            },
            [0, 0],
        )?;
        b.self_out.run(&self.blas, n, attention, projected)?;
        if let Some(preparation) = producer_bias {
            preparation.row(
                d,
                [
                    hidden,
                    projected,
                    bm + (2 * h * 4) as u64,
                    b.norm_weight,
                    b.norm_bias,
                ],
                [norm, q, b.cross_q.bias.ptr],
                1,
            )?;
        } else if row_fused {
            k.prepare_row_norm(
                d,
                [
                    hidden,
                    projected,
                    bm + (2 * h * 4) as u64,
                    b.norm_weight,
                    b.norm_bias,
                ],
                norm,
                [n, h],
                1,
            )?;
        } else {
            self.gated_add(projected, bm, 2)?;
            k.norm(
                d,
                hidden,
                norm,
                [n, h],
                [b.norm_weight, b.norm_bias],
                (1e-6, false),
            )?;
        }
        if producer_bias.is_some() {
            b.cross_q.run_prefilled(&self.blas, n, norm, q)?;
        } else {
            b.cross_q.run(&self.blas, n, norm, q)?;
        }
        if T::HALF {
            let context_rows = conditioning.global.len() / self.spec.cond_channels;
            let (keys, values, offsets) = if let Some(slot) = cache_slot {
                let cache = self
                    .context_cache
                    .as_ref()
                    .context("context cache missing")?;
                let offsets =
                    context_cache_offsets([self.blocks.len(), context_rows, h], slot, index)?;
                ensure!(
                    offsets[1] + context_rows * h <= cache.len(),
                    "context cache slot exceeds capacity"
                );
                (cache, cache, offsets)
            } else {
                b.cross_kv.run(
                    &self.blas,
                    context_rows,
                    ptr(&conditioning.global, d),
                    ptr(&self.work.cross_kv, d),
                )?;
                k.unpack(
                    d,
                    ptr(&self.work.cross_kv, d),
                    [0, ptr(&self.work.cross_kn, d), ptr(&self.work.cross_v, d)],
                    context_rows,
                    2,
                )?;
                #[cfg(test)]
                k.context_chains.set(k.context_chains.get() + 1);
                (&self.work.cross_kn, &self.work.cross_v, [0, 0])
            };
            let CrossAttention::F16(cross) = &mut self.cross else {
                anyhow::bail!("F16 cross binding missing");
            };
            if CONTROL {
                zero_attention_tail(
                    d,
                    [
                        q,
                        ptr(keys, d) + (offsets[0] * size_of::<T>()) as u64,
                        ptr(values, d) + (offsets[1] * size_of::<T>()) as u64,
                    ],
                    [n, context_rows, context_rows],
                    h,
                )?;
            }
            T::attention(
                cross,
                &self.work.q,
                keys,
                values,
                &mut self.work.attention,
                Shape {
                    rows: n,
                    start: 0,
                    capacity: context_rows,
                    q_heads: heads,
                    kv_heads: heads,
                    head_dim: dim,
                    causal: false,
                    window_left: None,
                },
                offsets,
            )?;
            b.cross_out.run(&self.blas, n, attention, projected)?;
            if !row_fused {
                k.residual(d, hidden, projected, n * h)?;
            }
        } else {
            b.cross_kv.run(
                &self.blas,
                5,
                ptr(&conditioning.global, d),
                ptr(&self.work.cross_kv, d),
            )?;
            k.unpack(
                d,
                ptr(&self.work.cross_kv, d),
                [0, ptr(&self.work.cross_k, d), ptr(&self.work.cross_v, d)],
                5,
                2,
            )?;
            k.l2(d, q, q, b.gamma[2], n, false)?;
            k.l2(
                d,
                ptr(&self.work.cross_k, d),
                ptr(&self.work.cross_kn, d),
                b.gamma[3],
                5,
                true,
            )?;
            k.upcast(d, q, ptr(&self.work.cross_q32, d), n * h)?;
            k.upcast(
                d,
                ptr(&self.work.cross_kn, d),
                ptr(&self.work.cross_k32, d),
                5 * h,
            )?;
            k.upcast(
                d,
                ptr(&self.work.cross_v, d),
                ptr(&self.work.cross_v32, d),
                5 * h,
            )?;
            let q32 = view(
                &self.work.cross_q32,
                &d.stream,
                0,
                [heads as u32, n as u32, dim as u32],
                [dim as u64, h as u64, 1],
            )?;
            let k32 = dense(&self.work.cross_k32, d, [heads as u32, 5, dim as u32])?;
            let v32 = dense(&self.work.cross_v32, d, [heads as u32, 5, dim as u32])?;
            let out32 = view(
                &self.work.cross_out32,
                &d.stream,
                0,
                [heads as u32, n as u32, dim as u32],
                [dim as u64, h as u64, 1],
            )?;
            let CrossAttention::F32(cross) = &mut self.cross else {
                anyhow::bail!("F32 cross binding missing");
            };
            cross.run(
                &self.blas,
                q32,
                k32,
                v32,
                &mut TensorMut(out32),
                F32Mask {
                    causal: false,
                    valid_keys: 5,
                    query_base: 0,
                    window: 0,
                },
            )?;
            k.cast(d, ptr(&self.work.cross_out32, d), attention, n * h)?;
            b.cross_out.run(&self.blas, n, attention, projected)?;
            b.projection
                .context("projected cross branch missing")?
                .run(&self.blas, n, ptr(&conditioning.projected, d), norm)?;
            k.residual(d, norm, projected, n * h)?;
            k.residual(d, hidden, norm, n * h)?;
        }
        if let Some(preparation) = producer_bias {
            preparation.row(
                d,
                [hidden, projected, bm + (3 * h * 4) as u64, 0, 0],
                [norm, ptr(&self.work.ff, d), b.fc1.bias.ptr],
                2,
            )?;
        } else if row_fused {
            k.prepare_row_norm(
                d,
                [hidden, projected, bm + (3 * h * 4) as u64, 0, 0],
                norm,
                [n, h],
                2,
            )?;
        } else {
            k.norm(d, hidden, norm, [n, h], [0, 0], (1e-6, false))?;
            self.modulate(norm, bm, 3)?;
        }
        if producer_bias.is_some() {
            b.fc1
                .run_prefilled(&self.blas, n, norm, ptr(&self.work.ff, d))?;
        } else {
            b.fc1.run(&self.blas, n, norm, ptr(&self.work.ff, d))?;
        }
        if let Some(preparation) = producer_bias {
            preparation.gelu(d, ptr(&self.work.ff, d), projected, b.fc2.bias.ptr)?;
        } else {
            unsafe {
                d.stream
                    .launch_builder(&k.gelu)
                    .arg(&self.work.ff)
                    .arg(&self.work.ff)
                    .arg(&(n as u32))
                    .arg(&1u32)
                    .arg(&(ff as u32))
                    .arg(&(ff as u64))
                    .arg(&0u64)
                    .arg(&1u64)
                    .arg(&(ff as u64))
                    .arg(&0u64)
                    .arg(&1u64)
                    .arg(&1i32)
                    .launch(grid((n * ff).div_ceil(256).min(65535), 256))?;
            }
        }
        if producer_bias.is_some() {
            b.fc2
                .run_prefilled(&self.blas, n, ptr(&self.work.ff, d), projected)?;
        } else {
            b.fc2.run(&self.blas, n, ptr(&self.work.ff, d), projected)?;
        }
        self.gated_add(projected, bm, 5)?;
        Ok(())
    }
    fn modulate(&self, values: u64, modulation: u64, part: usize) -> Result<()> {
        let h = self.spec.model_channels;
        let shift = modulation + (part * h * size_of::<T::Parameters>()) as u64;
        let scale = shift + (h * size_of::<T::Parameters>()) as u64;
        unsafe {
            self.device
                .stream
                .launch_builder(&self.kernels.modulate)
                .arg(&values)
                .arg(&shift)
                .arg(&scale)
                .arg(&self.work.row_index)
                .arg(&values)
                .arg(&(self.rows as i32))
                .arg(&(h as i32))
                .arg(&((6 * h) as u64))
                .arg(&1u32)
                .launch(crate::backend::gpu::LaunchConfig {
                    grid_dim: (self.rows as u32, h.div_ceil(256) as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }
    fn gated_add(&self, values: u64, modulation: u64, part: usize) -> Result<()> {
        let h = self.spec.model_channels;
        let gate = modulation + (part * h * size_of::<T::Parameters>()) as u64;
        unsafe {
            self.device
                .stream
                .launch_builder(&self.kernels.gated)
                .arg(&self.work.hidden)
                .arg(&values)
                .arg(&gate)
                .arg(&self.work.row_index)
                .arg(&(self.rows as i32))
                .arg(&(h as i32))
                .arg(&((6 * h) as u64))
                .arg(&1u32)
                .launch(crate::backend::gpu::LaunchConfig {
                    grid_dim: (self.rows as u32, h.div_ceil(256) as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }
    fn forward(
        &mut self,
        input: &CudaSlice<f32>,
        time: usize,
        conditioning: &Conditioning<T>,
        output: &mut CudaSlice<f32>,
    ) -> Result<()> {
        self.forward_observe(input, time, conditioning, output, |_, _| Ok(()))
    }
    fn forward_observe(
        &mut self,
        input: &CudaSlice<f32>,
        time: usize,
        conditioning: &Conditioning<T>,
        output: &mut CudaSlice<f32>,
        mut observe: impl FnMut(&Self, usize) -> Result<()>,
    ) -> Result<()> {
        ensure!(
            time < self.time_count
                && input.len()
                    == self.io.as_ref().map_or(self.rows, |io| io.rows) * self.spec.out_channels
                && output.len() == input.len(),
            "flow input/time extent differs from setup"
        );
        validate_conditioning_origin(&conditioning.origin, &self.origin)?;
        let cache_slot = self.context_slot(conditioning)?;
        #[cfg(test)]
        let context_chains = self.kernels.context_chains.get();
        ensure!(
            conditioning.global.len() == (if T::HALF { 2917 } else { 5 }) * self.spec.cond_channels
                && conditioning.projected.len()
                    == if T::HALF {
                        0
                    } else {
                        self.rows * self.spec.projection_width()
                    },
            "conditioning belongs to different flow geometry"
        );
        let n = self.rows;
        let h = self.spec.model_channels;
        if let Some(io) = &mut self.io {
            let d = &self.device;
            let embedding = ptr(&self.time_embeddings, d) + (time * h * 2) as u64;
            io.encode(
                d,
                &self.blas,
                &self.kernels,
                self.input,
                [ptr(input, d), embedding, ptr(&self.work.hidden, d)],
            )?;
        } else {
            let input_ptr = if self.spec.texture() {
                let concat = conditioning
                    .concat
                    .as_ref()
                    .context("texture shape condition missing")?;
                ensure!(concat.len() == input.len(), "shape concat extent mismatch");
                unsafe {
                    self.device
                        .stream
                        .launch_builder(&self.kernels.concat)
                        .arg(input)
                        .arg(concat)
                        .arg(&self.work.input_rows)
                        .arg(&(n as i64))
                        .arg(&(self.spec.out_channels as i32))
                        .arg(&(self.spec.out_channels as i32))
                        .launch(flat(n * self.spec.in_channels))?;
                }
                ptr(&self.work.input_rows, &self.device)
            } else if self.spec.sparse() {
                ptr(input, &self.device)
            } else {
                unsafe {
                    self.device
                        .stream
                        .launch_builder(&self.kernels.transpose)
                        .arg(input)
                        .arg(&self.work.input_rows)
                        .arg(&(self.spec.in_channels as i64))
                        .arg(&(n as i64))
                        .launch(flat(input.len()))?;
                }
                ptr(&self.work.input_rows, &self.device)
            };
            self.linear_f32(
                self.input,
                n,
                input_ptr,
                ptr(&self.work.outer, &self.device),
                false,
            )?;
            self.kernels.cast(
                &self.device,
                ptr(&self.work.outer, &self.device),
                ptr(&self.work.hidden, &self.device),
                n * h,
            )?;
        }
        let modulation = ptr(&self.mods, &self.device) + (time * 6 * h * 2) as u64;
        for i in 0..self.blocks.len() {
            self.block::<false>(i, modulation, conditioning, cache_slot)?;
            observe(self, i)?;
        }
        #[cfg(test)]
        if T::HALF {
            let chains = self.kernels.context_chains.get() - context_chains;
            ensure!(
                chains
                    == if cache_slot.is_some() {
                        0
                    } else {
                        self.blocks.len()
                    },
                "flow context projection count differs from prepared mode"
            );
            println!(
                "F16_CONTEXT_FORWARD enabled={} chains={chains} blocks={} setup_only=false",
                cache_slot.is_some(),
                self.blocks.len()
            );
        }
        if let Some(io) = &mut self.io {
            let d = &self.device;
            return io.decode(
                d,
                &self.blas,
                &self.kernels,
                self.output,
                ptr(&self.work.hidden, d),
                ptr(output, d),
            );
        }
        self.kernels.upcast(
            &self.device,
            ptr(&self.work.hidden, &self.device),
            ptr(&self.work.outer, &self.device),
            n * h,
        )?;
        self.kernels.norm(
            &self.device,
            ptr(&self.work.outer, &self.device),
            ptr(&self.work.outer, &self.device),
            [n, h],
            [0, 0],
            (1e-5, true),
        )?;
        let out = if self.spec.sparse() {
            ptr(output, &self.device)
        } else {
            ptr(&self.work.output_rows, &self.device)
        };
        self.linear_f32(
            self.output,
            n,
            ptr(&self.work.outer, &self.device),
            out,
            false,
        )?;
        if !self.spec.sparse() {
            unsafe {
                self.device
                    .stream
                    .launch_builder(&self.kernels.transpose)
                    .arg(&out)
                    .arg(&*output)
                    .arg(&(n as i64))
                    .arg(&(self.spec.out_channels as i64))
                    .launch(flat(output.len()))?;
            }
        }
        Ok(())
    }
}

pub(crate) struct Sampler {
    pub(crate) state: CudaSlice<f32>,
    pub(crate) positive: CudaSlice<f32>,
    pub(crate) negative: Option<CudaSlice<f32>>,
    velocity: CudaSlice<f32>,
    x0: Option<CudaSlice<f32>>,
    positive_x0: Option<CudaSlice<f32>>,
    std: Option<CudaSlice<f32>>,
    guided: CudaFunction,
    convert: CudaFunction,
    std_kernel: CudaFunction,
    rescale: CudaFunction,
    euler: CudaFunction,
}
impl Sampler {
    pub(crate) fn new(d: &Device, noise: &[f32], guided: bool) -> Result<Self> {
        ensure!(
            noise.len() > 1 && noise.iter().all(|x| x.is_finite()),
            "invalid flow noise"
        );
        Self::from_state(d, d.stream.clone_htod(noise)?, guided)
    }
    fn from_state(d: &Device, state: CudaSlice<f32>, guided: bool) -> Result<Self> {
        let count = state.len();
        ensure!(count > 1, "invalid flow state extent");
        wait_input(d, &state)?;
        let common = cubin::module(&d.ctx, "q38_common")?;
        let flow = cubin::module(&d.ctx, "flow")?;
        let guided_buffer = |n| -> Result<Option<CudaSlice<f32>>> {
            if guided {
                Ok(Some(d.stream.alloc_zeros(n)?))
            } else {
                Ok(None)
            }
        };
        Ok(Self {
            state,
            positive: d.stream.alloc_zeros(count)?,
            negative: guided_buffer(count)?,
            velocity: d.stream.alloc_zeros(count)?,
            x0: guided_buffer(count)?,
            positive_x0: guided_buffer(count)?,
            std: guided_buffer(2)?,
            guided: common.load_function("guided_euler_f32")?,
            convert: flow.load_function("flow_x0_f32")?,
            std_kernel: cubin::module(&d.ctx, "norm")?.load_function("sample_std_f32")?,
            rescale: flow.load_function("flow_rescale_f32")?,
            euler: flow.load_function("euler_blend_f32")?,
        })
    }
    fn convert(&self, d: &Device, input: u64, output: u64, t: f64, inverse: bool) -> Result<()> {
        let alpha = (1.0f64 - 1e-5) as f32;
        let beta = (1e-5 + (1.0 - 1e-5) * t) as f32;
        unsafe {
            d.stream
                .launch_builder(&self.convert)
                .arg(&self.state)
                .arg(&input)
                .arg(&output)
                .arg(&(self.state.len() as u64))
                .arg(&alpha)
                .arg(&beta)
                .arg(&(inverse as i32))
                .launch(flat(self.state.len()))?;
        }
        Ok(())
    }
    pub(crate) fn step(&mut self, d: &Device, t: f64, next: f64, sparse: bool) -> Result<()> {
        ensure!(
            t.is_finite() && next.is_finite() && next >= 0.0 && next < t && t <= 1.0,
            "invalid Euler time pair"
        );
        let count = self.state.len();
        if let Some(negative) = self.negative.as_ref().filter(|_| (0.6..=1.0).contains(&t)) {
            let x0 = self.x0.as_ref().context("CFG x0 buffer missing")?;
            let positive_x0 = self
                .positive_x0
                .as_ref()
                .context("CFG positive x0 buffer missing")?;
            let std = self.std.as_ref().context("CFG std buffer missing")?;
            unsafe {
                d.stream
                    .launch_builder(&self.guided)
                    .arg(&0u64)
                    .arg(&self.positive)
                    .arg(negative)
                    .arg(&7.5f32)
                    .arg(&0f32)
                    .arg(&(count as i64))
                    .arg(&1i32)
                    .arg(&self.velocity)
                    .launch(flat(count))?;
            }
            self.convert(d, ptr(&self.positive, d), ptr(positive_x0, d), t, false)?;
            self.convert(d, ptr(&self.velocity, d), ptr(x0, d), t, false)?;
            for (i, input) in [positive_x0, x0].into_iter().enumerate() {
                let output = ptr(std, d) + (i * 4) as u64;
                unsafe {
                    d.stream
                        .launch_builder(&self.std_kernel)
                        .arg(input)
                        .arg(&output)
                        .arg(&(count as u64))
                        .arg(&(sparse as i32))
                        .launch(grid(1, 256))?;
                }
            }
            let std_pos = ptr(std, d);
            let std_cfg = std_pos + 4;
            unsafe {
                d.stream
                    .launch_builder(&self.rescale)
                    .arg(x0)
                    .arg(&std_pos)
                    .arg(&std_cfg)
                    .arg(&(count as u64))
                    .arg(&(if sparse { 0.5f32 } else { 0.7f32 }))
                    .launch(flat(count))?;
            }
            self.convert(d, ptr(x0, d), ptr(&self.velocity, d), t, true)?;
        } else {
            d.stream.memcpy_dtod(&self.positive, &mut self.velocity)?;
        }
        unsafe {
            d.stream
                .launch_builder(&self.euler)
                .arg(&self.state)
                .arg(&self.velocity)
                .arg(&(count as u64))
                .arg(&(-(t - next) as f32))
                .arg(&0f32)
                .arg(&1f32)
                .launch(flat(count))?;
        }
        Ok(())
    }
}

pub(crate) fn schedule(sparse: bool, steps: usize) -> Vec<f64> {
    let scale = if sparse { 3.0 } else { 5.0 };
    (0..=steps)
        .map(|i| {
            let t = if i == steps {
                0.0
            } else {
                1.0 + i as f64 * (-1.0 / steps as f64)
            };
            scale * t / (1.0 + (scale - 1.0) * t)
        })
        .collect()
}
impl Flow<bf16> {
    pub fn sample(
        &mut self,
        noise: &[f32],
        positive: &Conditioning,
        negative: Option<&Conditioning>,
    ) -> Result<Vec<f32>> {
        ensure!(noise.iter().all(|x| x.is_finite()), "non-finite flow noise");
        let noise = self.device.stream.clone_htod(noise)?;
        let state = self.sample_state(noise, positive, negative)?;
        Ok(self.device.stream.clone_dtoh(&state)?)
    }
    pub fn sample_device(
        &mut self,
        noise: CudaSlice<f32>,
        positive: &Conditioning,
        negative: Option<&Conditioning>,
    ) -> Result<CudaSlice<f32>> {
        finite_copy(&self.device, &noise, &noise)?;
        self.sample_state(noise, positive, negative)
    }
    fn sample_state(
        &mut self,
        noise: CudaSlice<f32>,
        positive: &Conditioning,
        negative: Option<&Conditioning>,
    ) -> Result<CudaSlice<f32>> {
        validate_conditioning_origin(&positive.origin, &self.origin)?;
        ensure!(
            negative.is_none() == self.spec.texture(),
            "sampler conditioning branch mismatch"
        );
        if let Some(negative) = negative {
            validate_conditioning_origin(&negative.origin, &self.origin)?;
        }
        ensure!(
            noise.len() == self.rows * self.spec.out_channels,
            "noise extent differs from flow geometry"
        );
        let times = schedule(self.spec.sparse(), self.steps);
        let model_times: Vec<_> = times[..self.steps]
            .iter()
            .map(|t| (1000.0 * t) as f32)
            .collect();
        self.set_times(&model_times)?;
        let mut sampler = Sampler::from_state(&self.device, noise, !self.spec.texture())?;
        #[cfg(test)]
        let mut calls = [0usize; 2];
        for step in 0..self.steps {
            self.forward(&sampler.state, step, positive, &mut sampler.positive)?;
            #[cfg(test)]
            {
                calls[0] += 1;
            }
            if let Some(negative) = negative.filter(|_| (0.6..=1.0).contains(&times[step])) {
                self.forward(
                    &sampler.state,
                    step,
                    negative,
                    sampler.negative.as_mut().context("CFG buffer missing")?,
                )?;
                #[cfg(test)]
                {
                    calls[1] += 1;
                }
            }
            sampler.step(
                &self.device,
                times[step],
                times[step + 1],
                self.spec.sparse(),
            )?;
        }
        #[cfg(test)]
        flow_call_contract(&self.spec, calls)?;
        if let Some(normalization) = &self.normalization {
            #[cfg(test)]
            let normalized = self.device.stream.clone_dtoh(&sampler.state)?;
            channel_transform(
                &self.device,
                &mut sampler.state,
                [&normalization[1][0], &normalization[1][1]],
                false,
            )?;
            #[cfg(test)]
            transform_contract(
                &self.device,
                &normalized,
                &sampler.state,
                [&normalization[1][0], &normalization[1][1]],
            )?;
        }
        Ok(sampler.state)
    }
}

#[cfg(test)]
fn flow_call_contract(spec: &FlowSpec, calls: [usize; 2]) -> Result<()> {
    let expected = if spec.texture() {
        [12, 0]
    } else if spec.sparse() {
        [12, 9]
    } else {
        [12, 10]
    };
    ensure!(
        calls == expected,
        "flow forward branch count mismatch: {calls:?} expected {expected:?}"
    );
    println!(
        "P1_M4C_FORWARD PASS sparse={} texture={} positive={} negative={}",
        spec.sparse(),
        spec.texture(),
        calls[0],
        calls[1]
    );
    Ok(())
}
#[cfg(test)]
fn decoder_pass_contract(passes: usize) -> Result<()> {
    ensure!(passes == 1, "decoder executed {passes} complete passes");
    println!("P1_M4C_DECODE_PASS PASS count={passes}");
    Ok(())
}
#[cfg(test)]
fn transform_contract(
    device: &Device,
    before: &[f32],
    after: &CudaSlice<f32>,
    statistics: [&[f32]; 2],
) -> Result<()> {
    let [mean, std] = statistics;
    let expected: Vec<_> = before
        .iter()
        .enumerate()
        .map(|(i, &v)| v * std[i % std.len()] + mean[i % mean.len()])
        .collect();
    ensure!(
        device.stream.clone_dtoh(after)? == expected,
        "latent transform was omitted, repeated or used different coefficients"
    );
    println!(
        "P1_M4C_TRANSFORM PASS values={} applications=1 exact=true",
        expected.len()
    );
    Ok(())
}

fn channel_transform(
    d: &Device,
    values: &mut CudaSlice<f32>,
    normalization: [&[f32]; 2],
    inverse: bool,
) -> Result<()> {
    let [mean, std] = normalization;
    let channels = mean.len();
    ensure!(
        channels > 0
            && std.len() == channels
            && !values.is_empty()
            && values.len().is_multiple_of(channels),
        "shape latent normalization extent mismatch"
    );
    ensure!(
        mean.iter().all(|x| x.is_finite()) && std.iter().all(|x| x.is_finite() && *x > 0.0),
        "invalid shape normalization coefficients"
    );
    let mean = d.stream.clone_htod(mean)?;
    let std = d.stream.clone_htod(std)?;
    let kernel = cubin::module(&d.ctx, "q38_common")?.load_function(if inverse {
        "channel_affine_inverse_f32"
    } else {
        "channel_affine_f32"
    })?;
    unsafe {
        d.stream
            .launch_builder(&kernel)
            .arg(&ptr(values, d))
            .arg(&std)
            .arg(&mean)
            .arg(&ptr(values, d))
            .arg(&u32::try_from(values.len() / channels)?)
            .arg(&u32::try_from(channels)?)
            .launch(flat(values.len()))?;
    }
    Ok(())
}

struct Weights {
    half: BTreeMap<String, CudaSlice<f16>>,
    float: BTreeMap<String, CudaSlice<f32>>,
}
impl Weights {
    fn load(d: &Device, checkpoint: &DecoderCheckpoint) -> Result<Self> {
        let mut half = BTreeMap::new();
        let mut float = BTreeMap::new();
        for (name, (_, _, f32_runtime)) in checkpoint.kind.inventory() {
            let t = checkpoint.mapped.tensor(&name)?;
            if t.dtype() == Dtype::I64 {
                continue;
            }
            if f32_runtime {
                let values: Vec<f32> = if t.dtype() == Dtype::F32 {
                    t.data()
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b))
                        .collect()
                } else {
                    t.data()
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| f16::from_le_bytes(*b).to_f32())
                        .collect()
                };
                ensure!(
                    values.iter().all(|x| x.is_finite()),
                    "nonfinite decoder weight {name}"
                );
                float.insert(name, d.stream.clone_htod(&values)?);
            } else {
                let values: Vec<f16> = if t.dtype() == Dtype::F32 {
                    t.data()
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f16::from_f32(f32::from_le_bytes(*b)))
                        .collect()
                } else {
                    t.data()
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| f16::from_le_bytes(*b))
                        .collect()
                };
                ensure!(
                    values.iter().all(|x| x.is_finite()),
                    "nonfinite decoder weight {name}"
                );
                half.insert(name, d.stream.clone_htod(&values)?);
            }
        }
        Ok(Self { half, float })
    }
    fn affine(&self, d: &Device, name: Option<&str>) -> Result<[u64; 2]> {
        match name {
            Some(name) => Ok([
                ptr(
                    self.float
                        .get(&format!("{name}.weight"))
                        .context("norm weight missing")?,
                    d,
                ),
                ptr(
                    self.float
                        .get(&format!("{name}.bias"))
                        .context("norm bias missing")?,
                    d,
                ),
            ]),
            None => Ok([0, 0]),
        }
    }
}
struct DecoderKernels {
    ln16: CudaFunction,
    ln32: CudaFunction,
    ln16_to32: CudaFunction,
    act16: CudaFunction,
    act32: CudaFunction,
    cast: CudaFunction,
    upcast: CudaFunction,
    bias16: CudaFunction,
    fill16: CudaFunction,
    bias32: CudaFunction,
    residual: CudaFunction,
    shuffle: CudaFunction,
    c2s: CudaFunction,
    conv16: CudaFunction,
    conv32: CudaFunction,
    rounded_swish: Option<[CudaFunction; 2]>,
    gelu16: Option<CudaFunction>,
    add32: Option<CudaFunction>,
    add16_to32: Option<CudaFunction>,
    corner_fields: Option<CudaFunction>,
}
impl DecoderKernels {
    fn load(d: &Device, point: bool) -> Result<Self> {
        let q = cubin::module(&d.ctx, "q38_common")?;
        let v = cubin::module(&d.ctx, "vision_ops")?;
        Ok(Self {
            ln16: q.load_function("q38_layer_norm_f16")?,
            ln32: q.load_function("q38_layer_norm_f32")?,
            ln16_to32: q.load_function("q38_layer_norm_f16_to_f32")?,
            act16: v.load_function("activation_f16")?,
            act32: v.load_function("activation_f32")?,
            cast: v.load_function("vision_downcast_f32_to_f16")?,
            upcast: v.load_function("vision_upcast_f16_to_f32")?,
            bias16: q.load_function("q38_bias_add_f16")?,
            fill16: q.load_function("q38_bias_fill_f16")?,
            bias32: v.load_function("vision_bias_f32")?,
            residual: q.load_function("channel_scale_residual_f16")?,
            shuffle: v.load_function("pixel_shuffle_3d_f16")?,
            c2s: v.load_function("channel_to_spatial_f16")?,
            conv16: v.load_function("conv3d_im2col_f16")?,
            conv32: v.load_function("conv3d_im2col_f32")?,
            rounded_swish: if point {
                Some([
                    v.load_function("swish_rounded_f32")?,
                    v.load_function("swish_rounded_f16")?,
                ])
            } else {
                None
            },
            gelu16: if point {
                Some(q.load_function("q38_gelu_f16")?)
            } else {
                None
            },
            add32: if point {
                Some(q.load_function("add_scale_f32")?)
            } else {
                None
            },
            add16_to32: if point {
                Some(v.load_function("residual_add_f16_to_f32")?)
            } else {
                None
            },
            corner_fields: if point {
                Some(v.load_function("corner_attributes_f32")?)
            } else {
                None
            },
        })
    }
    pub(crate) fn norm(
        &self,
        d: &Device,
        io: [u64; 2],
        shape: [usize; 2],
        affine: [u64; 2],
        f32_io: [bool; 2],
        eps: f32,
    ) -> Result<()> {
        let [rows, width] = shape;
        let kernel = match f32_io {
            [false, false] => &self.ln16,
            [true, true] => &self.ln32,
            [false, true] => &self.ln16_to32,
            [true, false] => anyhow::bail!("unsupported decoder normalization dtype pair"),
        };
        unsafe {
            d.stream
                .launch_builder(kernel)
                .arg(&io[0])
                .arg(&affine[0])
                .arg(&affine[1])
                .arg(&io[1])
                .arg(&(rows as u32))
                .arg(&1u32)
                .arg(&(width as u32))
                .arg(&(width as u64))
                .arg(&0u64)
                .arg(&1u64)
                .arg(&(width as u64))
                .arg(&0u64)
                .arg(&1u64)
                .arg(&eps)
                .launch(grid(rows, 256))?;
        }
        Ok(())
    }
    fn act(&self, d: &Device, x: u64, rows: usize, width: usize, f32_io: bool) -> Result<()> {
        unsafe {
            d.stream
                .launch_builder(if f32_io { &self.act32 } else { &self.act16 })
                .arg(&x)
                .arg(&x)
                .arg(&(rows as u32))
                .arg(&(width as u32))
                .arg(&1i32)
                .launch(flat(rows * width))?;
        }
        Ok(())
    }
    pub(crate) fn cast(&self, d: &Device, io: [u64; 2], count: usize, up: bool) -> Result<()> {
        unsafe {
            d.stream
                .launch_builder(if up { &self.upcast } else { &self.cast })
                .arg(&io[0])
                .arg(&io[1])
                .arg(&u32::try_from(count)?)
                .launch(flat(count))?;
        }
        Ok(())
    }
    fn bias(
        &self,
        d: &Device,
        y: u64,
        bias: u64,
        shape: [usize; 2],
        overwrite: bool,
        f32_io: bool,
    ) -> Result<()> {
        let [rows, width] = shape;
        unsafe {
            if f32_io {
                d.stream
                    .launch_builder(&self.bias32)
                    .arg(&y)
                    .arg(&bias)
                    .arg(&y)
                    .arg(&(rows as u32))
                    .arg(&(width as u32))
                    .launch(flat(rows * width))?;
            } else {
                d.stream
                    .launch_builder(if overwrite {
                        &self.fill16
                    } else {
                        &self.bias16
                    })
                    .arg(&y)
                    .arg(&bias)
                    .arg(&((rows * width) as u64))
                    .arg(&(width as u32))
                    .launch(flat(rows * width))?;
            }
        }
        Ok(())
    }
    fn add(
        &self,
        d: &Device,
        x: &CudaSlice<f16>,
        y: &CudaSlice<f16>,
        ones: &CudaSlice<f16>,
        shape: [usize; 2],
    ) -> Result<()> {
        let [rows, width] = shape;
        unsafe {
            d.stream
                .launch_builder(&self.residual)
                .arg(x)
                .arg(y)
                .arg(ones)
                .arg(x)
                .arg(&(rows as u32))
                .arg(&(width as u32))
                .launch(flat(rows * width))?;
        }
        Ok(())
    }
}

/// Published [batch,x,y,z,channel] noise values; the request selects one batch.
pub struct CoordinateNoise<'a> {
    pub values: &'a [f32],
    pub shape: [usize; 5],
    pub batch: usize,
}
impl CoordinateNoise<'_> {
    fn gather(
        &self,
        coordinates: &[[i32; 3]],
        channels: usize,
        channels_first: bool,
    ) -> Result<Vec<f32>> {
        let count = self
            .shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .context("noise extent overflow")?;
        ensure!(
            self.shape.iter().all(|&n| n > 0)
                && count == self.values.len()
                && self.batch < self.shape[0]
                && channels == self.shape[4],
            "noise domain extent mismatch"
        );
        let count = coordinates
            .len()
            .checked_mul(channels)
            .context("selected noise extent overflow")?;
        let mut output = vec![0.0; count];
        for (row, point) in coordinates.iter().enumerate() {
            let mut index = self.batch;
            for (axis, &coordinate) in point.iter().enumerate() {
                let coordinate =
                    usize::try_from(coordinate).context("negative noise coordinate")?;
                ensure!(
                    coordinate < self.shape[axis + 1],
                    "noise coordinate outside supplied domain"
                );
                index = index * self.shape[axis + 1] + coordinate;
            }
            for channel in 0..channels {
                let value = self.values[index * channels + channel];
                ensure!(value.is_finite(), "nonfinite selected noise");
                output[if channels_first {
                    channel * coordinates.len() + row
                } else {
                    row * channels + channel
                }] = value;
            }
        }
        Ok(output)
    }
}

pub struct PixalView<'a> {
    pub rgba: &'a [u8],
    /// [height,width].
    pub dimensions: [usize; 2],
    pub camera: crate::backend::ProjectionCamera,
}

pub struct PixalGeneration<'a> {
    pub model: &'a Path,
    pub dino: &'a Path,
    pub naf: &'a Path,
    pub runtime: &'a Path,
    pub views: &'a [PixalView<'a>],
    pub multiview: bool,
    pub requested_resolution: u32,
    /// Structure, low-resolution shape, high-resolution shape, texture.
    pub noise: [CoordinateNoise<'a>; 4],
}

pub struct Hi3DGeneration<'a> {
    pub model: &'a Path,
    pub dino: &'a Path,
    pub clip: &'a Path,
    pub runtime: &'a Path,
    pub rgba: &'a [u8],
    /// [height,width].
    pub dimensions: [usize; 2],
    pub orientation: u8,
    /// Initial mesh; voxelized before any boundary fill.
    pub mesh: &'a crate::sparse::SparseMesh,
    pub requested_resolution: u32,
    pub seed: u64,
}

impl Hi3DGeneration<'_> {
    /// Final filled, component-filtered, simplified and unit-radius mesh, with its resolution.
    pub fn run_processed(&self, device: &Device) -> Result<(crate::sparse::SparseMesh, u32)> {
        let (mut mesh, resolution) = self.run(device)?;
        let capacity = mesh
            .faces
            .len()
            .checked_mul(4)
            .context("filled mesh capacity overflow")?
            / 3;
        let reserve =
            |vertices: usize, faces: usize, per_vertex: usize, per_face: usize| -> Result<()> {
                let bytes = vertices
                    .checked_mul(per_vertex)
                    .and_then(|v| faces.checked_mul(per_face).and_then(|f| v.checked_add(f)))
                    .context("mesh processing host reservation overflow")?;
                let available = super::calibrate::host_memory(Path::new("/"))?;
                ensure!(
                    bytes as u64 <= available.saturating_sub(available / 16),
                    "mesh processing host reservation exceeds detected available memory"
                );
                Ok(())
            };
        reserve(mesh.vertices.len() / 3, capacity, 160, 48)?;
        mesh.fill_holes(capacity)?;
        reserve(mesh.vertices.len() / 3, mesh.faces.len() / 3, 160, 192)?;
        mesh.filter_components(0.005)?;
        let target = if resolution == 1536 {
            2_000_000
        } else {
            1_000_000
        };
        mesh.simplify(target)?;
        mesh.normalize()?;
        Ok((mesh, resolution))
    }

    fn validate(&self) -> Result<()> {
        let mesh = self.mesh;
        ensure!(
            !mesh.vertices.is_empty()
                && mesh.vertices.len().is_multiple_of(3)
                && !mesh.faces.is_empty()
                && mesh.faces.len().is_multiple_of(3)
                && mesh.vertices.iter().all(|v| v.is_finite())
                && mesh
                    .faces
                    .iter()
                    .all(|&v| v >= 0 && (v as usize) < mesh.vertices.len() / 3),
            "invalid supplied refinement mesh"
        );
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for point in mesh.vertices.as_chunks::<3>().0 {
            for axis in 0..3 {
                lo[axis] = lo[axis].min(point[axis]);
                hi[axis] = hi[axis].max(point[axis]);
            }
        }
        let extent = (0..3).map(|a| hi[a] - lo[a]).fold(0.0f32, f32::max);
        ensure!(
            extent.is_finite() && extent > 0.0,
            "supplied mesh has no finite positive extent"
        );
        ensure!(
            matches!(self.requested_resolution, 512 | 1024 | 1536),
            "unsupported refinement resolution"
        );
        ensure!(
            self.dimensions.iter().all(|&n| n > 0)
                && self.dimensions[0]
                    .checked_mul(self.dimensions[1])
                    .and_then(|n| n.checked_mul(4))
                    == Some(self.rgba.len())
                && (1..=8).contains(&self.orientation)
                && self.rgba.as_chunks::<4>().0.iter().any(|p| p[3] != 0),
            "invalid supplied RGBA image"
        );
        Ok(())
    }

    fn budget(&self, device: &Device) -> Result<usize> {
        let (free, total) = device.ctx.mem_get_info()?;
        let envelope = total - total / 8;
        envelope
            .checked_sub(total - free)
            .context("refinement live data exceeds detected device envelope")
    }

    /// Raw mesh before final fill, filtering, simplification and normalization, with its resolution.
    pub fn run(&self, device: &Device) -> Result<(crate::sparse::SparseMesh, u32)> {
        self.validate()?;
        let ops = Ops::new(&device.ctx)?;
        let mut blas = Blas::new(
            device,
            &ops,
            self.runtime,
            Rc::new(Closure::new(false, "image refinement".into())),
        )?;
        let mut conditioner = crate::backend::Hi3DImageConditioner::load(
            device,
            &mut blas,
            self.dino,
            self.clip,
            self.budget(device)?,
        )?;
        conditioner.encode_rgba(&blas, self.rgba, self.dimensions, self.orientation)?;
        ensure!(
            conditioner.encoder_calls() == [1, 1],
            "refinement encoder call count differs"
        );
        let context = conditioner.into_context()?;
        drop(blas);
        println!(
            "HI3D_GENERATION_MEDIA encoder_calls=1,1 context_rows=2917 context_channels=2048 context_reused_per_rung=true"
        );
        let mut previous: Option<crate::sparse::SparseMesh> = None;
        for (stream, nominal, limit) in [
            (0u64, 64u32, 80_000usize),
            (1, 128, 90_000),
            (2, 192, 110_000),
        ] {
            let mut added = 0;
            if let Some(mesh) = previous.as_mut() {
                let available = super::calibrate::host_memory(Path::new("/"))?;
                let fill_bytes = mesh
                    .faces
                    .len()
                    .checked_add(mesh.vertices.len())
                    .and_then(|v| v.checked_mul(24))
                    .context("boundary host reservation overflow")?;
                ensure!(
                    fill_bytes as u64 <= available.saturating_sub(available / 16),
                    "boundary host reservation exceeds available memory"
                );
                let capacity = mesh
                    .faces
                    .len()
                    .checked_mul(4)
                    .context("boundary face capacity overflow")?
                    / 3;
                added = mesh.fill_holes(capacity)?;
            }
            let mesh = previous.as_ref().unwrap_or(self.mesh);
            let (free, _) = device.ctx.mem_get_info()?;
            let (accepted, voxels) = adaptive_mesh_support(
                device,
                mesh,
                nominal,
                limit,
                usize::MAX,
                [free, self.budget(device)?],
            )?;
            let coordinates = voxels.sites.coordinates().to_vec();
            let grid = coordinates
                .iter()
                .map(|p| [p[1], p[2], p[3]])
                .collect::<Vec<_>>();
            let frame = voxels.sites.unit_ball_frame()?;
            drop(voxels);
            drop(previous.take());
            let latent = self.flow_stage(device, &context, &grid, (stream, nominal))?;
            device.stream.synchronize()?;
            let decoder_budget = self.budget(device)?;
            let decoder = PixalDecoder::open(
                device,
                self.model,
                DecoderKind::PointTransformer,
                self.runtime,
                decoder_budget,
            )?;
            let mesh = decoder.points_mesh(coordinates, &latent, frame, nominal)?;
            drop(decoder);
            ensure!(
                !mesh.vertices.is_empty()
                    && mesh.vertices.len().is_multiple_of(3)
                    && !mesh.faces.is_empty()
                    && mesh.faces.len().is_multiple_of(3)
                    && mesh.vertices.iter().all(|v| v.is_finite())
                    && mesh
                        .faces
                        .iter()
                        .all(|&v| v >= 0 && (v as usize) < mesh.vertices.len() / 3),
                "invalid raw refinement mesh"
            );
            println!(
                "HI3D_GENERATION_RUNG resolution={} nominal_grid={nominal} accepted_grid={accepted} filled_faces_added={added} input_rows={} updates=25 component={} vertices={} faces={} noise=philox seed={} stream={stream} noise_boundary_D2H_H2D_bytes={} raw=true timing=false",
                nominal * 8,
                grid.len(),
                if stream == 0 {
                    "transformer_512"
                } else {
                    "transformer_1024"
                },
                mesh.vertices.len() / 3,
                mesh.faces.len() / 3,
                self.seed,
                grid.len() * 32 * 8
            );
            if nominal * 8 == self.requested_resolution {
                return Ok((mesh, nominal * 8));
            }
            previous = Some(mesh);
        }
        anyhow::bail!("refinement did not reach requested resolution")
    }

    fn flow_stage(
        &self,
        device: &Device,
        context: &CudaSlice<f32>,
        grid: &[[i32; 3]],
        stage: (u64, u32),
    ) -> Result<Vec<f32>> {
        let (stream, nominal) = stage;
        let elements = grid
            .len()
            .checked_mul(32)
            .context("refinement noise extent overflow")?;
        let noise = device.stream.alloc_zeros::<f32>(elements)?;
        unsafe {
            device
                .stream
                .launch_builder(
                    &cubin::module(&device.ctx, "sample")?.load_function("philox_normal_f32")?,
                )
                .arg(&self.seed)
                .arg(&stream)
                .arg(&noise)
                .arg(&i64::try_from(elements)?)
                .launch(flat(elements.div_ceil(4)))?;
        }
        let noise_host = device.stream.clone_dtoh(&noise)?;
        drop(noise);
        let component = if stream == 0 {
            "transformer_512"
        } else {
            "transformer_1024"
        };
        let mut flow = Flow::<f16>::open_hi3d(
            device,
            self.model,
            component,
            self.runtime,
            grid,
            nominal as usize,
            self.budget(device)?,
        )?;
        ensure!(
            flow.coordinates == grid,
            "refinement support did not reach flow"
        );
        let positive = flow.conditioning_device(context, grid)?;
        let zero = device.stream.alloc_zeros::<f32>(2917 * 2048)?;
        let negative = flow.conditioning_device(&zero, grid)?;
        flow.sample_heun(&noise_host, &positive, &negative)
    }
}

pub struct PixalOutput {
    pub mesh: crate::sparse::SparseMesh,
    /// RGB, metallic, roughness, alpha at the final voxel coordinates.
    pub material: SparseDecoderOutput,
    pub resolution: u32,
    pub origin: [f32; 3],
    pub voxel_size: f32,
}

impl PixalGeneration<'_> {
    pub fn run(&self, device: &Device) -> Result<PixalOutput> {
        self.run_observed(device, usize::MAX, |_, _, _, _| Ok(()))
    }
    fn run_observed<F>(
        &self,
        device: &Device,
        retention_limit: usize,
        mut observe: F,
    ) -> Result<PixalOutput>
    where
        F: FnMut(usize, u32, &[[i32; 3]], &CudaSlice<f32>) -> Result<()>,
    {
        ensure!(
            matches!(self.requested_resolution, 1024 | 1536),
            "unsupported requested resolution"
        );
        ensure!(
            !self.views.is_empty() && (self.multiview || self.views.len() == 1),
            "invalid single/multi-view request"
        );
        for view in self.views {
            let count = view
                .dimensions
                .iter()
                .try_fold(4usize, |n, &d| n.checked_mul(d))
                .context("RGBA size overflow")?;
            ensure!(
                view.dimensions.iter().all(|&n| n > 0)
                    && view.rgba.len() == count
                    && view.rgba.as_chunks::<4>().0.iter().any(|p| p[3] > 0)
                    && view.rgba.as_chunks::<4>().0.iter().any(|p| p[3] != 255),
                "request requires nonempty supplied RGBA foreground"
            );
        }
        let pipeline: serde_json::Value =
            serde_json::from_slice(&std::fs::read(self.model.join(if self.multiview {
                "pipeline_mv.json"
            } else {
                "pipeline.json"
            }))?)?;
        for (key, kind) in [
            ("sparse_structure_decoder", DecoderKind::Structure),
            ("shape_slat_decoder", DecoderKind::Shape),
            ("tex_slat_decoder", DecoderKind::Texture),
        ] {
            ensure!(
                pipeline["args"]["models"][key] == format!("ckpts/{}", kind.stem()),
                "decoder component differs from published geometry"
            );
        }
        let request = Rc::new(());
        let ss: Vec<_> = (0..16usize.pow(3))
            .map(|i| [(i / 256) as i32, ((i / 16) % 16) as i32, (i % 16) as i32])
            .collect();
        let structure = self.flow_stage(
            device,
            &pipeline,
            (0, 16),
            &ss,
            (&request, None),
            &mut observe,
        )?;
        let decoder = self.decoder(device, DecoderKind::Structure)?;
        let logits = decoder.structure_device(structure)?;
        let low = PixalDecoder::structure_sites(&logits, 64, 32)?;
        drop(logits);
        #[cfg(test)]
        decoder_pass_contract(decoder.decode_passes.get())?;
        drop(decoder);
        let low_xyz: Vec<_> = low
            .coordinates()
            .iter()
            .map(|p| [p[1], p[2], p[3]])
            .collect();
        let low_latent = self.flow_stage(
            device,
            &pipeline,
            (1, 32),
            &low_xyz,
            (&request, None),
            &mut observe,
        )?;
        drop(low_xyz);
        let decoder = self.decoder(device, DecoderKind::Shape)?;
        let (resolution, high) = decoder.cascade_coordinates_device(
            low.coordinates().to_vec(),
            low_latent,
            self.requested_resolution,
            49152,
        )?;
        drop(low);
        #[cfg(test)]
        decoder_pass_contract(decoder.decode_passes.get())?;
        let required = self.following_peak(&pipeline, high.coordinates().len(), resolution / 16)?;
        let measured = self.budget(device)?;
        let available = measured.min(retention_limit);
        let keep = required <= available;
        #[cfg(test)]
        {
            if retention_limit == 0 {
                ensure!(!keep, "test retention cap was ignored");
            }
            let weight_bytes = decoder
                .weights
                .half
                .values()
                .map(CudaSlice::num_bytes)
                .sum::<usize>()
                + decoder
                    .weights
                    .float
                    .values()
                    .map(CudaSlice::num_bytes)
                    .sum::<usize>();
            println!(
                "P1_M4C_STAGING required_bytes={required} measured_bytes={measured} available_bytes={available} retained_shape_decoder={keep} avoided_weight_upload_bytes={}",
                if keep { weight_bytes } else { 0 }
            );
        }
        let decoder = if keep {
            Some(decoder)
        } else {
            drop(decoder);
            None
        };
        let high_xyz: Vec<_> = high
            .coordinates()
            .iter()
            .map(|p| [p[1], p[2], p[3]])
            .collect();
        let shape = self.flow_stage(
            device,
            &pipeline,
            (2, resolution / 16),
            &high_xyz,
            (&request, None),
            &mut observe,
        )?;
        let texture = self.flow_stage(
            device,
            &pipeline,
            (3, resolution / 16),
            &high_xyz,
            (&request, Some(&shape)),
            &mut observe,
        )?;
        drop(high_xyz);
        let decoder = match decoder {
            Some(decoder) => decoder,
            None => self.decoder(device, DecoderKind::Shape)?,
        };
        #[cfg(test)]
        let previous_passes = decoder.decode_passes.get();
        let (mesh, guides) =
            decoder.shape_mesh_device(high.coordinates().to_vec(), shape, resolution)?;
        #[cfg(test)]
        decoder_pass_contract(decoder.decode_passes.get() - previous_passes)?;
        drop(decoder);
        let decoder = self.decoder(device, DecoderKind::Texture)?;
        let (material, predicted) =
            decoder.decode_device(high.coordinates().to_vec(), texture, &guides)?;
        #[cfg(test)]
        decoder_pass_contract(decoder.decode_passes.get())?;
        ensure!(
            predicted.is_empty(),
            "texture decoder predicted subdivision"
        );
        Ok(PixalOutput {
            mesh,
            material,
            resolution,
            origin: [-0.5; 3],
            voxel_size: 1.0 / resolution as f32,
        })
    }
    fn budget(&self, device: &Device) -> Result<usize> {
        let (free, total) = device.ctx.mem_get_info()?;
        device.workspace_budget(free, total, total / 16)
    }
    fn decoder(&self, device: &Device, kind: DecoderKind) -> Result<PixalDecoder> {
        PixalDecoder::open(device, self.model, kind, self.runtime, self.budget(device)?)
    }
    fn flow_geometry<'a>(
        &self,
        pipeline: &'a serde_json::Value,
        stage: usize,
        grid_side: u32,
    ) -> Result<(&'a str, FlowSpec, crate::backend::PixalConditioningShape)> {
        let key = [
            "sparse_structure_flow_model",
            "shape_slat_flow_model_512",
            "shape_slat_flow_model_1024",
            "tex_slat_flow_model_1024",
        ][stage];
        let stem = pipeline["args"]["models"][key]
            .as_str()
            .context("flow component missing")?
            .strip_prefix("ckpts/")
            .context("flow checkpoint component path")?;
        ensure!(
            !stem.is_empty()
                && !stem.contains('/')
                && !stem.contains('\\')
                && stem.ends_with("_mv") == self.multiview,
            "invalid flow component variant"
        );
        let spec = FlowSpec::read(&self.model.join("ckpts"), stem)?;
        ensure!(
            spec.sparse() == (stage > 0) && spec.texture() == (stage == 3),
            "flow component stage mismatch"
        );
        let shape = crate::backend::PixalConditioningShape {
            image_side: if stage < 2 { 512 } else { 1024 },
            grid_side,
            naf_side: (stage > 0).then_some(if stage == 3 { 1024 } else { 512 }),
        };
        Ok((stem, spec, shape))
    }
    fn following_peak(
        &self,
        pipeline: &serde_json::Value,
        rows: usize,
        grid_side: u32,
    ) -> Result<usize> {
        let mut peak = 0;
        for stage in [2, 3] {
            let (stem, spec, shape) = self.flow_geometry(pipeline, stage, grid_side)?;
            let conditioning = crate::backend::PixalConditioningRuntime::planned_bytes(
                self.dino,
                shape.naf_side.map(|_| self.naf),
                shape,
                rows,
            )?;
            let checkpoint = FlowCheckpoint::open(&self.model.join("ckpts"), stem)?;
            let sampler = if spec.texture() {
                "tex_slat_sampler"
            } else {
                "shape_slat_sampler"
            };
            let steps = usize::try_from(
                pipeline["args"][sampler]["params"]["steps"]
                    .as_u64()
                    .context("sampler steps missing")?,
            )?;
            let flow = Flow::<bf16>::planned_bytes(&checkpoint, rows, steps)?;
            let retained = rows
                .checked_mul(spec.out_channels * 4)
                .context("retained latent bytes overflow")?;
            peak = peak.max(
                conditioning
                    .max(flow)
                    .checked_add(retained)
                    .context("stage peak overflow")?,
            );
        }
        Ok(peak)
    }
    fn flow_stage<F>(
        &self,
        device: &Device,
        pipeline: &serde_json::Value,
        stage: (usize, u32),
        coordinates: &[[i32; 3]],
        input: (&Rc<()>, Option<&CudaSlice<f32>>),
        observe: &mut F,
    ) -> Result<CudaSlice<f32>>
    where
        F: FnMut(usize, u32, &[[i32; 3]], &CudaSlice<f32>) -> Result<()>,
    {
        let (request, physical_shape) = input;
        let (stage, grid_side) = stage;
        let (stem, spec, shape) = self.flow_geometry(pipeline, stage, grid_side)?;
        let ops = Ops::new(&device.ctx)?;
        let mut blas = Blas::new(
            device,
            &ops,
            self.runtime,
            Rc::new(Closure::new(false, "Pixal conditioning".into())),
        )?;
        let mut encoder = crate::backend::PixalConditioningRuntime::load(
            device,
            &mut blas,
            self.dino,
            shape.naf_side.map(|_| self.naf),
            shape,
            Some(coordinates),
            request,
        )?;
        for (index, view) in self.views.iter().enumerate() {
            if self.multiview {
                encoder.stage_view_rgba(&device.stream, view.rgba, view.dimensions, view.camera)?;
            } else {
                encoder.stage_rgba(&device.stream, view.rgba, view.dimensions, view.camera)?;
            }
            encoder.run_view(
                &device.stream,
                &blas,
                u32::try_from(index)?,
                u32::try_from(self.views.len())?,
            )?;
        }
        let projected = encoder.finish()?;
        drop(blas);
        let mut flow = Flow::open(
            device,
            &self.model.join("ckpts"),
            stem,
            self.runtime,
            (coordinates, grid_side),
            request,
            self.budget(device)?,
        )?;
        let cameras: Vec<_> = self.views.iter().map(|v| v.camera).collect();
        let mut positive = flow.conditioning_projected(&projected, &cameras)?;
        drop(projected);
        ensure!(
            physical_shape.is_some() == spec.texture(),
            "shape conditioning stage mismatch"
        );
        if let Some(shape) = physical_shape {
            flow.condition_shape_device(&mut positive, shape, coordinates)?;
        }
        let negative = if spec.texture() {
            None
        } else {
            Some(Conditioning {
                origin: flow.origin.clone(),
                identity: Rc::new(()),
                global: device.stream.alloc_zeros(positive.global.len())?,
                projected: device.stream.alloc_zeros(positive.projected.len())?,
                concat: None,
            })
        };
        let noise = self.noise[stage].gather(coordinates, spec.out_channels, !spec.sparse())?;
        let noise = device.stream.clone_htod(&noise)?;
        let mut output = flow.sample_device(noise, &positive, negative.as_ref())?;
        #[cfg(test)]
        let before_transform = device.stream.clone_dtoh(&output)?;
        if matches!(stage, 1 | 2) {
            let normalization = &pipeline["args"]["shape_slat_normalization"];
            let mean: Vec<f32> = serde_json::from_value(normalization["mean"].clone())?;
            let std: Vec<f32> = serde_json::from_value(normalization["std"].clone())?;
            ensure!(
                mean.len() == spec.out_channels && std.len() == spec.out_channels,
                "shape normalization channel count"
            );
            channel_transform(device, &mut output, [&mean, &std], false)?;
            #[cfg(test)]
            transform_contract(device, &before_transform, &output, [&mean, &std])?;
        }
        #[cfg(test)]
        if stage == 0 || stage == 3 {
            ensure!(
                device.stream.clone_dtoh(&output)? == before_transform,
                "unchanged latent boundary differs"
            );
        }
        observe(stage, grid_side, coordinates, &output)?;
        Ok(output)
    }
}

pub struct PixalDecoder {
    device: Device,
    blas: Blas,
    weights: Weights,
    k: DecoderKernels,
    kind: DecoderKind,
    initial_free: usize,
    device_budget: usize,
    ones: CudaSlice<f16>,
    columns: Option<CudaSlice<u8>>,
    #[cfg(test)]
    subdivision_launches: std::cell::Cell<usize>,
    #[cfg(test)]
    decode_passes: std::cell::Cell<usize>,
}
pub struct SparseDecoderOutput {
    pub coordinates: Vec<[i32; 4]>,
    pub features: Vec<f32>,
    pub channels: usize,
}
pub struct CornerDecoderOutput {
    pub fields: SparseDecoderOutput,
    /// Row-major [sites,8] corner sign logits.
    pub sign_logits: Vec<f32>,
    pub matrix: [[f32; 4]; 4],
    pub resolution: u32,
}
struct PointFields {
    sites: SparseSites,
    attributes: CudaSlice<f32>,
    resolution: u32,
}
struct PointWorkspace<T> {
    conv: SparseConv<f32>,
    norms: BTreeMap<usize, BatchGroupNorm>,
    scratch: [CudaSlice<T>; 2],
}
impl<T: DeviceRepr + Copy + ValidAsZeroBits + 'static> PointWorkspace<T> {
    fn new(
        decoder: &PixalDecoder,
        sites: &SparseSites,
        channels: [usize; 2],
        final_stage: bool,
    ) -> Result<Self> {
        let n = sites.coordinates().len();
        DecoderKind::validate_point_rows(n)?;
        let [ci, co] = channels;
        let count = n
            .checked_mul(ci)
            .context("point stage feature extent overflow")?;
        let scratch_bytes = count
            .checked_mul(2 * size_of::<T>())
            .context("point stage scratch overflow")?;
        let mut reserve = if final_stage {
            n.checked_mul(4 * (co + 40))
                .and_then(|v| v.checked_add(scratch_bytes))
                .context("point field workspace overflow")?
        } else {
            let projection = n
                .checked_mul(32 * co)
                .context("point subdivision workspace overflow")?;
            projection.max(
                scratch_bytes
                    .checked_add(n * co * size_of::<T>())
                    .and_then(|v| v.checked_add(n * (4 * co + 32)))
                    .context("point residual workspace overflow")?,
            )
        };
        let d = &decoder.device;
        let mut offsets = vec![0];
        let mut first = 0;
        while first < n {
            let batch = sites.coordinates()[first][0];
            first += sites.coordinates()[first..].partition_point(|p| p[0] == batch);
            offsets.push(u32::try_from(first)?);
        }
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let padded: usize = offsets
                .windows(2)
                .map(|v| {
                    let count = (v[1] - v[0]) as usize;
                    if count > 48 {
                        count.div_ceil(48) * 48
                    } else {
                        count
                    }
                })
                .sum();
            let windows: usize = offsets
                .windows(2)
                .map(|v| ((v[1] - v[0]) as usize).div_ceil(48))
                .sum();
            let mid_bytes = n
                .checked_mul((3 * 512 * 4) + ((2048 + 3 * 512) * 2) + 4)
                .and_then(|v| v.checked_add(padded * ((4 * 512 * 2) + (32 * 4) + 4)))
                .and_then(|v| v.checked_add((windows + 1) * 4))
                .context("point mid workspace overflow")?;
            reserve = reserve.max(mid_bytes);
        }
        let mut norms = BTreeMap::new();
        for c in [ci, co] {
            if let std::collections::btree_map::Entry::Vacant(entry) = norms.entry(c) {
                entry.insert(BatchGroupNorm::new(d, &offsets, c, 32, 1e-6, 64 << 20)?);
            }
        }
        #[cfg(test)]
        println!(
            "SPARSE_POINT_WORKSPACE rows={n} input_channels={ci} output_channels={co} element_bytes={} final_stage={final_stage} scratch_bytes={scratch_bytes} live_bytes={reserve}",
            size_of::<T>()
        );
        let available = d.workspace_budget(decoder.initial_free, decoder.device_budget, reserve)?;
        let conv = SparseConv::<f32>::promoted(d, sites, [ci, co], available)?;
        let scratch = [d.stream.alloc_zeros(count)?, d.stream.alloc_zeros(count)?];
        Ok(Self {
            conv,
            norms,
            scratch,
        })
    }
}
pub(crate) struct SubdivisionGuide<'a> {
    pub coordinates: &'a [[i32; 4]],
    pub logits: &'a [f32],
}
impl SubdivisionGuide<'_> {
    fn validate(&self, coordinates: &[[i32; 4]]) -> Result<()> {
        ensure!(
            self.coordinates == coordinates
                && self.logits.len() == coordinates.len() * 8
                && self.logits.iter().all(|x| x.is_finite()),
            "subdivision guide does not match stage coordinates"
        );
        Ok(())
    }
}
fn validate_guide_count(kind: DecoderKind, count: usize) -> Result<()> {
    ensure!(
        count <= 4 && (!matches!(kind, DecoderKind::Texture) || count == 4),
        "sparse subdivision guide count mismatch"
    );
    Ok(())
}
impl PixalDecoder {
    pub fn open(
        device: &Device,
        dir: &Path,
        kind: DecoderKind,
        runtime: &Path,
        device_budget: usize,
    ) -> Result<Self> {
        let component = if matches!(kind, DecoderKind::PointTransformer) {
            "vae"
        } else {
            "ckpts"
        };
        let checkpoint = DecoderCheckpoint::open(&dir.join(component), kind)?;
        let device = device.clone();
        let (free, _) = device.ctx.mem_get_info()?;
        #[cfg(test)]
        println!("SPARSE_DECODER_PLACEMENT initial_free={free} device_budget={device_budget}");
        let ops = Ops::new(&device.ctx)?;
        let blas = Blas::new(
            &device,
            &ops,
            runtime,
            Rc::new(Closure::new(false, "Pixal decoder".into())),
        )?;
        let k = DecoderKernels::load(&device, matches!(kind, DecoderKind::PointTransformer))?;
        let weights = Weights::load(&device, &checkpoint)?;
        let ones = device.stream.clone_htod(&vec![f16::ONE; 1024])?;
        device.workspace_budget(free, device_budget, 0)?;
        let columns = if matches!(kind, DecoderKind::Structure) {
            let live = 64usize.pow(3) * 32 * 8;
            let available = device.workspace_budget(free, device_budget, live)?;
            let full = 64usize.pow(3) * 32 * 27 * size_of::<f32>();
            let bytes = available.min(full);
            ensure!(
                bytes >= 512 * 27 * size_of::<f16>(),
                "dense columns cannot fit one row"
            );
            Some(device.stream.alloc_zeros::<u8>(bytes)?)
        } else {
            None
        };
        Ok(Self {
            device,
            blas,
            weights,
            k,
            kind,
            #[cfg(test)]
            subdivision_launches: std::cell::Cell::new(0),
            #[cfg(test)]
            decode_passes: std::cell::Cell::new(0),
            initial_free: free,
            device_budget,
            ones,
            columns,
        })
    }
    fn serialized_qkv(
        &self,
        maps: &SerializedWindows,
        qkv: Tensor<f16, 2>,
        order: usize,
    ) -> Result<CudaSlice<f16>> {
        ensure!(
            matches!(self.kind, DecoderKind::PointTransformer) && order < 2,
            "invalid serialized decoder attention configuration"
        );
        let rows = maps.order[order].len();
        let packed_rows = maps.pad.len();
        ensure!(
            maps.offsets.first() == Some(&0)
                && maps
                    .offsets
                    .last()
                    .is_some_and(|&n| n as usize == packed_rows),
            "serialized window offsets do not cover packed rows"
        );
        ensure!(
            rows > 0
                && qkv.shape == [u32::try_from(rows)?, 1536]
                && qkv.stride == [1536, 1]
                && maps.inverse[order].len() == rows
                && maps.unpad.len() == rows,
            "serialized QKV/map extent mismatch"
        );
        qkv.validate()?;
        let gather: Vec<_> = maps
            .pad
            .iter()
            .map(|&row| {
                maps.order[order]
                    .get(row as usize)
                    .copied()
                    .context("serialized pad row outside order")
            })
            .collect::<Result<_>>()?;
        let inverse: Vec<_> = maps.inverse[order]
            .iter()
            .map(|&row| {
                maps.unpad
                    .get(row as usize)
                    .copied()
                    .context("serialized inverse row outside unpad")
            })
            .collect::<Result<_>>()?;
        ensure!(
            gather.iter().all(|&row| (row as usize) < rows)
                && inverse.iter().all(|&row| (row as usize) < packed_rows),
            "serialized physical row outside features"
        );
        let live = packed_rows
            .checked_mul(4 * 512 * size_of::<f16>())
            .and_then(|n| n.checked_add(rows * 512 * size_of::<f16>()))
            .and_then(|n| {
                n.checked_add(
                    (gather.len() + inverse.len() + maps.offsets.len()) * size_of::<u32>(),
                )
            })
            .and_then(|n| n.checked_add(packed_rows * 32 * size_of::<f32>()))
            .context("serialized attention workspace overflow")?;
        let d = &self.device;
        d.workspace_budget(self.initial_free, self.device_budget, live)?;
        let gather = d.stream.clone_htod(&gather)?;
        let inverse = d.stream.clone_htod(&inverse)?;
        let q = d.stream.alloc_zeros::<f16>(packed_rows * 512)?;
        let k = d.stream.alloc_zeros::<f16>(packed_rows * 512)?;
        let v = d.stream.alloc_zeros::<f16>(packed_rows * 512)?;
        let mut packed = d.stream.alloc_zeros::<f16>(packed_rows * 512)?;
        let output = d.stream.alloc_zeros::<f16>(rows * 512)?;
        let module = cubin::module(&d.ctx, "vision_ops")?;
        let mut attention = Flash::new_f16(
            d.stream.clone(),
            packed_rows,
            32,
            32,
            16,
            Score {
                scale: 0.25,
                window_left: None,
                window_right: None,
            },
        )?;
        attention.bind_f16_windows(&maps.offsets)?;
        unsafe {
            d.stream
                .launch_builder(&module.load_function("qkv_gather_head_major_f16")?)
                .arg(&qkv.ptr)
                .arg(&gather)
                .arg(&q)
                .arg(&k)
                .arg(&v)
                .arg(&u32::try_from(rows)?)
                .arg(&u32::try_from(packed_rows)?)
                .arg(&32u32)
                .arg(&16u32)
                .launch(flat(packed_rows * 512))?;
        }
        attention.run_f16(
            &q.slice(..),
            512,
            &k.slice(..),
            &v.slice(..),
            &mut packed.slice_mut(..),
            (packed_rows, packed_rows, packed_rows),
        )?;
        unsafe {
            d.stream
                .launch_builder(&module.load_function("sparse_index_rows_f16")?)
                .arg(&packed)
                .arg(&inverse)
                .arg(&output)
                .arg(&u32::try_from(rows)?)
                .arg(&512u32)
                .launch(flat(rows * 512))?;
        }
        Ok(output)
    }
    fn point_norm<I: DeviceRepr + Copy + 'static, O: DeviceRepr + Copy + 'static>(
        &self,
        norms: &BTreeMap<usize, BatchGroupNorm>,
        name: &str,
        input: Tensor<I, 2>,
        output: &mut TensorMut<O, 2>,
    ) -> Result<()> {
        let c = input.shape[1] as usize;
        let plan = norms.get(&c).context("point norm width not planned")?;
        let w = self
            .weights
            .float
            .get(&format!("{name}.weights"))
            .context("point group norm weight missing")?;
        let b = self
            .weights
            .float
            .get(&format!("{name}.bias"))
            .context("point group norm bias missing")?;
        let stream = &self.device.stream;
        plan.run::<I, O>(
            input,
            [
                view(w, stream, 0, [c as u32], [1])?,
                view(b, stream, 0, [c as u32], [1])?,
            ],
            output,
        )
    }
    fn point_conv<I: DeviceRepr + Copy + 'static, O: DeviceRepr + Copy + 'static>(
        &self,
        conv: &SparseConv<f32>,
        name: &str,
        input: Tensor<I, 2>,
        output: &mut TensorMut<O, 2>,
    ) -> Result<()> {
        let ci = input.shape[1] as usize;
        let co = output.0.shape[1] as usize;
        let w = self
            .weights
            .float
            .get(&format!("{name}.weight"))
            .context("point convolution weight missing")?;
        let b = self
            .weights
            .float
            .get(&format!("{name}.bias"))
            .context("point convolution bias missing")?;
        ensure!(
            w.len() == 27 * ci * co && b.len() == co,
            "point convolution parameter extent mismatch"
        );
        conv.run_promoted(
            &self.blas,
            input,
            view(
                w,
                &self.device.stream,
                0,
                [co as u32, (27 * ci) as u32],
                [(27 * ci) as u64, 1],
            )?,
            view(b, &self.device.stream, 0, [co as u32], [1])?,
            output,
        )
    }
    fn point_swish<T: DeviceRepr + Copy + 'static>(&self, x: Tensor<T, 2>) -> Result<()> {
        use std::any::TypeId;
        let half = TypeId::of::<T>() == TypeId::of::<f16>();
        ensure!(
            half || TypeId::of::<T>() == TypeId::of::<f32>(),
            "point activation feature dtype"
        );
        x.validate()?;
        let functions = self
            .k
            .rounded_swish
            .as_ref()
            .context("rounded activation not loaded")?;
        let count = u64::from(x.shape[0]) * u64::from(x.shape[1]);
        unsafe {
            self.device
                .stream
                .launch_builder(&functions[usize::from(half)])
                .arg(&x.ptr)
                .arg(&x.ptr)
                .arg(&count)
                .launch(flat(count as usize))?;
        }
        Ok(())
    }
    fn point_linear(
        &self,
        name: &str,
        input: Tensor<f16, 2>,
        output: &mut TensorMut<f16, 2>,
        biased: bool,
    ) -> Result<()> {
        let n = input.shape[0] as usize;
        let ci = input.shape[1] as usize;
        let co = output.0.shape[1] as usize;
        let d = &self.device;
        let w = self
            .weights
            .half
            .get(&format!("{name}.weight"))
            .context("point linear weight missing")?;
        ensure!(w.len() == ci * co, "point linear parameter extent mismatch");
        if biased {
            let b = self
                .weights
                .half
                .get(&format!("{name}.bias"))
                .context("point linear bias missing")?;
            ensure!(b.len() == co, "point linear bias extent mismatch");
            self.k
                .bias(d, output.0.ptr, ptr(b, d), [n, co], true, false)?;
        }
        self.blas.linear_f16(
            view(w, &d.stream, 0, [co as u32, ci as u32], [ci as u64, 1])?,
            input,
            output,
            biased,
        )
    }
    fn point_residual<T: DeviceRepr + Copy + ValidAsZeroBits + 'static>(
        &self,
        workspace: &PointWorkspace<T>,
        name: &str,
        input: CudaSlice<T>,
        channels: [usize; 2],
        last_conv: &SparseConv<f32>,
    ) -> Result<CudaSlice<T>> {
        use std::any::TypeId;
        let [ci, co] = channels;
        let half = TypeId::of::<T>() == TypeId::of::<f16>();
        ensure!(
            half || TypeId::of::<T>() == TypeId::of::<f32>(),
            "point residual feature dtype"
        );
        ensure!(
            ci == co || half,
            "point channel transition requires F16 features"
        );
        let n = input.len() / ci;
        ensure!(
            n <= 12_000_000 && input.len() == n * ci,
            "point residual capacity or extent mismatch"
        );
        let d = &self.device;
        let a = matrix::<T>(ptr(&workspace.scratch[0], d), n, ci)?;
        self.point_norm(
            &workspace.norms,
            &format!("{name}.block.0"),
            matrix::<T>(ptr(&input, d), n, ci)?,
            &mut TensorMut(a),
        )?;
        self.point_swish(a)?;
        let b = matrix::<T>(ptr(&workspace.scratch[1], d), n, co)?;
        self.point_conv(
            &workspace.conv,
            &format!("{name}.block.2"),
            a,
            &mut TensorMut(b),
        )?;
        self.point_norm(
            &workspace.norms,
            &format!("{name}.block.3"),
            b,
            &mut TensorMut(b),
        )?;
        self.point_swish(b)?;
        let branch = matrix::<T>(ptr(&workspace.scratch[0], d), n, co)?;
        self.point_conv(
            last_conv,
            &format!("{name}.block.5"),
            b,
            &mut TensorMut(branch),
        )?;
        let output = if ci == co {
            input
        } else {
            let output = d.stream.alloc_zeros::<T>(n * co)?;
            self.point_linear(
                &format!("{name}.res_conv"),
                matrix::<f16>(ptr(&input, d), n, ci)?,
                &mut TensorMut(matrix::<f16>(ptr(&output, d), n, co)?),
                true,
            )?;
            output
        };
        unsafe {
            if half {
                d.stream
                    .launch_builder(&self.k.residual)
                    .arg(&ptr(&output, d))
                    .arg(&branch.ptr)
                    .arg(&self.ones)
                    .arg(&ptr(&output, d))
                    .arg(&(n as u32))
                    .arg(&(co as u32))
                    .launch(flat(n * co))?;
            } else {
                d.stream
                    .launch_builder(
                        self.k
                            .add32
                            .as_ref()
                            .context("point F32 residual not loaded")?,
                    )
                    .arg(&ptr(&output, d))
                    .arg(&branch.ptr)
                    .arg(&i64::try_from(n * co)?)
                    .arg(&1f32)
                    .launch(grid((n * co).div_ceil(256), 256))?;
            }
        }
        Ok(output)
    }
    fn point_blocks<T: DeviceRepr + Copy + ValidAsZeroBits + 'static>(
        &self,
        workspace: &PointWorkspace<T>,
        mut input: CudaSlice<T>,
        stage: usize,
        channels: [usize; 2],
    ) -> Result<CudaSlice<T>> {
        let [ci, co] = channels;
        for block in 0..2 {
            input = self.point_residual(
                workspace,
                &format!("decoder.dec.dec{stage}.block{block}"),
                input,
                [if block == 0 { ci } else { co }, co],
                &workspace.conv,
            )?;
        }
        Ok(input)
    }
    fn point_mid(
        &self,
        workspace: &PointWorkspace<f32>,
        maps: &SerializedWindows,
        mut input: CudaSlice<f32>,
        last_conv: &SparseConv<f32>,
    ) -> Result<CudaSlice<f32>> {
        let d = &self.device;
        let n = input.len() / 512;
        let half_a = d.stream.alloc_zeros::<f16>(n * 512)?;
        let half_b = d.stream.alloc_zeros::<f16>(n * 512)?;
        let wide = d.stream.alloc_zeros::<f16>(n * 2048)?;
        input = self.point_residual(
            workspace,
            "decoder.mid.block_in",
            input,
            [512, 512],
            &workspace.conv,
        )?;
        let a = matrix::<f32>(ptr(&workspace.scratch[0], d), n, 512)?;
        let ha = matrix::<f16>(ptr(&half_a, d), n, 512)?;
        let hb = matrix::<f16>(ptr(&half_b, d), n, 512)?;
        for block in 0..2 {
            let name = format!("decoder.mid.block{block}");
            self.point_conv(
                &workspace.conv,
                &format!("{name}.cpe.0"),
                matrix::<f32>(ptr(&input, d), n, 512)?,
                &mut TensorMut(a),
            )?;
            self.k.cast(d, [a.ptr, ha.ptr], n * 512, false)?;
            self.point_linear(&format!("{name}.cpe.1"), ha, &mut TensorMut(hb), true)?;
            self.k.norm(
                d,
                [hb.ptr, hb.ptr],
                [n, 512],
                self.weights.affine(d, Some(&format!("{name}.cpe.2")))?,
                [false, false],
                1e-5,
            )?;
            unsafe {
                d.stream
                    .launch_builder(
                        self.k
                            .add16_to32
                            .as_ref()
                            .context("mixed residual not loaded")?,
                    )
                    .arg(&ptr(&input, d))
                    .arg(&hb.ptr)
                    .arg(&((n * 512) as u64))
                    .launch(flat(n * 512))?;
            }
            self.k.norm(
                d,
                [ptr(&input, d), a.ptr],
                [n, 512],
                self.weights.affine(d, Some(&format!("{name}.norm1.0")))?,
                [true, true],
                1e-5,
            )?;
            self.k.cast(d, [a.ptr, ha.ptr], n * 512, false)?;
            let qkv = matrix::<f16>(ptr(&wide, d), n, 1536)?;
            self.point_linear(&format!("{name}.attn.qkv"), ha, &mut TensorMut(qkv), true)?;
            let attended = self.serialized_qkv(maps, qkv, block)?;
            self.point_linear(
                &format!("{name}.attn.proj"),
                matrix::<f16>(ptr(&attended, d), n, 512)?,
                &mut TensorMut(hb),
                true,
            )?;
            unsafe {
                d.stream
                    .launch_builder(
                        self.k
                            .add16_to32
                            .as_ref()
                            .context("mixed residual not loaded")?,
                    )
                    .arg(&ptr(&input, d))
                    .arg(&hb.ptr)
                    .arg(&((n * 512) as u64))
                    .launch(flat(n * 512))?;
            }
            drop(attended);
            self.k.norm(
                d,
                [ptr(&input, d), a.ptr],
                [n, 512],
                self.weights.affine(d, Some(&format!("{name}.norm2.0")))?,
                [true, true],
                1e-5,
            )?;
            self.k.cast(d, [a.ptr, ha.ptr], n * 512, false)?;
            let hidden = matrix::<f16>(ptr(&wide, d), n, 2048)?;
            self.point_linear(
                &format!("{name}.mlp.0.fc1"),
                ha,
                &mut TensorMut(hidden),
                true,
            )?;
            unsafe {
                d.stream
                    .launch_builder(self.k.gelu16.as_ref().context("point GELU not loaded")?)
                    .arg(&hidden.ptr)
                    .arg(&hidden.ptr)
                    .arg(&(n as u32))
                    .arg(&1u32)
                    .arg(&2048u32)
                    .arg(&2048u64)
                    .arg(&0u64)
                    .arg(&1u64)
                    .arg(&2048u64)
                    .arg(&0u64)
                    .arg(&1u64)
                    .arg(&0i32)
                    .launch(flat(n * 2048))?;
            }
            self.point_linear(
                &format!("{name}.mlp.0.fc2"),
                hidden,
                &mut TensorMut(hb),
                true,
            )?;
            unsafe {
                d.stream
                    .launch_builder(
                        self.k
                            .add16_to32
                            .as_ref()
                            .context("mixed residual not loaded")?,
                    )
                    .arg(&ptr(&input, d))
                    .arg(&hb.ptr)
                    .arg(&((n * 512) as u64))
                    .launch(flat(n * 512))?;
            }
        }
        self.point_residual(
            workspace,
            "decoder.mid.block_out",
            input,
            [512, 512],
            last_conv,
        )
    }
    fn point_head<T: DeviceRepr + Copy + 'static>(
        &self,
        workspace: &PointWorkspace<T>,
        input: Tensor<T, 2>,
        stage: usize,
    ) -> Result<CudaSlice<f32>> {
        let d = &self.device;
        let [n, c] = input.shape.map(|v| v as usize);
        let normalized = d.stream.alloc_zeros::<f32>(n * c)?;
        let norm = matrix::<f32>(ptr(&normalized, d), n, c)?;
        let name = format!("decoder.dec.dec{stage}.up.upsample.cls_head.fc");
        self.point_norm(
            &workspace.norms,
            &format!("{name}.0"),
            input,
            &mut TensorMut(norm),
        )?;
        self.point_swish(norm)?;
        let output = d.stream.alloc_zeros::<f32>(n * 8)?;
        self.point_conv(
            &workspace.conv,
            &format!("{name}.2"),
            norm,
            &mut TensorMut(matrix::<f32>(ptr(&output, d), n, 8)?),
        )?;
        Ok(output)
    }
    fn point_project<T: DeviceRepr + Copy + 'static>(
        &self,
        input: Tensor<T, 2>,
        stage: usize,
    ) -> Result<CudaSlice<f16>> {
        use std::any::TypeId;
        let d = &self.device;
        let [n, c] = input.shape.map(|v| v as usize);
        let half = TypeId::of::<T>() == TypeId::of::<f16>();
        ensure!(
            half || TypeId::of::<T>() == TypeId::of::<f32>(),
            "point projection feature dtype"
        );
        let converted = if half {
            None
        } else {
            Some(d.stream.alloc_zeros::<f16>(n * c)?)
        };
        let address = if let Some(x) = &converted {
            self.k.cast(d, [input.ptr, ptr(x, d)], n * c, false)?;
            ptr(x, d)
        } else {
            input.ptr
        };
        let output = d.stream.alloc_zeros::<f16>(n * 8 * c)?;
        self.point_linear(
            &format!("decoder.dec.dec{stage}.up.upsample.proj"),
            matrix(address, n, c)?,
            &mut TensorMut(matrix(ptr(&output, d), n, 8 * c)?),
            false,
        )?;
        Ok(output)
    }
    fn point_up<T: DeviceRepr + Copy + 'static>(
        &self,
        workspace: PointWorkspace<T>,
        sites: SparseSites,
        input: CudaSlice<T>,
        stage: usize,
    ) -> Result<(SparseSites, CudaSlice<f16>)> {
        let d = &self.device;
        let n = sites.coordinates().len();
        let c = input.len() / n;
        let x = matrix::<T>(ptr(&input, d), n, c)?;
        let logits = self.point_head(&workspace, x, stage)?;
        let logits = d.stream.clone_dtoh(&logits)?;
        ensure!(
            logits.iter().all(|v| v.is_finite()),
            "nonfinite point occupancy logits"
        );
        let active: Vec<[bool; 8]> = logits
            .as_chunks::<8>()
            .0
            .iter()
            .map(|row| row.map(|v| v > 0.0))
            .collect();
        let offsets = std::array::from_fn(|child| {
            [
                ((child >> 2) & 1) as i32,
                ((child >> 1) & 1) as i32,
                (child & 1) as i32,
            ]
        });
        let (children, source) = sites.subdivide_with_offsets(&active, 12_000_000, &offsets)?;
        drop(workspace);
        let rows = children.coordinates().len();
        let reserve = n
            .checked_mul(18 * c)
            .and_then(|v| v.checked_add(rows * (2 * c + 8)))
            .context("point subdivision allocation overflow")?;
        d.workspace_budget(self.initial_free, self.device_budget, reserve)?;
        let projection = self.point_project(x, stage)?;
        drop(input);
        let pairs = d.stream.clone_htod(source.as_flattened())?;
        let output = d.stream.alloc_zeros::<f16>(rows * c)?;
        self.c2s(&projection, &pairs, &output, [n, rows, c, c], false)?;
        Ok((children, output))
    }
    fn point_field<T: DeviceRepr + Copy + 'static>(
        &self,
        workspace: &PointWorkspace<T>,
        input: &CudaSlice<T>,
        last_conv: &SparseConv<f32>,
    ) -> Result<CudaSlice<f32>> {
        let d = &self.device;
        let n = input.len() / 128;
        let normalized = d.stream.alloc_zeros::<f32>(n * 128)?;
        let norm = matrix::<f32>(ptr(&normalized, d), n, 128)?;
        self.point_norm(
            &workspace.norms,
            "decoder.conv_out.conv.0",
            matrix::<T>(ptr(input, d), n, 128)?,
            &mut TensorMut(norm),
        )?;
        self.point_swish(norm)?;
        let output = d.stream.alloc_zeros::<f32>(n * 40)?;
        self.point_conv(
            last_conv,
            "decoder.conv_out.conv.2",
            norm,
            &mut TensorMut(matrix::<f32>(ptr(&output, d), n, 40)?),
        )?;
        Ok(output)
    }
    fn points_device(
        &self,
        coordinates: Vec<[i32; 4]>,
        latent: &[f32],
        frame: [[f32; 4]; 4],
        resolution: u32,
    ) -> Result<PointFields> {
        ensure!(
            matches!(self.kind, DecoderKind::PointTransformer),
            "point decoder requires point transformer weights"
        );
        DecoderKind::validate_point_rows(coordinates.len())?;
        ensure!(
            latent.len() == coordinates.len() * 32
                && latent.iter().all(|v| v.is_finite())
                && frame.iter().flatten().all(|v| v.is_finite()),
            "invalid point decoder input extent or values"
        );
        let output_resolution = resolution
            .checked_mul(8)
            .filter(|_| resolution > 0)
            .context("point decoder resolution overflow")?;
        ensure!(
            coordinates
                .iter()
                .all(|p| p[1..].iter().all(|&v| v >= 0 && (v as u32) < resolution)),
            "point coordinate outside input resolution"
        );
        let mut sites = SparseSites::new(coordinates)?;
        let n = sites.coordinates().len();
        let d = &self.device;
        d.workspace_budget(self.initial_free, self.device_budget, n * (32 + 512) * 4)?;
        let input = d.stream.clone_htod(latent)?;
        let mut features = d.stream.alloc_zeros::<f32>(n * 512)?;
        let workspace = PointWorkspace::new(self, &sites, [512, 512], false)?;
        self.point_conv(
            &workspace.conv,
            "decoder.z_to_f.conv.0",
            matrix::<f32>(ptr(&input, d), n, 32)?,
            &mut TensorMut(matrix::<f32>(ptr(&features, d), n, 512)?),
        )?;
        drop(input);
        let maps = sites.serialized_windows(48)?;
        features = self.point_mid(&workspace, &maps, features, &workspace.conv)?;
        drop(maps);
        features = self.point_blocks(&workspace, features, 3, [512, 512])?;
        let (next, mut features) = self.point_up(workspace, sites, features, 3)?;
        sites = next;
        for (stage, ci, co) in [(2, 512, 256), (1, 256, 128)] {
            let workspace = PointWorkspace::new(self, &sites, [ci, co], false)?;
            features = self.point_blocks(&workspace, features, stage, [ci, co])?;
            (sites, features) = self.point_up(workspace, sites, features, stage)?;
        }
        let workspace = PointWorkspace::new(self, &sites, [128, 128], true)?;
        features = self.point_blocks(&workspace, features, 0, [128, 128])?;
        let fields = self.point_field(&workspace, &features, &workspace.conv)?;
        drop(features);
        let rows = sites.coordinates().len();
        unsafe {
            d.stream
                .launch_builder(
                    self.k
                        .corner_fields
                        .as_ref()
                        .context("corner activation not loaded")?,
                )
                .arg(&fields)
                .arg(&u32::try_from(rows)?)
                .launch(flat(rows * 32))?;
        }
        Ok(PointFields {
            sites,
            attributes: fields,
            resolution: output_resolution,
        })
    }
    pub fn points(
        &self,
        coordinates: Vec<[i32; 4]>,
        latent: &[f32],
        frame: [[f32; 4]; 4],
        resolution: u32,
    ) -> Result<CornerDecoderOutput> {
        let output = self.points_device(coordinates, latent, frame, resolution)?;
        let rows = output.sites.coordinates().len();
        let values = self.device.stream.clone_dtoh(&output.attributes)?;
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite point decoder fields"
        );
        let mut sign_logits = Vec::with_capacity(rows * 8);
        let mut features = Vec::with_capacity(rows * 32);
        for row in values.as_chunks::<40>().0 {
            sign_logits.extend_from_slice(&row[..8]);
            features.extend_from_slice(&row[8..]);
        }
        Ok(CornerDecoderOutput {
            fields: SparseDecoderOutput {
                coordinates: output.sites.coordinates().to_vec(),
                features,
                channels: 32,
            },
            sign_logits,
            matrix: frame,
            resolution: output.resolution,
        })
    }
    pub fn points_mesh(
        &self,
        coordinates: Vec<[i32; 4]>,
        latent: &[f32],
        frame: [[f32; 4]; 4],
        resolution: u32,
    ) -> Result<crate::sparse::SparseMesh> {
        ensure!(
            coordinates.iter().all(|p| p[0] == 0),
            "corner mesh requires one batch"
        );
        let output = self.points_device(coordinates, latent, frame, resolution)?;
        let d = &self.device;
        let budget = [self.initial_free, self.device_budget];
        let fields =
            super::sparse::corner_fields(d, &output.sites, &output.attributes, frame, budget)?;
        drop(output.attributes);
        drop(fields.layout);
        drop(fields.means);
        super::sparse::cube_mesh(
            d,
            output.sites,
            fields.cube_to_corner,
            fields.positions,
            fields.scalar,
            budget,
        )
    }
    fn linear16(
        &self,
        name: &str,
        input: &CudaSlice<f16>,
        output: &mut CudaSlice<f16>,
        shape: [usize; 3],
    ) -> Result<()> {
        let [rows, ci, co] = shape;
        let d = &self.device;
        let w = self
            .weights
            .half
            .get(&format!("{name}.weight"))
            .context("linear16 weight missing")?;
        let b = self
            .weights
            .half
            .get(&format!("{name}.bias"))
            .context("linear16 bias missing")?;
        self.k
            .bias(d, ptr(output, d), ptr(b, d), [rows, co], true, false)?;
        self.blas.linear_f16(
            matrix(ptr(w, d), co, ci)?,
            matrix(ptr(input, d), rows, ci)?,
            &mut TensorMut(matrix(ptr(output, d), rows, co)?),
            true,
        )
    }
    fn linear32(
        &self,
        name: &str,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        shape: [usize; 3],
    ) -> Result<()> {
        let [rows, ci, co] = shape;
        let d = &self.device;
        let w = self
            .weights
            .float
            .get(&format!("{name}.weight"))
            .context("linear32 weight missing")?;
        let b = self
            .weights
            .float
            .get(&format!("{name}.bias"))
            .context("linear32 bias missing")?;
        self.blas.linear_f32(
            matrix(ptr(w, d), co, ci)?,
            matrix(ptr(input, d), rows, ci)?,
            &mut TensorMut(matrix(ptr(output, d), rows, co)?),
        )?;
        self.k
            .bias(d, ptr(output, d), ptr(b, d), [rows, co], false, true)
    }
    fn sparse_conv(
        &self,
        conv: &SparseConv,
        name: &str,
        input: &CudaSlice<f16>,
        output: &mut CudaSlice<f16>,
        ci: usize,
    ) -> Result<()> {
        let d = &self.device;
        let w = self
            .weights
            .half
            .get(&format!("{name}.weight"))
            .context("sparse conv weight missing")?;
        let b = self
            .weights
            .half
            .get(&format!("{name}.bias"))
            .context("sparse conv bias missing")?;
        conv.run(
            &self.blas,
            input,
            matrix(ptr(w, d), b.len(), 27 * ci)?,
            view(b, &d.stream, 0, [b.len() as u32], [1])?,
            output,
        )
    }
    fn norm16(
        &self,
        name: Option<&str>,
        input: &CudaSlice<f16>,
        output: &CudaSlice<f16>,
        rows: usize,
        c: usize,
        eps: f32,
    ) -> Result<()> {
        self.k.norm(
            &self.device,
            [ptr(input, &self.device), ptr(output, &self.device)],
            [rows, c],
            self.weights.affine(&self.device, name)?,
            [false, false],
            eps,
        )
    }
    fn next_blocks(
        &self,
        stage: usize,
        conv: &SparseConv,
        x: CudaSlice<f16>,
        c: usize,
        blocks: usize,
    ) -> Result<CudaSlice<f16>> {
        self.next_blocks_observed(stage, conv, x, c, blocks, |_, _| Ok(()))
    }
    fn next_blocks_observed<F>(
        &self,
        stage: usize,
        conv: &SparseConv,
        x: CudaSlice<f16>,
        c: usize,
        blocks: usize,
        mut observe: F,
    ) -> Result<CudaSlice<f16>>
    where
        F: FnMut(usize, &CudaSlice<f16>) -> Result<()>,
    {
        let d = &self.device;
        let n = x.len() / c;
        let features = n.checked_mul(c).context("sparse features overflow")?;
        let additional = features
            .checked_mul(14)
            .context("sparse workspace overflow")?;
        d.workspace_budget(self.initial_free, self.device_budget, additional)?;
        let mut a = d.stream.alloc_zeros::<f16>(features)?;
        let b = d.stream.alloc_zeros::<f16>(features)?;
        let mut wide = d.stream.alloc_zeros::<f16>(features * 4)?;
        for block in 0..blocks {
            let name = format!("blocks.{stage}.{block}");
            self.sparse_conv(conv, &format!("{name}.conv"), &x, &mut a, c)?;
            self.norm16(Some(&format!("{name}.norm")), &a, &b, n, c, 1e-6)?;
            self.linear16(&format!("{name}.mlp.0"), &b, &mut wide, [n, c, 4 * c])?;
            self.k.act(d, ptr(&wide, d), n, 4 * c, false)?;
            self.linear16(&format!("{name}.mlp.2"), &wide, &mut a, [n, 4 * c, c])?;
            self.k.add(d, &x, &a, &self.ones, [n, c])?;
            observe(block, &x)?;
        }
        Ok(x)
    }
    fn c2s(
        &self,
        x: &CudaSlice<f16>,
        pairs: &CudaSlice<u32>,
        y: &CudaSlice<f16>,
        shape: [usize; 4],
        add: bool,
    ) -> Result<()> {
        let [parents, rows, ci, co] = shape;
        unsafe {
            self.device
                .stream
                .launch_builder(&self.k.c2s)
                .arg(x)
                .arg(pairs)
                .arg(y)
                .arg(&(parents as u32))
                .arg(&(rows as u32))
                .arg(&(ci as u32))
                .arg(&(co as u32))
                .arg(&(add as i32))
                .launch(flat(rows * co))?;
        }
        Ok(())
    }
    fn up_stage(
        &self,
        stage: usize,
        sites: &SparseSites,
        x: CudaSlice<f16>,
        conv: SparseConv,
        guide: Option<&SubdivisionGuide<'_>>,
    ) -> Result<(SparseSites, CudaSlice<f16>, Vec<f32>, SparseConv)> {
        let c = 1024 >> stage;
        let co = c / 2;
        let n = sites.coordinates().len();
        let blocks = [4, 16, 8, 4][stage];
        if let Some(guide) = guide {
            guide.validate(sites.coordinates())?;
        }
        let d = &self.device;
        let available = d
            .workspace_budget(self.initial_free, self.device_budget, 0)?
            .checked_add(conv.workspace_bytes())
            .context("released sparse workspace overflow")?;
        let capacity = available / (co * 16 + 27 * size_of::<i32>() + size_of::<[i32; 4]>());
        let name = format!("blocks.{stage}.{blocks}");
        let live = n
            .checked_mul(c * 10 + 16)
            .context("C2S workspace overflow")?;
        d.workspace_budget(self.initial_free, self.device_budget, live)?;
        let predicted = if matches!(self.kind, DecoderKind::Shape) {
            #[cfg(test)]
            self.subdivision_launches
                .set(self.subdivision_launches.get() + 1);
            let mut subdivision = d.stream.alloc_zeros::<f16>(n * 8)?;
            self.linear16(
                &format!("{name}.to_subdiv"),
                &x,
                &mut subdivision,
                [n, c, 8],
            )?;
            let predicted: Vec<f32> = d
                .stream
                .clone_dtoh(&subdivision)?
                .into_iter()
                .map(f16::to_f32)
                .collect();
            drop(subdivision);
            predicted
        } else {
            ensure!(guide.is_some(), "texture subdivision guide missing");
            Vec::new()
        };
        ensure!(
            predicted.iter().all(|x| x.is_finite()),
            "nonfinite subdivision logits"
        );
        let active: Vec<[bool; 8]> = guide
            .map_or(predicted.as_slice(), |g| g.logits)
            .as_chunks::<8>()
            .0
            .iter()
            .map(|row| row.map(|x| x > 0.0))
            .collect();
        let (children, source) = sites.subdivide(&active, capacity)?;
        let child_n = children.coordinates().len();
        let normalized = d.stream.alloc_zeros::<f16>(n * c)?;
        self.norm16(Some(&format!("{name}.norm1")), &x, &normalized, n, c, 1e-6)?;
        self.k.act(d, ptr(&normalized, d), n, c, false)?;
        let mut expanded = d.stream.alloc_zeros::<f16>(n * 8 * co)?;
        self.sparse_conv(
            &conv,
            &format!("{name}.conv1"),
            &normalized,
            &mut expanded,
            c,
        )?;
        drop(conv);
        drop(normalized);
        let live = child_n
            .checked_mul(co * 2 + size_of::<[u32; 2]>())
            .context("C2S child workspace overflow")?;
        d.workspace_budget(self.initial_free, self.device_budget, live)?;
        let pairs = d.stream.clone_htod(source.as_flattened())?;
        let child = d.stream.alloc_zeros::<f16>(child_n * co)?;
        self.c2s(&expanded, &pairs, &child, [n, child_n, co, co], false)?;
        drop(expanded);
        let normalized = d.stream.alloc_zeros::<f16>(child_n * co)?;
        self.norm16(None, &child, &normalized, child_n, co, 1e-6)?;
        drop(child);
        self.k.act(d, ptr(&normalized, d), child_n, co, false)?;
        let mut output = d.stream.alloc_zeros::<f16>(child_n * co)?;
        let additional = if stage + 1 == 4 {
            0
        } else {
            child_n
                .checked_mul(co * 14)
                .context("child workspace overflow")?
        };
        let budget = d.workspace_budget(self.initial_free, self.device_budget, additional)?;
        let conv = SparseConv::new(d, &children, co, budget)?;
        self.sparse_conv(
            &conv,
            &format!("{name}.conv2"),
            &normalized,
            &mut output,
            co,
        )?;
        self.c2s(&x, &pairs, &output, [n, child_n, c / 8, co], true)?;
        Ok((children, output, predicted, conv))
    }
    fn sparse_decode<F>(
        &self,
        coordinates: Vec<[i32; 4]>,
        latent: &[f32],
        guides: &[SubdivisionGuide<'_>],
        observe: F,
    ) -> Result<(SparseSites, CudaSlice<f16>)>
    where
        F: FnMut([usize; 3], (&SparseSites, &[f32]), (&SparseSites, &CudaSlice<f16>)) -> Result<()>,
    {
        ensure!(
            latent.len() == coordinates.len() * 32 && latent.iter().all(|x| x.is_finite()),
            "invalid denormalized sparse latent"
        );
        self.device.workspace_budget(
            self.initial_free,
            self.device_budget,
            latent
                .len()
                .checked_mul(size_of::<f32>())
                .context("latent bytes overflow")?,
        )?;
        let input = self.device.stream.clone_htod(latent)?;
        self.sparse_decode_rows(coordinates, input, guides, observe)
    }
    fn sparse_decode_rows<F>(
        &self,
        coordinates: Vec<[i32; 4]>,
        input: CudaSlice<f32>,
        guides: &[SubdivisionGuide<'_>],
        mut observe: F,
    ) -> Result<(SparseSites, CudaSlice<f16>)>
    where
        F: FnMut([usize; 3], (&SparseSites, &[f32]), (&SparseSites, &CudaSlice<f16>)) -> Result<()>,
    {
        ensure!(
            matches!(self.kind, DecoderKind::Shape | DecoderKind::Texture),
            "sparse decoder required"
        );
        validate_guide_count(self.kind, guides.len())?;
        #[cfg(test)]
        self.decode_passes.set(self.decode_passes.get() + 1);
        let mut sites = SparseSites::new(coordinates)?;
        let n = sites.coordinates().len();
        ensure!(
            input.len() == n * 32,
            "invalid denormalized sparse latent extent"
        );
        let d = &self.device;
        wait_input(d, &input)?;
        let live = n
            .checked_mul(1024 * 6)
            .context("sparse input live bytes overflow")?;
        d.workspace_budget(self.initial_free, self.device_budget, live)?;
        let mut float = d.stream.alloc_zeros::<f32>(n * 1024)?;
        self.linear32("from_latent", &input, &mut float, [n, 32, 1024])?;
        drop(input);
        let mut x = d.stream.alloc_zeros::<f16>(n * 1024)?;
        self.k
            .cast(d, [ptr(&float, d), ptr(&x, d)], n * 1024, false)?;
        drop(float);
        let additional = n
            .checked_mul(1024 * 14)
            .context("initial sparse workspace overflow")?;
        let budget = d.workspace_budget(self.initial_free, self.device_budget, additional)?;
        let mut conv = SparseConv::new(d, &sites, 1024, budget)?;
        for stage in 0..4 {
            let parent_workspace = conv.workspace_bytes();
            x = self.next_blocks(stage, &conv, x, 1024 >> stage, [4, 16, 8, 4][stage])?;
            let (next, features, predicted, next_conv) =
                self.up_stage(stage, &sites, x, conv, guides.get(stage))?;
            observe(
                [stage, parent_workspace, next_conv.workspace_bytes()],
                (&sites, &predicted),
                (&next, &features),
            )?;
            sites = next;
            x = features;
            conv = next_conv;
        }
        Ok((sites, x))
    }
    #[cfg(test)]
    pub(crate) fn shape_prefix<F>(
        &self,
        coordinates: Vec<[i32; 4]>,
        latent: &[f32],
        guides: &[SubdivisionGuide<'_>],
        mut observe: F,
    ) -> Result<SparseDecoderOutput>
    where
        F: FnMut(usize, &SparseDecoderOutput, &[f32]) -> Result<()>,
    {
        let d = &self.device;
        let (sites, x) = self.sparse_decode(
            coordinates,
            latent,
            guides,
            |[stage, _, _], (_, predicted), (next, features)| {
                let host = SparseDecoderOutput {
                    coordinates: next.coordinates().to_vec(),
                    features: d
                        .stream
                        .clone_dtoh(features)?
                        .into_iter()
                        .map(f16::to_f32)
                        .collect(),
                    channels: 512 >> stage,
                };
                ensure!(
                    host.features.iter().all(|x| x.is_finite()),
                    "nonfinite sparse stage output"
                );
                observe(stage, &host, predicted)
            },
        )?;
        Ok(SparseDecoderOutput {
            coordinates: sites.coordinates().to_vec(),
            features: d
                .stream
                .clone_dtoh(&x)?
                .into_iter()
                .map(f16::to_f32)
                .collect(),
            channels: 64,
        })
    }
    pub fn decode(
        &self,
        coordinates: Vec<[i32; 4]>,
        latent: &[f32],
        guides: &[SparseDecoderOutput],
    ) -> Result<(SparseDecoderOutput, Vec<SparseDecoderOutput>)> {
        ensure!(
            latent.len() == coordinates.len() * 32 && latent.iter().all(|x| x.is_finite()),
            "invalid denormalized sparse latent"
        );
        let input = self.device.stream.clone_htod(latent)?;
        self.decode_output(coordinates, input, guides)
    }
    pub fn decode_device(
        &self,
        coordinates: Vec<[i32; 4]>,
        latent: CudaSlice<f32>,
        guides: &[SparseDecoderOutput],
    ) -> Result<(SparseDecoderOutput, Vec<SparseDecoderOutput>)> {
        finite_copy(&self.device, &latent, &latent)?;
        self.decode_output(coordinates, latent, guides)
    }
    fn decode_output(
        &self,
        coordinates: Vec<[i32; 4]>,
        latent: CudaSlice<f32>,
        guides: &[SparseDecoderOutput],
    ) -> Result<(SparseDecoderOutput, Vec<SparseDecoderOutput>)> {
        let (sites, output, generated) = self.decode_fields(coordinates, latent, guides)?;
        let features = self.device.stream.clone_dtoh(&output)?;
        ensure!(
            features.iter().all(|x| x.is_finite()),
            "nonfinite decoder output"
        );
        Ok((
            SparseDecoderOutput {
                coordinates: sites.coordinates().to_vec(),
                features,
                channels: output.len() / sites.coordinates().len(),
            },
            generated,
        ))
    }
    fn decode_fields(
        &self,
        coordinates: Vec<[i32; 4]>,
        latent: CudaSlice<f32>,
        guides: &[SparseDecoderOutput],
    ) -> Result<(SparseSites, CudaSlice<f32>, Vec<SparseDecoderOutput>)> {
        ensure!(
            guides.iter().all(|g| g.channels == 8),
            "subdivision guide width mismatch"
        );
        ensure!(
            !matches!(self.kind, DecoderKind::Shape) || guides.is_empty(),
            "shape decoder predicts its own guides"
        );
        let borrowed: Vec<_> = guides
            .iter()
            .map(|g| SubdivisionGuide {
                coordinates: &g.coordinates,
                logits: &g.features,
            })
            .collect();
        let mut generated = Vec::new();
        let (sites, features) =
            self.sparse_decode_rows(coordinates, latent, &borrowed, |_, (parents, logits), _| {
                if matches!(self.kind, DecoderKind::Shape) {
                    generated.push(SparseDecoderOutput {
                        coordinates: parents.coordinates().to_vec(),
                        features: logits.to_vec(),
                        channels: 8,
                    });
                }
                Ok(())
            })?;
        let mut output = self.output_head(&sites, features)?;
        if matches!(self.kind, DecoderKind::Texture) {
            channel_transform(&self.device, &mut output, [&[0.5; 6], &[0.5; 6]], false)?;
        }
        Ok((sites, output, generated))
    }
    pub fn shape_mesh_device(
        &self,
        coordinates: Vec<[i32; 4]>,
        latent: CudaSlice<f32>,
        resolution: u32,
    ) -> Result<(crate::sparse::SparseMesh, Vec<SparseDecoderOutput>)> {
        ensure!(
            matches!(self.kind, DecoderKind::Shape),
            "shape decoder required for mesh"
        );
        finite_copy(&self.device, &latent, &latent)?;
        let (sites, fields, guides) = self.decode_fields(coordinates, latent, &[])?;
        let mesh = super::sparse::dual_grid_mesh(
            &self.device,
            sites,
            fields,
            resolution,
            [self.initial_free, self.device_budget],
        )?;
        Ok((mesh, guides))
    }
    pub fn shape_mesh(
        &self,
        coordinates: Vec<[i32; 4]>,
        latent: &[f32],
        resolution: u32,
    ) -> Result<crate::sparse::SparseMesh> {
        ensure!(
            matches!(self.kind, DecoderKind::Shape),
            "shape decoder required for mesh"
        );
        let (sites, features) = self.sparse_decode(coordinates, latent, &[], |_, _, _| Ok(()))?;
        let fields = self.output_head(&sites, features)?;
        super::sparse::dual_grid_mesh(
            &self.device,
            sites,
            fields,
            resolution,
            [self.initial_free, self.device_budget],
        )
    }
    pub fn cascade_coordinates(
        &self,
        coordinates: Vec<[i32; 4]>,
        latent: &[f32],
        requested_resolution: u32,
        token_limit: usize,
    ) -> Result<(u32, SparseSites)> {
        ensure!(
            matches!(self.kind, DecoderKind::Shape),
            "shape decoder required for cascade"
        );
        ensure!(
            matches!(requested_resolution, 1024 | 1536) && token_limit > 0,
            "invalid cascade resolution or token limit"
        );
        let (sites, features) = self.sparse_decode(coordinates, latent, &[], |_, _, _| Ok(()))?;
        drop(features);
        Self::cascade_grid(&sites, requested_resolution, token_limit)
    }
    pub fn cascade_coordinates_device(
        &self,
        coordinates: Vec<[i32; 4]>,
        latent: CudaSlice<f32>,
        requested_resolution: u32,
        token_limit: usize,
    ) -> Result<(u32, SparseSites)> {
        ensure!(
            matches!(self.kind, DecoderKind::Shape),
            "shape decoder required for cascade"
        );
        finite_copy(&self.device, &latent, &latent)?;
        let (sites, features) =
            self.sparse_decode_rows(coordinates, latent, &[], |_, _, _| Ok(()))?;
        drop(features);
        Self::cascade_grid(&sites, requested_resolution, token_limit)
    }
    fn cascade_grid(
        sites: &SparseSites,
        requested_resolution: u32,
        token_limit: usize,
    ) -> Result<(u32, SparseSites)> {
        ensure!(
            matches!(requested_resolution, 1024 | 1536) && token_limit > 0,
            "invalid cascade resolution or token limit"
        );
        let mut resolution = requested_resolution;
        loop {
            let quantized = sites.round_unique(512, resolution / 16)?;
            if quantized.coordinates().len() < token_limit || resolution == 1024 {
                return Ok((resolution, quantized));
            }
            resolution -= 128;
        }
    }
    fn output_head(&self, sites: &SparseSites, features: CudaSlice<f16>) -> Result<CudaSlice<f32>> {
        let d = &self.device;
        let rows = sites.coordinates().len();
        let channels = features.len() / rows;
        let output_channels = self
            .weights
            .float
            .get("output_layer.bias")
            .context("shape output bias missing")?
            .len();
        let bytes = rows
            .checked_mul(channels + output_channels)
            .and_then(|n| n.checked_mul(size_of::<f32>()))
            .context("shape output workspace overflow")?;
        d.workspace_budget(self.initial_free, self.device_budget, bytes)?;
        let normalized = d.stream.alloc_zeros::<f32>(rows * channels)?;
        self.k.norm(
            d,
            [ptr(&features, d), ptr(&normalized, d)],
            [rows, channels],
            [0, 0],
            [false, true],
            1e-5,
        )?;
        drop(features);
        let mut output = d.stream.alloc_zeros::<f32>(rows * output_channels)?;
        self.linear32(
            "output_layer",
            &normalized,
            &mut output,
            [rows, channels, output_channels],
        )?;
        drop(normalized);
        Ok(output)
    }
    fn dense_conv(&self, name: &str, io: [u64; 2], shape: [usize; 3], half: bool) -> Result<()> {
        let [side, ci, co] = shape;
        let n = side.pow(3);
        let width = 27 * ci;
        let d = &self.device;
        let columns = self.columns.as_ref().context("dense columns unavailable")?;
        let bytes = if half { 2 } else { 4 };
        let tile = (columns.len() / (width * bytes)).min(n);
        ensure!(tile > 0, "dense conv cannot fit one row");
        let (weight, bias) = if half {
            (
                ptr(
                    self.weights
                        .half
                        .get(&format!("{name}.weight"))
                        .context("dense F16 weight missing")?,
                    d,
                ),
                ptr(
                    self.weights
                        .half
                        .get(&format!("{name}.bias"))
                        .context("dense F16 bias missing")?,
                    d,
                ),
            )
        } else {
            (
                ptr(
                    self.weights
                        .float
                        .get(&format!("{name}.weight"))
                        .context("dense F32 weight missing")?,
                    d,
                ),
                ptr(
                    self.weights
                        .float
                        .get(&format!("{name}.bias"))
                        .context("dense F32 bias missing")?,
                    d,
                ),
            )
        };
        for first in (0..n).step_by(tile) {
            let rows = tile.min(n - first);
            let y = io[1] + (first * co * bytes) as u64;
            unsafe {
                d.stream
                    .launch_builder(if half { &self.k.conv16 } else { &self.k.conv32 })
                    .arg(&io[0])
                    .arg(columns)
                    .arg(&(ci as u32))
                    .arg(&(side as u32))
                    .arg(&(side as u32))
                    .arg(&(side as u32))
                    .arg(&3u32)
                    .arg(&(first as u64))
                    .arg(&(rows as u32))
                    .arg(&0u32)
                    .launch(flat(rows * width))?;
            }
            if half {
                self.blas.linear_f16(
                    matrix::<f16>(weight, co, width)?,
                    matrix(ptr(columns, d), rows, width)?,
                    &mut TensorMut(matrix(y, rows, co)?),
                    false,
                )?;
            } else {
                self.blas.linear_f32(
                    matrix::<f32>(weight, co, width)?,
                    matrix(ptr(columns, d), rows, width)?,
                    &mut TensorMut(matrix(y, rows, co)?),
                )?;
            }
            self.k.bias(d, y, bias, [rows, co], false, !half)?;
        }
        Ok(())
    }
    fn dense_residual(&self, name: &str, x: &CudaSlice<f16>, side: usize, c: usize) -> Result<()> {
        let d = &self.device;
        let rows = side.pow(3);
        let a = d.stream.alloc_zeros::<f16>(rows * c)?;
        let b = d.stream.alloc_zeros::<f16>(rows * c)?;
        self.norm16(Some(&format!("{name}.norm1")), x, &a, rows, c, 1e-5)?;
        self.k.act(d, ptr(&a, d), rows, c, false)?;
        self.dense_conv(
            &format!("{name}.conv1"),
            [ptr(&a, d), ptr(&b, d)],
            [side, c, c],
            true,
        )?;
        self.norm16(Some(&format!("{name}.norm2")), &b, &a, rows, c, 1e-5)?;
        self.k.act(d, ptr(&a, d), rows, c, false)?;
        self.dense_conv(
            &format!("{name}.conv2"),
            [ptr(&a, d), ptr(&b, d)],
            [side, c, c],
            true,
        )?;
        self.k.add(d, x, &b, &self.ones, [rows, c])
    }
    pub fn structure(&self, latent_chw: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            matches!(self.kind, DecoderKind::Structure),
            "structure decoder required"
        );
        ensure!(
            latent_chw.len() == 8 * 16usize.pow(3) && latent_chw.iter().all(|x| x.is_finite()),
            "structure latent must be finite [8,16,16,16]"
        );
        self.device.workspace_budget(
            self.initial_free,
            self.device_budget,
            64usize.pow(3) * 32 * 8,
        )?;
        let rows = 16usize.pow(3);
        let input: Vec<f32> = (0..rows * 8)
            .map(|i| latent_chw[(i % 8) * rows + i / 8])
            .collect();
        let d = &self.device;
        let input = d.stream.clone_htod(&input)?;
        self.structure_rows(input)
    }
    pub fn structure_device(&self, latent_chw: CudaSlice<f32>) -> Result<Vec<f32>> {
        ensure!(
            matches!(self.kind, DecoderKind::Structure) && latent_chw.len() == 8 * 16usize.pow(3),
            "structure latent must be [8,16,16,16]"
        );
        let d = &self.device;
        finite_copy(d, &latent_chw, &latent_chw)?;
        d.workspace_budget(
            self.initial_free,
            self.device_budget,
            64usize.pow(3) * 32 * 8,
        )?;
        let rows = 16usize.pow(3);
        let input = d.stream.alloc_zeros::<f32>(latent_chw.len())?;
        let transpose = cubin::module(&d.ctx, "q38_common")?.load_function("transpose_f32")?;
        unsafe {
            d.stream
                .launch_builder(&transpose)
                .arg(&latent_chw)
                .arg(&input)
                .arg(&8i64)
                .arg(&(rows as i64))
                .launch(flat(input.len()))?;
        }
        drop(latent_chw);
        self.structure_rows(input)
    }
    fn structure_rows(&self, input: CudaSlice<f32>) -> Result<Vec<f32>> {
        #[cfg(test)]
        self.decode_passes.set(self.decode_passes.get() + 1);
        let d = &self.device;
        let rows = 16usize.pow(3);
        let float = d.stream.alloc_zeros::<f32>(rows * 512)?;
        self.dense_conv(
            "input_layer",
            [ptr(&input, d), ptr(&float, d)],
            [16, 8, 512],
            false,
        )?;
        drop(input);
        let mut x = d.stream.alloc_zeros::<f16>(rows * 512)?;
        self.k
            .cast(d, [ptr(&float, d), ptr(&x, d)], rows * 512, false)?;
        drop(float);
        for block in 0..2 {
            self.dense_residual(&format!("middle_block.{block}"), &x, 16, 512)?;
        }
        let mut side = 16usize;
        let mut c = 512;
        for block in 0..8 {
            if block == 2 || block == 5 {
                let co = c / 4;
                let expanded = d.stream.alloc_zeros::<f16>(side.pow(3) * co * 8)?;
                self.dense_conv(
                    &format!("blocks.{block}.conv"),
                    [ptr(&x, d), ptr(&expanded, d)],
                    [side, c, co * 8],
                    true,
                )?;
                drop(x);
                let next = d.stream.alloc_zeros::<f16>((side * 2).pow(3) * co)?;
                unsafe {
                    d.stream
                        .launch_builder(&self.k.shuffle)
                        .arg(&expanded)
                        .arg(&next)
                        .arg(&(side as u32))
                        .arg(&(co as u32))
                        .launch(flat(next.len()))?;
                }
                x = next;
                side *= 2;
                c = co;
            } else {
                self.dense_residual(&format!("blocks.{block}"), &x, side, c)?;
            }
        }
        let rows = side.pow(3);
        let float = d.stream.alloc_zeros::<f32>(rows * c)?;
        self.k
            .cast(d, [ptr(&x, d), ptr(&float, d)], rows * c, true)?;
        drop(x);
        let normalized = d.stream.alloc_zeros::<f32>(rows * c)?;
        self.k.norm(
            d,
            [ptr(&float, d), ptr(&normalized, d)],
            [rows, c],
            self.weights.affine(d, Some("out_layer.0"))?,
            [true, true],
            1e-5,
        )?;
        drop(float);
        self.k.act(d, ptr(&normalized, d), rows, c, true)?;
        let logits = d.stream.alloc_zeros::<f32>(rows)?;
        self.dense_conv(
            "out_layer.2",
            [ptr(&normalized, d), ptr(&logits, d)],
            [side, c, 1],
            false,
        )?;
        let logits = d.stream.clone_dtoh(&logits)?;
        ensure!(
            logits.iter().all(|x| x.is_finite()),
            "nonfinite structure logits"
        );
        Ok(logits)
    }
    pub fn structure_sites(logits: &[f32], side: usize, target: usize) -> Result<SparseSites> {
        ensure!(
            side > 0
                && target > 0
                && side.is_multiple_of(target)
                && side.checked_pow(3) == Some(logits.len())
                && logits.iter().all(|x| x.is_finite()),
            "invalid occupancy grid"
        );
        let factor = side / target;
        let mut points = Vec::new();
        for x in 0..target {
            for y in 0..target {
                for z in 0..target {
                    let mut occupied = false;
                    for dx in 0..factor {
                        for dy in 0..factor {
                            for dz in 0..factor {
                                occupied |= logits[((x * factor + dx) * side + y * factor + dy)
                                    * side
                                    + z * factor
                                    + dz]
                                    > 0.0;
                            }
                        }
                    }
                    if occupied {
                        points.push([0, i32::try_from(x)?, i32::try_from(y)?, i32::try_from(z)?]);
                    }
                }
            }
        }
        SparseSites::new(points)
    }
}
