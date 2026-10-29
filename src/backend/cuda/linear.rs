//! Typed Linear views; all BF16 plans execute through the shared Blas handle.
//! Typed linear views with explicit native input and output formats.
use super::{
    contracts::{RowDomain, Scratch, Tensor, TensorMut},
    weights::WeightCatalog,
};
#[cfg(feature = "cuda")]
use crate::backend::gpu::CudaSlice;
use crate::backend::gpu::{CudaContext, CudaFunction, CudaStream, DevicePtr, PushKernelArg};
use crate::fp8::Fp8ScaleLayout;
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::sync::Arc;
/// Native BF16 operands with an unrounded FP32 output on the shared skinny tile.
pub struct Bf16F32Plan {
    kernel: CudaFunction,
    input: Tensor<bf16, 2>,
    weight: Tensor<bf16, 2>,
    output: Tensor<f32, 2>,
    shape: [i32; 3],
}

fn bf16_f32_shape(
    input: Tensor<bf16, 2>,
    weight: Tensor<bf16, 2>,
    output: Tensor<f32, 2>,
) -> Result<[i32; 3]> {
    use super::contracts::{dense_layout, spans_disjoint};
    input.validate()?;
    weight.validate()?;
    output.validate()?;
    let [rows, width] = input.shape;
    let columns = weight.shape[0];
    ensure!(
        (1..=16).contains(&rows)
            && width.is_multiple_of(256)
            && width <= i32::MAX as u32
            && columns <= i32::MAX as u32,
        "BF16 to F32 skinny projection requires M1..16 and K divisible by256"
    );
    ensure!(
        input.stride == dense_layout(input.shape)?.0
            && weight.stride == dense_layout(weight.shape)?.0
            && output.stride == dense_layout(output.shape)?.0
            && weight.shape[1] == width
            && output.shape == [rows, columns],
        "BF16 to F32 projection shape/stride mismatch"
    );
    let spans = [
        (input.ptr, input.span_bytes),
        (weight.ptr, weight.span_bytes),
        (output.ptr, output.span_bytes),
    ];
    ensure!(
        input.ptr.is_multiple_of(16) && weight.ptr.is_multiple_of(16),
        "BF16 to F32 projection requires16byte input/weight alignment"
    );
    for (i, &(ptr, bytes)) in spans.iter().enumerate() {
        for &(other, count) in &spans[i + 1..] {
            ensure!(
                spans_disjoint(ptr, bytes, other, count)?,
                "BF16 to F32 projection overlap"
            );
        }
    }
    Ok([rows as i32, columns as i32, width as i32])
}

impl Bf16F32Plan {
    pub fn new(
        ctx: &Arc<CudaContext>,
        input: Tensor<bf16, 2>,
        weight: Tensor<bf16, 2>,
        output: Tensor<f32, 2>,
    ) -> Result<Self> {
        let shape = bf16_f32_shape(input, weight, output)?;
        Ok(Self {
            kernel: super::cubin::module(ctx, "gemm_skinny")?
                .load_function("gemm_skinny_bf16_f32")?,
            input,
            weight,
            output,
            shape,
        })
    }

    pub fn run(&self, stream: &Arc<CudaStream>) -> Result<()> {
        let [rows, columns, width] = self.shape;
        let leading = i64::from(width);
        unsafe {
            stream
                .launch_builder(&self.kernel)
                .arg(&self.weight.ptr)
                .arg(&self.input.ptr)
                .arg(&self.output.ptr)
                .arg(&columns)
                .arg(&width)
                .arg(&leading)
                .arg(&rows)
                .launch(crate::backend::gpu::LaunchConfig {
                    grid_dim: ((columns as u32).div_ceil(16), 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: if cfg!(feature = "cuda") { 40960 } else { 0 },
                })?;
        }
        Ok(())
    }
}

/// Explicit visible bias boundary; the caller's numerical contract selects it.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum LinearBiasMode {
    BeforeOutputRound,
    AfterOutputRound,
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum Fp8ScaleRef {
    Bf16(Tensor<bf16, 2>),
    F32(Tensor<f32, 2>),
    E8m0(Tensor<u8, 2>),
}

impl Fp8ScaleRef {
    fn ptr(self) -> u64 {
        match self {
            Self::Bf16(t) => t.ptr,
            Self::F32(t) => t.ptr,
            Self::E8m0(t) => t.ptr,
        }
    }

    fn disjoint<T: Copy>(self, other: &Tensor<T, 2>) -> Result<()> {
        match self {
            Self::Bf16(t) => disjoint(&t, other),
            Self::F32(t) => disjoint(&t, other),
            Self::E8m0(t) => disjoint(&t, other),
        }
    }
    /// # Safety
    /// The native allocation matches dtype and remains live through every use of this view.
    unsafe fn new(
        ptr: u64,
        shape: [u32; 2],
        bytes: u64,
        dtype: safetensors::Dtype,
    ) -> Result<Self> {
        let stride = [u64::from(shape[1]), 1];
        Ok(match dtype {
            safetensors::Dtype::BF16 => {
                Self::Bf16(unsafe { Tensor::new(ptr, shape, stride, bytes)? })
            }
            safetensors::Dtype::F32 => {
                Self::F32(unsafe { Tensor::new(ptr, shape, stride, bytes)? })
            }
            safetensors::Dtype::F8_E8M0 => {
                Self::E8m0(unsafe { Tensor::new(ptr, shape, stride, bytes)? })
            }
            _ => anyhow::bail!("unsupported native FP8 scale dtype {dtype:?}"),
        })
    }

    pub(crate) fn validate(self, codes: &Tensor<u8, 2>, layout: Fp8ScaleLayout) -> Result<()> {
        match self {
            Self::Bf16(scale) => {
                Self::validate_tensor(codes, scale, layout, safetensors::Dtype::BF16)
            }
            Self::F32(scale) => {
                Self::validate_tensor(codes, scale, layout, safetensors::Dtype::F32)
            }
            Self::E8m0(scale) => {
                Self::validate_tensor(codes, scale, layout, safetensors::Dtype::F8_E8M0)
            }
        }
    }

    fn validate_tensor<T: Copy>(
        codes: &Tensor<u8, 2>,
        scale: Tensor<T, 2>,
        layout: Fp8ScaleLayout,
        dtype: safetensors::Dtype,
    ) -> Result<()> {
        codes.validate()?;
        scale.validate()?;
        ensure!(
            codes.shape.iter().all(|&n| n <= i32::MAX as u32)
                && codes.stride == [u64::from(codes.shape[1]), 1]
                && codes.span_bytes == u64::from(codes.shape[0]) * u64::from(codes.shape[1]),
            "invalid native FP8 code extent"
        );
        let shape = layout.scale_shape(codes.shape[0], codes.shape[1]);
        ensure!(
            dtype == layout.scale_dtype()
                && scale.shape.map(|n| n as usize) == shape
                && scale.stride == [shape[1] as u64, 1],
            "native FP8 scale dtype/grid mismatch: view={dtype:?} layout={layout:?} shape={:?} stride={:?}",
            scale.shape,
            scale.stride
        );
        let bytes = u64::from(scale.shape[0])
            .checked_mul(u64::from(scale.shape[1]))
            .and_then(|n| n.checked_mul(size_of::<T>() as u64))
            .context("native FP8 scale byte extent overflow")?;
        ensure!(
            scale.span_bytes == bytes,
            "native FP8 scale allocation extent mismatch"
        );
        super::contracts::no_overlap(codes, &scale, "FP8 codes", "FP8 scales")
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum LinearWeightRef {
    Bf16(Tensor<bf16, 2>),
    Fp8 {
        codes: Tensor<u8, 2>,
        scale: Fp8ScaleRef,
        layout: Fp8ScaleLayout,
    },
}
pub(crate) struct Fp8Ops {
    #[cfg(feature = "cuda")]
    quant_block_f32: CudaFunction,
    #[cfg(feature = "cuda")]
    dequant: [CudaFunction; 2],
    #[cfg(feature = "cuda")]
    small: [Vec<CudaFunction>; 2],
    #[cfg(feature = "rocm")]
    channel: [CudaFunction; 2],
    #[cfg(feature = "rocm")]
    block: [CudaFunction; 2],
    #[cfg(feature = "rocm")]
    context: Arc<CudaContext>,
    fake_quant: CudaFunction,
    native_block: [[CudaFunction; 2]; 2],
}

/// Bound native codes/scales and prepared activations; storage remains caller-owned.
pub(crate) struct Fp8BlockPlan {
    kernel: CudaFunction,
    grid: crate::backend::gpu::LaunchConfig,
    pointers: [u64; 5],
    shape: [i32; 3],
}

impl Fp8BlockPlan {
    pub(crate) fn run(&self, stream: &Arc<CudaStream>) -> Result<()> {
        unsafe {
            stream
                .launch_builder(&self.kernel)
                .arg(&self.pointers[0])
                .arg(&self.pointers[1])
                .arg(&self.pointers[2])
                .arg(&self.pointers[3])
                .arg(&self.pointers[4])
                .arg(&self.shape[0])
                .arg(&self.shape[1])
                .arg(&self.shape[2])
                .launch(self.grid)?;
        }
        Ok(())
    }
}

impl Fp8Ops {
    pub(crate) fn bind_block(
        &self,
        weight: LinearWeightRef,
        codes: Tensor<u8, 2>,
        scale: Fp8ScaleRef,
        output: &TensorMut<bf16, 2>,
    ) -> Result<Fp8BlockPlan> {
        use crate::fp8::Fp8BlockScale;
        use safetensors::Dtype;
        let LinearWeightRef::Fp8 {
            codes: weight,
            scale: weight_scale,
            layout,
        } = weight
        else {
            anyhow::bail!("prepared FP8 linear requires native FP8 weights");
        };
        weight_scale.validate(&weight, layout)?;
        let block = layout
            .block_scale()
            .context("native FP8 block layout missing")?;
        let family = match (block.dtype(), block.block()) {
            (Dtype::F32, [128, 128]) => 0,
            (Dtype::F8_E8M0, [32, 32]) => 1,
            _ => anyhow::bail!("unsupported prepared FP8 block layout {layout:?}"),
        };
        scale.validate(
            &codes,
            Fp8ScaleLayout::Block(Fp8BlockScale::new([1, block.block()[1]], block.dtype())?),
        )?;
        output.0.validate()?;
        let [m, k] = codes.shape;
        let n = weight.shape[0];
        ensure!(
            k == weight.shape[1]
                && output.0.shape == [m, n]
                && output.0.stride == [u64::from(n), 1],
            "native FP8 linear activation/output shape mismatch"
        );
        disjoint(&weight, &output.0)?;
        disjoint(&codes, &output.0)?;
        weight_scale.disjoint(&output.0)?;
        scale.disjoint(&output.0)?;
        let tiled = m > 8;
        let blocks = if tiled {
            u64::from(m.div_ceil(16)) * u64::from(n.div_ceil(32))
        } else {
            u64::from(m) * u64::from(n.div_ceil(8))
        };
        ensure!(blocks <= i32::MAX as u64, "native FP8 grid extent overflow");
        Ok(Fp8BlockPlan {
            kernel: self.native_block[family][usize::from(tiled)].clone(),
            grid: super::ops::grid(blocks as usize, if tiled { 128 } else { 256 }),
            pointers: [
                weight.ptr,
                weight_scale.ptr(),
                codes.ptr,
                scale.ptr(),
                output.0.ptr,
            ],
            shape: [m as i32, n as i32, k as i32],
        })
    }
    #[cfg(feature = "cuda")]
    pub(super) fn require_rows(&self, rows: usize) -> Result<()> {
        ensure!(
            rows > 0 && self.small.iter().all(|table| table.len() >= rows),
            "direct FP8 rows={rows} exceeds bound tables={:?}",
            self.small.each_ref().map(Vec::len)
        );
        Ok(())
    }
    #[cfg(feature = "cuda")]
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        Self::for_rows(ctx, crate::mtp::MAX_VERIFY_ROWS)
    }
    #[cfg(feature = "cuda")]
    pub(super) fn for_rows(ctx: &Arc<CudaContext>, rows: usize) -> Result<Self> {
        ensure!(
            (1..=crate::linear_plan::MAX_SKINNY_ROWS).contains(&rows),
            "FP8 skinny row bound={rows} exceeds {}",
            crate::linear_plan::MAX_SKINNY_ROWS
        );
        let module = super::cubin::module(ctx, "fp8_linear")?;
        let load = |suffix: &str| {
            (1..=rows)
                .map(|row| {
                    module.load_function(&if row == 1 {
                        format!("fp8_gemv_bf16{suffix}")
                    } else {
                        format!("fp8_skinny_bf16_m{row}{suffix}")
                    })
                })
                .collect::<std::result::Result<Vec<_>, _>>()
        };
        Ok(Self {
            fake_quant: module.load_function("e4m3_fake_quant_bf16")?,
            quant_block_f32: module.load_function("fp8_quant_f32_128")?,
            dequant: [
                module.load_function("fp8_dequant_bf16")?,
                module.load_function("fp8_dequant_bf16_per_channel")?,
            ],
            small: [load("")?, load("_per_channel")?],
            native_block: [
                [
                    module.load_function("fp8_block_f32_128_decode")?,
                    module.load_function("fp8_block_f32_128_mma")?,
                ],
                [
                    module.load_function("fp8_block_e8m0_32_decode")?,
                    module.load_function("fp8_block_e8m0_32_mma")?,
                ],
            ],
        })
    }
    #[cfg(feature = "rocm")]
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        ctx.require_wmma()?;
        let module = super::cubin::module(ctx, "fp8_linear")?;
        let channel = [
            module.load_function("fp8_channel_bf16_simt")?,
            module.load_function("fp8_channel_bf16_wmma")?,
        ];
        let block = [
            module.load_function("fp8_block_bf16_simt")?,
            module.load_function("fp8_block_bf16_wmma")?,
        ];
        for function in channel.iter().chain(&block) {
            ensure!(
                function.max_threads_per_block()? >= 128
                    && function.occupancy_max_active_blocks_per_multiprocessor(128, 0, None)? > 0,
                "HIP native FP8 requires a resident 128-thread block"
            );
        }
        Ok(Self {
            channel,
            block,
            context: ctx.clone(),
            fake_quant: module.load_function("e4m3_fake_quant_bf16")?,
            native_block: [
                [
                    module.load_function("fp8_block_f32_128_decode")?,
                    module.load_function("fp8_block_f32_128_mma")?,
                ],
                [
                    module.load_function("fp8_block_e8m0_32_decode")?,
                    module.load_function("fp8_block_e8m0_32_mma")?,
                ],
            ],
        })
    }

    #[cfg(feature = "rocm")]
    pub fn bind_native(
        &self,
        weight: LinearWeightRef,
        x: Tensor<bf16, 2>,
        y: &TensorMut<bf16, 2>,
        invariant: bool,
    ) -> Result<Fp8Projection> {
        let LinearWeightRef::Fp8 {
            codes,
            scale,
            layout,
        } = weight
        else {
            anyhow::bail!("HIP native projection requires FP8 weight");
        };
        let scale = Self::check(&codes, scale, layout)?;
        validate_call(codes.shape, x, &y.0)?;
        disjoint(&codes, &y.0)?;
        disjoint(&scale, &y.0)?;
        let [rows, inner] = x.shape;
        let output = codes.shape[0];
        ensure!(
            rows <= i32::MAX as u32,
            "HIP per-channel row extent exceeds i32"
        );
        let tiled = rows > 1 && !invariant;
        let grid = if tiled {
            (output.div_ceil(64), rows.div_ceil(16))
        } else {
            (output.div_ceil(4), rows)
        };
        ensure!(
            grid.1 <= self.context.properties.max_grid_y as u32,
            "HIP per-channel launch exceeds device grid limits"
        );
        Ok(Fp8Projection {
            function: if layout == Fp8ScaleLayout::PerChannel {
                &self.channel
            } else {
                &self.block
            }[usize::from(tiled)]
            .clone(),
            context: self.context.clone(),
            pointers: [codes.ptr, scale.ptr, x.ptr, y.0.ptr],
            shape: [rows as i32, output as i32, inner as i32],
            launch: crate::backend::gpu::LaunchConfig {
                grid_dim: (grid.0, grid.1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            },
        })
    }
    pub fn fake_quant(
        &self,
        stream: &Arc<CudaStream>,
        x: &mut TensorMut<bf16, 2>,
        rows: Option<RowDomain>,
    ) -> Result<()> {
        x.0.validate()?;
        ensure!(
            x.0.shape.iter().all(|&n| n <= i32::MAX as u32)
                && x.0.stride == [u64::from(x.0.shape[1]), 1],
            "invalid E4M3 activation layout"
        );
        if let Some(rows) = rows {
            rows.validate(x.0.shape[0])?;
        }
        let (meta, row_base, row_capacity) = rows.map_or((0, 0, x.0.shape[0]), |rows| {
            (rows.meta.ptr, rows.row_base, rows.row_capacity)
        });
        unsafe {
            stream
                .launch_builder(&self.fake_quant)
                .arg(&x.0.ptr)
                .arg(&(x.0.shape[0] as i32))
                .arg(&(x.0.shape[1] as i32))
                .arg(&meta)
                .arg(&row_base)
                .arg(&row_capacity)
                .launch(super::ops::grid(row_capacity as usize, 256))?;
        }
        Ok(())
    }
    fn check(
        codes: &Tensor<u8, 2>,
        scale: Fp8ScaleRef,
        layout: Fp8ScaleLayout,
    ) -> Result<Tensor<bf16, 2>> {
        scale.validate(codes, layout)?;
        let Fp8ScaleRef::Bf16(scale) = scale else {
            anyhow::bail!(
                "native FP8 scale storage is valid but its execution kernel is not bound: {layout:?}"
            );
        };
        if let Some(descriptor) = layout.block_scale() {
            ensure!(
                descriptor.block() == [128, 128] && descriptor.dtype() == safetensors::Dtype::BF16,
                "CUDA FP8 block kernel does not support block={:?} dtype={:?}",
                descriptor.block(),
                descriptor.dtype()
            );
        }
        Ok(scale)
    }
    #[cfg(feature = "cuda")]
    pub fn materialize(
        &self,
        stream: &Arc<CudaStream>,
        codes: Tensor<u8, 2>,
        scale: Fp8ScaleRef,
        layout: Fp8ScaleLayout,
        dst: &mut TensorMut<bf16, 2>,
    ) -> Result<()> {
        let scale = Self::check(&codes, scale, layout)?;
        dst.0.validate()?;
        ensure!(
            dst.0.shape == codes.shape && dst.0.stride == codes.stride,
            "dequant output mismatch"
        );
        disjoint(&codes, &dst.0)?;
        disjoint(&scale, &dst.0)?;
        let count = u64::from(codes.shape[0]) * u64::from(codes.shape[1]);
        unsafe {
            stream
                .launch_builder(match layout {
                    Fp8ScaleLayout::Block(_) => &self.dequant[0],
                    Fp8ScaleLayout::PerChannel => &self.dequant[1],
                })
                .arg(&codes.ptr)
                .arg(&scale.ptr)
                .arg(&dst.0.ptr)
                .arg(&(codes.shape[0] as i32))
                .arg(&(codes.shape[1] as i32))
                .launch(super::ops::grid(
                    count.div_ceil(256).min(65535) as usize,
                    256,
                ))?;
        }
        Ok(())
    }
    #[cfg(feature = "cuda")]
    pub fn small_gemm(
        &self,
        stream: &Arc<CudaStream>,
        codes: Tensor<u8, 2>,
        scale: Fp8ScaleRef,
        layout: Fp8ScaleLayout,
        x: Tensor<bf16, 2>,
        y: &mut TensorMut<bf16, 2>,
    ) -> Result<()> {
        let scale = Self::check(&codes, scale, layout)?;
        validate_call(codes.shape, x, &y.0)?;
        self.require_rows(x.shape[0] as usize)?;
        disjoint(&codes, &y.0)?;
        disjoint(&scale, &y.0)?;
        unsafe {
            stream
                .launch_builder(
                    &match layout {
                        Fp8ScaleLayout::Block(_) => &self.small[0],
                        Fp8ScaleLayout::PerChannel => &self.small[1],
                    }[x.shape[0] as usize - 1],
                )
                .arg(&codes.ptr)
                .arg(&scale.ptr)
                .arg(&x.ptr)
                .arg(&y.0.ptr)
                .arg(&(codes.shape[0] as i32))
                .arg(&(codes.shape[1] as i32))
                .launch(super::ops::grid(codes.shape[0].div_ceil(4) as usize, 128))?;
        }
        Ok(())
    }
}
#[cfg(feature = "rocm")]
pub(crate) struct Fp8Projection {
    function: CudaFunction,
    context: Arc<CudaContext>,
    pointers: [u64; 4],
    shape: [i32; 3],
    launch: crate::backend::gpu::LaunchConfig,
}
#[cfg(feature = "rocm")]
impl Fp8Projection {
    pub fn run(&self, stream: &Arc<CudaStream>) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.context, stream.context()),
            "HIP per-channel projection context mismatch"
        );
        unsafe {
            stream
                .launch_builder(&self.function)
                .arg(&self.pointers[0])
                .arg(&self.pointers[1])
                .arg(&self.pointers[2])
                .arg(&self.pointers[3])
                .arg(&self.shape[0])
                .arg(&self.shape[1])
                .arg(&self.shape[2])
                .launch(self.launch)?;
        }
        Ok(())
    }
}
pub(crate) fn disjoint<T: Copy, U: Copy, const R: usize, const S: usize>(
    a: &Tensor<T, R>,
    b: &Tensor<U, S>,
) -> Result<()> {
    a.validate()?;
    b.validate()?;
    super::contracts::no_overlap(a, b, "linear input", "linear output")
}
pub(crate) fn validate_call(
    weight: [u32; 2],
    x: Tensor<bf16, 2>,
    y: &Tensor<bf16, 2>,
) -> Result<()> {
    x.validate()?;
    y.validate()?;
    ensure!(
        weight[0] > 0 && weight[1] > 0 && weight.iter().all(|&v| v <= i32::MAX as u32),
        "invalid linear dimensions"
    );
    ensure!(
        x.shape[1] == weight[1] && y.shape == [x.shape[0], weight[0]],
        "linear activation shape mismatch"
    );
    ensure!(
        x.stride == [u64::from(weight[1]), 1] && y.stride == [u64::from(weight[0]), 1],
        "linear requires row-major activations"
    );
    disjoint(&x, y)
}
impl WeightCatalog {
    fn fp8_scale_ref(&self, name: &str, stream: &Arc<CudaStream>) -> Result<Fp8ScaleRef> {
        use super::weights::StoredTensor;
        use safetensors::Dtype;
        let entry = self
            .manifest
            .entries
            .get(name)
            .context("FP8 scale absent from manifest")?;
        ensure!(
            entry.load && entry.shape.len() == 2,
            "FP8 scale must be a loaded matrix: {name}"
        );
        let (ptr, bytes) = match self.tensors.get(name).context("FP8 scale not uploaded")? {
            StoredTensor::Bf16(buffer) => {
                ensure!(
                    entry.dtype == Dtype::BF16,
                    "FP8 scale native storage mismatch: {name}"
                );
                (
                    buffer.device_ptr(stream).0,
                    buffer.len().checked_mul(2).context("scale byte overflow")?,
                )
            }
            StoredTensor::Bytes(buffer) => {
                ensure!(
                    matches!(entry.dtype, Dtype::F32 | Dtype::F8_E8M0),
                    "FP8 scale native storage mismatch: {name}"
                );
                (buffer.device_ptr(stream).0, buffer.len())
            }
        };
        ensure!(
            bytes == entry.bytes,
            "FP8 scale upload extent mismatch: {name}"
        );
        unsafe {
            Fp8ScaleRef::new(
                ptr,
                [
                    u32::try_from(entry.shape[0])?,
                    u32::try_from(entry.shape[1])?,
                ],
                bytes as u64,
                entry.dtype,
            )
        }
    }

    pub fn linear_ref(&self, name: &str, stream: &Arc<CudaStream>) -> Result<LinearWeightRef> {
        let entry = self
            .manifest
            .entries
            .get(name)
            .context("linear tensor missing from manifest")?;
        ensure!(entry.shape.len() == 2, "linear tensor must be a matrix");
        let shape = [
            u32::try_from(entry.shape[0])?,
            u32::try_from(entry.shape[1])?,
        ];
        let stride = [u64::from(shape[1]), 1];
        match self
            .tensors
            .get(name)
            .context("linear tensor not uploaded")?
        {
            super::weights::StoredTensor::Bf16(w) => {
                let (p, _r) = w.device_ptr(stream);
                Ok(LinearWeightRef::Bf16(unsafe {
                    Tensor::new(p, shape, stride, entry.bytes as u64)?
                }))
            }
            super::weights::StoredTensor::Bytes(w) => {
                ensure!(
                    entry.dtype == safetensors::Dtype::F8_E4M3,
                    "native linear weight is not FP8: {name}"
                );
                let scale_name = entry
                    .scale_name
                    .as_ref()
                    .context("FP8 scale descriptor missing")?;
                let layout = self
                    .linear_format
                    .fp8_layout()
                    .context("FP8 tensor declared as BF16")?;
                let scale = self.fp8_scale_ref(scale_name, stream)?;
                let (wp, _rw) = w.device_ptr(stream);
                ensure!(
                    w.len() == entry.bytes,
                    "FP8 code upload extent mismatch: {name}"
                );
                let codes = unsafe { Tensor::new(wp, shape, stride, entry.bytes as u64)? };
                scale.validate(&codes, layout)?;
                Ok(LinearWeightRef::Fp8 {
                    layout,
                    codes,
                    scale,
                })
            }
        }
    }
}
#[cfg(feature = "cuda")]
impl crate::linear_plan::LayerDequantPlan {
    pub(crate) fn weight(&self, name: &str, arena: &Scratch) -> Result<LinearWeightRef> {
        arena.validate(self.bytes)?;
        let slot = self
            .slots
            .get(name)
            .context("missing materialized weight")?;
        Ok(LinearWeightRef::Bf16(unsafe {
            Tensor::new(
                arena.ptr + slot.offset_bytes,
                [slot.n, slot.k],
                [u64::from(slot.k), 1],
                u64::from(slot.n) * u64::from(slot.k) * 2,
            )?
        }))
    }
}

struct BoundProjection {
    resident: LinearWeightRef,
    #[cfg(feature = "cuda")]
    prefill: Option<Tensor<bf16, 2>>,
    #[cfg(feature = "cuda")]
    native: std::cell::RefCell<Option<NativeProjection>>,
    #[cfg(feature = "rocm")]
    calls: std::cell::RefCell<std::collections::BTreeMap<(u32, u64, u64), Fp8Projection>>,
    invariant_rows: usize,
}

#[cfg(feature = "cuda")]
pub(crate) struct NativeProjection {
    weight: LinearWeightRef,
    shape: [u32; 2],
    quant: CudaFunction,
    _keepalive: (CudaSlice<u8>, CudaSlice<f32>),
    codes_view: crate::backend::cuda::contracts::Tensor<u8, 2>,
    scales_view: crate::backend::cuda::contracts::Tensor<f32, 2>,
}

#[cfg(feature = "cuda")]
impl NativeProjection {
    pub(crate) fn new(
        ops: &Fp8Ops,
        stream: &Arc<CudaStream>,
        weight: LinearWeightRef,
        rows: u32,
    ) -> Result<Self> {
        let LinearWeightRef::Fp8 {
            codes,
            scale,
            layout,
        } = weight
        else {
            anyhow::bail!("native projection requires FP8 weights");
        };
        scale.validate(&codes, layout)?;
        ensure!(
            layout.block_scale().is_some_and(
                |block| block.dtype() == safetensors::Dtype::F32 && block.block() == [128, 128]
            ),
            "layer FP8 activation preparation requires F32 block128 scales"
        );
        ensure!(
            rows > 0 && rows <= i32::MAX as u32,
            "invalid native projection rows"
        );
        let k = codes.shape[1];
        let elements = u64::from(rows) * u64::from(k);
        let cells = u64::from(rows) * u64::from(k.div_ceil(128));
        ensure!(
            cells.div_ceil(4) <= i32::MAX as u64,
            "FP8 quantization grid overflow"
        );
        let codes = stream.alloc_zeros(usize::try_from(elements)?)?;
        let scales = stream.alloc_zeros(usize::try_from(cells)?)?;
        let codes_view = super::contracts::view(&codes, stream, 0, [rows, k], [u64::from(k), 1])?;
        let scales_view = super::contracts::view(
            &scales,
            stream,
            0,
            [rows, k.div_ceil(128)],
            [u64::from(k.div_ceil(128)), 1],
        )?;
        Ok(Self {
            weight,
            shape: [rows, k],
            quant: ops.quant_block_f32.clone(),
            _keepalive: (codes, scales),
            codes_view,
            scales_view,
        })
    }

    pub(crate) fn run(
        &self,
        ops: &Fp8Ops,
        stream: &Arc<CudaStream>,
        x: Tensor<bf16, 2>,
        y: &TensorMut<bf16, 2>,
    ) -> Result<()> {
        let shape = match self.weight {
            LinearWeightRef::Fp8 { codes, .. } => codes.shape,
            LinearWeightRef::Bf16(weight) => weight.shape,
        };
        validate_call(shape, x, &y.0)?;
        let [capacity, k] = self.shape;
        let m = x.shape[0];
        ensure!(
            m > 0 && m <= capacity && x.shape[1] == k,
            "native FP8 projection row plan mismatch"
        );
        let codes = unsafe {
            Tensor::new(
                self.codes_view.ptr,
                [m, k],
                [u64::from(k), 1],
                u64::from(m) * u64::from(k),
            )?
        };
        let groups = k.div_ceil(128);
        let scales = unsafe {
            Tensor::new(
                self.scales_view.ptr,
                [m, groups],
                [u64::from(groups), 1],
                u64::from(m) * u64::from(groups) * 4,
            )?
        };
        disjoint(&x, &codes)?;
        disjoint(&x, &scales)?;
        unsafe {
            stream
                .launch_builder(&self.quant)
                .arg(&x.ptr)
                .arg(&codes.ptr)
                .arg(&scales.ptr)
                .arg(&(m as i32))
                .arg(&(k as i32))
                .launch(super::ops::grid(
                    (u64::from(m) * u64::from(groups)).div_ceil(4) as usize,
                    128,
                ))?;
        }
        ops.bind_block(self.weight, codes, Fp8ScaleRef::F32(scales), y)?
            .run(stream)
    }
}
/// Setup-bound views for a layer. Engine retains catalog/arena storage; run does
/// not construct tensor names, allocate scratch or invoke a Lt heuristic.
pub(crate) struct LayerLinearPlan {
    projections: std::collections::BTreeMap<String, BoundProjection>,
    #[cfg(feature = "cuda")]
    materialize: Vec<(LinearWeightRef, TensorMut<bf16, 2>)>,
}

#[derive(Clone, Debug)]
pub(crate) enum BoundFfn {
    Dense,
    #[cfg(feature = "cuda")]
    Native {
        router: Tensor<bf16, 2>,
        shared: Box<[LinearWeightRef; 3]>,
    },
    Moe {
        router: Tensor<bf16, 2>,
        score: crate::model_spec::RouterScore,
        shared: Box<[Tensor<bf16, 2>; 3]>,
        shared_gate: Option<Tensor<bf16, 1>>,
    },
}

impl BoundFfn {
    pub fn bind(spec: &crate::model_spec::ModelSpec, linear: &mut LayerLinearPlan) -> Result<Self> {
        let Some(moe) = spec
            .moe
            .filter(|_| linear.projections.contains_key("mlp.gate.weight"))
        else {
            return Ok(Self::Dense);
        };
        let h = spec.hidden;
        let i = moe.shared_intermediate;
        let router = linear.take_bf16("mlp.gate.weight", [moe.experts, h])?;
        #[cfg(feature = "cuda")]
        if spec.architecture == crate::model_spec::Architecture::Glm5Next {
            let mut take = |role: &str| -> Result<LinearWeightRef> {
                let projection = linear
                    .projections
                    .remove(role)
                    .with_context(|| format!("native FFN projection not bound: {role}"))?;
                ensure!(
                    matches!(
                        projection.resident,
                        LinearWeightRef::Fp8 {
                            scale: Fp8ScaleRef::F32(_),
                            ..
                        }
                    ),
                    "native FFN requires FP32-scale FP8 weights: {role}"
                );
                Ok(projection.resident)
            };
            return Ok(Self::Native {
                router,
                shared: Box::new([
                    take("mlp.shared_experts.gate_proj.weight")?,
                    take("mlp.shared_experts.up_proj.weight")?,
                    take("mlp.shared_experts.down_proj.weight")?,
                ]),
            });
        }
        let shared = Box::new([
            linear.take_bf16("mlp.shared_expert.gate_proj.weight", [i, h])?,
            linear.take_bf16("mlp.shared_expert.up_proj.weight", [i, h])?,
            linear.take_bf16("mlp.shared_expert.down_proj.weight", [h, i])?,
        ]);
        let shared_gate = if moe.shared_gate {
            let gate = linear.take_bf16("mlp.shared_expert_gate.weight", [1, h])?;
            Some(unsafe { Tensor::new(gate.ptr, [h], [1], gate.span_bytes)? })
        } else {
            None
        };
        Ok(Self::Moe {
            router,
            score: moe.score,
            shared,
            shared_gate,
        })
    }

    pub fn prepare(&self, blas: &super::blas::Blas, rows: u32) -> Result<()> {
        match self {
            Self::Dense => Ok(()),
            #[cfg(feature = "cuda")]
            Self::Native { .. } => Ok(()),
            Self::Moe {
                shared,
                shared_gate,
                ..
            } => {
                if let Some(gate) = shared_gate {
                    gate.validate()?;
                }
                for weight in shared.iter() {
                    blas.prepare_fixed(rows, weight.shape[0], weight.shape[1])?;
                }
                Ok(())
            }
        }
    }
}

impl LayerLinearPlan {
    #[cfg(feature = "rocm")]
    pub fn library_shapes(
        &self,
        rows: u32,
    ) -> impl Iterator<Item = crate::backend::setup::LinearShape> + '_ {
        self.projections.values().filter_map(move |projection| {
            let LinearWeightRef::Bf16(weight) = projection.resident else {
                return None;
            };
            (rows as usize > projection.invariant_rows).then_some(
                crate::backend::setup::LinearShape {
                    rows: rows as usize,
                    output: weight.shape[0] as usize,
                    input: weight.shape[1] as usize,
                },
            )
        })
    }

    fn take_bf16(&mut self, role: &str, expected: [u32; 2]) -> Result<Tensor<bf16, 2>> {
        let projection = self
            .projections
            .remove(role)
            .with_context(|| format!("FFN projection not bound: role={role}"))?;
        let LinearWeightRef::Bf16(weight) = projection.resident else {
            anyhow::bail!(
                "FFN projection format mismatch: role={role} format=Fp8E4m3Block expected=Bf16"
            );
        };
        ensure!(
            weight.shape == expected && weight.stride == [u64::from(expected[1]), 1],
            "FFN projection layout mismatch: role={role} shape={:?} stride={:?} expected={expected:?}",
            weight.shape,
            weight.stride
        );
        Ok(weight)
    }

    pub fn bind(
        plan: &crate::linear_plan::LayerDequantPlan,
        catalog: &WeightCatalog,
        arena: Option<&Scratch>,
        stream: &Arc<CudaStream>,
    ) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if plan.bytes > 0 {
            arena
                .context("FP8 layer has no materialization arena")?
                .validate(plan.bytes)?;
        }
        #[cfg(feature = "rocm")]
        ensure!(
            arena.is_none(),
            "HIP native FP8 layers do not use a dequantization arena"
        );
        let prefix = &plan.prefix;
        let mut projections = std::collections::BTreeMap::new();
        #[cfg(feature = "cuda")]
        let mut materialize = Vec::new();
        for (name, entry) in &catalog.manifest.entries {
            if !entry.is_linear_projection() || !name.starts_with(prefix) {
                continue;
            }
            let resident = catalog.linear_ref(name, stream)?;
            #[cfg(feature = "cuda")]
            let prefill = match &resident {
                LinearWeightRef::Bf16(w) => Some(*w),
                LinearWeightRef::Fp8 {
                    scale: Fp8ScaleRef::F32(_),
                    ..
                } => None,
                LinearWeightRef::Fp8 { .. } => {
                    let LinearWeightRef::Bf16(w) = plan.weight(
                        name,
                        arena.context("FP8 layer has no materialization arena")?,
                    )?
                    else {
                        unreachable!()
                    };
                    materialize.push((resident, TensorMut(w)));
                    Some(w)
                }
            };
            let role = &name[prefix.len()..];
            let invariant_rows = if matches!(
                role,
                "linear_attn.in_proj_a.weight" | "linear_attn.in_proj_b.weight"
            ) {
                ensure!(
                    matches!(resident, LinearWeightRef::Bf16(_)),
                    "GDN gate projection requires BF16: {role}"
                );
                crate::linear_plan::MAX_SKINNY_ROWS
            } else {
                0
            };
            projections.insert(
                role.to_owned(),
                BoundProjection {
                    resident,
                    #[cfg(feature = "cuda")]
                    prefill,
                    #[cfg(feature = "cuda")]
                    native: Default::default(),
                    #[cfg(feature = "rocm")]
                    calls: std::cell::RefCell::new(std::collections::BTreeMap::new()),
                    invariant_rows,
                },
            );
        }
        #[cfg(feature = "cuda")]
        ensure!(
            materialize.len() == plan.slots.len(),
            "layer linear plan does not cover materialization slots"
        );
        Ok(Self {
            projections,
            #[cfg(feature = "cuda")]
            materialize,
        })
    }
    #[cfg(feature = "cuda")]
    pub fn prepare_native(&self, ops: &Fp8Ops, stream: &Arc<CudaStream>, rows: u32) -> Result<()> {
        for projection in self.projections.values() {
            if projection.prefill.is_none() {
                let mut native = projection.native.borrow_mut();
                ensure!(native.is_none(), "native projection already bound");
                *native = Some(NativeProjection::new(
                    ops,
                    stream,
                    projection.resident,
                    rows,
                )?);
            }
        }
        Ok(())
    }
    pub fn prepare(&self, blas: &super::blas::Blas, rows: u32) -> Result<()> {
        for p in self.projections.values() {
            #[cfg(feature = "cuda")]
            let Some(weight) = p.prefill else {
                continue;
            };
            #[cfg(feature = "rocm")]
            let LinearWeightRef::Bf16(weight) = p.resident else {
                continue;
            };
            if p.invariant_rows > 0 {
                blas.prepare_skinny(crate::backend::setup::LinearShape {
                    rows: (rows as usize).min(p.invariant_rows),
                    output: weight.shape[0] as usize,
                    input: weight.shape[1] as usize,
                })?;
            }
            if rows as usize > p.invariant_rows {
                blas.prepare_fixed(rows, weight.shape[0], weight.shape[1])?;
            }
        }
        Ok(())
    }
    #[cfg(feature = "rocm")]
    pub fn bind_call(
        &self,
        ops: &Fp8Ops,
        role: &str,
        x: Tensor<bf16, 2>,
        y: &TensorMut<bf16, 2>,
        invariant: bool,
    ) -> Result<()> {
        let projection = self
            .projections
            .get(role)
            .with_context(|| format!("layer projection not bound: {role}"))?;
        if let LinearWeightRef::Bf16(weight) = projection.resident {
            validate_call(weight.shape, x, &y.0)?;
            return disjoint(&weight, &y.0);
        }
        let call = ops.bind_native(projection.resident, x, y, invariant)?;
        let key = (x.shape[0], x.ptr, y.0.ptr);
        let mut calls = projection.calls.borrow_mut();
        ensure!(
            !calls.contains_key(&key),
            "HIP layer projection call already bound: {role}"
        );
        calls.insert(key, call);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn materialize(
        &mut self,
        blas: &super::blas::Blas,
        stream: &Arc<CudaStream>,
    ) -> Result<()> {
        for (weight, dst) in &mut self.materialize {
            let LinearWeightRef::Fp8 {
                codes,
                scale,
                layout,
            } = *weight
            else {
                anyhow::bail!("materialization source requires an FP8 weight");
            };
            blas.fp8_ops()?
                .materialize(stream, codes, scale, layout, dst)?;
        }
        Ok(())
    }
    pub fn run(
        &self,
        blas: &super::blas::Blas,
        role: &str,
        phase: crate::op_contract::ExecutionPhase,
        x: Tensor<bf16, 2>,
        y: &mut TensorMut<bf16, 2>,
    ) -> Result<()> {
        let p = self
            .projections
            .get(role)
            .with_context(|| format!("layer projection not bound: {role}"))?;
        #[cfg(feature = "cuda")]
        if p.prefill.is_none() {
            let native = p.native.borrow();
            return native
                .as_ref()
                .context("native projection setup missing")?
                .run(blas.fp8_ops()?, blas.stream(), x, y);
        }
        #[cfg(feature = "rocm")]
        if let LinearWeightRef::Fp8 { codes, .. } = p.resident {
            validate_call(codes.shape, x, &y.0)?;
            ensure!(
                phase != crate::op_contract::ExecutionPhase::Decode || x.shape[0] == 1,
                "Decode requires M1"
            );
            let calls = p.calls.borrow();
            let call = calls
                .get(&(x.shape[0], x.ptr, y.0.ptr))
                .context("HIP native layer projection is outside the frozen setup closure")?;
            return call.run(blas.stream());
        }
        #[cfg(feature = "rocm")]
        let weight = p.resident;
        #[cfg(feature = "cuda")]
        let weight = if phase == crate::op_contract::ExecutionPhase::Prefill
            && !blas.direct_fp8_rows().contains(&(x.shape[0] as usize))
        {
            LinearWeightRef::Bf16(p.prefill.context("native projection was not bound")?)
        } else {
            p.resident
        };
        blas.linear_typed_bound(phase, weight, x, y, p.invariant_rows)
    }
}

#[cfg(all(test, feature = "rocm"))]
#[test]
fn hip_projection_closure_excludes_native_and_invariant_calls() -> Result<()> {
    use crate::backend::setup::LinearShape;
    let weights = [
        (
            LinearWeightRef::Bf16(unsafe {
                Tensor::new(0x1000, [512, 2048], [2048, 1], 512 * 2048 * 2)?
            }),
            0,
        ),
        (
            LinearWeightRef::Bf16(unsafe {
                Tensor::new(0x1000, [32, 2048], [2048, 1], 32 * 2048 * 2)?
            }),
            8,
        ),
        (
            LinearWeightRef::Fp8 {
                codes: unsafe { Tensor::new(0x1000, [512, 2048], [2048, 1], 512 * 2048)? },
                scale: Fp8ScaleRef::Bf16(unsafe {
                    Tensor::new(0x2000, [512, 1], [1, 1], 512 * 2)?
                }),
                layout: Fp8ScaleLayout::PerChannel,
            },
            0,
        ),
    ];
    let plan = LayerLinearPlan {
        projections: weights
            .into_iter()
            .enumerate()
            .map(|(i, (resident, invariant_rows))| {
                (
                    i.to_string(),
                    BoundProjection {
                        resident,
                        invariant_rows,
                        calls: Default::default(),
                    },
                )
            })
            .collect(),
    };
    for rows in [1, 3, 8, 9, 17, 32] {
        let mut expected = vec![LinearShape {
            rows: rows as usize,
            output: 512,
            input: 2048,
        }];
        if rows > 8 {
            expected.push(LinearShape {
                rows: rows as usize,
                output: 32,
                input: 2048,
            });
        }
        assert_eq!(plan.library_shapes(rows).collect::<Vec<_>>(), expected);
    }
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::backend::cuda::{
        Device, blas::Blas, closure::Closure, ops::Ops, weights::StoredTensor,
    };
    use crate::checkpoint::{TensorEntry, TensorManifest, TensorScope};
    use crate::op_contract::ExecutionPhase;
    use std::{path::PathBuf, rc::Rc};

    #[test]
    fn bf16_f32_projection_bounds() -> Result<()> {
        use super::super::contracts::dense_layout;
        fn view<T: Copy>(ptr: u64, shape: [u32; 2]) -> Result<Tensor<T, 2>> {
            let (stride, n) = dense_layout(shape)?;
            unsafe { Tensor::new(ptr, shape, stride, n * size_of::<T>() as u64) }
        }
        let x = view(0x100000, [16, 5120])?;
        let w = view(0x1000000, [512, 5120])?;
        let y = view(0x2000000, [16, 512])?;
        assert_eq!(bf16_f32_shape(x, w, y)?, [16, 512, 5120]);
        assert!(
            bf16_f32_shape(view(0x100000, [17, 5120])?, w, view(0x2000000, [17, 512])?).is_err()
        );
        assert!(bf16_f32_shape(view(0x100000, [16, 5118])?, w, y).is_err());
        assert!(bf16_f32_shape(x, w, view(0x100000, [16, 512])?).is_err());
        assert!(
            view(0x100002, [16, 5120])
                .and_then(|v| bf16_f32_shape(v, w, y))
                .is_err()
        );
        assert!(bf16_f32_shape(x, w, view(0x2000000, [16, 511])?).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires GPU lease; native BF16 weights with FP32 projection output"]
    fn bf16_f32_projection_gpu() -> Result<()> {
        use super::super::contracts::view;
        const K: usize = 5120;
        const N: usize = 512;
        let ctx = CudaContext::new(0)?;
        let stream = ctx.new_stream()?;
        let inputs: Vec<i32> = (0..16 * K)
            .map(|i| ((i % K * 37 + i / K * 17) % 251 % 15) as i32 - 7)
            .collect();
        let weights: Vec<i32> = (0..N * K)
            .map(|i| ((i % K * 29 + i / K * 13) % 241 % 15) as i32 - 7)
            .collect();
        let mut oracle = Vec::with_capacity(16 * N);
        for row in 0..16 {
            for col in 0..N {
                let sum: i64 = inputs[row * K..(row + 1) * K]
                    .iter()
                    .zip(&weights[col * K..(col + 1) * K])
                    .map(|(&a, &b)| i64::from(a) * i64::from(b))
                    .sum();
                // Every product is an integer multiple of 1/256; absolute partial sum <=5120*49<2^24.
                ensure!(sum.abs() < (1 << 24), "projection exact oracle range");
                oracle.push(sum as f32 / 256.0);
            }
        }
        let x = stream.clone_htod(
            &inputs
                .iter()
                .map(|&n| bf16::from_f32(n as f32 / 16.0))
                .collect::<Vec<_>>(),
        )?;
        let w = stream.clone_htod(
            &weights
                .iter()
                .map(|&n| bf16::from_f32(n as f32 / 16.0))
                .collect::<Vec<_>>(),
        )?;
        let mut cells = 0;
        for rows in [1u32, 2, 5, 8, 16] {
            let output = stream.clone_htod(&vec![127.0f32; rows as usize * N + 16])?;
            let xv = view(&x, &stream, 0, [rows, K as u32], [K as u64, 1])?;
            let wv = view(&w, &stream, 0, [N as u32, K as u32], [K as u64, 1])?;
            let yv = view(&output, &stream, 0, [rows, N as u32], [N as u64, 1])?;
            let plan = Bf16F32Plan::new(&ctx, xv, wv, yv)?;
            plan.run(&stream)?;
            let eager = stream.clone_dtoh(&output)?;
            ensure!(
                eager[..rows as usize * N] == oracle[..rows as usize * N]
                    && eager[rows as usize * N..].iter().all(|&v| v == 127.0),
                "BF16 F32 projection exact/guard DEFECT rows={rows}"
            );
            super::super::moe::testing::capture(&stream, || plan.run(&stream))?.launch()?;
            ensure!(
                stream.clone_dtoh(&output)? == eager,
                "F32 projection graph bits DEFECT rows={rows}"
            );
            // 2026-10-29: the control rounds measured FP32 output; it does not execute BF16 WMMA.
            let rounded: Vec<_> = eager[..rows as usize * N]
                .iter()
                .map(|&value| bf16::from_f32(value))
                .collect();
            let premature = rounded
                .iter()
                .zip(&oracle)
                .filter(|(a, b)| a.to_f32() != **b)
                .count();
            ensure!(
                premature > 0,
                "premature BF16 store control escaped rows={rows}"
            );
            println!(
                "BF16_F32_PROJECTION rows={rows} weights_bf16=[512,5120] input=[{rows},5120] output=[{rows},512] fp64_bits=0 graph_bits=0 premature_round_mismatches={premature}"
            );
            cells += 1;
        }
        println!(
            "PASS BF16_F32_PROJECTION cases={cells} exact_dyadic_fp64=true shape_bits=0 graph_bits=0 guards=true premature_round_controls={cells}"
        );
        Ok(())
    }

    fn native_block_oracle(
        codes: [&[u8]; 2],
        scales: [&[f32]; 2],
        shape: [usize; 3],
        block: usize,
        wrong_scale_cell: bool,
    ) -> Vec<(f64, f64)> {
        let [m, n, k] = shape;
        let groups = k.div_ceil(block);
        (0..m * n)
            .map(|index| {
                let row = index / n;
                let col = index % n;
                let scale_row = if wrong_scale_cell {
                    (col / block + 1) % n.div_ceil(block)
                } else {
                    col / block
                };
                let mut sum = 0.0;
                let mut absolute = 0.0;
                for inner in 0..k {
                    let weight = f64::from(crate::fp8::decode_e4m3(codes[0][col * k + inner]))
                        * f64::from(scales[0][scale_row * groups + inner / block]);
                    let input = f64::from(crate::fp8::decode_e4m3(codes[1][row * k + inner]))
                        * f64::from(scales[1][row * groups + inner / block]);
                    let product = weight * input;
                    sum += product;
                    absolute += product.abs();
                }
                // FP32 reduction bound plus one BF16 rounding unit and FP64 reference roundoff.
                let epsilon = f64::from(f32::EPSILON);
                let gamma =
                    (k + 2 * groups) as f64 * epsilon / (1.0 - (k + 2 * groups) as f64 * epsilon);
                let bound = gamma * absolute + sum.abs() / 256.0 + 1.0e-30;
                (sum, bound)
            })
            .collect()
    }

    #[test]
    #[ignore = "GPU native FP8 block-scale linear correctness"]
    fn fp8_native_block_linear_matches_fp64() -> Result<()> {
        use crate::fp8::Fp8BlockScale;
        use safetensors::Dtype;
        let device = Device::new(0)?;
        let stream = &device.stream;
        let ops = Fp8Ops::new(&device.ctx)?;
        let mut defects = 0;
        let mut cases = 0;
        for (dtype, block, production) in [
            (Dtype::F32, 128usize, [1536usize, 4096usize]),
            (Dtype::F8_E8M0, 32usize, [1280usize, 5120usize]),
        ] {
            let layout = Fp8ScaleLayout::Block(Fp8BlockScale::new([block as u32; 2], dtype)?);
            for [n, k] in [[block * 2 + 3, block * 2 + 5], [block * 2, 32], production] {
                let groups = k.div_ceil(block);
                let weight: Vec<_> = (0..n * k)
                    .map(|i| ((i * 37 + 11) % 67) as u8 | (((i * 13 + 7) % 5 < 2) as u8 * 128))
                    .collect();
                let weight_scales: Vec<_> = (0..n.div_ceil(block) * groups)
                    .map(|i| {
                        let value = 2.0f32.powi(((i * 17 + 3) % 7) as i32 - 4);
                        if dtype == Dtype::F32 {
                            value * (1.0 + (i * 23 % 11) as f32 / 31.0)
                        } else {
                            value
                        }
                    })
                    .collect();
                let weight_scale_bytes: Vec<u8> = if dtype == Dtype::F32 {
                    weight_scales.iter().flat_map(|v| v.to_le_bytes()).collect()
                } else {
                    weight_scales
                        .iter()
                        .map(|v| (v.log2() as i32 + 127) as u8)
                        .collect()
                };
                let wc = stream.clone_htod(&weight)?;
                let ws = stream.clone_htod(&weight_scale_bytes)?;
                let wc_view = unsafe {
                    Tensor::new(
                        wc.device_ptr(stream).0,
                        [n as u32, k as u32],
                        [k as u64, 1],
                        (n * k) as u64,
                    )?
                };
                let ws_view = unsafe {
                    Fp8ScaleRef::new(
                        ws.device_ptr(stream).0,
                        [n.div_ceil(block) as u32, groups as u32],
                        weight_scale_bytes.len() as u64,
                        dtype,
                    )?
                };
                let rows: &[usize] = if [n, k] == production {
                    &[1, 17]
                } else if [n, k] == [block * 2, 32] {
                    &[16]
                } else {
                    &[1, 3, 8, 9, 17, 33]
                };
                for &m in rows {
                    let input: Vec<_> = (0..m * k)
                        .map(|i| ((i * 43 + 19) % 71) as u8 | (((i * 11 + 5) % 7 < 3) as u8 * 128))
                        .collect();
                    let input_scales: Vec<_> = (0..m * groups)
                        .map(|i| {
                            let value = 2.0f32.powi(((i * 19 + 2) % 5) as i32 - 5);
                            if dtype == Dtype::F32 {
                                value * (1.0 + (i * 29 % 13) as f32 / 37.0)
                            } else {
                                value
                            }
                        })
                        .collect();
                    let input_scale_bytes: Vec<u8> = if dtype == Dtype::F32 {
                        input_scales.iter().flat_map(|v| v.to_le_bytes()).collect()
                    } else {
                        input_scales
                            .iter()
                            .map(|v| (v.log2() as i32 + 127) as u8)
                            .collect()
                    };
                    let oracle = native_block_oracle(
                        [&weight, &input],
                        [&weight_scales, &input_scales],
                        [m, n, k],
                        block,
                        false,
                    );
                    let mutant = native_block_oracle(
                        [&weight, &input],
                        [&weight_scales, &input_scales],
                        [m, n, k],
                        block,
                        true,
                    );
                    let control = oracle
                        .iter()
                        .zip(&mutant)
                        .filter(|((reference, bound), (wrong, _))| {
                            (reference - wrong).abs() > *bound
                        })
                        .count();
                    ensure!(control > 0, "wrong scale-cell control not discriminating");
                    let xc = stream.clone_htod(&input)?;
                    let xs = stream.clone_htod(&input_scale_bytes)?;
                    let canary = bf16::from_bits(0x7fc1);
                    let output = stream.clone_htod(&vec![canary; m * n + 16])?;
                    let xc_view = unsafe {
                        Tensor::new(
                            xc.device_ptr(stream).0,
                            [m as u32, k as u32],
                            [k as u64, 1],
                            (m * k) as u64,
                        )?
                    };
                    let xs_view = unsafe {
                        Fp8ScaleRef::new(
                            xs.device_ptr(stream).0,
                            [m as u32, groups as u32],
                            input_scale_bytes.len() as u64,
                            dtype,
                        )?
                    };
                    let output_view = unsafe {
                        TensorMut::new(
                            output.device_ptr(stream).0,
                            [m as u32, n as u32],
                            [n as u64, 1],
                            (m * n * 2) as u64,
                        )?
                    };
                    let plan = ops.bind_block(
                        LinearWeightRef::Fp8 {
                            codes: wc_view,
                            scale: ws_view,
                            layout,
                        },
                        xc_view,
                        xs_view,
                        &output_view,
                    )?;
                    device.finish_upload()?;
                    for graph in [false, true] {
                        if graph {
                            super::super::moe::testing::capture(stream, || plan.run(stream))?
                                .launch()?;
                        } else {
                            plan.run(stream)?;
                        }
                        stream.synchronize()?;
                        let got = stream.clone_dtoh(&output)?;
                        ensure!(
                            got[..m * n].iter().all(|v| v.is_finite()),
                            "non-finite native FP8 output"
                        );
                        ensure!(
                            got[m * n..].iter().all(|v| v.to_bits() == canary.to_bits()),
                            "native FP8 output guard overwritten"
                        );
                        let bad = got
                            .iter()
                            .zip(&oracle)
                            .filter(|(value, (reference, bound))| {
                                (f64::from(value.to_f32()) - reference).abs() > *bound
                            })
                            .count();
                        defects += bad;
                        cases += 1;
                        println!(
                            "FP8_NATIVE_BLOCK_CELL dtype={dtype:?} block={block} weight=[{n},{k}] input=[{m},{k}] graph={graph} defects={bad} scale_cell_control={control}"
                        );
                    }
                }
            }
        }
        println!(
            "{} FP8_NATIVE_BLOCK cases={cases} defects={defects} fp64=true scale_cell_control=true graph_eager=true no_materialization=true",
            if defects == 0 { "PASS" } else { "DEFECT" }
        );
        ensure!(defects == 0, "native FP8 block linear defects={defects}");
        Ok(())
    }

    #[test]
    fn fp8_native_scale_views_check_storage_and_block_geometry() -> Result<()> {
        use crate::fp8::Fp8BlockScale;
        use safetensors::Dtype;
        let mut cases = 0;
        let mut wrong_dtype = 0;
        for (dtype, block) in [
            (Dtype::BF16, [128, 128]),
            (Dtype::F32, [128, 128]),
            (Dtype::F8_E8M0, [32, 32]),
        ] {
            let layout = Fp8ScaleLayout::Block(Fp8BlockScale::new(block, dtype)?);
            for [n, k] in [[1u32, 1u32], [129, 257], [1536, 4096], [32768, 1280]] {
                // Metadata-only views; no address is dereferenced.
                let codes = unsafe {
                    Tensor::new(
                        0x1000,
                        [n, k],
                        [u64::from(k), 1],
                        u64::from(n) * u64::from(k),
                    )?
                };
                let shape = layout.scale_shape(n, k).map(|n| n as u32);
                let bytes =
                    u64::from(shape[0]) * u64::from(shape[1]) * (dtype.bitsize() / 8) as u64;
                let ptr = 0x1_0000_0000;
                let scale = unsafe { Fp8ScaleRef::new(ptr, shape, bytes, dtype)? };
                scale.validate(&codes, layout)?;
                match (scale, dtype) {
                    (Fp8ScaleRef::Bf16(t), Dtype::BF16) => assert_eq!(t.ptr, ptr),
                    (Fp8ScaleRef::F32(t), Dtype::F32) => assert_eq!(t.ptr, ptr),
                    (Fp8ScaleRef::E8m0(t), Dtype::F8_E8M0) => assert_eq!(t.ptr, ptr),
                    _ => panic!("native scale storage type changed"),
                }
                assert_eq!(
                    Fp8Ops::check(&codes, scale, layout).is_ok(),
                    dtype == Dtype::BF16
                );
                for wrong in [Dtype::BF16, Dtype::F32, Dtype::F8_E8M0]
                    .into_iter()
                    .filter(|d| *d != dtype)
                {
                    let wrong_bytes =
                        u64::from(shape[0]) * u64::from(shape[1]) * (wrong.bitsize() / 8) as u64;
                    let view = unsafe { Fp8ScaleRef::new(ptr, shape, wrong_bytes, wrong)? };
                    assert!(view.validate(&codes, layout).is_err());
                    wrong_dtype += 1;
                }
                assert!(unsafe { Fp8ScaleRef::new(ptr, shape, bytes - 1, dtype) }.is_err());
                let oversized = unsafe { Fp8ScaleRef::new(ptr, shape, bytes + 16, dtype)? };
                assert!(oversized.validate(&codes, layout).is_err());
                let overlap = unsafe { Fp8ScaleRef::new(codes.ptr, shape, bytes, dtype)? };
                assert!(overlap.validate(&codes, layout).is_err());
                let mut padded = codes;
                padded.span_bytes += 16;
                assert!(scale.validate(&padded, layout).is_err());
                let mut strided = codes;
                strided.stride[1] = 2;
                assert!(scale.validate(&strided, layout).is_err());
                let mut too_wide = codes;
                too_wide.shape = [1, i32::MAX as u32 + 1];
                too_wide.stride = [u64::from(too_wide.shape[1]), 1];
                too_wide.span_bytes = u64::from(too_wide.shape[1]);
                assert!(scale.validate(&too_wide, layout).is_err());
                let wrong_grid = [shape[0] + 1, shape[1]];
                let wrong_bytes = u64::from(wrong_grid[0])
                    * u64::from(wrong_grid[1])
                    * (dtype.bitsize() / 8) as u64;
                let wrong = unsafe { Fp8ScaleRef::new(ptr, wrong_grid, wrong_bytes, dtype)? };
                assert!(wrong.validate(&codes, layout).is_err());
                cases += 1;
            }
        }
        let codes = unsafe { Tensor::new(0x1000, [129, 257], [257, 1], 129 * 257)? };
        let per_channel = unsafe { Fp8ScaleRef::new(0x1_0000_0000, [129, 1], 258, Dtype::BF16)? };
        per_channel.validate(&codes, Fp8ScaleLayout::PerChannel)?;
        assert!(Fp8Ops::check(&codes, per_channel, Fp8ScaleLayout::PerChannel).is_ok());
        let Fp8ScaleRef::Bf16(mut wrong_stride) = per_channel else {
            unreachable!()
        };
        wrong_stride.stride[1] = 2;
        assert!(
            Fp8ScaleRef::Bf16(wrong_stride)
                .validate(&codes, Fp8ScaleLayout::PerChannel)
                .is_err()
        );
        assert!(unsafe { Fp8ScaleRef::new(0x1001, [1, 1], 4, Dtype::F32) }.is_err());
        assert!(unsafe { Fp8ScaleRef::new(0x1000, [1, 1], 1, Dtype::F8_E4M3) }.is_err());
        assert!(unsafe { Fp8ScaleRef::new(0x1000, [0, 1], 1, Dtype::F8_E8M0) }.is_err());
        println!(
            "PASS FP8_NATIVE_SCALE_VIEWS block_cells={cases} per_channel=1 wrong_dtype_controls={wrong_dtype} bounds_controls=true execution_unbound_f32_e8m0=true no_gpu=true"
        );
        Ok(())
    }

    fn invariant_rows(
        device: &Device,
        blas: &Blas,
        ops: &Ops,
        weight: LinearWeightRef,
        shape: (usize, usize),
    ) -> Result<usize> {
        use crate::backend::cuda::contracts::view;
        use crate::op_contract::ExecutionPhase;
        let (n, k) = shape;
        let rows = crate::linear_plan::MAX_SKINNY_ROWS;
        let format = match weight {
            LinearWeightRef::Bf16(_) => "BF16",
            LinearWeightRef::Fp8 { layout, .. } => match layout {
                Fp8ScaleLayout::Block(_) => "FP8_BLOCK",
                Fp8ScaleLayout::PerChannel => "FP8_PER_CHANNEL",
            },
        };
        let stream = &device.stream;
        let materialized = match weight {
            LinearWeightRef::Bf16(_) => None,
            LinearWeightRef::Fp8 { .. } => Some(stream.alloc_zeros::<bf16>(n * k)?),
        };
        let prefill = match weight {
            LinearWeightRef::Bf16(w) => w,
            LinearWeightRef::Fp8 {
                codes,
                scale,
                layout,
            } => {
                let mut output = TensorMut(view(
                    materialized.as_ref().context("FP8 test arena missing")?,
                    stream,
                    0,
                    [n as u32, k as u32],
                    [k as u64, 1],
                )?);
                blas.fp8_ops()?
                    .materialize(stream, codes, scale, layout, &mut output)?;
                output.read()
            }
        };
        let layer = LayerLinearPlan {
            projections: [(
                "projection.weight".into(),
                BoundProjection {
                    resident: weight,
                    prefill: Some(prefill),
                    native: Default::default(),
                    invariant_rows: 0,
                },
            )]
            .into(),
            materialize: Vec::new(),
        };
        layer.prepare(blas, 1)?;
        let values: Vec<_> = (0..rows * k)
            .map(|i| {
                bf16::from_f32((((i / k) * 71 + (i % k) * 13) % 509) as f32 / 256.0 - 254.0 / 256.0)
            })
            .collect();
        let input = stream.clone_htod(&values)?;
        let input_stride = k.next_multiple_of(16 / size_of::<bf16>());
        let mut serial_values = vec![bf16::ZERO; rows * input_stride];
        for row in 0..rows {
            serial_values[row * input_stride..row * input_stride + k]
                .copy_from_slice(&values[row * k..(row + 1) * k]);
        }
        let serial_input = stream.clone_htod(&serial_values)?;
        let serial_stride = n.next_multiple_of(16 / size_of::<bf16>());
        let expected = stream.alloc_zeros::<bf16>(rows * serial_stride)?;
        let canary = bf16::from_bits(0x7fc1);
        let output = stream.clone_htod(&vec![canary; (rows + 1) * n])?;
        let serial_ids = stream.alloc_zeros::<u32>(rows)?;
        let predictions = stream.clone_htod(&vec![u32::MAX; rows + 1])?;
        let mut valid = stream.clone_htod(&[0i32])?;
        let serial_ptr = serial_ids.device_ptr(stream).0;
        let predicted_ptr = predictions.device_ptr(stream).0;
        let output_ptr = output.device_ptr(stream).0;
        let valid_ptr = valid.device_ptr(stream).0;
        device.finish_upload()?;
        for row in 0..rows {
            layer.run(
                blas,
                "projection.weight",
                ExecutionPhase::Decode,
                view(
                    &serial_input,
                    stream,
                    row * input_stride,
                    [1, k as u32],
                    [k as u64, 1],
                )?,
                &mut TensorMut(view(
                    &expected,
                    stream,
                    row * serial_stride,
                    [1, n as u32],
                    [n as u64, 1],
                )?),
            )?;
            let input_ptr =
                expected.device_ptr(stream).0 + (row * serial_stride * size_of::<bf16>()) as u64;
            let token_ptr = serial_ptr + (row * size_of::<u32>()) as u64;
            // SAFETY: each M1 input covers n logits and each output covers one token.
            unsafe {
                stream
                    .launch_builder(&ops.argmax)
                    .arg(&input_ptr)
                    .arg(&token_ptr)
                    .arg(&(n as i32))
                    .launch(crate::backend::cuda::ops::grid(1, 1024))?;
            }
        }
        let expected: Vec<_> = stream
            .clone_dtoh(&expected)?
            .chunks_exact(serial_stride)
            .flat_map(|row| row[..n].iter().copied())
            .collect();
        let expected_ids = stream.clone_dtoh(&serial_ids)?;
        for (row, &id) in expected.chunks_exact(n).zip(&expected_ids) {
            let (want, _) = row.iter().enumerate().fold(
                (0usize, f32::NEG_INFINITY),
                |(index, value), (next, &logit)| {
                    if logit.to_f32() > value {
                        (next, logit.to_f32())
                    } else {
                        (index, value)
                    }
                },
            );
            ensure!(
                id as usize == want,
                "M1 argmax tie/index oracle id={id} expected={want}"
            );
        }
        let mut defects = 0;
        for m in 1..=rows as u32 {
            stream.memcpy_htod(&[m as i32], &mut valid)?;
            let x = view(&input, stream, 0, [m, k as u32], [k as u64, 1])?;
            let mut y = TensorMut(view(&output, stream, 0, [m, n as u32], [n as u64, 1])?);
            let mut execute = || -> Result<()> {
                layer.run(
                    blas,
                    "projection.weight",
                    ExecutionPhase::Prefill,
                    x,
                    &mut y,
                )?;
                // SAFETY: the logits and token slabs cover all rows; valid contains the live extent.
                unsafe {
                    stream
                        .launch_builder(&ops.argmax_rows)
                        .arg(&output_ptr)
                        .arg(&predicted_ptr)
                        .arg(&(n as i32))
                        .arg(&valid_ptr)
                        .launch(crate::backend::cuda::ops::grid(rows, 1024))?;
                }
                Ok(())
            };
            for graph in [false, true] {
                if graph {
                    use cudarc::driver::sys;
                    stream.synchronize()?;
                    stream.begin_capture(
                        sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                    )?;
                    let launched = execute();
                    let captured = stream.end_capture(sys::CUgraphInstantiate_flags(0));
                    launched?;
                    captured?
                        .context("empty invariant linear graph")?
                        .launch()?;
                } else {
                    execute()?;
                }
                let got = stream.clone_dtoh(&output)?;
                ensure!(
                    got[..m as usize * n].iter().all(|v| v.is_finite()),
                    "nonfinite invariant linear {shape:?} rows={m}"
                );
                ensure!(
                    got[m as usize * n..]
                        .iter()
                        .all(|v| v.to_bits() == canary.to_bits()),
                    "invariant linear wrote past rows={m} weight={shape:?}"
                );
                let mismatches = got[..m as usize * n]
                    .iter()
                    .zip(&expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let ids = stream.clone_dtoh(&predictions)?;
                ensure!(
                    ids[m as usize..].iter().all(|&id| id == u32::MAX),
                    "argmax wrote past rows={m}"
                );
                let id_mismatches = ids[..m as usize]
                    .iter()
                    .zip(&expected_ids)
                    .filter(|(a, b)| a != b)
                    .count();
                defects += usize::from(mismatches != 0 || id_mismatches != 0);
                println!(
                    "{} INVARIANT_LINEAR consumer=layer_plan format={format} weight=[{n},{k}] rows={m} graph={graph} bit_mismatches={mismatches} argmax_id_mismatches={id_mismatches}",
                    if mismatches == 0 && id_mismatches == 0 {
                        "PASS"
                    } else {
                        "DEFECT"
                    }
                );
            }
        }
        let mut wrong = values.clone();
        for row in 1..rows {
            wrong[row * k..(row + 1) * k].copy_from_slice(&values[..k]);
        }
        let wrong = stream.clone_htod(&wrong)?;
        layer.run(
            blas,
            "projection.weight",
            ExecutionPhase::Prefill,
            view(&wrong, stream, 0, [rows as u32, k as u32], [k as u64, 1])?,
            &mut TensorMut(view(
                &output,
                stream,
                0,
                [rows as u32, n as u32],
                [n as u64, 1],
            )?),
        )?;
        let got = stream.clone_dtoh(&output)?;
        let mismatches = got
            .iter()
            .zip(&expected)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        ensure!(
            mismatches > 0,
            "row0-reuse control escaped invariant linear {shape:?}"
        );
        println!(
            "PASS INVARIANT_LINEAR_CONTROL format={format} weight=[{n},{k}] row0_reuse=true bit_mismatches={mismatches}"
        );
        Ok(defects)
    }

    fn invariant_linears(device: &Device, root: &std::path::Path) -> Result<()> {
        use crate::backend::cuda::contracts::view;
        let stream = &device.stream;
        let ops = Ops::new(&device.ctx)?;
        let mut blas = Blas::new(
            device,
            &ops,
            root,
            Rc::new(Closure::new(false, "invariant linear gate".into())),
        )?;
        blas.enable_fp8(&device.ctx)?;
        blas.bind_invariant(&device.ctx, crate::linear_plan::MAX_SKINNY_ROWS)?;
        let mut bind_first = Blas::new(
            device,
            &ops,
            root,
            Rc::new(Closure::new(false, "invariant linear gate".into())),
        )?;
        bind_first.bind_invariant(&device.ctx, crate::linear_plan::MAX_SKINNY_ROWS)?;
        bind_first.enable_fp8(&device.ctx)?;
        bind_first
            .fp8_ops()?
            .require_rows(crate::linear_plan::MAX_SKINNY_ROWS)?;
        let mut defects = 0;
        for (n, k) in [
            (8192, 2048),
            (32, 2048),
            (2048, 4096),
            (248144, 2048),
            (512, 2048),
            (2048, 512),
            (63, 2048),
            (64, 2048),
            (65, 2048),
        ] {
            {
                let values: Vec<_> = (0..n * k)
                    .map(|i| {
                        bf16::from_f32(((i * 37 + i / k * 71) % 257) as f32 / 512.0 - 128.0 / 512.0)
                    })
                    .collect();
                let weight = stream.clone_htod(&values)?;
                let weight = LinearWeightRef::Bf16(view(
                    &weight,
                    stream,
                    0,
                    [n as u32, k as u32],
                    [k as u64, 1],
                )?);
                defects += invariant_rows(device, &blas, &ops, weight, (n, k))?;
            }
            if [63, 64, 65].contains(&n) {
                continue;
            }
            {
                let codes: Vec<_> = (0..n * k)
                    .map(|i| {
                        (0x10 + (i * 37 + i / k * 71) % 80) as u8
                            | if (i / 257) & 1 == 1 { 0x80 } else { 0 }
                    })
                    .collect();
                let codes = stream.clone_htod(&codes)?;
                for layout in [
                    Fp8ScaleLayout::Block(
                        crate::fp8::Fp8BlockScale::new([128, 128], safetensors::Dtype::BF16)
                            .expect("valid BF16 block scale"),
                    ),
                    Fp8ScaleLayout::PerChannel,
                ] {
                    let [scale_rows, scale_cols] = layout.scale_shape(n as u32, k as u32);
                    let scales: Vec<_> = (0..scale_rows * scale_cols)
                        .map(|i| bf16::from_f32(2f32.powi(-7 + (i % 3) as i32)))
                        .collect();
                    let scale = stream.clone_htod(&scales)?;
                    let weight = LinearWeightRef::Fp8 {
                        codes: view(&codes, stream, 0, [n as u32, k as u32], [k as u64, 1])?,
                        scale: Fp8ScaleRef::Bf16(view(
                            &scale,
                            stream,
                            0,
                            [scale_rows as u32, scale_cols as u32],
                            [scale_cols as u64, 1],
                        )?),
                        layout,
                    };
                    let bound = match layout {
                        Fp8ScaleLayout::Block(_) => &blas,
                        Fp8ScaleLayout::PerChannel => &bind_first,
                    };
                    defects += invariant_rows(device, bound, &ops, weight, (n, k))?;
                }
            }
        }
        println!(
            "INVARIANT_LINEAR_SUMMARY shapes=9 bf16_shapes=9 fp8_shapes=6 rows=1..16 eager_graph=true defects={defects}"
        );
        ensure!(defects == 0, "invariant linear DEFECT cells={defects}");
        for k in [128, 130] {
            let n = 32;
            let values: Vec<_> = (0..n * k)
                .map(|i| {
                    bf16::from_f32(((i * 37 + i / k * 71) % 257) as f32 / 512.0 - 128.0 / 512.0)
                })
                .collect();
            let weight = stream.clone_htod(&values)?;
            let weight = LinearWeightRef::Bf16(view(
                &weight,
                stream,
                0,
                [n as u32, k as u32],
                [k as u64, 1],
            )?);
            defects += invariant_rows(device, &blas, &ops, weight, (n, k))?;
        }
        println!(
            "INVARIANT_LINEAR_EVEN_K_SUMMARY shapes=2 rows=1..16 eager_graph=true defects={defects}"
        );
        ensure!(
            defects == 0,
            "even-K invariant linear DEFECT cells={defects}"
        );
        Ok(())
    }

    #[test]
    #[ignore = "GPU window, official FP8 model, T13 and projection reference required"]
    fn channel_projection_gpu() -> Result<()> {
        use super::super::moe::testing::view;
        use crate::fp8::{effective_weight, linear_reference};
        use cudarc::driver::sys;
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let root = PathBuf::from(
            std::env::var_os("FF_Q38_PROBE_DIR").context("FF_Q38_PROBE_DIR missing")?,
        );
        let reference_dir = PathBuf::from(
            std::env::var_os("FF_MINICPM_REFERENCE_DIR")
                .context("FF_MINICPM_REFERENCE_DIR missing")?,
        );
        let model = PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
        let spec = crate::model_spec::ModelSpec::read(&model)?;
        ensure!(
            spec.hidden == 2048
                && spec.q_width == 4096
                && spec.linear_format == crate::model_spec::LinearFormat::Fp8E4m3Channel,
            "invalid FP8 model geometry"
        );
        let checkpoint = crate::checkpoint::MappedCheckpoint::open(&model)?;
        let mut manifest = checkpoint.manifest(&spec, crate::checkpoint::LoadPolicy::TextOnly)?;
        let views = checkpoint.views()?;
        let tap_bytes = std::fs::read(reference_dir.join("fixtures/text_32_mixer.safetensors"))?;
        let taps = safetensors::SafeTensors::deserialize(&tap_bytes)?;
        let reference_bytes = std::fs::read(root.join("projection-reference.safetensors"))?;
        let references = safetensors::SafeTensors::deserialize(&reference_bytes)?;
        let device = Device::new(0)?;
        let stream = &device.stream;
        let ops = Ops::new(&device.ctx)?;
        let mut blas = Blas::new(
            &device,
            &ops,
            &root,
            Rc::new(Closure::new(false, "channel FP8 projection gate".into())),
        )?;
        blas.enable_fp8(&device.ctx)?;
        let mut defects = 0;
        let mut cases = 0;
        // Reference-only BF16 shape/FP32 floors on sm89, sm86 and cross-GPU; rows1..5,8,32.
        for (op, n, k, input_name, floors) in [
            (
                "q",
                8192u32,
                2048u32,
                "layer_03_mixer_in_norm",
                [0.03125f32; 3],
            ),
            (
                "k",
                512,
                2048,
                "layer_03_mixer_in_norm",
                [0.03125, 0.015625, 0.03125],
            ),
            ("v", 512, 2048, "layer_03_mixer_in_norm", [0.015625; 3]),
            ("o", 2048, 4096, "layer_03_o_proj_in", [0.00390625; 3]),
        ] {
            let two_arm = 2.0 * floors.into_iter().fold(0.0f32, f32::max);
            let rounded = bf16::from_f32(two_arm);
            let limit =
                bf16::from_bits(rounded.to_bits() + u16::from(rounded.to_f32() < two_arm)).to_f32();
            let role = format!("self_attn.{op}_proj.weight");
            let name = format!("model.language_model.layers.3.{role}");
            let scale_name = format!("{name}_scale");
            let entry = manifest
                .entries
                .remove(&name)
                .context("projection manifest entry missing")?;
            let scale_entry = manifest
                .entries
                .remove(&scale_name)
                .context("projection scale missing")?;
            let code = views[&entry.shard].tensor(&name)?;
            let scale = views[&scale_entry.shard].tensor(&scale_name)?;
            ensure!(
                code.dtype() == safetensors::Dtype::F8_E4M3
                    && code.shape() == [n as usize, k as usize]
                    && scale.dtype() == safetensors::Dtype::BF16
                    && scale.shape() == [n as usize, 1],
                "invalid published {op} layout"
            );
            let scales = scale
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_le_bytes(*b))
                .collect::<Vec<_>>();
            let host_weight = code
                .data()
                .iter()
                .enumerate()
                .map(|(i, &c)| effective_weight(c, scales[i / k as usize]))
                .collect::<Vec<_>>();
            ensure!(
                host_weight.iter().all(|v| v.is_finite()),
                "nonfinite published {op} weight"
            );
            let input = taps.tensor(input_name)?;
            ensure!(
                input.dtype() == safetensors::Dtype::BF16 && input.shape() == [32, k as usize],
                "invalid T13 projection input"
            );
            let inputs = input
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_le_bytes(*b))
                .collect::<Vec<_>>();
            let host_fake = inputs
                .chunks_exact(k as usize)
                .map(crate::fp8::fake_quant_reference)
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flat_map(|(_, _, f)| f)
                .collect::<Vec<_>>();
            let scalar = linear_reference(&host_weight, &host_fake[..5 * k as usize], k as usize)?;
            let codes = stream.clone_htod(code.data())?;
            let scales = stream.clone_htod(&scales)?;
            let host_materialized = stream.clone_htod(&host_weight)?;
            let dequant = stream.alloc_zeros::<u8>(host_weight.len() * 2)?;
            let bytes = entry.bytes + scale_entry.bytes;
            let entries = [(name.clone(), entry), (scale_name.clone(), scale_entry)].into();
            let small_manifest = TensorManifest {
                entries,
                resident_bytes: bytes,
            };
            let dequant_plan = crate::linear_plan::LayerDequantPlan::from_manifest(
                &small_manifest,
                3,
                spec.linear_format,
            )?;
            let catalog = WeightCatalog {
                linear_format: spec.linear_format,
                tensors: [
                    (name.clone(), StoredTensor::Bytes(codes)),
                    (scale_name.clone(), StoredTensor::Bf16(scales)),
                ]
                .into(),
                manifest: small_manifest,
                bytes,
                upload_content_sha256: None,
                upload_hashed_bytes: None,
                load_wall_ms: 0.0,
            };
            use cudarc::driver::DevicePtr;
            let scratch = Scratch {
                ptr: dequant.device_ptr(stream).0,
                bytes: dequant.len() as u64,
            };
            let mut plan = LayerLinearPlan::bind(&dequant_plan, &catalog, Some(&scratch), stream)?;
            plan.materialize(&blas, stream)?;
            stream.synchronize()?;
            let materialized = stream.clone_dtoh(&dequant)?;
            let weight_bad = materialized
                .iter()
                .copied()
                .ne(host_weight.iter().flat_map(|v| v.to_le_bytes()));
            println!(
                "FP8_MATERIALIZE op={op} stored_weight=[{n},{k}] mismatches_present={weight_bad} {}",
                if weight_bad { "DEFECT" } else { "PASS" }
            );
            defects += usize::from(weight_bad);
            for m in [1u32, 2, 3, 4, 5, 8, 32] {
                plan.prepare(&blas, m)?;
                let expected = if m <= 5 {
                    scalar[..m as usize * n as usize].to_vec()
                } else {
                    let x = stream.clone_htod(&host_fake[..m as usize * k as usize])?;
                    let y = stream.alloc_zeros::<bf16>(m as usize * n as usize)?;
                    blas.linear_typed(
                        ExecutionPhase::Prefill,
                        LinearWeightRef::Bf16(view(
                            &host_materialized,
                            stream,
                            [n, k],
                            [u64::from(k), 1],
                        )?),
                        view(&x, stream, [m, k], [u64::from(k), 1])?,
                        &mut TensorMut(view(&y, stream, [m, n], [u64::from(n), 1])?),
                    )?;
                    stream.clone_dtoh(&y)?
                };
                let torch = references.tensor(&format!("{op}_bf16_rows{m}"))?;
                ensure!(
                    torch.dtype() == safetensors::Dtype::BF16
                        && torch.shape() == [m as usize, n as usize],
                    "invalid torch projection reference"
                );
                let torch = torch
                    .data()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| bf16::from_le_bytes(*b))
                    .collect::<Vec<_>>();
                for graph in [false, true] {
                    let x = stream.clone_htod(&inputs[..m as usize * k as usize])?;
                    let y = stream.alloc_zeros::<bf16>(m as usize * n as usize)?;
                    let x_view = view(&x, stream, [m, k], [u64::from(k), 1])?;
                    let y_view = view(&y, stream, [m, n], [u64::from(n), 1])?;
                    device.finish_upload()?;
                    let execute = || -> Result<()> {
                        blas.fp8_ops()?
                            .fake_quant(stream, &mut TensorMut(x_view), None)?;
                        plan.run(
                            &blas,
                            &role,
                            if m == 1 {
                                ExecutionPhase::Decode
                            } else {
                                ExecutionPhase::Prefill
                            },
                            x_view,
                            &mut TensorMut(y_view),
                        )
                    };
                    if graph {
                        stream.begin_capture(
                            sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                        )?;
                        let run = execute();
                        let captured = stream.end_capture(sys::CUgraphInstantiate_flags(0));
                        run?;
                        captured?.context("empty FP8 projection graph")?.launch()?;
                    } else {
                        execute()?;
                    }
                    stream.synchronize()?;
                    let got = stream.clone_dtoh(&y)?;
                    ensure!(
                        got.iter().all(|v| v.is_finite()),
                        "nonfinite projection output"
                    );
                    let bit_bad = got
                        .iter()
                        .zip(&expected)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    let max_abs = got
                        .iter()
                        .zip(&torch)
                        .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
                        .fold(0.0f32, f32::max);
                    let mut bad = bit_bad > 0 || max_abs > limit;
                    if op == "o" {
                        let tap = taps.tensor("layer_03_mixer_out")?;
                        ensure!(
                            tap.dtype() == safetensors::Dtype::BF16
                                && tap.shape() == [32, n as usize],
                            "invalid T13 o output tap"
                        );
                        let max_t13 = got
                            .iter()
                            .zip(tap.data().as_chunks::<2>().0.iter())
                            .map(|(a, b)| (a.to_f32() - bf16::from_le_bytes(*b).to_f32()).abs())
                            .fold(0.0f32, f32::max);
                        bad |= max_t13 > limit;
                        println!(
                            "FP8_T13_O rows={m} graph={graph} max_abs={max_t13} limit={limit}"
                        );
                    }
                    println!(
                        "FP8_PROJECTION op={op} stored_weight=[{n},{k}] scale=[{n},1] rows={m} graph={graph} bit_mismatches={bit_bad} torch_max_abs={max_abs} limit={limit} {}",
                        if bad { "DEFECT" } else { "PASS" }
                    );
                    defects += usize::from(bad);
                    cases += 1;
                }
            }
        }
        println!("FP8_PROJECTION_SUMMARY cases={cases} defects={defects}");
        ensure!(defects == 0, "FP8 projection DEFECT cases={defects}");
        Ok(())
    }

    #[test]
    #[ignore = "GPU window and FF_MINICPM_REFERENCE_DIR required"]
    fn token_quantization_gpu() -> Result<()> {
        use super::super::moe::testing::view;
        use crate::fp8::fake_quant_reference;
        use cudarc::driver::sys;
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let root = PathBuf::from(
            std::env::var_os("FF_MINICPM_REFERENCE_DIR")
                .context("FF_MINICPM_REFERENCE_DIR missing")?,
        );
        let bytes = std::fs::read(root.join("fixtures/text_32_mixer.safetensors"))?;
        let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
        let mut inputs = Vec::new();
        for (name, prefix, width) in [
            ("layer_03_mixer_in_norm", "layer_03_attn_in", 2048usize),
            ("layer_03_o_proj_in", "layer_03_o_proj_in", 4096usize),
        ] {
            let input = tensors.tensor(name)?;
            let scales = tensors.tensor(&format!("{prefix}_scale"))?;
            let codes = tensors.tensor(&format!("{prefix}_fp8_q"))?;
            let fake = tensors.tensor(&format!("{prefix}_fake_quant"))?;
            ensure!(
                input.dtype() == safetensors::Dtype::BF16
                    && input.shape() == [32, width]
                    && scales.dtype() == safetensors::Dtype::BF16
                    && scales.shape() == [32, 1]
                    && codes.dtype() == safetensors::Dtype::F8_E4M3
                    && codes.shape() == input.shape()
                    && fake.dtype() == safetensors::Dtype::BF16
                    && fake.shape() == input.shape(),
                "invalid T13 activation fixture {name}"
            );
            let values = input
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| bf16::from_le_bytes([v[0], v[1]]))
                .collect::<Vec<_>>();
            for (row, x) in values.chunks_exact(width).enumerate() {
                let (s, c, f) = fake_quant_reference(x)?;
                ensure!(
                    s.to_le_bytes() == scales.data()[row * 2..row * 2 + 2]
                        && c == codes.data()[row * width..(row + 1) * width]
                        && f.iter()
                            .flat_map(|v| v.to_le_bytes())
                            .eq(fake.data()[row * width * 2..(row + 1) * width * 2]
                                .iter()
                                .copied()),
                    "T13 scalar oracle invalid source={name} row={row}"
                );
            }
            for rows in [1, 2, 3, 4, 5, 8, 32] {
                inputs.push((name, rows, width, values[..rows * width].to_vec()));
            }
        }
        let width = 257usize;
        let tiny = bf16::from_f32(bf16::EPSILON.to_f32() * 0.5);
        let values = (0..9 * width)
            .map(|i| match i / width {
                0 => bf16::ZERO,
                1 => bf16::NEG_ZERO,
                2 => {
                    if i % 17 == 0 {
                        bf16::from_f32(tiny.to_f32() / 262144.0)
                    } else if i % 2 == 0 {
                        tiny
                    } else {
                        -tiny
                    }
                }
                3 => bf16::from_bits((i % 127 + 1) as u16),
                _ => bf16::from_f32(((i * 37 % 101) as f32 - 50.0) / 32.0),
            })
            .collect::<Vec<_>>();
        let (_, _, right) = fake_quant_reference(&values[2 * width..3 * width])?;
        let mut wrong_input = values[2 * width..3 * width].to_vec();
        wrong_input.push(bf16::EPSILON);
        let (_, _, wrong) = fake_quant_reference(&wrong_input)?;
        let changed = right
            .iter()
            .zip(&wrong)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        ensure!(
            changed > 0,
            "max(amax,eps) output control is not discriminating"
        );
        println!("PASS E4M3_EPS_CONTROL fake_bit_mismatches={changed}");
        inputs.push(("zero_tiny_tail", 9, width, values));
        let device = Device::new(0)?;
        let stream = &device.stream;
        let op = Fp8Ops::new(&device.ctx)?;
        let mut defects = 0;
        let mut cases = 0;
        for (name, rows, width, values) in inputs {
            let mut want = Vec::new();
            for row in values.chunks_exact(width) {
                want.extend(fake_quant_reference(row)?.2);
            }
            for graph in [false, true] {
                let x = stream.clone_htod(&values)?;
                let x_view = view(&x, stream, [rows as u32, width as u32], [width as u64, 1])?;
                device.finish_upload()?;
                let execute = || op.fake_quant(stream, &mut TensorMut(x_view), None);
                if graph {
                    stream.begin_capture(
                        sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                    )?;
                    let run = execute();
                    let captured = stream.end_capture(sys::CUgraphInstantiate_flags(0));
                    run?;
                    captured?
                        .context("empty activation quantization graph")?
                        .launch()?;
                } else {
                    execute()?;
                }
                stream.synchronize()?;
                let got = stream.clone_dtoh(&x)?;
                ensure!(
                    got.iter().all(|v| v.is_finite()),
                    "nonfinite fake-quant output"
                );
                let bad = got
                    .iter()
                    .zip(&want)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                println!(
                    "E4M3_QUANT source={name} stored_input=[{rows},{width}] graph={graph} fake_bad={bad} {}",
                    if bad == 0 { "PASS" } else { "DEFECT" }
                );
                defects += usize::from(bad != 0);
                cases += 1;
            }
        }
        println!("E4M3_QUANT_SUMMARY cases={cases} defects={defects}");
        ensure!(
            defects == 0,
            "E4M3 activation quantization DEFECT cases={defects}"
        );
        Ok(())
    }

    #[test]
    #[ignore = "GPU window required"]
    fn token_quantization_live_rows_gpu() -> Result<()> {
        use super::super::{contracts::SeqRef, moe::testing::view};
        use crate::fp8::fake_quant_reference;
        use cudarc::driver::{DevicePtr, sys};
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let device = Device::new(0)?;
        let stream = &device.stream;
        let op = Fp8Ops::new(&device.ctx)?;
        let (rows, width) = (32usize, 257usize);
        let scale_sentinel = bf16::from_bits(0x7fc5);
        let mut defects = 0;
        let mut cases = 0;
        for (row_base, row_capacity) in [(0u32, 32u32), (3, 17)] {
            let values = (0..rows * width)
                .map(|i| {
                    let row = i / width;
                    if row < row_base as usize || row >= 20 {
                        scale_sentinel
                    } else {
                        bf16::from_f32((((row * 37 + (i % width) * 11) % 103) as f32 - 51.0) / 32.0)
                    }
                })
                .collect::<Vec<_>>();
            ensure!(
                values[20 * width..].iter().all(|v| v.is_nan()),
                "live-row inactive-NaN fixture invalid"
            );
            for graph in [false, true] {
                let mut x = stream.clone_htod(&values)?;
                let mut metadata = stream.clone_htod(&[3i32, 1, 4, 64, 1, 0, 0, 0])?;
                let domain = RowDomain {
                    meta: SeqRef {
                        ptr: metadata.device_ptr(stream).0,
                    },
                    row_base,
                    row_capacity,
                };
                let x_view = view(&x, stream, [rows as u32, width as u32], [width as u64, 1])?;
                device.finish_upload()?;
                let execute = || op.fake_quant(stream, &mut TensorMut(x_view), Some(domain));
                let captured = if graph {
                    stream.begin_capture(
                        sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                    )?;
                    let run = execute();
                    let captured = stream.end_capture(sys::CUgraphInstantiate_flags(0));
                    run?;
                    Some(captured?.context("empty live-row quantization graph")?)
                } else {
                    None
                };
                for valid in [1i32, 17, 20, 5] {
                    stream.memcpy_htod(&values, &mut x)?;
                    stream.memcpy_htod(&[3, valid, 3 + valid, 64, 1, 0, 0, 0], &mut metadata)?;
                    if let Some(captured) = &captured {
                        captured.launch()?;
                    } else {
                        execute()?;
                    }
                    stream.synchronize()?;
                    let got = stream.clone_dtoh(&x)?;
                    let mut bad = 0;
                    for row in 0..rows {
                        let range = row * width..(row + 1) * width;
                        let active = row >= row_base as usize
                            && row < (row_base + row_capacity) as usize
                            && row < valid as usize;
                        if active {
                            let (_, _, f) = fake_quant_reference(&values[range.clone()])?;
                            ensure!(
                                got[range.clone()].iter().all(|v| v.is_finite()),
                                "nonfinite active quantization output"
                            );
                            bad += got[range]
                                .iter()
                                .zip(&f)
                                .filter(|(a, b)| a.to_bits() != b.to_bits())
                                .count();
                        } else {
                            bad += got[range.clone()]
                                .iter()
                                .zip(&values[range])
                                .filter(|(a, b)| a.to_bits() != b.to_bits())
                                .count();
                        }
                    }
                    println!(
                        "E4M3_LIVE_ROWS stored_input=[{rows},{width}] row_base={row_base} row_capacity={row_capacity} valid={valid} graph={graph} bit_mismatches={bad} {}",
                        if bad == 0 { "PASS" } else { "DEFECT" }
                    );
                    defects += usize::from(bad != 0);
                    cases += 1;
                }
            }
        }
        println!("E4M3_LIVE_ROWS_SUMMARY cases={cases} defects={defects}");
        ensure!(defects == 0, "live-row quantization DEFECT cases={defects}");
        Ok(())
    }

    #[test]
    fn moe_ffn_binding_uses_native_router_and_shared_layouts() -> Result<()> {
        let spec = crate::model_spec::ModelSpec::minicpm(
            &serde_json::from_str(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/minicpm-config.json"
            )))?,
            &serde_json::json!({"eos_token_id": [248044, 248046]}),
        )?;
        let shapes = [
            ("mlp.gate.weight", [256, 2048]),
            ("mlp.shared_expert.gate_proj.weight", [512, 2048]),
            ("mlp.shared_expert.up_proj.weight", [512, 2048]),
            ("mlp.shared_expert.down_proj.weight", [2048, 512]),
            ("mlp.shared_expert_gate.weight", [1, 2048]),
        ];
        for wrong in std::iter::once(None).chain((0..shapes.len()).map(Some)) {
            let mut linear = LayerLinearPlan {
                projections: std::collections::BTreeMap::new(),
                materialize: Vec::new(),
            };
            let mut ptr = 0x1000u64;
            for (index, (role, mut shape)) in shapes.iter().copied().enumerate() {
                if wrong == Some(index) {
                    shape[1] /= 2;
                }
                let bytes = u64::from(shape[0]) * u64::from(shape[1]) * 2;
                let weight = unsafe { Tensor::new(ptr, shape, [u64::from(shape[1]), 1], bytes)? };
                linear.projections.insert(
                    role.into(),
                    BoundProjection {
                        resident: LinearWeightRef::Bf16(weight),
                        prefill: Some(weight),
                        native: Default::default(),
                        invariant_rows: 0,
                    },
                );
                ptr += bytes;
            }
            let bound = BoundFfn::bind(&spec, &mut linear);
            if let Some(index) = wrong {
                ensure!(
                    bound.is_err(),
                    "accepted wrong FFN layout: projection={index}"
                );
            } else {
                let BoundFfn::Moe {
                    router,
                    ref shared,
                    shared_gate,
                    ..
                } = bound?
                else {
                    anyhow::bail!("MoE layer selected the dense FFN plan");
                };
                let shared_gate = shared_gate.context("test FFN plan has no shared-expert gate")?;
                assert_eq!(router.shape, [256, 2048]);
                assert_eq!(
                    shared.map(|weight| weight.shape),
                    [[512, 2048], [512, 2048], [2048, 512]]
                );
                assert_eq!(shared_gate.shape, [2048]);
                assert_eq!(shared[0].ptr, router.ptr + router.span_bytes);
                assert_eq!(shared[1].ptr, shared[0].ptr + shared[0].span_bytes);
                assert_eq!(shared[2].ptr, shared[1].ptr + shared[1].span_bytes);
                assert_eq!(shared_gate.ptr, shared[2].ptr + shared[2].span_bytes);
                assert!(linear.projections.is_empty());
                assert!(linear.materialize.is_empty());
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "coordinator-granted GPU window and FF_Q38_PROBE_DIR required"]
    fn mixer_geometry_linear_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let device = Device::new(0)?;
        let stream = &device.stream;
        let root = PathBuf::from(
            std::env::var_os("FF_Q38_PROBE_DIR").context("FF_Q38_PROBE_DIR is required")?,
        );
        invariant_linears(&device, &root)?;
        let ops = Ops::new(&device.ctx)?;
        let blas = Blas::new(
            &device,
            &ops,
            &root,
            Rc::new(Closure::new(false, "native BF16 mixer gate".into())),
        )?;
        let mut defects = 0;
        let mut cases = 0;
        for (n, k) in [(8192u32, 2048u32), (32, 2048), (129, 257)] {
            let role = "linear_attn.in_proj_qkv.weight";
            let name = format!("model.language_model.layers.0.{role}");
            let values: Vec<_> = (0..n as usize * k as usize)
                .map(|i| {
                    bf16::from_f32(
                        (((i / k as usize) * 37 + i % k as usize * 11) % 53) as f32 / 32.0
                            - 26.0 / 32.0,
                    )
                })
                .collect();
            let weight = stream.clone_htod(&values)?;
            let bytes = values.len() * 2;
            let manifest = TensorManifest {
                entries: [(
                    name.clone(),
                    TensorEntry {
                        dtype: safetensors::Dtype::BF16,
                        shape: vec![n as usize, k as usize],
                        shard: "synthetic.safetensors".into(),
                        bytes,
                        load: true,
                        data_start: 0,
                        data_offsets: [0, bytes as u64],
                        scope: TensorScope::Text,
                        layer: Some(0),
                        role: role.into(),
                        scale_name: None,
                    },
                )]
                .into(),
                resident_bytes: bytes,
            };
            let catalog = WeightCatalog {
                linear_format: crate::model_spec::LinearFormat::Bf16,
                tensors: [(name.clone(), StoredTensor::Bf16(weight))].into(),
                manifest,
                bytes,
                upload_content_sha256: None,
                upload_hashed_bytes: None,
                load_wall_ms: 0.0,
            };
            let dequant = crate::linear_plan::LayerDequantPlan::from_manifest(
                &catalog.manifest,
                0,
                crate::model_spec::LinearFormat::Bf16,
            )?;
            ensure!(
                dequant.bytes == 0 && dequant.slots.is_empty(),
                "native BF16 fixture allocated materialization storage"
            );
            let plan = LayerLinearPlan::bind(&dequant, &catalog, None, stream)?;
            for m in [1u32, 2, 5, 8, 4096] {
                plan.prepare(&blas, m)?;
                let mut inputs = vec![bf16::ZERO; m as usize * k as usize];
                for row in 0..m as usize {
                    inputs[row * k as usize + (row * 17 + 13) % k as usize] = bf16::ONE;
                }
                let input_store = stream.clone_htod(&inputs)?;
                let output = stream.alloc_zeros::<bf16>(m as usize * n as usize)?;
                use cudarc::driver::DevicePtr;
                let input = unsafe {
                    Tensor::new(
                        input_store.device_ptr(stream).0,
                        [m, k],
                        [u64::from(k), 1],
                        (input_store.len() * 2) as u64,
                    )?
                };
                let target = TensorMut(unsafe {
                    Tensor::new(
                        output.device_ptr(stream).0,
                        [m, n],
                        [u64::from(n), 1],
                        (output.len() * 2) as u64,
                    )?
                });
                device.finish_upload()?;
                for (phase, graph) in [
                    (ExecutionPhase::Prefill, false),
                    (ExecutionPhase::Prefill, true),
                ]
                .into_iter()
                .chain((m == 1).then_some((ExecutionPhase::Decode, false)))
                {
                    let execute =
                        || plan.run(&blas, role, phase, input, &mut TensorMut(target.read()));
                    if graph {
                        use cudarc::driver::sys;
                        stream.begin_capture(
                            sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                        )?;
                        let run = execute();
                        let captured = stream.end_capture(sys::CUgraphInstantiate_flags(0));
                        run?;
                        captured?
                            .context("empty native BF16 geometry graph")?
                            .launch()?;
                    } else {
                        execute()?;
                    }
                    stream.synchronize()?;
                    let got = stream.clone_dtoh(&output)?;
                    ensure!(
                        got.iter().all(|v| v.is_finite()),
                        "nonfinite native BF16 linear output"
                    );
                    let mismatches = got
                        .iter()
                        .enumerate()
                        .filter(|(i, v)| {
                            let row = *i / n as usize;
                            let column = (row * 17 + 13) % k as usize;
                            v.to_bits() != values[(*i % n as usize) * k as usize + column].to_bits()
                        })
                        .count();
                    println!(
                        "BF16_LINEAR stored_weight=[{n},{k}] rows={m} phase={phase:?} graph={graph} dequant_bytes=0 mismatches={mismatches} {}",
                        if mismatches == 0 { "PASS" } else { "DEFECT" }
                    );
                    defects += usize::from(mismatches != 0);
                    cases += 1;
                }
            }
        }
        println!("BF16_LINEAR_SUMMARY cases={cases} defects={defects}");
        ensure!(defects == 0, "BF16_LINEAR DEFECT cases={defects}");
        Ok(())
    }
}
