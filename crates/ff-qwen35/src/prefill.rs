use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use ff_core::quant::QuantFormat;
use ff_edge0::gpu::{GpuContext, GpuGdn};
use std::sync::Arc;

pub const BLOCK: usize = 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    Gemv,
    Packed,
    Mma,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gemv => "gemv",
            Self::Packed => "packed",
            Self::Mma => "mma",
        }
    }
}

pub fn supports(text: &crate::config::TextConfig) -> bool {
    text.hidden_size > 0
        && text.hidden_size.is_multiple_of(64)
        && text.hidden_size <= 8192
        && text.intermediate_size > 0
        && text.intermediate_size.is_multiple_of(256)
        && text.intermediate_size <= 32768
}

pub struct Matrix<'a> {
    pub packed: &'a CudaSlice<u32>,
    pub scales: &'a CudaSlice<u16>,
    pub biases: &'a CudaSlice<u16>,
    pub rows: usize,
    pub cols: usize,
}

/// The slab dimensions a packed prefill block needs, derived from the
/// model config once at load.
pub struct BatchGeom {
    pub hidden: usize,
    pub inner: usize,
    pub tokens: usize,
    /// GDN in_proj_qkv output width.
    pub conv: usize,
    /// GDN z/out and attention scratch width.
    pub z: usize,
    /// GDN b/a projection width.
    pub ba: usize,
    /// Attention q output width (heads interleaved with the gate).
    pub q: usize,
    /// Attention k/v output width.
    pub kv: usize,
    pub key_heads: usize,
    pub conv_kernel: usize,
}

impl BatchGeom {
    pub fn of(text: &crate::config::TextConfig, tokens: usize) -> Self {
        Self {
            hidden: text.hidden_size,
            inner: text
                .intermediate_size
                .max(text.conv_dim())
                .max(text.linear_num_value_heads * (text.linear_value_head_dim + 2))
                .max(2 * text.num_attention_heads * text.head_dim)
                .max(2 * text.num_key_value_heads * text.head_dim),
            tokens,
            conv: text.conv_dim(),
            z: text.linear_num_value_heads * text.linear_value_head_dim,
            ba: text.linear_num_value_heads,
            q: 2 * text.num_attention_heads * text.head_dim,
            kv: text.num_key_value_heads * text.head_dim,
            key_heads: if (0..text.num_hidden_layers)
                .any(|layer| text.layer_kind(layer) == crate::config::LayerKind::LinearAttention)
            {
                text.linear_num_key_heads
            } else {
                0
            },
            conv_kernel: text.linear_conv_kernel_dim,
        }
    }
}

/// The token-parallel norm kernels (cuda/prefill_rows.cu): per-row
/// arithmetic identical to the scalar glue norms.
pub(crate) struct RowsKernels {
    norm: CudaFunction,
    add_norm: CudaFunction,
}

/// One qwen_rmsnorm_rows launch's buffers and geometry.
pub(crate) struct NormRows<'a> {
    pub x: &'a CudaSlice<f32>,
    pub w: &'a CudaSlice<f32>,
    pub out: &'a CudaSlice<f32>,
    pub tokens: usize,
    pub n: usize,
    pub eps: f32,
}

/// One qwen_add_rmsnorm_rows launch's buffers and geometry.
pub(crate) struct AddNormRows<'a> {
    pub acc: &'a CudaSlice<f32>,
    pub delta: &'a CudaSlice<f32>,
    pub w: &'a CudaSlice<f32>,
    pub out: &'a CudaSlice<f32>,
    pub tokens: usize,
    pub n: usize,
    pub eps: f32,
}

impl RowsKernels {
    pub(crate) fn load(ctx: &GpuContext) -> Result<Self> {
        let module =
            ff_edge0::kernel_assets::load_module(&ctx.context, &crate::kernel_assets::PREFILL_ROWS)
                .context("prefill_rows module load failed")?;
        Ok(Self {
            norm: module
                .load_function("qwen_rmsnorm_rows")
                .context("qwen_rmsnorm_rows missing")?,
            add_norm: module
                .load_function("qwen_add_rmsnorm_rows")
                .context("qwen_add_rmsnorm_rows missing")?,
        })
    }

    /// out[t] = rmsnorm(x[t]) * (1 + w), one block per token row.
    pub(crate) fn rmsnorm(&self, ctx: &GpuContext, args: NormRows<'_>) -> Result<()> {
        let n_i = i32::try_from(args.n).context("rmsnorm_rows width")?;
        let tokens = u32::try_from(args.tokens).context("rmsnorm_rows tokens")?;
        unsafe {
            ctx.stream
                .launch_builder(&self.norm)
                .arg(args.x)
                .arg(args.w)
                .arg(args.out)
                .arg(&n_i)
                .arg(&args.eps)
                .launch(LaunchConfig {
                    grid_dim: (tokens, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .context("rmsnorm_rows launch failed")?;
        Ok(())
    }

    /// acc[t] += delta[t]; out[t] = rmsnorm(acc[t]) * (1 + w), one block
    /// per token row, thread count matching the scalar kernel's rule.
    pub(crate) fn add_rmsnorm(&self, ctx: &GpuContext, args: AddNormRows<'_>) -> Result<()> {
        let n_i = i32::try_from(args.n).context("add_rmsnorm_rows width")?;
        let tokens = u32::try_from(args.tokens).context("add_rmsnorm_rows tokens")?;
        unsafe {
            ctx.stream
                .launch_builder(&self.add_norm)
                .arg(args.acc)
                .arg(args.delta)
                .arg(args.w)
                .arg(args.out)
                .arg(&n_i)
                .arg(&args.eps)
                .launch(LaunchConfig {
                    grid_dim: (tokens, 1, 1),
                    block_dim: (args.n.min(1024) as u32, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .context("add_rmsnorm_rows launch failed")?;
        Ok(())
    }
}

pub struct Packed {
    stream: Arc<CudaStream>,
    dot: CudaFunction,
    sum: CudaFunction,
}

impl Packed {
    pub fn new(ctx: &GpuContext) -> Result<Self> {
        let module = ff_edge0::kernel_assets::load_module(
            &ctx.context,
            &crate::kernel_assets::INT4_GEMV_WIDE,
        )?;
        Ok(Self {
            stream: ctx.stream.clone(),
            dot: module.load_function("int4_gemv_rows")?,
            sum: module.load_function("int4_gemv_rows_sum")?,
        })
    }

    pub fn project(
        &self,
        matrix: Matrix<'_>,
        x: &CudaSlice<f32>,
        y: &mut (impl cudarc::driver::DevicePtrMut<f32> + ?Sized),
        scratch: &mut CudaSlice<f32>,
        tokens: usize,
        split: usize,
    ) -> Result<()> {
        let Matrix {
            packed,
            scales,
            biases,
            rows,
            cols,
        } = matrix;
        ensure!(matches!(split, 1 | 2 | 4), "packed split must be 1, 2 or 4");
        ensure!(
            rows > 0
                && rows.is_multiple_of(crate::wide::WIDE_RPB)
                && cols > 0
                && cols.is_multiple_of(64 * split)
                && cols / (32 * split) <= 256,
            "unsupported packed projection shape: rows {rows}, cols {cols}, split {split}, tokens {tokens}"
        );
        let elements = rows.checked_mul(cols).context("packed shape overflow")?;
        let inputs = tokens.checked_mul(cols).context("packed input overflow")?;
        let outputs = tokens.checked_mul(rows).context("packed output overflow")?;
        let partials = outputs
            .checked_mul(split)
            .context("packed scratch overflow")?;
        ensure!(
            tokens > 0
                && packed.len() >= elements / 8
                && scales.len() >= elements / 64
                && biases.len() >= elements / 64,
            "short packed projection"
        );
        ensure!(
            x.len() >= inputs && y.len() >= outputs && scratch.len() >= partials,
            "short packed workspace"
        );
        let blocks = u32::try_from(rows / crate::wide::WIDE_RPB * split)?;
        ensure!(blocks <= 65535, "packed grid exceeds CUDA y limit");
        let tokens = u32::try_from(tokens)?;
        let rows = i32::try_from(rows)?;
        let cols = i32::try_from(cols)?;
        let split = split as i32;
        unsafe {
            self.stream
                .launch_builder(&self.dot)
                .arg(packed)
                .arg(scales)
                .arg(biases)
                .arg(x)
                .arg(&*scratch)
                .arg(&rows)
                .arg(&cols)
                .arg(&split)
                .launch(LaunchConfig {
                    grid_dim: (tokens, blocks, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
            let (y, _y_guard) = y.device_ptr_mut(&self.stream);
            self.stream
                .launch_builder(&self.sum)
                .arg(&*scratch)
                .arg(&y)
                .arg(&rows)
                .arg(&split)
                .launch(LaunchConfig {
                    grid_dim: (tokens, (rows as u32).div_ceil(256), 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }
}

pub(crate) enum Projector {
    Gemv {
        input: CudaSlice<f32>,
    },
    Packed {
        engine: Packed,
        scratch: CudaSlice<f32>,
    },
    Mma {
        engine: Box<crate::mma::MmaKernels>,
        xb: CudaSlice<u8>,
        xsum: CudaSlice<f32>,
        stream: Arc<CudaStream>,
        format: ff_core::quant::QuantFormat,
    },
}

impl Projector {
    pub fn project(
        &mut self,
        matrix: Matrix<'_>,
        x: &CudaSlice<f32>,
        y: &mut (impl cudarc::driver::DevicePtrMut<f32> + ?Sized),
        tokens: usize,
        split: usize,
    ) -> Result<()> {
        match self {
            Self::Gemv { .. } => anyhow::bail!("GEMV projections use the scalar composition"),
            Self::Packed { engine, scratch } => {
                engine.project(matrix, x, y, scratch, tokens, split)
            }
            Self::Mma {
                engine,
                xb,
                xsum,
                stream,
                format,
            } => {
                ensure!(
                    *format == crate::weights::INT4,
                    "the packed Matrix form is int4-only"
                );
                // The MMA covers any in_dim without split-K; `split` is the
                // packed engine's knob and does not apply.
                let _ = split;
                engine.stage_x(stream, x, xb, *format)?;
                engine.group_sums(stream, x, xsum, matrix.cols, tokens)?;
                engine.project_int4(
                    stream,
                    &crate::mma::Int4Args {
                        packed: matrix.packed,
                        scales: matrix.scales,
                        biases: matrix.biases,
                        xs: xsum,
                        xb,
                        rows: matrix.rows,
                        in_dim: matrix.cols,
                        tokens,
                    },
                    y,
                )
            }
        }
    }

    /// Raw 16-bit projection (bf16/f16) through the MMA engine.
    pub fn project16(
        &mut self,
        w: &CudaSlice<u8>,
        x: &CudaSlice<f32>,
        y: &mut (impl cudarc::driver::DevicePtrMut<f32> + ?Sized),
        rows: usize,
        in_dim: usize,
        tokens: usize,
    ) -> Result<()> {
        match self {
            Self::Mma {
                engine,
                xb,
                stream,
                format,
                ..
            } => {
                engine.stage_x(stream, x, xb, *format)?;
                let x: &CudaSlice<u8> = xb;
                match *format {
                    ff_core::quant::QuantFormat::Bf16 => engine.project_bf16(
                        stream,
                        &crate::mma::RawArgs {
                            w,
                            x,
                            rows,
                            in_dim,
                            tokens,
                        },
                        y,
                    ),
                    ff_core::quant::QuantFormat::F16 => engine.project_f16(
                        stream,
                        &crate::mma::RawArgs {
                            w,
                            x,
                            rows,
                            in_dim,
                            tokens,
                        },
                        y,
                    ),
                    QuantFormat::GroupAffine { .. } => {
                        anyhow::bail!("project16 needs a 16-bit checkpoint")
                    }
                    QuantFormat::BlockFp8 { .. } => {
                        anyhow::bail!("project16 needs a 16-bit checkpoint")
                    }
                }
            }
            Self::Packed { .. } | Self::Gemv { .. } => {
                anyhow::bail!("16-bit batched projections need the Mma engine")
            }
        }
    }
}

pub fn workspace_bytes(
    mode: Mode,
    geom: &BatchGeom,
    format: ff_core::quant::QuantFormat,
) -> Result<usize> {
    let (hidden, inner, tokens) = (geom.hidden, geom.inner, geom.tokens);
    let (weights, reserve) = match mode {
        Mode::Gemv => (hidden.max(inner), 0),
        Mode::Packed | Mode::Mma => (
            hidden
                .checked_mul(4)
                .map(|n| n.max(inner))
                .and_then(|n| n.checked_mul(tokens))
                .context("packed scratch overflow")?,
            0,
        ),
    };
    let width = inner
        .checked_mul(3)
        .and_then(|m| hidden.checked_add(m))
        .context("prefill width overflow")?;
    let activations = weights
        .checked_add(
            tokens
                .checked_mul(width)
                .context("prefill activations overflow")?,
        )
        .and_then(|n| n.checked_mul(size_of::<f32>()))
        .and_then(|n| n.checked_add(reserve))
        .context("prefill workspace overflow")?;
    // hidden + x1 per token; the mixer slabs alias the MLP buffers.
    let mixer = geom
        .hidden
        .checked_mul(2)
        .and_then(|n| n.checked_mul(tokens))
        .and_then(|n| n.checked_mul(size_of::<f32>()))
        .context("prefill mixer slab overflow")?;
    let aux = if mode == Mode::Mma {
        // xb 16-bit staging [T, max(inner, hidden)] for every format;
        // xsum [G, T] for the int4 bias path.
        let xb = inner
            .max(hidden)
            .checked_mul(tokens)
            .and_then(|n| n.checked_mul(2))
            .context("prefill xb staging overflow")?;
        let int4 = if format == crate::weights::INT4 {
            inner
                .checked_div(64)
                .and_then(|n| n.checked_mul(tokens))
                .and_then(|n| n.checked_mul(size_of::<f32>()))
                .context("prefill xsum overflow")?
        } else {
            0
        };
        xb.checked_add(int4).context("prefill mma aux overflow")?
    } else {
        0
    };
    let attn_slabs = tokens
        .checked_mul(geom.q)
        .and_then(|n| n.checked_mul(size_of::<f32>()))
        .context("attn slab overflow")?;
    activations
        .checked_add(mixer)
        .and_then(|n| n.checked_add(aux))
        .and_then(|n| n.checked_add(attn_slabs))
        .context("prefill workspace overflow")
}

pub(crate) struct Batch {
    pub projector: Projector,
    /// Roped queries and gate logits, [T, heads, D].
    pub qslab: CudaSlice<f32>,
    pub gslab: CudaSlice<f32>,
    pub gate: CudaSlice<f32>,
    pub up: CudaSlice<f32>,
    pub inner: CudaSlice<f32>,
    pub out: CudaSlice<f32>,
    /// Per-token residual stream and normed input, [T, hidden].
    pub hidden: CudaSlice<f32>,
    pub x1: CudaSlice<f32>,
    pub rows: RowsKernels,
    pub gdn: Option<GdnPrefill>,
}

impl Batch {
    pub fn new(
        ctx: &GpuContext,
        device: usize,
        mode: Mode,
        geom: &BatchGeom,
        format: ff_core::quant::QuantFormat,
    ) -> Result<Self> {
        let bytes = workspace_bytes(mode, geom, format)?;
        let (hidden, inner, tokens) = (geom.hidden, geom.inner, geom.tokens);
        let build = || -> Result<Self> {
            Ok(Self {
                gdn: None,
                qslab: ctx
                    .stream
                    .alloc_zeros::<f32>(tokens * geom.q / 2)
                    .context("q slab")?,
                gslab: ctx
                    .stream
                    .alloc_zeros::<f32>(tokens * geom.q / 2)
                    .context("g slab")?,
                projector: match mode {
                    Mode::Gemv => Projector::Gemv {
                        input: ctx.stream.alloc_zeros::<f32>(hidden.max(inner))?,
                    },
                    Mode::Packed => Projector::Packed {
                        engine: Packed::new(ctx)?,
                        scratch: ctx
                            .stream
                            .alloc_zeros::<f32>(tokens * (4 * hidden).max(inner))?,
                    },
                    Mode::Mma => Projector::Mma {
                        engine: Box::new(crate::mma::MmaKernels::load(ctx)?),
                        xb: ctx
                            .stream
                            .alloc_zeros::<u8>(tokens * geom.inner.max(geom.hidden) * 2)?,
                        xsum: ctx
                            .stream
                            .alloc_zeros::<f32>(if format == crate::weights::INT4 {
                                tokens * (geom.inner / 64).max(1)
                            } else {
                                1
                            })?,
                        stream: ctx.stream.clone(),
                        format,
                    },
                },
                gate: ctx.stream.alloc_zeros::<f32>(tokens * inner)?,
                up: ctx.stream.alloc_zeros::<f32>(tokens * inner)?,
                inner: ctx.stream.alloc_zeros::<f32>(tokens * inner)?,
                out: ctx.stream.alloc_zeros::<f32>(tokens * hidden)?,
                hidden: ctx.stream.alloc_zeros::<f32>(tokens * hidden)?,
                x1: ctx.stream.alloc_zeros::<f32>(tokens * hidden)?,
                rows: RowsKernels::load(ctx)?,
            })
        };
        build().with_context(|| {
            format!("allocate prefill workspace of {bytes} bytes on device {device}")
        })
    }
}

#[derive(Clone)]
struct GdnChoice {
    chunk: usize,
    values: usize,
    prepare: CudaFunction,
    recurrence: CudaFunction,
    prepare_smem: u32,
    recurrence_smem: u32,
    bytes: usize,
}

pub(crate) struct GdnPrefill {
    key: (String, Vec<i32>, Vec<(usize, usize)>),
    conv: CudaFunction,
    norm: CudaFunction,
    choice: GdnChoice,
    qk: CudaSlice<u16>,
    v: CudaSlice<u16>,
    control: CudaSlice<f32>,
    history: CudaSlice<f32>,
    state: CudaSlice<f32>,
    channels: usize,
    kernel: usize,
    keys: usize,
    values: usize,
    tokens: usize,
}

impl GdnPrefill {
    fn sizes(geom: &BatchGeom, chunk: usize) -> Result<[usize; 5]> {
        if geom.key_heads == 0 {
            return Ok([0; 5]);
        }
        ensure!(
            geom.tokens > 0
                && matches!(chunk, 16 | 32 | 64)
                && geom.ba > 0
                && geom.ba.is_multiple_of(geom.key_heads)
                && geom.conv_kernel > 0
                && geom.z == geom.ba * 128
                && geom.conv == (2 * geom.key_heads + geom.ba) * 128,
            "chunked GDN requires 128-dimensional heads and an integral value/key head ratio"
        );
        let size = |dims: &[usize]| {
            dims.iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .context("GDN workspace overflow")
        };
        let chunks = geom.tokens.div_ceil(chunk);
        Ok([
            size(&[chunks, chunk, 2, geom.key_heads, 128])?,
            size(&[geom.tokens, geom.z])?,
            size(&[chunks, geom.ba, 2 * chunk * chunk + 2 * chunk])?,
            size(&[geom.conv, geom.conv_kernel - 1])?,
            size(&[geom.ba, 128, 128])?,
        ])
    }

    pub fn bytes(geom: &BatchGeom, chunk: usize) -> Result<usize> {
        let [qk, v, control, history, state] = Self::sizes(geom, chunk)?;
        qk.checked_add(v)
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| {
                control
                    .checked_add(history)?
                    .checked_add(state)?
                    .checked_mul(4)?
                    .checked_add(n)
            })
            .context("GDN workspace byte count overflow")
    }

    pub fn select(
        ctx: &GpuContext,
        device: usize,
        geom: &BatchGeom,
        g: &mut GpuGdn,
        batch: &Batch,
        eps: f32,
        previous: &[&Self],
    ) -> Result<Self> {
        let started = std::time::Instant::now();
        ensure!(geom.key_heads > 0, "no GDN heads to calibrate");
        let module =
            ff_edge0::kernel_assets::load_module(&ctx.context, &crate::kernel_assets::GDN_PREFILL)?;
        let conv = module.load_function("gdn_conv_rows")?;
        let norm = module.load_function("gdn_norm_rows")?;
        let shared: usize = ctx.context.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )?.try_into().context("invalid GDN shared-memory capacity")?;
        let available = ctx
            .context
            .mem_get_info()?
            .0
            .saturating_sub(ff_core::probe::device_admission_reserve_bytes() as usize);
        let mut choices = Vec::new();
        for chunk in [16, 32, 64] {
            let bytes = Self::bytes(geom, chunk)?;
            let prepare_smem = 2 * chunk * 128 * 2 + (3 * chunk * chunk + 2 * chunk) * 4;
            if bytes > available || prepare_smem > shared {
                continue;
            }
            let prepare = module.load_function(&format!("gdn_prepare_c{chunk}"))?;
            prepare.set_attribute(
                cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                prepare_smem as i32,
            )?;
            if prepare.occupancy_max_active_blocks_per_multiprocessor(256, prepare_smem, None)? == 0
            {
                continue;
            }
            for values in [128, 64] {
                let recurrence_smem =
                    2 * chunk * 128 * 2 + (chunk * chunk + 2 * chunk + chunk * values) * 4;
                if recurrence_smem > shared {
                    continue;
                }
                let recurrence =
                    module.load_function(&format!("gdn_recurrence_c{chunk}_dv{values}"))?;
                recurrence.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    recurrence_smem as i32,
                )?;
                if recurrence.occupancy_max_active_blocks_per_multiprocessor(
                    256,
                    recurrence_smem,
                    None,
                )? == 0
                {
                    continue;
                }
                choices.push(GdnChoice {
                    chunk,
                    values,
                    prepare: prepare.clone(),
                    recurrence,
                    prepare_smem: prepare_smem as u32,
                    recurrence_smem: recurrence_smem as u32,
                    bytes,
                });
            }
        }
        ensure!(
            !choices.is_empty(),
            "no chunked GDN variant fits: at least {} B workspace required, {available} B available after reserve; shared capacity {shared} B",
            Self::bytes(geom, 16)?
        );
        use cudarc::driver::sys::CUdevice_attribute as Attr;
        let mut limits = vec![shared as i32];
        for attribute in [
            Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            Attr::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            Attr::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
            Attr::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
            Attr::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
            Attr::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            Attr::CU_DEVICE_ATTRIBUTE_WARP_SIZE,
        ] {
            limits.push(ctx.context.attribute(attribute)?);
        }
        let key = (
            ctx.context.name()?,
            limits,
            choices.iter().map(|v| (v.chunk, v.values)).collect(),
        );
        let cached = previous.iter().find(|v| {
            v.key == key
                && v.channels == geom.conv
                && v.kernel == geom.conv_kernel
                && v.keys == geom.key_heads
                && v.values == geom.ba
                && v.tokens == geom.tokens
        });
        let largest = match cached {
            Some(v) => v.choice.chunk,
            None => choices
                .iter()
                .map(|v| v.chunk)
                .max()
                .context("no GDN chunk")?,
        };
        let [qk, v, control, history, state] = Self::sizes(geom, largest)?;
        let mut op = Self {
            key,
            conv,
            norm,
            choice: choices[0].clone(),
            qk: ctx.stream.alloc_zeros(qk)?,
            v: ctx.stream.alloc_zeros(v)?,
            control: ctx.stream.alloc_zeros(control)?,
            history: ctx.stream.alloc_zeros(history)?,
            state: ctx.stream.alloc_zeros(state)?,
            channels: geom.conv,
            kernel: geom.conv_kernel,
            keys: geom.key_heads,
            values: geom.ba,
            tokens: geom.tokens,
        };
        if let Some(cached) = cached {
            op.choice = choices
                .into_iter()
                .find(|v| v.chunk == cached.choice.chunk && v.values == cached.choice.values)
                .context("shared GDN choice is absent from its identical feasible set")?;
            ctx.stream.synchronize()?;
            eprintln!(
                "qwen35: device {device}: GDN C{} DV{} [shared calibration], {} B workspace | setup {:.3} ms",
                op.choice.chunk,
                op.choice.values,
                op.choice.bytes,
                started.elapsed().as_secs_f64() * 1000.0
            );
            return Ok(op);
        }
        let mut timings = Vec::with_capacity(choices.len());
        for choice in &choices {
            op.choice = choice.clone();
            let mut samples = Vec::with_capacity(12);
            for rep in 0..15 {
                let (history, state) = ctx.gdn_state_mut(g);
                ctx.stream.memset_zeros(history)?;
                ctx.stream.memset_zeros(state)?;
                ctx.stream.synchronize()?;
                let begin = std::time::Instant::now();
                op.run(ctx, g, batch, geom.tokens, eps)?;
                ctx.stream.synchronize()?;
                if rep >= 3 {
                    samples.push(begin.elapsed().as_secs_f64() * 1000.0);
                }
            }
            ensure!(
                samples.iter().all(|v| v.is_finite() && *v > 0.0),
                "invalid GDN probe sample"
            );
            samples.sort_by(f64::total_cmp);
            timings.push([samples[0], samples[6], samples[11]]);
        }
        let (history, state) = ctx.gdn_state_mut(g);
        ctx.stream.memset_zeros(history)?;
        ctx.stream.memset_zeros(state)?;
        ctx.stream.synchronize()?;
        let best = (0..choices.len())
            .min_by(|&a, &b| timings[a][1].total_cmp(&timings[b][1]))
            .context("no GDN samples")?;
        let selected = (0..choices.len())
            .filter(|&i| timings[i][0] <= timings[best][2] && timings[best][0] <= timings[i][2])
            .min_by_key(|&i| (choices[i].bytes, std::cmp::Reverse(choices[i].values)))
            .context("no GDN choice overlaps its best sample")?;
        let facts = choices
            .iter()
            .zip(&timings)
            .map(|(c, t)| {
                format!(
                    "C{} DV{} min {:.4} med {:.4} max {:.4} ms",
                    c.chunk, c.values, t[0], t[1], t[2]
                )
            })
            .collect::<Vec<_>>()
            .join(" | ");
        op.choice = choices.swap_remove(selected);
        let [qk, _, control, _, _] = Self::sizes(geom, op.choice.chunk)?;
        if op.qk.len() != qk || op.control.len() != control {
            drop(op.qk);
            drop(op.control);
            ctx.stream.synchronize()?;
            op.qk = ctx.stream.alloc_zeros(qk)?;
            op.control = ctx.stream.alloc_zeros(control)?;
        }
        eprintln!(
            "qwen35: device {device}: GDN probe C{} DV{} [derived], {} B workspace | {facts} | setup {:.3} ms",
            op.choice.chunk,
            op.choice.values,
            op.choice.bytes,
            started.elapsed().as_secs_f64() * 1000.0
        );
        Ok(op)
    }

    pub fn run(
        &self,
        ctx: &GpuContext,
        g: &GpuGdn,
        batch: &Batch,
        tokens: usize,
        eps: f32,
    ) -> Result<()> {
        ensure!(
            tokens > 0 && tokens <= self.tokens,
            "GDN block exceeds admitted capacity"
        );
        let c = &self.choice;
        let chunks = tokens.div_ceil(c.chunk);
        let width = self.values * 128;
        let (w, al, dt, norm, history, state) = ctx.gdn_parts(g);
        let z = batch.up.slice(..tokens * width);
        let b = batch
            .up
            .slice(tokens * width..tokens * (width + self.values));
        let a = batch
            .up
            .slice(tokens * (width + self.values)..tokens * (width + 2 * self.values));
        unsafe {
            ctx.stream
                .launch_builder(&self.conv)
                .arg(&batch.gate)
                .arg(w)
                .arg(history)
                .arg(&self.history)
                .arg(&batch.inner)
                .arg(&(tokens as i32))
                .arg(&(self.channels as i32))
                .arg(&(self.kernel as i32))
                .launch(LaunchConfig {
                    grid_dim: (self.channels.div_ceil(256) as u32, tokens as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
            ctx.stream
                .launch_builder(&c.prepare)
                .arg(&batch.inner)
                .arg(&a)
                .arg(&b)
                .arg(al)
                .arg(dt)
                .arg(&self.qk)
                .arg(&self.v)
                .arg(&self.control)
                .arg(&(tokens as i32))
                .arg(&(self.keys as i32))
                .arg(&(self.values as i32))
                .launch(LaunchConfig {
                    grid_dim: ((chunks * self.values) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: c.prepare_smem,
                })?;
            ctx.stream
                .launch_builder(&c.recurrence)
                .arg(&self.qk)
                .arg(&self.v)
                .arg(&self.control)
                .arg(state)
                .arg(&self.state)
                .arg(&batch.inner)
                .arg(&(tokens as i32))
                .arg(&(chunks as i32))
                .arg(&(self.keys as i32))
                .arg(&(self.values as i32))
                .arg(&(1.0f32 / 128.0f32.sqrt()))
                .launch(LaunchConfig {
                    grid_dim: ((self.values * (128 / c.values)) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: c.recurrence_smem,
                })?;
            ctx.stream
                .launch_builder(&self.norm)
                .arg(&batch.inner)
                .arg(&z)
                .arg(norm)
                .arg(&batch.inner)
                .arg(&128i32)
                .arg(&eps)
                .launch(LaunchConfig {
                    grid_dim: ((tokens * self.values) as u32, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        ctx.context.bind_to_thread()?;
        for (source, target) in [(&self.history, history), (&self.state, state)] {
            ensure!(source.len() == target.len(), "GDN state slot shape differs");
            let (source_ptr, _source_guard) = source.device_ptr(&ctx.stream);
            let (target_ptr, _target_guard) = target.device_ptr(&ctx.stream);
            // State accesses and commit copies use the owning stream.
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    target_ptr,
                    source_ptr,
                    source.num_bytes(),
                    ctx.stream.cu_stream(),
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wide::{DownBuffers, GroupPair, WideGeom, WideKernels};

    #[test]
    fn scratch_rejects_overflow() {
        let geom = BatchGeom {
            hidden: 128,
            inner: 256,
            tokens: usize::MAX,
            conv: 64,
            z: 64,
            ba: 1,
            q: 128,
            kv: 64,
            key_heads: 0,
            conv_kernel: 1,
        };
        assert!(workspace_bytes(Mode::Packed, &geom, crate::weights::INT4).is_err());
    }

    /// The wide-kernel route the decode path takes for this shape:
    /// down (v4d8) above 8192, group below.
    fn wide_reference(
        ctx: &GpuContext,
        wide: &WideKernels,
        q: &ff_edge0::gpu::GpuQuant,
        x: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        if cols > 8192 {
            let (w, s, b) = q.tensors();
            let y = q.y_ref();
            wide.down(
                ctx,
                &DownBuffers {
                    packed: w,
                    scales: s,
                    biases: b,
                    x,
                    xb: x,
                    y,
                    yb: y,
                },
                &WideGeom {
                    rows,
                    in_dim: cols,
                    cols: 1,
                },
            )
        } else {
            let (w, s, b) = q.tensors();
            let y = q.y_ref();
            let seg = |rows| ff_edge0::gpu::GroupSeg {
                packed: w,
                scales: s,
                biases: b,
                y,
                rows,
                lora: None,
            };
            wide.group(
                ctx,
                &[seg(rows), seg(0), seg(0), seg(0)],
                [y; 4],
                &GroupPair { x, xb: x },
                cols,
                1,
            )
        }
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn packed_matches_scalar_bits() -> Result<()> {
        use ff_edge0::int4::GroupQuant;
        let ctx = GpuContext::new(0)?;
        let packed = Packed::new(&ctx)?;
        let wide = WideKernels::load(&ctx)?;
        wide.capture_body(Some(0))?;
        let shapes: &[(usize, usize, &str)] = &[
            (5120, 1, "linear_attn.in_proj_qkv"),
            (5120, 1, "linear_attn.in_proj_z"),
            (5120, 1, "linear_attn.in_proj_ba"),
            (6144, 1, "linear_attn.out_proj"),
            (5120, 1, "self_attn.q_proj"),
            (5120, 1, "self_attn.kv_proj"),
            (5120, 1, "mlp.gate_up_proj"),
            (17408, 4, "mlp.down_proj"),
        ];
        let rows = 32;
        for (cols, split) in [(5120, 1), (6144, 1), (17408, 4)] {
            let words: Vec<u32> = (0..rows * cols / 8)
                .map(|i| (i as u32).wrapping_mul(2654435761))
                .collect();
            let scales: Vec<f32> = (0..rows * cols / 64)
                .map(|i| 0.01 + (i % 7) as f32 * 0.003)
                .collect();
            let biases = scales.iter().map(|s| -7.0 * s).collect();
            let quant = GroupQuant::new(words, scales, biases, rows, cols, 4)?;
            let q = ctx.upload(&quant, None)?;
            let (w, s, b) = q.tensors();
            let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.017).sin()).collect();
            let dx = ctx.upload_f32(&input)?;
            let matrix = || Matrix {
                packed: w,
                scales: s,
                biases: b,
                rows,
                cols,
            };
            let mut y = ctx.stream.alloc_zeros::<f32>(rows)?;
            let mut projector = Projector::Packed {
                engine: Packed::new(&ctx)?,
                scratch: ctx.stream.alloc_zeros::<f32>(rows * split)?,
            };
            for &(_, _, name) in shapes.iter().filter(|&&(c, s, _)| c == cols && s == split) {
                projector
                    .project(matrix(), &dx, &mut y, 1, split)
                    .with_context(|| format!("{name} rejected by the packed engine"))?;
                wide_reference(&ctx, &wide, &q, &dx, rows, cols)?;
                let expect = ctx.dtoh(q.y_ref())?;
                let got = ctx.dtoh(&y)?;
                ensure!(
                    got.iter()
                        .zip(&expect)
                        .all(|(a, b)| (a - b).abs() <= 1e-4 * a.abs().max(1.0)),
                    "{name}: packed projection differs from the group kernel beyond 1e-4 relative"
                );
                println!("packed production {name}: cols={cols}, split={split}, tokens=1");
            }
            if cols == 6144 {
                continue;
            }
            for tokens in [1, 3, 33] {
                let input: Vec<f32> = (0..tokens * cols)
                    .map(|i| {
                        if i < cols {
                            -0.0
                        } else {
                            (i as f32 * 0.017).sin()
                        }
                    })
                    .collect();
                let x = ctx.upload_f32(&input)?;
                let mut y = ctx.stream.alloc_zeros::<f32>(tokens * rows)?;
                let mut scratch = ctx.stream.alloc_zeros::<f32>(tokens * rows * split)?;
                let matrix = || Matrix {
                    packed: w,
                    scales: s,
                    biases: b,
                    rows,
                    cols,
                };
                assert!(
                    packed
                        .project(matrix(), &x, &mut y, &mut scratch, tokens + 1, split)
                        .is_err()
                );
                packed.project(matrix(), &x, &mut y, &mut scratch, tokens, split)?;
                let got = ctx.dtoh(&y)?;
                let mut expected = Vec::new();
                for token in input.chunks(cols) {
                    let dx = ctx.upload_f32(token)?;
                    wide_reference(&ctx, &wide, &q, &dx, rows, cols)?;
                    expected.extend(ctx.dtoh(q.y_ref())?);
                }
                ensure!(
                    got.iter()
                        .zip(&expected)
                        .all(|(a, b)| (a - b).abs() <= 1e-4 * a.abs().max(1.0)),
                    "packed exceeds 1e-4 relative vs the group kernel for {cols}/{tokens}"
                );
            }
        }
        let geom = BatchGeom {
            hidden: 5120,
            inner: 17408,
            tokens: 128,
            conv: 10240,
            z: 6144,
            ba: 48,
            q: 12288,
            kv: 1024,
            key_heads: 16,
            conv_kernel: 4,
        };
        let batch = Batch::new(&ctx, 0, Mode::Packed, &geom, crate::weights::INT4)?;
        let scratch = match &batch.projector {
            Projector::Packed { scratch, .. } => scratch,
            _ => unreachable!("packed corpus uses the packed projector"),
        };
        let actual = [
            scratch,
            &batch.gate,
            &batch.up,
            &batch.inner,
            &batch.out,
            &batch.hidden,
            &batch.x1,
            &batch.qslab,
            &batch.gslab,
        ]
        .iter()
        .map(|b| b.len() * 4)
        .sum::<usize>();
        assert_eq!(
            actual,
            workspace_bytes(Mode::Packed, &geom, crate::weights::INT4)?
        );
        let conv = vec![0.0; geom.conv * geom.conv_kernel];
        let gates = vec![0.0; geom.ba];
        let norm = vec![1.0; 128];
        let mut g = GpuGdn::upload(
            &ctx,
            ff_edge0::gpu::GdnUpload {
                conv1d: &conv,
                a_log: &gates,
                dt_bias: &gates,
                norm: &norm,
                conv_dim: geom.conv,
                kernel: geom.conv_kernel,
                num_v: geom.ba,
                num_k: geom.key_heads,
                dk: 128,
                dv: 128,
                eps: 1e-6,
            },
        )?;
        let op = GdnPrefill::select(&ctx, 0, &geom, &mut g, &batch, 1e-6, &[])?;
        let allocated = (op.qk.len() + op.v.len()) * 2
            + (op.control.len() + op.history.len() + op.state.len()) * 4;
        assert_eq!(allocated, GdnPrefill::bytes(&geom, op.choice.chunk)?);
        let reused = GdnPrefill::select(&ctx, 0, &geom, &mut g, &batch, 1e-6, &[&op])?;
        assert_eq!(reused.key, op.key);
        assert_eq!(reused.choice.chunk, op.choice.chunk);
        assert_eq!(reused.choice.values, op.choice.values);
        assert_eq!(reused.qk.len(), op.qk.len());
        assert_eq!(reused.control.len(), op.control.len());
        let (_, _, _, _, history, state) = ctx.gdn_parts(&g);
        for buffer in [history, state] {
            assert!(
                ctx.stream
                    .clone_dtoh(buffer)?
                    .iter()
                    .all(|v| v.to_bits() == 0)
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn workspace_failure_names_device() -> Result<()> {
        let ctx = GpuContext::new(0)?;
        let geom = BatchGeom {
            hidden: 1 << 26,
            inner: 1 << 20,
            tokens: 128,
            conv: 64,
            z: 64,
            ba: 1,
            q: 128,
            kv: 64,
            key_heads: 0,
            conv_kernel: 1,
        };
        let error = match Batch::new(&ctx, 3, Mode::Packed, &geom, crate::weights::INT4) {
            Ok(_) => anyhow::bail!("oversized workspace unexpectedly succeeded"),
            Err(error) => error,
        };
        let message = error.to_string();
        assert!(message.contains("allocate prefill workspace"), "{message}");
        assert!(message.contains("device 3"), "{message}");
        Ok(())
    }
}
