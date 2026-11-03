//! MiniMax-H3 FL2VA transformer execution.
use crate::backend::cuda::{
    Device,
    blas::Blas,
    contracts::{Tensor, TensorMut, view},
    cubin,
    flash::Flash,
    linear::LinearBiasMode,
    ops::{self, flat},
    verification::Shape,
};
use crate::backend::gpu::{
    CudaSlice, CudaStream, CudaViewMut, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
};
use crate::checkpoint::{MappedCheckpoint, TensorEntry, TensorManifest};
use crate::experts::disk::{AlignedBytes, SpanReader};
use crate::minimax_h3::{AdalnPlan, H3Spec, MODALITIES, adaln_input};
use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use safetensors::Dtype;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
};

/// Direct-IO read granularity.
const READ_CHUNK_BYTES: u64 = 32 << 20;

/// Published element types a tensor upload accepts.
trait Element: crate::backend::gpu::DeviceRepr {
    const DTYPE: Dtype;
    fn finite(bytes: &[u8]) -> bool;
}

impl Element for bf16 {
    const DTYPE: Dtype = Dtype::BF16;
    fn finite(bytes: &[u8]) -> bool {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .all(|v| u16::from_le_bytes(*v) & 0x7f80 != 0x7f80)
    }
}

impl Element for f32 {
    const DTYPE: Dtype = Dtype::F32;
    fn finite(bytes: &[u8]) -> bool {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .all(|v| f32::from_le_bytes(*v).is_finite())
    }
}

/// O_DIRECT reads of single manifest tensors through one aligned pageable staging buffer.
struct EntryReader {
    dir: PathBuf,
    readers: BTreeMap<String, SpanReader>,
    staging: Option<AlignedBytes>,
}

impl EntryReader {
    fn new(dir: &Path) -> Self {
        Self {
            dir: dir.to_owned(),
            readers: BTreeMap::new(),
            staging: None,
        }
    }

    /// Copies a finite BF16 or F32 tensor into a buffer of its element type; the pageable source is consumed when the
    /// copy call returns.
    fn upload<T: Element>(
        &mut self,
        stream: &Arc<CudaStream>,
        name: &str,
        entry: &TensorEntry,
        destination: &mut CudaViewMut<'_, T>,
    ) -> Result<()> {
        ensure!(
            entry.dtype == T::DTYPE && entry.bytes == std::mem::size_of::<T>() * destination.len(),
            "{name}: {:?} {} B does not fill a {}-element {:?} buffer",
            entry.dtype,
            entry.bytes,
            destination.len(),
            T::DTYPE
        );
        let bytes = self.read(name, entry)?;
        ensure!(T::finite(bytes), "{name}: nonfinite value");
        let (pointer, _record) = destination.device_ptr_mut(stream);
        unsafe {
            cudarc::driver::result::memcpy_htod_async(pointer, bytes, stream.cu_stream())?;
        }
        Ok(())
    }

    /// Copies a finite BF16 tensor into host memory of exactly its size.
    fn read_into(&mut self, name: &str, entry: &TensorEntry, destination: &mut [u8]) -> Result<()> {
        ensure!(
            entry.dtype == Dtype::BF16 && entry.bytes == destination.len(),
            "{name}: {:?} {} B does not fill {} B",
            entry.dtype,
            entry.bytes,
            destination.len()
        );
        let bytes = self.read(name, entry)?;
        ensure!(<bf16 as Element>::finite(bytes), "{name}: nonfinite value");
        destination.copy_from_slice(bytes);
        Ok(())
    }

    /// The tensor's bytes, read with O_DIRECT into the reusable aligned staging buffer.
    fn read(&mut self, name: &str, entry: &TensorEntry) -> Result<&[u8]> {
        if !self.readers.contains_key(&entry.shard) {
            let reader = SpanReader::open(&self.dir.join(&entry.shard), true)?;
            self.readers.insert(entry.shard.clone(), reader);
        }
        let reader = &self.readers[&entry.shard];
        let offset = entry
            .data_start
            .checked_add(entry.data_offsets[0])
            .context("tensor offset overflow")?;
        let span = reader
            .plan(&[(offset, entry.bytes as u64)], READ_CHUNK_BYTES)?
            .pop()
            .with_context(|| format!("{name}: no read span"))?;
        let alignment = reader.alignment as usize;
        let needed = usize::try_from(span.end - span.begin)?.next_multiple_of(alignment);
        if self.staging.as_ref().is_none_or(|s| s.len() < needed) {
            self.staging = Some(AlignedBytes::new(needed, alignment)?);
        }
        let staging = self.staging.as_mut().context("staging buffer")?;
        let begin = usize::try_from(offset - span.begin)?;
        Ok(&reader.read_span(&span, READ_CHUNK_BYTES, staging)?[begin..begin + entry.bytes])
    }
}

fn matrix(
    b: &CudaSlice<bf16>,
    s: &Arc<CudaStream>,
    rows: usize,
    width: usize,
) -> Result<Tensor<bf16, 2>> {
    view(
        b,
        s,
        0,
        [u32::try_from(rows)?, u32::try_from(width)?],
        [width as u64, 1],
    )
}

fn vector(b: &CudaSlice<bf16>, s: &Arc<CudaStream>) -> Result<Tensor<bf16, 1>> {
    view(b, s, 0, [u32::try_from(b.len())?], [1])
}

/// Setup-time modulation tables, BF16: block rows `[layers, pairs, 6 * hidden]` in `AdalnPlan::pairs` order
/// (shift, scale, gate for attention, then for the FFN) and output-norm rows `[timesteps, 2 * hidden]` (shift, scale).
pub(crate) struct AdalnTable {
    pub blocks: CudaSlice<bf16>,
    pub norm_out: CudaSlice<bf16>,
}

impl AdalnTable {
    /// One pass over the 50 `adaln_proj` weights: each block's `[18 * hidden, time_dim]` projection of the plan's
    /// timesteps (bias before output rounding), of which only the addressed `(timestep, tag)` slices are kept.
    pub(crate) fn build(
        d: &Device,
        blas: &mut Blas,
        transformer_dir: &Path,
        checkpoint: &MappedCheckpoint,
        manifest: &TensorManifest,
        spec: &H3Spec,
        plan: &AdalnPlan,
    ) -> Result<Self> {
        let s = &d.stream;
        let timesteps = plan.timesteps.len();
        let width = 6 * spec.hidden;
        let modulation = width * MODALITIES as usize;
        let entry = |name: &str| {
            manifest
                .entries
                .get(name)
                .with_context(|| format!("manifest lacks {name}"))
        };
        let x = s.clone_htod(&adaln_input(spec, checkpoint, &plan.timesteps)?)?;
        let mut reader = EntryReader::new(transformer_dir);
        let mut weight = s.alloc_zeros::<bf16>(modulation * spec.time_dim)?;
        let mut bias = s.alloc_zeros::<bf16>(modulation)?;
        let y = s.alloc_zeros::<bf16>(timesteps * modulation)?;
        let mut blocks = s.alloc_zeros::<bf16>(spec.layers * plan.pairs.len() * width)?;
        let mode = LinearBiasMode::BeforeOutputRound;
        blas.prepare_biased(
            u32::try_from(timesteps)?,
            u32::try_from(modulation)?,
            u32::try_from(spec.time_dim)?,
            vector(&bias, s)?,
            mode,
        )?;
        for layer in 0..spec.layers {
            let base = format!("transformer_blocks.{layer}.adaln_proj.linear");
            for (suffix, buffer) in [("weight", &mut weight), ("bias", &mut bias)] {
                let name = format!("{base}.{suffix}");
                reader.upload(s, &name, entry(&name)?, &mut buffer.slice_mut(..))?;
            }
            blas.linear_biased(
                matrix(&weight, s, modulation, spec.time_dim)?,
                vector(&bias, s)?,
                matrix(&x, s, timesteps, spec.time_dim)?,
                &mut TensorMut(matrix(&y, s, timesteps, modulation)?),
                mode,
            )?;
            for (pair, &(t, tag)) in plan.pairs.iter().enumerate() {
                let source = (t as usize * MODALITIES as usize + usize::from(tag)) * width;
                let target = (layer * plan.pairs.len() + pair) * width;
                s.memcpy_dtod(
                    &y.slice(source..source + width),
                    &mut blocks.slice_mut(target..target + width),
                )?;
            }
        }
        let norm = 2 * spec.hidden;
        let mut norm_weight = s.alloc_zeros::<bf16>(norm * spec.time_dim)?;
        let mut norm_bias = s.alloc_zeros::<bf16>(norm)?;
        for (suffix, buffer) in [("weight", &mut norm_weight), ("bias", &mut norm_bias)] {
            let name = format!("norm_out.linear.{suffix}");
            reader.upload(s, &name, entry(&name)?, &mut buffer.slice_mut(..))?;
        }
        let norm_out = s.alloc_zeros::<bf16>(timesteps * norm)?;
        blas.prepare_biased(
            u32::try_from(timesteps)?,
            u32::try_from(norm)?,
            u32::try_from(spec.time_dim)?,
            vector(&norm_bias, s)?,
            mode,
        )?;
        blas.linear_biased(
            matrix(&norm_weight, s, norm, spec.time_dim)?,
            vector(&norm_bias, s)?,
            matrix(&x, s, timesteps, spec.time_dim)?,
            &mut TensorMut(matrix(&norm_out, s, timesteps, norm)?),
            mode,
        )?;
        s.synchronize()?;
        Ok(Self { blocks, norm_out })
    }
}

/// Upper bound on rows per FFN chunk.
const FFN_CHUNK_ROWS: usize = 1024;

/// Rows per FFN chunk: `rows` split evenly over `ceil(rows / FFN_CHUNK_ROWS)` chunks, so the tail chunk holds at
/// least `chunk - (chunks - 1)` rows.
fn ffn_chunk(rows: usize) -> usize {
    rows.div_ceil(rows.div_ceil(FFN_CHUNK_ROWS))
}
/// Threads per hidden-width RMSNorm row: 672 eight-element vectors of a 5376 row fit one register pass.
const HIDDEN_NORM_THREADS: u32 = 768;
/// Threads per head-width RMSNorm row.
const HEAD_NORM_THREADS: u32 = 32;

/// One block's BF16 weights: the base linears as row-major views (`LINEAR_ROLES` order) into `storage` or a dense
/// streaming slot, norms and LoRA factors owned; the q, k, v LoRA A factors are stacked `[3 * rank, hidden]`.
pub(crate) struct BlockWeights {
    norm1: CudaSlice<bf16>,
    norm2: CudaSlice<bf16>,
    norm_q: CudaSlice<bf16>,
    norm_k: CudaSlice<bf16>,
    linears: [Tensor<bf16, 2>; 6],
    storage: Option<CudaSlice<bf16>>,
    qkv_a: CudaSlice<bf16>,
    q_b: CudaSlice<bf16>,
    k_b: CudaSlice<bf16>,
    v_b: CudaSlice<bf16>,
    out_a: CudaSlice<bf16>,
    out_b: CudaSlice<bf16>,
    in_a: CudaSlice<bf16>,
    in_b: CudaSlice<bf16>,
    ffo_a: CudaSlice<bf16>,
    ffo_b: CudaSlice<bf16>,
    rank: usize,
}

/// A checkpoint directory read through its manifest.
pub(crate) struct Source<'a> {
    reader: EntryReader,
    manifest: &'a TensorManifest,
}

impl<'a> Source<'a> {
    pub(crate) fn new(dir: &Path, manifest: &'a TensorManifest) -> Self {
        Self {
            reader: EntryReader::new(dir),
            manifest,
        }
    }

    fn into<T: Element>(
        &mut self,
        s: &Arc<CudaStream>,
        name: &str,
        destination: &mut CudaViewMut<'_, T>,
    ) -> Result<()> {
        let entry = self
            .manifest
            .entries
            .get(name)
            .with_context(|| format!("manifest lacks {name}"))?;
        self.reader.upload(s, name, entry, destination)
    }

    fn tensor<T: Element + crate::backend::gpu::ValidAsZeroBits>(
        &mut self,
        s: &Arc<CudaStream>,
        name: &str,
        elements: usize,
    ) -> Result<CudaSlice<T>> {
        let mut buffer = s.alloc_zeros::<T>(elements)?;
        self.into(s, name, &mut buffer.slice_mut(..))?;
        Ok(buffer)
    }
}

/// Roles of `BlockWeights::linears`, in order.
const LINEAR_ROLES: [&str; 6] = [
    "attn.to_q.weight",
    "attn.to_k.weight",
    "attn.to_v.weight",
    "attn.to_out.0.weight",
    "ff.net.0.proj.weight",
    "ff.net.2.weight",
];
const Q: usize = 0;
const K: usize = 1;
const V: usize = 2;
const OUT: usize = 3;
const FF_IN: usize = 4;
const FF_OUT: usize = 5;

/// Element offset and shape of each loaded matrix of `block` in manifest name order, the order `DenseStreaming`
/// fills a layer slot in, and the block's total matrix elements.
type MatrixLayout = BTreeMap<String, (usize, [u32; 2])>;

fn matrix_layout(manifest: &TensorManifest, block: &str) -> Result<(MatrixLayout, usize)> {
    let prefix = format!("{block}.");
    let mut layout = BTreeMap::new();
    let mut elements = 0usize;
    for (name, entry) in &manifest.entries {
        let Some(role) = name.strip_prefix(&prefix) else {
            continue;
        };
        if !entry.load || entry.shape.len() != 2 {
            continue;
        }
        ensure!(entry.dtype == Dtype::BF16, "{name} is not BF16");
        layout.insert(
            role.to_owned(),
            (
                elements,
                [
                    u32::try_from(entry.shape[0])?,
                    u32::try_from(entry.shape[1])?,
                ],
            ),
        );
        elements += entry.bytes / 2;
    }
    ensure!(
        layout.len() == LINEAR_ROLES.len() && LINEAR_ROLES.iter().all(|r| layout.contains_key(*r)),
        "{block}: matrix set {:?}",
        layout.keys().collect::<Vec<_>>()
    );
    Ok((layout, elements))
}

impl BlockWeights {
    /// Base linears read into one owned buffer in `matrix_layout` order.
    pub(crate) fn resident(
        s: &Arc<CudaStream>,
        spec: &H3Spec,
        rank: usize,
        block: &str,
        base: &mut Source<'_>,
        lora: &mut Source<'_>,
    ) -> Result<Self> {
        let (layout, elements) = matrix_layout(base.manifest, block)?;
        let mut storage = s.alloc_zeros::<bf16>(elements)?;
        for (role, (offset, shape)) in &layout {
            let count = shape[0] as usize * shape[1] as usize;
            base.into(
                s,
                &format!("{block}.{role}"),
                &mut storage.slice_mut(*offset..offset + count),
            )?;
        }
        let at = pointer(&storage, s);
        let mut weights = Self::with_linears(s, spec, rank, block, (at, &layout), base, lora)?;
        weights.storage = Some(storage);
        Ok(weights)
    }

    /// Base linears read from the `DenseStreaming` slot at `slot`, valid between the layer's `begin` and `end`.
    pub(crate) fn streamed(
        s: &Arc<CudaStream>,
        spec: &H3Spec,
        rank: usize,
        block: &str,
        slot: u64,
        base: &mut Source<'_>,
        lora: &mut Source<'_>,
    ) -> Result<Self> {
        let (layout, _) = matrix_layout(base.manifest, block)?;
        Self::with_linears(s, spec, rank, block, (slot, &layout), base, lora)
    }

    fn with_linears(
        s: &Arc<CudaStream>,
        spec: &H3Spec,
        rank: usize,
        block: &str,
        (at, layout): (u64, &MatrixLayout),
        base: &mut Source<'_>,
        lora: &mut Source<'_>,
    ) -> Result<Self> {
        let mut linears = Vec::with_capacity(LINEAR_ROLES.len());
        for role in LINEAR_ROLES {
            let (offset, shape) = layout[role];
            let bytes = 2 * u64::from(shape[0]) * u64::from(shape[1]);
            linears.push(unsafe {
                Tensor::new(
                    at + 2 * offset as u64,
                    shape,
                    [u64::from(shape[1]), 1],
                    bytes,
                )?
            });
        }
        let linears: [Tensor<bf16, 2>; 6] = linears
            .try_into()
            .map_err(|_| anyhow::anyhow!("block linear count"))?;
        let (h, w, f, r) = (spec.hidden, spec.heads * spec.head_dim, spec.ffn, rank);
        let mut qkv_a = s.alloc_zeros::<bf16>(3 * r * h)?;
        for (i, name) in ["to_q", "to_k", "to_v"].iter().enumerate() {
            lora.into(
                s,
                &format!("{block}.attn.{name}.lora_A.default.weight"),
                &mut qkv_a.slice_mut(i * r * h..(i + 1) * r * h),
            )?;
        }
        let mut n =
            |name: &str, count: usize| base.tensor(s, &format!("{block}.{name}.weight"), count);
        let (norm1, norm2) = (n("norm1", h)?, n("norm2", h)?);
        let (norm_q, norm_k) = (
            n("attn.norm_q", spec.head_dim)?,
            n("attn.norm_k", spec.head_dim)?,
        );
        let mut l = |name: &str, factor: &str, count: usize| {
            lora.tensor(
                s,
                &format!("{block}.{name}.lora_{factor}.default.weight"),
                count,
            )
        };
        Ok(Self {
            q_b: l("attn.to_q", "B", w * r)?,
            k_b: l("attn.to_k", "B", w * r)?,
            v_b: l("attn.to_v", "B", w * r)?,
            out_a: l("attn.to_out.0", "A", r * w)?,
            out_b: l("attn.to_out.0", "B", h * r)?,
            in_a: l("ff.net.0.proj", "A", r * h)?,
            in_b: l("ff.net.0.proj", "B", 2 * f * r)?,
            ffo_a: l("ff.net.2", "A", r * f)?,
            ffo_b: l("ff.net.2", "B", h * r)?,
            norm1,
            norm2,
            norm_q,
            norm_k,
            linears,
            storage: None,
            qkv_a,
            rank,
        })
    }
}

pub(crate) struct BlockKernels {
    norm: crate::backend::gpu::CudaFunction,
    rotary: crate::backend::gpu::CudaFunction,
    swiglu: crate::backend::gpu::CudaFunction,
    modulate: crate::backend::gpu::CudaFunction,
    gated_add: crate::backend::gpu::CudaFunction,
    residual: crate::backend::gpu::CudaFunction,
}

impl BlockKernels {
    pub(crate) fn new(ctx: &Arc<crate::backend::gpu::CudaContext>) -> Result<Self> {
        let modulate = cubin::module(ctx, "modulate")?;
        let norm = cubin::module(ctx, "norm")?;
        Ok(Self {
            norm: norm.load_function("rms_norm_bf16")?,
            residual: norm.load_function("residual")?,
            rotary: cubin::module(ctx, "rope")?.load_function("rotary_prefix_kv")?,
            swiglu: cubin::module(ctx, "act")?.load_function("silu_mul")?,
            modulate: modulate.load_function("row_modulate")?,
            gated_add: modulate.load_function("row_gated_add")?,
        })
    }
}

/// Activations of one block forward over `rows` sequence rows.
pub(crate) struct BlockWork {
    rows: usize,
    rank: usize,
    normed: CudaSlice<bf16>,
    q_raw: CudaSlice<bf16>,
    k_raw: CudaSlice<bf16>,
    v_raw: CudaSlice<bf16>,
    q: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    lora: CudaSlice<bf16>,
    hidden: CudaSlice<bf16>,
    gated: CudaSlice<bf16>,
}

impl BlockWork {
    /// Binds the fixed GEMM plans of every base and LoRA-A linear shape these rows run: q/k/v, out, the stacked
    /// q/k/v LoRA A and the out LoRA A over all rows; the two FFN projections and their LoRA A over full and tail
    /// chunks.
    pub(crate) fn prepare(&self, blas: &Blas, spec: &H3Spec) -> Result<()> {
        let (h, width, f, r) = (spec.hidden, spec.heads * spec.head_dim, spec.ffn, self.rank);
        let mut shapes = vec![
            (self.rows, width, h),
            (self.rows, h, width),
            (self.rows, 3 * r, h),
            (self.rows, r, width),
        ];
        let chunk = ffn_chunk(self.rows);
        let tail = self.rows % chunk;
        for rows in [chunk, tail].into_iter().filter(|&n| n > 0) {
            shapes.push((rows, 2 * f, h));
            shapes.push((rows, h, f));
            shapes.push((rows, r, h));
            shapes.push((rows, r, f));
        }
        for (rows, output, input) in shapes {
            blas.prepare_fixed(
                u32::try_from(rows)?,
                u32::try_from(output)?,
                u32::try_from(input)?,
            )?;
        }
        Ok(())
    }

    pub(crate) fn new(
        s: &Arc<CudaStream>,
        spec: &H3Spec,
        rank: usize,
        rows: usize,
    ) -> Result<Self> {
        launch_extents(spec, rows)?;
        let width = spec.heads * spec.head_dim;
        let chunk = ffn_chunk(rows);
        let attention = || s.alloc_zeros::<bf16>(rows * width);
        Ok(Self {
            rows,
            rank,
            normed: s.alloc_zeros::<bf16>(rows * spec.hidden)?,
            q_raw: attention()?,
            k_raw: attention()?,
            v_raw: attention()?,
            q: attention()?,
            k: attention()?,
            lora: s.alloc_zeros::<bf16>(rows * 3 * rank)?,
            hidden: s.alloc_zeros::<bf16>(chunk * 2 * spec.ffn)?,
            gated: s.alloc_zeros::<bf16>(chunk * spec.ffn)?,
        })
    }
}

/// Transformer blocks read AdaLN rows and rotate the first `6 * rope_freq_dim` channels of q and k; token refiner
/// blocks do neither and add both residuals unmodulated.
pub(crate) enum Conditioning<'a> {
    Modulated {
        rotary: &'a CudaSlice<bf16>,
        modulation: &'a Modulation<'a>,
    },
    Plain,
}

/// Block modulation: `entries` rows of `6 * hidden` (shift, scale, gate of attention, then of the FFN) starting
/// `offset` elements into `table`, and the entry of every sequence row.
pub(crate) struct Modulation<'a> {
    pub table: &'a CudaSlice<bf16>,
    pub offset: usize,
    pub entries: u32,
    pub index: &'a CudaSlice<u32>,
}

fn pointer<T>(b: &CudaSlice<T>, s: &Arc<CudaStream>) -> u64 {
    b.device_ptr(s).0
}

fn rows_view(
    b: &CudaSlice<bf16>,
    s: &Arc<CudaStream>,
    offset: usize,
    rows: usize,
    width: usize,
    pitch: usize,
) -> Result<Tensor<bf16, 2>> {
    view(
        b,
        s,
        offset,
        [u32::try_from(rows)?, u32::try_from(width)?],
        [pitch as u64, 1],
    )
}

/// Stream, BLAS, kernels, geometry and LoRA scale shared by every block launch.
pub(crate) struct BlockContext<'a> {
    pub s: &'a Arc<CudaStream>,
    pub blas: &'a Blas,
    pub kernels: &'a BlockKernels,
    pub spec: &'a H3Spec,
    pub lora_scale: f32,
}

/// Dense rows of `width` elements starting `offset` elements into `buffer`.
#[derive(Clone, Copy)]
struct Operand<'a> {
    buffer: &'a CudaSlice<bf16>,
    offset: usize,
    width: usize,
}

/// `y += lora_scale * (x * a^T) * b^T` for one LoRA pair; `x * a^T` is staged in `work` with row pitch `rank`.
fn lora_pair(
    c: &BlockContext<'_>,
    rank: usize,
    x: Operand<'_>,
    (a, b): (&CudaSlice<bf16>, &CudaSlice<bf16>),
    work: &mut CudaSlice<bf16>,
    y: Operand<'_>,
    rows: usize,
) -> Result<()> {
    c.blas.linear_typed(
        crate::op_contract::ExecutionPhase::Prefill,
        crate::backend::cuda::linear::LinearWeightRef::Bf16(matrix(a, c.s, rank, x.width)?),
        rows_view(x.buffer, c.s, x.offset, rows, x.width, x.width)?,
        &mut TensorMut(rows_view(work, c.s, 0, rows, rank, rank)?),
    )?;
    c.blas.linear_accumulate(
        matrix(b, c.s, y.width, rank)?,
        rows_view(work, c.s, 0, rows, rank, rank)?,
        &mut TensorMut(rows_view(y.buffer, c.s, y.offset, rows, y.width, y.width)?),
        c.lora_scale,
    )
}

/// One MiniMax-H3 transformer or token refiner block over `x [rows, hidden]` in place, BF16, LoRA applied online
/// at `c.lora_scale`. A modulated block's `rotary` is `[rows, 6 * rope_freq_dim]` from `rotary_table`.
pub(crate) fn block_forward(
    c: &BlockContext<'_>,
    flash: &mut Flash,
    w: &BlockWeights,
    conditioning: &Conditioning<'_>,
    x: &mut CudaSlice<bf16>,
    work: &mut BlockWork,
) -> Result<()> {
    let (s, blas, kernels, spec, lora_scale) = (c.s, c.blas, c.kernels, c.spec, c.lora_scale);
    let rows = work.rows;
    let (h, width, f, r) = (spec.hidden, spec.heads * spec.head_dim, spec.ffn, w.rank);
    let (rotary_width, rotary_table, modulation) = match conditioning {
        Conditioning::Modulated { rotary, modulation } => {
            ensure!(
                rotary.len() == rows * 6 * spec.rope_freq_dim && modulation.index.len() == rows,
                "conditioning extents differ from the work rows"
            );
            ensure!(
                modulation.offset + modulation.entries as usize * 6 * h <= modulation.table.len(),
                "modulation table extent"
            );
            (6 * spec.rope_freq_dim, pointer(rotary, s), Some(modulation))
        }
        Conditioning::Plain => (0, pointer(&work.q, s), None),
    };
    ensure!(x.len() == rows * h, "block rows differ from the work rows");
    let stride = 6 * h;
    let (table, index, entries) = match modulation {
        Some(m) => (
            pointer(m.table, s) + 2 * m.offset as u64,
            pointer(m.index, s),
            m.entries,
        ),
        None => (0, 0, 0),
    };
    let chunk = |c: usize| table + (2 * c * h) as u64;
    let rows_i = i32::try_from(rows)?;
    let row_grid = LaunchConfig {
        grid_dim: (u32::try_from(rows)?, u32::try_from(h.div_ceil(256))?, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let norm = |input: u64,
                weight: u64,
                output: u64,
                rows: usize,
                dim: usize,
                threads: u32,
                eps: f32|
     -> Result<()> {
        kernels.rms_norm(
            s,
            NormRows {
                input,
                weight,
                output,
                rows,
                dim,
                threads,
                eps,
            },
        )
    };
    let modulate = |data: u64, shift: usize, scale: usize| -> Result<()> {
        unsafe {
            s.launch_builder(&kernels.modulate)
                .arg(&data)
                .arg(&chunk(shift))
                .arg(&chunk(scale))
                .arg(&index)
                .arg(&data)
                .arg(&rows_i)
                .arg(&i32::try_from(h)?)
                .arg(&(stride as u64))
                .arg(&entries)
                .launch(row_grid)?;
        }
        Ok(())
    };
    let gated_add = |residual: u64, update: u64, gate: usize| -> Result<()> {
        unsafe {
            s.launch_builder(&kernels.gated_add)
                .arg(&residual)
                .arg(&update)
                .arg(&chunk(gate))
                .arg(&index)
                .arg(&rows_i)
                .arg(&i32::try_from(h)?)
                .arg(&(stride as u64))
                .arg(&entries)
                .launch(row_grid)?;
        }
        Ok(())
    };
    let residual = |residual: u64, update: u64| -> Result<()> {
        unsafe {
            s.launch_builder(&kernels.residual)
                .arg(&residual)
                .arg(&update)
                .arg(&i32::try_from(rows * h)?)
                .launch(flat(rows * h / 8))?;
        }
        Ok(())
    };
    let base = |index: usize, x: Operand<'_>, y: Operand<'_>, count: usize| -> Result<()> {
        blas.linear_typed(
            crate::op_contract::ExecutionPhase::Prefill,
            crate::backend::cuda::linear::LinearWeightRef::Bf16(w.linears[index]),
            rows_view(x.buffer, s, x.offset, count, x.width, x.width)?,
            &mut TensorMut(rows_view(y.buffer, s, y.offset, count, y.width, y.width)?),
        )
    };
    let xp = pointer(x, s);
    let normed = pointer(&work.normed, s);

    norm(
        xp,
        pointer(&w.norm1, s),
        normed,
        rows,
        h,
        HIDDEN_NORM_THREADS,
        spec.norm_eps,
    )?;
    if modulation.is_some() {
        modulate(normed, 0, 1)?;
    }
    for (index, out) in [(Q, &work.q_raw), (K, &work.k_raw), (V, &work.v_raw)] {
        base(
            index,
            Operand {
                buffer: &work.normed,
                offset: 0,
                width: h,
            },
            Operand {
                buffer: out,
                offset: 0,
                width,
            },
            rows,
        )?;
    }
    blas.linear_typed(
        crate::op_contract::ExecutionPhase::Prefill,
        crate::backend::cuda::linear::LinearWeightRef::Bf16(matrix(&w.qkv_a, s, 3 * r, h)?),
        rows_view(&work.normed, s, 0, rows, h, h)?,
        &mut TensorMut(rows_view(&work.lora, s, 0, rows, 3 * r, 3 * r)?),
    )?;
    for (i, (b, out)) in [
        (&w.q_b, &work.q_raw),
        (&w.k_b, &work.k_raw),
        (&w.v_b, &work.v_raw),
    ]
    .into_iter()
    .enumerate()
    {
        blas.linear_accumulate(
            matrix(b, s, width, r)?,
            rows_view(&work.lora, s, i * r, rows, r, 3 * r)?,
            &mut TensorMut(matrix(out, s, rows, width)?),
            lora_scale,
        )?;
    }
    let head_rows = rows * spec.heads;
    norm(
        pointer(&work.q_raw, s),
        pointer(&w.norm_q, s),
        pointer(&work.q, s),
        head_rows,
        spec.head_dim,
        HEAD_NORM_THREADS,
        spec.qk_norm_eps,
    )?;
    norm(
        pointer(&work.k_raw, s),
        pointer(&w.norm_k, s),
        pointer(&work.k, s),
        head_rows,
        spec.head_dim,
        HEAD_NORM_THREADS,
        spec.qk_norm_eps,
    )?;
    // q rotates in place; k and v land head-major [heads, rows, head_dim] in k_raw and q_raw.
    let elements = rows * 2 * spec.heads * spec.head_dim;
    unsafe {
        s.launch_builder(&kernels.rotary)
            .arg(&pointer(&work.q, s))
            .arg(&pointer(&work.k, s))
            .arg(&pointer(&work.v_raw, s))
            .arg(&pointer(&work.q_raw, s))
            .arg(&pointer(&work.k_raw, s))
            .arg(&rotary_table)
            .arg(&0i32)
            .arg(&rows_i)
            .arg(&i32::try_from(spec.heads)?)
            .arg(&i32::try_from(spec.heads)?)
            .arg(&i32::try_from(spec.head_dim)?)
            .arg(&i32::try_from(rotary_width)?)
            .arg(&rows_i)
            .launch(flat(elements))?;
    }
    flash.run(
        &work.q,
        width,
        &work.q_raw,
        &work.k_raw,
        &mut work.v_raw,
        Shape {
            rows,
            start: 0,
            q_heads: spec.heads,
            kv_heads: spec.heads,
            capacity: rows,
            head_dim: spec.head_dim,
            causal: false,
            window_left: None,
        },
    )?;
    base(
        OUT,
        Operand {
            buffer: &work.v_raw,
            offset: 0,
            width,
        },
        Operand {
            buffer: &work.normed,
            offset: 0,
            width: h,
        },
        rows,
    )?;
    lora_pair(
        c,
        r,
        Operand {
            buffer: &work.v_raw,
            offset: 0,
            width,
        },
        (&w.out_a, &w.out_b),
        &mut work.lora,
        Operand {
            buffer: &work.normed,
            offset: 0,
            width: h,
        },
        rows,
    )?;
    if modulation.is_some() {
        gated_add(xp, normed, 2)?;
    } else {
        residual(xp, normed)?;
    }

    norm(
        xp,
        pointer(&w.norm2, s),
        normed,
        rows,
        h,
        HIDDEN_NORM_THREADS,
        spec.norm_eps,
    )?;
    if modulation.is_some() {
        modulate(normed, 3, 4)?;
    }
    let chunk = ffn_chunk(rows);
    for start in (0..rows).step_by(chunk) {
        let n = chunk.min(rows - start);
        base(
            FF_IN,
            Operand {
                buffer: &work.normed,
                offset: start * h,
                width: h,
            },
            Operand {
                buffer: &work.hidden,
                offset: 0,
                width: 2 * f,
            },
            n,
        )?;
        lora_pair(
            c,
            r,
            Operand {
                buffer: &work.normed,
                offset: start * h,
                width: h,
            },
            (&w.in_a, &w.in_b),
            &mut work.lora,
            Operand {
                buffer: &work.hidden,
                offset: 0,
                width: 2 * f,
            },
            n,
        )?;
        let hidden = pointer(&work.hidden, s);
        let gated = pointer(&work.gated, s);
        let cfg = ops::silu_mul_grid(n, f, hidden, gated)?;
        unsafe {
            s.launch_builder(&kernels.swiglu)
                .arg(&hidden)
                .arg(&gated)
                .arg(&i32::try_from(n)?)
                .arg(&i32::try_from(f)?)
                .arg(&3i32)
                .launch(cfg)?;
        }
        base(
            FF_OUT,
            Operand {
                buffer: &work.gated,
                offset: 0,
                width: f,
            },
            Operand {
                buffer: &work.normed,
                offset: start * h,
                width: h,
            },
            n,
        )?;
        lora_pair(
            c,
            r,
            Operand {
                buffer: &work.gated,
                offset: 0,
                width: f,
            },
            (&w.ffo_a, &w.ffo_b),
            &mut work.lora,
            Operand {
                buffer: &work.normed,
                offset: start * h,
                width: h,
            },
            n,
        )?;
    }
    if modulation.is_some() {
        gated_add(xp, normed, 5)?;
    } else {
        residual(xp, normed)?;
    }
    Ok(())
}

/// One RMSNorm launch: `rows` rows of `dim` elements, one block of `threads` per row.
struct NormRows {
    input: u64,
    weight: u64,
    output: u64,
    rows: usize,
    dim: usize,
    threads: u32,
    eps: f32,
}

/// One `row_modulate` launch over dense `[rows, dim]` data with table rows of `stride` elements.
struct ModulateRows {
    data: u64,
    shift: u64,
    scale: u64,
    index: u64,
    rows: usize,
    dim: usize,
    stride: usize,
    entries: u32,
}

impl BlockKernels {
    fn rms_norm(&self, s: &Arc<CudaStream>, n: NormRows) -> Result<()> {
        unsafe {
            s.launch_builder(&self.norm)
                .arg(&n.input)
                .arg(&n.weight)
                .arg(&n.output)
                .arg(&i32::try_from(n.rows)?)
                .arg(&i32::try_from(n.dim)?)
                .arg(&n.eps)
                .launch(LaunchConfig {
                    grid_dim: (u32::try_from(n.rows)?, 1, 1),
                    block_dim: (n.threads, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    fn row_modulate(&self, s: &Arc<CudaStream>, m: ModulateRows) -> Result<()> {
        unsafe {
            s.launch_builder(&self.modulate)
                .arg(&m.data)
                .arg(&m.shift)
                .arg(&m.scale)
                .arg(&m.index)
                .arg(&m.data)
                .arg(&i32::try_from(m.rows)?)
                .arg(&i32::try_from(m.dim)?)
                .arg(&(m.stride as u64))
                .arg(&m.entries)
                .launch(LaunchConfig {
                    grid_dim: (
                        u32::try_from(m.rows)?,
                        u32::try_from(m.dim.div_ceil(256))?,
                        1,
                    ),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }
}

/// `[text | conditioning video | audio | generated video]` row ranges of a packed layout.
#[derive(Clone, Copy)]
struct Ranges {
    text: usize,
    condition: usize,
    audio_start: usize,
    audio: usize,
    video_start: usize,
    rows: usize,
}

impl Ranges {
    fn of(layout: &crate::minimax_h3::PackedLayout) -> Result<Self> {
        let text = layout.text_indices.len();
        let condition = layout.condition_video_rows;
        let audio = layout.audio_indices.len();
        let audio_start = text + condition;
        let video_start = audio_start + audio;
        let rows = layout.rows();
        ensure!(
            layout
                .text_indices
                .iter()
                .enumerate()
                .all(|(i, &r)| r as usize == i)
                && layout
                    .audio_indices
                    .iter()
                    .enumerate()
                    .all(|(i, &r)| r as usize == audio_start + i)
                && layout.video_indices.iter().enumerate().all(|(i, &r)| {
                    r as usize
                        == if i < condition {
                            text + i
                        } else {
                            video_start + i - condition
                        }
                }),
            "packed layout is not [text | conditions | audio | video]"
        );
        Ok(Self {
            text,
            condition,
            audio_start,
            audio,
            video_start,
            rows,
        })
    }
}

/// Teacher-forced inputs of one evaluation: patchified video rows `[video rows, 96]` (conditioning first) and audio
/// rows `[audio rows, 32]` in F32, text states `[text rows, text_dim]` in BF16, and each sequence row's AdaLN pair and
/// timestep from `AdalnPlan`.
pub(crate) struct StepInputs<'a> {
    pub video: &'a CudaSlice<f32>,
    pub audio: &'a CudaSlice<f32>,
    pub text: &'a CudaSlice<bf16>,
    pub row_pairs: &'a CudaSlice<u32>,
    pub row_timesteps: &'a CudaSlice<u32>,
}

/// Transformer-block placement: `Derived` from the sensed VRAM and host budgets; `Streamed` keeps no block resident
/// (two device slots, host tier from the sensed host budget, rest disk); `StreamedDisk` streams every block from disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Placement {
    Derived,
    Streamed,
    StreamedDisk,
}

impl Placement {
    /// `FF_H3_PLACEMENT` = `derived`, `streamed` or `streamed-disk`.
    pub(crate) fn from_env() -> Result<Self> {
        match std::env::var("FF_H3_PLACEMENT").as_deref() {
            Ok("derived") => Ok(Self::Derived),
            Ok("streamed") => Ok(Self::Streamed),
            Ok("streamed-disk") => Ok(Self::StreamedDisk),
            other => bail!("FF_H3_PLACEMENT must be derived, streamed or streamed-disk: {other:?}"),
        }
    }
}

/// `rows` sequence rows keep every launch of the block path inside its index type: `rotary_prefix_kv` indexes
/// `rows * 2 * heads * head_dim` elements and `residual` `rows * hidden` elements in i32, the q/k `rms_norm_bf16`
/// runs `rows * heads` rows in i32, and `vision_upcast_f32` / `vision_bias_f32` take u32 counts of `rows * hidden`.
fn launch_extents(spec: &H3Spec, rows: usize) -> Result<()> {
    let limit = i32::MAX as usize;
    for (kernel, per_row) in [
        ("rotary_prefix_kv", 2 * spec.heads * spec.head_dim),
        ("residual", spec.hidden),
        ("rms_norm_bf16", spec.heads),
    ] {
        let extent = rows
            .checked_mul(per_row)
            .with_context(|| format!("{kernel} extent of {rows} rows overflows"))?;
        ensure!(
            extent <= limit,
            "{rows} sequence rows give {kernel} {extent} indices, above its i32 limit {limit} (at most {} rows)",
            limit / per_row
        );
    }
    ensure!(
        rows * spec.hidden <= u32::MAX as usize,
        "{rows} sequence rows exceed the u32 counts of vision_upcast_f32 / vision_bias_f32"
    );
    Ok(())
}

/// `vision_affine_cast_bf16` of `count` F32 values at `source` to BF16 at `target`; `count == 0` (the conditioning
/// rows of a layout with no keyframe anchors, the `t2va` workflow) launches nothing.
fn cast_rows(
    s: &Arc<CudaStream>,
    kernel: &crate::backend::gpu::CudaFunction,
    source: u64,
    target: u64,
    count: usize,
) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    unsafe {
        s.launch_builder(kernel)
            .arg(&source)
            .arg(&target)
            .arg(&(count as u64))
            .arg(&0f32)
            .arg(&1f32)
            .launch(flat(count))?;
    }
    Ok(())
}

/// The transformer over one request layout: F32 patch and output projections, BF16 text embedder, norms, refiner
/// blocks and LoRA factors resident; the 50 transformer blocks' matrices placed by `DensePlacement` from the sensed
/// VRAM and host budgets (resident, or two device slots fed from pinned host memory or the disk tier).
pub(crate) struct Dit {
    ranges: Ranges,
    proj_in: [CudaSlice<f32>; 2],
    audio_proj_in: [CudaSlice<f32>; 2],
    proj_out: [CudaSlice<f32>; 2],
    audio_proj_out: [CudaSlice<f32>; 2],
    context: [CudaSlice<bf16>; 2],
    final_norm: CudaSlice<bf16>,
    norm_out: CudaSlice<bf16>,
    refiners: Vec<BlockWeights>,
    blocks: Vec<BlockWeights>,
    dense: Option<crate::backend::cuda::weights::DenseStreaming>,
    rotary: CudaSlice<bf16>,
    x: CudaSlice<bf16>,
    work: BlockWork,
    flash: Flash,
    text: CudaSlice<bf16>,
    text_work: BlockWork,
    text_flash: Flash,
    scratch: CudaSlice<f32>,
    upcast: crate::backend::gpu::CudaFunction,
    bias: crate::backend::gpu::CudaFunction,
    round: crate::backend::gpu::CudaFunction,
    euler: crate::backend::gpu::CudaFunction,
    /// Velocity of the generated video rows, `[generated rows, 96]`.
    pub video: CudaSlice<f32>,
    /// Velocity of the audio rows, `[audio rows, 32]`.
    pub audio: CudaSlice<f32>,
}

impl Dit {
    pub(crate) fn new(
        d: &Device,
        blas: &mut Blas,
        spec: &H3Spec,
        (rank, layout): (usize, &crate::minimax_h3::PackedLayout),
        mut base: Source<'_>,
        mut lora: Source<'_>,
        placement: Placement,
    ) -> Result<Self> {
        use crate::prefill::DensePlacement;
        let s = &d.stream;
        let ranges = Ranges::of(layout)?;
        let (h, r) = (spec.hidden, rank);
        let patch = spec.video_row_width();
        let mut pair = |name: &str, n: usize, k: usize| -> Result<[CudaSlice<f32>; 2]> {
            Ok([
                base.tensor(s, &format!("{name}.weight"), n * k)?,
                base.tensor(s, &format!("{name}.bias"), n)?,
            ])
        };
        let proj_in = pair("proj_in", h, patch)?;
        let audio_proj_in = pair("audio_proj_in", h, spec.audio_in_channels)?;
        let proj_out = pair("proj_out", patch, h)?;
        let audio_proj_out = pair("audio_proj_out", spec.audio_in_channels, h)?;
        let context = [
            base.tensor(s, "context_embedder.weight", h * spec.text_dim)?,
            base.tensor(s, "context_embedder.bias", h)?,
        ];
        let final_norm = base.tensor(s, "token_refiner.final_norm.weight", h)?;
        let norm_out = base.tensor(s, "norm_out.norm.weight", h)?;
        let refiners = (0..spec.refiner_layers)
            .map(|i| {
                BlockWeights::resident(
                    s,
                    spec,
                    r,
                    &format!("token_refiner.refiner_blocks.{i}"),
                    &mut base,
                    &mut lora,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        blas.prepare_biased(
            u32::try_from(ranges.text)?,
            u32::try_from(h)?,
            u32::try_from(spec.text_dim)?,
            vector(&context[1], s)?,
            LinearBiasMode::BeforeOutputRound,
        )?;
        let work = BlockWork::new(s, spec, r, ranges.rows)?;
        work.prepare(blas, spec)?;
        let text_work = BlockWork::new(s, spec, r, ranges.text)?;
        text_work.prepare(blas, spec)?;
        let flash = Flash::new(
            s.clone(),
            ranges.rows,
            spec.heads,
            spec.heads,
            spec.head_dim,
        )?;
        let text_flash = Flash::new(
            s.clone(),
            ranges.text,
            spec.heads,
            spec.heads,
            spec.head_dim,
        )?;
        let generated = ranges.rows - ranges.video_start;
        let scratch_rows = (ranges.condition + generated).max(ranges.audio);
        let x = s.alloc_zeros::<bf16>(ranges.rows * h)?;
        let text = s.alloc_zeros::<bf16>(ranges.text * h)?;
        let scratch = s.alloc_zeros::<f32>(scratch_rows * h)?;
        let video = s.alloc_zeros::<f32>(generated * patch)?;
        let audio = s.alloc_zeros::<f32>(ranges.audio * spec.audio_in_channels)?;
        let rotary = s.clone_htod(&crate::minimax_h3::rotary_table(spec, layout))?;

        let mut layers = vec![0usize; spec.layers];
        let mut largest = 0usize;
        for entry in base.manifest.entries.values() {
            if let Some(layer) = entry.layer
                && entry.load
                && entry.shape.len() == 2
            {
                layers[layer as usize] += entry.bytes;
                largest = largest.max(entry.bytes);
            }
        }
        let owned = lora
            .manifest
            .entries
            .iter()
            .filter(|(name, _)| name.starts_with("transformer_blocks."))
            .map(|(_, entry)| entry.bytes)
            .sum::<usize>()
            + spec.layers * 2 * (2 * h + 2 * spec.head_dim);
        let (free, total) = d.ctx.mem_get_info()?;
        let mut budget = DensePlacement::device_budget(free, total, owned, 0, 0)?;
        if placement != Placement::Derived {
            let slots = 2 * layers
                .iter()
                .copied()
                .max()
                .context("no transformer blocks")?;
            ensure!(slots <= budget, "two block slots exceed the device budget");
            budget = slots;
        }
        let mut plan = DensePlacement::new(&layers, budget, 0)?;
        let mut host = None;
        if plan.slot_bytes > 0 {
            let available = usize::try_from(crate::backend::cuda::calibrate::host_memory(
                Path::new("/"),
            )?)?;
            let reserved = 2 * plan.slot_bytes + largest + (4 << 30);
            ensure!(
                available >= reserved,
                "dense pinned staging and host reserve exceed available RAM: need={reserved} available={available}"
            );
            let host_bytes = match placement {
                Placement::StreamedDisk => 0,
                Placement::Derived | Placement::Streamed => available - reserved,
            };
            plan = DensePlacement::new(&layers, budget, host_bytes)?;
            host = Some(available);
        }
        ensure!(
            placement == Placement::Derived || plan.resident_bytes == 0,
            "{placement:?} placement kept a resident block"
        );
        eprintln!(
            "{}",
            serde_json::json!({"h3_dense_placement": &plan, "mode": format!("{placement:?}"), "free_bytes": free, "total_bytes": total, "owned_block_bytes": owned, "host_available_bytes": host})
        );
        let dense = if plan.slot_bytes > 0 {
            let dir = base.reader.dir.clone();
            let host_tier = plan.host.clone();
            let mut dense =
                crate::backend::cuda::weights::DenseStreaming::new(&dir, base.manifest, plan, d)?;
            for layer in (0..spec.layers).filter(|&layer| host_tier[layer]) {
                let name = format!("transformer_blocks.{layer}");
                let (layout, elements) = matrix_layout(base.manifest, &name)?;
                let host = dense
                    .host_layer(layer)?
                    .with_context(|| format!("{name}: host tier without a pinned copy"))?;
                ensure!(host.len() == 2 * elements, "{name}: pinned copy size");
                for (role, (offset, _)) in &layout {
                    let tensor = format!("{name}.{role}");
                    let entry = &base.manifest.entries[&tensor];
                    base.reader.read_into(
                        &tensor,
                        entry,
                        &mut host[2 * offset..2 * offset + entry.bytes],
                    )?;
                }
            }
            Some(dense)
        } else {
            None
        };
        let mut blocks = Vec::with_capacity(spec.layers);
        for (layer, &bytes) in layers.iter().enumerate() {
            let name = format!("transformer_blocks.{layer}");
            let slot = match &dense {
                Some(dense) if dense.plan.slots[layer].is_some() => {
                    Some(dense.pointer(layer, 0, bytes)?)
                }
                _ => None,
            };
            blocks.push(match slot {
                Some(slot) => {
                    BlockWeights::streamed(s, spec, r, &name, slot, &mut base, &mut lora)?
                }
                None => BlockWeights::resident(s, spec, r, &name, &mut base, &mut lora)?,
            });
        }
        let vision = cubin::module(s.context(), "vision_ops")?;
        Ok(Self {
            ranges,
            proj_in,
            audio_proj_in,
            proj_out,
            audio_proj_out,
            context,
            final_norm,
            norm_out,
            refiners,
            blocks,
            dense,
            rotary,
            x,
            work,
            flash,
            text,
            text_work,
            text_flash,
            scratch,
            upcast: vision.load_function("vision_upcast_f32")?,
            bias: vision.load_function("vision_bias_f32")?,
            round: vision.load_function("vision_affine_cast_bf16")?,
            euler: cubin::module(s.context(), "flow")?.load_function("euler_blend_f32")?,
            video,
            audio,
        })
    }

    /// `y [rows, n] = x [rows, k] * w^T + b` in F32 on rows starting at element `x_offset` of `x`.
    fn projection(
        &self,
        c: &BlockContext<'_>,
        [w, b]: &[CudaSlice<f32>; 2],
        (x, x_offset): (&CudaSlice<f32>, usize),
        y: &CudaSlice<f32>,
        rows: usize,
    ) -> Result<()> {
        let s = c.s;
        let (n, k) = (b.len(), w.len() / b.len());
        c.blas.linear_f32(
            view(
                w,
                s,
                0,
                [u32::try_from(n)?, u32::try_from(k)?],
                [k as u64, 1],
            )?,
            view(
                x,
                s,
                x_offset,
                [u32::try_from(rows)?, u32::try_from(k)?],
                [k as u64, 1],
            )?,
            &mut TensorMut(view(
                y,
                s,
                0,
                [u32::try_from(rows)?, u32::try_from(n)?],
                [n as u64, 1],
            )?),
        )?;
        let out = pointer(y, s);
        unsafe {
            s.launch_builder(&self.bias)
                .arg(&out)
                .arg(&pointer(b, s))
                .arg(&out)
                .arg(&u32::try_from(rows)?)
                .arg(&u32::try_from(n)?)
                .launch(flat(rows * n))?;
        }
        Ok(())
    }

    /// Rounds `count` F32 values to BF16 sequence rows starting at element `target` of the hidden state.
    fn round_into(
        &self,
        s: &Arc<CudaStream>,
        source: u64,
        target: usize,
        count: usize,
    ) -> Result<()> {
        cast_rows(
            s,
            &self.round,
            source,
            pointer(&self.x, s) + 2 * target as u64,
            count,
        )
    }

    /// Widens `count` BF16 values starting at element `source` of `normed` into the F32 scratch.
    fn upcast_from(&self, s: &Arc<CudaStream>, source: usize, count: usize) -> Result<()> {
        unsafe {
            s.launch_builder(&self.upcast)
                .arg(&(pointer(&self.work.normed, s) + 2 * source as u64))
                .arg(&pointer(&self.scratch, s))
                .arg(&u32::try_from(count)?)
                .launch(flat(count))?;
        }
        Ok(())
    }

    /// `x = r * x + (1 - r) * (x + (1 - t) * v)` over `count` F32 latents at element `offset` of `x`, from grid
    /// points `k` and `k + 1`.
    fn euler(
        &self,
        s: &Arc<CudaStream>,
        (x, offset, count): (&CudaSlice<f32>, usize, usize),
        v: &CudaSlice<f32>,
        grid: &crate::minimax_h3::Schedule,
        k: usize,
    ) -> Result<()> {
        ensure!(
            v.len() == count && offset + count <= x.len(),
            "Euler extent"
        );
        let step = 1.0f32 - grid.timesteps[k];
        let ratio = grid.sigmas[k + 1] / grid.sigmas[k];
        unsafe {
            s.launch_builder(&self.euler)
                .arg(&(pointer(x, s) + 4 * offset as u64))
                .arg(&pointer(v, s))
                .arg(&(count as u64))
                .arg(&step)
                .arg(&ratio)
                .arg(&(1.0f32 - ratio))
                .launch(flat(count))?;
        }
        Ok(())
    }

    /// The sampling chain in place on `[video rows, 96]` and `[audio rows, 32]` F32 latents: per evaluation of
    /// `plan`, the forward, then the Euler step of the generated video rows on the video grid and of every audio row
    /// on the audio grid; conditioning video rows stay fixed.
    pub(crate) fn sample(
        &mut self,
        c: &BlockContext<'_>,
        adaln: (&AdalnTable, &AdalnPlan),
        (video_grid, audio_grid): (&crate::minimax_h3::Schedule, &crate::minimax_h3::Schedule),
        (video, audio): (&CudaSlice<f32>, &CudaSlice<f32>),
        text: &CudaSlice<bf16>,
    ) -> Result<()> {
        let (table, plan) = adaln;
        let s = c.s;
        let patch = c.spec.video_row_width();
        let counts = (
            u32::try_from(plan.pairs.len())?,
            u32::try_from(plan.timesteps.len())?,
        );
        ensure!(
            plan.row_pairs.len() == video_grid.timesteps.len()
                && plan.row_pairs.len() == audio_grid.timesteps.len(),
            "plan and grids differ in evaluations"
        );
        let condition = self.ranges.condition;
        let generated = self.ranges.rows - self.ranges.video_start;
        for k in 0..plan.row_pairs.len() {
            let row_pairs = s.clone_htod(&plan.row_pairs[k])?;
            let row_timesteps = s.clone_htod(&plan.row_timesteps[k])?;
            self.step(
                c,
                table,
                counts,
                &StepInputs {
                    video,
                    audio,
                    text,
                    row_pairs: &row_pairs,
                    row_timesteps: &row_timesteps,
                },
            )?;
            self.euler(
                s,
                (video, condition * patch, generated * patch),
                &self.video,
                video_grid,
                k,
            )?;
            self.euler(s, (audio, 0, audio.len()), &self.audio, audio_grid, k)?;
        }
        s.synchronize()?;
        Ok(())
    }
    /// One teacher-forced evaluation; the velocities land in `self.video` and `self.audio`.
    pub(crate) fn step(
        &mut self,
        c: &BlockContext<'_>,
        adaln: &AdalnTable,
        (pairs, timesteps): (u32, u32),
        inputs: &StepInputs<'_>,
    ) -> Result<()> {
        let (s, spec, r) = (c.s, c.spec, self.ranges);
        let h = spec.hidden;
        let video_rows = r.condition + (r.rows - r.video_start);
        let patch = spec.video_row_width();
        ensure!(
            inputs.video.len() == video_rows * patch
                && inputs.audio.len() == r.audio * spec.audio_in_channels
                && inputs.text.len() == r.text * spec.text_dim
                && inputs.row_pairs.len() == r.rows
                && inputs.row_timesteps.len() == r.rows,
            "evaluation inputs differ from the layout"
        );
        c.blas.linear_biased(
            matrix(&self.context[0], s, h, spec.text_dim)?,
            vector(&self.context[1], s)?,
            matrix(inputs.text, s, r.text, spec.text_dim)?,
            &mut TensorMut(matrix(&self.text, s, r.text, h)?),
            LinearBiasMode::BeforeOutputRound,
        )?;
        for refiner in &self.refiners {
            block_forward(
                c,
                &mut self.text_flash,
                refiner,
                &Conditioning::Plain,
                &mut self.text,
                &mut self.text_work,
            )?;
        }
        c.kernels.rms_norm(
            s,
            NormRows {
                input: pointer(&self.text, s),
                weight: pointer(&self.final_norm, s),
                output: pointer(&self.x, s),
                rows: r.text,
                dim: h,
                threads: HIDDEN_NORM_THREADS,
                eps: spec.final_norm_eps,
            },
        )?;
        self.projection(
            c,
            &self.proj_in,
            (inputs.video, 0),
            &self.scratch,
            video_rows,
        )?;
        let scratch = pointer(&self.scratch, s);
        self.round_into(s, scratch, r.text * h, r.condition * h)?;
        self.round_into(
            s,
            scratch + 4 * (r.condition * h) as u64,
            r.video_start * h,
            (video_rows - r.condition) * h,
        )?;
        self.projection(
            c,
            &self.audio_proj_in,
            (inputs.audio, 0),
            &self.scratch,
            r.audio,
        )?;
        self.round_into(s, scratch, r.audio_start * h, r.audio * h)?;

        let block_rows = 6 * h * pairs as usize;
        for (layer, block) in self.blocks.iter().enumerate() {
            if let Some(dense) = self.dense.as_mut() {
                dense.begin(layer, s)?;
            }
            let modulation = Modulation {
                table: &adaln.blocks,
                offset: layer * block_rows,
                entries: pairs,
                index: inputs.row_pairs,
            };
            block_forward(
                c,
                &mut self.flash,
                block,
                &Conditioning::Modulated {
                    rotary: &self.rotary,
                    modulation: &modulation,
                },
                &mut self.x,
                &mut self.work,
            )?;
            if let Some(dense) = self.dense.as_mut() {
                dense.end(layer, s)?;
            }
        }
        let normed = pointer(&self.work.normed, s);
        c.kernels.rms_norm(
            s,
            NormRows {
                input: pointer(&self.x, s),
                weight: pointer(&self.norm_out, s),
                output: normed,
                rows: r.rows,
                dim: h,
                threads: HIDDEN_NORM_THREADS,
                eps: spec.final_norm_eps,
            },
        )?;
        ensure!(
            adaln.norm_out.len() == timesteps as usize * 2 * h,
            "output-norm table extent"
        );
        let table = pointer(&adaln.norm_out, s);
        c.kernels.row_modulate(
            s,
            ModulateRows {
                data: normed,
                shift: table,
                scale: table + 2 * h as u64,
                index: pointer(inputs.row_timesteps, s),
                rows: r.rows,
                dim: h,
                stride: 2 * h,
                entries: timesteps,
            },
        )?;
        let generated = r.rows - r.video_start;
        self.upcast_from(s, r.video_start * h, generated * h)?;
        self.projection(
            c,
            &self.proj_out,
            (&self.scratch, 0),
            &self.video,
            generated,
        )?;
        self.upcast_from(s, r.audio_start * h, r.audio * h)?;
        self.projection(
            c,
            &self.audio_proj_out,
            (&self.scratch, 0),
            &self.audio,
            r.audio,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cuda::{closure::Closure, ops::Ops};
    use crate::minimax_h3::{
        Geometry, KeyframeAnchor, PackedLayout, RowTimesteps, Schedule, row_timestep_plan,
    };
    use safetensors::SafeTensors;
    use std::{fs, rc::Rc};

    /// The production (60,431) and small (40,045) cell rows and short multiples: every FFN chunk is within
    /// `FFN_CHUNK_ROWS` and no tail falls in the skinny-row class, which has no frozen LoRA GEMM choice.
    #[test]
    fn ffn_chunks_leave_no_skinny_tail() {
        for rows in [60_431, 40_045, 1_025, 2_049, 3_073, 1_024, 300] {
            let chunk = ffn_chunk(rows);
            let tail = rows % chunk;
            assert!(chunk <= FFN_CHUNK_ROWS, "rows={rows} chunk={chunk}");
            assert!(
                tail == 0 || tail > crate::linear_plan::MAX_SKINNY_ROWS,
                "rows={rows} chunk={chunk} tail={tail}"
            );
        }
    }

    /// The small (40,045) and production (60,431) cell rows fit every launch's index type; the 1024x1792 canvas at
    /// 345 frames does not, and the rotary limit is named.
    #[test]
    fn launch_extents_reject_rows_beyond_the_rotary_index() -> Result<()> {
        let spec = H3Spec::parse(&serde_json::from_str(include_str!(
            "../../../tests/data/minimax-h3-transformer-config.json"
        ))?)?;
        for rows in [40_045, 60_431, 149_796] {
            launch_extents(&spec, rows)?;
        }
        let tags = vec![crate::minimax_h3::TEXT_TAG; 1327];
        let geometry = Geometry::resolve(&spec, 1024, 1792, 345)?;
        let layout = PackedLayout::build(&spec, &geometry, &tags, &[KeyframeAnchor::First])?;
        let error = launch_extents(&spec, layout.rows())
            .err()
            .context("1024x1792 at 345 frames passed the launch extents")?;
        ensure!(
            layout.rows() > 149_796
                && error.to_string().contains("rotary_prefix_kv")
                && error.to_string().contains("at most 149796 rows"),
            "rows={} error={error}",
            layout.rows()
        );
        ensure!(
            launch_extents(&spec, 149_797).is_err(),
            "149,797 rows passed"
        );
        Ok(())
    }

    /// Token-refiner block 0 (published weights, Turbo LoRA at its scale) on 1, 12, 16 and 17 text rows: the LoRA A
    /// projections of the skinny class (rows <= 16, input a multiple of 256) run on the bound fixed plans; every
    /// output is finite.
    #[test]
    #[ignore = "GPU; FF_MODEL and FF_H3_LORA"]
    fn refiner_block_runs_on_skinny_text_rows_gpu() -> Result<()> {
        let path = |key: &str| {
            std::env::var_os(key)
                .map(PathBuf::from)
                .with_context(|| format!("{key} missing"))
        };
        let model = path("FF_MODEL")?;
        let lora_path = path("FF_H3_LORA")?;
        let spec = H3Spec::read(&model)?;
        let transformer = model.join("transformer");
        let manifest = MappedCheckpoint::open_component(&transformer, "diffusion_pytorch_model")?
            .minimax_h3_manifest(&spec)?;
        let lora_dir = lora_path.parent().context("LoRA directory")?.to_owned();
        let stem = lora_path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix(".safetensors"))
            .context("LoRA file stem")?;
        let (lora_manifest, adapter) =
            MappedCheckpoint::open_component(&lora_dir, stem)?.minimax_h3_lora_manifest(&spec)?;
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let s = &device.stream;
        let runtime = std::env::temp_dir().join(format!("ff-h3-refiner-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let blas = Blas::new(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(
                true,
                "MiniMax-H3 refiner skinny-row test".into(),
            )),
        )?;
        let kernels = BlockKernels::new(&device.ctx)?;
        let weights = BlockWeights::resident(
            s,
            &spec,
            adapter.rank,
            "token_refiner.refiner_blocks.0",
            &mut Source::new(&transformer, &manifest),
            &mut Source::new(&lora_dir, &lora_manifest),
        )?;
        for rows in [1usize, 12, 16, 17] {
            let mut work = BlockWork::new(s, &spec, adapter.rank, rows)?;
            work.prepare(&blas, &spec)?;
            let mut flash = Flash::new(s.clone(), rows, spec.heads, spec.heads, spec.head_dim)?;
            let input: Vec<bf16> = (0..rows * spec.hidden)
                .map(|i| bf16::from_f32(((i * 37 % 509) as f32 - 254.0) / 127.0))
                .collect();
            let mut x = s.clone_htod(&input)?;
            let context = BlockContext {
                s,
                blas: &blas,
                kernels: &kernels,
                spec: &spec,
                lora_scale: adapter.scale,
            };
            block_forward(
                &context,
                &mut flash,
                &weights,
                &Conditioning::Plain,
                &mut x,
                &mut work,
            )?;
            let output = s.clone_dtoh(&x)?;
            ensure!(
                output.iter().all(|v| v.to_f32().is_finite()),
                "rows={rows}: non-finite refiner output"
            );
            println!("H3_REFINER_SKINNY rows={rows} finite=true");
        }
        Ok(())
    }

    /// `cast_rows` with `count == 0` launches nothing (the kernel traps on a zero count); a non-zero count rounds.
    #[test]
    #[ignore = "GPU test"]
    fn cast_rows_skips_an_empty_conditioning_range_gpu() -> Result<()> {
        let device = Device::new(0)?;
        let s = &device.stream;
        let kernel =
            cubin::module(s.context(), "vision_ops")?.load_function("vision_affine_cast_bf16")?;
        let values: Vec<f32> = (0..16).map(|i| 1.0 + i as f32 / 3.0).collect();
        let source = s.clone_htod(&values)?;
        let target = s.alloc_zeros::<bf16>(16)?;
        cast_rows(s, &kernel, pointer(&source, s), pointer(&target, s), 0)?;
        s.synchronize()?;
        ensure!(
            s.clone_dtoh(&target)?.iter().all(|v| v.to_bits() == 0),
            "empty range wrote"
        );
        cast_rows(s, &kernel, pointer(&source, s), pointer(&target, s), 16)?;
        ensure!(
            s.clone_dtoh(&target)?
                .iter()
                .zip(&values)
                .all(|(got, &v)| got.to_bits() == bf16::from_f32(v).to_bits()),
            "cast differs from BF16 rounding"
        );
        Ok(())
    }

    fn decoded(file: &SafeTensors<'_>, name: &str) -> Result<(Vec<f64>, Vec<usize>)> {
        let view = file.tensor(name).with_context(|| format!("lacks {name}"))?;
        let values = match view.dtype() {
            Dtype::BF16 => view
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f64::from(bf16::from_le_bytes(*b).to_f32()))
                .collect(),
            Dtype::F32 => view
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f64::from(f32::from_le_bytes(*b)))
                .collect(),
            Dtype::I64 => view
                .data()
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_le_bytes(*b) as f64)
                .collect(),
            other => anyhow::bail!("{name}: unsupported dtype {other:?}"),
        };
        Ok((values, view.shape().to_vec()))
    }

    /// `[max_abs, rel_rms, cosine]` of `candidate` against `reference`.
    fn distance(candidate: &[f64], reference: &[f64]) -> [f64; 3] {
        let (mut max_abs, mut diff, mut cc, mut rr, mut cr) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (&c, &r) in candidate.iter().zip(reference) {
            max_abs = max_abs.max((c - r).abs());
            diff += (c - r) * (c - r);
            cc += c * c;
            rr += r * r;
            cr += c * r;
        }
        [max_abs, (diff / rr).sqrt(), cr / (cc.sqrt() * rr.sqrt())]
    }

    fn within(d: [f64; 3], threshold: &serde_json::Value) -> Result<bool> {
        let get = |key: &str| {
            threshold[key]
                .as_f64()
                .with_context(|| format!("threshold lacks {key}"))
        };
        Ok(d[0] <= get("max_abs")? && d[1] <= get("rel_rms")? && d[2] >= get("cosine_min")?)
    }

    fn mapped(path: &Path) -> Result<memmap2::Mmap> {
        let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        Ok(unsafe { memmap2::Mmap::map(&file)? })
    }

    /// The engine's own layout for a Stage 0 run directory: geometry from `meta.json`, text tags from the capture.
    fn capture_layout(spec: &H3Spec, capture: &Path) -> Result<PackedLayout> {
        let meta: serde_json::Value =
            serde_json::from_slice(&fs::read(capture.join("meta.json"))?)?;
        let size = |v: &serde_json::Value| v.as_u64().map(|n| n as usize).context("meta size");
        let geometry = Geometry::resolve(
            spec,
            size(&meta["resolved"]["height"])?,
            size(&meta["resolved"]["width"])?,
            size(&meta["cell"]["frames"])?,
        )?;
        let layout_map = mapped(&capture.join("layout.safetensors"))?;
        let layout_file = SafeTensors::deserialize(&layout_map)?;
        let (tags, _) = decoded(&layout_file, "token_tags")?;
        let (text, _) = decoded(&layout_file, "text_indices")?;
        let text_tags: Vec<u8> = text.iter().map(|&row| tags[row as usize] as u8).collect();
        let anchors: &[KeyframeAnchor] = if meta["cell"]["last_image"].is_null() {
            &[KeyframeAnchor::First]
        } else {
            &[KeyframeAnchor::First, KeyframeAnchor::Last]
        };
        PackedLayout::build(spec, &geometry, &text_tags, anchors)
    }

    /// AdaLN projections read once from disk, eager. Candidate rows of every block and step against the FP32 replay
    /// on the addressed rows, within the frozen `A.adaln.*` thresholds; control: the same rows scored against the
    /// replay's video/audio-swapped rows must leave a threshold.
    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted, FF_MODEL, FF_H3_CAPTURE (BF16 run, --capture-adaln), FF_H3_REPLAY (its FP32 replay with the adaln stage), FF_H3_THRESHOLDS (frozen JSON)"]
    fn adaln_table_matches_the_fp32_replay() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let path = |key: &str| {
            std::env::var_os(key)
                .map(PathBuf::from)
                .with_context(|| format!("{key} missing"))
        };
        let model = path("FF_MODEL")?;
        let capture = path("FF_H3_CAPTURE")?;
        let thresholds: serde_json::Value =
            serde_json::from_slice(&fs::read(path("FF_H3_THRESHOLDS")?)?)?;
        let thresholds = &thresholds["thresholds"];
        let spec = H3Spec::read(&model)?;
        let transformer = model.join("transformer");
        let checkpoint = MappedCheckpoint::open_component(&transformer, "diffusion_pytorch_model")?;
        let manifest = checkpoint.minimax_h3_manifest(&spec)?;

        let layout = capture_layout(&spec, &capture)?;
        let plan: Vec<RowTimesteps> =
            row_timestep_plan(&layout, &Schedule::new(9, 6.0)?, &Schedule::new(9, 3.0)?)?;
        let adaln = AdalnPlan::build(&layout, &plan);

        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let runtime = std::env::temp_dir().join(format!("ff-h3-adaln-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let mut blas = Blas::new(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(true, "MiniMax-H3 AdaLN gate".into())),
        )?;
        let table = AdalnTable::build(
            &device,
            &mut blas,
            &transformer,
            &checkpoint,
            &manifest,
            &spec,
            &adaln,
        )?;
        let blocks: Vec<f64> = device
            .stream
            .clone_dtoh(&table.blocks)?
            .into_iter()
            .map(|v| f64::from(v.to_f32()))
            .collect();
        let norm_out: Vec<f64> = device
            .stream
            .clone_dtoh(&table.norm_out)?
            .into_iter()
            .map(|v| f64::from(v.to_f32()))
            .collect();
        fs::remove_dir(&runtime)?;

        let replay_map = mapped(&path("FF_H3_REPLAY")?.join("replay.safetensors"))?;
        let replay = SafeTensors::deserialize(&replay_map)?;
        let width = 6 * spec.hidden;
        let norm = 2 * spec.hidden;
        let global = |t: f32| adaln.timesteps.partition_point(|&v| v < t);
        let (mut checked, mut worst, mut control_failed) = (0usize, [0f64, 0f64, 1f64], false);
        for (k, step) in plan.iter().enumerate() {
            let mut rows: Vec<u32> = step
                .indices
                .iter()
                .zip(&layout.token_tags)
                .map(|(&t, &tag)| t * MODALITIES + u32::from(tag))
                .collect();
            rows.sort_unstable();
            rows.dedup();
            let pair = |row: u32| {
                let t = global(step.timesteps[(row / MODALITIES) as usize]) as u32;
                let key = (t, (row % MODALITIES) as u8);
                adaln.pairs.binary_search(&key).ok()
            };
            for layer in 0..spec.layers {
                let name = format!("adaln.block{layer}.step{k}");
                let (reference, shape) = decoded(&replay, &name)?;
                ensure!(
                    shape == [step.timesteps.len() * 3, width],
                    "{name} shape {shape:?}"
                );
                let (mut candidate, mut expected, mut swapped) = (vec![], vec![], vec![]);
                for &row in &rows {
                    let p = pair(row).with_context(|| format!("{name} row {row} has no pair"))?;
                    let c = &blocks[(layer * adaln.pairs.len() + p) * width..][..width];
                    candidate.extend_from_slice(c);
                    expected.extend_from_slice(&reference[row as usize * width..][..width]);
                    let tag = row % MODALITIES;
                    if tag != 1 {
                        let other = row - tag + (2 - tag);
                        swapped.push((
                            c.to_vec(),
                            reference[other as usize * width..][..width].to_vec(),
                        ));
                    }
                }
                let threshold = &thresholds[format!("A.{name}")];
                let d = distance(&candidate, &expected);
                ensure!(within(d, threshold)?, "{name} {d:?} outside {threshold}");
                let (c, r): (Vec<f64>, Vec<f64>) = swapped
                    .iter()
                    .flat_map(|(c, r)| c.iter().copied().zip(r.iter().copied()))
                    .unzip();
                control_failed |= !within(distance(&c, &r), threshold)?;
                worst = [worst[0].max(d[0]), worst[1].max(d[1]), worst[2].min(d[2])];
                checked += 1;
            }
            let name = format!("adaln.norm_out.step{k}");
            let (reference, _) = decoded(&replay, &name)?;
            let mut used: Vec<u32> = step.indices.clone();
            used.sort_unstable();
            used.dedup();
            let (mut candidate, mut expected) = (vec![], vec![]);
            for &t in &used {
                let g = global(step.timesteps[t as usize]);
                candidate.extend_from_slice(&norm_out[g * norm..][..norm]);
                expected.extend_from_slice(&reference[t as usize * norm..][..norm]);
            }
            let d = distance(&candidate, &expected);
            let threshold = &thresholds[format!("A.{name}")];
            ensure!(within(d, threshold)?, "{name} {d:?} outside {threshold}");
            checked += 1;
        }
        ensure!(
            control_failed,
            "video/audio tag-swap control stayed within thresholds"
        );
        println!(
            "H3_ADALN_GATE PASS tables={checked} worst_max_abs={:.6e} worst_rel_rms={:.6e} worst_cosine={:.9} control=FAIL_AS_REQUIRED",
            worst[0], worst[1], worst[2]
        );
        Ok(())
    }

    /// Resident block 0, eager. The native block on the captured block-0 input with the captured BF16 modulation
    /// rows, against the FP32 replay of block 0 within the frozen `A.block0.output` threshold; control: LoRA scale 0
    /// must leave it.
    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted, FF_MODEL, FF_H3_LORA, FF_H3_CAPTURE (BF16 run, --capture-block-io --capture-adaln), FF_H3_REPLAY (FP32 replay with the block0 stage), FF_H3_THRESHOLDS"]
    fn block0_matches_the_fp32_replay() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let path = |key: &str| {
            std::env::var_os(key)
                .map(PathBuf::from)
                .with_context(|| format!("{key} missing"))
        };
        let model = path("FF_MODEL")?;
        let capture = path("FF_H3_CAPTURE")?;
        let lora_path = path("FF_H3_LORA")?;
        let thresholds: serde_json::Value =
            serde_json::from_slice(&fs::read(path("FF_H3_THRESHOLDS")?)?)?;
        let threshold = &thresholds["thresholds"]["A.block0.output"];
        let spec = H3Spec::read(&model)?;
        let transformer = model.join("transformer");
        let manifest = MappedCheckpoint::open_component(&transformer, "diffusion_pytorch_model")?
            .minimax_h3_manifest(&spec)?;
        let lora_dir = lora_path.parent().context("LoRA directory")?.to_owned();
        let stem = lora_path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix(".safetensors"))
            .context("LoRA file stem")?;
        let (lora_manifest, adapter) =
            MappedCheckpoint::open_component(&lora_dir, stem)?.minimax_h3_lora_manifest(&spec)?;
        let layout = capture_layout(&spec, &capture)?;
        let rows = layout.rows();
        let rotary_host = crate::minimax_h3::rotary_table(&spec, &layout);

        let io_map = mapped(&capture.join("block_io.safetensors"))?;
        let io = SafeTensors::deserialize(&io_map)?;
        let hidden = io.tensor("block0.hidden_states")?;
        ensure!(
            hidden.dtype() == Dtype::BF16 && hidden.shape() == [1, rows, spec.hidden],
            "block0.hidden_states {:?} {:?}",
            hidden.dtype(),
            hidden.shape()
        );
        let input: Vec<bf16> = hidden
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect();
        let (indices, _) = decoded(&io, "block0.adaln_indices")?;
        let index: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        let (cos, _) = decoded(&io, "block0.rotary_cos")?;
        let (sin, _) = decoded(&io, "block0.rotary_sin")?;
        let half = 3 * spec.rope_freq_dim;
        let rotary_width = 2 * half;
        let mut rotary_differs = 0usize;
        for row in 0..rows {
            for j in 0..half {
                let table = &rotary_host[row * rotary_width..][..rotary_width];
                let reference_cos = bf16::from_f64(cos[row * rotary_width + j]);
                let reference_sin = bf16::from_f64(sin[row * rotary_width + j]);
                rotary_differs += usize::from(table[j] != reference_cos)
                    + usize::from(table[half + j] != reference_sin);
            }
        }
        let adaln_map = mapped(&capture.join("adaln.safetensors"))?;
        let adaln = SafeTensors::deserialize(&adaln_map)?;
        let rows_table = adaln.tensor("block0.step0")?;
        ensure!(rows_table.dtype() == Dtype::BF16, "block0.step0 dtype");
        let entries = u32::try_from(rows_table.shape()[0])?;
        let table_host: Vec<bf16> = rows_table
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect();

        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let s = &device.stream;
        let runtime = std::env::temp_dir().join(format!("ff-h3-block0-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let blas = Blas::new(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(true, "MiniMax-H3 block gate".into())),
        )?;
        let r = adapter.rank;
        let mut flash = Flash::new(s.clone(), rows, spec.heads, spec.heads, spec.head_dim)?;
        let kernels = BlockKernels::new(&device.ctx)?;
        let weights = BlockWeights::resident(
            s,
            &spec,
            r,
            "transformer_blocks.0",
            &mut Source::new(&transformer, &manifest),
            &mut Source::new(&lora_dir, &lora_manifest),
        )?;
        let mut work = BlockWork::new(s, &spec, r, rows)?;
        work.prepare(&blas, &spec)?;
        let rotary = s.clone_htod(&rotary_host)?;
        let table = s.clone_htod(&table_host)?;
        let index = s.clone_htod(&index)?;
        let modulation = Modulation {
            table: &table,
            offset: 0,
            entries,
            index: &index,
        };
        let mut run = |lora_scale: f32| -> Result<Vec<f64>> {
            let mut x = s.clone_htod(&input)?;
            let context = BlockContext {
                s,
                blas: &blas,
                kernels: &kernels,
                spec: &spec,
                lora_scale,
            };
            block_forward(
                &context,
                &mut flash,
                &weights,
                &Conditioning::Modulated {
                    rotary: &rotary,
                    modulation: &modulation,
                },
                &mut x,
                &mut work,
            )?;
            Ok(s.clone_dtoh(&x)?
                .into_iter()
                .map(|v| f64::from(v.to_f32()))
                .collect())
        };
        let candidate = run(adapter.scale)?;
        let control = run(0.0)?;
        fs::remove_dir(&runtime)?;

        let replay_map = mapped(&path("FF_H3_REPLAY")?.join("replay.safetensors"))?;
        let replay = SafeTensors::deserialize(&replay_map)?;
        let (reference, _) = decoded(&replay, "block0.output")?;
        let (captured, _) = decoded(&io, "block0.output")?;
        ensure!(
            candidate.iter().all(|v| v.is_finite()),
            "non-finite block output"
        );
        let d = distance(&candidate, &reference);
        let d_capture = distance(&candidate, &captured);
        let d_control = distance(&control, &reference);
        println!(
            "H3_BLOCK0 candidate_vs_fp32={d:?} candidate_vs_bf16_capture={d_capture:?} lora0_vs_fp32={d_control:?} rotary_bf16_differs={rotary_differs}/{} threshold={threshold}",
            rows * rotary_width
        );
        ensure!(within(d, threshold)?, "block0 {d:?} outside {threshold}");
        ensure!(
            !within(d_control, threshold)?,
            "LoRA-scale-0 control stayed within {threshold}"
        );
        println!("H3_BLOCK0_GATE PASS control=FAIL_AS_REQUIRED");
        Ok(())
    }

    /// Blocks placed per `FF_H3_PLACEMENT`, eager (the H3 path captures no graph), native AdaLN table. Keyframe
    /// conditioning rows are the captured FP32 VAE-encoder latents, the form under which the frozen keyframe cells
    /// are 0. Every step in `FF_H3_STEPS` (each held by `FF_H3_REPLAY`): the native velocity of the generated video
    /// rows and of the audio rows on the run's teacher-forced inputs, within the frozen `{FF_H3_CELL}.step{k}.*`
    /// thresholds; control: text rows rotated by one must leave a threshold.
    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted, FF_MODEL, FF_H3_LORA, FF_H3_CAPTURE (BF16 run), FF_H3_REPLAY (the replay the frozen step thresholds were measured against), FF_H3_THRESHOLDS, FF_H3_CELL, FF_H3_PLACEMENT, FF_H3_STEPS"]
    fn dit_steps_match_the_fp32_replay() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let path = |key: &str| {
            std::env::var_os(key)
                .map(PathBuf::from)
                .with_context(|| format!("{key} missing"))
        };
        let cell = std::env::var("FF_H3_CELL").context("FF_H3_CELL missing")?;
        let model = path("FF_MODEL")?;
        let capture = path("FF_H3_CAPTURE")?;
        let lora_path = path("FF_H3_LORA")?;
        let thresholds: serde_json::Value =
            serde_json::from_slice(&fs::read(path("FF_H3_THRESHOLDS")?)?)?;
        let thresholds = &thresholds["thresholds"];
        let spec = H3Spec::read(&model)?;
        let transformer = model.join("transformer");
        let checkpoint = MappedCheckpoint::open_component(&transformer, "diffusion_pytorch_model")?;
        let manifest = checkpoint.minimax_h3_manifest(&spec)?;
        let lora_dir = lora_path.parent().context("LoRA directory")?.to_owned();
        let stem = lora_path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix(".safetensors"))
            .context("LoRA file stem")?;
        let (lora_manifest, adapter) =
            MappedCheckpoint::open_component(&lora_dir, stem)?.minimax_h3_lora_manifest(&spec)?;
        let layout = capture_layout(&spec, &capture)?;
        let plan = row_timestep_plan(&layout, &Schedule::new(9, 6.0)?, &Schedule::new(9, 3.0)?)?;
        let adaln_plan = AdalnPlan::build(&layout, &plan);
        let replay_map = mapped(&path("FF_H3_REPLAY")?.join("replay.safetensors"))?;
        let replay = SafeTensors::deserialize(&replay_map)?;
        let steps = std::env::var("FF_H3_STEPS")
            .context("FF_H3_STEPS missing")?
            .split(',')
            .map(|k| Ok(k.parse::<usize>()?))
            .collect::<Result<Vec<usize>>>()?;
        for &k in &steps {
            ensure!(
                k < plan.len() && replay.tensor(&format!("step{k}.noise_pred")).is_ok(),
                "replay holds no step {k} velocity"
            );
        }
        let text_map = mapped(&capture.join("dit_text_input.safetensors"))?;
        let text_file = SafeTensors::deserialize(&text_map)?;
        let text_view = text_file.tensor("encoder_hidden_states")?;
        ensure!(text_view.dtype() == Dtype::BF16, "text input dtype");
        let text_host: Vec<bf16> = text_view
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect();

        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let s = &device.stream;
        let runtime = std::env::temp_dir().join(format!("ff-h3-dit-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let mut blas = Blas::new(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(true, "MiniMax-H3 DiT gate".into())),
        )?;
        let table = AdalnTable::build(
            &device,
            &mut blas,
            &transformer,
            &checkpoint,
            &manifest,
            &spec,
            &adaln_plan,
        )?;
        let mut dit = Dit::new(
            &device,
            &mut blas,
            &spec,
            (adapter.rank, &layout),
            Source::new(&transformer, &manifest),
            Source::new(&lora_dir, &lora_manifest),
            Placement::from_env()?,
        )?;
        let kernels = BlockKernels::new(&device.ctx)?;
        let context = BlockContext {
            s,
            blas: &blas,
            kernels: &kernels,
            spec: &spec,
            lora_scale: adapter.scale,
        };
        let counts = (
            u32::try_from(adaln_plan.pairs.len())?,
            u32::try_from(adaln_plan.timesteps.len())?,
        );
        let text = s.clone_htod(&text_host)?;
        let rotated: Vec<bf16> = text_host[spec.text_dim..]
            .iter()
            .chain(&text_host[..spec.text_dim])
            .copied()
            .collect();
        let rotated = s.clone_htod(&rotated)?;
        let condition = layout.condition_video_rows;
        let patch = spec.video_row_width();
        let (mut control_failed, mut defects) = (false, vec![]);
        for &k in &steps {
            let step_map = mapped(&capture.join(format!("step{k}.safetensors")))?;
            let step_file = SafeTensors::deserialize(&step_map)?;
            let to_f32 = |name: &str| -> Result<Vec<f32>> {
                Ok(decoded(&step_file, name)?
                    .0
                    .into_iter()
                    .map(|v| v as f32)
                    .collect())
            };
            let video = s.clone_htod(&to_f32("hidden_states")?)?;
            let audio = s.clone_htod(&to_f32("audio_hidden_states")?)?;
            let row_pairs = s.clone_htod(&adaln_plan.row_pairs[k])?;
            let row_timesteps = s.clone_htod(&adaln_plan.row_timesteps[k])?;
            let mut evaluate = |text: &CudaSlice<bf16>| -> Result<[[f64; 3]; 2]> {
                dit.step(
                    &context,
                    &table,
                    counts,
                    &StepInputs {
                        video: &video,
                        audio: &audio,
                        text,
                        row_pairs: &row_pairs,
                        row_timesteps: &row_timesteps,
                    },
                )?;
                let candidate: Vec<f64> = s
                    .clone_dtoh(&dit.video)?
                    .into_iter()
                    .map(f64::from)
                    .collect();
                let audio_candidate: Vec<f64> = s
                    .clone_dtoh(&dit.audio)?
                    .into_iter()
                    .map(f64::from)
                    .collect();
                ensure!(
                    candidate
                        .iter()
                        .chain(&audio_candidate)
                        .all(|v| v.is_finite()),
                    "step {k}: non-finite velocity"
                );
                let (reference, _) = decoded(&replay, &format!("step{k}.noise_pred"))?;
                let (audio_reference, _) = decoded(&replay, &format!("step{k}.audio_noise_pred"))?;
                Ok([
                    distance(&candidate, &reference[condition * patch..]),
                    distance(&audio_candidate, &audio_reference),
                ])
            };
            let [d_video, d_audio] = evaluate(&text)?;
            let video_threshold = &thresholds[format!("{cell}.step{k}.noise_pred")];
            let audio_threshold = &thresholds[format!("{cell}.step{k}.audio_noise_pred")];
            let verdict = [
                within(d_video, video_threshold)?,
                within(d_audio, audio_threshold)?,
            ];
            println!(
                "H3_DIT step{k} video={d_video:?} audio={d_audio:?} thresholds={video_threshold} {audio_threshold} within={verdict:?}"
            );
            if verdict != [true, true] {
                defects.push(k);
            }
            if k == steps[0] {
                let [c_video, c_audio] = evaluate(&rotated)?;
                println!("H3_DIT step{k} rotated-text control video={c_video:?} audio={c_audio:?}");
                control_failed =
                    !within(c_video, video_threshold)? || !within(c_audio, audio_threshold)?;
            }
        }
        fs::remove_dir(&runtime)?;
        ensure!(
            control_failed,
            "rotated-text control stayed within the thresholds"
        );
        ensure!(
            defects.is_empty(),
            "H3_DIT_GATE DEFECT cell={cell} steps outside the frozen thresholds: {defects:?} of {steps:?}"
        );
        println!("H3_DIT_GATE PASS cell={cell} steps={steps:?} control=FAIL_AS_REQUIRED");
        Ok(())
    }

    /// Blocks placed per `FF_H3_PLACEMENT`, eager (the H3 path captures no graph), native AdaLN table and grids.
    /// Keyframe conditioning rows are the captured FP32 VAE-encoder latents, the form under which the frozen keyframe
    /// cells are 0. The full chain from the run's step-0 latents against the run's final latents (generated video
    /// rows, audio rows) within the frozen `{FF_H3_CELL}.final.*` thresholds; control: the video and audio grids
    /// swapped must leave a threshold.
    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted, FF_MODEL, FF_H3_LORA, FF_H3_CAPTURE (BF16 run), FF_H3_THRESHOLDS, FF_H3_CELL, FF_H3_PLACEMENT"]
    fn dit_chain_matches_the_reference_final_latents() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let path = |key: &str| {
            std::env::var_os(key)
                .map(PathBuf::from)
                .with_context(|| format!("{key} missing"))
        };
        let cell = std::env::var("FF_H3_CELL").context("FF_H3_CELL missing")?;
        let model = path("FF_MODEL")?;
        let capture = path("FF_H3_CAPTURE")?;
        let lora_path = path("FF_H3_LORA")?;
        let thresholds: serde_json::Value =
            serde_json::from_slice(&fs::read(path("FF_H3_THRESHOLDS")?)?)?;
        let thresholds = &thresholds["thresholds"];
        let spec = H3Spec::read(&model)?;
        let transformer = model.join("transformer");
        let checkpoint = MappedCheckpoint::open_component(&transformer, "diffusion_pytorch_model")?;
        let manifest = checkpoint.minimax_h3_manifest(&spec)?;
        let lora_dir = lora_path.parent().context("LoRA directory")?.to_owned();
        let stem = lora_path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix(".safetensors"))
            .context("LoRA file stem")?;
        let (lora_manifest, adapter) =
            MappedCheckpoint::open_component(&lora_dir, stem)?.minimax_h3_lora_manifest(&spec)?;
        let layout = capture_layout(&spec, &capture)?;
        let video_grid = Schedule::new(9, 6.0)?;
        let audio_grid = Schedule::new(9, 3.0)?;
        let plan = row_timestep_plan(&layout, &video_grid, &audio_grid)?;
        let adaln_plan = AdalnPlan::build(&layout, &plan);
        let read = |file: &str, name: &str| -> Result<(Vec<f64>, Vec<usize>)> {
            let map = mapped(&capture.join(file))?;
            decoded(&SafeTensors::deserialize(&map)?, name)
        };
        let (video0, _) = read("step0.safetensors", "hidden_states")?;
        let (audio0, _) = read("step0.safetensors", "audio_hidden_states")?;
        let (final_video, video_shape) = read("final.safetensors", "latents")?;
        let (final_audio, audio_shape) = read("final.safetensors", "audio_latents")?;
        let patch = spec.video_row_width();
        let geometry = (layout.rows()
            - layout.condition_video_rows
            - layout.audio_indices.len()
            - layout.text_indices.len())
            * patch;
        ensure!(
            video_shape.len() == 5
                && video_shape[0] == 1
                && video_shape[1] * spec.patch.iter().product::<usize>() == patch
                && final_video.len() == geometry
                && audio_shape.len() == 3
                && audio_shape[1] == spec.audio_in_channels
                && final_audio.len() == audio0.len(),
            "final latents {video_shape:?} {audio_shape:?} do not match the layout"
        );
        // The reference stores `[1, channels, frames, height, width]` video and `[audio_channels, channels, latents]`
        // audio; packed rows are frame-major then row-major with columns (channel, patch_h, patch_w), audio rows
        // channel-major.
        let (channels, frames, height, width) = (
            video_shape[1],
            video_shape[2],
            video_shape[3],
            video_shape[4],
        );
        let (ph, pw) = (spec.patch[1], spec.patch[2]);
        let mut final_video_rows = Vec::with_capacity(final_video.len());
        for frame in 0..frames {
            for i in 0..height / ph {
                for j in 0..width / pw {
                    for c in 0..channels {
                        for y in 0..ph {
                            for x in 0..pw {
                                final_video_rows.push(
                                    final_video[((c * frames + frame) * height + i * ph + y)
                                        * width
                                        + j * pw
                                        + x],
                                );
                            }
                        }
                    }
                }
            }
        }
        let (stereo, latents) = (audio_shape[0], audio_shape[2]);
        let mut final_audio_rows = Vec::with_capacity(final_audio.len());
        for ch in 0..stereo {
            for t in 0..latents {
                for c in 0..audio_shape[1] {
                    final_audio_rows.push(final_audio[(ch * audio_shape[1] + c) * latents + t]);
                }
            }
        }
        let text_map = mapped(&capture.join("dit_text_input.safetensors"))?;
        let text_file = SafeTensors::deserialize(&text_map)?;
        let text_host: Vec<bf16> = text_file
            .tensor("encoder_hidden_states")?
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect();

        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let s = &device.stream;
        let runtime = std::env::temp_dir().join(format!("ff-h3-chain-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let mut blas = Blas::new(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(true, "MiniMax-H3 chain gate".into())),
        )?;
        let table = AdalnTable::build(
            &device,
            &mut blas,
            &transformer,
            &checkpoint,
            &manifest,
            &spec,
            &adaln_plan,
        )?;
        let mut dit = Dit::new(
            &device,
            &mut blas,
            &spec,
            (adapter.rank, &layout),
            Source::new(&transformer, &manifest),
            Source::new(&lora_dir, &lora_manifest),
            Placement::from_env()?,
        )?;
        let kernels = BlockKernels::new(&device.ctx)?;
        let context = BlockContext {
            s,
            blas: &blas,
            kernels: &kernels,
            spec: &spec,
            lora_scale: adapter.scale,
        };
        let text = s.clone_htod(&text_host)?;
        let condition = layout.condition_video_rows * patch;
        let mut chain = |grids: (&Schedule, &Schedule)| -> Result<[[f64; 3]; 2]> {
            let video = s.clone_htod(&video0.iter().map(|&v| v as f32).collect::<Vec<_>>())?;
            let audio = s.clone_htod(&audio0.iter().map(|&v| v as f32).collect::<Vec<_>>())?;
            dit.sample(
                &context,
                (&table, &adaln_plan),
                grids,
                (&video, &audio),
                &text,
            )?;
            let video: Vec<f64> = s.clone_dtoh(&video)?.into_iter().map(f64::from).collect();
            let audio: Vec<f64> = s.clone_dtoh(&audio)?.into_iter().map(f64::from).collect();
            ensure!(
                video.iter().chain(&audio).all(|v| v.is_finite()),
                "non-finite latents"
            );
            ensure!(
                video[..condition] == video0[..condition],
                "conditioning rows moved"
            );
            Ok([
                distance(&video[condition..], &final_video_rows),
                distance(&audio, &final_audio_rows),
            ])
        };
        let [d_video, d_audio] = chain((&video_grid, &audio_grid))?;
        let video_threshold = &thresholds[format!("{cell}.final.latents")];
        let audio_threshold = &thresholds[format!("{cell}.final.audio_latents")];
        println!(
            "H3_CHAIN video={d_video:?} audio={d_audio:?} thresholds={video_threshold} {audio_threshold}"
        );
        ensure!(
            within(d_video, video_threshold)? && within(d_audio, audio_threshold)?,
            "final latents outside the frozen thresholds"
        );
        let [c_video, c_audio] = chain((&audio_grid, &video_grid))?;
        println!("H3_CHAIN swapped-grid control video={c_video:?} audio={c_audio:?}");
        fs::remove_dir(&runtime)?;
        ensure!(
            !within(c_video, video_threshold)? || !within(c_audio, audio_threshold)?,
            "swapped-grid control stayed within the thresholds"
        );
        println!("H3_CHAIN_GATE PASS cell={cell} control=FAIL_AS_REQUIRED");
        Ok(())
    }
}
