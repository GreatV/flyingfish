//! MiniMax-H3 FL2VA transformer execution.
use crate::backend::cuda::{
    Device,
    blas::Blas,
    contracts::{Tensor, TensorMut, view},
    cubin,
    flash::Flash,
    linear::LinearBiasMode,
    ops::flat,
    verification::Shape,
};
use crate::backend::gpu::{
    CudaSlice, CudaStream, CudaViewMut, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
};
use crate::checkpoint::{MappedCheckpoint, TensorEntry, TensorManifest};
use crate::experts::disk::{AlignedBytes, SpanReader};
use crate::minimax_h3::{AdalnPlan, H3Spec, MODALITIES, adaln_input};
use anyhow::{Context, Result, ensure};
use half::bf16;
use safetensors::Dtype;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
};

/// Direct-IO read granularity.
const READ_CHUNK_BYTES: u64 = 32 << 20;

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

    /// Copies a finite BF16 tensor into `destination`; the pageable source is consumed when the copy call returns.
    fn upload(
        &mut self,
        stream: &Arc<CudaStream>,
        name: &str,
        entry: &TensorEntry,
        destination: &mut CudaViewMut<'_, bf16>,
    ) -> Result<()> {
        ensure!(
            entry.dtype == Dtype::BF16 && entry.bytes == 2 * destination.len(),
            "{name}: {:?} {} B does not fill a {}-element BF16 buffer",
            entry.dtype,
            entry.bytes,
            destination.len()
        );
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
        let bytes =
            &reader.read_span(&span, READ_CHUNK_BYTES, staging)?[begin..begin + entry.bytes];
        ensure!(
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .all(|v| u16::from_le_bytes(*v) & 0x7f80 != 0x7f80),
            "{name}: nonfinite BF16 value"
        );
        let (pointer, _record) = destination.device_ptr_mut(stream);
        unsafe {
            cudarc::driver::result::memcpy_htod_async(pointer, bytes, stream.cu_stream())?;
        }
        Ok(())
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

/// Rows per FFN chunk.
const FFN_CHUNK_ROWS: usize = 1024;
/// Threads per hidden-width RMSNorm row: 672 eight-element vectors of a 5376 row fit one register pass.
const HIDDEN_NORM_THREADS: u32 = 768;
/// Threads per head-width RMSNorm row.
const HEAD_NORM_THREADS: u32 = 32;

/// One transformer block's BF16 weights and LoRA factors; the q, k, v LoRA A factors are stacked `[3 * rank, hidden]`.
pub(crate) struct BlockWeights {
    norm1: CudaSlice<bf16>,
    norm2: CudaSlice<bf16>,
    norm_q: CudaSlice<bf16>,
    norm_k: CudaSlice<bf16>,
    to_q: CudaSlice<bf16>,
    to_k: CudaSlice<bf16>,
    to_v: CudaSlice<bf16>,
    to_out: CudaSlice<bf16>,
    ff_in: CudaSlice<bf16>,
    ff_out: CudaSlice<bf16>,
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

    fn into(
        &mut self,
        s: &Arc<CudaStream>,
        name: &str,
        destination: &mut CudaViewMut<'_, bf16>,
    ) -> Result<()> {
        let entry = self
            .manifest
            .entries
            .get(name)
            .with_context(|| format!("manifest lacks {name}"))?;
        self.reader.upload(s, name, entry, destination)
    }

    fn tensor(
        &mut self,
        s: &Arc<CudaStream>,
        name: &str,
        elements: usize,
    ) -> Result<CudaSlice<bf16>> {
        let mut buffer = s.alloc_zeros::<bf16>(elements)?;
        self.into(s, name, &mut buffer.slice_mut(..))?;
        Ok(buffer)
    }
}

impl BlockWeights {
    pub(crate) fn load(
        s: &Arc<CudaStream>,
        spec: &H3Spec,
        rank: usize,
        block: &str,
        base: &mut Source<'_>,
        lora: &mut Source<'_>,
    ) -> Result<Self> {
        let (h, w, f, r) = (spec.hidden, spec.heads * spec.head_dim, spec.ffn, rank);
        let mut qkv_a = s.alloc_zeros::<bf16>(3 * r * h)?;
        for (i, name) in ["to_q", "to_k", "to_v"].iter().enumerate() {
            lora.into(
                s,
                &format!("{block}.attn.{name}.lora_A.default.weight"),
                &mut qkv_a.slice_mut(i * r * h..(i + 1) * r * h),
            )?;
        }
        let mut b = |name: &str, n| base.tensor(s, &format!("{block}.{name}.weight"), n);
        let (norm1, norm2) = (b("norm1", h)?, b("norm2", h)?);
        let (norm_q, norm_k) = (
            b("attn.norm_q", spec.head_dim)?,
            b("attn.norm_k", spec.head_dim)?,
        );
        let (to_q, to_k, to_v) = (
            b("attn.to_q", w * h)?,
            b("attn.to_k", w * h)?,
            b("attn.to_v", w * h)?,
        );
        let (to_out, ff_in, ff_out) = (
            b("attn.to_out.0", h * w)?,
            b("ff.net.0.proj", 2 * f * h)?,
            b("ff.net.2", h * f)?,
        );
        let mut l = |name: &str, factor: &str, n| {
            lora.tensor(
                s,
                &format!("{block}.{name}.lora_{factor}.default.weight"),
                n,
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
            to_q,
            to_k,
            to_v,
            to_out,
            ff_in,
            ff_out,
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
}

impl BlockKernels {
    pub(crate) fn new(ctx: &Arc<crate::backend::gpu::CudaContext>) -> Result<Self> {
        let modulate = cubin::module(ctx, "modulate")?;
        Ok(Self {
            norm: cubin::module(ctx, "norm")?.load_function("rms_norm_bf16")?,
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
    pub(crate) fn new(
        s: &Arc<CudaStream>,
        spec: &H3Spec,
        rank: usize,
        rows: usize,
    ) -> Result<Self> {
        let width = spec.heads * spec.head_dim;
        let chunk = rows.min(FFN_CHUNK_ROWS);
        let attention = || s.alloc_zeros::<bf16>(rows * width);
        Ok(Self {
            rows,
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
    c.blas.linear(
        a,
        &x.buffer.slice(x.offset..x.offset + rows * x.width),
        &mut work.slice_mut(..rows * rank),
        rows,
        rank,
        x.width,
    )?;
    c.blas.linear_accumulate(
        matrix(b, c.s, y.width, rank)?,
        rows_view(work, c.s, 0, rows, rank, rank)?,
        &mut TensorMut(rows_view(y.buffer, c.s, y.offset, rows, y.width, y.width)?),
        c.lora_scale,
    )
}

/// One MiniMax-H3 transformer block over `x [rows, hidden]` in place, BF16, LoRA applied online at
/// `c.lora_scale`. `rotary` is `[rows, 6 * rope_freq_dim]` from `rotary_table`.
pub(crate) fn block_forward(
    c: &BlockContext<'_>,
    flash: &mut Flash,
    w: &BlockWeights,
    rotary: &CudaSlice<bf16>,
    modulation: &Modulation<'_>,
    x: &mut CudaSlice<bf16>,
    work: &mut BlockWork,
) -> Result<()> {
    let (s, blas, kernels, spec, lora_scale) = (c.s, c.blas, c.kernels, c.spec, c.lora_scale);
    let rows = work.rows;
    let (h, width, f, r) = (spec.hidden, spec.heads * spec.head_dim, spec.ffn, w.rank);
    let rotary_width = 6 * spec.rope_freq_dim;
    ensure!(
        x.len() == rows * h
            && rotary.len() == rows * rotary_width
            && modulation.index.len() == rows,
        "block extents differ from the work rows"
    );
    let stride = 6 * h;
    ensure!(
        modulation.offset + modulation.entries as usize * stride <= modulation.table.len(),
        "modulation table extent"
    );
    let table = pointer(modulation.table, s) + 2 * modulation.offset as u64;
    let chunk = |c: usize| table + (2 * c * h) as u64;
    let index = pointer(modulation.index, s);
    let rows_i = i32::try_from(rows)?;
    let row_grid = LaunchConfig {
        grid_dim: (u32::try_from(rows)?, u32::try_from(h.div_ceil(256))?, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let norm = |input: u64,
                weight: u64,
                output: u64,
                count: usize,
                dim: usize,
                threads: u32,
                eps: f32|
     -> Result<()> {
        unsafe {
            s.launch_builder(&kernels.norm)
                .arg(&input)
                .arg(&weight)
                .arg(&output)
                .arg(&i32::try_from(count)?)
                .arg(&i32::try_from(dim)?)
                .arg(&eps)
                .launch(LaunchConfig {
                    grid_dim: (u32::try_from(count)?, 1, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
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
                .arg(&modulation.entries)
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
                .arg(&modulation.entries)
                .launch(row_grid)?;
        }
        Ok(())
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
    modulate(normed, 0, 1)?;
    for (weight, out) in [
        (&w.to_q, &mut work.q_raw),
        (&w.to_k, &mut work.k_raw),
        (&w.to_v, &mut work.v_raw),
    ] {
        blas.linear(
            weight,
            &work.normed.slice(..),
            &mut out.slice_mut(..),
            rows,
            width,
            h,
        )?;
    }
    blas.linear(
        &w.qkv_a,
        &work.normed.slice(..),
        &mut work.lora.slice_mut(..),
        rows,
        3 * r,
        h,
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
            .arg(&pointer(rotary, s))
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
    blas.linear(
        &w.to_out,
        &work.v_raw.slice(..),
        &mut work.normed.slice_mut(..),
        rows,
        h,
        width,
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
    gated_add(xp, normed, 2)?;

    norm(
        xp,
        pointer(&w.norm2, s),
        normed,
        rows,
        h,
        HIDDEN_NORM_THREADS,
        spec.norm_eps,
    )?;
    modulate(normed, 3, 4)?;
    for start in (0..rows).step_by(FFN_CHUNK_ROWS) {
        let n = FFN_CHUNK_ROWS.min(rows - start);
        blas.linear(
            &w.ff_in,
            &work.normed.slice(start * h..(start + n) * h),
            &mut work.hidden.slice_mut(..n * 2 * f),
            n,
            2 * f,
            h,
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
        unsafe {
            s.launch_builder(&kernels.swiglu)
                .arg(&pointer(&work.hidden, s))
                .arg(&pointer(&work.gated, s))
                .arg(&i32::try_from(n)?)
                .arg(&i32::try_from(f)?)
                .arg(&3i32)
                .launch(flat(n * f))?;
        }
        blas.linear(
            &w.ff_out,
            &work.gated.slice(..n * f),
            &mut work.normed.slice_mut(start * h..(start + n) * h),
            n,
            h,
            f,
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
    gated_add(xp, normed, 5)?;
    Ok(())
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

    /// Streamed (disk tier), eager. Candidate rows of every block and step against the FP32 replay on the
    /// addressed rows, within the frozen `A.adaln.*` thresholds; control: the same rows scored against the replay's
    /// video/audio-swapped rows must leave a threshold.
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
        let mut blas = Blas::new(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(true, "MiniMax-H3 block gate".into())),
        )?;
        let (h, width, ffn, r) = (
            spec.hidden,
            spec.heads * spec.head_dim,
            spec.ffn,
            adapter.rank,
        );
        blas.declare(
            rows,
            &[
                (width, h),
                (h, width),
                (2 * ffn, h),
                (h, ffn),
                (3 * r, h),
                (r, width),
                (r, h),
                (r, ffn),
            ]
            .into_iter()
            .collect(),
        );
        let mut flash = Flash::new(s.clone(), rows, spec.heads, spec.heads, spec.head_dim)?;
        let kernels = BlockKernels::new(&device.ctx)?;
        let weights = BlockWeights::load(
            s,
            &spec,
            r,
            "transformer_blocks.0",
            &mut Source::new(&transformer, &manifest),
            &mut Source::new(&lora_dir, &lora_manifest),
        )?;
        let mut work = BlockWork::new(s, &spec, r, rows)?;
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
                &rotary,
                &modulation,
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
}
