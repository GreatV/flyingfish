//! MiniMax-H3 FL2VA transformer execution.
use crate::backend::cuda::weights::{
    DequantCopy, QuantKernels, dequantizes, device_slot_bytes, quant_source_finite,
};
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
    CudaFunction, CudaSlice, CudaStream, CudaViewMut, DevicePtr, DevicePtrMut, LaunchConfig,
    PushKernelArg,
};
use crate::checkpoint::{TensorEntry, TensorManifest};
use crate::experts::disk::{AlignedBytes, SpanReader};
use crate::gguf::GgmlType;
use crate::minimax_h3::{AdalnPlan, AdalnTimeEmbedder, H3Spec, MODALITIES, adaln_input};
use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use safetensors::Dtype;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

#[path = "engine_minimax_h3_audio.rs"]
mod audio;
#[path = "engine_minimax_h3_pipeline.rs"]
mod pipeline;
#[path = "engine_minimax_h3_vae.rs"]
mod vae;

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

    /// The tensor's source bytes in a device buffer, for a quantized GGUF entry.
    fn read_device(
        &mut self,
        stream: &Arc<CudaStream>,
        name: &str,
        entry: &TensorEntry,
    ) -> Result<CudaSlice<u8>> {
        let bytes = self.read(name, entry)?;
        quant_source_finite(name, entry, bytes)?;
        let mut buffer = stream.alloc_zeros::<u8>(entry.bytes)?;
        let (pointer, record) = buffer.device_ptr_mut(stream);
        unsafe {
            cudarc::driver::result::memcpy_htod_async(pointer, bytes, stream.cu_stream())?;
        }
        drop(record);
        Ok(buffer)
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

    /// Copies a quantized tensor's stored source bytes into host memory of exactly its size.
    fn read_raw(&mut self, name: &str, entry: &TensorEntry, destination: &mut [u8]) -> Result<()> {
        ensure!(
            entry.bytes == destination.len(),
            "{name}: {} B does not fill {} B",
            entry.bytes,
            destination.len()
        );
        destination.copy_from_slice(self.read(name, entry)?);
        Ok(())
    }

    /// The tensor's bytes, read with O_DIRECT into the reusable aligned staging buffer.
    fn read(&mut self, name: &str, entry: &TensorEntry) -> Result<&[u8]> {
        self.read_range(name, entry, 0, entry.bytes)
    }

    /// Bytes `start..start + bytes` of the tensor, read with O_DIRECT into the reusable aligned staging buffer.
    fn read_range(
        &mut self,
        name: &str,
        entry: &TensorEntry,
        start: usize,
        bytes: usize,
    ) -> Result<&[u8]> {
        ensure!(
            start
                .checked_add(bytes)
                .is_some_and(|end| end <= entry.bytes),
            "{name}: range {start}+{bytes} outside {} B",
            entry.bytes
        );
        if !self.readers.contains_key(&entry.shard) {
            let reader = SpanReader::open(&self.dir.join(&entry.shard), true)?;
            self.readers.insert(entry.shard.clone(), reader);
        }
        let reader = &self.readers[&entry.shard];
        let offset = entry
            .data_start
            .checked_add(entry.data_offsets[0])
            .and_then(|offset| offset.checked_add(start as u64))
            .context("tensor offset overflow")?;
        let span = reader
            .plan(&[(offset, bytes as u64)], READ_CHUNK_BYTES)?
            .pop()
            .with_context(|| format!("{name}: no read span"))?;
        let alignment = reader.alignment as usize;
        let needed = usize::try_from(span.end - span.begin)?.next_multiple_of(alignment);
        if self.staging.as_ref().is_none_or(|s| s.len() < needed) {
            self.staging = Some(AlignedBytes::new(needed, alignment)?);
        }
        let staging = self.staging.as_mut().context("staging buffer")?;
        let begin = usize::try_from(offset - span.begin)?;
        Ok(&reader.read_span(&span, READ_CHUNK_BYTES, staging)?[begin..begin + bytes])
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
        manifest: &TensorManifest,
        spec: &H3Spec,
        plan: &AdalnPlan,
    ) -> Result<Self> {
        let s = &d.stream;
        let timesteps = plan.timesteps.len();
        let width = 6 * spec.hidden;
        let modulation = width * MODALITIES as usize;
        let mut source = Source::new(transformer_dir, manifest);
        let time = AdalnTimeEmbedder {
            linear_1_weight: source.host_f32(
                s,
                "time_embedder.linear_1.weight",
                spec.time_hidden * spec.freq_dim,
            )?,
            linear_1_bias: source.host_f32(s, "time_embedder.linear_1.bias", spec.time_hidden)?,
            linear_2_weight: source.host_f32(
                s,
                "time_embedder.linear_2.weight",
                spec.time_dim * spec.time_hidden,
            )?,
            linear_2_bias: source.host_f32(s, "time_embedder.linear_2.bias", spec.time_dim)?,
        };
        let x = s.clone_htod(&adaln_input(spec, &time, &plan.timesteps)?)?;
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
                source.fill(s, &name, &mut buffer.slice_mut(..))?;
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
            source.fill(s, &name, &mut buffer.slice_mut(..))?;
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

/// Initial FFN chunk ceiling before weight placement.
const FFN_CHUNK_ROWS: usize = 1024;

/// Rows per FFN chunk: `rows` split evenly over `ceil(rows / FFN_CHUNK_ROWS)` chunks, so the tail chunk holds at
/// least `chunk - (chunks - 1)` rows.
fn ffn_chunk(rows: usize) -> usize {
    rows.div_ceil(rows.div_ceil(FFN_CHUNK_ROWS))
}

#[derive(Clone, Copy, Debug)]
struct FfnPlan {
    chunk: usize,
    borrowed: bool,
    allocation: usize,
    workspace: usize,
    tensor_cost: f64,
    launches: usize,
}

impl FfnPlan {
    fn alias_capacity(attention: (usize, usize), ffn: usize) -> Result<usize> {
        ensure!(ffn > 0, "zero FFN width");
        let hidden = ffn.checked_mul(2).context("FFN hidden width overflow")?;
        Ok((attention.0 / hidden).min(attention.1 / ffn))
    }

    fn allocation(chunk: usize, ffn: usize) -> Result<usize> {
        chunk
            .checked_mul(ffn)
            .and_then(|n| n.checked_mul(3 * std::mem::size_of::<bf16>()))
            .context("FFN workspace extent overflow")
    }

    fn select(choices: &[Self], budget: usize) -> Result<Self> {
        let mut best = None;
        for &plan in choices {
            let allocation = plan
                .allocation
                .checked_add(plan.workspace)
                .context("FFN allocation overflow")?;
            if allocation > budget {
                continue;
            }
            ensure!(
                plan.tensor_cost.is_finite() && plan.tensor_cost >= 0.0,
                "invalid FFN tensor cost"
            );
            let score = (plan.tensor_cost, plan.launches, allocation);
            if best.as_ref().is_none_or(|(_, previous)| score < *previous) {
                best = Some((plan, score));
            }
        }
        Ok(best
            .context("no FFN plan fits the measured workspace budget")?
            .0)
    }
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
    quant: OnceLock<QuantKernels>,
}

impl<'a> Source<'a> {
    pub(crate) fn new(dir: &Path, manifest: &'a TensorManifest) -> Self {
        Self {
            reader: EntryReader::new(dir),
            manifest,
            quant: OnceLock::new(),
        }
    }

    fn quant_kernels(&self, s: &Arc<CudaStream>) -> Result<&QuantKernels> {
        if let Some(kernels) = self.quant.get() {
            return Ok(kernels);
        }
        let kernels = QuantKernels::new(s.context())?;
        let _ = self.quant.set(kernels);
        self.quant.get().context("quant kernels")
    }

    fn fill<T: Element>(
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
        let quant = match entry.quant {
            Some(quant) => quant,
            None => return self.reader.upload(s, name, entry, destination),
        };
        match quant {
            GgmlType::F32 | GgmlType::Bf16 => self.reader.upload(s, name, entry, destination),
            GgmlType::F16 | GgmlType::Q4_0 => {
                let src = self.reader.read_device(s, name, entry)?;
                let kernels = self.quant_kernels(s)?;
                let elements = destination.len();
                let (pointer, _record) = destination.device_ptr_mut(s);
                kernels.dequant(
                    s,
                    name,
                    entry,
                    DequantCopy {
                        source: src.device_ptr(s).0,
                        destination: pointer,
                        elements,
                        dtype: T::DTYPE,
                    },
                )?;
                Ok(())
            }
            other => bail!("{name}: unsupported GGUF type {other:?}"),
        }
    }

    fn tensor<T: Element + crate::backend::gpu::ValidAsZeroBits>(
        &mut self,
        s: &Arc<CudaStream>,
        name: &str,
        elements: usize,
    ) -> Result<CudaSlice<T>> {
        let mut buffer = s.alloc_zeros::<T>(elements)?;
        self.fill(s, name, &mut buffer.slice_mut(..))?;
        Ok(buffer)
    }

    fn host_f32(&mut self, s: &Arc<CudaStream>, name: &str, elements: usize) -> Result<Vec<f32>> {
        let tensor = self.tensor::<f32>(s, name, elements)?;
        Ok(s.clone_dtoh(&tensor)?)
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

fn matrix_layout(
    manifest: &TensorManifest,
    block: &str,
    roles: &[&str],
) -> Result<(MatrixLayout, usize)> {
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
        elements += entry.shape[0] * entry.shape[1];
    }
    ensure!(
        layout.len() == roles.len() && roles.iter().all(|r| layout.contains_key(*r)),
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
        let (layout, elements) = matrix_layout(base.manifest, block, &LINEAR_ROLES)?;
        let mut storage = s.alloc_zeros::<bf16>(elements)?;
        for (role, (offset, shape)) in &layout {
            let count = shape[0] as usize * shape[1] as usize;
            base.fill(
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
        let (layout, _) = matrix_layout(base.manifest, block, &LINEAR_ROLES)?;
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
            lora.fill(
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
    chunk: usize,
    fixed_lora: bool,
    #[cfg(test)]
    repeat_ffn_input: bool,
    normed: CudaSlice<bf16>,
    q_raw: CudaSlice<bf16>,
    k_raw: CudaSlice<bf16>,
    v_raw: CudaSlice<bf16>,
    q: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    lora: CudaSlice<bf16>,
    hidden: Tensor<bf16, 2>,
    gated: Tensor<bf16, 2>,
    ffn_owned: Option<(CudaSlice<bf16>, CudaSlice<bf16>)>,
}

impl BlockWork {
    /// Binds the fixed GEMM plans of every base and LoRA-A linear shape these rows run: q/k/v, out, the stacked
    /// q/k/v LoRA A and the out LoRA A over all rows; the two FFN projections and their LoRA A over full and tail
    /// chunks.
    pub(crate) fn prepare(&self, blas: &Blas, spec: &H3Spec) -> Result<()> {
        let (h, width, f, r) = (spec.hidden, spec.heads * spec.head_dim, spec.ffn, self.rank);
        let mut shapes = vec![
            (self.rows, width, h, false),
            (self.rows, h, width, false),
            (self.rows, 3 * r, h, false),
            (self.rows, r, width, false),
        ];
        for rows in [self.chunk, self.rows % self.chunk]
            .into_iter()
            .filter(|&n| n > 0)
        {
            shapes.extend([
                (rows, 2 * f, h, false),
                (rows, h, f, false),
                (rows, r, h, self.fixed_lora),
                (rows, r, f, self.fixed_lora),
            ]);
            if self.fixed_lora {
                shapes.extend([(rows, 2 * f, r, true), (rows, h, r, true)]);
            }
        }
        for (rows, output, input, family) in shapes {
            if family {
                blas.prepare_row_family(crate::backend::setup::LinearShape {
                    rows,
                    output,
                    input,
                })?;
            } else {
                blas.prepare_fixed(
                    u32::try_from(rows)?,
                    u32::try_from(output)?,
                    u32::try_from(input)?,
                )?;
            }
        }
        Ok(())
    }

    fn ffn_cost(
        &self,
        blas: &Blas,
        spec: &H3Spec,
        chunk: usize,
        describe: bool,
    ) -> Result<FfnPlan> {
        let mut plan = FfnPlan {
            chunk,
            borrowed: false,
            allocation: if chunk == self.chunk {
                0
            } else {
                FfnPlan::allocation(chunk, spec.ffn)?
            },
            workspace: 0,
            tensor_cost: 0.0,
            launches: 0,
        };
        for (rows, count) in [(chunk, self.rows / chunk), (self.rows % chunk, 1)] {
            if rows == 0 || count == 0 {
                continue;
            }
            let mut launches = 1;
            for (index, (output, input)) in [
                (2 * spec.ffn, spec.hidden),
                (spec.hidden, spec.ffn),
                (self.rank, spec.hidden),
                (self.rank, spec.ffn),
                (2 * spec.ffn, self.rank),
                (spec.hidden, self.rank),
            ]
            .into_iter()
            .enumerate()
            {
                let (wave_cost, reduction) = blas.fixed_plan_cost(
                    crate::backend::setup::LinearShape {
                        rows,
                        output,
                        input,
                    },
                    describe,
                    index >= 2,
                )?;
                plan.tensor_cost +=
                    2.0 * rows as f64 * output as f64 * input as f64 * count as f64 * wave_cost;
                plan.workspace = plan.workspace.max(blas.fixed_workspace(
                    crate::backend::setup::LinearShape {
                        rows,
                        output,
                        input,
                    },
                    index >= 2,
                )?);
                launches += 1 + reduction;
            }
            plan.launches += count * launches;
        }
        Ok(plan)
    }

    fn ffn_choices(&self, blas: &Blas, spec: &H3Spec, budget: usize) -> Result<Vec<FfnPlan>> {
        let mut choices = Vec::new();
        for chunks in 1..=self.rows.div_ceil(self.chunk) {
            let chunk = self.rows.div_ceil(chunks);
            let tail = self.rows % chunk;
            if tail > 0 && tail <= crate::linear_plan::MAX_SKINNY_ROWS {
                continue;
            }
            if chunk != self.chunk && FfnPlan::allocation(chunk, spec.ffn)? > budget {
                continue;
            }
            match self.ffn_cost(blas, spec, chunk, false) {
                Ok(plan) => choices.push(plan),
                Err(error) if error.downcast_ref::<cudarc::cublaslt::result::CublasError>()
                    .is_some_and(|e| e.0 == cudarc::cublaslt::sys::cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED) => {
                    eprintln!("FFN_PLAN_REJECT rows={} chunk={chunk} error={error:#}",self.rows);
                }
                Err(error) => return Err(error.context(format!("FFN plan rows={} chunk={chunk}",self.rows))),
            }
        }
        ensure!(
            choices
                .iter()
                .any(|p| p.chunk == self.chunk && p.allocation == 0),
            "existing FFN plan missing from workspace choices"
        );
        Ok(choices)
    }

    fn alias_choices(&self, blas: &Blas, spec: &H3Spec, owned: FfnPlan) -> Result<Vec<FfnPlan>> {
        let capacity = FfnPlan::alias_capacity((self.q_raw.len(), self.k_raw.len()), spec.ffn)?;
        let mut choices = vec![owned];
        for chunks in 1..=self.rows.div_ceil(self.chunk) {
            let chunk = self.rows.div_ceil(chunks);
            let tail = self.rows % chunk;
            if chunk > capacity || (tail > 0 && tail <= crate::linear_plan::MAX_SKINNY_ROWS) {
                continue;
            }
            let mut plan = match self.ffn_cost(blas, spec, chunk, false) {
                Ok(plan) => plan,
                Err(error) if error.downcast_ref::<cudarc::cublaslt::result::CublasError>()
                    .is_some_and(|e| e.0 == cudarc::cublaslt::sys::cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED) => {
                    eprintln!("FFN_ALIAS_REJECT rows={} chunk={chunk} error={error:#}",self.rows);
                    continue;
                }
                Err(error) => return Err(error.context(format!("FFN alias rows={} chunk={chunk}",self.rows))),
            };
            if plan.tensor_cost <= owned.tensor_cost && plan.launches <= owned.launches {
                plan.borrowed = true;
                plan.allocation = 0;
                choices.push(plan);
            }
        }
        Ok(choices)
    }

    fn bind_ffn(
        &mut self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        spec: &H3Spec,
        plan: FfnPlan,
        x: &CudaSlice<bf16>,
    ) -> Result<()> {
        let allocation = if plan.borrowed || plan.chunk == self.chunk {
            0
        } else {
            FfnPlan::allocation(plan.chunk, spec.ffn)?
        };
        ensure!(
            plan.chunk >= self.chunk && plan.chunk <= self.rows && plan.allocation == allocation,
            "FFN binding differs from the counted allocation"
        );
        if plan.borrowed {
            self.hidden = rows_view(&self.q_raw, s, 0, plan.chunk, 2 * spec.ffn, 2 * spec.ffn)?;
            self.gated = rows_view(&self.k_raw, s, 0, plan.chunk, spec.ffn, spec.ffn)?;
            self.ffn_owned = None;
        } else if plan.chunk != self.chunk {
            let hidden = s.alloc_zeros::<bf16>(plan.chunk * 2 * spec.ffn)?;
            let gated = s.alloc_zeros::<bf16>(plan.chunk * spec.ffn)?;
            self.hidden = matrix(&hidden, s, plan.chunk, 2 * spec.ffn)?;
            self.gated = matrix(&gated, s, plan.chunk, spec.ffn)?;
            self.ffn_owned = Some((hidden, gated));
        }
        crate::backend::cuda::contracts::no_overlap(
            &self.hidden,
            &self.gated,
            "FFN hidden",
            "FFN gated",
        )?;
        for (name, protected) in [
            ("normed", &self.normed),
            ("LoRA", &self.lora),
            ("residual", x),
        ] {
            let protected = matrix(protected, s, 1, protected.len())?;
            crate::backend::cuda::contracts::no_overlap(
                &self.hidden,
                &protected,
                "FFN hidden",
                name,
            )?;
            crate::backend::cuda::contracts::no_overlap(
                &self.gated,
                &protected,
                "FFN gated",
                name,
            )?;
        }
        self.chunk = plan.chunk;
        ensure!(
            self.hidden.ptr.is_multiple_of(256) && self.gated.ptr.is_multiple_of(256),
            "FFN view alignment"
        );
        self.fixed_lora = true;
        self.prepare(blas, spec)?;
        if self.rows > FFN_CHUNK_ROWS {
            self.ffn_cost(blas, spec, self.chunk, true)?;
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
        let normed = s.alloc_zeros::<bf16>(rows * spec.hidden)?;
        let (q_raw, k_raw, v_raw, q, k) = (
            attention()?,
            attention()?,
            attention()?,
            attention()?,
            attention()?,
        );
        let lora = s.alloc_zeros::<bf16>(rows * 3 * rank)?;
        let hidden = s.alloc_zeros::<bf16>(chunk * 2 * spec.ffn)?;
        let gated = s.alloc_zeros::<bf16>(chunk * spec.ffn)?;
        let hidden_view = matrix(&hidden, s, chunk, 2 * spec.ffn)?;
        let gated_view = matrix(&gated, s, chunk, spec.ffn)?;
        Ok(Self {
            rows,
            rank,
            chunk,
            fixed_lora: false,
            #[cfg(test)]
            repeat_ffn_input: false,
            normed,
            q_raw,
            k_raw,
            v_raw,
            q,
            k,
            lora,
            hidden: hidden_view,
            gated: gated_view,
            ffn_owned: Some((hidden, gated)),
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

/// `y += lora_scale * (x * a^T) * b^T`; the intermediate row pitch is `rank`.
fn lora_pair(
    c: &BlockContext<'_>,
    x: Tensor<bf16, 2>,
    (a, b): (&CudaSlice<bf16>, &CudaSlice<bf16>),
    work: &mut CudaSlice<bf16>,
    y: Tensor<bf16, 2>,
    fixed: bool,
) -> Result<()> {
    let input = x.shape[1] as usize;
    let output = y.shape[1] as usize;
    let rows = x.shape[0] as usize;
    ensure!(a.len().is_multiple_of(input), "LoRA-A extent");
    let rank = a.len() / input;
    let aw = matrix(a, c.s, rank, input)?;
    let mut ay = TensorMut(rows_view(work, c.s, 0, rows, rank, rank)?);
    if fixed {
        c.blas.linear_row_family(aw, x, &mut ay, 1.0, 0.0)?;
    } else {
        c.blas.linear_typed(
            crate::op_contract::ExecutionPhase::Prefill,
            crate::backend::cuda::linear::LinearWeightRef::Bf16(aw),
            x,
            &mut ay,
        )?;
    }
    let bw = matrix(b, c.s, output, rank)?;
    let bx = rows_view(work, c.s, 0, rows, rank, rank)?;
    let mut by = TensorMut(y);
    if fixed {
        c.blas.linear_row_family(bw, bx, &mut by, c.lora_scale, 1.0)
    } else {
        c.blas.linear_accumulate(bw, bx, &mut by, c.lora_scale)
    }
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
    let (h, width, r) = (spec.hidden, spec.heads * spec.head_dim, w.rank);
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
    let base = |index: usize, x: Tensor<bf16, 2>, y: Tensor<bf16, 2>| -> Result<()> {
        blas.linear_typed(
            crate::op_contract::ExecutionPhase::Prefill,
            crate::backend::cuda::linear::LinearWeightRef::Bf16(w.linears[index]),
            x,
            &mut TensorMut(y),
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
            rows_view(&work.normed, s, 0, rows, h, h)?,
            rows_view(out, s, 0, rows, width, width)?,
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
        rows_view(&work.v_raw, s, 0, rows, width, width)?,
        rows_view(&work.normed, s, 0, rows, h, h)?,
    )?;
    lora_pair(
        c,
        rows_view(&work.v_raw, s, 0, rows, width, width)?,
        (&w.out_a, &w.out_b),
        &mut work.lora,
        rows_view(&work.normed, s, 0, rows, h, h)?,
        false,
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
    ffn_forward::<false>(c, w, work)?;
    if modulation.is_some() {
        gated_add(xp, normed, 5)?;
    } else {
        residual(xp, normed)?;
    }
    Ok(())
}

/// FFN rows overwrite their input after the gated projection; scratch is reused between chunks.
fn ffn_forward<const SHIFT_INPUT: bool>(
    c: &BlockContext<'_>,
    w: &BlockWeights,
    work: &mut BlockWork,
) -> Result<()> {
    let (s, h, f, rows) = (c.s, c.spec.hidden, c.spec.ffn, work.rows);
    let base = |index: usize, x: Tensor<bf16, 2>, y: Tensor<bf16, 2>| -> Result<()> {
        c.blas.linear_typed(
            crate::op_contract::ExecutionPhase::Prefill,
            crate::backend::cuda::linear::LinearWeightRef::Bf16(w.linears[index]),
            x,
            &mut TensorMut(y),
        )
    };
    let chunk = work.chunk;
    for start in (0..rows).step_by(chunk) {
        let n = chunk.min(rows - start);
        let input_start = start;
        #[cfg(test)]
        let input_start = if work.repeat_ffn_input {
            0
        } else {
            input_start
        };
        base(
            FF_IN,
            rows_view(&work.normed, s, input_start * h, n, h, h)?,
            work.hidden.prefix_rows(u32::try_from(n)?)?,
        )?;
        lora_pair(
            c,
            rows_view(&work.normed, s, input_start * h, n, h, h)?,
            (&w.in_a, &w.in_b),
            &mut work.lora,
            work.hidden.prefix_rows(u32::try_from(n)?)?,
            work.fixed_lora,
        )?;
        let hidden = work.hidden.ptr + if SHIFT_INPUT { 256 } else { 0 };
        let gated = work.gated.ptr;
        let cfg = ops::silu_mul_grid(n, f, hidden, gated)?;
        unsafe {
            s.launch_builder(&c.kernels.swiglu)
                .arg(&hidden)
                .arg(&gated)
                .arg(&i32::try_from(n)?)
                .arg(&i32::try_from(f)?)
                .arg(&3i32)
                .launch(cfg)?;
        }
        base(
            FF_OUT,
            work.gated.prefix_rows(u32::try_from(n)?)?,
            rows_view(&work.normed, s, start * h, n, h, h)?,
        )?;
        lora_pair(
            c,
            work.gated.prefix_rows(u32::try_from(n)?)?,
            (&w.ffo_a, &w.ffo_b),
            &mut work.lora,
            rows_view(&work.normed, s, start * h, n, h, h)?,
            work.fixed_lora,
        )?;
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

/// Transformer-block placement: `Derived` from the sensed VRAM and host budgets; the gates also force `Streamed` (no
/// block resident: two device slots, host tier from the sensed host budget, rest disk) and `StreamedDisk` (every block
/// from disk).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Placement {
    Derived,
    #[cfg(test)]
    Streamed,
    #[cfg(test)]
    StreamedDisk,
}

#[cfg(test)]
impl Placement {
    /// `FF_H3_PLACEMENT` = `derived`, `streamed` or `streamed-disk`.
    pub(crate) fn from_env() -> Result<Self> {
        match std::env::var("FF_H3_PLACEMENT").as_deref() {
            Ok("derived") => Ok(Self::Derived),
            Ok("streamed") => Ok(Self::Streamed),
            Ok("streamed-disk") => Ok(Self::StreamedDisk),
            other => anyhow::bail!(
                "FF_H3_PLACEMENT must be derived, streamed or streamed-disk: {other:?}"
            ),
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

/// Per-layer device slot bytes, stored source bytes, the largest slot and the staging/transient bytes of a
/// manifest's dense linears.
struct DenseSizing {
    layers: Vec<usize>,
    source_layers: Vec<usize>,
    largest: usize,
    staging: usize,
    transient: usize,
}

/// Sizing of a manifest's dense linears: a dequantized entry's device slot holds its BF16 tile, a pass-through
/// entry's holds its stored bytes, and a stored type with no device path is an error.
fn dense_sizing(manifest: &TensorManifest, layers: usize) -> Result<DenseSizing> {
    let mut device_layers = vec![0usize; layers];
    let mut source_layers = vec![0usize; layers];
    let mut dequant_layer = vec![false; layers];
    let mut largest = 0usize;
    for (name, entry) in &manifest.entries {
        if let Some(layer) = entry.layer
            && entry.load
            && entry.shape.len() == 2
        {
            let device_bytes = device_slot_bytes(name, entry)?;
            device_layers[layer as usize] += device_bytes;
            source_layers[layer as usize] += entry.bytes;
            largest = largest.max(device_bytes);
            if dequantizes(entry.quant) {
                dequant_layer[layer as usize] = true;
            }
        }
    }
    let staging = dequant_layer
        .iter()
        .zip(&source_layers)
        .filter_map(|(dequant, source)| dequant.then_some(*source))
        .max()
        .unwrap_or(0);
    let transient = manifest
        .entries
        .iter()
        .filter(|(_, entry)| entry.load && dequantizes(entry.quant))
        .map(|(_, entry)| entry.bytes)
        .max()
        .unwrap_or(0);
    Ok(DenseSizing {
        layers: device_layers,
        source_layers,
        largest,
        staging,
        transient,
    })
}

/// Dense streaming of `layers` (per-layer device matrix bytes, `largest` device matrix) beside `owned` resident bytes:
/// placement derived from free device memory and host RAM, or forced streamed / disk-only. `source_layers` sizes the
/// host/disk tier; it equals `layers` for a native BF16 manifest and is the stored source size for a quantized one.
/// Host-tier copies are filled from `base` in `roles` order before the first `begin`.
fn place_layers(
    d: &Device,
    base: &mut Source<'_>,
    (layers, source_layers, largest, owned, media): (&[usize], &[usize], usize, usize, usize),
    placement: Placement,
    name: impl Fn(usize) -> String,
    roles: &[&str],
) -> Result<Option<crate::backend::cuda::weights::DenseStreaming>> {
    use crate::prefill::DensePlacement;
    let (free, total) = d.ctx.mem_get_info()?;
    let mut budget = DensePlacement::device_budget(free, total, owned, 0, media)?;
    if placement != Placement::Derived {
        let slots = 2 * layers.iter().copied().max().context("no streamed layers")?;
        ensure!(slots <= budget, "two block slots exceed the device budget");
        budget = slots;
    }
    let mut plan = DensePlacement::new(layers, Some(source_layers), budget, 0)?;
    let mut host = None;
    if plan.slot_bytes > 0 {
        let available = usize::try_from(crate::backend::cuda::calibrate::host_memory(Path::new(
            "/",
        ))?)?;
        let reserved = 2 * plan.slot_bytes + largest + (4 << 30);
        ensure!(
            available >= reserved,
            "dense pinned staging and host reserve exceed available RAM: need={reserved} available={available}"
        );
        let host_bytes = match placement {
            Placement::Derived => available - reserved,
            #[cfg(test)]
            Placement::Streamed => available - reserved,
            #[cfg(test)]
            Placement::StreamedDisk => 0,
        };
        plan = DensePlacement::new(layers, Some(source_layers), budget, host_bytes)?;
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
    Ok(if plan.slot_bytes > 0 {
        let dir = base.reader.dir.clone();
        let host_tier = plan.host.clone();
        let mut dense =
            crate::backend::cuda::weights::DenseStreaming::new(&dir, base.manifest, plan, d, None)?;
        for layer in (0..layers.len()).filter(|&layer| host_tier[layer]) {
            let name = name(layer);
            let (layout, _) = matrix_layout(base.manifest, &name, roles)?;
            let host = dense
                .host_layer(layer)?
                .with_context(|| format!("{name}: host tier without a pinned copy"))?;
            ensure!(
                host.len() == source_layers[layer],
                "{name}: pinned copy size"
            );
            let mut source_offset = 0usize;
            for (role, (offset, _)) in &layout {
                let tensor = format!("{name}.{role}");
                let entry = &base.manifest.entries[&tensor];
                if entry.quant.is_some() {
                    base.reader.read_raw(
                        &tensor,
                        entry,
                        &mut host[source_offset..source_offset + entry.bytes],
                    )?;
                } else {
                    base.reader.read_into(
                        &tensor,
                        entry,
                        &mut host[2 * offset..2 * offset + entry.bytes],
                    )?;
                }
                source_offset += entry.bytes;
            }
        }
        Some(dense)
    } else {
        None
    })
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
        let mut work = BlockWork::new(s, spec, r, ranges.rows)?;
        work.prepare(blas, spec)?;
        let text_work = BlockWork::new(s, spec, r, ranges.text)?;
        text_work.prepare(blas, spec)?;
        let flash = Flash::new(
            s.clone(),
            ranges.rows,
            spec.heads,
            spec.heads,
            spec.head_dim,
        )?
        .with_full_single_probability()?;
        let text_flash = Flash::new(
            s.clone(),
            ranges.text,
            spec.heads,
            spec.heads,
            spec.head_dim,
        )?
        .with_full_single_probability()?;
        let generated = ranges.rows - ranges.video_start;
        let scratch_rows = (ranges.condition + generated).max(ranges.audio);
        let x = s.alloc_zeros::<bf16>(ranges.rows * h)?;
        let text = s.alloc_zeros::<bf16>(ranges.text * h)?;
        let scratch = s.alloc_zeros::<f32>(scratch_rows * h)?;
        let video = s.alloc_zeros::<f32>(generated * patch)?;
        let audio = s.alloc_zeros::<f32>(ranges.audio * spec.audio_in_channels)?;
        let rotary = s.clone_htod(&crate::minimax_h3::rotary_table(spec, layout))?;

        let sizing = dense_sizing(base.manifest, spec.layers)?;
        let owned = lora
            .manifest
            .entries
            .iter()
            .filter(|(name, _)| name.starts_with("transformer_blocks."))
            .map(|(_, entry)| entry.bytes)
            .sum::<usize>()
            + spec.layers * 2 * (2 * h + 2 * spec.head_dim)
            + sizing.staging
            + sizing.transient;
        let dense = place_layers(
            d,
            &mut base,
            (
                &sizing.layers,
                &sizing.source_layers,
                sizing.largest,
                owned,
                0,
            ),
            placement,
            |layer| format!("transformer_blocks.{layer}"),
            &LINEAR_ROLES,
        )?;
        let mut blocks = Vec::with_capacity(spec.layers);
        for (layer, &bytes) in sizing.layers.iter().enumerate() {
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
        let (free, total) = d.ctx.mem_get_info()?;
        let reserve = (total / 16)
            .checked_add(sizing.transient)
            .context("FFN live reserve overflow")?;
        for (output, input) in [(r, h), (r, spec.ffn), (2 * spec.ffn, r), (h, r)] {
            blas.bind_row_family(crate::backend::setup::LinearShape {
                rows: FFN_CHUNK_ROWS,
                output,
                input,
            })?;
        }
        let budget = d.workspace_budget(free, free, reserve)?;
        let choices = work.ffn_choices(blas, spec, budget)?;
        let owned = FfnPlan::select(&choices, budget)?;
        let aliases = work.alias_choices(blas, spec, owned)?;
        let chosen = FfnPlan::select(&aliases, budget)?;
        let previous = *choices
            .iter()
            .find(|p| p.chunk == work.chunk)
            .context("main FFN baseline missing")?;
        if work.rows > FFN_CHUNK_ROWS {
            work.ffn_cost(blas, spec, work.chunk, true)?;
        }
        let description = |plan: FfnPlan| {
            serde_json::json!({
            "rows":work.rows,"previous_chunk":work.chunk,"chunk":plan.chunk,
            "storage":if plan.borrowed {"attention_scratch"} else {"owned"},
            "chunks":work.rows.div_ceil(plan.chunk),"tail":work.rows%plan.chunk,
            "new_allocation_bytes":plan.allocation,"estimated_tensor_cost":plan.tensor_cost,
            "estimated_launches_per_block":plan.launches,"previous_tensor_cost":previous.tensor_cost,
            "previous_launches_per_block":previous.launches})
        };
        eprintln!(
            "{}",
            serde_json::json!({"h3_ffn_plan":{
            "workspace_budget":budget,"free_bytes":free,"live_reserve_bytes":reserve,
            "peak_additional_bytes":chosen.allocation+chosen.workspace,
            "lt_workspace_replacement_bytes":chosen.workspace,"main":description(chosen),
            "owned_alternative":description(owned),
            "alias_capacity_rows":FfnPlan::alias_capacity((work.q_raw.len(),work.k_raw.len()),spec.ffn)?,
            "refiner":{"rows":text_work.rows,"previous_chunk":text_work.chunk,"chunk":text_work.chunk,
                "chunks":text_work.rows.div_ceil(text_work.chunk),"tail":text_work.rows%text_work.chunk,
                "new_allocation_bytes":0,"ordinary_plans":true}}})
        );
        blas.bind_workspace(chosen.workspace)?;
        work.bind_ffn(s, blas, spec, chosen, &x)?;
        s.synchronize()?;
        eprintln!(
            "{}",
            serde_json::json!({"h3_ffn_storage":{
            "hidden_ptr":work.hidden.ptr,"hidden_shape":work.hidden.shape,"hidden_span_bytes":work.hidden.span_bytes,
            "gated_ptr":work.gated.ptr,"gated_shape":work.gated.shape,"gated_span_bytes":work.gated.span_bytes,
            "q_raw_ptr":pointer(&work.q_raw,s),"q_raw_bytes":2*work.q_raw.len(),
            "k_raw_ptr":pointer(&work.k_raw,s),"k_raw_bytes":2*work.k_raw.len(),
            "owned":work.ffn_owned.is_some(),"free_after_binding":d.ctx.mem_get_info()?.0}})
        );
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

/// Roles of a Qwen3-VL decoder layer's matrices in manifest name order; `mlp.gate_proj` and `mlp.up_proj` are
/// adjacent, so one `[2 * ffn, hidden]` view projects both.
const TEXT_ROLES: [&str; 7] = [
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
];

/// One Qwen3-VL decoder layer: matrices as views into `storage` or a dense streaming slot, norms owned.
pub(crate) struct TextLayer {
    input_norm: CudaSlice<bf16>,
    post_norm: CudaSlice<bf16>,
    q_norm: CudaSlice<bf16>,
    k_norm: CudaSlice<bf16>,
    q: Tensor<bf16, 2>,
    k: Tensor<bf16, 2>,
    v: Tensor<bf16, 2>,
    o: Tensor<bf16, 2>,
    gate_up: Tensor<bf16, 2>,
    down: Tensor<bf16, 2>,
    storage: Option<CudaSlice<bf16>>,
}

impl TextLayer {
    fn new(
        s: &Arc<CudaStream>,
        block: &str,
        (hidden, head_dim): (usize, usize),
        slot: Option<u64>,
        base: &mut Source<'_>,
    ) -> Result<Self> {
        let (layout, elements) = matrix_layout(base.manifest, block, &TEXT_ROLES)?;
        let mut storage = None;
        let at = match slot {
            Some(slot) => slot,
            None => {
                let mut owned = s.alloc_zeros::<bf16>(elements)?;
                for (role, (offset, shape)) in &layout {
                    let count = shape[0] as usize * shape[1] as usize;
                    base.fill(
                        s,
                        &format!("{block}.{role}"),
                        &mut owned.slice_mut(*offset..offset + count),
                    )?;
                }
                let at = pointer(&owned, s);
                storage = Some(owned);
                at
            }
        };
        let view = |offset: usize, shape: [u32; 2]| unsafe {
            Tensor::new(
                at + 2 * offset as u64,
                shape,
                [u64::from(shape[1]), 1],
                2 * u64::from(shape[0]) * u64::from(shape[1]),
            )
        };
        let role = |name: &str| -> Result<Tensor<bf16, 2>> {
            let (offset, shape) = layout[name];
            view(offset, shape)
        };
        let (gate, gate_shape) = layout["mlp.gate_proj.weight"];
        let (up, up_shape) = layout["mlp.up_proj.weight"];
        ensure!(
            gate_shape == up_shape && up == gate + gate_shape[0] as usize * gate_shape[1] as usize,
            "{block}: gate and up projections are not adjacent"
        );
        let norm = |base: &mut Source<'_>, name: &str, n: usize| {
            base.tensor::<bf16>(s, &format!("{block}.{name}.weight"), n)
        };
        let mut layer = Self {
            input_norm: norm(base, "input_layernorm", hidden)?,
            post_norm: norm(base, "post_attention_layernorm", hidden)?,
            q_norm: norm(base, "self_attn.q_norm", head_dim)?,
            k_norm: norm(base, "self_attn.k_norm", head_dim)?,
            q: role("self_attn.q_proj.weight")?,
            k: role("self_attn.k_proj.weight")?,
            v: role("self_attn.v_proj.weight")?,
            o: role("self_attn.o_proj.weight")?,
            gate_up: view(gate, [2 * gate_shape[0], gate_shape[1]])?,
            down: role("mlp.down_proj.weight")?,
            storage: None,
        };
        layer.storage = storage;
        Ok(layer)
    }
}

/// Conditioner input over `rows` prompt rows: the BF16 input embeddings `[rows, hidden]` with the image rows already
/// spliced, the deepstack features added to the image rows after decoder layers `0..deepstack.len()`
/// (`[image rows, hidden]` each, image runs in order), the image runs as `(first row, rows)`, and the BF16 mRoPE
/// table `[rows, head_dim]` (cosines, then sines).
pub(crate) struct ConditionerInput<'a> {
    pub embeds: &'a [bf16],
    pub deepstack: &'a [CudaSlice<bf16>],
    pub runs: &'a [(usize, usize)],
    pub rope: &'a [bf16],
}

/// The Qwen3-VL conditioner text decoder: layers `0..layers.len()` of the published checkpoint, placed through
/// dense streaming; the output is the residual after the last layer (`hidden_states[layers]`). Activations are
/// allocated before the placement is derived; `media` bytes stay unplaced for the vision request.
pub(crate) struct Conditioner {
    hidden: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    ffn: usize,
    eps: f32,
    capacity: usize,
    layers: Vec<TextLayer>,
    dense: Option<crate::backend::cuda::weights::DenseStreaming>,
    kernels: BlockKernels,
    flash: Flash,
    x: CudaSlice<bf16>,
    normed: CudaSlice<bf16>,
    q: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    keys: CudaSlice<bf16>,
    values: CudaSlice<bf16>,
    attention: CudaSlice<bf16>,
    gate_up: CudaSlice<bf16>,
    gated: CudaSlice<bf16>,
    rope: CudaSlice<bf16>,
}

impl Conditioner {
    pub(crate) fn new(
        d: &Device,
        spec: &crate::model_spec::ModelSpec,
        (layers, capacity, media): (usize, usize, usize),
        mut base: Source<'_>,
        placement: Placement,
    ) -> Result<Self> {
        ensure!(
            spec.architecture == crate::model_spec::Architecture::Qwen3Vl
                && layers <= spec.layers.len(),
            "conditioner needs Qwen3-VL with at least {layers} decoder layers"
        );
        let s = &d.stream;
        let (hidden, head_dim, ffn) = (
            spec.hidden as usize,
            spec.head_dim as usize,
            spec.intermediate as usize,
        );
        let (q_heads, kv_heads) = (spec.q_heads as usize, spec.kv_heads as usize);
        let name = |layer: usize| format!("model.language_model.layers.{layer}");
        let mut bytes = vec![0usize; layers];
        let mut largest = 0usize;
        for entry in base.manifest.entries.values() {
            if let Some(layer) = entry.layer
                && entry.load
                && entry.shape.len() == 2
            {
                let layer = layer as usize;
                ensure!(layer < layers, "manifest loads decoder layer {layer}");
                bytes[layer] += entry.bytes;
                largest = largest.max(entry.bytes);
            }
        }
        let rows = |width: usize| s.alloc_zeros::<bf16>(capacity * width);
        let (x, normed, rope) = (rows(hidden)?, rows(hidden)?, rows(head_dim)?);
        let (q, attention) = (rows(q_heads * head_dim)?, rows(q_heads * head_dim)?);
        let (k, v) = (rows(kv_heads * head_dim)?, rows(kv_heads * head_dim)?);
        let (keys, values) = (rows(kv_heads * head_dim)?, rows(kv_heads * head_dim)?);
        let (gate_up, gated) = (rows(2 * ffn)?, rows(ffn)?);
        let flash = Flash::new(s.clone(), capacity, q_heads, kv_heads, head_dim)?;
        let owned = layers * 2 * (2 * hidden + 2 * head_dim);
        let dense = place_layers(
            d,
            &mut base,
            (&bytes, &bytes, largest, owned, media),
            placement,
            name,
            &TEXT_ROLES,
        )?;
        let mut text_layers = Vec::with_capacity(layers);
        for (layer, &layer_bytes) in bytes.iter().enumerate() {
            let slot = match &dense {
                Some(dense) if dense.plan.slots[layer].is_some() => {
                    Some(dense.pointer(layer, 0, layer_bytes)?)
                }
                _ => None,
            };
            text_layers.push(TextLayer::new(
                s,
                &name(layer),
                (hidden, head_dim),
                slot,
                &mut base,
            )?);
        }
        Ok(Self {
            hidden,
            q_heads,
            kv_heads,
            head_dim,
            ffn,
            eps: spec.rms_eps,
            capacity,
            layers: text_layers,
            dense,
            kernels: BlockKernels::new(s.context())?,
            flash,
            x,
            normed,
            q,
            k,
            v,
            keys,
            values,
            attention,
            gate_up,
            gated,
            rope,
        })
    }

    /// Binds the fixed GEMM plans of every decoder-layer shape at `rows`.
    pub(crate) fn prepare(&self, blas: &Blas, rows: usize) -> Result<()> {
        let (h, q, kv, f) = (
            self.hidden,
            self.q_heads * self.head_dim,
            self.kv_heads * self.head_dim,
            self.ffn,
        );
        for (output, input) in [(q, h), (kv, h), (h, q), (2 * f, h), (h, f)] {
            blas.prepare_fixed(
                u32::try_from(rows)?,
                u32::try_from(output)?,
                u32::try_from(input)?,
            )?;
        }
        Ok(())
    }

    /// Runs every decoder layer on `input` and returns the residual after layer `tap - 1` for each `tap`, in order.
    pub(crate) fn forward(
        &mut self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        input: &ConditionerInput<'_>,
        taps: &[usize],
    ) -> Result<Vec<Vec<bf16>>> {
        let (h, d) = (self.hidden, self.head_dim);
        let (qw, kvw) = (self.q_heads * d, self.kv_heads * d);
        ensure!(
            !input.embeds.is_empty() && input.embeds.len().is_multiple_of(h),
            "embedding extent"
        );
        let rows = input.embeds.len() / h;
        ensure!(
            rows <= self.capacity,
            "{rows} rows exceed capacity {}",
            self.capacity
        );
        ensure!(input.rope.len() == rows * d, "mRoPE table extent");
        let image_rows: usize = input.runs.iter().map(|&(_, n)| n).sum();
        ensure!(
            input.runs.iter().all(|&(first, n)| first + n <= rows)
                && input.deepstack.len() <= self.layers.len()
                && input.deepstack.iter().all(|f| f.len() == image_rows * h),
            "image runs or deepstack features differ from the prompt"
        );
        ensure!(
            taps.iter()
                .all(|&tap| (1..=self.layers.len()).contains(&tap)),
            "taps outside 1..={}",
            self.layers.len()
        );
        s.memcpy_htod(input.embeds, &mut self.x.slice_mut(..rows * h))?;
        s.memcpy_htod(input.rope, &mut self.rope.slice_mut(..rows * d))?;
        let rows_i = i32::try_from(rows)?;
        let linear = |weight: Tensor<bf16, 2>,
                      x: &CudaSlice<bf16>,
                      k: usize,
                      y: &CudaSlice<bf16>,
                      n: usize|
         -> Result<()> {
            blas.linear_typed(
                crate::op_contract::ExecutionPhase::Prefill,
                crate::backend::cuda::linear::LinearWeightRef::Bf16(weight),
                rows_view(x, s, 0, rows, k, k)?,
                &mut TensorMut(rows_view(y, s, 0, rows, n, n)?),
            )
        };
        let kernels = &self.kernels;
        let norm = |input: &CudaSlice<bf16>,
                    weight: &CudaSlice<bf16>,
                    output: &CudaSlice<bf16>,
                    (rows, dim, threads): (usize, usize, u32)|
         -> Result<()> {
            kernels.rms_norm(
                s,
                NormRows {
                    input: pointer(input, s),
                    weight: pointer(weight, s),
                    output: pointer(output, s),
                    rows,
                    dim,
                    threads,
                    eps: self.eps,
                },
            )
        };
        let residual = |residual: u64, update: u64, count: usize| -> Result<()> {
            unsafe {
                s.launch_builder(&kernels.residual)
                    .arg(&residual)
                    .arg(&update)
                    .arg(&i32::try_from(count)?)
                    .launch(flat(count / 8))?;
            }
            Ok(())
        };
        let mut tapped = Vec::with_capacity(taps.len());
        let x = pointer(&self.x, s);
        let normed = pointer(&self.normed, s);
        for (index, layer) in self.layers.iter().enumerate() {
            if let Some(dense) = self.dense.as_mut() {
                dense.begin(index, s)?;
            }
            norm(
                &self.x,
                &layer.input_norm,
                &self.normed,
                (rows, h, HIDDEN_NORM_THREADS),
            )?;
            linear(layer.q, &self.normed, h, &self.attention, qw)?;
            linear(layer.k, &self.normed, h, &self.keys, kvw)?;
            linear(layer.v, &self.normed, h, &self.v, kvw)?;
            norm(
                &self.attention,
                &layer.q_norm,
                &self.q,
                (rows * self.q_heads, d, HEAD_NORM_THREADS),
            )?;
            norm(
                &self.keys,
                &layer.k_norm,
                &self.k,
                (rows * self.kv_heads, d, HEAD_NORM_THREADS),
            )?;
            // q rotates in place; k and v land head-major [kv_heads, rows, head_dim] in keys and values.
            unsafe {
                s.launch_builder(&kernels.rotary)
                    .arg(&pointer(&self.q, s))
                    .arg(&pointer(&self.k, s))
                    .arg(&pointer(&self.v, s))
                    .arg(&pointer(&self.keys, s))
                    .arg(&pointer(&self.values, s))
                    .arg(&pointer(&self.rope, s))
                    .arg(&0i32)
                    .arg(&rows_i)
                    .arg(&i32::try_from(self.q_heads)?)
                    .arg(&i32::try_from(self.kv_heads)?)
                    .arg(&i32::try_from(d)?)
                    .arg(&i32::try_from(d)?)
                    .arg(&rows_i)
                    .launch(flat(rows * (self.q_heads + self.kv_heads) * d))?;
            }
            self.flash.run(
                &self.q,
                qw,
                &self.keys,
                &self.values,
                &mut self.attention,
                Shape {
                    rows,
                    start: 0,
                    q_heads: self.q_heads,
                    kv_heads: self.kv_heads,
                    capacity: rows,
                    head_dim: d,
                    causal: true,
                    window_left: None,
                },
            )?;
            linear(layer.o, &self.attention, qw, &self.normed, h)?;
            residual(x, normed, rows * h)?;
            norm(
                &self.x,
                &layer.post_norm,
                &self.normed,
                (rows, h, HIDDEN_NORM_THREADS),
            )?;
            linear(layer.gate_up, &self.normed, h, &self.gate_up, 2 * self.ffn)?;
            let (gate_up, gated) = (pointer(&self.gate_up, s), pointer(&self.gated, s));
            let cfg = ops::silu_mul_grid(rows, self.ffn, gate_up, gated)?;
            unsafe {
                s.launch_builder(&kernels.swiglu)
                    .arg(&gate_up)
                    .arg(&gated)
                    .arg(&rows_i)
                    .arg(&i32::try_from(self.ffn)?)
                    .arg(&0i32)
                    .launch(cfg)?;
            }
            linear(layer.down, &self.gated, self.ffn, &self.normed, h)?;
            residual(x, normed, rows * h)?;
            if let Some(features) = input.deepstack.get(index) {
                let mut feature = 0usize;
                for &(first, n) in input.runs {
                    residual(
                        x + (2 * first * h) as u64,
                        pointer(features, s) + (2 * feature * h) as u64,
                        n * h,
                    )?;
                    feature += n;
                }
            }
            if let Some(dense) = self.dense.as_mut() {
                dense.end(index, s)?;
            }
            if taps.contains(&(index + 1)) {
                tapped.push((index + 1, s.clone_dtoh(&self.x.slice(..rows * h))?));
            }
        }
        Ok(taps
            .iter()
            .map(|tap| {
                tapped
                    .iter()
                    .find(|(t, _)| t == tap)
                    .map(|(_, v)| v.clone())
                    .unwrap_or_default()
            })
            .collect())
    }
}

/// `[ids.len(), hidden]` BF16 rows of the published embedding table `embed`, read row by row.
pub(crate) fn embedding_rows(
    base: &mut Source<'_>,
    embed: &str,
    hidden: usize,
    ids: &[u32],
) -> Result<Vec<bf16>> {
    let entry = base
        .manifest
        .entries
        .get(embed)
        .with_context(|| format!("manifest lacks {embed}"))?;
    ensure!(
        entry.dtype == Dtype::BF16 && entry.shape.len() == 2 && entry.shape[1] == hidden,
        "{embed}: {:?} {:?}",
        entry.dtype,
        entry.shape
    );
    let row = 2 * hidden;
    let mut out = Vec::with_capacity(ids.len() * hidden);
    for &id in ids {
        ensure!((id as usize) < entry.shape[0], "token {id} outside {embed}");
        let bytes = base
            .reader
            .read_range(embed, entry, id as usize * row, row)?;
        ensure!(
            <bf16 as Element>::finite(bytes),
            "{embed}: nonfinite row {id}"
        );
        out.extend(
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_le_bytes(*b)),
        );
    }
    Ok(out)
}

/// Fills `out` with `count` standard normals from one documented `(seed, stream)` Philox counter range.
/// `block` selects the launch geometry; the counter-based output is identical for every geometry.
pub(crate) fn draw_normals(
    s: &Arc<CudaStream>,
    normal: &CudaFunction,
    (seed, stream): (u64, u64),
    count: usize,
    block: u32,
) -> Result<CudaSlice<f32>> {
    // The kernel writes every one of the `count` outputs.
    let out = unsafe { s.alloc::<f32>(count) }?;
    let threads = count.div_ceil(4);
    unsafe {
        s.launch_builder(normal)
            .arg(&seed)
            .arg(&stream)
            .arg(&out)
            .arg(&(count as i64))
            .launch(ops::grid(threads.div_ceil(block as usize), block))?;
    }
    Ok(out)
}

/// Runs one MiniMax-H3 i2va request on GPU `device` with the transformer LoRA at `lora` (cuBLAS calibration under
/// `runtime`) and returns its u8 frames and stereo PCM. The four noise draws are generated natively on device from
/// `request.seed`; there is no recorded-draw path. `gguf` replaces the transformer safetensors with one
/// community GGUF file.
pub fn generate(
    model: &Path,
    lora: &Path,
    request: &crate::minimax_h3::H3Request,
    (device, runtime): (usize, &Path),
    gguf: Option<&Path>,
) -> Result<crate::minimax_h3::H3Output> {
    Ok(pipeline::run(model, lora, request, (device, runtime), gguf)?.output)
}

/// One Qwen3-VL conditioner prompt: prompt tokens with their image grids and packed `pixel_values`.
pub(crate) struct ConditionerPrompt {
    pub tokens: Vec<u32>,
    pub modality: Vec<u8>,
    pub grids: Vec<crate::vision_input::ImageGrid>,
    pub pixels: Vec<f32>,
}

/// Image-token runs of a prompt as `(first row, rows)`.
pub(crate) fn image_runs(tokens: &[u32], image_token: u32) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (row, &id) in tokens.iter().enumerate() {
        if id != image_token {
            continue;
        }
        match runs.last_mut() {
            Some((first, n)) if *first + *n == row => *n += 1,
            _ => runs.push((row, 1)),
        }
    }
    runs
}

/// BF16 mRoPE table `[rows, head_dim]` of a conditioner cell from its validated position plan.
pub(crate) fn conditioner_rope(
    spec: &crate::model_spec::ModelSpec,
    cell: &ConditionerPrompt,
) -> Result<Vec<bf16>> {
    let vision = spec.vision.as_ref().context("vision missing")?;
    let section = vision.mrope_section.context("mRoPE section missing")?;
    let plan = crate::vision_input::MultimodalInput {
        input_ids: &cell.tokens,
        modality_ids: &cell.modality,
        pixel_values: &cell.pixels,
        grids: &cell.grids,
        audio: &[],
    }
    .validate(spec, cell.tokens.len())?;
    Ok(
        crate::minimax_h3::mrope_table(&plan, section, spec.rope_theta, spec.head_dim)?
            .into_iter()
            .map(bf16::from_f32)
            .collect(),
    )
}

/// Conditioner input of one cell: table rows for text tokens, `merged` for the image rows in order.
pub(crate) fn conditioner_embeds(
    cell: &ConditionerPrompt,
    image_token: u32,
    table: &BTreeMap<u32, Vec<bf16>>,
    merged: &[bf16],
    hidden: usize,
) -> Result<Vec<bf16>> {
    let mut embeds = Vec::with_capacity(cell.tokens.len() * hidden);
    let mut feature = 0usize;
    for &id in &cell.tokens {
        if id == image_token {
            embeds.extend_from_slice(
                merged
                    .get(feature * hidden..(feature + 1) * hidden)
                    .context("fewer image features than image tokens")?,
            );
            feature += 1;
        } else {
            embeds.extend_from_slice(table.get(&id).context("embedding row missing")?);
        }
    }
    ensure!(
        feature * hidden == merged.len(),
        "more image features than image tokens"
    );
    Ok(embeds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cuda::{closure::Closure, ops::Ops};
    use crate::checkpoint::MappedCheckpoint;
    use crate::minimax_h3::{
        Geometry, KeyframeAnchor, PackedLayout, RowTimesteps, Schedule, noise_draws,
        row_timestep_plan,
    };
    use anyhow::bail;
    use safetensors::SafeTensors;
    use std::{fs, rc::Rc};

    /// Initial chunks avoid skinny tails; selection counts scratch and workspace replacements.
    #[test]
    fn ffn_chunks_leave_no_skinny_tail() -> Result<()> {
        for rows in [60_431, 40_045, 1_025, 2_049, 3_073, 1_024, 300] {
            let chunk = ffn_chunk(rows);
            let tail = rows % chunk;
            assert!(chunk <= FFN_CHUNK_ROWS, "rows={rows} chunk={chunk}");
            assert!(
                tail == 0 || tail > crate::linear_plan::MAX_SKINNY_ROWS,
                "rows={rows} chunk={chunk} tail={tail}"
            );
        }
        let baseline = FfnPlan {
            chunk: 1_002,
            borrowed: false,
            allocation: 0,
            workspace: 0,
            tensor_cost: 80.0,
            launches: 280,
        };
        let efficient = FfnPlan {
            chunk: 6_675,
            borrowed: false,
            allocation: FfnPlan::allocation(6_675, 14_336)?,
            workspace: 0,
            tensor_cost: 54.0,
            launches: 42,
        };
        let larger = FfnPlan {
            chunk: 8_009,
            borrowed: false,
            allocation: FfnPlan::allocation(8_009, 14_336)?,
            workspace: 0,
            tensor_cost: 55.0,
            launches: 35,
        };
        let choices = [baseline, efficient, larger];
        let full = efficient.allocation;
        assert_eq!(FfnPlan::select(&choices, full)?.chunk, 6_675);
        assert_eq!(FfnPlan::select(&choices, full - 1)?.chunk, 1_002);
        assert_eq!(FfnPlan::select(&choices, 0)?.chunk, 1_002);
        let with_workspace = FfnPlan {
            workspace: 4096,
            ..efficient
        };
        assert!(FfnPlan::select(&[with_workspace], full + 4095).is_err());
        assert_eq!(
            FfnPlan::select(&[with_workspace], full + 4096)?.chunk,
            6_675
        );
        assert!(FfnPlan::allocation(usize::MAX, 2).is_err());
        for (rows, expected) in [(40_045, 10_011), (60_431, 15_107)] {
            let elements = rows * 7_168;
            let capacity = FfnPlan::alias_capacity((elements, elements), 14_336)?;
            assert_eq!(capacity, expected);
            assert!(capacity * 28_672 <= elements);
            assert!((capacity + 1) * 28_672 > elements);
            assert_eq!(FfnPlan::alias_capacity((elements, 14_335), 14_336)?, 0);
        }
        assert!(FfnPlan::alias_capacity((usize::MAX, usize::MAX), usize::MAX).is_err());
        assert!(FfnPlan::alias_capacity((1, 1), 0).is_err());
        assert!(
            FfnPlan::select(
                &[FfnPlan {
                    allocation: usize::MAX,
                    workspace: 1,
                    ..baseline
                }],
                usize::MAX
            )
            .is_err()
        );
        Ok(())
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
            let mut flash = Flash::new(s.clone(), rows, spec.heads, spec.heads, spec.head_dim)?
                .with_full_single_probability()?;
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
        let table = AdalnTable::build(&device, &mut blas, &transformer, &manifest, &spec, &adaln)?;
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

    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted, FF_MODEL, FF_H3_LORA, FF_H3_CAPTURE"]
    fn ffn_lora_is_row_count_invariant() -> Result<()> {
        use crate::backend::setup::LinearShape;
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let path = |key| {
            std::env::var_os(key)
                .map(PathBuf::from)
                .with_context(|| format!("{key} missing"))
        };
        let spec = H3Spec::read(&path("FF_MODEL")?)?;
        let file = path("FF_H3_LORA")?;
        let dir = file.parent().context("LoRA directory")?;
        let stem = file
            .file_stem()
            .and_then(|s| s.to_str())
            .context("LoRA stem")?;
        let (manifest, adapter) =
            MappedCheckpoint::open_component(dir, stem)?.minimax_h3_lora_manifest(&spec)?;
        let captured = fs::read(path("FF_H3_CAPTURE")?.join("lora_a_up_input.bf16"))?;
        ensure!(
            captured.len() == 1511 * spec.hidden * 2,
            "captured input must be [1511,hidden] BF16"
        );
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let stream = &device.stream;
        let runtime = std::env::temp_dir().join(format!("ff-h3-row-gate-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let partitions: Vec<usize> = serde_json::from_slice(&fs::read(
            path("FF_H3_CAPTURE")?.join("row_partitions.json"),
        )?)?;
        ensure!(
            partitions.contains(&1008)
                && partitions.contains(&1511)
                && partitions.iter().all(|&n| n > 0 && n <= 60_431),
            "row partitions must cover the discriminator within production extents"
        );
        let total = *partitions.iter().max().context("row partitions empty")?;
        let mut source = Source::new(dir, &manifest);
        {
            let rows = 664;
            let weights = source.tensor::<bf16>(
                stream,
                "token_refiner.refiner_blocks.0.ff.net.0.proj.lora_A.default.weight",
                adapter.rank * spec.hidden,
            )?;
            let bytes = fs::read(path("FF_H3_CAPTURE")?.join("refiner_a_up_input.bf16"))?;
            ensure!(
                bytes.len() == rows * spec.hidden * 2,
                "refiner common input must be [664,hidden] BF16"
            );
            let host: Vec<bf16> = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_le_bytes(*b))
                .collect();
            ensure!(
                host.iter().all(|v| v.is_finite()),
                "RUNNER_FAIL nonfinite refiner input"
            );
            let input = stream.clone_htod(&host)?;
            let output = stream.alloc_zeros::<bf16>(rows * adapter.rank)?;
            let preservation = Blas::new(
                &device,
                &ops,
                &runtime,
                Rc::new(Closure::new(true, "ordinary row preservation".into())),
            )?;
            preservation.prepare_fixed(rows as u32, adapter.rank as u32, spec.hidden as u32)?;
            let shape = LinearShape {
                rows,
                output: adapter.rank,
                input: spec.hidden,
            };
            let ordinary = || -> Result<Vec<bf16>> {
                preservation.linear_typed(
                    crate::op_contract::ExecutionPhase::Prefill,
                    crate::backend::cuda::linear::LinearWeightRef::Bf16(matrix(
                        &weights,
                        stream,
                        adapter.rank,
                        spec.hidden,
                    )?),
                    matrix(&input, stream, rows, spec.hidden)?,
                    &mut TensorMut(matrix(&output, stream, rows, adapter.rank)?),
                )?;
                Ok(stream.clone_dtoh(&output)?)
            };
            let expected = ordinary()?;
            let before = preservation.bound_plan_receipt()?;
            preservation.bind_row_family(LinearShape {
                rows: FFN_CHUNK_ROWS,
                ..shape
            })?;
            preservation.bind_workspace(preservation.fixed_workspace(shape, true)?)?;
            for executed_family in [false, true] {
                if executed_family {
                    preservation.linear_row_family(
                        matrix(&weights, stream, adapter.rank, spec.hidden)?,
                        matrix(&input, stream, rows, spec.hidden)?,
                        &mut TensorMut(matrix(&output, stream, rows, adapter.rank)?),
                        1.0,
                        0.0,
                    )?;
                }
                let actual = ordinary()?;
                ensure!(
                    actual.iter().all(|v| v.is_finite()),
                    "RUNNER_FAIL nonfinite ordinary output"
                );
                let differences = expected
                    .iter()
                    .zip(&actual)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                println!(
                    "H3_ORDINARY_PRESERVATION rows={rows} elements={} family_executed={executed_family} bit_differences={differences}",
                    expected.len()
                );
                ensure!(
                    differences == 0,
                    "DEFECT ordinary route changed after row-family binding"
                );
            }
            let after = preservation.bound_plan_receipt()?;
            let ordinary_after: Vec<_> = after["lt"]
                .as_array()
                .context("plan list missing")?
                .iter()
                .filter(|p| p["row_family"] == false)
                .cloned()
                .collect();
            ensure!(
                before["lt"] == serde_json::to_value(ordinary_after)?,
                "DEFECT ordinary plan identity changed"
            );
            println!(
                "H3_ORDINARY_PRESERVATION_GATE PASS before_after_binding=true after_family_execution=true bit_difference_limit=0"
            );
        }
        let mut cells = 0;
        for (name, output, input, accumulate) in [
            ("ff.net.0.proj.lora_A", adapter.rank, spec.hidden, false),
            ("ff.net.2.lora_A", adapter.rank, spec.ffn, false),
            ("ff.net.0.proj.lora_B", 2 * spec.ffn, adapter.rank, true),
            ("ff.net.2.lora_B", spec.hidden, adapter.rank, true),
        ] {
            let weight = source.tensor::<bf16>(
                stream,
                &format!("transformer_blocks.0.{name}.default.weight"),
                output * input,
            )?;
            let host: Vec<bf16> = if name == "ff.net.0.proj.lora_A" {
                captured
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| bf16::from_le_bytes(*b))
                    .cycle()
                    .take(total * input)
                    .collect()
            } else {
                (0..total * input)
                    .map(|i| {
                        bf16::from_f32((((i * 37 + i / input * 11) % 251) as f32 - 125.0) / 127.0)
                    })
                    .collect()
            };
            let x = stream.clone_htod(&host)?;
            let initial: Vec<bf16> = (0..total * output)
                .map(|i| bf16::from_f32(((i * 19 % 257) as f32 - 128.0) / 131.0))
                .collect();
            let make_blas = || {
                Blas::new(
                    &device,
                    &ops,
                    &runtime,
                    Rc::new(Closure::new(true, "FFN row invariance".into())),
                )
            };
            let fixed = make_blas()?;
            fixed.bind_row_family(LinearShape {
                rows: FFN_CHUNK_ROWS,
                output,
                input,
            })?;
            let heuristic = make_blas()?;
            let shape = |rows| LinearShape {
                rows,
                output,
                input,
            };
            let mut workspace = 0;
            for &chunk in &partitions {
                for rows in [chunk, total % chunk].into_iter().filter(|&r| r > 0) {
                    workspace = workspace.max(fixed.fixed_workspace(shape(rows), true)?);
                    heuristic.prepare_fixed(
                        u32::try_from(rows)?,
                        u32::try_from(output)?,
                        u32::try_from(input)?,
                    )?;
                }
            }
            let composition = if accumulate {
                let width = if output == 2 * spec.ffn {
                    spec.hidden
                } else {
                    spec.ffn
                };
                let key = name.replace("lora_B", "lora_A");
                let a = source.tensor::<bf16>(
                    stream,
                    &format!("transformer_blocks.0.{key}.default.weight"),
                    input * width,
                )?;
                let values: Vec<bf16> = (0..total * width)
                    .map(|i| {
                        bf16::from_f32((((i * 37 + i / width * 11) % 251) as f32 - 125.0) / 127.0)
                    })
                    .collect();
                let raw = stream.clone_htod(&values)?;
                fixed.bind_row_family(LinearShape {
                    rows: FFN_CHUNK_ROWS,
                    output: input,
                    input: width,
                })?;
                for &chunk in &partitions {
                    for rows in [chunk, total % chunk].into_iter().filter(|&r| r > 0) {
                        workspace = workspace.max(fixed.fixed_workspace(
                            LinearShape {
                                rows,
                                output: input,
                                input: width,
                            },
                            true,
                        )?);
                    }
                }
                Some((a, raw, width))
            } else {
                None
            };
            fixed.bind_workspace(workspace)?;
            let run = |blas: &Blas,
                       chunk: usize,
                       graph: bool,
                       fixed_b: bool,
                       composed: bool|
             -> Result<Vec<bf16>> {
                let y = stream.clone_htod(&initial)?;
                let intermediate = stream.alloc_zeros::<bf16>(total * input)?;
                if graph {
                    stream.begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                }
                let launch = (|| -> Result<()> {
                    for start in (0..total).step_by(chunk) {
                        let rows = chunk.min(total - start);
                        let w = matrix(&weight, stream, output, input)?;
                        let x = if composed {
                            let (a, raw, width) =
                                composition.as_ref().context("composed LoRA-A missing")?;
                            blas.linear_row_family(
                                matrix(a, stream, input, *width)?,
                                rows_view(raw, stream, start * width, rows, *width, *width)?,
                                &mut TensorMut(rows_view(
                                    &intermediate,
                                    stream,
                                    start * input,
                                    rows,
                                    input,
                                    input,
                                )?),
                                1.0,
                                0.0,
                            )?;
                            rows_view(&intermediate, stream, start * input, rows, input, input)?
                        } else {
                            rows_view(&x, stream, start * input, rows, input, input)?
                        };
                        let mut y =
                            TensorMut(rows_view(&y, stream, start * output, rows, output, output)?);
                        if accumulate {
                            if fixed_b {
                                blas.linear_row_family(w, x, &mut y, adapter.scale, 1.0)?;
                            } else {
                                blas.linear_accumulate(w, x, &mut y, adapter.scale)?;
                            }
                        } else if fixed_b {
                            blas.linear_row_family(w, x, &mut y, 1.0, 0.0)?;
                        } else {
                            blas.linear_typed(
                                crate::op_contract::ExecutionPhase::Prefill,
                                crate::backend::cuda::linear::LinearWeightRef::Bf16(w),
                                x,
                                &mut y,
                            )?;
                        }
                    }
                    Ok(())
                })();
                if graph {
                    let captured =
                        stream.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
                    launch?;
                    captured?.context("row gate captured no graph")?.launch()?;
                } else {
                    launch?;
                }
                let actual = stream.clone_dtoh(&y)?;
                ensure!(
                    actual.iter().all(|v| v.is_finite()),
                    "RUNNER_FAIL nonfinite LoRA row output"
                );
                Ok(actual)
            };
            if name == "ff.net.0.proj.lora_A" {
                let a = run(&heuristic, 1008, false, false, false)?;
                let b = run(&heuristic, 1511, false, false, false)?;
                let differences = a
                    .iter()
                    .zip(&b)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                println!("H3_ROW_CONTROL heuristic_1008_vs_1511 bit_differences={differences}");
                ensure!(
                    differences > 0,
                    "row-count discriminator did not detect old heuristic"
                );
            }
            for composed in [false, true].into_iter().filter(|&c| !c || accumulate) {
                let expected = run(&fixed, total, false, true, composed)?;
                for &chunk in &partitions {
                    for graph in [false, true] {
                        let actual = run(&fixed, chunk, graph, true, composed)?;
                        let differences = actual
                            .iter()
                            .zip(&expected)
                            .filter(|(a, b)| a.to_bits() != b.to_bits())
                            .count();
                        println!(
                            "H3_ROW_CELL operation={name} composed={composed} chunk={chunk} graph={graph} bit_differences={differences} elements={}",
                            actual.len()
                        );
                        ensure!(
                            differences == 0,
                            "DEFECT row-count invariance {name} chunk={chunk} graph={graph}"
                        );
                        cells += 1;
                    }
                }
            }
        }
        fs::remove_dir_all(runtime)?;
        println!("H3_ROW_GATE PASS cells={cells} bit_difference_limit=0 control=FAIL_AS_REQUIRED");
        Ok(())
    }

    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted, FF_MODEL, FF_H3_CAPTURE"]
    fn base_ffn_is_row_count_invariant() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let model = std::env::var_os("FF_MODEL")
            .map(PathBuf::from)
            .context("FF_MODEL missing")?;
        let capture = std::env::var_os("FF_H3_CAPTURE")
            .map(PathBuf::from)
            .context("FF_H3_CAPTURE missing")?;
        let partitions: Vec<usize> =
            serde_json::from_slice(&fs::read(capture.join("row_partitions.json"))?)?;
        ensure!(
            partitions.contains(&1008)
                && partitions.contains(&1511)
                && partitions.iter().all(|&n| n > 0 && n <= 60_431),
            "base row partitions outside production extents"
        );
        let spec = H3Spec::read(&model)?;
        let dir = model.join("transformer");
        let manifest = MappedCheckpoint::open_component(&dir, "diffusion_pytorch_model")?
            .minimax_h3_manifest(&spec)?;
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let stream = &device.stream;
        let runtime =
            std::env::temp_dir().join(format!("ff-h3-base-row-gate-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let blas = Blas::new(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(true, "base row invariance".into())),
        )?;
        let mut source = Source::new(&dir, &manifest);
        let mut defects = 0;
        let mut cells = 0;
        for layer in [0, spec.layers / 2] {
            for (name, output, input) in [
                ("ff.net.0.proj", 2 * spec.ffn, spec.hidden),
                ("ff.net.2", spec.hidden, spec.ffn),
            ] {
                let weight = source.tensor::<bf16>(
                    stream,
                    &format!("transformer_blocks.{layer}.{name}.weight"),
                    output * input,
                )?;
                for rows in [40_045usize, 60_431] {
                    for &other in &partitions {
                        for chunk in [1008, other] {
                            for n in [chunk, rows % chunk].into_iter().filter(|&n| n > 0) {
                                blas.prepare_fixed(
                                    u32::try_from(n)?,
                                    u32::try_from(output)?,
                                    u32::try_from(input)?,
                                )?;
                            }
                        }
                        let run = |start: usize, n: usize| -> Result<Vec<bf16>> {
                            let host: Vec<bf16> = (start * input..(start + n) * input)
                                .map(|i| {
                                    bf16::from_f32(
                                        (((i * 37 + i / input * 11) % 251) as f32 - 125.0) / 127.0,
                                    )
                                })
                                .collect();
                            let x = stream.clone_htod(&host)?;
                            let y = stream.alloc_zeros::<bf16>(n * output)?;
                            blas.linear_typed(
                                crate::op_contract::ExecutionPhase::Prefill,
                                crate::backend::cuda::linear::LinearWeightRef::Bf16(matrix(
                                    &weight, stream, output, input,
                                )?),
                                matrix(&x, stream, n, input)?,
                                &mut TensorMut(matrix(&y, stream, n, output)?),
                            )?;
                            let actual = stream.clone_dtoh(&y)?;
                            ensure!(
                                actual.iter().all(|v| v.is_finite()),
                                "RUNNER_FAIL nonfinite base row output"
                            );
                            Ok(actual)
                        };
                        let mut buffers = [Vec::new(), Vec::new()];
                        let mut begin = [0usize; 2];
                        let mut end = [0usize; 2];
                        let mut offset = 0;
                        let mut differences = 0;
                        while offset < rows {
                            for (arm, chunk) in [1008, other].into_iter().enumerate() {
                                if offset == end[arm] {
                                    begin[arm] = offset;
                                    end[arm] = (offset + chunk).min(rows);
                                    buffers[arm] = run(offset + 17, end[arm] - offset)?;
                                }
                            }
                            let stop = end[0].min(end[1]);
                            let a = &buffers[0]
                                [(offset - begin[0]) * output..(stop - begin[0]) * output];
                            let b = &buffers[1]
                                [(offset - begin[1]) * output..(stop - begin[1]) * output];
                            differences += a
                                .iter()
                                .zip(b)
                                .filter(|(a, b)| a.to_bits() != b.to_bits())
                                .count();
                            offset = stop;
                        }
                        println!(
                            "H3_BASE_ROW_CELL layer={layer} operation={name} rows={rows} partition=1008_vs_{other} global_row_offset=17 bit_differences={differences} elements={}",
                            rows * output
                        );
                        defects += usize::from(differences > 0);
                        cells += 1;
                    }
                }
            }
        }
        fs::remove_dir_all(runtime)?;
        println!("H3_BASE_ROW_GATE cells={cells} defects={defects} bit_difference_limit=0");
        ensure!(
            defects == 0,
            "DEFECT base FFN row invariance cells={defects}"
        );
        Ok(())
    }

    /// Owned and borrowed FFN storage use identical plans, including poisoned scratch and subsequent blocks.
    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted, FF_MODEL, FF_H3_LORA, FF_H3_CAPTURE, FF_H3_REPLAY, FF_H3_THRESHOLDS"]
    fn ffn_alias_matches_owned_gpu() -> Result<()> {
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
        let replay = path("FF_H3_REPLAY")?;
        let frozen: serde_json::Value =
            serde_json::from_slice(&fs::read(path("FF_H3_THRESHOLDS")?)?)?;
        ensure!(frozen["bit_difference_limit"] == 0, "alias gate bound");
        let spec = H3Spec::read(&model)?;
        let transformer = model.join("transformer");
        let manifest = MappedCheckpoint::open_component(&transformer, "diffusion_pytorch_model")?
            .minimax_h3_manifest(&spec)?;
        let lora_path = path("FF_H3_LORA")?;
        let lora_dir = lora_path.parent().context("LoRA directory")?;
        let stem = lora_path
            .file_stem()
            .and_then(|v| v.to_str())
            .context("LoRA stem")?;
        let (lora_manifest, adapter) =
            MappedCheckpoint::open_component(lora_dir, stem)?.minimax_h3_lora_manifest(&spec)?;
        let bytes = fs::read(capture.join("lora_a_up_input.bf16"))?;
        ensure!(
            bytes.len() == 1511 * spec.hidden * 2,
            "common input [1511,hidden] BF16"
        );
        let common: Vec<bf16> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect();
        ensure!(
            common.iter().all(|v| v.is_finite()),
            "RUNNER_FAIL nonfinite common input"
        );
        let adaln_map = mapped(&replay.join("A-bf16/adaln.safetensors"))?;
        let adaln = SafeTensors::deserialize(&adaln_map)?;
        let device = Device::new(0)?;
        let s = &device.stream;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let kernels = BlockKernels::new(&device.ctx)?;
        let weights: Vec<BlockWeights> = (0..2)
            .map(|layer| {
                BlockWeights::resident(
                    s,
                    &spec,
                    adapter.rank,
                    &format!("transformer_blocks.{layer}"),
                    &mut Source::new(&transformer, &manifest),
                    &mut Source::new(lora_dir, &lora_manifest),
                )
            })
            .collect::<Result<_>>()?;
        let mut cells = 0;
        let mut defects = 0;
        let mut control_bits = 0;
        for case in ["A", "P"] {
            let layout = capture_layout(&spec, &replay.join(format!("{case}-bf16")))?;
            let rows = layout.rows();
            let chunk = frozen["runs"][format!("{case}-derived")]["plan"]["main"]["chunk"]
                .as_u64()
                .context("frozen main chunk")? as usize;
            let input: Vec<bf16> = (0..rows * spec.hidden)
                .map(|i| common[((i / spec.hidden + 17) % 1511) * spec.hidden + i % spec.hidden])
                .collect();
            let runtime =
                std::env::temp_dir().join(format!("ff-h3-alias-{}-{case}", std::process::id()));
            fs::create_dir(&runtime)?;
            let blas = Blas::new(
                &device,
                &ops,
                &runtime,
                Rc::new(Closure::new(true, "H3 alias identity".into())),
            )?;
            let mut owned = BlockWork::new(s, &spec, adapter.rank, rows)?;
            let mut borrowed = BlockWork::new(s, &spec, adapter.rank, rows)?;
            owned.prepare(&blas, &spec)?;
            borrowed.prepare(&blas, &spec)?;
            for (output, input) in [
                (adapter.rank, spec.hidden),
                (adapter.rank, spec.ffn),
                (2 * spec.ffn, adapter.rank),
                (spec.hidden, adapter.rank),
            ] {
                blas.bind_row_family(crate::backend::setup::LinearShape {
                    rows: FFN_CHUNK_ROWS,
                    output,
                    input,
                })?;
            }
            let plan = owned.ffn_cost(&blas, &spec, chunk, false)?;
            blas.bind_workspace(plan.workspace)?;
            let mut x = s.clone_htod(&input)?;
            owned.bind_ffn(s, &blas, &spec, plan, &x)?;
            borrowed.bind_ffn(
                s,
                &blas,
                &spec,
                FfnPlan {
                    borrowed: true,
                    allocation: 0,
                    ..plan
                },
                &x,
            )?;
            ensure!(
                borrowed.hidden.ptr == pointer(&borrowed.q_raw, s)
                    && borrowed.gated.ptr == pointer(&borrowed.k_raw, s)
                    && borrowed.hidden.span_bytes + 256 <= (borrowed.q_raw.len() * 2) as u64,
                "borrowed address/control capacity"
            );
            let receipt = blas.bound_plan_receipt()?;
            let rotary = s.clone_htod(&crate::minimax_h3::rotary_table(&spec, &layout))?;
            let tables: Vec<_> = (0..2)
                .map(|layer| -> Result<_> {
                    let view = adaln.tensor(&format!("block{layer}.step0"))?;
                    ensure!(
                        view.dtype() == Dtype::BF16 && view.shape()[1] == 6 * spec.hidden,
                        "modulation layout"
                    );
                    let values: Vec<bf16> = view
                        .data()
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| bf16::from_le_bytes(*b))
                        .collect();
                    Ok((s.clone_htod(&values)?, u32::try_from(view.shape()[0])?))
                })
                .collect::<Result<_>>()?;
            ensure!(
                tables[0].1 == tables[1].1 && tables[0].1 > 0,
                "modulation entries"
            );
            let index = s.clone_htod(
                &(0..rows)
                    .map(|r| (r % tables[0].1 as usize) as u32)
                    .collect::<Vec<_>>(),
            )?;
            let mut flash = Flash::new(s.clone(), rows, spec.heads, spec.heads, spec.head_dim)?
                .with_full_single_probability()?;
            let context = BlockContext {
                s,
                blas: &blas,
                kernels: &kernels,
                spec: &spec,
                lora_scale: adapter.scale,
            };
            let mut run = |work: &mut BlockWork,
                           graph: bool,
                           full: bool,
                           shifted: bool|
             -> Result<Vec<bf16>> {
                s.memcpy_htod(&input, &mut x)?;
                s.memcpy_htod(&input, &mut work.normed)?;
                let poison = if work.ffn_owned.is_some() { 0x3f } else { 0xbf };
                unsafe {
                    crate::backend::cuda::copy::memset_bytes(
                        work.hidden.ptr,
                        poison,
                        usize::try_from(work.hidden.span_bytes)?,
                        s,
                    )?;
                    crate::backend::cuda::copy::memset_bytes(
                        work.gated.ptr,
                        poison,
                        usize::try_from(work.gated.span_bytes)?,
                        s,
                    )?;
                }
                s.synchronize()?;
                let mut launch = || -> Result<()> {
                    if full {
                        for (layer, weight) in weights.iter().enumerate() {
                            let modulation = Modulation {
                                table: &tables[layer].0,
                                offset: 0,
                                entries: tables[layer].1,
                                index: &index,
                            };
                            block_forward(
                                &context,
                                &mut flash,
                                weight,
                                &Conditioning::Modulated {
                                    rotary: &rotary,
                                    modulation: &modulation,
                                },
                                &mut x,
                                work,
                            )?;
                        }
                    } else if shifted {
                        ffn_forward::<true>(&context, &weights[0], work)?;
                    } else {
                        ffn_forward::<false>(&context, &weights[0], work)?;
                    }
                    Ok(())
                };
                if graph {
                    s.begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                    let result = launch();
                    let captured = s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
                    result?;
                    let captured = captured?.context("alias gate captured no graph")?;
                    captured.launch()?;
                    captured.launch()?;
                } else {
                    launch()?;
                    launch()?;
                }
                let output = s.clone_dtoh(if full { &x } else { &work.normed })?;
                ensure!(
                    output.iter().all(|v| v.is_finite()),
                    "RUNNER_FAIL nonfinite alias output"
                );
                Ok(output)
            };
            for full in [false, true] {
                let reference = run(&mut owned, false, full, false)?;
                for (borrow, graph) in [(false, true), (true, false), (true, true)] {
                    let actual = run(
                        if borrow { &mut borrowed } else { &mut owned },
                        graph,
                        full,
                        false,
                    )?;
                    let bits = reference
                        .iter()
                        .zip(&actual)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    cells += 1;
                    defects += usize::from(bits > 0);
                    println!(
                        "H3_ALIAS_CELL case={case} rows={rows} chunk={chunk} tail={} borrowed={borrow} graph={graph} next_block={full} replays=2 bit_differences={bits} limit=0",
                        rows % chunk
                    );
                }
                if !full && case == "A" {
                    let control = run(&mut borrowed, false, false, true)?;
                    control_bits = reference
                        .iter()
                        .zip(&control)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    println!(
                        "H3_ALIAS_CONTROL consumer_offset_bytes=256 bit_differences={control_bits}"
                    );
                }
            }
            ensure!(
                receipt == blas.bound_plan_receipt()?,
                "alias test changed bound algorithms"
            );
            s.synchronize()?;
            drop(blas);
            fs::remove_dir_all(runtime)?;
        }
        println!(
            "H3_ALIAS_GATE cells={cells} defects={defects} bit_difference_limit=0 control_bit_differences={control_bits}"
        );
        ensure!(
            control_bits > 0,
            "RUNNER_FAIL alias-address control undetected"
        );
        ensure!(defects == 0, "DEFECT FFN storage alias cells={defects}");
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
        let mut flash = Flash::new(s.clone(), rows, spec.heads, spec.heads, spec.head_dim)?
            .with_full_single_probability()?;
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

    /// The GGUF layer gate: one H3 transformer block with the gguf-py-dequantized weights, run by the engine on the
    /// definition reference's dumped input and conditioning, compared with the definition's own output. The floor is
    /// the max over the reference's pairwise arm spreads and the engine's official-weight spread; the control
    /// re-runs with the fused gate+up halves unswapped.
    /// 2026-11-04: the `official` arm keeps the GGUF AdaLN and the GGUF fixed projections, and the floor adds the
    /// `bf16_strict` arm. The earlier floor was the FP32-vs-BF16 spread alone, and the `official` dump it scored
    /// against predated the harness revision that put the GGUF modulation in that arm. The thresholds below were
    /// re-derived from the regenerated arms before the candidate was rerun. The `bf16_strict` spread against
    /// `bf16` is exactly 0 because `allow_bf16_reduced_precision_reduction=False` is a no-op on the CPU
    /// reference, so the floor is the FP32-vs-BF16 spread in practice.
    #[test]
    #[ignore = "FF_MODEL, FF_H3_GGUF, FF_H3_GGUF_BLOCK_OUT, FF_H3_LORA"]
    fn gguf_block_matches_the_definition_reference() -> Result<()> {
        let var = |key: &str| {
            std::env::var_os(key)
                .map(PathBuf::from)
                .with_context(|| format!("{key} missing"))
        };
        let model = var("FF_MODEL")?;
        let gguf_path = var("FF_H3_GGUF")?;
        let out = var("FF_H3_GGUF_BLOCK_OUT")?;
        let lora_path = var("FF_H3_LORA")?;
        let spec = H3Spec::read(&model)?;
        let gguf = crate::gguf::GgufFile::open(&gguf_path)?;
        let manifest = MappedCheckpoint::minimax_h3_gguf_manifest(&spec, &gguf)?;
        let transformer = model.join("transformer");
        let official = MappedCheckpoint::open_component(&transformer, "diffusion_pytorch_model")?
            .minimax_h3_manifest(&spec)?;
        let lora_dir = lora_path.parent().context("LoRA directory")?.to_owned();
        let stem = lora_path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix(".safetensors"))
            .context("LoRA file stem")?;
        let (lora_manifest, adapter) =
            MappedCheckpoint::open_component(&lora_dir, stem)?.minimax_h3_lora_manifest(&spec)?;

        let input_map = mapped(&out.join("input.safetensors"))?;
        let input = SafeTensors::deserialize(&input_map)?;
        let hidden = input.tensor("hidden_states")?;
        ensure!(
            hidden.dtype() == Dtype::BF16 && hidden.shape().len() == 3,
            "hidden_states {:?} {:?}",
            hidden.dtype(),
            hidden.shape()
        );
        let rows = hidden.shape()[1];
        let input_host: Vec<bf16> = hidden
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect();
        let table = input.tensor("modulation")?;
        ensure!(
            table.dtype() == Dtype::BF16 && table.shape().len() == 2,
            "modulation {:?} {:?}",
            table.dtype(),
            table.shape()
        );
        let entries = u32::try_from(table.shape()[0])?;
        let table_host: Vec<bf16> = table
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect();
        let rotary = input.tensor("rotary_table")?;
        ensure!(
            rotary.dtype() == Dtype::BF16 && rotary.shape() == [rows, 6 * spec.rope_freq_dim],
            "rotary_table {:?} {:?}",
            rotary.dtype(),
            rotary.shape()
        );
        let rotary_host: Vec<bf16> = rotary
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect();
        let (indices, _) = decoded(&input, "adaln_indices")?;
        let index_host: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        let output = |arm: &str| -> Result<Vec<f64>> {
            let map = mapped(&out.join(format!("output_{arm}.safetensors")))?;
            let file = SafeTensors::deserialize(&map)?;
            let tensor = file.tensor("output")?;
            ensure!(
                tensor.shape() == [1, rows, spec.hidden],
                "{arm} output {:?} {:?}",
                tensor.dtype(),
                tensor.shape()
            );
            Ok(match tensor.dtype() {
                Dtype::BF16 => tensor
                    .data()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| f64::from(bf16::from_le_bytes(*b).to_f32()))
                    .collect(),
                Dtype::F32 => tensor
                    .data()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f64::from(f32::from_le_bytes(*b)))
                    .collect(),
                other => bail!("{arm} output dtype {other:?}"),
            })
        };
        let reference_bf16 = output("bf16")?;
        let reference_fp32 = output("fp32")?;
        let reference_bf16_strict = output("bf16_strict")?;
        let reference_official = output("official")?;

        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let s = &device.stream;
        let runtime = std::env::temp_dir().join(format!("ff-h3-gguf-block-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let blas = Blas::new(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(true, "MiniMax-H3 GGUF block gate".into())),
        )?;
        let r = adapter.rank;
        let gguf_dir = gguf_path.parent().context("GGUF directory")?;
        let mut flash = Flash::new(s.clone(), rows, spec.heads, spec.heads, spec.head_dim)?
            .with_full_single_probability()?;
        let kernels = BlockKernels::new(&device.ctx)?;
        let weights = BlockWeights::resident(
            s,
            &spec,
            r,
            "transformer_blocks.0",
            &mut Source::new(gguf_dir, &manifest),
            &mut Source::new(&lora_dir, &lora_manifest),
        )?;
        let mut wrong = manifest.clone();
        wrong
            .entries
            .get_mut("transformer_blocks.0.ff.net.0.proj.weight")
            .context("fc1 entry")?
            .swap_halves = false;
        let wrong_weights = BlockWeights::resident(
            s,
            &spec,
            r,
            "transformer_blocks.0",
            &mut Source::new(gguf_dir, &wrong),
            &mut Source::new(&lora_dir, &lora_manifest),
        )?;
        let official_weights = BlockWeights::resident(
            s,
            &spec,
            r,
            "transformer_blocks.0",
            &mut Source::new(&transformer, &official),
            &mut Source::new(&lora_dir, &lora_manifest),
        )?;
        let mut work = BlockWork::new(s, &spec, r, rows)?;
        work.prepare(&blas, &spec)?;
        let rotary_device = s.clone_htod(&rotary_host)?;
        let table_device = s.clone_htod(&table_host)?;
        let index_device = s.clone_htod(&index_host)?;
        let modulation = Modulation {
            table: &table_device,
            offset: 0,
            entries,
            index: &index_device,
        };
        let conditioning = Conditioning::Modulated {
            rotary: &rotary_device,
            modulation: &modulation,
        };
        let context = BlockContext {
            s,
            blas: &blas,
            kernels: &kernels,
            spec: &spec,
            lora_scale: 0.0,
        };
        let mut run = |weights: &BlockWeights| -> Result<Vec<f64>> {
            let mut x = s.clone_htod(&input_host)?;
            block_forward(
                &context,
                &mut flash,
                weights,
                &conditioning,
                &mut x,
                &mut work,
            )?;
            Ok(s.clone_dtoh(&x)?
                .into_iter()
                .map(|v| f64::from(v.to_f32()))
                .collect())
        };
        let candidate = run(&weights)?;
        let control = run(&wrong_weights)?;
        let engine_official = run(&official_weights)?;

        // The floor on this gate's own input: the reference's max pairwise arm spread, and the engine's
        // block-vs-definition arithmetic spread measured here with the official BF16 weights on the same
        // conditioning (the `official` arm shares the GGUF temb/rotary and the GGUF AdaLN table).
        let arm_pairs = [
            ("bf16_vs_fp32", distance(&reference_bf16, &reference_fp32)),
            (
                "bf16_vs_bf16_strict",
                distance(&reference_bf16, &reference_bf16_strict),
            ),
            (
                "fp32_vs_bf16_strict",
                distance(&reference_fp32, &reference_bf16_strict),
            ),
        ];
        let arm_floor = arm_pairs.iter().fold([0.0f64, 0.0, 2.0], |floor, (_, d)| {
            [floor[0].max(d[0]), floor[1].max(d[1]), floor[2].min(d[2])]
        });
        let engine_floor = distance(&engine_official, &reference_official);
        let floor = [
            arm_floor[0].max(engine_floor[0]),
            arm_floor[1].max(engine_floor[1]),
            arm_floor[2].min(engine_floor[2]),
        ];
        let d = distance(&candidate, &reference_bf16);
        let d_control = distance(&control, &reference_bf16);
        let abs_tau = (2.0 * floor[0]).max(1e-5);
        let rel_tau = (2.0 * floor[1]).max(1e-6);
        let cos_tau = 1.0 - 2.0 * (1.0 - floor[2]);
        let ok = d[0] <= abs_tau && d[1] <= rel_tau && d[2] >= cos_tau;
        let control_abs_ratio = d_control[0] / abs_tau;
        let control_rel_ratio = d_control[1] / rel_tau;
        let caught = control_abs_ratio > 10.0 || control_rel_ratio > 10.0;
        println!(
            "H3_GGUF_BLOCK candidate={d:?} arm_pairs={arm_pairs:?} arm_floor={arm_floor:?} engine_floor={engine_floor:?} floor={floor:?} control={d_control:?} control_abs_ratio={control_abs_ratio:.3} control_rel_ratio={control_rel_ratio:.3} abs_tau={abs_tau} rel_tau={rel_tau} within={ok}"
        );
        ensure!(
            candidate.iter().all(|v| v.is_finite()),
            "non-finite block output"
        );
        ensure!(ok, "GGUF block output {d:?} outside the floor {floor:?}");
        ensure!(
            caught,
            "unswapped-fc1 control stayed within 10x tau (abs {:.3}x, rel {:.3}x)",
            control_abs_ratio,
            control_rel_ratio
        );
        fs::remove_dir(&runtime)?;
        println!("H3_GGUF_BLOCK_GATE PASS control=FAIL_AS_REQUIRED");
        Ok(())
    }

    /// The streamed quantized path gate: the real target's GGUF manifest placed `Streamed` (host tier) and
    /// `StreamedDisk` (disk tier), every layer's device slot compared bit-exact against `BlockWeights::resident`
    /// for the same layer. The control shifts one entry's source offset by one Q4_0 block (18 B) and must fail.
    #[test]
    #[ignore = "FF_MODEL, FF_H3_GGUF, FF_H3_LORA"]
    fn gguf_streamed_slots_match_resident_weights() -> Result<()> {
        let var = |key: &str| -> Result<PathBuf> {
            Ok(PathBuf::from(
                std::env::var_os(key).with_context(|| format!("{key} missing"))?,
            ))
        };
        let model = var("FF_MODEL")?;
        let gguf_path = var("FF_H3_GGUF")?;
        let lora_path = var("FF_H3_LORA")?;
        let spec = H3Spec::read(&model)?;
        let gguf = crate::gguf::GgufFile::open(&gguf_path)?;
        let manifest = MappedCheckpoint::minimax_h3_gguf_manifest(&spec, &gguf)?;
        let lora_dir = lora_path.parent().context("LoRA directory")?.to_owned();
        let stem = lora_path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix(".safetensors"))
            .context("LoRA file stem")?;
        let (lora_manifest, adapter) =
            MappedCheckpoint::open_component(&lora_dir, stem)?.minimax_h3_lora_manifest(&spec)?;
        let gguf_dir = gguf_path.parent().context("GGUF directory")?.to_owned();

        let sizing = dense_sizing(&manifest, spec.layers)?;
        let owned = lora_manifest
            .entries
            .iter()
            .filter(|(name, _)| name.starts_with("transformer_blocks."))
            .map(|(_, entry)| entry.bytes)
            .sum::<usize>()
            + spec.layers * 2 * (2 * spec.hidden + 2 * spec.head_dim)
            + sizing.staging
            + sizing.transient;

        let device = Device::new(0)?;
        let stream = &device.stream;
        let mismatches =
            |placement: Placement, perturb: Option<(usize, &str, isize)>| -> Result<Vec<usize>> {
                let mut base = Source::new(&gguf_dir, &manifest);
                let mut dense = place_layers(
                    &device,
                    &mut base,
                    (
                        &sizing.layers,
                        &sizing.source_layers,
                        sizing.largest,
                        owned,
                        0,
                    ),
                    placement,
                    |layer| format!("transformer_blocks.{layer}"),
                    &LINEAR_ROLES,
                )?
                .context("streamed placement")?;
                if let Some((layer, name, delta)) = perturb {
                    dense.shift_source_offset(layer, name, delta)?;
                }
                let mut mismatches = Vec::new();
                for (layer, &layer_bytes) in sizing.layers.iter().enumerate() {
                    let name = format!("transformer_blocks.{layer}");
                    let resident = BlockWeights::resident(
                        stream,
                        &spec,
                        adapter.rank,
                        &name,
                        &mut Source::new(&gguf_dir, &manifest),
                        &mut Source::new(&lora_dir, &lora_manifest),
                    )?;
                    let expected = stream
                        .clone_dtoh(resident.storage.as_ref().context("resident storage")?)?;
                    dense.begin(layer, stream)?;
                    let pointer = dense.pointer(layer, 0, layer_bytes)?;
                    let actual = stream.alloc_zeros::<bf16>(expected.len())?;
                    unsafe {
                        cudarc::driver::result::memcpy_dtod_async(
                            actual.device_ptr(stream).0,
                            pointer,
                            layer_bytes,
                            stream.cu_stream(),
                        )?;
                    }
                    dense.end(layer, stream)?;
                    let actual = stream.clone_dtoh(&actual)?;
                    if actual
                        .iter()
                        .map(|v| v.to_bits())
                        .ne(expected.iter().map(|v| v.to_bits()))
                    {
                        mismatches.push(layer);
                    }
                }
                Ok(mismatches)
            };
        for placement in [Placement::Streamed, Placement::StreamedDisk] {
            let clean = mismatches(placement, None)?;
            ensure!(
                clean.is_empty(),
                "{placement:?}: streamed slots differ from resident in layers {clean:?}"
            );
            let control = mismatches(
                placement,
                Some((0, "transformer_blocks.0.attn.to_q.weight", 18)),
            )?;
            ensure!(
                control == vec![0],
                "{placement:?}: the shifted source offset control did not fail exactly layer 0: {control:?}"
            );
            println!(
                "H3_GGUF_STREAMED {placement:?} layers={} bit_exact_mismatches=0 control=FAIL_AS_REQUIRED",
                spec.layers
            );
        }
        println!("H3_GGUF_STREAMED_GATE PASS");
        Ok(())
    }

    /// The GGUF AdaLN gate: the 50 `adaln_proj` weights dequantized from the GGUF into the engine's AdaLN table,
    /// against the definition's per-block modulation for the same timesteps (the harness dump, layer-major). The
    /// floor is the reference's own FP32-vs-BF16 arm spread; the control swaps the video/audio tags.
    #[test]
    #[ignore = "FF_MODEL, FF_H3_GGUF, FF_H3_GGUF_ADALN_OUT"]
    fn gguf_adaln_matches_the_definition_modulation() -> Result<()> {
        let var = |key: &str| {
            std::env::var_os(key)
                .map(PathBuf::from)
                .with_context(|| format!("{key} missing"))
        };
        let model = var("FF_MODEL")?;
        let gguf_path = var("FF_H3_GGUF")?;
        let out = var("FF_H3_GGUF_ADALN_OUT")?;
        let spec = H3Spec::read(&model)?;
        let gguf = crate::gguf::GgufFile::open(&gguf_path)?;
        let manifest = MappedCheckpoint::minimax_h3_gguf_manifest(&spec, &gguf)?;

        let adaln_map = mapped(&out.join("adaln.safetensors"))?;
        let adaln = SafeTensors::deserialize(&adaln_map)?;
        let timesteps_tensor = adaln.tensor("timesteps")?;
        ensure!(
            timesteps_tensor.dtype() == Dtype::F32 && timesteps_tensor.shape().len() == 1,
            "timesteps {:?} {:?}",
            timesteps_tensor.dtype(),
            timesteps_tensor.shape()
        );
        let timesteps: Vec<f32> = timesteps_tensor
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let width = 6 * spec.hidden;
        let modulation = |arm: &str| -> Result<Vec<bf16>> {
            let tensor = adaln.tensor(&format!("modulation_{arm}"))?;
            ensure!(
                tensor.shape().len() == 2
                    && tensor.shape()[0] == spec.layers * timesteps.len() * MODALITIES as usize
                    && tensor.shape()[1] == width,
                "modulation_{arm} {:?} {:?} for {} layers x {} timesteps",
                tensor.dtype(),
                tensor.shape(),
                spec.layers,
                timesteps.len()
            );
            Ok(match tensor.dtype() {
                Dtype::BF16 => tensor
                    .data()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| bf16::from_le_bytes(*b))
                    .collect(),
                Dtype::F32 => tensor
                    .data()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| bf16::from_f32(f32::from_le_bytes(*b)))
                    .collect(),
                other => bail!("modulation_{arm} dtype {other:?}"),
            })
        };
        let to_f64 = |values: &[bf16]| -> Vec<f64> {
            values.iter().map(|&v| f64::from(v.to_f32())).collect()
        };
        let expected_bf16 = to_f64(&modulation("bf16")?);
        let expected_fp32 = to_f64(&modulation("fp32")?);
        let floor = distance(&expected_bf16, &expected_fp32);

        let pairs: Vec<(u32, u8)> = (0..timesteps.len() as u32)
            .flat_map(|t| (0..MODALITIES).map(move |tag| (t, tag as u8)))
            .collect();
        let plan = AdalnPlan {
            timesteps: timesteps.clone(),
            pairs: pairs.clone(),
            row_pairs: Vec::new(),
            row_timesteps: Vec::new(),
        };

        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let s = &device.stream;
        let runtime = std::env::temp_dir().join(format!("ff-h3-gguf-adaln-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let mut blas = Blas::new(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(true, "MiniMax-H3 GGUF AdaLN gate".into())),
        )?;
        let gguf_dir = gguf_path.parent().context("GGUF directory")?;
        let table = AdalnTable::build(&device, &mut blas, gguf_dir, &manifest, &spec, &plan)?;
        let blocks: Vec<f64> = s
            .clone_dtoh(&table.blocks)?
            .into_iter()
            .map(|v| f64::from(v.to_f32()))
            .collect();
        fs::remove_dir(&runtime)?;

        let mut candidate = Vec::new();
        let mut expected = Vec::new();
        let mut control_candidate = Vec::new();
        let mut control_expected = Vec::new();
        for layer in 0..spec.layers {
            for (pair, &(t, tag)) in pairs.iter().enumerate() {
                let row = layer * pairs.len() + pair;
                let c = &blocks[row * width..][..width];
                candidate.extend_from_slice(c);
                expected.extend_from_slice(&expected_bf16[row * width..][..width]);
                if tag != 1 {
                    control_candidate.extend_from_slice(c);
                    let other = layer * pairs.len()
                        + t as usize * MODALITIES as usize
                        + (2 - usize::from(tag));
                    control_expected.extend_from_slice(&expected_bf16[other * width..][..width]);
                }
            }
        }
        let d = distance(&candidate, &expected);
        let d_control = distance(&control_candidate, &control_expected);
        let abs_tau = (2.0 * floor[0]).max(1e-5);
        let rel_tau = (2.0 * floor[1]).max(1e-6);
        let cos_tau = 1.0 - 2.0 * (1.0 - floor[2]);
        let ok = d[0] <= abs_tau && d[1] <= rel_tau && d[2] >= cos_tau;
        let caught = d_control[0] > 10.0 * abs_tau || d_control[1] > 10.0 * rel_tau;
        println!(
            "H3_GGUF_ADALN timesteps={} pairs={} candidate={d:?} arm_floor={floor:?} control={d_control:?} abs_tau={abs_tau} rel_tau={rel_tau} within={ok}",
            timesteps.len(),
            pairs.len()
        );
        ensure!(
            candidate.iter().all(|v| v.is_finite()),
            "non-finite AdaLN table"
        );
        ensure!(ok, "GGUF AdaLN table {d:?} outside the arm floor {floor:?}");
        ensure!(caught, "tag-swap control stayed within 10x tau");
        println!("H3_GGUF_ADALN_GATE PASS control=FAIL_AS_REQUIRED");
        Ok(())
    }

    /// Blocks placed per `FF_H3_PLACEMENT`, eager (the H3 path captures no graph), native AdaLN table. Keyframe
    /// conditioning rows are the captured FP32 VAE-encoder latents, the form under which the frozen keyframe cells
    /// are 0. Every step in `FF_H3_STEPS` (each held by `FF_H3_REPLAY`): the native velocity of the generated video
    /// rows and of the audio rows on the run's teacher-forced inputs, within the frozen `{FF_H3_CELL}.step{k}.*`
    /// thresholds; chunked controls repeat FFN input row 0, and single-chunk controls rotate text rows by one.
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
            let chunked_control = dit.work.chunk < dit.work.rows;
            let mut evaluate = |text: &CudaSlice<bf16>,
                                repeat_ffn_input: bool|
             -> Result<[[f64; 3]; 2]> {
                dit.work.repeat_ffn_input = repeat_ffn_input;
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
            let [d_video, d_audio] = evaluate(&text, false)?;
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
                let control_text = if chunked_control { &text } else { &rotated };
                let kind = if chunked_control {
                    "wrong-ffn-input-offset"
                } else {
                    "rotated-text"
                };
                let [c_video, c_audio] = evaluate(control_text, chunked_control)?;
                println!("H3_DIT step{k} {kind} control video={c_video:?} audio={c_audio:?}");
                control_failed =
                    !within(c_video, video_threshold)? || !within(c_audio, audio_threshold)?;
            }
        }
        fs::remove_dir(&runtime)?;
        ensure!(control_failed, "DiT control stayed within the thresholds");
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

    /// Cells A, D and E of the N6 freeze: tokens and grids of `tokens_from`, pixel rows of `pixel_sources` in order.
    fn conditioner_cells(
        s0: &Path,
        image_token: u32,
    ) -> Result<Vec<(&'static str, ConditionerPrompt)>> {
        let mut cells = Vec::new();
        for (name, tokens_from, pixel_sources) in [
            ("A", "A-bf16", &["A-bf16"][..]),
            ("D", "D-bf16", &["D-bf16"][..]),
            ("E", "D-bf16", &["A-bf16", "B-bf16"][..]),
        ] {
            let map = mapped(&s0.join(tokens_from).join("text.safetensors"))?;
            let text = SafeTensors::deserialize(&map)?;
            let tokens: Vec<u32> = decoded(&text, "token_ids")?
                .0
                .into_iter()
                .map(|v| v as u32)
                .collect();
            let (grid, _) = decoded(&text, "image_grid_thw")?;
            let grids = grid
                .as_chunks::<3>()
                .0
                .iter()
                .map(|g| crate::vision_input::ImageGrid {
                    t: g[0] as u32,
                    h: g[1] as u32,
                    w: g[2] as u32,
                })
                .collect();
            let mut pixels = Vec::new();
            for source in pixel_sources {
                let map = mapped(&s0.join(source).join("text.safetensors"))?;
                let file = SafeTensors::deserialize(&map)?;
                pixels.extend(
                    decoded(&file, "pixel_values")?
                        .0
                        .into_iter()
                        .map(|v| v as f32),
                );
            }
            let modality = tokens
                .iter()
                .map(|&id| u8::from(id == image_token))
                .collect();
            cells.push((
                name,
                ConditionerPrompt {
                    tokens,
                    modality,
                    grids,
                    pixels,
                },
            ));
        }
        Ok(cells)
    }

    /// N6b gate: the Qwen3-VL vision tower on the captured `pixel_values` of cells A, D and E; the merged features and
    /// the three deepstack outputs are scored against the fp32 composition legs at the frozen `thresholds-n6-r2.json`
    /// cells. Eager (the tower captures no graph), all tower weights resident. Control: deepstack taps shifted one
    /// block later (9/17/25) must leave every deepstack output of cell A outside its cell.
    #[test]
    #[ignore = "GPU gate; set FF_H3_TEXT_ENCODER, FF_H3_S0 (Stage 0 runs), FF_H3_N6_LEGS (N6 reference legs) and FF_H3_N6_THRESHOLDS"]
    fn conditioner_vision_matches_the_fp32_reference() -> Result<()> {
        use crate::backend::cuda::{ops::CommonOpsRef, weights::WeightCatalog};
        let var = |key: &str| -> Result<PathBuf> {
            Ok(PathBuf::from(
                std::env::var_os(key).with_context(|| format!("{key} missing"))?,
            ))
        };
        let (encoder, s0, legs) = (
            var("FF_H3_TEXT_ENCODER")?,
            var("FF_H3_S0")?,
            var("FF_H3_N6_LEGS")?,
        );
        let frozen: serde_json::Value =
            serde_json::from_slice(&fs::read(var("FF_H3_N6_THRESHOLDS")?)?)?;
        let spec = crate::model_spec::ModelSpec::read(&encoder)?;
        ensure!(
            spec.architecture == crate::model_spec::Architecture::Qwen3Vl,
            "text encoder parsed as {:?}",
            spec.architecture
        );
        let image_token = spec.vision.as_ref().context("vision missing")?.image_token;
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        let common = CommonOpsRef::new(&device.ctx)?;
        let mut catalog = WeightCatalog::load_with_policy(
            &encoder,
            &spec,
            &device,
            crate::checkpoint::LoadPolicy::VisionOnly,
        )?;
        ensure!(
            catalog.bytes == 1_190_533_600,
            "vision weight ledger {} B",
            catalog.bytes
        );
        let mut tower = super::super::vision_runtime::LayerNormTower::from_catalog(
            &spec,
            &mut catalog,
            &device,
            common,
        )?;
        ensure!(
            catalog.tensors.is_empty(),
            "vision binding left unloaded owners"
        );
        device.finish_upload()?;
        let runtime = std::env::temp_dir().join(format!("ff-h3-vision-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let mut blas = Blas::new(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(
                true,
                "MiniMax-H3 conditioner vision gate".into(),
            )),
        )?;
        let cells = conditioner_cells(&s0, image_token)?;
        let mut run = |tower: &super::super::vision_runtime::LayerNormTower,
                       cell: &ConditionerPrompt|
         -> Result<Vec<Vec<f64>>> {
            let input = crate::vision_input::MultimodalInput {
                input_ids: &cell.tokens,
                modality_ids: &cell.modality,
                pixel_values: &cell.pixels,
                grids: &cell.grids,
                audio: &[],
            };
            let request = tower.prepare(&device, &mut blas, &input, cell.tokens.len(), 0)?;
            let encoded = tower.encode(&device.stream, &blas, request, None)?;
            std::iter::once(&encoded.features)
                .chain(&encoded.deepstack)
                .map(|f| {
                    Ok(device
                        .stream
                        .clone_dtoh(f)?
                        .into_iter()
                        .map(|v| f64::from(v.to_f32()))
                        .collect())
                })
                .collect()
        };
        let names = ["merged", "deepstack0", "deepstack1", "deepstack2"];
        let reference = |cell: &str| -> Result<Vec<Vec<f64>>> {
            let map = mapped(
                &legs
                    .join(format!("C-{cell}-fp32"))
                    .join("conditioner.safetensors"),
            )?;
            let file = SafeTensors::deserialize(&map)?;
            names
                .iter()
                .map(|n| Ok(decoded(&file, &format!("vision.{n}"))?.0))
                .collect()
        };
        let mut defects = Vec::new();
        for (cell, inputs) in &cells {
            let outputs = run(&tower, inputs)?;
            let expected = reference(cell)?;
            ensure!(
                outputs.len() == names.len(),
                "cell {cell}: {} outputs",
                outputs.len()
            );
            for ((name, got), want) in names.iter().zip(&outputs).zip(&expected) {
                ensure!(
                    got.len() == want.len(),
                    "cell {cell} {name}: {} vs {}",
                    got.len(),
                    want.len()
                );
                ensure!(
                    got.iter().all(|v| v.is_finite()),
                    "cell {cell} {name}: non-finite output"
                );
                let d = distance(got, want);
                let ok = within(d, &frozen["thresholds"][format!("{cell}.vision.{name}")])?;
                println!("H3_VISION cell={cell} {name}={d:?} within={ok}");
                if !ok {
                    defects.push(format!("{cell}.{name}"));
                }
            }
        }
        tower.shift_deepstack_taps()?;
        let (_, a) = cells.first().context("cell A missing")?;
        let shifted = run(&tower, a)?;
        let expected = reference("A")?;
        for (k, name) in names.iter().enumerate().skip(1) {
            let d = distance(&shifted[k], &expected[k]);
            println!("H3_VISION shifted-taps control cell=A {name}={d:?}");
            ensure!(
                !within(d, &frozen["thresholds"][format!("A.vision.{name}")])?,
                "shifted deepstack taps passed A.vision.{name}"
            );
        }
        if !defects.is_empty() {
            bail!("H3_VISION_GATE DEFECT cells outside the frozen thresholds: {defects:?}");
        }
        println!("H3_VISION_GATE PASS cells=3 control=FAIL_AS_REQUIRED");
        Ok(())
    }

    /// Scores `got` against `want` (`[rows, width]`) under a frozen sink-rule cell: the candidate's sink rows `C`
    /// satisfy `I <= C <= U`, every row of `I` meets its row bound, and the rows outside `U` meet the aggregate bound.
    fn sink_rule(
        got: &[f64],
        want: &[f64],
        width: usize,
        cell: &serde_json::Value,
    ) -> Result<(bool, String)> {
        use std::collections::BTreeSet;
        let rows = want.len() / width;
        let number = |v: &serde_json::Value, key: &str| {
            v[key]
                .as_f64()
                .with_context(|| format!("sink cell lacks {key}"))
        };
        let (median, factor) = (
            number(cell, "median_norm_fp32")?,
            number(cell, "sink_factor")?,
        );
        let row = |v: &[f64], r: usize| v[r * width..(r + 1) * width].to_vec();
        let norm = |v: &[f64], r: usize| row(v, r).iter().map(|x| x * x).sum::<f64>().sqrt();
        let set = |key: &str| -> Result<BTreeSet<usize>> {
            cell[key]
                .as_array()
                .with_context(|| format!("sink cell lacks {key}"))?
                .iter()
                .map(|r| r.as_u64().map(|r| r as usize).context("sink row"))
                .collect()
        };
        let candidate: BTreeSet<usize> = (0..rows)
            .filter(|&r| norm(got, r) > factor * median)
            .collect();
        let (inter, union) = (set("I")?, set("U")?);
        let mut ok = inter.is_subset(&candidate) && candidate.is_subset(&union);
        let mut sink_rows = Vec::new();
        for &r in &inter {
            let d = distance(&row(got, r), &row(want, r));
            let bound = &cell["rows"][r.to_string()];
            let within_row = d[1] <= number(bound, "rel_max")? && d[2] >= number(bound, "cos_min")?;
            ok &= within_row;
            sink_rows.push((r, d[1], d[2], within_row));
        }
        let keep: Vec<usize> = (0..rows).filter(|r| !union.contains(r)).collect();
        let gather = |v: &[f64]| keep.iter().flat_map(|&r| row(v, r)).collect::<Vec<_>>();
        let aggregate = distance(&gather(got), &gather(want));
        ok &= within(aggregate, &cell["aggregate"])?;
        Ok((
            ok,
            format!(
                "sinks={candidate:?} I={inter:?} U={union:?} sink_rows={sink_rows:?} aggregate={aggregate:?}"
            ),
        ))
    }

    /// BF16 copy of an F32 reference tensor.
    fn bf16_of(file: &SafeTensors<'_>, name: &str) -> Result<Vec<bf16>> {
        Ok(decoded(file, name)?
            .0
            .into_iter()
            .map(bf16::from_f64)
            .collect())
    }

    /// Conditioner text gate state shared by the N6c and N6d gates: the checkpoint manifest with `layers` decoder
    /// layers loading, the embedding rows of every cell's text tokens, and the frozen thresholds.
    struct TextGate {
        encoder: PathBuf,
        legs: PathBuf,
        spec: crate::model_spec::ModelSpec,
        frozen: serde_json::Value,
        cells: Vec<(&'static str, ConditionerPrompt)>,
        table: BTreeMap<u32, Vec<bf16>>,
        manifest: TensorManifest,
    }

    impl TextGate {
        fn new(layers: usize) -> Result<Self> {
            let var = |key: &str| -> Result<PathBuf> {
                Ok(PathBuf::from(
                    std::env::var_os(key).with_context(|| format!("{key} missing"))?,
                ))
            };
            let (encoder, s0, legs) = (
                var("FF_H3_TEXT_ENCODER")?,
                var("FF_H3_S0")?,
                var("FF_H3_N6_LEGS")?,
            );
            let frozen = serde_json::from_slice(&fs::read(var("FF_H3_N6_THRESHOLDS")?)?)?;
            let spec = crate::model_spec::ModelSpec::read(&encoder)?;
            let image_token = spec.vision.as_ref().context("vision missing")?.image_token;
            let manifest = MappedCheckpoint::open_component(&encoder, "model")?
                .minimax_h3_conditioner_manifest(&spec, layers)?;
            let cells = conditioner_cells(&s0, image_token)?;
            let ids: std::collections::BTreeSet<u32> = cells
                .iter()
                .flat_map(|(_, c)| c.tokens.iter().copied())
                .filter(|&id| id != image_token)
                .collect();
            let ids: Vec<u32> = ids.into_iter().collect();
            let hidden = spec.hidden as usize;
            let rows = embedding_rows(
                &mut Source::new(&encoder, &manifest),
                "model.language_model.embed_tokens.weight",
                hidden,
                &ids,
            )?;
            let table = ids
                .iter()
                .zip(rows.chunks(hidden))
                .map(|(&id, row)| (id, row.to_vec()))
                .collect();
            println!(
                "H3_TEXT embedding_rows={} bytes={}",
                ids.len(),
                2 * rows.len()
            );
            Ok(Self {
                encoder,
                legs,
                spec,
                frozen,
                cells,
                table,
                manifest,
            })
        }

        fn image_token(&self) -> Result<u32> {
            Ok(self
                .spec
                .vision
                .as_ref()
                .context("vision missing")?
                .image_token)
        }

        /// Scores `taps` (hidden_states indices) of one cell against `{kind}-{cell}-fp32` and the frozen `{cell}.{label}` cells;
        /// returns whether the gating hidden_states[50] passed.
        fn score(
            &self,
            cell: &str,
            (kind, label): (&str, &str),
            outputs: &[Vec<bf16>],
            taps: &[usize],
        ) -> Result<bool> {
            let map = mapped(
                &self
                    .legs
                    .join(format!("{kind}-{cell}-fp32"))
                    .join("conditioner.safetensors"),
            )?;
            let file = SafeTensors::deserialize(&map)?;
            let mut gating = true;
            for (output, &tap) in outputs.iter().zip(taps) {
                let got: Vec<f64> = output.iter().map(|v| f64::from(v.to_f32())).collect();
                let want = decoded(&file, &format!("text.hidden_states_{tap}"))?.0;
                ensure!(
                    got.len() == want.len(),
                    "cell {cell} hidden_states[{tap}] extent"
                );
                ensure!(
                    got.iter().all(|v| v.is_finite()),
                    "cell {cell} hidden_states[{tap}] non-finite"
                );
                let (ok, summary) = sink_rule(
                    &got,
                    &want,
                    self.spec.hidden as usize,
                    &self.frozen["thresholds"][format!("{cell}.{label}.hidden_states_{tap}")],
                )?;
                println!("H3_TEXT {label} cell={cell} hidden_states_{tap} within={ok} {summary}");
                if tap == 50 {
                    gating = ok;
                }
            }
            Ok(gating)
        }
    }

    const TEXT_TAPS: [usize; 4] = [1, 4, 25, 50];

    /// N6c gate: Qwen3-VL decoder layers 0-49 on the fp32 reference vision features of cells A, D and E (image rows
    /// spliced as BF16, the three deepstack features added after layers 0-2); hidden_states[50] is scored under the
    /// frozen sink rule of `thresholds-n6-r2.json` against the `T-<cell>-fp32` legs, hidden_states[1/4/25] printed.
    /// Layers placed per `FF_H3_PLACEMENT`, eager (no graph capture). Layer 50 also loads for the control tap.
    /// Controls on cell A: hidden_states[51] scored as hidden_states[50], and deepstack off, must both fail.
    #[test]
    #[ignore = "GPU gate; set FF_H3_TEXT_ENCODER, FF_H3_S0, FF_H3_N6_LEGS, FF_H3_N6_THRESHOLDS and FF_H3_PLACEMENT"]
    fn conditioner_text_matches_the_fp32_reference() -> Result<()> {
        let gate = TextGate::new(51)?;
        let (hidden, image_token) = (gate.spec.hidden as usize, gate.image_token()?);
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let runtime = std::env::temp_dir().join(format!("ff-h3-text-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let blas = Blas::new(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(
                true,
                "MiniMax-H3 conditioner text gate".into(),
            )),
        )?;
        let capacity = gate
            .cells
            .iter()
            .map(|(_, c)| c.tokens.len())
            .max()
            .context("no cells")?;
        let mut conditioner = Conditioner::new(
            &device,
            &gate.spec,
            (51, capacity, 0),
            Source::new(&gate.encoder, &gate.manifest),
            Placement::from_env()?,
        )?;
        let mut defects = Vec::new();
        for (cell, inputs) in &gate.cells {
            conditioner.prepare(&blas, inputs.tokens.len())?;
            let map = mapped(
                &gate
                    .legs
                    .join(format!("C-{cell}-fp32"))
                    .join("conditioner.safetensors"),
            )?;
            let reference = SafeTensors::deserialize(&map)?;
            let merged = bf16_of(&reference, "vision.merged")?;
            let deepstack = (0..3)
                .map(|k| {
                    Ok(device
                        .stream
                        .clone_htod(&bf16_of(&reference, &format!("vision.deepstack{k}"))?)?)
                })
                .collect::<Result<Vec<_>>>()?;
            let embeds = conditioner_embeds(inputs, image_token, &gate.table, &merged, hidden)?;
            let runs = image_runs(&inputs.tokens, image_token);
            let rope = conditioner_rope(&gate.spec, inputs)?;
            let input = ConditionerInput {
                embeds: &embeds,
                deepstack: &deepstack,
                runs: &runs,
                rope: &rope,
            };
            let mut taps = TEXT_TAPS.to_vec();
            taps.push(51);
            let outputs = conditioner.forward(&device.stream, &blas, &input, &taps)?;
            if !gate.score(cell, ("T", "text"), &outputs[..4], &TEXT_TAPS)? {
                defects.push(*cell);
            }
            if *cell == "A" {
                ensure!(
                    !gate.score(cell, ("T", "text"), &outputs[4..], &[50])?,
                    "hidden_states[51] passed the hidden_states[50] cell"
                );
                let flat = ConditionerInput {
                    deepstack: &[],
                    ..input
                };
                let outputs = conditioner.forward(&device.stream, &blas, &flat, &[50])?;
                ensure!(
                    !gate.score(cell, ("T", "text"), &outputs, &[50])?,
                    "deepstack-off control passed cell A"
                );
                println!("H3_TEXT controls cell=A tap51=FAIL deepstack_off=FAIL as required");
            }
        }
        if !defects.is_empty() {
            bail!("H3_TEXT_GATE DEFECT cells outside the frozen thresholds: {defects:?}");
        }
        println!("H3_TEXT_GATE PASS cells=3 control=FAIL_AS_REQUIRED");
        Ok(())
    }

    /// N6d gate: the native vision tower on the captured `pixel_values` feeds the native decoder layers 0-49 (merged
    /// features spliced, deepstack added) for cells A, D and E; hidden_states[50] is scored under the frozen sink
    /// rule against the `C-<cell>-fp32` composition legs. Tower eager with resident weights; layers placed per
    /// `FF_H3_PLACEMENT`, eager.
    #[test]
    #[ignore = "GPU gate; set FF_H3_TEXT_ENCODER, FF_H3_S0, FF_H3_N6_LEGS, FF_H3_N6_THRESHOLDS and FF_H3_PLACEMENT"]
    fn conditioner_composition_matches_the_fp32_reference() -> Result<()> {
        use crate::backend::cuda::{ops::CommonOpsRef, weights::WeightCatalog};
        let gate = TextGate::new(50)?;
        let (hidden, image_token) = (gate.spec.hidden as usize, gate.image_token()?);
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        let mut catalog = WeightCatalog::load_with_policy(
            &gate.encoder,
            &gate.spec,
            &device,
            crate::checkpoint::LoadPolicy::VisionOnly,
        )?;
        let tower = super::super::vision_runtime::LayerNormTower::from_catalog(
            &gate.spec,
            &mut catalog,
            &device,
            CommonOpsRef::new(&device.ctx)?,
        )?;
        ensure!(
            catalog.tensors.is_empty(),
            "vision binding left unloaded owners"
        );
        device.finish_upload()?;
        let runtime =
            std::env::temp_dir().join(format!("ff-h3-composition-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let mut blas = Blas::new(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(
                true,
                "MiniMax-H3 conditioner composition gate".into(),
            )),
        )?;
        let capacity = gate
            .cells
            .iter()
            .map(|(_, c)| c.tokens.len())
            .max()
            .context("no cells")?;
        let mut media = 0;
        for (_, c) in &gate.cells {
            let input = crate::vision_input::MultimodalInput {
                input_ids: &c.tokens,
                modality_ids: &c.modality,
                pixel_values: &c.pixels,
                grids: &c.grids,
                audio: &[],
            };
            media = media.max(super::super::vision_runtime::LayerNormTower::request_bytes(
                &gate.spec,
                &input,
                c.tokens.len(),
            )?);
        }
        let mut conditioner = Conditioner::new(
            &device,
            &gate.spec,
            (50, capacity, media),
            Source::new(&gate.encoder, &gate.manifest),
            Placement::from_env()?,
        )?;
        let mut defects = Vec::new();
        for (cell, inputs) in &gate.cells {
            let input = crate::vision_input::MultimodalInput {
                input_ids: &inputs.tokens,
                modality_ids: &inputs.modality,
                pixel_values: &inputs.pixels,
                grids: &inputs.grids,
                audio: &[],
            };
            let request = tower.prepare(&device, &mut blas, &input, inputs.tokens.len(), 0)?;
            let encoded = tower.encode(&device.stream, &blas, request, None)?;
            let merged = device.stream.clone_dtoh(&encoded.features)?;
            conditioner.prepare(&blas, inputs.tokens.len())?;
            let embeds = conditioner_embeds(inputs, image_token, &gate.table, &merged, hidden)?;
            let runs = image_runs(&inputs.tokens, image_token);
            let rope = conditioner_rope(&gate.spec, inputs)?;
            let outputs = conditioner.forward(
                &device.stream,
                &blas,
                &ConditionerInput {
                    embeds: &embeds,
                    deepstack: &encoded.deepstack,
                    runs: &runs,
                    rope: &rope,
                },
                &TEXT_TAPS,
            )?;
            if !gate.score(cell, ("C", "composition"), &outputs, &TEXT_TAPS)? {
                defects.push(*cell);
            }
        }
        if !defects.is_empty() {
            bail!("H3_COMPOSITION_GATE DEFECT cells outside the frozen thresholds: {defects:?}");
        }
        println!("H3_COMPOSITION_GATE PASS cells=3");
        Ok(())
    }

    /// `[max_abs, rel_rms, cosine]` of BF16 decoder output `[channels, frames, plane]` against the F32 reference bytes of
    /// the same layout, over the listed frames (all frames when `frames` is None), accumulated in f64.
    fn video_distance(
        candidate: &[bf16],
        reference: &[u8],
        (channels, total, plane): (usize, usize, usize),
        frames: Option<&[usize]>,
    ) -> Result<[f64; 3]> {
        ensure!(
            candidate.len() == channels * total * plane && reference.len() == 4 * candidate.len(),
            "video extents differ: {} vs {} B",
            candidate.len(),
            reference.len()
        );
        let all: Vec<usize> = (0..total).collect();
        let (mut dd, mut rr, mut cc, mut cr, mut max) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for c in 0..channels {
            for &f in frames.unwrap_or(&all) {
                let base = (c * total + f) * plane;
                for (x, r) in candidate[base..base + plane]
                    .iter()
                    .zip(reference[4 * base..4 * (base + plane)].as_chunks::<4>().0)
                {
                    let (x, r) = (f64::from(x.to_f32()), f64::from(f32::from_le_bytes(*r)));
                    let d = x - r;
                    max = max.max(d.abs());
                    dd += d * d;
                    rr += r * r;
                    cc += x * x;
                    cr += x * r;
                }
            }
        }
        Ok([max, (dd / rr).sqrt(), cr / (cc * rr).sqrt()])
    }

    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW, FF_MODEL, FF_H3_CAPTURE (frozen setup directory)"]
    fn video_decoder_biased_rows_are_bitwise_invariant() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window missing"
        );
        let model = PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
        let capture =
            PathBuf::from(std::env::var_os("FF_H3_CAPTURE").context("FF_H3_CAPTURE missing")?);
        let frozen: serde_json::Value =
            serde_json::from_slice(&fs::read(capture.join("freeze.json"))?)?;
        let spec = crate::minimax_h3::VaeSpec::read(&model)?;
        let dir = model.join("vae");
        let manifest = MappedCheckpoint::open_component(&dir, "diffusion_pytorch_model")?
            .minimax_h3_vae_manifest(&spec)?;
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let runtime =
            std::env::temp_dir().join(format!("ff-h3-biased-rows-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let result = (|| -> Result<()> {
            let mut blas = Blas::new(
                &device,
                &ops,
                &runtime,
                Rc::new(Closure::new(true, "VAE biased row gate".into())),
            )?;
            let decoder = vae::VideoDecoder::new(&device, &spec, &dir, &manifest)?;
            decoder.biased_row_gate(&device, &mut blas, &frozen)
        })();
        let cleanup = fs::remove_dir_all(&runtime);
        result?;
        cleanup?;
        Ok(())
    }

    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW, FF_MODEL, FF_H3_CAPTURE (frozen setup), FF_H3_S0"]
    fn video_decoder_spatial_groups_are_bitwise_invariant() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window missing"
        );
        let path = |key| {
            std::env::var_os(key)
                .map(PathBuf::from)
                .with_context(|| format!("{key} missing"))
        };
        let model = path("FF_MODEL")?;
        let frozen: serde_json::Value =
            serde_json::from_slice(&fs::read(path("FF_H3_CAPTURE")?.join("freeze.json"))?)?;
        let spec = crate::minimax_h3::VaeSpec::read(&model)?;
        let mut inputs = Vec::new();
        for cell in ["A", "P"] {
            let map = mapped(&path("FF_H3_S0")?.join(format!("{cell}-bf16/decode.safetensors")))?;
            let file = SafeTensors::deserialize(&map)?;
            let input = file.tensor("video_decoder_input")?;
            let shape = input.shape();
            ensure!(
                input.dtype() == Dtype::F32
                    && shape.len() == 5
                    && shape[0] == 1
                    && shape[1] == spec.latent_channels,
                "group latent layout"
            );
            let values = input
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|x| f32::from_le_bytes(*x))
                .collect::<Vec<_>>();
            inputs.push((values, [shape[2], shape[3], shape[4]]));
        }
        let tiles = crate::minimax_h3::VaeTiles::released(
            &spec,
            inputs[0].1[1] * spec.spatial_ratio,
            inputs[0].1[2] * spec.spatial_ratio,
        )?;
        let dir = model.join("vae");
        let manifest = MappedCheckpoint::open_component(&dir, "diffusion_pytorch_model")?
            .minimax_h3_vae_manifest(&spec)?;
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let runtime = std::env::temp_dir().join(format!("ff-h3-group-gate-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let result = (|| -> Result<()> {
            let mut blas = Blas::new(
                &device,
                &ops,
                &runtime,
                Rc::new(Closure::new(true, "VAE spatial group gate".into())),
            )?;
            let decoder = vae::VideoDecoder::new(&device, &spec, &dir, &manifest)?;
            decoder.batch_identity_gate(&device, &mut blas, &tiles, inputs.into_iter(), &frozen)
        })();
        let cleanup = fs::remove_dir_all(&runtime);
        result?;
        cleanup?;
        Ok(())
    }

    #[test]
    fn video_tile_batch_capacity() -> Result<()> {
        let spec = crate::minimax_h3::VaeSpec {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 24,
            block_out_channels: vec![128, 256, 256, 512, 512, 1024],
            layers_per_block: 2,
            spatial_factors: vec![2, 2, 2, 2, 1, 1],
            temporal_factors: vec![1, 2, 2, 1, 1, 1],
            spatial_ratio: 16,
            temporal_ratio: 4,
            dim: 2048,
            layers: 36,
            heads: 32,
            head_dim: 64,
            registers: 4,
            ffn: 8192,
            rope_theta: 100.0,
            rope_dims: 48,
            norm_eps: 1e-5,
            clip_length: 17,
            token_drop: 3,
            groups: 32,
            encoder_eps: 1e-6,
            latents_mean: vec![0.0; 24],
            latents_std: vec![1.0; 24],
        };
        let shape = (7, 16, 16);
        assert_eq!(
            vae::TileWork::batch_bytes(&spec, shape, 1, 28)?,
            vae::TileWork::bytes(&spec, shape)
        );
        assert_eq!(
            vae::TileWork::batch_bytes(&spec, shape, 3, 28)?,
            630_933_552
        );
        assert_eq!(
            vae::TileWork::batch_bytes(&spec, shape, 28, 28)?,
            5_885_131_636
        );
        assert!(vae::TileWork::batch_bytes(&spec, shape, 0, 28).is_err());
        assert!(vae::TileWork::batch_bytes(&spec, shape, 29, 28).is_err());
        for shape in [(0, 16, 16), (7, 0, 16), (7, 16, 0)] {
            assert!(vae::TileWork::batch_bytes(&spec, shape, 1, 28).is_err());
        }
        assert!(vae::TileWork::batch_bytes(&spec, (usize::MAX, 16, 16), 1, 28).is_err());
        Ok(())
    }

    /// Setup receipts precede the first video tile in standalone and released-transformer pipeline contexts.
    #[test]
    #[ignore = "GPU; FF_MODEL, FF_H3_LORA, FF_H3_S0, FF_H3_CELL, FF_H3_PLACEMENT"]
    fn video_decoder_batch_setup_gpu() -> Result<()> {
        let path = |key: &str| {
            std::env::var_os(key)
                .map(PathBuf::from)
                .with_context(|| format!("{key} missing"))
        };
        let model = path("FF_MODEL")?;
        let s0 = path("FF_H3_S0")?;
        let cell = std::env::var("FF_H3_CELL")?;
        ensure!(
            matches!(cell.as_str(), "A" | "P"),
            "unknown setup cell {cell}"
        );
        let placement = std::env::var("FF_H3_PLACEMENT")?;
        let runtime = std::env::temp_dir().join(format!("ff-h3-vae-setup-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let result = (|| -> Result<()> {
            match placement.as_str() {
                "standalone" => {
                    let spec = crate::minimax_h3::VaeSpec::read(&model)?;
                    let dir = model.join("vae");
                    let manifest =
                        MappedCheckpoint::open_component(&dir, "diffusion_pytorch_model")?
                            .minimax_h3_vae_manifest(&spec)?;
                    let input_map = mapped(&s0.join(format!("{cell}-bf16/decode.safetensors")))?;
                    let file = SafeTensors::deserialize(&input_map)?;
                    let input = file.tensor("video_decoder_input")?;
                    let shape = input.shape();
                    ensure!(
                        input.dtype() == Dtype::F32
                            && shape.len() == 5
                            && shape[0] == 1
                            && shape[1] == spec.latent_channels,
                        "decoder setup input layout"
                    );
                    let device = Device::new(0)?;
                    let ops = Ops::new(&device.ctx)?;
                    device.finish_upload()?;
                    let mut blas = Blas::new(
                        &device,
                        &ops,
                        &runtime,
                        Rc::new(Closure::new(true, "VAE setup gate".into())),
                    )?;
                    let decoder = vae::VideoDecoder::new(&device, &spec, &dir, &manifest)?;
                    let tiles = crate::minimax_h3::VaeTiles::released(
                        &spec,
                        shape[3] * spec.spatial_ratio,
                        shape[4] * spec.spatial_ratio,
                    )?;
                    decoder.batch_setup(
                        &device,
                        &mut blas,
                        &tiles,
                        (shape[2], shape[3], shape[4]),
                    )?;
                }
                "pipeline" => {
                    let request = seeded_request(&s0, &cell, 42)?;
                    let result =
                        pipeline::decoder_setup(&model, &path("FF_H3_LORA")?, &request, &runtime);
                    match result {
                        Err(error) if error.to_string() == "H3_VAE_SETUP_COMPLETE" => (),
                        Err(error) => return Err(error),
                        Ok(_) => bail!("pipeline setup probe ran past the decoder boundary"),
                    }
                }
                _ => bail!("unknown setup placement {placement}"),
            }
            Ok(())
        })();
        let cleanup = fs::remove_dir_all(&runtime);
        result?;
        cleanup?;
        println!(
            "H3_VAE_BATCH_SETUP_GATE PASS cell={cell} placement={placement} numerical_candidate_executed=false model_cells_scored=0"
        );
        Ok(())
    }

    /// N7a gate: the native video VAE decoder on the Stage 0 captured decoder input of cells A and P against the CUDA
    /// FP32 replay, whole video and the temporally cross-faded frames, under the frozen thresholds-n7n8.json bounds;
    /// the engine's BF16 cast set equals the reference load's dtype dump; tiling off must fail.
    #[test]
    #[ignore = "requires FF_MODEL, FF_H3_S0, FF_H3_N7_THRESHOLDS, FF_H3_VAE_DTYPES and the GPU"]
    fn video_decoder_matches_the_fp32_replay() -> Result<()> {
        use super::vae::{LatentVideo, VideoDecoder, bf16_parameter};
        let var = |key: &str| -> Result<PathBuf> {
            Ok(PathBuf::from(
                std::env::var_os(key).with_context(|| format!("{key} missing"))?,
            ))
        };
        let (model, s0) = (var("FF_MODEL")?, var("FF_H3_S0")?);
        let frozen: serde_json::Value =
            serde_json::from_slice(&fs::read(var("FF_H3_N7_THRESHOLDS")?)?)?;
        let dtypes: serde_json::Value =
            serde_json::from_slice(&fs::read(var("FF_H3_VAE_DTYPES")?)?)?;
        let spec = crate::minimax_h3::VaeSpec::read(&model)?;
        let vae_dir = model.join("vae");
        let checkpoint = MappedCheckpoint::open_component(&vae_dir, "diffusion_pytorch_model")?;
        let manifest = checkpoint.minimax_h3_vae_manifest(&spec)?;
        let dumped = dtypes["vae"]["parameters"]
            .as_object()
            .context("dtype dump lacks vae parameters")?;
        ensure!(
            dumped.len() == manifest.entries.len(),
            "dtype dump lists {} parameters, the manifest {}",
            dumped.len(),
            manifest.entries.len()
        );
        for name in manifest.entries.keys() {
            let reference = dumped
                .get(name)
                .and_then(|v| v.as_str())
                .with_context(|| format!("dtype dump lacks {name}"))?;
            ensure!(
                bf16_parameter(name) == (reference == "torch.bfloat16"),
                "{name}: engine BF16 {} vs reference {reference}",
                bf16_parameter(name)
            );
        }
        println!("H3_VAE_DTYPE_MAP PASS parameters={}", dumped.len());
        let device = Device::new(0)?;
        let s = &device.stream;
        let ops = Ops::new(&device.ctx)?;
        let runtime = std::env::temp_dir().join(format!("ff-h3-vae-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let mut blas = Blas::new(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(true, "MiniMax-H3 video decoder gate".into())),
        )?;
        let decoder = VideoDecoder::new(&device, &spec, &vae_dir, &manifest)?;
        let clip = spec.chunk_latents() + spec.overlap_latents();
        let r = spec.spatial_ratio;
        let decode = |blas: &mut Blas,
                      cell: &str,
                      tiles: &crate::minimax_h3::VaeTiles|
         -> Result<(Vec<bf16>, usize, usize)> {
            let map = mapped(&s0.join(format!("{cell}-bf16")).join("decode.safetensors"))?;
            let file = SafeTensors::deserialize(&map)?;
            let input = file.tensor("video_decoder_input")?;
            ensure!(
                input.dtype() == Dtype::F32,
                "decoder input dtype {:?}",
                input.dtype()
            );
            let shape = input.shape().to_vec();
            ensure!(
                shape.len() == 5 && shape[0] == 1 && shape[1] == spec.latent_channels,
                "decoder input shape {shape:?}"
            );
            let (frames, height, width) = (shape[2], shape[3], shape[4]);
            let mut work = decoder.work(&device, blas, tiles, (frames, height, width))?;
            let z = s.clone_htod(
                &input
                    .data()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| f32::from_le_bytes(*v))
                    .collect::<Vec<_>>(),
            )?;
            let plan = crate::minimax_h3::DecodePlan::new(&spec, frames)?;
            let (lead, pre) = (
                spec.chunk_latents() * spec.temporal_ratio,
                spec.frame_pre_padding(),
            );
            let total = plan.chunks * (lead - pre) + clip * spec.temporal_ratio - lead - pre;
            let mut out =
                s.alloc_zeros::<bf16>(spec.out_channels * total * height * r * width * r)?;
            decoder.decode(
                s,
                blas,
                &mut work,
                tiles,
                LatentVideo {
                    data: &z,
                    frames,
                    height,
                    width,
                },
                &mut out,
            )?;
            s.synchronize()?;
            Ok((s.clone_dtoh(&out)?, total, height * r * width * r))
        };
        let mut defects = Vec::new();
        let mut tiling_off = None;
        for cell in ["A", "P"] {
            let reference_map = mapped(
                &s0.join(format!("{cell}-fp32-replay"))
                    .join("replay.safetensors"),
            )?;
            let reference_file = SafeTensors::deserialize(&reference_map)?;
            let reference = reference_file.tensor("decode.video_decoder_output")?;
            ensure!(
                reference.dtype() == Dtype::F32,
                "reference dtype {:?}",
                reference.dtype()
            );
            let tiles = crate::minimax_h3::VaeTiles::released(
                &spec,
                reference.shape()[3],
                reference.shape()[4],
            )?;
            let (video, frames, plane) = decode(&mut blas, cell, &tiles)?;
            ensure!(
                frames == reference.shape()[2],
                "decoded {frames} frames, reference {}",
                reference.shape()[2]
            );
            let extents = (spec.out_channels, frames, plane);
            let whole = video_distance(&video, reference.data(), extents, None)?;
            let key = format!("{cell}.decode.video_decoder_output");
            let ok = within(whole, &frozen["cells"][&key])?;
            println!("H3_VAE cell={cell} video={whole:?} within={ok}");
            if !ok {
                defects.push(key.clone());
            }
            let fade_key = format!("{key}.crossfade_frames");
            let fade: Vec<usize> = frozen["cells"][&fade_key]["frames"]
                .as_array()
                .context("crossfade frame list missing")?
                .iter()
                .map(|v| v.as_u64().map(|v| v as usize).context("crossfade frame"))
                .collect::<Result<_>>()?;
            let faded = video_distance(&video, reference.data(), extents, Some(&fade))?;
            let ok = within(faded, &frozen["cells"][&fade_key])?;
            println!(
                "H3_VAE cell={cell} crossfade_frames={faded:?} frames={} within={ok}",
                fade.len()
            );
            if !ok {
                defects.push(fade_key);
            }
            if cell == "A" {
                let whole_frame =
                    crate::minimax_h3::VaeTiles::whole(reference.shape()[3], reference.shape()[4]);
                let (untiled, _, _) = decode(&mut blas, cell, &whole_frame)?;
                let d = video_distance(&untiled, reference.data(), extents, None)?;
                let failed = !within(d, &frozen["cells"][&key])?;
                println!("H3_VAE tiling-off control cell=A video={d:?} fails={failed}");
                tiling_off = Some(failed);
            }
        }
        ensure!(
            tiling_off == Some(true),
            "tiling-off control passed the A bound"
        );
        if !defects.is_empty() {
            bail!("H3_VAE_DECODE_GATE DEFECT cells outside the frozen thresholds: {defects:?}");
        }
        println!("H3_VAE_DECODE_GATE PASS cells=2 control=FAIL_AS_REQUIRED");
        Ok(())
    }

    /// N7b gate: the native keyframe encoder on the Stage 0 captured k0 pixels and posterior noise of cells A and P
    /// against the captured CUDA FP32 posterior mean, std and normalised latents, under the frozen thresholds-n7n8.json
    /// bounds; encoder tiling off must fail the latents bound.
    #[test]
    #[ignore = "requires FF_MODEL, FF_H3_S0, FF_H3_N7_THRESHOLDS and the GPU"]
    fn keyframe_encoder_matches_the_fp32_capture() -> Result<()> {
        use super::vae::{EncodeWork, KeyframeEncoder};
        let var = |key: &str| -> Result<PathBuf> {
            Ok(PathBuf::from(
                std::env::var_os(key).with_context(|| format!("{key} missing"))?,
            ))
        };
        let (model, s0) = (var("FF_MODEL")?, var("FF_H3_S0")?);
        let frozen: serde_json::Value =
            serde_json::from_slice(&fs::read(var("FF_H3_N7_THRESHOLDS")?)?)?;
        let spec = crate::minimax_h3::VaeSpec::read(&model)?;
        let vae_dir = model.join("vae");
        let checkpoint = MappedCheckpoint::open_component(&vae_dir, "diffusion_pytorch_model")?;
        let manifest = checkpoint.minimax_h3_vae_manifest(&spec)?;
        let device = Device::new(0)?;
        let s = &device.stream;
        let ops = Ops::new(&device.ctx)?;
        let runtime = std::env::temp_dir().join(format!("ff-h3-vae-enc-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let blas = Blas::new(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(
                true,
                "MiniMax-H3 keyframe encoder gate".into(),
            )),
        )?;
        let encoder = KeyframeEncoder::new(&device, &spec, &vae_dir, &manifest)?;
        let f32s = |bytes: &[u8]| -> Vec<f32> {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect()
        };
        let keys = ["posterior_mean", "posterior_std", "latents"];
        let mut defects = Vec::new();
        let mut tiling_off = None;
        for cell in ["A", "P"] {
            let map = mapped(
                &s0.join(format!("{cell}-bf16"))
                    .join("keyframes.safetensors"),
            )?;
            let file = SafeTensors::deserialize(&map)?;
            let pixels = file.tensor("k0.pixels")?;
            let shape = pixels.shape().to_vec();
            ensure!(
                pixels.dtype() == Dtype::U8
                    && shape.len() == 5
                    && shape[..3] == [1, spec.in_channels, 1],
                "keyframe pixels {:?} {shape:?}",
                pixels.dtype()
            );
            let (height, width) = (shape[3], shape[4]);
            let eps = file.tensor("k0.posterior_eps")?;
            ensure!(
                eps.dtype() == Dtype::F32,
                "posterior noise dtype {:?}",
                eps.dtype()
            );
            let frame = s.clone_htod(pixels.data())?;
            let eps = s.clone_htod(&f32s(eps.data()))?;
            let encode = |tiles: &crate::minimax_h3::VaeTiles| -> Result<Vec<Vec<f32>>> {
                let mut work = EncodeWork::new(s, &encoder, (tiles.height, tiles.width))?;
                let out =
                    encoder.encode((s, &blas), &mut work, tiles, (&frame, height, width), &eps)?;
                s.synchronize()?;
                out.iter().map(|b| Ok(s.clone_dtoh(b)?)).collect()
            };
            let tiles = crate::minimax_h3::VaeTiles::released(&spec, height, width)?;
            let native = encode(&tiles)?;
            let as64 = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
            for (key, candidate) in keys.iter().zip(&native) {
                let reference = file.tensor(&format!("k0.{key}"))?;
                ensure!(
                    reference.dtype() == Dtype::F32
                        && reference.data().len() == 4 * candidate.len(),
                    "reference k0.{key} {:?} {:?}",
                    reference.dtype(),
                    reference.shape()
                );
                let d = distance(&as64(candidate), &as64(&f32s(reference.data())));
                let name = format!("{cell}.keyframe.k0.{key}");
                let ok = within(d, &frozen["cells"][&name])?;
                println!("H3_VAE_KEYFRAME cell={cell} {key}={d:?} within={ok}");
                if !ok {
                    defects.push(name);
                }
            }
            if cell == "A" {
                let untiled = encode(&crate::minimax_h3::VaeTiles::whole(height, width))?;
                let reference = f32s(file.tensor("k0.latents")?.data());
                let d = distance(&as64(&untiled[2]), &as64(&reference));
                let failed = !within(d, &frozen["cells"]["A.keyframe.k0.latents"])?;
                println!("H3_VAE_KEYFRAME tiling-off control cell=A latents={d:?} fails={failed}");
                tiling_off = Some(failed);
            }
        }
        ensure!(
            tiling_off == Some(true),
            "encoder tiling-off control passed the A latents bound"
        );
        if !defects.is_empty() {
            bail!("H3_VAE_KEYFRAME_GATE DEFECT cells outside the frozen thresholds: {defects:?}");
        }
        println!("H3_VAE_KEYFRAME_GATE PASS cells=2 control=FAIL_AS_REQUIRED");
        Ok(())
    }

    /// N8a gate: the native audio VAE decoder on the Stage 0 captured audio decoder input of cells A and P against the
    /// CUDA FP32 replay under the frozen thresholds-n7n8.json bounds; activations without their FIR resampling must
    /// fail.
    #[test]
    #[ignore = "requires FF_MODEL, FF_H3_S0, FF_H3_N7_THRESHOLDS and the GPU"]
    fn audio_decoder_matches_the_fp32_replay() -> Result<()> {
        use super::audio::AudioDecoder;
        let var = |key: &str| -> Result<PathBuf> {
            Ok(PathBuf::from(
                std::env::var_os(key).with_context(|| format!("{key} missing"))?,
            ))
        };
        let (model, s0) = (var("FF_MODEL")?, var("FF_H3_S0")?);
        let frozen: serde_json::Value =
            serde_json::from_slice(&fs::read(var("FF_H3_N7_THRESHOLDS")?)?)?;
        let spec = crate::minimax_h3::AudioVaeSpec::read(&model)?;
        let audio_dir = model.join("audio_vae");
        let checkpoint = MappedCheckpoint::open_component(&audio_dir, "diffusion_pytorch_model")?;
        let manifest = checkpoint.minimax_h3_audio_vae_manifest(&spec)?;
        let device = Device::new(0)?;
        let s = &device.stream;
        let ops = Ops::new(&device.ctx)?;
        let runtime = std::env::temp_dir().join(format!("ff-h3-audio-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let blas = Blas::new(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(true, "MiniMax-H3 audio decoder gate".into())),
        )?;
        let f32s = |bytes: &[u8]| -> Vec<f64> {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f64::from(f32::from_le_bytes(*v)))
                .collect()
        };
        let decode = |decoder: &AudioDecoder, cell: &str| -> Result<(Vec<f64>, Vec<f64>)> {
            let map = mapped(&s0.join(format!("{cell}-bf16")).join("decode.safetensors"))?;
            let file = SafeTensors::deserialize(&map)?;
            let input = file.tensor("audio_decoder_input")?;
            let shape = input.shape().to_vec();
            ensure!(
                input.dtype() == Dtype::F32 && shape.len() == 3 && shape[1] == spec.latent_channels,
                "audio decoder input {:?} {shape:?}",
                input.dtype()
            );
            let latents = s.clone_htod(
                &input
                    .data()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| f32::from_le_bytes(*v))
                    .collect::<Vec<_>>(),
            )?;
            let out = decoder.decode((s, &blas), &latents, (shape[0], shape[2]))?;
            s.synchronize()?;
            let candidate: Vec<f64> = s.clone_dtoh(&out)?.into_iter().map(f64::from).collect();
            let reference_map = mapped(
                &s0.join(format!("{cell}-fp32-replay"))
                    .join("replay.safetensors"),
            )?;
            let reference_file = SafeTensors::deserialize(&reference_map)?;
            let reference = reference_file.tensor("decode.audio_decoder_output")?;
            ensure!(
                reference.dtype() == Dtype::F32
                    && reference.shape() == [shape[0], 1, shape[2] * spec.hop()]
                    && reference.data().len() == 4 * candidate.len(),
                "reference audio {:?} {:?}",
                reference.dtype(),
                reference.shape()
            );
            Ok((candidate, f32s(reference.data())))
        };
        let decoder = AudioDecoder::new(&device, &spec, &audio_dir, &manifest)?;
        let mut defects = Vec::new();
        for cell in ["A", "P"] {
            let (candidate, reference) = decode(&decoder, cell)?;
            let d = distance(&candidate, &reference);
            let key = format!("{cell}.decode.audio_decoder_output");
            let ok = within(d, &frozen["cells"][&key])?;
            println!("H3_AUDIO cell={cell} pcm={d:?} within={ok}");
            if !ok {
                defects.push(key);
            }
        }
        let control = decoder.without_resampling();
        let (candidate, reference) = decode(&control, "A")?;
        let d = distance(&candidate, &reference);
        let failed = !within(d, &frozen["cells"]["A.decode.audio_decoder_output"])?;
        println!("H3_AUDIO fir-off control cell=A pcm={d:?} fails={failed}");
        ensure!(failed, "FIR-off control passed the A audio bound");
        if !defects.is_empty() {
            bail!("H3_AUDIO_DECODE_GATE DEFECT cells outside the frozen thresholds: {defects:?}");
        }
        println!("H3_AUDIO_DECODE_GATE PASS cells=2 control=FAIL_AS_REQUIRED");
        Ok(())
    }

    /// The recorded i2va request of Stage 0 run `<cell>-bf16`: prompt tokens, image grid and `pixel_values`, the text
    /// tags of the run's layout, the keyframe u8 pixels, and the run's sampling (8 evaluations, shifts 6 / 3). The
    /// four noise draws are generated natively on device from `generator_seed`; no recorded draws are loaded.
    fn seeded_request(
        s0: &Path,
        cell: &str,
        generator_seed: u64,
    ) -> Result<crate::minimax_h3::H3Request> {
        let run = s0.join(format!("{cell}-bf16"));
        let meta: serde_json::Value = serde_json::from_slice(&fs::read(run.join("meta.json"))?)?;
        let text_map = mapped(&run.join("text.safetensors"))?;
        let text = SafeTensors::deserialize(&text_map)?;
        let token_ids: Vec<u32> = decoded(&text, "token_ids")?
            .0
            .into_iter()
            .map(|v| v as u32)
            .collect();
        let grids = decoded(&text, "image_grid_thw")?
            .0
            .as_chunks::<3>()
            .0
            .iter()
            .map(|g| crate::vision_input::ImageGrid {
                t: g[0] as u32,
                h: g[1] as u32,
                w: g[2] as u32,
            })
            .collect();
        let f32s = |v: Vec<f64>| v.into_iter().map(|x| x as f32).collect::<Vec<_>>();
        let pixel_values = f32s(decoded(&text, "pixel_values")?.0);
        let layout_map = mapped(&run.join("layout.safetensors"))?;
        let layout = SafeTensors::deserialize(&layout_map)?;
        let (tags, _) = decoded(&layout, "token_tags")?;
        let (rows, _) = decoded(&layout, "text_indices")?;
        let text_tags = rows.iter().map(|&row| tags[row as usize] as u8).collect();
        let key_map = mapped(&run.join("keyframes.safetensors"))?;
        let keys = SafeTensors::deserialize(&key_map)?;
        let pixels = keys.tensor("k0.pixels")?;
        let shape = pixels.shape().to_vec();
        ensure!(
            pixels.dtype() == Dtype::U8 && shape.len() == 5 && shape[..3] == [1, 3, 1],
            "keyframe {shape:?}"
        );
        Ok(crate::minimax_h3::H3Request {
            token_ids,
            text_tags,
            grids,
            pixel_values,
            keyframe: pixels.data().to_vec(),
            height: shape[3],
            width: shape[4],
            frames: meta["cell"]["frames"].as_u64().context("meta frames")? as usize,
            seed: generator_seed,
            evaluations: 8,
            shifts: (6.0, 3.0),
        })
    }

    /// N9d run: `seeded_request`'s recorded presentation and keyframe with a request seed, through the production
    /// pipeline for cells A, P and B at generator seed 42, plus cell A at generator seed 43 for the seed control. B uses
    /// seed 42 so the A-vs-B media control varies only the keyframe; A at 43 isolates the seed. Writes
    /// `<FF_H3_E2E_OUT>/<cell>/native.safetensors` (denormalised decoder
    /// inputs, the BF16 video decoder output, u8 frames, PCM) for the splice gate. The media control requires A's and B's
    /// frames and PCM to differ; the seed control requires A at seeds 42 and 43 to differ. The envelope comparison is a
    /// measurement (N9-0: the x0.9 video-noise control is not detected), reported in the PR, not a gate.
    #[test]
    #[ignore = "GPU; FF_MODEL, FF_H3_LORA, FF_H3_S0, FF_H3_E2E_OUT"]
    fn end_to_end_generates_from_seeds() -> Result<()> {
        let var = |key: &str| -> Result<PathBuf> {
            Ok(PathBuf::from(
                std::env::var_os(key).with_context(|| format!("{key} missing"))?,
            ))
        };
        let (model, lora, s0, out) = (
            var("FF_MODEL")?,
            var("FF_H3_LORA")?,
            var("FF_H3_S0")?,
            var("FF_H3_E2E_OUT")?,
        );
        let runtime = std::env::temp_dir().join(format!("ff-h3-e2e-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let mut media = Vec::new();
        let mut seed43 = None;
        for (cell, seed) in [("A", 42), ("P", 42), ("B", 42), ("A", 43)] {
            let request = seeded_request(&s0, cell, seed)?;
            let generated = super::pipeline::run(&model, &lora, &request, (0, &runtime), None)?;
            let o = &generated.output;
            println!(
                "H3_E2E cell={cell} seed={seed} frames={} size={}x{} samples={} rate={} video_u8_mean={:.3}",
                o.frames,
                o.height,
                o.width,
                o.samples,
                o.sampling_rate,
                o.video.iter().map(|&v| f64::from(v)).sum::<f64>() / o.video.len() as f64
            );
            if seed == 43 && cell == "A" {
                seed43 = Some(generated.output.clone());
                continue;
            }
            let dir = out.join(cell);
            fs::create_dir_all(&dir)?;
            let mut t = crate::tensors::Tensors::default();
            t.f32(
                "video_latents".into(),
                generated.video_latents.0.clone(),
                &generated.video_latents.1,
            )?;
            t.f32(
                "audio_latents".into(),
                generated.audio_latents.0.clone(),
                &generated.audio_latents.1,
            )?;
            t.bf16(
                "video_decoded".into(),
                generated.video_decoded.0.clone(),
                &generated.video_decoded.1,
            )?;
            t.f32("audio_decoded".into(), vec![2, 1, o.samples], &o.audio)?;
            t.u8(
                "video_u8".into(),
                vec![o.frames, o.height, o.width, 3],
                &o.video,
            )?;
            t.save_new(
                &dir.join("native.safetensors"),
                [("sampling_rate".to_owned(), o.sampling_rate.to_string())].into(),
            )?;
            media.push((cell, generated.output));
        }
        fs::remove_dir_all(&runtime)?;
        let (a, b) = (&media[0].1, &media[2].1);
        ensure!(
            a.video.len() == b.video.len() && a.audio.len() == b.audio.len(),
            "A and B extents differ"
        );
        let video_diff = a
            .video
            .iter()
            .zip(&b.video)
            .map(|(&x, &y)| f64::from(x.abs_diff(y)))
            .sum::<f64>()
            / a.video.len() as f64;
        let audio_rms = (a
            .audio
            .iter()
            .zip(&b.audio)
            .map(|(&x, &y)| f64::from(x - y).powi(2))
            .sum::<f64>()
            / a.audio.len() as f64)
            .sqrt();
        let differ = a.video != b.video && a.audio != b.audio;
        println!(
            "H3_E2E media control A vs B video_mean_abs_u8={video_diff:.3} audio_rms={audio_rms:.5} differ={differ}"
        );
        ensure!(differ, "the second keyframe left the output unchanged");
        let alt = seed43.context("seed-43 run missing")?;
        let seed_differ = a.video != alt.video && a.audio != alt.audio;
        println!("H3_E2E seed control A seed42 vs seed43 differ={seed_differ}");
        ensure!(
            seed_differ,
            "generator seed 42 and 43 left the output unchanged"
        );
        println!(
            "H3_E2E_RUN PASS cells=3 media_control=DIFFER seed_control=DIFFER envelope=MEASUREMENT"
        );
        Ok(())
    }

    /// N9c: the frozen per-stream normal-distribution bounds and the planted biased-draw controls of the s9 freeze
    /// spec, applied to `philox_normal_f32` at each production count. Thresholds and planted offsets are read from
    /// `FF_H3_N9`; a candidate DEFECT is scored, not hidden. Two grid sizes and two launches are forced.
    #[test]
    #[ignore = "GPU; FF_H3_N9 (s9 freeze dir)"]
    fn native_noise_matches_the_frozen_distribution() -> Result<()> {
        fn norm_cdf(x: f64) -> f64 {
            let z = x / std::f64::consts::SQRT_2;
            let t = 1.0 / (1.0 + 0.3275911 * z.abs());
            let mut p = 1.061405429f64;
            for &c in &[-1.453152027f64, 1.421413741, -0.284496736, 0.254829592] {
                p = c + t * p;
            }
            let e = 1.0 - p * t * (-z * z).exp();
            if z >= 0.0 {
                0.5 * (1.0 + e)
            } else {
                0.5 * (1.0 - e)
            }
        }
        fn f64s(x: &[f32]) -> Vec<f64> {
            x.iter().map(|&v| f64::from(v)).collect()
        }
        fn mean_var(a: &[f64]) -> (f64, f64) {
            let n = a.len() as f64;
            let m = a.iter().sum::<f64>() / n;
            (m, a.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / n)
        }
        fn ks(a: &[f64]) -> f64 {
            let n = a.len() as f64;
            let mut s = a.to_vec();
            s.sort_by(f64::total_cmp);
            let mut d = 0.0f64;
            for (i, &v) in s.iter().enumerate() {
                let cdf = norm_cdf(v);
                d = d
                    .max((cdf - i as f64 / n).abs())
                    .max((cdf - (i as f64 + 1.0) / n).abs());
            }
            d
        }
        fn corr(a: &[f64], b: &[f64]) -> f64 {
            let (am, av) = mean_var(a);
            let (bm, bv) = mean_var(b);
            let cov = a
                .iter()
                .zip(b)
                .map(|(&x, &y)| (x - am) * (y - bm))
                .sum::<f64>()
                / a.len() as f64;
            cov / (av * bv).sqrt()
        }

        let var = |key: &str| -> Result<PathBuf> {
            Ok(PathBuf::from(
                std::env::var_os(key).with_context(|| format!("{key} missing"))?,
            ))
        };
        let n9 = var("FF_H3_N9")?;
        let add: serde_json::Value =
            serde_json::from_slice(&fs::read(n9.join("freeze-spec-n9-addendum1.json"))?)?;
        let spec: serde_json::Value =
            serde_json::from_slice(&fs::read(n9.join("freeze-spec-n9.json"))?)?;
        let rules = spec["N9c"]["gates_per_stream_on_production_count_N"]
            .as_array()
            .context("freeze-spec N9c gate rules missing")?;
        ensure!(
            rules.len() == 6,
            "Runner FAIL: freeze-spec N9c has {} gate rules, expected 6",
            rules.len()
        );
        let streams = add["streams"]
            .as_object()
            .context("addendum 1 streams missing")?;
        ensure!(
            streams.len() == 8,
            "addendum 1 declares 8 production streams, got {}",
            streams.len()
        );

        let device = Device::new(0)?;
        let s = &device.stream;
        let normal = cubin::module(&device.ctx, "sample")?.load_function("philox_normal_f32")?;
        let pairs = noise_draws(42);
        let site = |name: &str| -> Result<usize> {
            Ok(if name.ends_with("posterior_eps") {
                0
            } else if name.ends_with("anchor_noise") {
                1
            } else if name.ends_with("video_noise") {
                2
            } else if name.ends_with("audio_rows") {
                3
            } else {
                bail!("Runner FAIL: unknown draw site {name}");
            })
        };

        let mut defects = Vec::new();
        for (name, entry) in streams {
            let n = entry["N"].as_u64().context("N missing")? as usize;
            let mean_limit = entry["mean_limit"].as_f64().context("mean_limit missing")?;
            let ks_critical = entry["ks_critical"]
                .as_f64()
                .context("ks_critical missing")?;
            let delta = entry["planted_offset"]
                .as_f64()
                .context("planted_offset missing")?;
            let detected_by: Vec<&str> = entry["detected_by"]
                .as_array()
                .context("detected_by missing")?
                .iter()
                .map(|v| v.as_str().context("detected_by entry"))
                .collect::<Result<Vec<_>>>()?;
            let var_limit = 5.0 * (2.0 / n as f64).sqrt();
            let corr_limit = 5.0 / (n as f64).sqrt();
            let (seed, stream) = pairs[site(name)?];

            let d = super::draw_normals(s, &normal, (seed, stream), n, 256)?;
            let x = s.clone_dtoh(&d)?;
            ensure!(
                x.len() == n && x.iter().all(|v| v.is_finite()),
                "Runner FAIL: {name} produced {} values or non-finite draws",
                x.len()
            );
            let xf = f64s(&x);
            let (mean, vari) = mean_var(&xf);
            let ks_d = ks(&xf);
            let lag1 = corr(&xf[..n - 1], &xf[1..n]);

            // Determinism: two launches and two grid sizes bit-identical.
            let d2 = super::draw_normals(s, &normal, (seed, stream), n, 256)?;
            let d3 = super::draw_normals(s, &normal, (seed, stream), n, 1024)?;
            let (x2, x3) = (s.clone_dtoh(&d2)?, s.clone_dtoh(&d3)?);
            let deterministic = x.len() == x2.len()
                && x.len() == x3.len()
                && x.iter()
                    .zip(&x2)
                    .zip(&x3)
                    .all(|((a, b), c)| a.to_bits() == b.to_bits() && a.to_bits() == c.to_bits());

            // Seed control 42 vs 43 differ with |corr| <= 5/sqrt(N).
            let alt = super::draw_normals(s, &normal, (seed + 1, stream), n, 256)?;
            let af = f64s(&s.clone_dtoh(&alt)?);
            let seed_corr = corr(&xf, &af);
            let seed_differ = xf != af;

            // Cross-stream |corr| <= 5/sqrt(N) against a distinct counter range.
            let cross = super::draw_normals(s, &normal, (seed, stream + 4), n, 256)?;
            let cf = f64s(&s.clone_dtoh(&cross)?);
            let cross_corr = corr(&xf, &cf);

            // Planted biased-draw control: output x 0.999 + delta_s. Each frozen `detected_by` detector must
            // fire on the biased draws; a stream with an empty detected_by is "declared as having none".
            let biased: Vec<f64> = xf.iter().map(|&v| 0.999 * v + delta).collect();
            let (bm, bv) = mean_var(&biased);
            let bks = ks(&biased);
            let blag = corr(&biased[..n - 1], &biased[1..n]);
            let fires = |d: &str| match d {
                "mean" => bm.abs() > mean_limit,
                "var" => (bv - 1.0).abs() > var_limit,
                "ks" => bks > ks_critical,
                "corr" => blag.abs() > corr_limit,
                _ => false,
            };
            let detected = detected_by.is_empty() || detected_by.iter().all(|d| fires(d));

            let within = mean.abs() <= mean_limit
                && (vari - 1.0).abs() <= var_limit
                && ks_d <= ks_critical
                && lag1.abs() <= corr_limit
                && cross_corr.abs() <= corr_limit;
            let seed_ok = seed_corr.abs() <= corr_limit && seed_differ;
            let pass = within && deterministic && seed_ok && detected;
            println!(
                "H3_N9C {name} N={n} mean={mean:.6e} var={vari:.6} ks={ks_d:.6} lag1={lag1:.6} \
                 cross={cross_corr:.6} seed_corr={seed_corr:.6} deterministic={deterministic} \
                 control_mean={bm:.6e} control_ks={bks:.6} detected={detected} pass={pass}"
            );
            if !pass {
                defects.push(name.clone());
            }
        }
        if !defects.is_empty() {
            bail!("H3_N9C_GATE DEFECT streams outside frozen bounds: {defects:?}");
        }
        println!("H3_N9C_GATE PASS streams=8 controls_detected=8");
        Ok(())
    }

    /// GGUF measurement: one seeded cell A request run twice through `generate`, once with the official BF16
    /// transformer manifest and once with the community GGUF, and the frames/PCM distance printed. The two
    /// arms share the request, so only the transformer weights differ; the numbers are a measurement, not a gate.
    #[test]
    #[ignore = "FF_MODEL, FF_H3_GGUF, FF_H3_LORA, FF_H3_S0, FF_H3_E2E_OUT"]
    fn gguf_end_to_end_measurement_against_the_bf16_path() -> Result<()> {
        let var = |key: &str| -> Result<PathBuf> {
            Ok(PathBuf::from(
                std::env::var_os(key).with_context(|| format!("{key} missing"))?,
            ))
        };
        let (model, gguf, lora, s0, out) = (
            var("FF_MODEL")?,
            var("FF_H3_GGUF")?,
            var("FF_H3_LORA")?,
            var("FF_H3_S0")?,
            var("FF_H3_E2E_OUT")?,
        );
        let request = seeded_request(&s0, "A", 42)?;
        let runtime = std::env::temp_dir().join(format!("ff-h3-gguf-e2e-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let bf16 = super::generate(&model, &lora, &request, (0, &runtime), None)?;
        let quantized = super::generate(&model, &lora, &request, (0, &runtime), Some(&gguf))?;
        fs::remove_dir_all(&runtime)?;
        let (a, b) = (&bf16, &quantized);
        ensure!(
            a.frames == b.frames
                && a.height == b.height
                && a.width == b.width
                && a.video.len() == b.video.len()
                && a.samples == b.samples
                && a.audio.len() == b.audio.len(),
            "the two arms differ in extent"
        );
        ensure!(b.audio.iter().all(|v| v.is_finite()), "non-finite GGUF PCM");
        let video: Vec<f64> = a.video.iter().map(|&v| f64::from(v)).collect();
        let video_b: Vec<f64> = b.video.iter().map(|&v| f64::from(v)).collect();
        let pcm: Vec<f64> = a.audio.iter().map(|&v| f64::from(v)).collect();
        let pcm_b: Vec<f64> = b.audio.iter().map(|&v| f64::from(v)).collect();
        let video_d = distance(&video, &video_b);
        let pcm_d = distance(&pcm, &pcm_b);
        ensure!(
            a.video != b.video || a.audio != b.audio,
            "the GGUF arm left the output unchanged"
        );
        let dir = out.join("A-gguf");
        fs::create_dir_all(&dir)?;
        let mut t = crate::tensors::Tensors::default();
        t.u8(
            "video_u8".into(),
            vec![b.frames, b.height, b.width, 3],
            &b.video,
        )?;
        t.f32("audio_decoded".into(), vec![2, 1, b.samples], &b.audio)?;
        t.save_new(
            &dir.join("native.safetensors"),
            [("sampling_rate".to_owned(), b.sampling_rate.to_string())].into(),
        )?;
        println!(
            "H3_GGUF_E2E cell=A frames={} video_max_abs={:.6e} video_rel_rms={:.6e} video_cosine={:.9} pcm_max_abs={:.6e} pcm_rel_rms={:.6e} pcm_cosine={:.9}",
            b.frames, video_d[0], video_d[1], video_d[2], pcm_d[0], pcm_d[1], pcm_d[2]
        );
        println!(
            "H3_GGUF_E2E_MEASUREMENT frames={} video={video_d:?} pcm={pcm_d:?}",
            b.frames
        );
        Ok(())
    }

    /// N8b splice gate: for cells A and P, the native decoder outputs of the end-to-end run against the model
    /// definition's CUDA FP32 decoders on the same native latents (`reference.safetensors`, harness leg) under the
    /// frozen N7a / N8a bounds (video whole and cross-fade frames, audio); control: the native video with frame k
    /// replaced by frame k + 1 must leave the whole-video bound.
    #[test]
    #[ignore = "FF_H3_E2E_OUT (run and harness dumps), FF_H3_N7_THRESHOLDS"]
    fn end_to_end_splice_matches_the_definition_decoders() -> Result<()> {
        let var = |key: &str| -> Result<PathBuf> {
            Ok(PathBuf::from(
                std::env::var_os(key).with_context(|| format!("{key} missing"))?,
            ))
        };
        let out = var("FF_H3_E2E_OUT")?;
        let frozen: serde_json::Value =
            serde_json::from_slice(&fs::read(var("FF_H3_N7_THRESHOLDS")?)?)?;
        let mut defects = Vec::new();
        for cell in ["A", "P"] {
            let native_map = mapped(&out.join(cell).join("native.safetensors"))?;
            let native = SafeTensors::deserialize(&native_map)?;
            let reference_map = mapped(&out.join(cell).join("reference.safetensors"))?;
            let reference = SafeTensors::deserialize(&reference_map)?;
            let video = native.tensor("video_decoded")?;
            let want = reference.tensor("video_decoder_output")?;
            ensure!(
                video.dtype() == Dtype::BF16
                    && want.dtype() == Dtype::F32
                    && video.shape() == want.shape(),
                "splice video {:?} {:?} vs {:?} {:?}",
                video.dtype(),
                video.shape(),
                want.dtype(),
                want.shape()
            );
            let shape = video.shape().to_vec();
            let extents = (shape[1], shape[2], shape[3] * shape[4]);
            let candidate: Vec<bf16> = video
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_le_bytes(*b))
                .collect();
            let key = format!("{cell}.decode.video_decoder_output");
            let whole = video_distance(&candidate, want.data(), extents, None)?;
            let ok = within(whole, &frozen["cells"][&key])?;
            println!("H3_SPLICE cell={cell} video={whole:?} within={ok}");
            if !ok {
                defects.push(key.clone());
            }
            let fade_key = format!("{key}.crossfade_frames");
            let fade: Vec<usize> = frozen["cells"][&fade_key]["frames"]
                .as_array()
                .context("crossfade frame list missing")?
                .iter()
                .map(|v| v.as_u64().map(|v| v as usize).context("crossfade frame"))
                .collect::<Result<_>>()?;
            let faded = video_distance(&candidate, want.data(), extents, Some(&fade))?;
            let ok = within(faded, &frozen["cells"][&fade_key])?;
            println!("H3_SPLICE cell={cell} crossfade_frames={faded:?} within={ok}");
            if !ok {
                defects.push(fade_key);
            }
            let (channels, frames, plane) = extents;
            let mut shifted = candidate.clone();
            for c in 0..channels {
                for f in 0..frames {
                    let from = (c * frames + (f + 1).min(frames - 1)) * plane;
                    shifted[(c * frames + f) * plane..(c * frames + f + 1) * plane]
                        .copy_from_slice(&candidate[from..from + plane]);
                }
            }
            let mutant = video_distance(&shifted, want.data(), extents, None)?;
            let caught = !within(mutant, &frozen["cells"][&key])?;
            println!("H3_SPLICE frame-shift control cell={cell} video={mutant:?} fails={caught}");
            ensure!(caught, "frame-shift control passed the {cell} video bound");
            let pcm = native.tensor("audio_decoded")?;
            let want = reference.tensor("audio_decoder_output")?;
            ensure!(
                pcm.dtype() == Dtype::F32
                    && want.dtype() == Dtype::F32
                    && pcm.shape() == want.shape(),
                "splice audio {:?} vs {:?}",
                pcm.shape(),
                want.shape()
            );
            let as64 = |b: &[u8]| {
                b.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| f64::from(f32::from_le_bytes(*v)))
                    .collect::<Vec<_>>()
            };
            let d = distance(&as64(pcm.data()), &as64(want.data()));
            let key = format!("{cell}.decode.audio_decoder_output");
            let ok = within(d, &frozen["cells"][&key])?;
            println!("H3_SPLICE cell={cell} pcm={d:?} within={ok}");
            if !ok {
                defects.push(key);
            }
        }
        if !defects.is_empty() {
            bail!("H3_SPLICE_GATE DEFECT cells outside the frozen thresholds: {defects:?}");
        }
        println!("H3_SPLICE_GATE PASS cells=2 control=FAIL_AS_REQUIRED");
        Ok(())
    }
}
