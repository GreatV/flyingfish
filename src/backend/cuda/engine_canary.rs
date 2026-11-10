//! Teacher-forced Canary decoder over the shared FastConformer encoder.
use super::{
    Device,
    contracts::{TensorMut, view},
    cubin,
    engine_parakeet::ParakeetEncoder,
    flash::{F32Attention, F32Mask},
    ops,
};
use crate::{
    backend::gpu::{CudaFunction, CudaGraph, CudaSlice, CudaStream, DevicePtr, PushKernelArg},
    canary::CanarySpec,
    checkpoint::MappedCheckpoint,
};
use anyhow::{Context, Result, ensure};
use cudarc::cublas::{CudaBlas, sys as cublas_sys};
#[cfg(test)]
use std::cell::RefCell;
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};
use std::{cell::Cell, path::Path, sync::Arc};

const HEADS: u32 = 8;
const HEAD_DIM: u32 = 128;
const INTERMEDIATE: u32 = 4096;
const MAX_POSITIONS: usize = 1024;

struct Linear {
    weight: CudaSlice<f32>,
    bias: CudaSlice<f32>,
    output: u32,
    input: u32,
}

struct Norm {
    weight: CudaSlice<f32>,
    bias: CudaSlice<f32>,
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
}

struct Layer {
    input_norm: Norm,
    post_norm: Norm,
    final_norm: Norm,
    self_attn: Attention,
    cross_attn: Attention,
    fc1: Linear,
    fc2: Linear,
}

struct Ops {
    gather: CudaFunction,
    place: CudaFunction,
    residual: CudaFunction,
    activation: CudaFunction,
    bias: CudaFunction,
    layer_norm: CudaFunction,
    argmax: CudaFunction,
    finite: CudaFunction,
}

enum Mark {
    GemmEx,
    Bias,
    Norm,
    Residual,
    Activation,
    Gather,
    Place,
    Argmax,
    Finite,
    H2d,
    D2h,
    D2d,
    GraphLaunch,
    GraphCapture,
}

enum TokenSource {
    Host(u32),
    Device,
}

struct Pass<'a> {
    encoder: &'a CudaSlice<f32>,
    queries: u32,
    keys: u32,
    valid: u32,
    self_attn: &'a F32Attention,
    cross_attn: &'a F32Attention,
}

struct RowScratch {
    mix: CudaSlice<f32>,
    pos: CudaSlice<f32>,
    state: CudaSlice<f32>,
    q: CudaSlice<f32>,
    kv: CudaSlice<f32>,
    attended: CudaSlice<f32>,
    delta: CudaSlice<f32>,
    expanded: CudaSlice<f32>,
    activated: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    token: CudaSlice<i32>,
    bounds: CudaSlice<i32>,
    picked: CudaSlice<i32>,
    bound_ptr: u64,
    graph: Option<CudaGraph>,
}

struct KvCache {
    self_k: Vec<CudaSlice<f32>>,
    self_v: Vec<CudaSlice<f32>>,
    cross_k: Vec<CudaSlice<f32>>,
    cross_v: Vec<CudaSlice<f32>>,
    rows: u32,
    valid: u32,
    filled: u32,
    self_attn: F32Attention,
    cross_attn: F32Attention,
    scratch: RowScratch,
}

fn require_prompt_length(len: usize) -> Result<()> {
    ensure!(len > 0, "decoder sequence length is 0");
    ensure!(
        len <= MAX_POSITIONS,
        "decoder sequence length {len} exceeds {MAX_POSITIONS}"
    );
    Ok(())
}

fn require_step_position(filled: usize) -> Result<()> {
    ensure!(filled < MAX_POSITIONS, "decoder position {MAX_POSITIONS}");
    Ok(())
}

fn require_token_ids(tokens: &[u32], vocab: u32) -> Result<()> {
    ensure!(tokens.iter().all(|&id| id < vocab), "token id >= {vocab}");
    Ok(())
}

pub struct Canary {
    encoder: ParakeetEncoder,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    embed: CudaSlice<f32>,
    positions: CudaSlice<f32>,
    embedding_norm: Norm,
    layers: Vec<Layer>,
    norm: Norm,
    proj: Linear,
    ops: Ops,
    vocab: u32,
    hidden: u32,
    graph_on: Cell<bool>,
    #[cfg(test)]
    tally: Tally,
    #[cfg(test)]
    steps: RefCell<Vec<StepTally>>,
    #[cfg(test)]
    cross_writes: AtomicU64,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StepTally {
    pub gemm_ex: u64,
    pub bias: u64,
    pub norm: u64,
    pub residual: u64,
    pub activation: u64,
    pub gather: u64,
    pub place: u64,
    pub exclude: u64,
    pub softmax: u64,
    pub sum_parts: u64,
    pub strided: u64,
    pub argmax: u64,
    pub finite: u64,
    pub h2d: u64,
    pub d2h: u64,
    pub d2d: u64,
    pub graph_launches: u64,
    pub graph_captures: u64,
}

#[cfg(test)]
struct Tally {
    gemm_ex: Cell<u64>,
    bias: Cell<u64>,
    norm: Cell<u64>,
    residual: Cell<u64>,
    activation: Cell<u64>,
    gather: Cell<u64>,
    place: Cell<u64>,
    argmax: Cell<u64>,
    finite: Cell<u64>,
    h2d: Cell<u64>,
    d2h: Cell<u64>,
    d2d: Cell<u64>,
    graph_launches: Cell<u64>,
    graph_captures: Cell<u64>,
}

#[cfg(test)]
struct Snap {
    host: StepTally,
    self_attn: [u64; 4],
    cross_attn: [u64; 4],
}

#[cfg(test)]
impl Tally {
    fn new() -> Self {
        Self {
            gemm_ex: Cell::new(0),
            bias: Cell::new(0),
            norm: Cell::new(0),
            residual: Cell::new(0),
            activation: Cell::new(0),
            gather: Cell::new(0),
            place: Cell::new(0),
            argmax: Cell::new(0),
            finite: Cell::new(0),
            h2d: Cell::new(0),
            d2h: Cell::new(0),
            d2d: Cell::new(0),
            graph_launches: Cell::new(0),
            graph_captures: Cell::new(0),
        }
    }

    fn mark(&self, kind: Mark) {
        let cell = match kind {
            Mark::GemmEx => &self.gemm_ex,
            Mark::Bias => &self.bias,
            Mark::Norm => &self.norm,
            Mark::Residual => &self.residual,
            Mark::Activation => &self.activation,
            Mark::Gather => &self.gather,
            Mark::Place => &self.place,
            Mark::Argmax => &self.argmax,
            Mark::Finite => &self.finite,
            Mark::GraphLaunch => &self.graph_launches,
            Mark::GraphCapture => &self.graph_captures,
            Mark::H2d | Mark::D2h | Mark::D2d => return,
        };
        cell.set(cell.get() + 1);
    }

    fn add_bytes(&self, kind: Mark, n: u64) {
        let cell = match kind {
            Mark::H2d => &self.h2d,
            Mark::D2h => &self.d2h,
            Mark::D2d => &self.d2d,
            _ => return,
        };
        cell.set(cell.get() + n);
    }

    fn host(&self) -> StepTally {
        StepTally {
            gemm_ex: self.gemm_ex.get(),
            bias: self.bias.get(),
            norm: self.norm.get(),
            residual: self.residual.get(),
            activation: self.activation.get(),
            gather: self.gather.get(),
            place: self.place.get(),
            exclude: 0,
            softmax: 0,
            sum_parts: 0,
            strided: 0,
            argmax: self.argmax.get(),
            finite: self.finite.get(),
            h2d: self.h2d.get(),
            d2h: self.d2h.get(),
            d2d: self.d2d.get(),
            graph_launches: self.graph_launches.get(),
            graph_captures: self.graph_captures.get(),
        }
    }

    fn store(&self, host: StepTally) {
        self.gemm_ex.set(host.gemm_ex);
        self.bias.set(host.bias);
        self.norm.set(host.norm);
        self.residual.set(host.residual);
        self.activation.set(host.activation);
        self.gather.set(host.gather);
        self.place.set(host.place);
        self.argmax.set(host.argmax);
        self.finite.set(host.finite);
        self.h2d.set(host.h2d);
        self.d2h.set(host.d2h);
        self.d2d.set(host.d2d);
        self.graph_launches.set(host.graph_launches);
        self.graph_captures.set(host.graph_captures);
    }
}

#[cfg(test)]
impl StepTally {
    fn sub(self, earlier: Self) -> Self {
        Self {
            gemm_ex: self.gemm_ex - earlier.gemm_ex,
            bias: self.bias - earlier.bias,
            norm: self.norm - earlier.norm,
            residual: self.residual - earlier.residual,
            activation: self.activation - earlier.activation,
            gather: self.gather - earlier.gather,
            place: self.place - earlier.place,
            exclude: self.exclude - earlier.exclude,
            softmax: self.softmax - earlier.softmax,
            sum_parts: self.sum_parts - earlier.sum_parts,
            strided: self.strided - earlier.strided,
            argmax: self.argmax - earlier.argmax,
            finite: self.finite - earlier.finite,
            h2d: self.h2d - earlier.h2d,
            d2h: self.d2h - earlier.d2h,
            d2d: self.d2d - earlier.d2d,
            graph_launches: self.graph_launches - earlier.graph_launches,
            graph_captures: self.graph_captures - earlier.graph_captures,
        }
    }
}

fn tensor(
    checkpoint: &MappedCheckpoint,
    stream: &Arc<CudaStream>,
    name: &str,
    shape: &[usize],
) -> Result<CudaSlice<f32>> {
    let values = checkpoint.f32_tensor(name, shape)?;
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "non-finite published tensor {name}"
    );
    Ok(stream.clone_htod(&values)?)
}

fn linear(
    checkpoint: &MappedCheckpoint,
    stream: &Arc<CudaStream>,
    name: &str,
    output: u32,
    input: u32,
) -> Result<Linear> {
    Ok(Linear {
        weight: tensor(
            checkpoint,
            stream,
            &format!("{name}.weight"),
            &[output as usize, input as usize],
        )?,
        bias: tensor(
            checkpoint,
            stream,
            &format!("{name}.bias"),
            &[output as usize],
        )?,
        output,
        input,
    })
}

fn norm(checkpoint: &MappedCheckpoint, stream: &Arc<CudaStream>, prefix: &str) -> Result<Norm> {
    Ok(Norm {
        weight: tensor(checkpoint, stream, &format!("{prefix}.weight"), &[1024])?,
        bias: tensor(checkpoint, stream, &format!("{prefix}.bias"), &[1024])?,
    })
}

fn attention(
    checkpoint: &MappedCheckpoint,
    stream: &Arc<CudaStream>,
    prefix: &str,
) -> Result<Attention> {
    Ok(Attention {
        q: linear(checkpoint, stream, &format!("{prefix}.q_proj"), 1024, 1024)?,
        k: linear(checkpoint, stream, &format!("{prefix}.k_proj"), 1024, 1024)?,
        v: linear(checkpoint, stream, &format!("{prefix}.v_proj"), 1024, 1024)?,
        o: linear(checkpoint, stream, &format!("{prefix}.o_proj"), 1024, 1024)?,
    })
}

fn sinusoidal_rows(length: usize, channels: usize) -> Result<Vec<f32>> {
    ensure!(
        channels == 1024 && length > 0 && length <= MAX_POSITIONS,
        "decoder position shape"
    );
    let max_timescale = 10000f64.powf((channels as f64 - 2.0) / channels as f64);
    let increment = (max_timescale.ln() / (channels as f64 / 2.0 - 1.0)) as f32;
    let half = channels / 2;
    let scale = (channels as f32).sqrt();
    let mut rows = vec![0f32; length * channels];
    for position in 0..length {
        for index in 0..half {
            let phase = position as f32 * (-increment * index as f32).exp();
            let base = position * channels + index;
            rows[base] = phase.sin() / scale;
            rows[base + half] = phase.cos() / scale;
        }
    }
    Ok(rows)
}

fn attention_workspace(queries: u32, keys: u32) -> Result<usize> {
    let key_chunk = keys.div_ceil(32).max(512).next_multiple_of(4);
    let parts = keys.div_ceil(key_chunk) as usize;
    let row_bytes = (HEADS as usize)
        .checked_mul((keys as usize) * 8 + parts * (HEAD_DIM as usize) * 4)
        .context("F32 attention workspace overflow")?;
    row_bytes
        .checked_mul(queries as usize)
        .context("F32 attention workspace overflow")
}

impl Canary {
    pub fn load(directory: &Path, device: &Device) -> Result<Self> {
        let spec = CanarySpec::read(directory)?;
        ensure!(
            spec.hidden == 1024
                && spec.heads == HEADS
                && spec.decoder_layers == 8
                && spec.vocab == 16384,
            "unsupported Canary decoder geometry"
        );
        let encoder = ParakeetEncoder::load_canary(directory, device)?;
        let checkpoint = MappedCheckpoint::open(directory)?;
        let stream = Arc::clone(&device.stream);
        let blas = CudaBlas::new(Arc::clone(&stream))?;
        unsafe {
            cublas_sys::cublasSetMathMode(
                *blas.handle(),
                cublas_sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
            )
            .result()?;
        }
        let vision = cubin::module(&device.ctx, "vision_ops")?;
        let speech = cubin::module(&device.ctx, "speech_ops")?;
        let common = cubin::module(&device.ctx, "q38_common")?;
        let sample = cubin::module(&device.ctx, "sample")?;
        let ops = Ops {
            gather: vision.load_function("vision_row_gather_f32")?,
            place: vision.load_function("row_place_f32")?,
            residual: speech.load_function("speech_residual_f32")?,
            activation: vision.load_function("activation_f32")?,
            bias: vision.load_function("vision_bias_f32")?,
            layer_norm: common.load_function("q38_layer_norm_f32")?,
            argmax: sample.load_function("argmax_f32")?,
            finite: common.load_function("copy_finite_f32")?,
        };
        let mut layers = Vec::with_capacity(spec.decoder_layers as usize);
        for index in 0..spec.decoder_layers {
            let prefix = format!("model.decoder.layers.{index}");
            layers.push(Layer {
                input_norm: norm(&checkpoint, &stream, &format!("{prefix}.input_layernorm"))?,
                post_norm: norm(
                    &checkpoint,
                    &stream,
                    &format!("{prefix}.post_attention_layernorm"),
                )?,
                final_norm: norm(&checkpoint, &stream, &format!("{prefix}.final_layernorm"))?,
                self_attn: attention(&checkpoint, &stream, &format!("{prefix}.self_attn"))?,
                cross_attn: attention(&checkpoint, &stream, &format!("{prefix}.encoder_attn"))?,
                fc1: linear(
                    &checkpoint,
                    &stream,
                    &format!("{prefix}.mlp.fc1"),
                    INTERMEDIATE,
                    spec.hidden,
                )?,
                fc2: linear(
                    &checkpoint,
                    &stream,
                    &format!("{prefix}.mlp.fc2"),
                    spec.hidden,
                    INTERMEDIATE,
                )?,
            });
        }
        let decoder = Self {
            encoder,
            embed: tensor(
                &checkpoint,
                &stream,
                "model.decoder.embed_tokens.weight",
                &[spec.vocab as usize, spec.hidden as usize],
            )?,
            positions: stream.clone_htod(&sinusoidal_rows(MAX_POSITIONS, spec.hidden as usize)?)?,
            embedding_norm: norm(&checkpoint, &stream, "model.decoder.embedding_layernorm")?,
            layers,
            norm: norm(&checkpoint, &stream, "model.decoder.norm")?,
            proj: linear(&checkpoint, &stream, "proj_out", spec.vocab, spec.hidden)?,
            ops,
            vocab: spec.vocab,
            hidden: spec.hidden,
            graph_on: Cell::new(true),
            blas,
            stream,
            #[cfg(test)]
            tally: Tally::new(),
            #[cfg(test)]
            steps: RefCell::new(Vec::new()),
            #[cfg(test)]
            cross_writes: AtomicU64::new(0),
        };
        // Device::new opens a second stream, so event tracking stays on until finish_upload.
        device.finish_upload()?;
        Ok(decoder)
    }

    fn linear(&self, input: &CudaSlice<f32>, rows: u32, weight: &Linear) -> Result<CudaSlice<f32>> {
        let mut output = self
            .stream
            .alloc_zeros(rows as usize * weight.output as usize)?;
        self.linear_into(input, rows, weight, &mut output, 0)?;
        Ok(output)
    }

    fn linear_into(
        &self,
        input: &CudaSlice<f32>,
        rows: u32,
        weight: &Linear,
        output: &mut CudaSlice<f32>,
        row0: u32,
    ) -> Result<()> {
        ensure!(rows > 0, "linear requires rows");
        ensure!(
            input.len() == rows as usize * weight.input as usize,
            "linear input length"
        );
        let width = weight.output as usize;
        let start = row0 as usize * width;
        let count = rows as usize * width;
        ensure!(start + count <= output.len(), "linear output length");
        let alpha = 1.0f32;
        let beta = 0.0f32;
        let destination = output.slice(start..start + count);
        let pointer = destination.device_ptr(&self.stream).0;
        unsafe {
            cublas_sys::cublasGemmEx(
                *self.blas.handle(),
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                weight.output as i32,
                rows as i32,
                weight.input as i32,
                (&alpha as *const f32).cast(),
                weight.weight.device_ptr(&self.stream).0 as *const core::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_32F,
                weight.input as i32,
                input.device_ptr(&self.stream).0 as *const core::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_32F,
                weight.input as i32,
                (&beta as *const f32).cast(),
                pointer as *mut core::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_32F,
                weight.output as i32,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .result()?;
            self.mark(Mark::GemmEx);
            self.stream
                .launch_builder(&self.ops.bias)
                .arg(&destination)
                .arg(&weight.bias)
                .arg(&destination)
                .arg(&rows)
                .arg(&weight.output)
                .launch(ops::flat(count))?;
            self.mark(Mark::Bias);
        }
        Ok(())
    }

    fn mark(&self, kind: Mark) {
        #[cfg(test)]
        self.tally.mark(kind);
        #[cfg(not(test))]
        let _ = kind;
    }

    fn mark_bytes(&self, kind: Mark, bytes: u64) {
        #[cfg(test)]
        self.tally.add_bytes(kind, bytes);
        #[cfg(not(test))]
        let _ = (kind, bytes);
    }

    fn normalize_into(
        &self,
        input: &CudaSlice<f32>,
        rows: u32,
        norm: &Norm,
        output: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let width = self.hidden;
        ensure!(
            input.len() == rows as usize * width as usize && output.len() == input.len(),
            "layer norm length"
        );
        unsafe {
            self.stream
                .launch_builder(&self.ops.layer_norm)
                .arg(input)
                .arg(&norm.weight)
                .arg(&norm.bias)
                .arg(&*output)
                .arg(&rows)
                .arg(&1u32)
                .arg(&width)
                .arg(&u64::from(width))
                .arg(&u64::from(width))
                .arg(&1u64)
                .arg(&u64::from(width))
                .arg(&u64::from(width))
                .arg(&1u64)
                .arg(&1e-5f32)
                .launch(ops::grid(rows as usize, 256))?;
        }
        self.mark(Mark::Norm);
        Ok(())
    }

    fn normalize(&self, input: &CudaSlice<f32>, rows: u32, norm: &Norm) -> Result<CudaSlice<f32>> {
        let mut output = self.stream.alloc_zeros(input.len())?;
        self.normalize_into(input, rows, norm, &mut output)?;
        Ok(output)
    }

    fn residual(
        &self,
        hidden: &mut CudaSlice<f32>,
        delta: &CudaSlice<f32>,
        scale: f32,
    ) -> Result<()> {
        ensure!(hidden.len() == delta.len(), "residual length mismatch");
        let count = hidden.len() as u64;
        unsafe {
            self.stream
                .launch_builder(&self.ops.residual)
                .arg(&*hidden)
                .arg(delta)
                .arg(&*hidden)
                .arg(&count)
                .arg(&scale)
                .launch(ops::flat(hidden.len()))?;
        }
        self.mark(Mark::Residual);
        Ok(())
    }

    fn activate_into(
        &self,
        input: &CudaSlice<f32>,
        rows: u32,
        width: u32,
        output: &mut CudaSlice<f32>,
    ) -> Result<()> {
        ensure!(
            input.len() == rows as usize * width as usize && output.len() == input.len(),
            "activation length"
        );
        let mode = 0i32;
        unsafe {
            self.stream
                .launch_builder(&self.ops.activation)
                .arg(input)
                .arg(&*output)
                .arg(&rows)
                .arg(&width)
                .arg(&mode)
                .launch(ops::flat(input.len()))?;
        }
        self.mark(Mark::Activation);
        Ok(())
    }

    fn activate(&self, input: &CudaSlice<f32>, rows: u32, width: u32) -> Result<CudaSlice<f32>> {
        let mut output = self.stream.alloc_zeros(input.len())?;
        self.activate_into(input, rows, width, &mut output)?;
        Ok(output)
    }

    fn gather_device(
        &self,
        table: &CudaSlice<f32>,
        indices: &CudaSlice<i32>,
        rows: u32,
        in_rows: u32,
        output: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let width = self.hidden;
        ensure!(
            rows > 0
                && indices.len() >= rows as usize
                && table.len() == in_rows as usize * width as usize
                && output.len() == rows as usize * width as usize,
            "gather length"
        );
        unsafe {
            self.stream
                .launch_builder(&self.ops.gather)
                .arg(table)
                .arg(indices)
                .arg(&*output)
                .arg(&rows)
                .arg(&in_rows)
                .arg(&width)
                .launch(ops::flat(output.len()))?;
        }
        self.mark(Mark::Gather);
        Ok(())
    }

    fn gather(
        &self,
        table: &CudaSlice<f32>,
        indices: &[i32],
        in_rows: u32,
    ) -> Result<CudaSlice<f32>> {
        let rows = u32::try_from(indices.len()).context("gather rows")?;
        let mut output = self
            .stream
            .alloc_zeros(indices.len() * self.hidden as usize)?;
        let index = self.stream.clone_htod(indices)?;
        self.gather_device(table, &index, rows, in_rows, &mut output)?;
        Ok(output)
    }

    fn place_row(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        row: &CudaSlice<i32>,
        rows: u32,
    ) -> Result<()> {
        let width = self.hidden;
        ensure!(
            src.len() == width as usize && dst.len() == rows as usize * width as usize,
            "row place length"
        );
        unsafe {
            self.stream
                .launch_builder(&self.ops.place)
                .arg(src)
                .arg(&*dst)
                .arg(row)
                .arg(&width)
                .arg(&rows)
                .launch(ops::flat(width as usize))?;
        }
        self.mark(Mark::Place);
        Ok(())
    }

    fn attend_into(
        &self,
        attention: &F32Attention,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        keys_mask: (u32, F32Mask),
    ) -> Result<()> {
        let (keys, mask) = keys_mask;
        let hidden = self.hidden as usize;
        ensure!(
            hidden > 0 && q.len().is_multiple_of(hidden) && output.len() == q.len(),
            "attention query length"
        );
        let queries = u32::try_from(q.len() / hidden).context("attention queries")?;
        ensure!(
            k.len() >= keys as usize * hidden && v.len() >= keys as usize * hidden,
            "attention buffer length"
        );
        let stride = [u64::from(HEAD_DIM), u64::from(self.hidden), 1];
        let q = view(q, &self.stream, 0, [HEADS, queries, HEAD_DIM], stride)?;
        let k = view(k, &self.stream, 0, [HEADS, keys, HEAD_DIM], stride)?;
        let v = view(v, &self.stream, 0, [HEADS, keys, HEAD_DIM], stride)?;
        let seen = view(
            &*output,
            &self.stream,
            0,
            [HEADS, queries, HEAD_DIM],
            stride,
        )?;
        attention.run_cublas(&self.blas, q, k, v, &mut TensorMut(seen), mask)?;
        Ok(())
    }

    fn attend(
        &self,
        attention: &F32Attention,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        keys: u32,
        mask: F32Mask,
    ) -> Result<CudaSlice<f32>> {
        let mut output = self.stream.alloc_zeros(q.len())?;
        self.attend_into(attention, q, k, v, &mut output, (keys, mask))?;
        Ok(output)
    }

    fn layer(&self, layer: &Layer, hidden: &mut CudaSlice<f32>, pass: &Pass<'_>) -> Result<()> {
        let normed = self.normalize(hidden, pass.queries, &layer.input_norm)?;
        let q = self.linear(&normed, pass.queries, &layer.self_attn.q)?;
        let k = self.linear(&normed, pass.queries, &layer.self_attn.k)?;
        let v = self.linear(&normed, pass.queries, &layer.self_attn.v)?;
        let attended = self.attend(
            pass.self_attn,
            &q,
            &k,
            &v,
            pass.queries,
            F32Mask {
                causal: true,
                valid_keys: pass.queries,
                query_base: 0,
                window: 0,
            },
        )?;
        let delta = self.linear(&attended, pass.queries, &layer.self_attn.o)?;
        self.residual(hidden, &delta, 1.0)?;
        let normed = self.normalize(hidden, pass.queries, &layer.post_norm)?;
        let q = self.linear(&normed, pass.queries, &layer.cross_attn.q)?;
        let k = self.linear(pass.encoder, pass.keys, &layer.cross_attn.k)?;
        let v = self.linear(pass.encoder, pass.keys, &layer.cross_attn.v)?;
        let attended = self.attend(
            pass.cross_attn,
            &q,
            &k,
            &v,
            pass.keys,
            F32Mask {
                causal: false,
                valid_keys: pass.valid,
                query_base: 0,
                window: 0,
            },
        )?;
        let delta = self.linear(&attended, pass.queries, &layer.cross_attn.o)?;
        self.residual(hidden, &delta, 1.0)?;
        let normed = self.normalize(hidden, pass.queries, &layer.final_norm)?;
        let expanded = self.linear(&normed, pass.queries, &layer.fc1)?;
        let activated = self.activate(&expanded, pass.queries, layer.fc1.output)?;
        let delta = self.linear(&activated, pass.queries, &layer.fc2)?;
        self.residual(hidden, &delta, 1.0)?;
        Ok(())
    }

    /// `encoder` is `[rows, 1024]` including padded rows. Returns embedding, layer 0 and logits.
    pub fn decode(
        &self,
        encoder: &[f32],
        valid: usize,
        tokens: &[u32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        require_prompt_length(tokens.len())?;
        require_token_ids(tokens, self.vocab)?;
        let queries = u32::try_from(tokens.len()).context("decoder sequence length")?;
        let (keys, valid_keys) = self.encoder_rows(encoder, valid)?;
        let encoder_state = self.stream.clone_htod(encoder)?;
        let token_index: Vec<i32> = tokens.iter().map(|&id| id as i32).collect();
        let position_index: Vec<i32> = (0..queries as i32).collect();
        let mut hidden = self.gather(&self.embed, &token_index, self.vocab)?;
        let positions = self.gather(&self.positions, &position_index, MAX_POSITIONS as u32)?;
        self.residual(&mut hidden, &positions, 1.0)?;
        hidden = self.normalize(&hidden, queries, &self.embedding_norm)?;
        let embeddings = self.stream.clone_dtoh(&hidden)?;
        let self_attn = F32Attention::new(
            Arc::clone(&self.stream),
            [HEADS, queries, queries, HEAD_DIM, HEAD_DIM],
            attention_workspace(queries, queries)?,
        )?;
        let cross_attn = F32Attention::new(
            Arc::clone(&self.stream),
            [HEADS, queries, keys, HEAD_DIM, HEAD_DIM],
            attention_workspace(queries, keys)?,
        )?;
        let pass = Pass {
            encoder: &encoder_state,
            queries,
            keys,
            valid: valid_keys,
            self_attn: &self_attn,
            cross_attn: &cross_attn,
        };
        let mut layer0 = None;
        for (index, layer) in self.layers.iter().enumerate() {
            self.layer(layer, &mut hidden, &pass)?;
            if index == 0 {
                layer0 = Some(self.stream.clone_dtoh(&hidden)?);
            }
        }
        hidden = self.normalize(&hidden, queries, &self.norm)?;
        let logits = self.linear(&hidden, queries, &self.proj)?;
        Ok((
            embeddings,
            layer0.context("decoder layer 0 missing")?,
            self.stream.clone_dtoh(&logits)?,
        ))
    }

    /// Mel is `[frames, 128]`. `valid_frames` is the unpadded frame count.
    pub fn logits_from_mel(
        &self,
        mel: &[f32],
        valid_frames: usize,
        tokens: &[u32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let (hidden, _mask, valid) = self.encoder.encode(mel, valid_frames)?;
        self.decode(&hidden, valid, tokens)
    }

    fn encoder_rows(&self, encoder: &[f32], valid: usize) -> Result<(u32, u32)> {
        let width = self.hidden as usize;
        ensure!(
            width > 0 && encoder.len().is_multiple_of(width),
            "encoder length is not rows*{width}"
        );
        let rows = u32::try_from(encoder.len() / width).context("encoder rows")?;
        let valid_keys = u32::try_from(valid).context("encoder valid count")?;
        ensure!(valid_keys > 0, "encoder valid count is 0");
        ensure!(
            valid_keys <= rows,
            "encoder valid count {valid_keys} exceeds rows {rows}"
        );
        ensure!(
            encoder.iter().all(|value| value.is_finite()),
            "non-finite encoder state"
        );
        Ok((rows, valid_keys))
    }

    fn open_cache(&self, encoder: &[f32], valid: usize) -> Result<KvCache> {
        let (rows, valid_keys) = self.encoder_rows(encoder, valid)?;
        let mut self_k = Vec::with_capacity(self.layers.len());
        let mut self_v = Vec::with_capacity(self.layers.len());
        let mut cross_k = Vec::with_capacity(self.layers.len());
        let mut cross_v = Vec::with_capacity(self.layers.len());
        for _ in 0..self.layers.len() {
            self_k.push(
                self.stream
                    .alloc_zeros(MAX_POSITIONS * self.hidden as usize)?,
            );
            self_v.push(
                self.stream
                    .alloc_zeros(MAX_POSITIONS * self.hidden as usize)?,
            );
            cross_k.push(
                self.stream
                    .alloc_zeros(rows as usize * self.hidden as usize)?,
            );
            cross_v.push(
                self.stream
                    .alloc_zeros(rows as usize * self.hidden as usize)?,
            );
        }
        let mut cache = KvCache {
            self_k,
            self_v,
            cross_k,
            cross_v,
            rows,
            valid: valid_keys,
            filled: 0,
            self_attn: F32Attention::new(
                Arc::clone(&self.stream),
                [
                    HEADS,
                    MAX_POSITIONS as u32,
                    MAX_POSITIONS as u32,
                    HEAD_DIM,
                    HEAD_DIM,
                ],
                attention_workspace(MAX_POSITIONS as u32, MAX_POSITIONS as u32)?,
            )?,
            cross_attn: F32Attention::new(
                Arc::clone(&self.stream),
                [HEADS, MAX_POSITIONS as u32, rows, HEAD_DIM, HEAD_DIM],
                attention_workspace(MAX_POSITIONS as u32, rows)?,
            )?,
            scratch: self.row_scratch()?,
        };
        let encoder_state = self.stream.clone_htod(encoder)?;
        self.project_cross(&encoder_state, &mut cache)?;
        Ok(cache)
    }

    fn project_cross(&self, encoder: &CudaSlice<f32>, cache: &mut KvCache) -> Result<()> {
        ensure!(
            encoder.len() == cache.rows as usize * self.hidden as usize,
            "encoder length is not rows*{}",
            self.hidden
        );
        for index in 0..self.layers.len() {
            self.linear_into(
                encoder,
                cache.rows,
                &self.layers[index].cross_attn.k,
                &mut cache.cross_k[index],
                0,
            )?;
            self.linear_into(
                encoder,
                cache.rows,
                &self.layers[index].cross_attn.v,
                &mut cache.cross_v[index],
                0,
            )?;
            #[cfg(test)]
            self.cross_writes.fetch_add(2, Ordering::Relaxed);
        }
        Ok(())
    }

    fn row_scratch(&self) -> Result<RowScratch> {
        let hidden = self.hidden as usize;
        let bounds = self.stream.alloc_zeros::<i32>(2)?;
        let bound_ptr = bounds.device_ptr(&self.stream).0;
        Ok(RowScratch {
            mix: self.stream.alloc_zeros(hidden)?,
            pos: self.stream.alloc_zeros(hidden)?,
            state: self.stream.alloc_zeros(hidden)?,
            q: self.stream.alloc_zeros(hidden)?,
            kv: self.stream.alloc_zeros(hidden)?,
            attended: self.stream.alloc_zeros(hidden)?,
            delta: self.stream.alloc_zeros(hidden)?,
            expanded: self.stream.alloc_zeros(INTERMEDIATE as usize)?,
            activated: self.stream.alloc_zeros(INTERMEDIATE as usize)?,
            logits: self.stream.alloc_zeros(self.vocab as usize)?,
            token: self.stream.alloc_zeros(1)?,
            bounds,
            picked: self.stream.alloc_zeros(1)?,
            bound_ptr,
            graph: None,
        })
    }

    fn row_embed(&self, cache: &mut KvCache) -> Result<()> {
        let vocab = self.vocab;
        let positions = MAX_POSITIONS as u32;
        self.gather_device(
            &self.embed,
            &cache.scratch.token,
            1,
            vocab,
            &mut cache.scratch.mix,
        )?;
        self.gather_device(
            &self.positions,
            &cache.scratch.bounds,
            1,
            positions,
            &mut cache.scratch.pos,
        )?;
        self.residual(&mut cache.scratch.mix, &cache.scratch.pos, 1.0)?;
        self.normalize_into(
            &cache.scratch.mix,
            1,
            &self.embedding_norm,
            &mut cache.scratch.state,
        )?;
        Ok(())
    }

    fn row_self(&self, cache: &mut KvCache, index: usize) -> Result<()> {
        self.normalize_into(
            &cache.scratch.state,
            1,
            &self.layers[index].input_norm,
            &mut cache.scratch.mix,
        )?;
        self.linear_into(
            &cache.scratch.mix,
            1,
            &self.layers[index].self_attn.q,
            &mut cache.scratch.q,
            0,
        )?;
        self.linear_into(
            &cache.scratch.mix,
            1,
            &self.layers[index].self_attn.k,
            &mut cache.scratch.kv,
            0,
        )?;
        self.place_row(
            &cache.scratch.kv,
            &mut cache.self_k[index],
            &cache.scratch.bounds,
            MAX_POSITIONS as u32,
        )?;
        self.linear_into(
            &cache.scratch.mix,
            1,
            &self.layers[index].self_attn.v,
            &mut cache.scratch.kv,
            0,
        )?;
        self.place_row(
            &cache.scratch.kv,
            &mut cache.self_v[index],
            &cache.scratch.bounds,
            MAX_POSITIONS as u32,
        )?;
        let query = cache.scratch.bound_ptr;
        cache.self_attn.set_length_device(query, query + 4);
        let attended = self.attend_into(
            &cache.self_attn,
            &cache.scratch.q,
            &cache.self_k[index],
            &cache.self_v[index],
            &mut cache.scratch.attended,
            (
                MAX_POSITIONS as u32,
                F32Mask {
                    causal: true,
                    valid_keys: MAX_POSITIONS as u32,
                    query_base: 0,
                    window: 0,
                },
            ),
        );
        cache.self_attn.set_length_device(0, 0);
        attended?;
        self.linear_into(
            &cache.scratch.attended,
            1,
            &self.layers[index].self_attn.o,
            &mut cache.scratch.delta,
            0,
        )?;
        self.residual(&mut cache.scratch.state, &cache.scratch.delta, 1.0)?;
        Ok(())
    }

    fn row_cross(&self, cache: &mut KvCache, index: usize) -> Result<()> {
        self.normalize_into(
            &cache.scratch.state,
            1,
            &self.layers[index].post_norm,
            &mut cache.scratch.mix,
        )?;
        self.linear_into(
            &cache.scratch.mix,
            1,
            &self.layers[index].cross_attn.q,
            &mut cache.scratch.q,
            0,
        )?;
        self.attend_into(
            &cache.cross_attn,
            &cache.scratch.q,
            &cache.cross_k[index],
            &cache.cross_v[index],
            &mut cache.scratch.attended,
            (
                cache.rows,
                F32Mask {
                    causal: false,
                    valid_keys: cache.valid,
                    query_base: 0,
                    window: 0,
                },
            ),
        )?;
        self.linear_into(
            &cache.scratch.attended,
            1,
            &self.layers[index].cross_attn.o,
            &mut cache.scratch.delta,
            0,
        )?;
        self.residual(&mut cache.scratch.state, &cache.scratch.delta, 1.0)?;
        Ok(())
    }

    fn row_mlp(&self, cache: &mut KvCache, index: usize) -> Result<()> {
        let width = self.layers[index].fc1.output;
        self.normalize_into(
            &cache.scratch.state,
            1,
            &self.layers[index].final_norm,
            &mut cache.scratch.mix,
        )?;
        self.linear_into(
            &cache.scratch.mix,
            1,
            &self.layers[index].fc1,
            &mut cache.scratch.expanded,
            0,
        )?;
        self.activate_into(
            &cache.scratch.expanded,
            1,
            width,
            &mut cache.scratch.activated,
        )?;
        self.linear_into(
            &cache.scratch.activated,
            1,
            &self.layers[index].fc2,
            &mut cache.scratch.delta,
            0,
        )?;
        self.residual(&mut cache.scratch.state, &cache.scratch.delta, 1.0)?;
        Ok(())
    }

    fn row_head(&self, cache: &mut KvCache) -> Result<()> {
        self.normalize_into(&cache.scratch.state, 1, &self.norm, &mut cache.scratch.mix)?;
        self.linear_into(
            &cache.scratch.mix,
            1,
            &self.proj,
            &mut cache.scratch.logits,
            0,
        )?;
        Ok(())
    }

    fn row_body(&self, cache: &mut KvCache) -> Result<()> {
        self.row_embed(cache)?;
        for index in 0..self.layers.len() {
            self.row_self(cache, index)?;
            self.row_cross(cache, index)?;
            self.row_mlp(cache, index)?;
        }
        self.row_head(cache)
    }

    fn stage_bounds(&self, cache: &mut KvCache) -> Result<()> {
        let position = cache.filled;
        let filled = position
            .checked_add(1)
            .context("decoder position {MAX_POSITIONS}")?;
        ensure!(
            (filled as usize) <= MAX_POSITIONS,
            "decoder position {MAX_POSITIONS}"
        );
        self.stream
            .memcpy_htod(&[position as i32, filled as i32], &mut cache.scratch.bounds)?;
        self.mark_bytes(Mark::H2d, 8);
        Ok(())
    }

    fn stage(&self, cache: &mut KvCache, token: TokenSource) -> Result<()> {
        require_step_position(cache.filled as usize)?;
        match token {
            TokenSource::Host(id) => {
                ensure!(id < self.vocab, "token id >= {}", self.vocab);
                self.stream
                    .memcpy_htod(&[id as i32], &mut cache.scratch.token)?;
                self.mark_bytes(Mark::H2d, 4);
            }
            TokenSource::Device => {
                self.stream
                    .memcpy_dtod(&cache.scratch.picked, &mut cache.scratch.token)?;
                self.mark_bytes(Mark::D2d, 4);
            }
        }
        self.stage_bounds(cache)
    }

    fn device_argmax(&self, ptr: u64, picked: &mut CudaSlice<i32>) -> Result<u32> {
        let count = u64::from(self.vocab);
        let vocab = self.vocab as i32;
        unsafe {
            self.stream
                .launch_builder(&self.ops.finite)
                .arg(&ptr)
                .arg(&ptr)
                .arg(&count)
                .launch(ops::flat(self.vocab as usize))?;
            self.mark(Mark::Finite);
            self.stream
                .launch_builder(&self.ops.argmax)
                .arg(&ptr)
                .arg(&mut *picked)
                .arg(&vocab)
                .launch(ops::grid(1, 1024))?;
        }
        self.mark(Mark::Argmax);
        let mut id = [0i32; 1];
        self.stream
            .memcpy_dtoh(&*picked, &mut id)
            .context("device argmax")?;
        self.mark_bytes(Mark::D2h, 4);
        ensure!(id[0] >= 0, "device argmax id");
        let id = id[0] as u32;
        ensure!(id < self.vocab, "token id >= {}", self.vocab);
        Ok(id)
    }

    fn replay_row(&self, cache: &mut KvCache) -> Result<()> {
        if !self.graph_on.get() {
            return self.row_body(cache);
        }
        if cache.scratch.graph.is_some() {
            cache
                .scratch
                .graph
                .as_ref()
                .context("graph")?
                .launch()
                .context("graph launch")?;
            self.mark(Mark::GraphLaunch);
            return Ok(());
        }
        #[cfg(test)]
        let saved = self.snap(cache);
        self.row_body(cache)?;
        self.stream.synchronize().context("graph warmup")?;
        #[cfg(test)]
        self.restore(cache, &saved);
        self.stream
            .begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .context("graph begin")?;
        let ran = self.row_body(cache);
        let captured = self
            .stream
            .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
        ran.context("graph body")?;
        let graph = captured?.context("empty graph")?;
        graph.upload().context("graph upload")?;
        self.stream.synchronize().context("graph upload")?;
        cache.scratch.graph = Some(graph);
        self.mark(Mark::GraphCapture);
        Ok(())
    }

    fn cached_row(&self, cache: &mut KvCache, token: TokenSource) -> Result<u32> {
        #[cfg(test)]
        let before = self.snap(cache);
        self.stage(cache, token)?;
        self.replay_row(cache)?;
        let ptr = cache.scratch.logits.device_ptr(&self.stream).0;
        let id = self.device_argmax(ptr, &mut cache.scratch.picked)?;
        #[cfg(test)]
        self.note(cache, &before);
        #[cfg(test)]
        self.check_scan(&cache.scratch.logits, id)?;
        cache.filled += 1;
        Ok(id)
    }

    fn argmax_last(
        &self,
        logits: &CudaSlice<f32>,
        rows: u32,
        picked: &mut CudaSlice<i32>,
    ) -> Result<u32> {
        let width = self.vocab as usize;
        ensure!(
            rows > 0 && logits.len() == rows as usize * width,
            "logit length"
        );
        let row = logits.slice((rows as usize - 1) * width..rows as usize * width);
        let ptr = row.device_ptr(&self.stream).0;
        self.device_argmax(ptr, picked)
    }

    fn cached_block(&self, cache: &mut KvCache, tokens: &[u32]) -> Result<CudaSlice<f32>> {
        require_prompt_length(tokens.len())?;
        require_step_position(cache.filled as usize)?;
        require_token_ids(tokens, self.vocab)?;
        ensure!(
            cache.filled as usize + tokens.len() <= MAX_POSITIONS,
            "decoder position {MAX_POSITIONS}"
        );
        let queries = u32::try_from(tokens.len()).context("decoder sequence length")?;
        let filled_before = cache.filled;
        let token_index: Vec<i32> = tokens.iter().map(|&id| id as i32).collect();
        let position_index: Vec<i32> = (filled_before..filled_before + queries)
            .map(|position| position as i32)
            .collect();
        let mut hidden = self.gather(&self.embed, &token_index, self.vocab)?;
        let positions = self.gather(&self.positions, &position_index, MAX_POSITIONS as u32)?;
        self.residual(&mut hidden, &positions, 1.0)?;
        hidden = self.normalize(&hidden, queries, &self.embedding_norm)?;
        let filled_after = filled_before + queries;
        for index in 0..self.layers.len() {
            let normed = self.normalize(&hidden, queries, &self.layers[index].input_norm)?;
            let q = self.linear(&normed, queries, &self.layers[index].self_attn.q)?;
            self.linear_into(
                &normed,
                queries,
                &self.layers[index].self_attn.k,
                &mut cache.self_k[index],
                filled_before,
            )?;
            self.linear_into(
                &normed,
                queries,
                &self.layers[index].self_attn.v,
                &mut cache.self_v[index],
                filled_before,
            )?;
            let attended = self.attend(
                &cache.self_attn,
                &q,
                &cache.self_k[index],
                &cache.self_v[index],
                filled_after,
                F32Mask {
                    causal: true,
                    valid_keys: filled_after,
                    query_base: filled_before,
                    window: 0,
                },
            )?;
            let delta = self.linear(&attended, queries, &self.layers[index].self_attn.o)?;
            self.residual(&mut hidden, &delta, 1.0)?;
            let normed = self.normalize(&hidden, queries, &self.layers[index].post_norm)?;
            let q = self.linear(&normed, queries, &self.layers[index].cross_attn.q)?;
            let attended = self.attend(
                &cache.cross_attn,
                &q,
                &cache.cross_k[index],
                &cache.cross_v[index],
                cache.rows,
                F32Mask {
                    causal: false,
                    valid_keys: cache.valid,
                    query_base: 0,
                    window: 0,
                },
            )?;
            let delta = self.linear(&attended, queries, &self.layers[index].cross_attn.o)?;
            self.residual(&mut hidden, &delta, 1.0)?;
            let normed = self.normalize(&hidden, queries, &self.layers[index].final_norm)?;
            let expanded = self.linear(&normed, queries, &self.layers[index].fc1)?;
            let activated = self.activate(&expanded, queries, self.layers[index].fc1.output)?;
            let delta = self.linear(&activated, queries, &self.layers[index].fc2)?;
            self.residual(&mut hidden, &delta, 1.0)?;
        }
        hidden = self.normalize(&hidden, queries, &self.norm)?;
        let logits = self.linear(&hidden, queries, &self.proj)?;
        cache.filled = filled_after;
        Ok(logits)
    }

    fn last_logit_row(&self, logits: &CudaSlice<f32>, rows: u32) -> Result<Vec<f32>> {
        let width = self.vocab as usize;
        ensure!(
            rows > 0 && logits.len() == rows as usize * width,
            "logit length"
        );
        let row = logits.slice((rows as usize - 1) * width..rows as usize * width);
        let mut host = vec![0f32; width];
        self.stream.memcpy_dtoh(&row, &mut host)?;
        ensure!(
            host.iter().all(|value| value.is_finite()),
            "non-finite logits"
        );
        Ok(host)
    }

    fn argmax_row(row: &[f32]) -> Result<u32> {
        ensure!(
            row.iter().all(|value| value.is_finite()),
            "non-finite logits"
        );
        let mut best = 0usize;
        for (index, value) in row.iter().enumerate().skip(1) {
            if *value > row[best] {
                best = index;
            }
        }
        u32::try_from(best).context("argmax index")
    }

    fn greedy_loop(&self, mut cache: KvCache, prompt: &[u32]) -> Result<(Vec<u32>, Vec<f32>)> {
        self.clear_steps();
        require_prompt_length(prompt.len())?;
        require_token_ids(prompt, self.vocab)?;
        let mut ids = prompt.to_vec();
        let rows = u32::try_from(prompt.len()).context("decoder sequence length")?;
        let (mut next, prefill_logits) = if prompt.len() == 1 {
            let id = self.cached_row(&mut cache, TokenSource::Host(prompt[0]))?;
            let prefill_logits = self
                .stream
                .clone_dtoh(&cache.scratch.logits)
                .context("prefill logits")?;
            (id, prefill_logits)
        } else {
            let logits = self.cached_block(&mut cache, prompt)?;
            let prefill_logits = self.last_logit_row(&logits, rows)?;
            let id = self.argmax_last(&logits, rows, &mut cache.scratch.picked)?;
            ensure!(
                id == Self::argmax_row(&prefill_logits)?,
                "prefill device argmax {id}"
            );
            (id, prefill_logits)
        };
        loop {
            ids.push(next);
            if next == 3 {
                break;
            }
            next = self.cached_row(&mut cache, TokenSource::Device)?;
        }
        Ok((ids, prefill_logits))
    }

    fn shift_cross_k(&self, cache: &mut KvCache) -> Result<()> {
        let rows = cache.rows as usize;
        let width = self.hidden as usize;
        for keys in &mut cache.cross_k {
            let host = self.stream.clone_dtoh(keys)?;
            let mut shifted = vec![0f32; host.len()];
            if rows >= 2 {
                let count = (rows - 1) * width;
                shifted[..count].copy_from_slice(&host[width..width + count]);
            }
            self.stream.memcpy_htod(&shifted, keys)?;
        }
        Ok(())
    }

    /// One encoder call, then greedy decode. Returns prompt ids, generated ids including eos, and the prefill's last logit row.
    pub fn transcribe(
        &self,
        mel: &[f32],
        valid_frames: usize,
        prompt: &[u32],
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let (hidden, _mask, valid) = self.encoder.encode(mel, valid_frames)?;
        self.greedy(&hidden, valid, prompt)
    }

    /// Greedy decode over a recorded encoder state. Does not call the encoder.
    pub fn greedy(
        &self,
        encoder: &[f32],
        valid: usize,
        prompt: &[u32],
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let cache = self.open_cache(encoder, valid)?;
        self.greedy_loop(cache, prompt)
    }

    /// Prefill `prefill` tokens, then one recorded id at a time. Returns `[tokens.len(), vocab]` logits.
    pub fn cached_logits(
        &self,
        encoder: &[f32],
        valid: usize,
        tokens: &[u32],
        prefill: usize,
    ) -> Result<Vec<f32>> {
        ensure!(
            prefill > 0 && prefill <= tokens.len(),
            "decoder prefill length"
        );
        self.clear_steps();
        let mut cache = self.open_cache(encoder, valid)?;
        let mut logits = Vec::with_capacity(tokens.len() * self.vocab as usize);
        if prefill == 1 {
            let id = self.cached_row(&mut cache, TokenSource::Host(tokens[0]))?;
            let row = self.stream.clone_dtoh(&cache.scratch.logits)?;
            ensure!(id == Self::argmax_row(&row)?, "prefill device argmax {id}");
            logits.extend(row);
        } else {
            let first = self.cached_block(&mut cache, &tokens[..prefill])?;
            let host = self.stream.clone_dtoh(&first)?;
            let rows = u32::try_from(prefill).context("decoder sequence length")?;
            let id = self.argmax_last(&first, rows, &mut cache.scratch.picked)?;
            let start = (prefill - 1) * self.vocab as usize;
            ensure!(
                id == Self::argmax_row(&host[start..start + self.vocab as usize])?,
                "prefill device argmax {id}"
            );
            logits.extend(host);
        }
        for id in &tokens[prefill..] {
            let chosen = self.cached_row(&mut cache, TokenSource::Host(*id))?;
            let row = self.stream.clone_dtoh(&cache.scratch.logits)?;
            ensure!(chosen == Self::argmax_row(&row)?, "device argmax {chosen}");
            logits.extend(row);
        }
        Ok(logits)
    }

    /// After the prefill stores cross K, move each K row one frame later and run one forced step.
    pub fn cached_step_with_shifted_cross_k(
        &self,
        encoder: &[f32],
        valid: usize,
        tokens: &[u32],
        prefill: usize,
    ) -> Result<Vec<f32>> {
        ensure!(tokens.len() == prefill + 1, "decoder shift length");
        self.clear_steps();
        let mut cache = self.open_cache(encoder, valid)?;
        if prefill == 1 {
            self.cached_row(&mut cache, TokenSource::Host(tokens[0]))?;
        } else {
            self.cached_block(&mut cache, &tokens[..prefill])?;
        }
        self.shift_cross_k(&mut cache)?;
        self.cached_row(&mut cache, TokenSource::Host(tokens[prefill]))?;
        self.stream
            .clone_dtoh(&cache.scratch.logits)
            .context("shifted logits")
    }

    #[cfg(test)]
    pub(crate) fn set_graph(&self, on: bool) {
        self.graph_on.set(on);
    }

    fn clear_steps(&self) {
        #[cfg(test)]
        self.steps.borrow_mut().clear();
    }

    #[cfg(test)]
    fn check_scan(&self, logits: &CudaSlice<f32>, id: u32) -> Result<()> {
        let host = self.stream.clone_dtoh(logits).context("host scan")?;
        let host_id = Self::argmax_row(&host)?;
        ensure!(host_id == id, "device argmax {id} host scan {host_id}");
        Ok(())
    }

    #[cfg(test)]
    fn snap(&self, cache: &KvCache) -> Snap {
        let self_attn = cache.self_attn.attention_launches();
        let cross_attn = cache.cross_attn.attention_launches();
        let mut host = self.tally.host();
        host.strided = self_attn[0] + cross_attn[0];
        host.exclude = self_attn[1] + cross_attn[1];
        host.softmax = self_attn[2] + cross_attn[2];
        host.sum_parts = self_attn[3] + cross_attn[3];
        Snap {
            host,
            self_attn,
            cross_attn,
        }
    }

    #[cfg(test)]
    fn restore(&self, cache: &KvCache, saved: &Snap) {
        self.tally.store(saved.host);
        cache.self_attn.set_attention_launches(saved.self_attn);
        cache.cross_attn.set_attention_launches(saved.cross_attn);
    }

    #[cfg(test)]
    fn note(&self, cache: &KvCache, before: &Snap) {
        let after = self.snap(cache);
        self.steps.borrow_mut().push(after.host.sub(before.host));
    }

    #[cfg(test)]
    pub(crate) fn take_steps(&self) -> Vec<StepTally> {
        std::mem::take(&mut *self.steps.borrow_mut())
    }

    #[cfg(test)]
    pub(crate) fn encode_calls(&self) -> u64 {
        self.encoder.encode_calls()
    }

    #[cfg(test)]
    pub(crate) fn reset_encode_calls(&self) {
        self.encoder.reset_encode_calls();
    }

    #[cfg(test)]
    pub(crate) fn cross_writes(&self) -> u64 {
        self.cross_writes.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn reset_cross_writes(&self) {
        self.cross_writes.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod position_tests {
    use super::sinusoidal_rows;

    #[test]
    fn fixed_positions_are_split_and_scaled() {
        let table = sinusoidal_rows(1024, 1024).unwrap();
        assert_eq!(table.len(), 1024 * 1024);
        for column in 0..512 {
            assert_eq!(table[column], 0.0);
            assert_eq!(table[512 + column], 0.03125);
        }
        let row = &table[1024..2048];
        assert_eq!(row[0], 1f32.sin() / 32.0);
        assert_eq!(row[512], 1f32.cos() / 32.0);
        assert!(sinusoidal_rows(0, 1024).is_err());
        assert!(sinusoidal_rows(1025, 1024).is_err());
    }

    #[test]
    fn next_position_stops_at_1024() {
        assert!(super::require_prompt_length(0).is_err());
        assert!(super::require_prompt_length(1025).is_err());
        assert!(super::require_prompt_length(1).is_ok());
        assert!(super::require_prompt_length(1023).is_ok());
        assert!(super::require_prompt_length(1024).is_ok());
        assert!(super::require_step_position(1023).is_ok());
        assert!(super::require_step_position(1024).is_err());
    }
}
