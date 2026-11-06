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
    backend::gpu::{CudaFunction, CudaSlice, CudaStream, DevicePtr, PushKernelArg},
    canary::CanarySpec,
    checkpoint::MappedCheckpoint,
};
use anyhow::{Context, Result, ensure};
use cudarc::cublas::{CudaBlas, sys as cublas_sys};
use std::{path::Path, sync::Arc};

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
    residual: CudaFunction,
    activation: CudaFunction,
    bias: CudaFunction,
    layer_norm: CudaFunction,
}

struct Pass<'a> {
    encoder: &'a CudaSlice<f32>,
    queries: u32,
    keys: u32,
    valid: u32,
    self_attn: &'a F32Attention,
    cross_attn: &'a F32Attention,
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
        let ops = Ops {
            gather: vision.load_function("vision_row_gather_f32")?,
            residual: speech.load_function("speech_residual_f32")?,
            activation: vision.load_function("activation_f32")?,
            bias: vision.load_function("vision_bias_f32")?,
            layer_norm: common.load_function("q38_layer_norm_f32")?,
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
            blas,
            stream,
        };
        decoder.stream.synchronize()?;
        Ok(decoder)
    }

    fn linear(&self, input: &CudaSlice<f32>, rows: u32, weight: &Linear) -> Result<CudaSlice<f32>> {
        ensure!(rows > 0, "linear requires rows");
        ensure!(
            input.len() == rows as usize * weight.input as usize,
            "linear input length"
        );
        let output = self
            .stream
            .alloc_zeros(rows as usize * weight.output as usize)?;
        let alpha = 1.0f32;
        let beta = 0.0f32;
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
                output.device_ptr(&self.stream).0 as *mut core::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_32F,
                weight.output as i32,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .result()?;
            self.stream
                .launch_builder(&self.ops.bias)
                .arg(&output)
                .arg(&weight.bias)
                .arg(&output)
                .arg(&rows)
                .arg(&weight.output)
                .launch(ops::flat(output.len()))?;
        }
        Ok(output)
    }

    fn normalize(&self, input: &CudaSlice<f32>, rows: u32, norm: &Norm) -> Result<CudaSlice<f32>> {
        ensure!(
            input.len() == rows as usize * self.hidden as usize,
            "layer norm input length"
        );
        let output = self.stream.alloc_zeros(input.len())?;
        let width = self.hidden;
        unsafe {
            self.stream
                .launch_builder(&self.ops.layer_norm)
                .arg(input)
                .arg(&norm.weight)
                .arg(&norm.bias)
                .arg(&output)
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
        Ok(())
    }

    fn activate(&self, input: &CudaSlice<f32>, rows: u32, width: u32) -> Result<CudaSlice<f32>> {
        ensure!(
            input.len() == rows as usize * width as usize,
            "activation input length"
        );
        let output = self.stream.alloc_zeros(input.len())?;
        let mode = 0i32;
        unsafe {
            self.stream
                .launch_builder(&self.ops.activation)
                .arg(input)
                .arg(&output)
                .arg(&rows)
                .arg(&width)
                .arg(&mode)
                .launch(ops::flat(input.len()))?;
        }
        Ok(output)
    }

    fn gather(
        &self,
        table: &CudaSlice<f32>,
        indices: &[i32],
        in_rows: u32,
    ) -> Result<CudaSlice<f32>> {
        let rows = u32::try_from(indices.len()).context("gather rows")?;
        let width = self.hidden;
        ensure!(
            table.len() == in_rows as usize * width as usize,
            "gather table length"
        );
        let output = self.stream.alloc_zeros(indices.len() * width as usize)?;
        let index = self.stream.clone_htod(indices)?;
        unsafe {
            self.stream
                .launch_builder(&self.ops.gather)
                .arg(table)
                .arg(&index)
                .arg(&output)
                .arg(&rows)
                .arg(&in_rows)
                .arg(&width)
                .launch(ops::flat(output.len()))?;
        }
        Ok(output)
    }

    fn attend(
        &self,
        attention: &F32Attention,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        causal: bool,
        valid_keys: u32,
    ) -> Result<CudaSlice<f32>> {
        let hidden = self.hidden as usize;
        ensure!(
            hidden > 0
                && q.len().is_multiple_of(hidden)
                && k.len().is_multiple_of(hidden)
                && v.len() == k.len(),
            "attention buffer length"
        );
        let queries = u32::try_from(q.len() / hidden).context("attention queries")?;
        let keys = u32::try_from(k.len() / hidden).context("attention keys")?;
        let output = self.stream.alloc_zeros(q.len())?;
        let stride = [u64::from(HEAD_DIM), u64::from(self.hidden), 1];
        let q = view(q, &self.stream, 0, [HEADS, queries, HEAD_DIM], stride)?;
        let k = view(k, &self.stream, 0, [HEADS, keys, HEAD_DIM], stride)?;
        let v = view(v, &self.stream, 0, [HEADS, keys, HEAD_DIM], stride)?;
        let seen = view(&output, &self.stream, 0, [HEADS, queries, HEAD_DIM], stride)?;
        attention.run_cublas(
            &self.blas,
            q,
            k,
            v,
            &mut TensorMut(seen),
            F32Mask { causal, valid_keys },
        )?;
        Ok(output)
    }

    fn layer(&self, layer: &Layer, hidden: &mut CudaSlice<f32>, pass: &Pass<'_>) -> Result<()> {
        let normed = self.normalize(hidden, pass.queries, &layer.input_norm)?;
        let q = self.linear(&normed, pass.queries, &layer.self_attn.q)?;
        let k = self.linear(&normed, pass.queries, &layer.self_attn.k)?;
        let v = self.linear(&normed, pass.queries, &layer.self_attn.v)?;
        let attended = self.attend(pass.self_attn, &q, &k, &v, true, pass.queries)?;
        let delta = self.linear(&attended, pass.queries, &layer.self_attn.o)?;
        self.residual(hidden, &delta, 1.0)?;
        let normed = self.normalize(hidden, pass.queries, &layer.post_norm)?;
        let q = self.linear(&normed, pass.queries, &layer.cross_attn.q)?;
        let k = self.linear(pass.encoder, pass.keys, &layer.cross_attn.k)?;
        let v = self.linear(pass.encoder, pass.keys, &layer.cross_attn.v)?;
        let attended = self.attend(pass.cross_attn, &q, &k, &v, false, pass.valid)?;
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
        let queries = u32::try_from(tokens.len()).context("decoder sequence length")?;
        ensure!(queries > 0, "decoder sequence length is 0");
        ensure!(
            queries as usize <= MAX_POSITIONS,
            "decoder sequence length {queries} exceeds {MAX_POSITIONS}"
        );
        ensure!(
            tokens.iter().all(|&id| id < self.vocab),
            "token id >= {}",
            self.vocab
        );
        let width = self.hidden as usize;
        ensure!(
            width > 0 && encoder.len().is_multiple_of(width),
            "encoder length is not rows*{width}"
        );
        let rows = encoder.len() / width;
        let keys = u32::try_from(rows).context("encoder rows")?;
        let valid_keys = u32::try_from(valid).context("encoder valid count")?;
        ensure!(valid_keys > 0, "encoder valid count is 0");
        ensure!(
            valid_keys <= keys,
            "encoder valid count {valid_keys} exceeds rows {keys}"
        );
        ensure!(
            encoder.iter().all(|value| value.is_finite()),
            "non-finite encoder state"
        );
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
}
