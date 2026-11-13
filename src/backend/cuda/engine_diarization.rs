//! Resident F32 bidirectional encoder and subpixel frame-classification head.
use super::{
    Device, blas,
    contracts::{Tensor, TensorMut, view},
    cubin,
    flash::{F32Attention, F32Mask},
    ops,
};
use crate::{
    audio::{LogMel, LogMelBatch},
    backend::gpu::{CudaFunction, CudaSlice, CudaStream, PushKernelArg},
    checkpoint::MappedCheckpoint,
    diarization::{ChunkConfig, DiarizationSpec},
};
use anyhow::{Context, Result, ensure};
use cudarc::cublas::{CudaBlas, sys as bs};
use std::{borrow::Cow, collections::HashMap, path::Path, sync::Arc};

struct Linear {
    weight: CudaSlice<f32>,
    bias: Option<CudaSlice<f32>>,
    shape: [u32; 2],
}

struct Norm {
    weight: CudaSlice<f32>,
    bias: CudaSlice<f32>,
}

struct Layer {
    norm1: Norm,
    norm2: Norm,
    qkv: [Linear; 3],
    output: Linear,
    up: Linear,
    down: Linear,
}

struct Kernels {
    norm: CudaFunction,
    gelu: CudaFunction,
    relu: CudaFunction,
    bias: CudaFunction,
    residual: CudaFunction,
    rotary: CudaFunction,
    pad: CudaFunction,
    im2col: CudaFunction,
    pool: CudaFunction,
    gather: CudaFunction,
}

struct Workspace {
    features: CudaSlice<f32>,
    embeds: CudaSlice<f32>,
    input_embeds: CudaSlice<f32>,
    source: CudaSlice<f32>,
    hidden: CudaSlice<f32>,
    normalized: CudaSlice<f32>,
    qkv: [CudaSlice<f32>; 3],
    attention: CudaSlice<f32>,
    delta: CudaSlice<f32>,
    up: CudaSlice<f32>,
    projection: CudaSlice<f32>,
    padded: CudaSlice<f32>,
    columns: CudaSlice<f32>,
    subpixel: CudaSlice<f32>,
    classifier: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    probs: CudaSlice<f32>,
}

struct SpeakerState {
    cache: CudaSlice<f32>,
    fifo: CudaSlice<f32>,
    probs: Vec<f32>,
    cache_len: usize,
    fifo_len: usize,
    fifo_capacity: usize,
    update_period: usize,
    compressed: bool,
    initialized: bool,
}

pub struct CacheUpdate {
    pub source_embeds: Vec<f32>,
    pub source_probs: Vec<f32>,
    pub selected_indices: Vec<i32>,
    pub selected_embeds: Vec<f32>,
    pub selected_probs: Vec<f32>,
}

pub struct SpeakerSnapshot {
    pub cache: Vec<f32>,
    pub fifo: Vec<f32>,
    pub probs: Vec<f32>,
    pub cache_len: usize,
    pub fifo_len: usize,
    pub fifo_capacity: usize,
    pub update_period: usize,
    pub compressed: bool,
    pub initialized: bool,
}

pub struct Diarization {
    spec: DiarizationSpec,
    device: Device,
    stream: Arc<CudaStream>,
    initial_free: usize,
    device_budget: usize,
    blas: CudaBlas,
    kernels: Kernels,
    embedding: Linear,
    input_norm: Norm,
    layers: Vec<Layer>,
    final_norm: Norm,
    projection: Linear,
    convolution: Linear,
    dense: Linear,
    output: Linear,
    silence: CudaSlice<f32>,
    cos: CudaSlice<f32>,
    sin: CudaSlice<f32>,
    max_rows: u32,
    attention: F32Attention,
    workspace: Workspace,
    frontend: LogMel,
    states: [SpeakerState; 2],
    calls: [u64; 2],
}

fn upload(
    cp: &MappedCheckpoint,
    stream: &Arc<CudaStream>,
    name: &str,
    shape: &[usize],
) -> Result<CudaSlice<f32>> {
    let values = cp.f32_tensor(name, shape)?;
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "non-finite published tensor {name}"
    );
    Ok(stream.clone_htod(&values)?)
}

fn linear(
    cp: &MappedCheckpoint,
    stream: &Arc<CudaStream>,
    prefix: &str,
    shape: [u32; 2],
    biased: bool,
) -> Result<Linear> {
    Ok(Linear {
        weight: upload(
            cp,
            stream,
            &format!("{prefix}.weight"),
            &[shape[0] as usize, shape[1] as usize],
        )?,
        bias: biased
            .then(|| upload(cp, stream, &format!("{prefix}.bias"), &[shape[0] as usize]))
            .transpose()?,
        shape,
    })
}

fn norm(cp: &MappedCheckpoint, stream: &Arc<CudaStream>, prefix: &str, width: u32) -> Result<Norm> {
    Ok(Norm {
        weight: upload(cp, stream, &format!("{prefix}.weight"), &[width as usize])?,
        bias: upload(cp, stream, &format!("{prefix}.bias"), &[width as usize])?,
    })
}

fn matrix(
    x: &CudaSlice<f32>,
    stream: &Arc<CudaStream>,
    rows: u32,
    width: u32,
) -> Result<Tensor<f32, 2>> {
    view(x, stream, 0, [rows, width], [u64::from(width), 1])
}

fn gather_rows(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    table: &CudaSlice<f32>,
    indices: &[i32],
    in_rows: u32,
    width: u32,
    destination: &mut CudaSlice<f32>,
) -> Result<()> {
    let rows = u32::try_from(indices.len())?;
    if indices.is_empty() {
        return Ok(());
    }
    ensure!(
        (rows as usize) * (width as usize) <= destination.len()
            && (in_rows as usize) * (width as usize) <= table.len(),
        "diarization gather extent"
    );
    let index = stream.clone_htod(indices)?;
    unsafe {
        stream
            .launch_builder(kernel)
            .arg(table)
            .arg(&index)
            .arg(&*destination)
            .arg(&rows)
            .arg(&in_rows)
            .arg(&width)
            .launch(ops::flat(indices.len() * width as usize))?;
    }
    Ok(())
}

fn topk_indices(values: &[f32], count: usize) -> Vec<usize> {
    let mut order: Vec<(f32, usize)> = values
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| (value, index))
        .collect();
    order.select_nth_unstable_by(count - 1, |left, right| right.0.total_cmp(&left.0));
    order.truncate(count);
    order.into_iter().map(|(_, index)| index).collect()
}

fn boost_scores(scores: &mut [f32], frames: usize, speakers: usize, count: usize, boost: f32) {
    for speaker in 0..speakers {
        let column: Vec<f32> = (0..frames)
            .map(|row| scores[row * speakers + speaker])
            .collect();
        for row in topk_indices(&column, count) {
            scores[row * speakers + speaker] += boost;
        }
    }
}

fn select_indices(spec: &DiarizationSpec, probs: &[f32], frames: usize) -> Result<Vec<i32>> {
    let speakers = spec.speakers as usize;
    let capacity = spec.cache.frames as usize;
    let scored = frames + spec.cache.silence as usize;
    ensure!(
        probs.len() == frames * speakers && frames > capacity,
        "diarization compression extent"
    );
    let threshold = spec.cache.threshold;
    let half_log = 0.5f32.ln();
    let mut scores = vec![f32::NEG_INFINITY; frames * speakers];
    let mut positive_counts = vec![0usize; speakers];
    for row in 0..frames {
        let mut complements = vec![0.0f32; speakers];
        for speaker in 0..speakers {
            complements[speaker] = (1.0 - probs[row * speakers + speaker]).max(threshold).ln();
        }
        let complement_sum: f32 = complements.iter().sum();
        for speaker in 0..speakers {
            let probability = probs[row * speakers + speaker];
            let score =
                probability.max(threshold).ln() - complements[speaker] + complement_sum - half_log;
            let speech = probability > 0.5;
            let value = if speech { score } else { f32::NEG_INFINITY };
            scores[row * speakers + speaker] = value;
            positive_counts[speaker] += usize::from(value > 0.0);
        }
    }
    for row in 0..frames {
        for speaker in 0..speakers {
            let probability = probs[row * speakers + speaker];
            if positive_counts[speaker] >= spec.cache.positive as usize
                && probability > 0.5
                && scores[row * speakers + speaker] <= 0.0
            {
                scores[row * speakers + speaker] = f32::NEG_INFINITY;
            }
        }
    }
    for row in capacity..frames {
        for speaker in 0..speakers {
            scores[row * speakers + speaker] += spec.cache.latest_boost;
        }
    }
    boost_scores(
        &mut scores,
        frames,
        speakers,
        spec.cache.strong as usize,
        -2.0 * half_log,
    );
    boost_scores(
        &mut scores,
        frames,
        speakers,
        spec.cache.weak as usize,
        -half_log,
    );
    let sentinel = scored * speakers;
    let mut flat = vec![f32::NEG_INFINITY; sentinel];
    for row in 0..frames {
        for speaker in 0..speakers {
            flat[speaker * scored + row] = scores[row * speakers + speaker];
        }
    }
    for speaker in 0..speakers {
        flat[speaker * scored + frames] = f32::INFINITY;
    }
    let mut selected = topk_indices(&flat, capacity);
    for index in &mut selected {
        if flat[*index] == f32::NEG_INFINITY {
            *index = sentinel;
        }
    }
    selected.sort_unstable();
    Ok(selected
        .iter()
        .map(|&index| {
            if index == sentinel {
                frames as i32
            } else {
                (index % scored).min(frames) as i32
            }
        })
        .collect())
}

fn fifo_source_span(cache_len: usize, fifo_len: usize, chunk: usize) -> usize {
    cache_len + fifo_len + chunk
}

fn input_embedding_rows(max_rows: u32, max_positions: u32) -> u32 {
    max_rows.max(max_positions)
}

fn payload_key(embeds: &[f32], probs: &[f32]) -> Vec<u8> {
    let mut key = Vec::with_capacity((embeds.len() + probs.len()) * 4);
    for value in embeds.iter().chain(probs) {
        key.extend_from_slice(&value.to_le_bytes());
    }
    key
}

fn stack_features(
    features: &[f32],
    mel_offset: usize,
    rows: usize,
    mel_bins: usize,
    factor: usize,
) -> Result<Vec<f32>> {
    ensure!(
        mel_bins > 0 && factor > 0 && features.len().is_multiple_of(mel_bins),
        "diarization feature extent"
    );
    let frames = features.len() / mel_bins;
    ensure!(mel_offset <= frames, "diarization feature offset");
    let mut output = vec![0.0; rows * factor * mel_bins];
    for row in 0..rows {
        let mel = mel_offset + row * factor;
        let take = frames.saturating_sub(mel).min(factor);
        if take > 0 {
            let begin = row * factor * mel_bins;
            output[begin..begin + take * mel_bins]
                .copy_from_slice(&features[mel * mel_bins..(mel + take) * mel_bins]);
        }
    }
    Ok(output)
}

fn require_prefix_mask(mask: &[bool]) -> Result<()> {
    if let Some(hole) = mask.iter().position(|flag| !flag) {
        ensure!(
            mask[hole..].iter().all(|flag| !flag),
            "diarization mask is not a padding prefix: first hole at frame {hole}"
        );
    }
    Ok(())
}

fn step_mask(
    mask: &[bool],
    mel_offset: usize,
    rows: usize,
    cached: usize,
    factor: usize,
) -> Vec<bool> {
    let mut output = vec![true; cached];
    for row in 0..rows {
        let mel = mel_offset + row * factor;
        output.push(mel < mask.len() && mask[mel]);
    }
    output
}

impl Diarization {
    pub(crate) fn spec(&self) -> &DiarizationSpec {
        &self.spec
    }

    pub fn load(directory: &Path, device: &Device) -> Result<Self> {
        let spec = DiarizationSpec::read(directory)?;
        let frontend = LogMel::new(spec.mel)?;
        let cp = MappedCheckpoint::open(directory)?;
        let (initial_free, total) = device.ctx.mem_get_info()?;
        let device_budget = device.workspace_budget(initial_free, total, total / 16)?;
        let stream = device.stream.clone();
        let blas = CudaBlas::new(stream.clone())?;
        unsafe {
            bs::cublasSetMathMode(
                *blas.handle(),
                bs::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
            )
            .result()?;
        }
        let h = spec.hidden;
        let i = spec.intermediate;
        let c = spec.head_hidden;
        let factor = spec.factor;
        let max_rows = spec.max_encoder_rows()?;
        let common = cubin::module(&device.ctx, "q38_common")?;
        let vision = cubin::module(&device.ctx, "vision_ops")?;
        let kernels = Kernels {
            norm: common.load_function("q38_layer_norm_f32")?,
            gelu: common.load_function("q38_gelu_f32")?,
            relu: vision.load_function("activation_f32")?,
            bias: vision.load_function("vision_bias_f32")?,
            residual: cubin::module(&device.ctx, "speech_ops")?
                .load_function("speech_residual_f32")?,
            rotary: cubin::module(&device.ctx, "rope")?.load_function("partial_rotary_f32")?,
            pad: common.load_function("pad_rows_n_f32")?,
            im2col: common.load_function("conv_im2col_f32")?,
            pool: vision.load_function("sigmoid_avg_pool_f32")?,
            gather: vision.load_function("vision_row_gather_f32")?,
        };
        let embedding = linear(
            &cp,
            &stream,
            "model.audio_tower.embedder.projection",
            [h, factor * spec.mel.mel_bins],
            false,
        )?;
        let input_norm = norm(&cp, &stream, "model.audio_tower.input_layer_norm", h)?;
        let mut layers = Vec::with_capacity(spec.layers as usize);
        for index in 0..spec.layers {
            let prefix = format!("model.audio_tower.layers.{index}");
            layers.push(Layer {
                norm1: norm(&cp, &stream, &format!("{prefix}.layer_norm1"), h)?,
                norm2: norm(&cp, &stream, &format!("{prefix}.layer_norm2"), h)?,
                qkv: [
                    linear(
                        &cp,
                        &stream,
                        &format!("{prefix}.self_attn.q_proj"),
                        [h, h],
                        false,
                    )?,
                    linear(
                        &cp,
                        &stream,
                        &format!("{prefix}.self_attn.k_proj"),
                        [h, h],
                        false,
                    )?,
                    linear(
                        &cp,
                        &stream,
                        &format!("{prefix}.self_attn.v_proj"),
                        [h, h],
                        false,
                    )?,
                ],
                output: linear(
                    &cp,
                    &stream,
                    &format!("{prefix}.self_attn.o_proj"),
                    [h, h],
                    true,
                )?,
                up: linear(&cp, &stream, &format!("{prefix}.mlp.fc1"), [i, h], true)?,
                down: linear(&cp, &stream, &format!("{prefix}.mlp.fc2"), [h, i], true)?,
            });
        }
        let final_norm = norm(&cp, &stream, "model.audio_tower.layer_norm", h)?;
        let projection = linear(&cp, &stream, "model.proj", [c, h], true)?;
        let convolution = Linear {
            weight: upload(
                &cp,
                &stream,
                "model.upsampler.conv.weight",
                &[(factor * c) as usize, c as usize, 3],
            )?,
            bias: Some(upload(
                &cp,
                &stream,
                "model.upsampler.conv.bias",
                &[(factor * c) as usize],
            )?),
            shape: [factor * c, 3 * c],
        };
        let dense = linear(&cp, &stream, "classifier.dense", [c, c], true)?;
        let output = linear(
            &cp,
            &stream,
            "classifier.out_proj",
            [spec.speakers, c],
            true,
        )?;
        let silence = upload(&cp, &stream, "silence_embeds", &[h as usize])?;
        let dim = h / spec.heads;
        let mut cos = Vec::with_capacity((max_rows * dim) as usize);
        let mut sin = Vec::with_capacity((max_rows * dim) as usize);
        for position in 0..max_rows {
            for d in 0..dim {
                let inverse = 1.0f32
                    / spec
                        .rope_theta
                        .powf((2 * (d % (dim / 2))) as f32 / dim as f32);
                let angle = position as f32 * inverse;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        let cos = stream.clone_htod(&cos)?;
        let sin = stream.clone_htod(&sin)?;
        let alloc = |width: u32| stream.alloc_zeros::<f32>((max_rows as usize) * (width as usize));
        let input_rows = input_embedding_rows(max_rows, spec.max_positions);
        let workspace = Workspace {
            features: stream.alloc_zeros(
                (input_rows as usize) * (factor as usize) * (spec.mel.mel_bins as usize),
            )?,
            embeds: alloc(h)?,
            input_embeds: stream.alloc_zeros((input_rows as usize) * (h as usize))?,
            source: stream.alloc_zeros(((max_rows + 1) as usize) * (h as usize))?,
            hidden: alloc(h)?,
            normalized: alloc(h)?,
            qkv: [alloc(h)?, alloc(h)?, alloc(h)?],
            attention: alloc(h)?,
            delta: alloc(h)?,
            up: alloc(i)?,
            projection: alloc(c)?,
            padded: stream.alloc_zeros(((max_rows + 2) * c) as usize)?,
            columns: alloc(3 * c)?,
            subpixel: alloc(factor * c)?,
            classifier: alloc(factor * c)?,
            logits: alloc(factor * spec.speakers)?,
            probs: alloc(spec.speakers)?,
        };
        let geometry = [spec.heads, max_rows, max_rows, dim, dim];
        let available = device.workspace_budget(initial_free, initial_free, 0)?;
        let bytes = F32Attention::required_bytes(geometry)?.min(available);
        let attention = F32Attention::new(stream.clone(), geometry, bytes)?;
        let state = || -> Result<SpeakerState> {
            Ok(SpeakerState {
                cache: stream.alloc_zeros((spec.cache.frames as usize) * (h as usize))?,
                fifo: stream.alloc_zeros((spec.cache.fifo as usize) * (h as usize))?,
                probs: vec![0.0; (spec.cache.frames as usize) * (spec.speakers as usize)],
                cache_len: 0,
                fifo_len: 0,
                fifo_capacity: spec.cache.fifo as usize,
                update_period: spec.cache.update as usize,
                compressed: false,
                initialized: true,
            })
        };
        let states = [state()?, state()?];
        Ok(Self {
            spec,
            device: device.clone(),
            stream,
            initial_free,
            device_budget,
            blas,
            kernels,
            embedding,
            input_norm,
            layers,
            final_norm,
            projection,
            convolution,
            dense,
            output,
            silence,
            cos,
            sin,
            max_rows,
            attention,
            workspace,
            frontend,
            states,
            calls: [0; 2],
        })
    }

    fn apply_linear(
        &self,
        input: Tensor<f32, 2>,
        weight: &Linear,
        destination: Tensor<f32, 2>,
    ) -> Result<()> {
        let w = matrix(
            &weight.weight,
            &self.stream,
            weight.shape[0],
            weight.shape[1],
        )?;
        blas::linear_f32_on(&self.blas, w, input, &mut TensorMut(destination))?;
        if let Some(bias) = &weight.bias {
            unsafe {
                self.stream
                    .launch_builder(&self.kernels.bias)
                    .arg(&destination.ptr)
                    .arg(bias)
                    .arg(&destination.ptr)
                    .arg(&destination.shape[0])
                    .arg(&destination.shape[1])
                    .launch(ops::flat(
                        (destination.shape[0] as usize) * (destination.shape[1] as usize),
                    ))?;
            }
        }
        Ok(())
    }

    fn normalize(
        &self,
        input: Tensor<f32, 2>,
        weight: &Norm,
        output: Tensor<f32, 2>,
    ) -> Result<()> {
        ensure!(
            input.shape == output.shape,
            "diarization normalization extent"
        );
        let [rows, width] = input.shape;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.norm)
                .arg(&input.ptr)
                .arg(&weight.weight)
                .arg(&weight.bias)
                .arg(&output.ptr)
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
        Ok(())
    }

    fn activation(&self, values: Tensor<f32, 2>, gelu: bool) -> Result<()> {
        let [rows, width] = values.shape;
        let count = (rows as usize) * (width as usize);
        unsafe {
            if gelu {
                self.stream
                    .launch_builder(&self.kernels.gelu)
                    .arg(&values.ptr)
                    .arg(&values.ptr)
                    .arg(&rows)
                    .arg(&1u32)
                    .arg(&width)
                    .arg(&u64::from(width))
                    .arg(&u64::from(width))
                    .arg(&1u64)
                    .arg(&u64::from(width))
                    .arg(&u64::from(width))
                    .arg(&1u64)
                    .arg(&0i32)
                    .launch(ops::flat(count))?;
            } else {
                self.stream
                    .launch_builder(&self.kernels.relu)
                    .arg(&values.ptr)
                    .arg(&values.ptr)
                    .arg(&rows)
                    .arg(&width)
                    .arg(&0i32)
                    .launch(ops::flat(count))?;
            }
        }
        Ok(())
    }

    fn residual(&self, hidden: Tensor<f32, 2>, delta: Tensor<f32, 2>) -> Result<()> {
        ensure!(hidden.shape == delta.shape, "diarization residual extent");
        let count = u64::from(hidden.shape[0]) * u64::from(hidden.shape[1]);
        unsafe {
            self.stream
                .launch_builder(&self.kernels.residual)
                .arg(&hidden.ptr)
                .arg(&delta.ptr)
                .arg(&hidden.ptr)
                .arg(&count)
                .arg(&1f32)
                .launch(ops::flat(count as usize))?;
        }
        Ok(())
    }

    fn rotary(&self, values: Tensor<f32, 2>) -> Result<()> {
        let rows = values.shape[0];
        let dim = self.spec.hidden / self.spec.heads;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.rotary)
                .arg(&values.ptr)
                .arg(&self.cos)
                .arg(&self.sin)
                .arg(&(rows as i32))
                .arg(&(self.spec.heads as i32))
                .arg(&(dim as i32))
                .arg(&(dim as i32))
                .launch(ops::flat((rows as usize) * (self.spec.hidden as usize) / 2))?;
        }
        Ok(())
    }

    fn pool_probs(&self, rows: u32, mask: &[bool]) -> Result<Vec<f32>> {
        let width = self.spec.speakers;
        ensure!(
            mask.len() == rows as usize,
            "diarization probability mask extent"
        );
        unsafe {
            self.stream
                .launch_builder(&self.kernels.pool)
                .arg(&self.workspace.logits)
                .arg(&self.workspace.probs)
                .arg(&rows)
                .arg(&width)
                .launch(ops::flat((rows as usize) * (width as usize)))?;
        }
        let count = (rows as usize) * (width as usize);
        let mut probs = self
            .stream
            .clone_dtoh(&self.workspace.probs.slice(..count))?;
        for (row, valid) in mask.iter().enumerate().take(rows as usize) {
            if !valid {
                let begin = row * width as usize;
                probs[begin..begin + width as usize].fill(0.0);
            }
        }
        Ok(probs)
    }

    fn grow_input_capacity(&mut self, rows: usize) -> Result<()> {
        let hidden = self.spec.hidden as usize;
        let width = self.spec.factor as usize * self.spec.mel.mel_bins as usize;
        let current =
            (self.workspace.input_embeds.len() / hidden).min(self.workspace.features.len() / width);
        if rows <= current {
            return Ok(());
        }
        let feature_count = rows
            .checked_mul(width)
            .context("diarization input feature count overflow")?;
        let embed_count = rows
            .checked_mul(hidden)
            .context("diarization input embedding count overflow")?;
        let bytes = feature_count
            .checked_add(embed_count)
            .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
            .context("diarization input buffer bytes overflow")?;
        let available = self
            .device
            .workspace_budget(self.initial_free, self.device_budget, 0)
            .context("diarization input buffer budget")?;
        ensure!(
            bytes <= available,
            "diarization input buffers need {bytes} bytes; device budget {available} bytes"
        );
        self.workspace.features = self.stream.alloc_zeros(feature_count)?;
        self.workspace.input_embeds = self.stream.alloc_zeros(embed_count)?;
        Ok(())
    }

    fn embed_features(&mut self, features: &[f32], rows: u32, offset: usize) -> Result<()> {
        let mel_bins = self.spec.mel.mel_bins as usize;
        let width = (self.spec.factor as usize) * mel_bins;
        let count = (rows as usize) * width;
        let hidden = self.spec.hidden as usize;
        ensure!(features.len() == count, "diarization feature extent");
        ensure!(
            offset + (rows as usize) * hidden <= self.workspace.embeds.len(),
            "diarization embedding extent"
        );
        self.stream
            .memcpy_htod(features, &mut self.workspace.features.slice_mut(..count))?;
        let input = view(
            &self.workspace.features,
            &self.stream,
            0,
            [rows, width as u32],
            [width as u64, 1],
        )?;
        let destination = view(
            &self.workspace.embeds,
            &self.stream,
            offset,
            [rows, self.spec.hidden],
            [u64::from(self.spec.hidden), 1],
        )?;
        self.apply_linear(input, &self.embedding, destination)?;
        self.calls[0] += 1;
        Ok(())
    }

    fn embed_all_features(&mut self, features: &[f32], rows: u32) -> Result<()> {
        let mel_bins = self.spec.mel.mel_bins as usize;
        let width = (self.spec.factor as usize) * mel_bins;
        let hidden = self.spec.hidden as usize;
        let total_rows = rows as usize;
        let count = total_rows * width;
        ensure!(features.len() == count, "diarization feature extent");
        ensure!(
            total_rows * hidden <= self.workspace.input_embeds.len(),
            "diarization input embedding extent"
        );
        if rows == 0 {
            self.calls[0] += 1;
            return Ok(());
        }
        self.stream
            .memcpy_htod(features, &mut self.workspace.features.slice_mut(..count))?;
        let tile_rows = (self.spec.offline.frames + self.spec.offline.right) as usize;
        for tile_start in (0..total_rows).step_by(tile_rows) {
            let tile = tile_rows.min(total_rows - tile_start);
            let tile_u32 = u32::try_from(tile)?;
            let input = view(
                &self.workspace.features,
                &self.stream,
                tile_start * width,
                [tile_u32, width as u32],
                [width as u64, 1],
            )?;
            let destination = view(
                &self.workspace.input_embeds,
                &self.stream,
                tile_start * hidden,
                [tile_u32, self.spec.hidden],
                [u64::from(self.spec.hidden), 1],
            )?;
            self.apply_linear(input, &self.embedding, destination)?;
        }
        self.calls[0] += 1;
        Ok(())
    }

    fn run_embeddings(
        &mut self,
        rows: u32,
        valid: u32,
        mask: &[bool],
        update: Option<(usize, u32)>,
    ) -> Result<(Vec<f32>, Option<CacheUpdate>)> {
        self.calls[1] += 1;
        self.encode(rows, valid)?;
        self.classify(rows)?;
        let count = (rows as usize) * (self.spec.factor as usize) * (self.spec.speakers as usize);
        let logits = self
            .stream
            .clone_dtoh(&self.workspace.logits.slice(..count))?;
        let probs = self.pool_probs(rows, mask)?;
        let update = match update {
            Some((state, chunk)) => self.update_state(state, rows, chunk, mask, &probs)?,
            None => None,
        };
        Ok((logits, update))
    }

    fn update_state(
        &mut self,
        state_index: usize,
        rows: u32,
        num_chunk_frames: u32,
        mask: &[bool],
        probs: &[f32],
    ) -> Result<Option<CacheUpdate>> {
        ensure!(state_index < 2, "diarization state index");
        let hidden = self.spec.hidden as usize;
        let speakers = self.spec.speakers as usize;
        let capacity = self.spec.cache.frames as usize;
        let input_rows = rows as usize;
        ensure!(
            mask.len() == input_rows && probs.len() == input_rows * speakers,
            "diarization update extent"
        );
        let (cache_len, fifo_len, fifo_capacity, update_period, compressed, stored_probs) = {
            let state = &self.states[state_index];
            (
                state.cache_len,
                state.fifo_len,
                state.fifo_capacity,
                state.update_period,
                state.compressed,
                state.probs.clone(),
            )
        };
        let chunk = num_chunk_frames as usize;
        let fifo_rows = fifo_len + chunk;
        let fifo_end = fifo_source_span(cache_len, fifo_len, chunk);
        ensure!(
            fifo_end <= input_rows && input_rows <= self.max_rows as usize,
            "diarization state window"
        );
        let num_popped = if fifo_rows <= fifo_capacity {
            0
        } else {
            update_period.max(fifo_rows - fifo_capacity).min(fifo_rows)
        };
        let in_rows = u32::try_from(self.workspace.embeds.len() / hidden)?;
        if num_popped == 0 {
            let indices: Vec<i32> = (cache_len..cache_len + fifo_rows)
                .map(|row| row as i32)
                .collect();
            gather_rows(
                &self.stream,
                &self.kernels.gather,
                &self.workspace.embeds,
                &indices,
                in_rows,
                self.spec.hidden,
                &mut self.states[state_index].fifo,
            )?;
            self.states[state_index].fifo_len = fifo_rows;
            self.states[state_index].initialized = true;
            return Ok(None);
        }
        let mut cache_probs = if compressed {
            stored_probs[..cache_len * speakers].to_vec()
        } else {
            probs[..cache_len * speakers].to_vec()
        };
        cache_probs
            .extend_from_slice(&probs[cache_len * speakers..(cache_len + num_popped) * speakers]);
        let cache_rows = cache_len + num_popped;
        let fifo_begin = cache_len + num_popped;
        let mut result = None;
        if cache_rows > capacity {
            let source_bytes = cache_rows * hidden;
            let silence_begin = source_bytes;
            let silence_end = (cache_rows + 1) * hidden;
            ensure!(
                silence_end <= self.workspace.source.len(),
                "diarization compression extent"
            );
            self.stream.memcpy_dtod(
                &self.workspace.embeds.slice(..source_bytes),
                &mut self.workspace.source.slice_mut(..source_bytes),
            )?;
            self.stream.memcpy_dtod(
                &self.silence,
                &mut self.workspace.source.slice_mut(silence_begin..silence_end),
            )?;
            let source_embeds_full = self
                .stream
                .clone_dtoh(&self.workspace.source.slice(..silence_end))?;
            let source_embeds = source_embeds_full[..source_bytes].to_vec();
            let mut source_probs = cache_probs.clone();
            source_probs.resize((cache_rows + 1) * speakers, 0.0);
            let gather_indices = select_indices(&self.spec, &cache_probs, cache_rows)?;
            let mut selected_probs = Vec::with_capacity(capacity * speakers);
            for &index in &gather_indices {
                let row = index as usize;
                selected_probs
                    .extend_from_slice(&source_probs[row * speakers..(row + 1) * speakers]);
            }
            gather_rows(
                &self.stream,
                &self.kernels.gather,
                &self.workspace.source,
                &gather_indices,
                (cache_rows + 1) as u32,
                self.spec.hidden,
                &mut self.states[state_index].cache,
            )?;
            let selected_embeds = self
                .stream
                .clone_dtoh(&self.states[state_index].cache.slice(..capacity * hidden))?;
            let mut lookup = HashMap::new();
            for row in 0..=cache_rows {
                let key = payload_key(
                    &source_embeds_full[row * hidden..(row + 1) * hidden],
                    &source_probs[row * speakers..(row + 1) * speakers],
                );
                lookup.entry(key).or_insert(row as i32);
            }
            let mut selected_indices = Vec::with_capacity(capacity);
            for (embeds, probs) in selected_embeds
                .chunks_exact(hidden)
                .zip(selected_probs.chunks_exact(speakers))
            {
                let key = payload_key(embeds, probs);
                selected_indices.push(*lookup.get(&key).context("missing selected payload")?);
            }
            self.states[state_index].cache_len = capacity;
            self.states[state_index].probs = selected_probs.clone();
            self.states[state_index].compressed = true;
            result = Some(CacheUpdate {
                source_embeds,
                source_probs: cache_probs,
                selected_indices,
                selected_embeds,
                selected_probs,
            });
        } else {
            let indices: Vec<i32> = (0..cache_rows).map(|row| row as i32).collect();
            gather_rows(
                &self.stream,
                &self.kernels.gather,
                &self.workspace.embeds,
                &indices,
                in_rows,
                self.spec.hidden,
                &mut self.states[state_index].cache,
            )?;
            self.states[state_index].cache_len = cache_rows;
            self.states[state_index].probs.fill(0.0);
            let probability_len = cache_probs.len();
            ensure!(
                probability_len <= self.states[state_index].probs.len(),
                "diarization cache probability extent"
            );
            self.states[state_index].probs[..probability_len].copy_from_slice(&cache_probs);
        }
        let fifo_indices: Vec<i32> = (fifo_begin..fifo_end).map(|row| row as i32).collect();
        gather_rows(
            &self.stream,
            &self.kernels.gather,
            &self.workspace.embeds,
            &fifo_indices,
            in_rows,
            self.spec.hidden,
            &mut self.states[state_index].fifo,
        )?;
        self.states[state_index].fifo_len = fifo_end - fifo_begin;
        self.states[state_index].initialized = true;
        Ok(result)
    }

    pub fn configure(&mut self, state: usize, config: ChunkConfig) -> Result<()> {
        ensure!(state < 2, "diarization state index");
        let hidden = self.spec.hidden as usize;
        ensure!(
            (config.fifo as usize) * hidden <= self.states[state].fifo.len(),
            "diarization FIFO capacity"
        );
        self.stream.memset_zeros(&mut self.states[state].cache)?;
        self.stream.memset_zeros(&mut self.states[state].fifo)?;
        self.states[state].probs.fill(0.0);
        self.states[state].cache_len = 0;
        self.states[state].fifo_len = 0;
        self.states[state].fifo_capacity = config.fifo as usize;
        self.states[state].update_period = config.update as usize;
        self.states[state].compressed = false;
        self.states[state].initialized = true;
        Ok(())
    }

    pub fn call_counts(&self) -> (u64, u64) {
        (self.calls[0], self.calls[1])
    }

    pub fn reset_calls(&mut self) {
        self.calls = [0; 2];
    }

    pub fn state_counters(&self, state: usize) -> (usize, usize, bool) {
        let state = &self.states[state];
        (state.cache_len, state.fifo_len, state.compressed)
    }

    pub fn snapshot(&self, state: usize) -> Result<SpeakerSnapshot> {
        ensure!(state < 2, "diarization state index");
        let state = &self.states[state];
        Ok(SpeakerSnapshot {
            cache: self.stream.clone_dtoh(&state.cache)?,
            fifo: self.stream.clone_dtoh(&state.fifo)?,
            probs: state.probs.clone(),
            cache_len: state.cache_len,
            fifo_len: state.fifo_len,
            fifo_capacity: state.fifo_capacity,
            update_period: state.update_period,
            compressed: state.compressed,
            initialized: state.initialized,
        })
    }

    pub fn restore(&mut self, state: usize, snapshot: &SpeakerSnapshot) -> Result<()> {
        ensure!(state < 2, "diarization state index");
        self.stream.memset_zeros(&mut self.states[state].cache)?;
        self.stream.memset_zeros(&mut self.states[state].fifo)?;
        self.stream
            .memcpy_htod(&snapshot.cache, &mut self.states[state].cache)?;
        self.stream
            .memcpy_htod(&snapshot.fifo, &mut self.states[state].fifo)?;
        let count = snapshot.probs.len().min(self.states[state].probs.len());
        self.states[state].probs.fill(0.0);
        self.states[state].probs[..count].copy_from_slice(&snapshot.probs[..count]);
        self.states[state].cache_len = snapshot.cache_len;
        self.states[state].fifo_len = snapshot.fifo_len;
        self.states[state].fifo_capacity = snapshot.fifo_capacity;
        self.states[state].update_period = snapshot.update_period;
        self.states[state].compressed = snapshot.compressed;
        self.states[state].initialized = snapshot.initialized;
        Ok(())
    }

    pub fn update_cache(
        &mut self,
        state: usize,
        config: ChunkConfig,
        input_embeds: &[f32],
        logits: &[f32],
        mask: &[bool],
        num_chunk_frames: usize,
    ) -> Result<Option<CacheUpdate>> {
        ensure!(state < 2, "diarization state index");
        let hidden = self.spec.hidden as usize;
        let speakers = self.spec.speakers as usize;
        let factor = self.spec.factor as usize;
        let rows = input_embeds.len() / hidden;
        ensure!(
            input_embeds.len().is_multiple_of(hidden)
                && rows <= self.max_rows as usize
                && mask.len() == rows
                && logits.len() == rows * factor * speakers
                && num_chunk_frames <= rows,
            "diarization cache fixture extent"
        );
        self.states[state].fifo_capacity = config.fifo as usize;
        self.states[state].update_period = config.update as usize;
        self.stream.memcpy_htod(
            input_embeds,
            &mut self.workspace.embeds.slice_mut(..rows * hidden),
        )?;
        self.stream.memcpy_htod(
            logits,
            &mut self.workspace.logits.slice_mut(..rows * factor * speakers),
        )?;
        let probs = self.pool_probs(rows as u32, mask)?;
        self.update_state(state, rows as u32, num_chunk_frames as u32, mask, &probs)
    }

    pub fn offline(
        &mut self,
        features: &[f32],
        mask: &[bool],
        state: usize,
    ) -> Result<(Vec<f32>, Vec<CacheUpdate>)> {
        ensure!(state < 2, "diarization state index");
        let config = self.spec.offline;
        self.configure(state, config)?;
        let mel_bins = self.spec.mel.mel_bins as usize;
        let factor = self.spec.factor as usize;
        let speakers = self.spec.speakers as usize;
        let hidden = self.spec.hidden as usize;
        let feature_frames = features.len() / mel_bins;
        ensure!(
            features.len().is_multiple_of(mel_bins) && mask.len() == feature_frames,
            "diarization offline features"
        );
        require_prefix_mask(mask)?;
        let num_embeds = feature_frames.div_ceil(factor);
        self.grow_input_capacity(num_embeds)?;
        let full_features = stack_features(features, 0, num_embeds, mel_bins, factor)?;
        self.embed_all_features(&full_features, num_embeds as u32)?;
        let mut logits = Vec::with_capacity(feature_frames * speakers);
        let mut updates = Vec::new();
        let mut start = 0usize;
        while start < num_embeds {
            let chunk = (config.frames as usize).min(num_embeds - start);
            let end = start + chunk;
            let right = (config.right as usize).min(num_embeds - end);
            let chunk_rows = chunk + right;
            let (cache_len, fifo_len) = {
                let state = &self.states[state];
                (state.cache_len, state.fifo_len)
            };
            let cached = cache_len + fifo_len;
            let rows = cached + chunk_rows;
            ensure!(rows <= self.max_rows as usize, "diarization offline window");
            if cache_len > 0 {
                self.stream.memcpy_dtod(
                    &self.states[state].cache.slice(..cache_len * hidden),
                    &mut self.workspace.embeds.slice_mut(..cache_len * hidden),
                )?;
            }
            if fifo_len > 0 {
                self.stream.memcpy_dtod(
                    &self.states[state].fifo.slice(..fifo_len * hidden),
                    &mut self
                        .workspace
                        .embeds
                        .slice_mut(cache_len * hidden..cached * hidden),
                )?;
            }
            self.stream.memcpy_dtod(
                &self
                    .workspace
                    .input_embeds
                    .slice(start * hidden..(start + chunk_rows) * hidden),
                &mut self
                    .workspace
                    .embeds
                    .slice_mut(cached * hidden..(cached + chunk_rows) * hidden),
            )?;
            let chunk_mask = step_mask(mask, start * factor, chunk_rows, cached, factor);
            let valid = chunk_mask.iter().filter(|flag| **flag).count() as u32;
            let (step_logits, update) =
                self.run_embeddings(rows as u32, valid, &chunk_mask, Some((state, chunk as u32)))?;
            let begin = cached * factor * speakers;
            let stop = (cached + chunk) * factor * speakers;
            logits.extend_from_slice(&step_logits[begin..stop]);
            if let Some(update) = update {
                updates.push(update);
            }
            start = end;
        }
        logits.truncate(feature_frames * speakers);
        Ok((logits, updates))
    }

    pub fn stream_step(
        &mut self,
        features: &[f32],
        mask: &[bool],
        config: ChunkConfig,
        last: bool,
        state: usize,
    ) -> Result<(Vec<f32>, Option<CacheUpdate>)> {
        ensure!(state < 2, "diarization state index");
        let mel_bins = self.spec.mel.mel_bins as usize;
        let factor = self.spec.factor as usize;
        let speakers = self.spec.speakers as usize;
        let hidden = self.spec.hidden as usize;
        let feature_frames = features.len() / mel_bins;
        ensure!(
            features.len().is_multiple_of(mel_bins) && mask.len() == feature_frames,
            "diarization streaming features"
        );
        require_prefix_mask(mask)?;
        let num_embeds = feature_frames.div_ceil(factor);
        let lookahead = if last { 0 } else { config.right as usize };
        ensure!(num_embeds > lookahead, "diarization streaming lookahead");
        let chunk = num_embeds - lookahead;
        let (cache_len, fifo_len) = {
            let state = &self.states[state];
            (state.cache_len, state.fifo_len)
        };
        let cached = cache_len + fifo_len;
        let rows = cached + num_embeds;
        ensure!(
            rows <= self.max_rows as usize,
            "diarization streaming window"
        );
        if cache_len > 0 {
            self.stream.memcpy_dtod(
                &self.states[state].cache.slice(..cache_len * hidden),
                &mut self.workspace.embeds.slice_mut(..cache_len * hidden),
            )?;
        }
        if fifo_len > 0 {
            self.stream.memcpy_dtod(
                &self.states[state].fifo.slice(..fifo_len * hidden),
                &mut self
                    .workspace
                    .embeds
                    .slice_mut(cache_len * hidden..cached * hidden),
            )?;
        }
        let chunk_features = stack_features(features, 0, num_embeds, mel_bins, factor)?;
        let chunk_mask = step_mask(mask, 0, num_embeds, cached, factor);
        self.embed_features(&chunk_features, num_embeds as u32, cached * hidden)?;
        let valid = chunk_mask.iter().filter(|flag| **flag).count() as u32;
        let (step_logits, update) =
            self.run_embeddings(rows as u32, valid, &chunk_mask, Some((state, chunk as u32)))?;
        let begin = cached * factor * speakers;
        let stop = (cached + chunk) * factor * speakers;
        let mut logits = step_logits[begin..stop].to_vec();
        logits.truncate(feature_frames * speakers);
        Ok((logits, update))
    }

    fn encode(&self, rows: u32, valid: u32) -> Result<()> {
        let h = self.spec.hidden;
        let hidden = matrix(&self.workspace.hidden, &self.stream, rows, h)?;
        let normalized = matrix(&self.workspace.normalized, &self.stream, rows, h)?;
        self.normalize(
            matrix(&self.workspace.embeds, &self.stream, rows, h)?,
            &self.input_norm,
            hidden,
        )?;
        for layer in &self.layers {
            self.normalize(hidden, &layer.norm1, normalized)?;
            for (weight, values) in layer.qkv.iter().zip(&self.workspace.qkv) {
                self.apply_linear(normalized, weight, matrix(values, &self.stream, rows, h)?)?;
            }
            self.rotary(matrix(&self.workspace.qkv[0], &self.stream, rows, h)?)?;
            self.rotary(matrix(&self.workspace.qkv[1], &self.stream, rows, h)?)?;
            let heads = self.spec.heads;
            let dim = h / heads;
            let head_view = |buffer| {
                view(
                    buffer,
                    &self.stream,
                    0,
                    [heads, rows, dim],
                    [u64::from(dim), u64::from(h), 1],
                )
            };
            self.attention.run_cublas(
                &self.blas,
                head_view(&self.workspace.qkv[0])?,
                head_view(&self.workspace.qkv[1])?,
                head_view(&self.workspace.qkv[2])?,
                &mut TensorMut(head_view(&self.workspace.attention)?),
                F32Mask {
                    causal: false,
                    valid_keys: valid,
                    query_base: 0,
                    window: 0,
                },
            )?;
            let delta = matrix(&self.workspace.delta, &self.stream, rows, h)?;
            self.apply_linear(
                matrix(&self.workspace.attention, &self.stream, rows, h)?,
                &layer.output,
                delta,
            )?;
            self.residual(hidden, delta)?;
            self.normalize(hidden, &layer.norm2, normalized)?;
            let up = matrix(
                &self.workspace.up,
                &self.stream,
                rows,
                self.spec.intermediate,
            )?;
            self.apply_linear(normalized, &layer.up, up)?;
            self.activation(up, true)?;
            self.apply_linear(up, &layer.down, delta)?;
            self.residual(hidden, delta)?;
        }
        self.normalize(hidden, &self.final_norm, normalized)
    }

    fn classify(&self, rows: u32) -> Result<()> {
        let c = self.spec.head_hidden;
        let projected = matrix(&self.workspace.projection, &self.stream, rows, c)?;
        self.apply_linear(
            matrix(
                &self.workspace.normalized,
                &self.stream,
                rows,
                self.spec.hidden,
            )?,
            &self.projection,
            projected,
        )?;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.pad)
                .arg(&projected.ptr)
                .arg(&self.workspace.padded)
                .arg(&i64::from(rows))
                .arg(&1i64)
                .arg(&i64::from(c))
                .launch(ops::flat(((rows + 2) * c) as usize))?;
            self.stream
                .launch_builder(&self.kernels.im2col)
                .arg(&self.workspace.padded)
                .arg(&self.workspace.columns)
                .arg(&i64::from(rows))
                .arg(&(c as i32))
                .arg(&3i32)
                .arg(&1i32)
                .launch(ops::flat((rows * c * 3) as usize))?;
        }
        self.apply_linear(
            matrix(&self.workspace.columns, &self.stream, rows, c * 3)?,
            &self.convolution,
            matrix(
                &self.workspace.subpixel,
                &self.stream,
                rows,
                c * self.spec.factor,
            )?,
        )?;
        let output_rows = rows * self.spec.factor;
        let subpixel = matrix(&self.workspace.subpixel, &self.stream, output_rows, c)?;
        self.activation(subpixel, false)?;
        let classified = matrix(&self.workspace.classifier, &self.stream, output_rows, c)?;
        self.apply_linear(subpixel, &self.dense, classified)?;
        self.activation(classified, false)?;
        self.apply_linear(
            classified,
            &self.output,
            matrix(
                &self.workspace.logits,
                &self.stream,
                output_rows,
                self.spec.speakers,
            )?,
        )
    }

    /// Includes the zero-padded final stack's subpixel rows.
    pub fn frame_logits(&mut self, features: &[f32], valid_frames: usize) -> Result<Vec<f32>> {
        let mel = self.spec.mel.mel_bins as usize;
        ensure!(
            !features.is_empty()
                && features.len().is_multiple_of(mel)
                && features.iter().all(|x| x.is_finite()),
            "invalid diarization mel features"
        );
        let frames = features.len() / mel;
        ensure!(
            valid_frames > 0 && valid_frames <= frames,
            "invalid diarization valid frames"
        );
        let rows = u32::try_from(frames.div_ceil(self.spec.factor as usize))
            .context("diarization rows overflow")?;
        ensure!(
            rows <= self.max_rows,
            "diarization window {rows} exceeds {}",
            self.max_rows
        );
        let length = (rows as usize) * (self.spec.factor as usize) * mel;
        let padded = if length == features.len() {
            Cow::Borrowed(features)
        } else {
            let mut values = features.to_vec();
            values.resize(length, 0.0);
            Cow::Owned(values)
        };
        self.embed_features(padded.as_ref(), rows, 0)?;
        let valid = u32::try_from(valid_frames.div_ceil(self.spec.factor as usize))?;
        let mask = vec![true; rows as usize];
        let (logits, _) = self.run_embeddings(rows, valid, &mask, None)?;
        Ok(logits)
    }

    pub fn attention_workspace_bytes(&self) -> usize {
        self.attention.workspace_bytes()
    }

    pub fn audio_features(
        &self,
        waveform: &[f32],
        centered: bool,
        last: bool,
    ) -> Result<LogMelBatch> {
        self.frontend.extract(&[waveform], false, centered, last)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::{Dtype, SafeTensors};
    use serde_json::Value;
    use std::path::{Path, PathBuf};

    fn read_f32(fixture: &SafeTensors<'_>, key: &str) -> Result<Vec<f32>> {
        let tensor = fixture.tensor(key)?;
        ensure!(tensor.dtype() == Dtype::F32, "fixture {key} is not F32");
        Ok(tensor
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|value| f32::from_le_bytes(*value))
            .collect())
    }

    fn read_i64(fixture: &SafeTensors<'_>, key: &str) -> Result<Vec<i64>> {
        let tensor = fixture.tensor(key)?;
        ensure!(tensor.dtype() == Dtype::I64, "fixture {key} is not I64");
        Ok(tensor
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|value| i64::from_le_bytes(*value))
            .collect())
    }

    fn read_bool(fixture: &SafeTensors<'_>, key: &str) -> Result<Vec<bool>> {
        let tensor = fixture.tensor(key)?;
        ensure!(tensor.dtype() == Dtype::BOOL, "fixture {key} is not BOOL");
        Ok(tensor.data().iter().map(|value| *value != 0).collect())
    }

    fn batch_f32(values: &[f32], batch: usize, batch_size: usize) -> Result<&[f32]> {
        ensure!(
            batch_size > 0 && values.len().is_multiple_of(batch_size) && batch < batch_size,
            "fixture batch extent"
        );
        let width = values.len() / batch_size;
        Ok(&values[batch * width..(batch + 1) * width])
    }

    fn batch_bool(values: &[bool], batch: usize, batch_size: usize) -> Result<&[bool]> {
        ensure!(
            batch_size > 0 && values.len().is_multiple_of(batch_size) && batch < batch_size,
            "fixture batch extent"
        );
        let width = values.len() / batch_size;
        Ok(&values[batch * width..(batch + 1) * width])
    }

    fn measure(cell: &str, left: &[f32], right: &[f32], floors: &Value) -> Result<bool> {
        ensure!(
            left.len() == right.len()
                && !left.is_empty()
                && left.iter().chain(right).all(|value| value.is_finite()),
            "fixture {cell} has a non-finite or mismatched extent"
        );
        let tau = &floors["metrics"][cell]["tau"];
        let max_tau = tau["max_abs"]
            .as_f64()
            .context("missing frozen max_abs tau")?;
        let rmse_tau = tau["rmse"].as_f64().context("missing frozen rmse tau")?;
        let mut maximum = 0f64;
        let mut square = 0f64;
        for (&a, &b) in left.iter().zip(right) {
            let delta = f64::from(a) - f64::from(b);
            maximum = maximum.max(delta.abs());
            square += delta * delta;
        }
        let rmse = (square / left.len() as f64).sqrt();
        let pass = maximum <= max_tau && rmse <= rmse_tau;
        println!(
            "DIARIZATION_M2_CELL cell={cell} max_abs={maximum:e} tau={max_tau:e} rmse={rmse:e} tau_rmse={rmse_tau:e} {}",
            if pass { "PASS" } else { "DEFECT" }
        );
        Ok(pass)
    }

    fn exact_i64(cell: &str, left: &[i64], right: &[i64]) -> bool {
        let pass = left == right;
        println!(
            "DIARIZATION_M2_EXACT cell={cell} {}",
            if pass { "PASS" } else { "DEFECT" }
        );
        pass
    }

    fn exact_bool(cell: &str, left: &[bool], right: &[bool]) -> bool {
        let pass = left == right;
        println!(
            "DIARIZATION_M2_MASK cell={cell} {}",
            if pass { "PASS" } else { "DEFECT" }
        );
        pass
    }

    fn read_wave(path: &Path) -> Result<Vec<f32>> {
        crate::media::read_wav(std::fs::File::open(path)?, 16000)
    }

    fn before_snapshot(
        fixture: &SafeTensors<'_>,
        batch: usize,
        batch_size: usize,
        cache_frames: usize,
        hidden: usize,
        config: ChunkConfig,
    ) -> Result<SpeakerSnapshot> {
        let cache_elements = cache_frames * hidden;
        let fifo_elements = config.fifo as usize * hidden;
        let cache = read_f32(fixture, "before.embeds")?;
        let fifo = read_f32(fixture, "before.fifo")?;
        let probs = read_f32(fixture, "before.probs")?;
        let mut state_fifo = vec![0.0; cache_elements];
        state_fifo[..fifo_elements].copy_from_slice(batch_f32(&fifo, batch, batch_size)?);
        Ok(SpeakerSnapshot {
            cache: batch_f32(&cache, batch, batch_size)?.to_vec(),
            fifo: state_fifo,
            probs: batch_f32(&probs, batch, batch_size)?.to_vec(),
            cache_len: read_i64(fixture, "before.num_cache_frames")?[0] as usize,
            fifo_len: read_i64(fixture, "before.num_fifo_frames")?[0] as usize,
            fifo_capacity: config.fifo as usize,
            update_period: config.update as usize,
            compressed: read_i64(fixture, "before.is_compressed")?[0] != 0,
            initialized: read_i64(fixture, "before.is_initialized")?[0] != 0,
        })
    }

    fn tie_relaxed(floors: &Value, label: &str, number: usize) -> Result<bool> {
        let entries = floors["arm_reports"]["media-CPU-r2"]["cache_compressions"]
            .as_array()
            .context("missing canonical compression records")?;
        for entry in entries {
            if entry["label"].as_str() == Some(label)
                && entry["number"].as_u64() == Some(number as u64)
            {
                return Ok(entry["zero_finite_cutoff_gaps"]
                    .as_u64()
                    .context("missing zero-gap count")?
                    > 0);
            }
        }
        Ok(false)
    }

    struct CompressionContext<'a> {
        path: &'a Path,
        cell_prefix: &'a str,
        batch: usize,
        batch_size: usize,
        floors: &'a Value,
        relaxed: bool,
    }

    fn compression_compare(
        update: &CacheUpdate,
        context: &CompressionContext<'_>,
    ) -> Result<usize> {
        let bytes = std::fs::read(context.path)?;
        let fixture = SafeTensors::deserialize(&bytes)?;
        let mut defects = 0;
        for (name, candidate) in [
            ("source_embeds", &update.source_embeds),
            ("source_probs", &update.source_probs),
            ("selected_embeds", &update.selected_embeds),
            ("selected_probs", &update.selected_probs),
        ] {
            let values = read_f32(&fixture, name)?;
            let expected = batch_f32(&values, context.batch, context.batch_size)?;
            if !measure(
                &format!("{}/{name}", context.cell_prefix),
                expected,
                candidate,
                context.floors,
            )? {
                defects += 1;
            }
        }
        let selected = read_i64(&fixture, "selected_indices")?;
        ensure!(
            context.batch_size > 0
                && selected.len().is_multiple_of(context.batch_size)
                && context.batch < context.batch_size,
            "selected index batch extent"
        );
        let width = selected.len() / context.batch_size;
        let expected = selected[context.batch * width..(context.batch + 1) * width].to_vec();
        let actual: Vec<i64> = update
            .selected_indices
            .iter()
            .map(|&value| i64::from(value))
            .collect();
        if context.relaxed {
            println!(
                "DIARIZATION_M2_TIE cell={} payload_only=true",
                context.cell_prefix
            );
        } else if !exact_i64(
            &format!("{}/selected_indices", context.cell_prefix),
            &expected,
            &actual,
        ) {
            defects += 1;
        }
        Ok(defects)
    }

    #[test]
    fn diarization_fifo_source_span_is_cache_fifo_chunk() {
        let span = fifo_source_span(0, 255, 9);
        assert_eq!(span, 264);
        assert!(span <= 268);
    }

    #[test]
    fn diarization_full_input_embedding_capacity_covers_full_clip() {
        let rows = input_embedding_rows(684, 5000);
        assert_eq!(rows, 5000);
        assert!(rows as usize >= 12001usize.div_ceil(8));
    }

    #[test]
    fn diarization_rejects_non_prefix_masks() -> Result<()> {
        let holed: Vec<bool> = (0..64).map(|frame| frame != 20).collect();
        let error = require_prefix_mask(&holed).expect_err("interior-hole mask accepted");
        ensure!(
            error.to_string().contains("first hole at frame 20"),
            "diarization mask error does not name the first hole: {error}"
        );
        let padded: Vec<bool> = (0..64).map(|frame| frame < 50).collect();
        require_prefix_mask(&padded)?;
        require_prefix_mask(&[true; 64])?;
        Ok(())
    }

    #[test]
    #[ignore = "requires GPU lease and external fixtures"]
    fn diarization_encoder_and_head_match_frozen_reference() -> Result<()> {
        let model = PathBuf::from(
            std::env::var_os("FF_DIARIZATION_MODEL").context("diarization model missing")?,
        );
        let fixtures = PathBuf::from(
            std::env::var_os("FF_DIARIZATION_FIXTURES").context("diarization fixtures missing")?,
        );
        let floors = PathBuf::from(
            std::env::var_os("FF_DIARIZATION_FLOORS").context("diarization floors missing")?,
        );
        let floors: Value = serde_json::from_slice(&std::fs::read(floors)?)?;
        ensure!(
            floors["status"] == "FROZEN",
            "diarization references are not frozen"
        );
        let device = Device::new(0)?;
        let mut engine = Diarization::load(&model, &device)?;
        let difference = |left: &[f32], right: &[f32]| -> Result<(f64, f64)> {
            ensure!(
                left.len() == right.len() && !left.is_empty(),
                "frame-logit extent mismatch"
            );
            ensure!(
                left.iter().chain(right).all(|x| x.is_finite()),
                "non-finite frame logits"
            );
            let mut maximum = 0f64;
            let mut square = 0f64;
            for (&a, &b) in left.iter().zip(right) {
                let d = f64::from(a) - f64::from(b);
                maximum = maximum.max(d.abs());
                square += d * d;
            }
            Ok((maximum, (square / left.len() as f64).sqrt()))
        };
        let mut defects = 0;
        let mut first = None;
        for (case, valid) in [
            ("tail7", 7),
            ("production684", 5471),
            ("tail8", 8),
            ("production541", 4327),
            ("tail9", 9),
            ("tail65", 63),
        ] {
            let bytes = std::fs::read(fixtures.join(format!("components/{case}-b1.safetensors")))?;
            let fixture = SafeTensors::deserialize(&bytes)?;
            let read = |key: &str| -> Result<Vec<f32>> {
                let tensor = fixture.tensor(key)?;
                ensure!(tensor.dtype() == Dtype::F32, "frame fixture must be F32");
                Ok(tensor
                    .data()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| f32::from_le_bytes(*v))
                    .collect())
            };
            let features = read("features")?;
            let expected = read("logits")?;
            let actual = engine.frame_logits(&features, valid)?;
            let limit = &floors["metrics"][format!("components/{case}.safetensors/logits")]["tau"];
            let max_limit = limit["max_abs"]
                .as_f64()
                .context("missing frame max_abs floor")?;
            let rms_limit = limit["rmse"].as_f64().context("missing frame RMSE floor")?;
            let (maximum, rmse) = difference(&expected, &actual)?;
            let pass = maximum <= max_limit && rmse <= rms_limit;
            defects += usize::from(!pass);
            println!(
                "DIARIZATION_FRAME case={case} max_abs={maximum:e} tau={max_limit:e} rmse={rmse:e} tau_rmse={rms_limit:e} {}",
                if pass { "PASS" } else { "DEFECT" }
            );
            let mut mutant = actual.clone();
            let speakers = engine.spec.speakers as usize;
            for row in 0..actual.len() / speakers {
                let source = (row / 8 * 8) + ((row % 8) ^ 1);
                mutant[row * speakers..(row + 1) * speakers]
                    .copy_from_slice(&actual[source * speakers..(source + 1) * speakers]);
            }
            let (control_abs, control_rmse) = difference(&expected, &mutant)?;
            let detected = control_abs > 10. * max_limit && control_rmse > 10. * rms_limit;
            defects += usize::from(!detected);
            println!(
                "DIARIZATION_PHASE_CONTROL case={case} max_abs={control_abs:e} rmse={control_rmse:e} detected={detected}"
            );
            if case == "tail7" {
                first = Some((features, actual));
            }
        }
        let (features, expected) = first.context("missing repeat case")?;
        let repeat = engine.frame_logits(&features, 7)?;
        let reset = expected
            .iter()
            .zip(&repeat)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        defects += usize::from(!reset);
        let oversized = vec![
            0f32;
            ((engine.max_rows + 1) * engine.spec.factor * engine.spec.mel.mel_bins)
                as usize
        ];
        ensure!(
            engine
                .frame_logits(&oversized, oversized.len() / 128)
                .is_err(),
            "capacity bound accepted oversized window"
        );
        let verdict = if defects == 0 { "PASS" } else { "DEFECT" };
        println!(
            "DIARIZATION_FRAME_GATE {verdict} cells=6 layers=31 phases=8 eager=true scratch_replay_exact={reset} attention_workspace_bytes={} defects={defects}",
            engine.attention_workspace_bytes()
        );
        ensure!(
            defects == 0,
            "diarization frame classification has {defects} DEFECTs"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires GPU lease and external fixtures"]
    fn diarization_speaker_cache_matches_frozen_reference() -> Result<()> {
        let model = PathBuf::from(
            std::env::var_os("FF_DIARIZATION_MODEL").context("diarization model missing")?,
        );
        let root = PathBuf::from(
            std::env::var_os("FF_DIARIZATION_FIXTURES").context("diarization fixtures missing")?,
        );
        let floors_path = PathBuf::from(
            std::env::var_os("FF_DIARIZATION_FLOORS").context("diarization floors missing")?,
        );
        let floors: Value = serde_json::from_slice(&std::fs::read(floors_path)?)?;
        ensure!(
            floors["status"] == "FROZEN",
            "diarization references are not frozen"
        );
        let device = Device::new(0)?;
        let mut engine = Diarization::load(&model, &device)?;
        let bounded = root.join("bounded-CPU-r2");
        let hidden = engine.spec.hidden as usize;
        let speakers = engine.spec.speakers as usize;
        let factor = engine.spec.factor as usize;
        let cache_frames = engine.spec.cache.frames as usize;
        let streaming = ChunkConfig {
            frames: 9,
            right: 4,
            fifo: engine.spec.cache.fifo,
            update: engine.spec.cache.update,
        };
        let cases = [
            ("empty", false),
            ("fifo_exact264", false),
            ("fifo_overflow_first", false),
            ("cache_first_compression", false),
            ("cache_already_compressed", false),
            ("offline_first", true),
            ("offline_full", true),
            ("masked_tail", false),
        ];
        let mut defects = 0;
        for (case, offline) in cases {
            let config = if offline {
                engine.spec.offline
            } else {
                streaming
            };
            for batch_size in [1usize, 2usize] {
                let path = bounded.join(format!("cache/{case}-b{batch_size}.safetensors"));
                let bytes = std::fs::read(&path)?;
                let fixture = SafeTensors::deserialize(&bytes)?;
                let input_embeds = read_f32(&fixture, "input_embeds")?;
                let input_logits = read_f32(&fixture, "input_logits")?;
                let input_mask = read_bool(&fixture, "input_mask")?;
                let num_chunk_frames = read_i64(&fixture, "num_chunk_frames")?[0] as usize;
                let rows = input_mask.len() / batch_size;
                ensure!(
                    batch_size > 0
                        && input_mask.len().is_multiple_of(batch_size)
                        && input_embeds.len() == batch_size * rows * hidden
                        && input_logits.len() == batch_size * rows * factor * speakers,
                    "cache fixture extent"
                );
                let after_embeds = read_f32(&fixture, "after.embeds")?;
                let after_fifo = read_f32(&fixture, "after.fifo")?;
                let after_probs = read_f32(&fixture, "after.probs")?;
                let after_cache = read_i64(&fixture, "after.num_cache_frames")?[0];
                let after_fifo_len = read_i64(&fixture, "after.num_fifo_frames")?[0];
                let after_compressed = read_i64(&fixture, "after.is_compressed")?[0];
                let after_initialized = read_i64(&fixture, "after.is_initialized")?[0];
                let mut updates = Vec::new();
                for stream in 0..batch_size {
                    let before = before_snapshot(
                        &fixture,
                        stream,
                        batch_size,
                        cache_frames,
                        hidden,
                        config,
                    )?;
                    engine.configure(stream, config)?;
                    engine.restore(stream, &before)?;
                    let update = engine.update_cache(
                        stream,
                        config,
                        batch_f32(&input_embeds, stream, batch_size)?,
                        batch_f32(&input_logits, stream, batch_size)?,
                        batch_bool(&input_mask, stream, batch_size)?,
                        num_chunk_frames,
                    )?;
                    updates.push(update);
                    let snapshot = engine.snapshot(stream)?;
                    let prefix = format!("cache/{case}-b{batch_size}.safetensors");
                    if !measure(
                        &format!("{prefix}/after.embeds"),
                        batch_f32(&after_embeds, stream, batch_size)?,
                        &snapshot.cache,
                        &floors,
                    )? {
                        defects += 1;
                    }
                    if !measure(
                        &format!("{prefix}/after.fifo"),
                        batch_f32(&after_fifo, stream, batch_size)?,
                        &snapshot.fifo[..config.fifo as usize * hidden],
                        &floors,
                    )? {
                        defects += 1;
                    }
                    if !measure(
                        &format!("{prefix}/after.probs"),
                        batch_f32(&after_probs, stream, batch_size)?,
                        &snapshot.probs,
                        &floors,
                    )? {
                        defects += 1;
                    }
                    let (cache_len, fifo_len, compressed) = engine.state_counters(stream);
                    if !exact_i64(
                        &format!("{prefix}/after.counters"),
                        &[
                            after_cache,
                            after_fifo_len,
                            after_compressed,
                            after_initialized,
                        ],
                        &[
                            cache_len as i64,
                            fifo_len as i64,
                            i64::from(compressed),
                            i64::from(snapshot.initialized),
                        ],
                    ) {
                        defects += 1;
                    }
                }
                let compression = format!("cache/{case}-b{batch_size}/compression-0.safetensors");
                let compression_path = bounded.join(&compression);
                if compression_path.exists() {
                    for (stream, update) in updates.iter().enumerate() {
                        let update = update.as_ref().context("missing native compression")?;
                        let context = CompressionContext {
                            path: &compression_path,
                            cell_prefix: &compression,
                            batch: stream,
                            batch_size,
                            floors: &floors,
                            relaxed: false,
                        };
                        defects += compression_compare(update, &context)?;
                    }
                } else {
                    ensure!(
                        updates.iter().all(Option::is_none),
                        "unexpected native compression"
                    );
                }
                if case == "fifo_overflow_first" && batch_size == 1 {
                    let control_path = bounded.join("controls/fifo-pop-from-1.safetensors");
                    let control_bytes = std::fs::read(&control_path)?;
                    let control = SafeTensors::deserialize(&control_bytes)?;
                    let control_embeds = read_f32(&control, "embeds")?;
                    let mut mutant = input_embeds.clone();
                    let start = 0;
                    let stop = start + control_embeds.len();
                    ensure!(stop <= mutant.len(), "FIFO control extent");
                    mutant[start..stop].copy_from_slice(&control_embeds);
                    let before = before_snapshot(&fixture, 0, 1, cache_frames, hidden, config)?;
                    engine.configure(0, config)?;
                    engine.restore(0, &before)?;
                    engine.update_cache(
                        0,
                        config,
                        &mutant,
                        batch_f32(&input_logits, 0, 1)?,
                        batch_bool(&input_mask, 0, 1)?,
                        num_chunk_frames,
                    )?;
                    let snapshot = engine.snapshot(0)?;
                    let expected = batch_f32(&after_embeds, 0, 1)?;
                    let mismatches = expected
                        .iter()
                        .zip(&snapshot.cache)
                        .filter(|(left, right)| left.to_bits() != right.to_bits())
                        .count();
                    let detected = mismatches > 0;
                    println!(
                        "DIARIZATION_FIFO_CONTROL case=fifo_overflow_first mismatches={mismatches} {}",
                        if detected { "PASS" } else { "DEFECT" }
                    );
                    defects += usize::from(!detected);
                }
            }
        }
        let verdict = if defects == 0 { "PASS" } else { "DEFECT" };
        println!("DIARIZATION_M2_CACHE_GATE {verdict} cases=16 defects={defects}");
        ensure!(
            defects == 0,
            "diarization speaker cache has {defects} DEFECTs"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires GPU lease and external fixtures"]
    fn diarization_offline_grows_input_buffers_for_long_recording() -> Result<()> {
        let model = PathBuf::from(
            std::env::var_os("FF_DIARIZATION_MODEL").context("diarization model missing")?,
        );
        let device = Device::new(0)?;
        let mut engine = Diarization::load(&model, &device)?;
        let mel_bins = engine.spec.mel.mel_bins as usize;
        let factor = engine.spec.factor as usize;
        let speakers = engine.spec.speakers as usize;
        let config = engine.spec.offline;
        let short_frames = (config.frames + config.right) as usize * factor;
        let long_frames = 40_007;
        ensure!(
            long_frames > 40_000
                && short_frames < long_frames
                && long_frames.div_ceil(factor) > engine.spec.max_positions as usize,
            "long-recording control does not exceed the setup capacity"
        );
        let mut features = vec![0.0f32; long_frames * mel_bins];
        for (frame, row) in features.chunks_mut(mel_bins).enumerate() {
            for (bin, value) in row.iter_mut().enumerate() {
                let signal = (frame * 7 + bin * 13 + (frame / 19) * 23 + (bin / 29) * 31) % 37;
                *value = signal as f32 / 37.0 - 0.5;
            }
        }
        // padding prefix: 40,000 valid frames plus a 7-frame padded tail
        let mask: Vec<bool> = (0..long_frames).map(|frame| frame < 40_000).collect();
        let short_features = features[..short_frames * mel_bins].to_vec();
        let short_mask = mask[..short_frames].to_vec();
        let first_window = config.frames as usize * factor * speakers;
        engine.reset_calls();
        let (short_logits, _) = engine.offline(&short_features, &short_mask, 0)?;
        ensure!(
            engine.call_counts().0 == 1,
            "diarization short clip embedded more than once"
        );
        engine.reset_calls();
        let (long_logits, _) = engine.offline(&features, &mask, 0)?;
        let windows = long_frames
            .div_ceil(factor)
            .div_ceil(config.frames as usize);
        ensure!(
            engine.call_counts() == (1, windows as u64),
            "diarization long recording changed the embedding call count"
        );
        ensure!(
            short_logits.len() == short_frames * speakers
                && long_logits.len() == long_frames * speakers,
            "diarization long recording logits extent"
        );
        ensure!(
            short_logits
                .iter()
                .chain(&long_logits)
                .all(|value| value.is_finite()),
            "diarization long recording produced non-finite logits"
        );
        let mut max_abs = 0f64;
        let mut bit_count = 0u64;
        for (short, long) in short_logits[..first_window]
            .iter()
            .zip(&long_logits[..first_window])
        {
            max_abs = max_abs.max((f64::from(*short) - f64::from(*long)).abs());
            bit_count += u64::from((short.to_bits() ^ long.to_bits()).count_ones());
        }
        println!(
            "DIARIZATION_LONG_RECORDING_CELL cell=first_window/logits max_abs={max_abs:e} bit_count={bit_count}"
        );
        for (index, (short, long)) in short_logits[..first_window]
            .iter()
            .zip(&long_logits[..first_window])
            .enumerate()
        {
            ensure!(
                short.to_bits() == long.to_bits(),
                "diarization long recording first window differs at logit {index}"
            );
        }
        let mut wrong_features = short_features.clone();
        wrong_features[..mel_bins]
            .copy_from_slice(&features[short_frames * mel_bins..(short_frames + 1) * mel_bins]);
        engine.reset_calls();
        let (wrong_logits, _) = engine.offline(&wrong_features, &short_mask, 0)?;
        let differs = wrong_logits[..first_window]
            .iter()
            .zip(&short_logits[..first_window])
            .any(|(left, right)| left.to_bits() != right.to_bits());
        ensure!(
            differs,
            "diarization first window ignored a perturbed leading frame"
        );
        println!(
            "DIARIZATION_LONG_RECORDING_GATE PASS frames={long_frames} windows={windows} first_window_logits={first_window} embedding_calls=1 negative_control=true"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires GPU lease and external fixtures"]
    fn diarization_streaming_matches_frozen_reference() -> Result<()> {
        let model = PathBuf::from(
            std::env::var_os("FF_DIARIZATION_MODEL").context("diarization model missing")?,
        );
        let root = PathBuf::from(
            std::env::var_os("FF_DIARIZATION_FIXTURES").context("diarization fixtures missing")?,
        );
        let floors_path = PathBuf::from(
            std::env::var_os("FF_DIARIZATION_FLOORS").context("diarization floors missing")?,
        );
        let floors: Value = serde_json::from_slice(&std::fs::read(floors_path)?)?;
        ensure!(
            floors["status"] == "FROZEN",
            "diarization references are not frozen"
        );
        let device = Device::new(0)?;
        let mut engine = Diarization::load(&model, &device)?;
        let media = root.join("media-CPU-r2");
        let speakers = engine.spec.speakers as usize;
        let factor = engine.spec.factor as usize;
        let meeting_wave = read_wave(&root.join("inputs/meeting.wav"))?;
        let swap_wave = read_wave(&root.join("inputs/arrival-swap.wav"))?;
        let modes = [
            "offline",
            "low_latency",
            "very_low_latency",
            "ultra_low_latency",
        ];
        let mut defects = 0;
        let mut outputs: Vec<(&str, &str, Vec<f32>)> = Vec::new();
        for clip in ["meeting", "arrival-swap"] {
            let wave = if clip == "meeting" {
                &meeting_wave
            } else {
                &swap_wave
            };
            for mode in modes {
                let label = format!("{clip}-{mode}");
                let steps: Value = serde_json::from_slice(&std::fs::read(
                    media.join(format!("media/{label}-steps.json")),
                )?)?;
                let config = if mode == "offline" {
                    engine.spec.offline
                } else {
                    let pair = engine
                        .spec
                        .modes
                        .get(mode)
                        .context("missing streaming mode")?;
                    ChunkConfig {
                        frames: pair[0],
                        right: pair[1],
                        fifo: engine.spec.cache.fifo,
                        update: engine.spec.cache.update,
                    }
                };
                engine.reset_calls();
                let mut logits = Vec::new();
                let mut updates = Vec::new();
                if mode == "offline" {
                    let frontend = media.join(format!("frontend/{clip}-offline.safetensors"));
                    let bytes = std::fs::read(&frontend)?;
                    let feature_file = SafeTensors::deserialize(&bytes)?;
                    let features = read_f32(&feature_file, "features")?;
                    let mask = read_bool(&feature_file, "mask")?;
                    let (offline_logits, offline_updates) = engine.offline(&features, &mask, 0)?;
                    logits = offline_logits;
                    updates = offline_updates;
                } else {
                    engine.configure(0, config)?;
                    let step_rows = steps["steps"].as_array().context("missing step records")?;
                    let step_mel = config.frames as usize * factor;
                    for (step, record) in step_rows.iter().enumerate() {
                        let mel_index = step * step_mel;
                        let (features, mask, last) = crate::diarization::stream_frontend_step(
                            &engine, wave, step, config, mel_index,
                        )?;
                        let expected_last =
                            record["last"].as_bool().context("missing step last flag")?;
                        if !exact_bool(
                            &format!("media/{label}-step{step}/last"),
                            &[expected_last],
                            &[last],
                        ) {
                            defects += 1;
                        }
                        if step == 0 || step == 1 || last {
                            let frontend =
                                media.join(format!("frontend/{label}-step{step}.safetensors"));
                            let bytes = std::fs::read(&frontend)?;
                            let feature_file = SafeTensors::deserialize(&bytes)?;
                            let expected_features = read_f32(&feature_file, "features")?;
                            let expected_mask = read_bool(&feature_file, "mask")?;
                            if !measure(
                                &format!("frontend/{label}-step{step}.safetensors/features"),
                                &expected_features,
                                &features,
                                &floors,
                            )? {
                                defects += 1;
                            }
                            if !exact_bool(
                                &format!("frontend/{label}-step{step}.safetensors/mask"),
                                &expected_mask,
                                &mask,
                            ) {
                                defects += 1;
                            }
                        }
                        let before = engine.snapshot(0)?;
                        let (step_logits, update) =
                            engine.stream_step(&features, &mask, config, last, 0)?;
                        if let Some(update) = update {
                            updates.push(update);
                        }
                        let feature_frames = record["feature_frames"]
                            .as_u64()
                            .context("missing feature_frames")?
                            as usize;
                        let valid_frames = record["valid_frames"]
                            .as_u64()
                            .context("missing valid_frames")?
                            as usize;
                        let logit_frames = record["logit_frames"]
                            .as_u64()
                            .context("missing logit_frames")?
                            as usize;
                        let cache =
                            record["cache"].as_u64().context("missing cache counter")? as usize;
                        let fifo =
                            record["fifo"].as_u64().context("missing fifo counter")? as usize;
                        let compressed = record["compressed"]
                            .as_bool()
                            .context("missing compressed flag")?;
                        let (native_cache, native_fifo, native_compressed) =
                            engine.state_counters(0);
                        if !exact_i64(
                            &format!("media/{label}-step{step}/counters"),
                            &[
                                feature_frames as i64,
                                valid_frames as i64,
                                logit_frames as i64,
                                cache as i64,
                                fifo as i64,
                                i64::from(compressed),
                            ],
                            &[
                                mask.len() as i64,
                                mask.iter().filter(|flag| **flag).count() as i64,
                                (step_logits.len() / speakers) as i64,
                                native_cache as i64,
                                native_fifo as i64,
                                i64::from(native_compressed),
                            ],
                        ) {
                            defects += 1;
                        }
                        if step == 61 && clip == "meeting" && mode == "low_latency" {
                            let after = engine.snapshot(0)?;
                            engine.restore(0, &before)?;
                            let (repeat_logits, _) =
                                engine.stream_step(&features, &mask, config, last, 0)?;
                            let repeat = engine.snapshot(0)?;
                            let equal = repeat_logits == step_logits
                                && repeat.cache == after.cache
                                && repeat.fifo == after.fifo
                                && repeat.probs == after.probs;
                            println!(
                                "DIARIZATION_SERIALIZATION_REPEAT {}",
                                if equal { "PASS" } else { "DEFECT" }
                            );
                            defects += usize::from(!equal);
                            engine.restore(0, &after)?;
                        }
                        logits.extend_from_slice(&step_logits);
                        if last {
                            break;
                        }
                    }
                }
                let expected_logits = media.join(format!("media/{label}.safetensors"));
                let bytes = std::fs::read(&expected_logits)?;
                let logits_file = SafeTensors::deserialize(&bytes)?;
                let expected = read_f32(&logits_file, "logits")?;
                if !measure(
                    &format!("media/{label}.safetensors/logits"),
                    &expected,
                    &logits,
                    &floors,
                )? {
                    defects += 1;
                }
                outputs.push((clip, mode, logits.clone()));
                let compression_count = steps["compressions"]
                    .as_u64()
                    .context("missing compression count")?
                    as usize;
                if !exact_i64(
                    &format!("media/{label}/compression_count"),
                    &[compression_count as i64],
                    &[updates.len() as i64],
                ) {
                    defects += 1;
                }
                for (number, update) in updates.iter().enumerate() {
                    let prefix = format!("media/{label}/compression-{number}.safetensors");
                    let path = media.join(&prefix);
                    let relaxed = tie_relaxed(&floors, &format!("media/{label}"), number)?;
                    let context = CompressionContext {
                        path: &path,
                        cell_prefix: &prefix,
                        batch: 0,
                        batch_size: 1,
                        floors: &floors,
                        relaxed,
                    };
                    defects += compression_compare(update, &context)?;
                }
                if mode != "offline" {
                    let final_path = media.join(format!("media/{label}-final-state.safetensors"));
                    let bytes = std::fs::read(&final_path)?;
                    let final_file = SafeTensors::deserialize(&bytes)?;
                    let snapshot = engine.snapshot(0)?;
                    for (name, candidate) in [
                        ("embeds", &snapshot.cache),
                        ("fifo", &snapshot.fifo),
                        ("probs", &snapshot.probs),
                    ] {
                        let expected = read_f32(&final_file, name)?;
                        if !measure(
                            &format!("media/{label}-final-state.safetensors/{name}"),
                            &expected,
                            candidate,
                            &floors,
                        )? {
                            defects += 1;
                        }
                    }
                    let cache = read_i64(&final_file, "num_cache_frames")?[0];
                    let fifo = read_i64(&final_file, "num_fifo_frames")?[0];
                    let compressed = read_i64(&final_file, "is_compressed")?[0];
                    let initialized = read_i64(&final_file, "is_initialized")?[0];
                    let (native_cache, native_fifo, native_compressed) = engine.state_counters(0);
                    if !exact_i64(
                        &format!("media/{label}-final-state.safetensors/counters"),
                        &[cache, fifo, compressed, initialized],
                        &[
                            native_cache as i64,
                            native_fifo as i64,
                            i64::from(native_compressed),
                            i64::from(snapshot.initialized),
                        ],
                    ) {
                        defects += 1;
                    }
                }
                let expected_calls = &steps["calls"];
                let (embedding, audio_tower) = engine.call_counts();
                if !exact_i64(
                    &format!("media/{label}/calls"),
                    &[
                        expected_calls["embedding"]
                            .as_u64()
                            .context("missing embedding calls")? as i64,
                        expected_calls["audio_tower"]
                            .as_u64()
                            .context("missing audio_tower calls")? as i64,
                    ],
                    &[embedding as i64, audio_tower as i64],
                ) {
                    defects += 1;
                }
            }
        }
        for mode in modes {
            let left = outputs
                .iter()
                .find(|(clip, candidate, _)| *clip == "meeting" && *candidate == mode)
                .context("missing meeting output")?;
            let right = outputs
                .iter()
                .find(|(clip, candidate, _)| *clip == "arrival-swap" && *candidate == mode)
                .context("missing arrival-swap output")?;
            let cell = format!("media/meeting-{mode}.safetensors/logits");
            let tau = &floors["metrics"][cell]["tau"];
            let max_tau = tau["max_abs"]
                .as_f64()
                .context("missing frozen max_abs tau")?;
            let rmse_tau = tau["rmse"].as_f64().context("missing frozen rmse tau")?;
            let mut maximum = 0f64;
            let mut square = 0f64;
            for (&a, &b) in left.2.iter().zip(&right.2) {
                let delta = f64::from(a) - f64::from(b);
                maximum = maximum.max(delta.abs());
                square += delta * delta;
            }
            let rmse = (square / left.2.len() as f64).sqrt();
            let detected = left.2 != right.2 && maximum > 10.0 * max_tau && rmse > 10.0 * rmse_tau;
            println!(
                "DIARIZATION_ARRIVAL_CONTROL mode={mode} max_abs={maximum:e} rmse={rmse:e} nonidentical={} {}",
                left.2 != right.2,
                if detected { "PASS" } else { "DEFECT" }
            );
            defects += usize::from(!detected);
        }
        let meeting_features = {
            let path = media.join("frontend/meeting-offline.safetensors");
            let bytes = std::fs::read(&path)?;
            let fixture = SafeTensors::deserialize(&bytes)?;
            (
                read_f32(&fixture, "features")?,
                read_bool(&fixture, "mask")?,
            )
        };
        let swap_features = {
            let path = media.join("frontend/arrival-swap-offline.safetensors");
            let bytes = std::fs::read(&path)?;
            let fixture = SafeTensors::deserialize(&bytes)?;
            (
                read_f32(&fixture, "features")?,
                read_bool(&fixture, "mask")?,
            )
        };
        let (meeting_b2, meeting_updates) =
            engine.offline(&meeting_features.0, &meeting_features.1, 0)?;
        let (swap_b2, swap_updates) = engine.offline(&swap_features.0, &swap_features.1, 1)?;
        for (clip, candidate) in [("meeting", &meeting_b2), ("arrival-swap", &swap_b2)] {
            let path = media.join(format!("media/{clip}-offline-b2.safetensors"));
            let bytes = std::fs::read(&path)?;
            let fixture = SafeTensors::deserialize(&bytes)?;
            let expected = read_f32(&fixture, "logits")?;
            if !measure(
                &format!("media/{clip}-offline.safetensors/logits"),
                &expected,
                candidate,
                &floors,
            )? {
                defects += 1;
            }
        }
        for (stream, updates) in [meeting_updates, swap_updates].iter().enumerate() {
            for (number, update) in updates.iter().enumerate() {
                let prefix = format!("media/offline-b2/compression-{number}.safetensors");
                let path = media.join(&prefix);
                let relaxed = tie_relaxed(&floors, "media/offline-b2", number)?;
                let context = CompressionContext {
                    path: &path,
                    cell_prefix: &prefix,
                    batch: stream,
                    batch_size: 2,
                    floors: &floors,
                    relaxed,
                };
                defects += compression_compare(update, &context)?;
            }
        }
        let (replay, _) = engine.offline(&meeting_features.0, &meeting_features.1, 0)?;
        let initial = outputs
            .iter()
            .find(|(clip, mode, _)| *clip == "meeting" && *mode == "offline")
            .context("missing initial offline output")?;
        if !exact_bool(
            "media/meeting-offline-reset.safetensors/logits",
            &[replay == initial.2],
            &[true],
        ) {
            defects += 1;
        }
        let verdict = if defects == 0 { "PASS" } else { "DEFECT" };
        println!(
            "DIARIZATION_M2_MEDIA_GATE {verdict} clips=2 modes=4 eager=true b2_isolated=true defects={defects}"
        );
        ensure!(defects == 0, "diarization streaming has {defects} DEFECTs");
        Ok(())
    }
}
