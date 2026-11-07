//! Native-F32 Parakeet FastConformer encoder.
use crate::{
    backend::{
        cuda::{Device, cubin},
        gpu::{CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg},
    },
    checkpoint::MappedCheckpoint,
    parakeet::{FastConformer, ParakeetSpec, conv_out, position_rows_fit},
};
use anyhow::{Context, Result, ensure};
use cudarc::cublas::{CudaBlas, sys as cublas_sys};
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};
use std::{path::Path, sync::Arc};

#[cfg(test)]
static ENCODE_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
static ENCODE_LAUNCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
static POSITION_UPLOADS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
static ATTENTION_QUERY_ROWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

macro_rules! note_launch {
    () => {
        #[cfg(test)]
        {
            ENCODE_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    };
}

struct Linear {
    weight: CudaSlice<f32>,
    bias: Option<CudaSlice<f32>>,
    output: u32,
    input: u32,
}

struct Norm {
    weight: CudaSlice<f32>,
    bias: CudaSlice<f32>,
}

struct FeedForward {
    norm: Norm,
    up: Linear,
    down: Linear,
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    relative: Linear,
    bias_u: CudaSlice<f32>,
    bias_v: CudaSlice<f32>,
}

struct Convolution {
    norm: Norm,
    pointwise1: Linear,
    depthwise: CudaSlice<f32>,
    depthwise_bias: Option<CudaSlice<f32>>,
    conv_weight: CudaSlice<f32>,
    conv_bias: CudaSlice<f32>,
    running_mean: Option<CudaSlice<f32>>,
    running_var: Option<CudaSlice<f32>>,
    pointwise2: Linear,
}

struct Block {
    ff1: FeedForward,
    ff2: FeedForward,
    attention: Attention,
    convolution: Convolution,
    attention_norm: Norm,
    output_norm: Norm,
}

struct SubsampleConv {
    weight: CudaSlice<f32>,
    bias: CudaSlice<f32>,
}

struct Subsample {
    conv: [SubsampleConv; 3],
    pointwise: [Linear; 2],
}

struct Ops {
    conv2d: CudaFunction,
    mask_time: CudaFunction,
    pack: CudaFunction,
    depthwise: CudaFunction,
    batch_norm: CudaFunction,
    residual: CudaFunction,
    relative_attention: CudaFunction,
    activation: CudaFunction,
    bias: CudaFunction,
    layer_norm: CudaFunction,
}

pub struct ParakeetEncoder {
    conformer: FastConformer,
    stream: Arc<CudaStream>,
    subsample: Subsample,
    subsample_linear: Linear,
    blas: CudaBlas,
    blocks: Vec<Block>,
    ops: Ops,
    #[cfg(test)]
    encode_calls: AtomicU64,
}

struct SubsampleOutput {
    hidden: CudaSlice<f32>,
    mask: Vec<u8>,
    valid: usize,
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
    bias: bool,
) -> Result<Linear> {
    linear_shaped(
        checkpoint,
        stream,
        name,
        output,
        input,
        bias,
        &[output as usize, input as usize],
    )
}

fn linear_shaped(
    checkpoint: &MappedCheckpoint,
    stream: &Arc<CudaStream>,
    name: &str,
    output: u32,
    input: u32,
    bias: bool,
    stored_shape: &[usize],
) -> Result<Linear> {
    Ok(Linear {
        weight: tensor(checkpoint, stream, &format!("{name}.weight"), stored_shape)?,
        bias: if bias {
            Some(tensor(
                checkpoint,
                stream,
                &format!("{name}.bias"),
                &[output as usize],
            )?)
        } else {
            None
        },
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

fn flat(count: usize) -> LaunchConfig {
    super::ops::flat(count)
}

fn relative_positions(rows: usize, hidden: usize) -> Result<Vec<f32>> {
    ensure!(
        rows > 0 && hidden >= 2 && hidden.is_multiple_of(2),
        "relative position shape"
    );
    let inverse: Vec<f32> = (0..hidden / 2)
        .map(|index| 1.0f32 / 10000f32.powf((2 * index) as f32 / hidden as f32))
        .collect();
    let mut positions = Vec::with_capacity((2 * rows - 1) * hidden);
    for row in 0..2 * rows - 1 {
        let position = (rows as isize - 1 - row as isize) as f32;
        for &frequency in &inverse {
            let phase = position * frequency;
            positions.push(phase.sin());
            positions.push(phase.cos());
        }
    }
    Ok(positions)
}

struct AttentionLaunch<'a> {
    q: &'a CudaSlice<f32>,
    k: &'a CudaSlice<f32>,
    v: &'a CudaSlice<f32>,
    relative: &'a CudaSlice<f32>,
    bias_u: &'a CudaSlice<f32>,
    bias_v: &'a CudaSlice<f32>,
    mask: &'a CudaSlice<u8>,
    output: &'a CudaSlice<f32>,
    rows: u32,
    key_rows: u32,
    query_offset: u32,
    kv_offset: u32,
    right_chunks: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum StreamOmit {
    None,
    Keys,
    Conv,
}

pub struct StreamChunk {
    pub subsample: Vec<f32>,
    pub layer0: Vec<f32>,
    pub encoder: Vec<f32>,
}

struct ChunkCaches {
    conv: [CudaSlice<f32>; 3],
    keys: Vec<CudaSlice<f32>>,
    values: Vec<CudaSlice<f32>>,
    depthwise: Vec<CudaSlice<f32>>,
}

struct StreamWindow {
    key_rows: u32,
    query_offset: u32,
    kv_offset: u32,
    cached: u32,
    omit: StreamOmit,
}

struct LayerStep {
    hidden: CudaSlice<f32>,
    key: CudaSlice<f32>,
    value: CudaSlice<f32>,
    depthwise: CudaSlice<f32>,
}

pub struct NemotronStream {
    tables: Vec<CudaSlice<f32>>,
    keys: Vec<CudaSlice<f32>>,
    values: Vec<CudaSlice<f32>>,
    depthwise: Vec<CudaSlice<f32>>,
    conv: [Option<CudaSlice<f32>>; 3],
    mask: CudaSlice<u8>,
    key_frames: usize,
    cumulative: u32,
    chunks: u32,
}

impl NemotronStream {
    pub fn key_frames(&self) -> usize {
        self.key_frames
    }

    pub fn cumulative(&self) -> u32 {
        self.cumulative
    }

    pub fn chunks(&self) -> u32 {
        self.chunks
    }
}

fn table_index(key_rows: u32) -> Result<usize> {
    ensure!(
        key_rows.is_multiple_of(4) && (4..=60).contains(&key_rows),
        "streaming key rows {key_rows}"
    );
    Ok((key_rows / 4 - 1) as usize)
}

fn stream_window(cumulative_before: u32, query_rows: u32) -> Result<StreamWindow> {
    ensure!(query_rows == 4, "streaming query rows {query_rows}");
    let cached = if cumulative_before >= 57 {
        56
    } else {
        cumulative_before
    };
    let kv_offset = if cumulative_before >= 57 {
        cumulative_before - 56
    } else {
        0
    };
    let key_rows = cached + query_rows;
    table_index(key_rows)?;
    Ok(StreamWindow {
        key_rows,
        query_offset: cumulative_before,
        kv_offset,
        cached,
        omit: StreamOmit::None,
    })
}

#[cfg(test)]
fn absolute_key_kept(
    q_abs: u32,
    k_abs: u32,
    chunk_size: u32,
    left_chunks: u32,
    right_chunks: u32,
) -> bool {
    let q_chunk = q_abs / chunk_size;
    let k_chunk = k_abs / chunk_size;
    let start = q_chunk.saturating_sub(left_chunks);
    (start..q_chunk + 1 + right_chunks).contains(&k_chunk)
}

impl ParakeetEncoder {
    pub fn load(directory: &Path, device: &Device) -> Result<Self> {
        let spec = ParakeetSpec::read(directory)?;
        Self::load_conformer(directory, device, &spec.conformer())
    }

    pub fn load_canary(directory: &Path, device: &Device) -> Result<Self> {
        let spec = crate::canary::CanarySpec::read(directory)?;
        Self::load_conformer(directory, device, &spec.conformer())
    }

    pub fn load_nemotron(directory: &Path, device: &Device) -> Result<Self> {
        let spec = crate::nemotron::NemotronSpec::read(directory)?;
        Self::load_conformer(directory, device, &spec.conformer()?)
    }

    #[cfg(test)]
    pub(crate) fn reset_encode_counters() {
        ENCODE_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
        ENCODE_LAUNCHES.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn encode_calls() -> u64 {
        ENCODE_CALLS.load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn encode_launches() -> u64 {
        ENCODE_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn reset_position_uploads() {
        POSITION_UPLOADS.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn position_uploads() -> u64 {
        POSITION_UPLOADS.load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn attention_query_rows() -> u64 {
        ATTENTION_QUERY_ROWS.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn load_conformer(
        directory: &Path,
        device: &Device,
        conformer: &FastConformer,
    ) -> Result<Self> {
        let checkpoint = MappedCheckpoint::open(directory)?;
        let root = conformer.prefix;
        let stream = Arc::clone(&device.stream);
        let blas = CudaBlas::new(Arc::clone(&stream))?;
        unsafe {
            cublas_sys::cublasSetMathMode(
                *blas.handle(),
                cublas_sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
            )
            .result()?;
        }
        let speech = cubin::module(&device.ctx, "speech_ops")?;
        let vision = cubin::module(&device.ctx, "vision_ops")?;
        let common = cubin::module(&device.ctx, "q38_common")?;
        let ops = Ops {
            conv2d: speech.load_function("speech_conv2d_f32")?,
            mask_time: speech.load_function("speech_mask_time_f32")?,
            pack: speech.load_function("speech_pack_subsample_f32")?,
            depthwise: speech.load_function("speech_depthwise_conv_f32")?,
            batch_norm: speech.load_function("speech_batch_norm_f32")?,
            residual: speech.load_function("speech_residual_f32")?,
            relative_attention: speech.load_function("speech_relative_attention_f32")?,
            activation: vision.load_function("activation_f32")?,
            bias: vision.load_function("vision_bias_f32")?,
            layer_norm: common.load_function("q38_layer_norm_f32")?,
        };
        let mut subsample_conv = Vec::new();
        for name in conformer.conv_names {
            subsample_conv.push(SubsampleConv {
                weight: tensor(
                    &checkpoint,
                    &stream,
                    &format!("{root}.{name}.weight"),
                    &[256, 1, 3, 3],
                )?,
                bias: tensor(&checkpoint, &stream, &format!("{root}.{name}.bias"), &[256])?,
            });
        }
        let mut subsample_pointwise = Vec::new();
        for name in conformer.pointwise_names {
            subsample_pointwise.push(linear_shaped(
                &checkpoint,
                &stream,
                &format!("{root}.{name}"),
                256,
                256,
                true,
                &[256, 256, 1, 1],
            )?);
        }
        let subsample_linear = linear(
            &checkpoint,
            &stream,
            &format!("{root}.subsampling.linear"),
            1024,
            conformer.linear_in,
            true,
        )?;
        let subsample = Subsample {
            conv: subsample_conv
                .try_into()
                .map_err(|_| anyhow::anyhow!("subsample convolution count"))?,
            pointwise: subsample_pointwise
                .try_into()
                .map_err(|_| anyhow::anyhow!("subsample pointwise count"))?,
        };
        let mut blocks = Vec::new();
        for index in 0..conformer.layers {
            let prefix = format!("{root}.layers.{index}");
            let attention_bias = conformer.attention_bias;
            let convolution_bias = conformer.convolution_bias;
            let feed_forward = |branch: &str| -> Result<FeedForward> {
                Ok(FeedForward {
                    norm: norm(&checkpoint, &stream, &format!("{prefix}.norm_{branch}"))?,
                    up: linear(
                        &checkpoint,
                        &stream,
                        &format!("{prefix}.{branch}.linear1"),
                        4096,
                        1024,
                        attention_bias,
                    )?,
                    down: linear(
                        &checkpoint,
                        &stream,
                        &format!("{prefix}.{branch}.linear2"),
                        1024,
                        4096,
                        attention_bias,
                    )?,
                })
            };
            let attention = |name: &str, bias: bool| -> Result<Linear> {
                linear(
                    &checkpoint,
                    &stream,
                    &format!("{prefix}.self_attn.{name}"),
                    1024,
                    1024,
                    bias,
                )
            };
            blocks.push(Block {
                ff1: feed_forward("feed_forward1")?,
                ff2: feed_forward("feed_forward2")?,
                attention: Attention {
                    q: attention("q_proj", attention_bias)?,
                    k: attention("k_proj", attention_bias)?,
                    v: attention("v_proj", attention_bias)?,
                    o: attention("o_proj", attention_bias)?,
                    relative: attention("relative_k_proj", false)?,
                    bias_u: tensor(
                        &checkpoint,
                        &stream,
                        &format!("{prefix}.self_attn.bias_u"),
                        &[8, 128],
                    )?,
                    bias_v: tensor(
                        &checkpoint,
                        &stream,
                        &format!("{prefix}.self_attn.bias_v"),
                        &[8, 128],
                    )?,
                },
                convolution: Convolution {
                    norm: norm(&checkpoint, &stream, &format!("{prefix}.norm_conv"))?,
                    pointwise1: linear_shaped(
                        &checkpoint,
                        &stream,
                        &format!("{prefix}.conv.pointwise_conv1"),
                        2048,
                        1024,
                        convolution_bias,
                        &[2048, 1024, 1],
                    )?,
                    depthwise: tensor(
                        &checkpoint,
                        &stream,
                        &format!("{prefix}.conv.depthwise_conv.weight"),
                        &[1024, 1, conformer.conv_kernel as usize],
                    )?,
                    depthwise_bias: if convolution_bias {
                        Some(tensor(
                            &checkpoint,
                            &stream,
                            &format!("{prefix}.conv.depthwise_conv.bias"),
                            &[1024],
                        )?)
                    } else {
                        None
                    },
                    conv_weight: tensor(
                        &checkpoint,
                        &stream,
                        &format!("{prefix}.conv.norm.weight"),
                        &[1024],
                    )?,
                    conv_bias: tensor(
                        &checkpoint,
                        &stream,
                        &format!("{prefix}.conv.norm.bias"),
                        &[1024],
                    )?,
                    running_mean: if conformer.conv_batch_norm {
                        Some(tensor(
                            &checkpoint,
                            &stream,
                            &format!("{prefix}.conv.norm.running_mean"),
                            &[1024],
                        )?)
                    } else {
                        None
                    },
                    running_var: if conformer.conv_batch_norm {
                        Some(tensor(
                            &checkpoint,
                            &stream,
                            &format!("{prefix}.conv.norm.running_var"),
                            &[1024],
                        )?)
                    } else {
                        None
                    },
                    pointwise2: linear_shaped(
                        &checkpoint,
                        &stream,
                        &format!("{prefix}.conv.pointwise_conv2"),
                        1024,
                        1024,
                        convolution_bias,
                        &[1024, 1024, 1],
                    )?,
                },
                attention_norm: norm(&checkpoint, &stream, &format!("{prefix}.norm_self_att"))?,
                output_norm: norm(&checkpoint, &stream, &format!("{prefix}.norm_out"))?,
            });
        }
        stream.synchronize()?;
        Ok(Self {
            conformer: *conformer,
            stream,
            subsample,
            subsample_linear,
            blas,
            blocks,
            ops,
            #[cfg(test)]
            encode_calls: AtomicU64::new(0),
        })
    }

    fn linear(&self, input: &CudaSlice<f32>, rows: u32, weight: &Linear) -> Result<CudaSlice<f32>> {
        ensure!(rows > 0, "linear requires rows");
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
        }
        note_launch!();
        if let Some(bias) = &weight.bias {
            unsafe {
                self.stream
                    .launch_builder(&self.ops.bias)
                    .arg(&output)
                    .arg(bias)
                    .arg(&output)
                    .arg(&rows)
                    .arg(&weight.output)
                    .launch(flat(output.len()))?;
            }
            note_launch!();
        }
        Ok(output)
    }

    fn activate(
        &self,
        input: &CudaSlice<f32>,
        rows: u32,
        width: u32,
        mode: i32,
    ) -> Result<CudaSlice<f32>> {
        let expected = rows as usize * width as usize * if mode == 2 { 2 } else { 1 };
        ensure!(
            input.len() == expected,
            "activation shape mismatch: rows={rows} width={width} mode={mode} input={} expected={expected}",
            input.len()
        );
        let output = self.stream.alloc_zeros(if mode == 2 {
            input.len() / 2
        } else {
            input.len()
        })?;
        unsafe {
            self.stream
                .launch_builder(&self.ops.activation)
                .arg(input)
                .arg(&output)
                .arg(&rows)
                .arg(&width)
                .arg(&mode)
                .launch(flat(input.len()))?;
        }
        note_launch!();
        Ok(output)
    }

    fn normalize(&self, input: &CudaSlice<f32>, rows: u32, norm: &Norm) -> Result<CudaSlice<f32>> {
        self.normalize_pair(input, rows, &norm.weight, &norm.bias)
    }

    fn normalize_pair(
        &self,
        input: &CudaSlice<f32>,
        rows: u32,
        weight: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>> {
        let output = self.stream.alloc_zeros(input.len())?;
        let width = self.conformer.hidden;
        unsafe {
            self.stream
                .launch_builder(&self.ops.layer_norm)
                .arg(input)
                .arg(weight)
                .arg(bias)
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
                .launch(super::ops::grid(rows as usize, 256))?;
        }
        note_launch!();
        Ok(output)
    }

    fn residual(
        &self,
        hidden: &mut CudaSlice<f32>,
        delta: &CudaSlice<f32>,
        scale: f32,
    ) -> Result<()> {
        let count = hidden.len() as u64;
        unsafe {
            self.stream
                .launch_builder(&self.ops.residual)
                .arg(&*hidden)
                .arg(delta)
                .arg(&*hidden)
                .arg(&count)
                .arg(&scale)
                .launch(flat(hidden.len()))?;
        }
        note_launch!();
        Ok(())
    }

    fn subsample_device(&self, input: &[f32], valid: usize) -> Result<SubsampleOutput> {
        ensure!(
            valid > 0
                && valid * 128 <= input.len()
                && input.len().is_multiple_of(128)
                && input.iter().all(|value| value.is_finite()),
            "invalid mel extent or non-finite input"
        );
        let mut hidden = self.stream.clone_htod(input)?;
        let mut time = input.len() / 128;
        let mut frequency = 128usize;
        let mut current_valid = valid;
        let pad_left = self.conformer.pad_left;
        let pad_right = self.conformer.pad_right;
        for (index, conv) in self.subsample.conv.iter().enumerate() {
            let next_time = conv_out(time, pad_left, pad_right)?;
            let next_frequency = conv_out(frequency, pad_left, pad_right)?;
            let next_valid = conv_out(current_valid, pad_left, pad_right)?;
            let output = self.stream.alloc_zeros(next_time * next_frequency * 256)?;
            let in_channels = if index == 0 { 1u32 } else { 256 };
            let groups = if index == 0 { 1u32 } else { 256 };
            unsafe {
                self.stream
                    .launch_builder(&self.ops.conv2d)
                    .arg(&hidden)
                    .arg(&conv.weight)
                    .arg(&conv.bias)
                    .arg(&output)
                    .arg(&u32::try_from(time)?)
                    .arg(&u32::try_from(frequency)?)
                    .arg(&in_channels)
                    .arg(&256u32)
                    .arg(&groups)
                    .arg(&u32::try_from(next_valid)?)
                    .arg(&pad_left)
                    .arg(&pad_right)
                    .launch(flat(output.len()))?;
            }
            note_launch!();
            hidden = output;
            time = next_time;
            frequency = next_frequency;
            current_valid = next_valid;
            let spatial = u32::try_from(time * frequency)?;
            if index > 0 {
                hidden = self.linear(&hidden, spatial, &self.subsample.pointwise[index - 1])?;
                unsafe {
                    self.stream
                        .launch_builder(&self.ops.mask_time)
                        .arg(&hidden)
                        .arg(&spatial)
                        .arg(&u32::try_from(time)?)
                        .arg(&u32::try_from(frequency)?)
                        .arg(&256u32)
                        .arg(&u32::try_from(current_valid)?)
                        .launch(flat(hidden.len()))?;
                }
                note_launch!();
            }
            hidden = self.activate(&hidden, spatial, 256, 0)?;
        }
        ensure!(
            frequency * 256 == self.subsample_linear.input as usize,
            "packed frequency {frequency} does not match linear input {}",
            self.subsample_linear.input
        );
        let packed = self.stream.alloc_zeros(hidden.len())?;
        unsafe {
            self.stream
                .launch_builder(&self.ops.pack)
                .arg(&hidden)
                .arg(&packed)
                .arg(&u32::try_from(time)?)
                .arg(&u32::try_from(frequency)?)
                .arg(&256u32)
                .launch(flat(packed.len()))?;
        }
        note_launch!();
        let hidden = self.linear(&packed, u32::try_from(time)?, &self.subsample_linear)?;
        let mask = (0..time)
            .map(|index| u8::from(index < current_valid))
            .collect();
        Ok(SubsampleOutput {
            hidden,
            mask,
            valid: current_valid,
        })
    }

    /// Input is `[frames, 128]`. Output is the subsampled `[rows, 1024]` and its mask.
    pub fn subsample(&self, input: &[f32], valid: usize) -> Result<(Vec<f32>, Vec<u8>)> {
        let output = self.subsample_device(input, valid)?;
        Ok((self.stream.clone_dtoh(&output.hidden)?, output.mask))
    }

    fn positions(&self, rows: usize) -> Result<Vec<f32>> {
        position_rows_fit(rows, self.conformer.max_position)?;
        relative_positions(rows, self.conformer.hidden as usize)
    }

    fn check_right(&self, right_chunks: u32) -> Result<()> {
        ensure!(right_chunks <= 1, "right_chunks {right_chunks} exceeds 1");
        ensure!(
            self.conformer.chunk_size > 0 || right_chunks == 0,
            "right_chunks requires a chunk mask"
        );
        Ok(())
    }

    fn layer_device(
        &self,
        index: usize,
        input: &CudaSlice<f32>,
        mask: &[u8],
        positions: &CudaSlice<f32>,
        right_chunks: u32,
    ) -> Result<CudaSlice<f32>> {
        self.check_right(right_chunks)?;
        let block = self
            .blocks
            .get(index)
            .context("encoder layer outside range")?;
        let rows = u32::try_from(mask.len())?;
        ensure!(
            rows > 0
                && input.len() == rows as usize * self.conformer.hidden as usize
                && mask.iter().all(|value| *value <= 1)
                && positions.len() == (2 * rows - 1) as usize * self.conformer.hidden as usize,
            "encoder layer input, mask or position shape mismatch"
        );
        let mask = self.stream.clone_htod(mask)?;
        let feed_forward =
            |hidden: &mut CudaSlice<f32>, ff: &FeedForward, scale: f32| -> Result<()> {
                let normalized = self.normalize(hidden, rows, &ff.norm)?;
                let expanded = self.linear(&normalized, rows, &ff.up)?;
                let activated = self.activate(&expanded, rows, 4096, 1)?;
                let delta = self.linear(&activated, rows, &ff.down)?;
                self.residual(hidden, &delta, scale)
            };
        let mut hidden = input.try_clone()?;
        feed_forward(&mut hidden, &block.ff1, 0.5)?;
        let normalized = self.normalize(&hidden, rows, &block.attention_norm)?;
        let q = self.linear(&normalized, rows, &block.attention.q)?;
        let k = self.linear(&normalized, rows, &block.attention.k)?;
        let v = self.linear(&normalized, rows, &block.attention.v)?;
        let relative = self.linear(positions, 2 * rows - 1, &block.attention.relative)?;
        let core = self.stream.alloc_zeros(hidden.len())?;
        self.attend(AttentionLaunch {
            q: &q,
            k: &k,
            v: &v,
            relative: &relative,
            bias_u: &block.attention.bias_u,
            bias_v: &block.attention.bias_v,
            mask: &mask,
            output: &core,
            rows,
            key_rows: rows,
            query_offset: 0,
            kv_offset: 0,
            right_chunks,
        })?;
        let delta = self.linear(&core, rows, &block.attention.o)?;
        self.residual(&mut hidden, &delta, 1.0)?;
        let normalized = self.normalize(&hidden, rows, &block.convolution.norm)?;
        let expanded = self.linear(&normalized, rows, &block.convolution.pointwise1)?;
        let activated = self.activate(&expanded, rows, self.conformer.hidden, 2)?;
        let convolved: CudaSlice<f32> = self.stream.alloc_zeros(hidden.len())?;
        unsafe {
            self.stream
                .launch_builder(&self.ops.depthwise)
                .arg(&activated)
                .arg(&block.convolution.depthwise)
                .arg(&mask)
                .arg(&convolved)
                .arg(&rows)
                .arg(&self.conformer.hidden)
                .arg(&self.conformer.conv_kernel)
                .arg(&self.conformer.depthwise_left)
                .launch(flat(convolved.len()))?;
        }
        note_launch!();
        if let Some(bias) = &block.convolution.depthwise_bias {
            unsafe {
                self.stream
                    .launch_builder(&self.ops.bias)
                    .arg(&convolved)
                    .arg(bias)
                    .arg(&convolved)
                    .arg(&rows)
                    .arg(&self.conformer.hidden)
                    .launch(flat(convolved.len()))?;
            }
            note_launch!();
        }
        let normalized = if self.conformer.conv_batch_norm {
            let mean = block
                .convolution
                .running_mean
                .as_ref()
                .context("batch-norm convolution is missing running_mean")?;
            let variance = block
                .convolution
                .running_var
                .as_ref()
                .context("batch-norm convolution is missing running_var")?;
            let normalized = self.stream.alloc_zeros(hidden.len())?;
            unsafe {
                self.stream
                    .launch_builder(&self.ops.batch_norm)
                    .arg(&convolved)
                    .arg(&block.convolution.conv_weight)
                    .arg(&block.convolution.conv_bias)
                    .arg(mean)
                    .arg(variance)
                    .arg(&normalized)
                    .arg(&rows)
                    .arg(&self.conformer.hidden)
                    .arg(&1e-5f32)
                    .launch(flat(normalized.len()))?;
            }
            note_launch!();
            normalized
        } else {
            ensure!(
                block.convolution.running_mean.is_none() && block.convolution.running_var.is_none(),
                "layer-norm convolution has batch-norm statistics"
            );
            self.normalize_pair(
                &convolved,
                rows,
                &block.convolution.conv_weight,
                &block.convolution.conv_bias,
            )?
        };
        let activated = self.activate(&normalized, rows, self.conformer.hidden, 1)?;
        let delta = self.linear(&activated, rows, &block.convolution.pointwise2)?;
        self.residual(&mut hidden, &delta, 1.0)?;
        feed_forward(&mut hidden, &block.ff2, 0.5)?;
        let output = self.normalize(&hidden, rows, &block.output_norm)?;
        Ok(output)
    }

    /// Input and output are `[frames, 1024]` in native F32.
    pub fn layer(&self, index: usize, input: &[f32], mask: &[u8]) -> Result<Vec<f32>> {
        self.layer_right(index, input, mask, 0)
    }

    pub(crate) fn layer_right(
        &self,
        index: usize,
        input: &[f32],
        mask: &[u8],
        right_chunks: u32,
    ) -> Result<Vec<f32>> {
        let positions = self.stream.clone_htod(&self.positions(mask.len())?)?;
        let input = self.stream.clone_htod(input)?;
        let output = self.layer_device(index, &input, mask, &positions, right_chunks)?;
        Ok(self.stream.clone_dtoh(&output)?)
    }

    /// Input is `[frames, 128]`. Output is `[rows, 1024]`, its mask and valid row count.
    pub fn encode(&self, input: &[f32], valid: usize) -> Result<(Vec<f32>, Vec<u8>, usize)> {
        self.encode_right(input, valid, 0)
    }

    pub(crate) fn encode_right(
        &self,
        input: &[f32],
        valid: usize,
        right_chunks: u32,
    ) -> Result<(Vec<f32>, Vec<u8>, usize)> {
        #[cfg(test)]
        {
            self.encode_calls.fetch_add(1, Ordering::Relaxed);
            ENCODE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        self.check_right(right_chunks)?;
        let subsampled = self.subsample_device(input, valid)?;
        let positions = self
            .stream
            .clone_htod(&self.positions(subsampled.mask.len())?)?;
        let mut hidden = subsampled.hidden;
        for index in 0..self.blocks.len() {
            hidden =
                self.layer_device(index, &hidden, &subsampled.mask, &positions, right_chunks)?;
        }
        Ok((
            self.stream.clone_dtoh(&hidden)?,
            subsampled.mask,
            subsampled.valid,
        ))
    }

    #[cfg(test)]
    pub(crate) fn encode_calls(&self) -> u64 {
        self.encode_calls.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn reset_encode_calls(&self) {
        self.encode_calls.store(0, Ordering::Relaxed);
    }

    fn attend(&self, launch: AttentionLaunch<'_>) -> Result<()> {
        let hidden = self.conformer.hidden as usize;
        let rows = launch.rows as usize;
        let key_rows = launch.key_rows as usize;
        self.check_right(launch.right_chunks)?;
        ensure!(
            key_rows >= rows
                && launch.k.len() == key_rows * hidden
                && launch.v.len() == key_rows * hidden
                && launch.q.len() == rows * hidden
                && launch.output.len() == rows * hidden
                && launch.relative.len() >= (rows + key_rows - 1) * hidden
                && launch.mask.len() >= key_rows
                && (self.conformer.chunk_size != 0
                    || (launch.query_offset == 0
                        && launch.kv_offset == 0
                        && launch.key_rows == launch.rows)),
            "attention launch shape rows={rows} key_rows={key_rows}"
        );
        unsafe {
            self.stream
                .launch_builder(&self.ops.relative_attention)
                .arg(launch.q)
                .arg(launch.k)
                .arg(launch.v)
                .arg(launch.relative)
                .arg(launch.bias_u)
                .arg(launch.bias_v)
                .arg(launch.mask)
                .arg(launch.output)
                .arg(&launch.rows)
                .arg(&self.conformer.heads)
                .arg(&(1.0f32 / 128f32.sqrt()))
                .arg(&self.conformer.chunk_size)
                .arg(&self.conformer.left_chunks)
                .arg(&launch.right_chunks)
                .arg(&launch.key_rows)
                .arg(&launch.query_offset)
                .arg(&launch.kv_offset)
                .launch(LaunchConfig {
                    grid_dim: (launch.rows, self.conformer.heads, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: (8 + launch.key_rows) * 4,
                })?;
        }
        note_launch!();
        #[cfg(test)]
        ATTENTION_QUERY_ROWS.store(u64::from(launch.rows), std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn copy_device(
        &self,
        source: &CudaSlice<f32>,
        source_at: usize,
        dest: &mut CudaSlice<f32>,
        dest_at: usize,
        count: usize,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        ensure!(
            source_at
                .checked_add(count)
                .is_some_and(|end| end <= source.len())
                && dest_at
                    .checked_add(count)
                    .is_some_and(|end| end <= dest.len()),
            "device copy exceeds allocation"
        );
        self.stream.memcpy_dtod(
            &source.slice(source_at..source_at + count),
            &mut dest.slice_mut(dest_at..dest_at + count),
        )?;
        Ok(())
    }

    fn freq_pad(
        &self,
        input: &CudaSlice<f32>,
        time: usize,
        frequency: usize,
        channels: usize,
    ) -> Result<CudaSlice<f32>> {
        let left = self.conformer.pad_left as usize;
        let right = self.conformer.pad_right as usize;
        let padded_freq = frequency + left + right;
        let mut padded = self.stream.alloc_zeros(time * padded_freq * channels)?;
        let row = frequency * channels;
        let dest_row = padded_freq * channels;
        for frame in 0..time {
            self.copy_device(
                input,
                frame * row,
                &mut padded,
                frame * dest_row + left * channels,
                row,
            )?;
        }
        Ok(padded)
    }

    fn time_prepend(
        &self,
        input: &CudaSlice<f32>,
        frequency: usize,
        channels: usize,
        cache: Option<&CudaSlice<f32>>,
        zero_frames: usize,
    ) -> Result<CudaSlice<f32>> {
        let frame = frequency * channels;
        let extra = if let Some(cache) = cache {
            ensure!(
                cache.len() == frame && zero_frames == 0,
                "streaming conv prepend frame {frame}"
            );
            1
        } else {
            ensure!(zero_frames > 0, "streaming conv prepend is empty");
            zero_frames
        };
        let mut output = self.stream.alloc_zeros(input.len() + extra * frame)?;
        if let Some(cache) = cache {
            self.copy_device(cache, 0, &mut output, 0, frame)?;
        }
        self.copy_device(input, 0, &mut output, extra * frame, input.len())?;
        Ok(output)
    }

    /// Fifteen relative tables, L = 4, 8, ..., 60. `945 * 1024 * 4 = 3_870_720` bytes, uploaded once.
    pub fn stream(&self) -> Result<NemotronStream> {
        let conformer = &self.conformer;
        ensure!(
            conformer.chunk_size == 4
                && conformer.left_chunks == 14
                && conformer.pad_left == 2
                && conformer.pad_right == 1
                && conformer.depthwise_left == 8
                && conformer.hidden == 1024
                && conformer.heads == 8
                && conformer.layers == 24
                && conformer.conv_kernel == 9
                && !conformer.conv_batch_norm
                && !conformer.convolution_bias
                && self.blocks.len() == 24,
            "streaming session requires the published Nemotron encoder"
        );
        let mut tables = Vec::with_capacity(15);
        let mut bytes = 0usize;
        for step in 1..=15 {
            let values = self.positions(4 * step)?;
            bytes += values.len() * 4;
            tables.push(self.stream.clone_htod(&values)?);
        }
        ensure!(
            bytes == 3_870_720 && tables.len() == 15,
            "resident position tables are {bytes} bytes across {} lengths",
            tables.len()
        );
        let mut keys = Vec::with_capacity(24);
        let mut values = Vec::with_capacity(24);
        let mut depthwise = Vec::with_capacity(24);
        for _ in 0..24 {
            keys.push(self.stream.null()?);
            values.push(self.stream.null()?);
            depthwise.push(self.stream.alloc_zeros(8 * 1024)?);
        }
        #[cfg(test)]
        POSITION_UPLOADS.fetch_add(15, std::sync::atomic::Ordering::Relaxed);
        Ok(NemotronStream {
            tables,
            keys,
            values,
            depthwise,
            conv: [None, None, None],
            mask: self.stream.clone_htod(&vec![1u8; 60])?,
            key_frames: 0,
            cumulative: 0,
            chunks: 0,
        })
    }

    pub fn encode_chunk(&self, session: &mut NemotronStream, mel: &[f32]) -> Result<StreamChunk> {
        self.encode_chunk_omitting(session, mel, StreamOmit::None)
    }

    pub fn encode_chunk_omitting(
        &self,
        session: &mut NemotronStream,
        mel: &[f32],
        omit: StreamOmit,
    ) -> Result<StreamChunk> {
        #[cfg(test)]
        {
            ENCODE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            ATTENTION_QUERY_ROWS.store(0, std::sync::atomic::Ordering::Relaxed);
        }
        let (output, caches) = self.run_chunk(session, mel, omit)?;
        if omit == StreamOmit::None {
            self.commit_chunk(session, caches)?;
        }
        Ok(output)
    }

    fn commit_chunk(&self, session: &mut NemotronStream, caches: ChunkCaches) -> Result<()> {
        let next_frames = (session.key_frames + 4).min(56);
        let width = self.conformer.hidden as usize;
        ensure!(
            caches.keys.len() == 24
                && caches.values.len() == 24
                && caches.depthwise.len() == 24
                && caches
                    .keys
                    .iter()
                    .zip(caches.values.iter())
                    .all(
                        |(key, value)| key.len() == next_frames * width && value.len() == key.len()
                    )
                && caches
                    .depthwise
                    .iter()
                    .all(|cache| cache.len() == 8 * width)
                && caches.conv[0].len() == 131
                && caches.conv[1].len() == 256 * 68
                && caches.conv[2].len() == 256 * 36,
            "streaming cache commit shape"
        );
        session.keys = caches.keys;
        session.values = caches.values;
        session.depthwise = caches.depthwise;
        let [first, second, third] = caches.conv;
        session.conv = [Some(first), Some(second), Some(third)];
        session.key_frames = next_frames;
        session.cumulative += 4;
        session.chunks += 1;
        Ok(())
    }

    fn run_chunk(
        &self,
        session: &NemotronStream,
        mel: &[f32],
        omit: StreamOmit,
    ) -> Result<(StreamChunk, ChunkCaches)> {
        ensure!(
            mel.len().is_multiple_of(128) && mel.iter().all(|value| value.is_finite()),
            "streaming mel is misaligned or non-finite"
        );
        let frames = mel.len() / 128;
        let expected = if session.chunks == 0 { 25 } else { 32 };
        ensure!(
            frames == expected,
            "streaming mel frames {frames} require {expected}"
        );
        let mut window = stream_window(session.cumulative, 4)?;
        window.omit = omit;
        ensure!(
            session.key_frames == window.cached as usize
                && session.cumulative == session.chunks * 4
                && session.keys.len() == 24
                && session.values.len() == 24
                && session.depthwise.len() == 24
                && session.tables.len() == 15,
            "torn streaming cache: stored {} cumulative {} chunks {}",
            session.key_frames,
            session.cumulative,
            session.chunks
        );
        if omit == StreamOmit::Keys {
            ensure!(session.key_frames > 0, "omit keys requires stored keys");
        }
        if omit == StreamOmit::Conv {
            ensure!(
                session.conv.iter().all(|frame| frame.is_some()),
                "omit conv requires a stored frame"
            );
        }
        let (mut hidden, conv) = self.subsample_stream(mel, session, omit)?;
        let subsample = self.stream.clone_dtoh(&hidden)?;
        let mut keys = Vec::with_capacity(24);
        let mut values = Vec::with_capacity(24);
        let mut depthwise = Vec::with_capacity(24);
        let mut layer0 = None;
        for index in 0..self.blocks.len() {
            let step = self.layer_stream(index, &hidden, session, &window)?;
            hidden = step.hidden;
            if index == 0 {
                layer0 = Some(self.stream.clone_dtoh(&hidden)?);
            }
            keys.push(step.key);
            values.push(step.value);
            depthwise.push(step.depthwise);
        }
        let layer0 = layer0.context("streaming layer 0 missing")?;
        ensure!(
            subsample.len() == 4 * 1024 && layer0.len() == subsample.len(),
            "streaming chunk rows"
        );
        let encoder = self.stream.clone_dtoh(&hidden)?;
        ensure!(encoder.len() == subsample.len(), "streaming encoder rows");
        Ok((
            StreamChunk {
                subsample,
                layer0,
                encoder,
            },
            ChunkCaches {
                conv,
                keys,
                values,
                depthwise,
            },
        ))
    }

    fn subsample_stream(
        &self,
        input: &[f32],
        session: &NemotronStream,
        omit: StreamOmit,
    ) -> Result<(CudaSlice<f32>, [CudaSlice<f32>; 3])> {
        let mut hidden = self.stream.clone_htod(input)?;
        let mut time = input.len() / 128;
        let mut frequency = 128usize;
        let mut channels = 1usize;
        let mut caches = Vec::with_capacity(3);
        for (index, conv) in self.subsample.conv.iter().enumerate() {
            ensure!(
                hidden.len() == time * frequency * channels,
                "subsampling stage {index} shape"
            );
            let padded = self.freq_pad(&hidden, time, frequency, channels)?;
            let padded_freq =
                frequency + self.conformer.pad_left as usize + self.conformer.pad_right as usize;
            let frame = padded_freq * channels;
            let mut cache = self.stream.alloc_zeros(frame)?;
            self.copy_device(&padded, (time - 1) * frame, &mut cache, 0, frame)?;
            caches.push(cache);
            let zero_frames = if session.chunks == 0 {
                2
            } else if omit == StreamOmit::Conv {
                1
            } else {
                0
            };
            let stored = if zero_frames == 0 {
                Some(
                    session.conv[index]
                        .as_ref()
                        .context("streaming conv cache is missing")?,
                )
            } else {
                None
            };
            let conv_input =
                self.time_prepend(&padded, padded_freq, channels, stored, zero_frames)?;
            let conv_time = time + if stored.is_some() { 1 } else { zero_frames };
            let next_time = conv_out(conv_time, 0, 0)?;
            let next_frequency = conv_out(padded_freq, 0, 0)?;
            let expect_time = if session.chunks == 0 {
                [13, 7, 4][index]
            } else {
                [16, 8, 4][index]
            };
            let expect_frequency = [65, 33, 17][index];
            ensure!(
                next_time == expect_time && next_frequency == expect_frequency,
                "streaming conv stage {index} produced {next_time}x{next_frequency}"
            );
            let output = self.stream.alloc_zeros(next_time * next_frequency * 256)?;
            let in_channels = if index == 0 { 1u32 } else { 256 };
            let conv_time_u = u32::try_from(conv_time)?;
            let padded_freq_u = u32::try_from(padded_freq)?;
            let next_time_u = u32::try_from(next_time)?;
            unsafe {
                self.stream
                    .launch_builder(&self.ops.conv2d)
                    .arg(&conv_input)
                    .arg(&conv.weight)
                    .arg(&conv.bias)
                    .arg(&output)
                    .arg(&conv_time_u)
                    .arg(&padded_freq_u)
                    .arg(&in_channels)
                    .arg(&256u32)
                    .arg(&in_channels)
                    .arg(&next_time_u)
                    .arg(&0u32)
                    .arg(&0u32)
                    .launch(flat(output.len()))?;
            }
            note_launch!();
            hidden = output;
            time = next_time;
            frequency = next_frequency;
            channels = 256;
            let spatial = u32::try_from(time * frequency)?;
            if index > 0 {
                hidden = self.linear(&hidden, spatial, &self.subsample.pointwise[index - 1])?;
                let time_u = u32::try_from(time)?;
                let frequency_u = u32::try_from(frequency)?;
                unsafe {
                    self.stream
                        .launch_builder(&self.ops.mask_time)
                        .arg(&hidden)
                        .arg(&spatial)
                        .arg(&time_u)
                        .arg(&frequency_u)
                        .arg(&256u32)
                        .arg(&time_u)
                        .launch(flat(hidden.len()))?;
                }
                note_launch!();
            }
            hidden = self.activate(&hidden, spatial, 256, 0)?;
        }
        ensure!(
            time == 4 && frequency == 17,
            "streaming subsample rows {time} frequency {frequency}"
        );
        let packed = self.stream.alloc_zeros(hidden.len())?;
        unsafe {
            self.stream
                .launch_builder(&self.ops.pack)
                .arg(&hidden)
                .arg(&packed)
                .arg(&u32::try_from(time)?)
                .arg(&u32::try_from(frequency)?)
                .arg(&256u32)
                .launch(flat(packed.len()))?;
        }
        note_launch!();
        let hidden = self.linear(&packed, u32::try_from(time)?, &self.subsample_linear)?;
        let caches: [CudaSlice<f32>; 3] = caches
            .try_into()
            .map_err(|_| anyhow::anyhow!("streaming conv cache count"))?;
        Ok((hidden, caches))
    }

    fn layer_stream(
        &self,
        index: usize,
        input: &CudaSlice<f32>,
        session: &NemotronStream,
        window: &StreamWindow,
    ) -> Result<LayerStep> {
        let block = self
            .blocks
            .get(index)
            .context("encoder layer outside range")?;
        let rows = 4u32;
        let width = self.conformer.hidden as usize;
        ensure!(
            input.len() == rows as usize * width && block.convolution.depthwise_bias.is_none(),
            "streaming layer input"
        );
        let feed_forward =
            |hidden: &mut CudaSlice<f32>, ff: &FeedForward, scale: f32| -> Result<()> {
                let normalized = self.normalize(hidden, rows, &ff.norm)?;
                let expanded = self.linear(&normalized, rows, &ff.up)?;
                let activated = self.activate(&expanded, rows, 4096, 1)?;
                let delta = self.linear(&activated, rows, &ff.down)?;
                self.residual(hidden, &delta, scale)
            };
        let mut hidden = input.try_clone()?;
        feed_forward(&mut hidden, &block.ff1, 0.5)?;
        let normalized = self.normalize(&hidden, rows, &block.attention_norm)?;
        let q = self.linear(&normalized, rows, &block.attention.q)?;
        let k = self.linear(&normalized, rows, &block.attention.k)?;
        let v = self.linear(&normalized, rows, &block.attention.v)?;
        let table = session
            .tables
            .get(table_index(window.key_rows)?)
            .context("streaming position table")?;
        let relative_rows = 2 * window.key_rows - 1;
        let relative = self.linear(table, relative_rows, &block.attention.relative)?;
        let attend_rows = if window.omit == StreamOmit::Keys {
            rows
        } else {
            window.key_rows
        };
        let attend_offset = if window.omit == StreamOmit::Keys {
            window.query_offset
        } else {
            window.kv_offset
        };
        let cached = window.cached as usize;
        ensure!(
            session.keys[index].len() == cached * width
                && session.values[index].len() == cached * width,
            "layer {index} key cache length"
        );
        let mut cat_k = self.stream.alloc_zeros(attend_rows as usize * width)?;
        let mut cat_v = self.stream.alloc_zeros(attend_rows as usize * width)?;
        let dest = if window.omit == StreamOmit::Keys {
            0
        } else {
            cached * width
        };
        if window.omit != StreamOmit::Keys {
            self.copy_device(&session.keys[index], 0, &mut cat_k, 0, cached * width)?;
            self.copy_device(&session.values[index], 0, &mut cat_v, 0, cached * width)?;
        }
        self.copy_device(&k, 0, &mut cat_k, dest, rows as usize * width)?;
        self.copy_device(&v, 0, &mut cat_v, dest, rows as usize * width)?;
        let core = self.stream.alloc_zeros(hidden.len())?;
        self.attend(AttentionLaunch {
            q: &q,
            k: &cat_k,
            v: &cat_v,
            relative: &relative,
            bias_u: &block.attention.bias_u,
            bias_v: &block.attention.bias_v,
            mask: &session.mask,
            output: &core,
            rows,
            key_rows: attend_rows,
            query_offset: window.query_offset,
            kv_offset: attend_offset,
            right_chunks: 0,
        })?;
        let delta = self.linear(&core, rows, &block.attention.o)?;
        self.residual(&mut hidden, &delta, 1.0)?;
        let combined = cached + rows as usize;
        let keep = combined.min(56);
        let drop = combined - keep;
        let carried = cached - drop;
        let mut next_k = self.stream.alloc_zeros(keep * width)?;
        let mut next_v = self.stream.alloc_zeros(keep * width)?;
        self.copy_device(
            &session.keys[index],
            drop * width,
            &mut next_k,
            0,
            carried * width,
        )?;
        self.copy_device(&k, 0, &mut next_k, carried * width, rows as usize * width)?;
        self.copy_device(
            &session.values[index],
            drop * width,
            &mut next_v,
            0,
            carried * width,
        )?;
        self.copy_device(&v, 0, &mut next_v, carried * width, rows as usize * width)?;
        let normalized = self.normalize(&hidden, rows, &block.convolution.norm)?;
        let expanded = self.linear(&normalized, rows, &block.convolution.pointwise1)?;
        let activated = self.activate(&expanded, rows, self.conformer.hidden, 2)?;
        let launch_rows = if session.chunks == 0 || window.omit == StreamOmit::Conv {
            rows
        } else {
            12
        };
        ensure!(
            session.depthwise[index].len() == 8 * width && activated.len() == rows as usize * width,
            "layer {index} depthwise cache"
        );
        let convolved = self.stream.alloc_zeros(launch_rows as usize * width)?;
        if launch_rows == 12 {
            let mut joined = self.stream.alloc_zeros(12 * width)?;
            self.copy_device(&session.depthwise[index], 0, &mut joined, 0, 8 * width)?;
            self.copy_device(&activated, 0, &mut joined, 8 * width, rows as usize * width)?;
            self.launch_depthwise(
                &joined,
                &block.convolution.depthwise,
                &session.mask,
                &convolved,
                launch_rows,
            )?;
        } else {
            self.launch_depthwise(
                &activated,
                &block.convolution.depthwise,
                &session.mask,
                &convolved,
                launch_rows,
            )?;
        }
        // Rows 8..11 are the 4 kernel-9 outputs of the 12-frame cache concatenation. Rows 0..7 are discarded.
        let kept = if launch_rows == 12 {
            let mut rows_kept = self.stream.alloc_zeros(rows as usize * width)?;
            self.copy_device(
                &convolved,
                8 * width,
                &mut rows_kept,
                0,
                rows as usize * width,
            )?;
            rows_kept
        } else {
            convolved
        };
        let mut next_depth = self.stream.alloc_zeros(8 * width)?;
        self.copy_device(
            &session.depthwise[index],
            4 * width,
            &mut next_depth,
            0,
            4 * width,
        )?;
        self.copy_device(
            &activated,
            0,
            &mut next_depth,
            4 * width,
            rows as usize * width,
        )?;
        ensure!(
            block.convolution.running_mean.is_none() && block.convolution.running_var.is_none(),
            "streaming convolution has batch-norm statistics"
        );
        let normalized = self.normalize_pair(
            &kept,
            rows,
            &block.convolution.conv_weight,
            &block.convolution.conv_bias,
        )?;
        let activated = self.activate(&normalized, rows, self.conformer.hidden, 1)?;
        let delta = self.linear(&activated, rows, &block.convolution.pointwise2)?;
        self.residual(&mut hidden, &delta, 1.0)?;
        feed_forward(&mut hidden, &block.ff2, 0.5)?;
        let output = self.normalize(&hidden, rows, &block.output_norm)?;
        Ok(LayerStep {
            hidden: output,
            key: next_k,
            value: next_v,
            depthwise: next_depth,
        })
    }

    fn launch_depthwise(
        &self,
        input: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        mask: &CudaSlice<u8>,
        output: &CudaSlice<f32>,
        rows: u32,
    ) -> Result<()> {
        ensure!(
            input.len() == rows as usize * self.conformer.hidden as usize
                && output.len() == input.len()
                && mask.len() >= rows as usize,
            "depthwise launch shape"
        );
        unsafe {
            self.stream
                .launch_builder(&self.ops.depthwise)
                .arg(input)
                .arg(weight)
                .arg(mask)
                .arg(output)
                .arg(&rows)
                .arg(&self.conformer.hidden)
                .arg(&self.conformer.conv_kernel)
                .arg(&self.conformer.depthwise_left)
                .launch(flat(output.len()))?;
        }
        note_launch!();
        Ok(())
    }
}

#[cfg(test)]
mod position_tests {
    use super::*;

    #[test]
    fn relative_positions_are_signed_and_interleaved() -> Result<()> {
        let table = relative_positions(3, 4)?;
        assert_eq!(table.len(), 5 * 4);
        for (row, expected) in [2.0f32, 1.0, 0.0, -1.0, -2.0].into_iter().enumerate() {
            let first = row * 4;
            assert!((table[first] - expected.sin()).abs() < 1e-6);
            assert!((table[first + 1] - expected.cos()).abs() < 1e-6);
            let frequency = 10000f32.powf(-0.5);
            assert!((table[first + 2] - (expected * frequency).sin()).abs() < 1e-6);
            assert!((table[first + 3] - (expected * frequency).cos()).abs() < 1e-6);
        }
        assert!(relative_positions(0, 4).is_err());
        Ok(())
    }

    #[test]
    fn streaming_window_index_matches_absolute_chunks() -> Result<()> {
        let mut rows = 0usize;
        let mut bytes = 0usize;
        for step in 1..=15u32 {
            let length = 4 * step;
            ensure!(
                table_index(length)? == (step - 1) as usize,
                "table index {length}"
            );
            let table = relative_positions(length as usize, 1024)?;
            rows += 2 * length as usize - 1;
            bytes += table.len() * 4;
        }
        ensure!(
            rows == 945 && bytes == 3_870_720,
            "resident table bytes {bytes} rows {rows}"
        );
        ensure!(
            table_index(0).is_err() && table_index(6).is_err() && table_index(64).is_err(),
            "out-of-set key rows accepted"
        );
        let first = stream_window(0, 4)?;
        ensure!(
            first.key_rows == 4
                && first.query_offset == 0
                && first.kv_offset == 0
                && first.cached == 0,
            "first window"
        );
        let chunk14 = stream_window(56, 4)?;
        ensure!(
            chunk14.key_rows == 60
                && chunk14.query_offset == 56
                && chunk14.kv_offset == 0
                && chunk14.cached == 56,
            "chunk 14 window"
        );
        let chunk15 = stream_window(60, 4)?;
        ensure!(
            chunk15.key_rows == 60
                && chunk15.query_offset == 60
                && chunk15.kv_offset == 4
                && chunk15.cached == 56,
            "chunk 15 window"
        );
        let later = stream_window(5000, 4)?;
        ensure!(
            later.key_rows == 60 && later.cached == 56 && later.kv_offset == 5000 - 56,
            "cumulative 5000 changed the resident length"
        );
        ensure!(
            stream_window(0, 3).is_err(),
            "query rows other than 4 accepted"
        );
        ensure!(
            kept(8, 8, 0, 0, 8) == [4usize, 4, 4, 4, 8, 8, 8, 8],
            "full 8-frame grid"
        );
        ensure!(
            kept(8, 8, 0, 0, 5) == [4usize, 4, 4, 4, 5, 5, 5, 5],
            "padded 8-frame grid"
        );
        ensure!(
            kept(8, 8, 0, 0, 0) == [0usize, 0, 0, 0, 0, 0, 0, 0],
            "masked 8-frame grid"
        );
        let offset = kept(4, 60, 60, 4, 60);
        ensure!(
            offset == [60usize, 60, 60, 60] && offset.iter().sum::<usize>() == 240,
            "offset grid"
        );
        let dropped = kept(4, 60, 0, 0, 60);
        ensure!(
            dropped == [4usize, 4, 4, 4] && dropped.iter().sum::<usize>() == 16,
            "dropped offsets"
        );
        for (query_offset, kv_offset) in [(56u32, 0u32), (60, 4)] {
            for query in 0..4u32 {
                for key in 0..60u32 {
                    let index = 4 - 1 - query + key;
                    let distance = 59 - i32::try_from(index)?;
                    let q_abs = i32::try_from(query_offset + query)?;
                    let k_abs = i32::try_from(kv_offset + key)?;
                    ensure!(
                        distance == q_abs - k_abs,
                        "relative distance q={query} k={key}"
                    );
                }
            }
        }
        Ok(())
    }

    fn kept(rows: u32, keys: u32, query_offset: u32, kv_offset: u32, valid: u32) -> Vec<usize> {
        (0..rows)
            .map(|query| {
                (0..keys)
                    .filter(|&key| {
                        key < valid
                            && absolute_key_kept(query_offset + query, kv_offset + key, 4, 14, 0)
                    })
                    .count()
            })
            .collect()
    }
}
