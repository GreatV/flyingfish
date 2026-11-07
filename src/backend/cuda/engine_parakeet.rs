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
        unsafe {
            self.stream
                .launch_builder(&self.ops.relative_attention)
                .arg(&q)
                .arg(&k)
                .arg(&v)
                .arg(&relative)
                .arg(&block.attention.bias_u)
                .arg(&block.attention.bias_v)
                .arg(&mask)
                .arg(&core)
                .arg(&rows)
                .arg(&self.conformer.heads)
                .arg(&(1.0f32 / 128f32.sqrt()))
                .arg(&self.conformer.chunk_size)
                .arg(&self.conformer.left_chunks)
                .arg(&right_chunks)
                .launch(LaunchConfig {
                    grid_dim: (rows, self.conformer.heads, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: (8 + rows) * 4,
                })?;
        }
        note_launch!();
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
}
