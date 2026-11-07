//! Native-F32 Parakeet FastConformer encoder.
use crate::{
    backend::{
        cuda::{Device, cubin},
        gpu::{CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg},
    },
    checkpoint::MappedCheckpoint,
    parakeet::{FastConformer, ParakeetSpec},
};
use anyhow::{Context, Result, ensure};
use cudarc::cublas::{CudaBlas, sys as cublas_sys};
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};
use std::{path::Path, sync::Arc};

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
    batch_weight: CudaSlice<f32>,
    batch_bias: CudaSlice<f32>,
    running_mean: CudaSlice<f32>,
    running_var: CudaSlice<f32>,
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
    lstm: CudaFunction,
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

struct LstmLayer {
    input: Linear,
    recurrent: Linear,
    bias_ih: CudaSlice<f32>,
    bias_hh: CudaSlice<f32>,
    hidden: CudaSlice<f32>,
    cell: CudaSlice<f32>,
    input_gates: CudaSlice<f32>,
    recurrent_gates: CudaSlice<f32>,
}

pub struct ParakeetTdt {
    encoder: ParakeetEncoder,
    spec: ParakeetSpec,
    embedding: CudaSlice<f32>,
    lstm: Vec<LstmLayer>,
    decoder_projector: Linear,
    encoder_projector: Linear,
    joint: Linear,
    input: CudaSlice<f32>,
    cache: CudaSlice<f32>,
    joint_frame: CudaSlice<f32>,
    joint_sum: CudaSlice<f32>,
    joint_activated: CudaSlice<f32>,
    joint_logits: CudaSlice<f32>,
    initialized: bool,
    #[cfg(test)]
    lstm_calls: std::cell::Cell<usize>,
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
            lstm: speech.load_function("lstm_step_f32")?,
            activation: vision.load_function("activation_f32")?,
            bias: vision.load_function("vision_bias_f32")?,
            layer_norm: common.load_function("q38_layer_norm_f32")?,
        };
        let mut subsample_conv = Vec::new();
        for index in [0, 2, 5] {
            subsample_conv.push(SubsampleConv {
                weight: tensor(
                    &checkpoint,
                    &stream,
                    &format!("{root}.subsampling.layers.{index}.weight"),
                    &[256, 1, 3, 3],
                )?,
                bias: tensor(
                    &checkpoint,
                    &stream,
                    &format!("{root}.subsampling.layers.{index}.bias"),
                    &[256],
                )?,
            });
        }
        let mut subsample_pointwise = Vec::new();
        for index in [3, 6] {
            subsample_pointwise.push(linear_shaped(
                &checkpoint,
                &stream,
                &format!("{root}.subsampling.layers.{index}"),
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
            4096,
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
                    batch_weight: tensor(
                        &checkpoint,
                        &stream,
                        &format!("{prefix}.conv.norm.weight"),
                        &[1024],
                    )?,
                    batch_bias: tensor(
                        &checkpoint,
                        &stream,
                        &format!("{prefix}.conv.norm.bias"),
                        &[1024],
                    )?,
                    running_mean: tensor(
                        &checkpoint,
                        &stream,
                        &format!("{prefix}.conv.norm.running_mean"),
                        &[1024],
                    )?,
                    running_var: tensor(
                        &checkpoint,
                        &stream,
                        &format!("{prefix}.conv.norm.running_var"),
                        &[1024],
                    )?,
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
        let mut output = self
            .stream
            .alloc_zeros(rows as usize * weight.output as usize)?;
        self.linear_into(input, rows, weight, &mut output)?;
        Ok(output)
    }

    fn linear_into(
        &self,
        input: &CudaSlice<f32>,
        rows: u32,
        weight: &Linear,
        output: &mut CudaSlice<f32>,
    ) -> Result<()> {
        ensure!(rows > 0, "linear requires rows");
        ensure!(
            input.len() == rows as usize * weight.input as usize
                && output.len() == rows as usize * weight.output as usize,
            "linear buffer shape mismatch"
        );
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
        if let Some(bias) = &weight.bias {
            unsafe {
                self.stream
                    .launch_builder(&self.ops.bias)
                    .arg(&*output)
                    .arg(bias)
                    .arg(&*output)
                    .arg(&rows)
                    .arg(&weight.output)
                    .launch(flat(output.len()))?;
            }
        }
        Ok(())
    }

    fn activate(
        &self,
        input: &CudaSlice<f32>,
        rows: u32,
        width: u32,
        mode: i32,
    ) -> Result<CudaSlice<f32>> {
        let mut output = self.stream.alloc_zeros(if mode == 2 {
            input.len() / 2
        } else {
            input.len()
        })?;
        self.activate_into(input, rows, width, mode, &mut output)?;
        Ok(output)
    }

    fn activate_into(
        &self,
        input: &CudaSlice<f32>,
        rows: u32,
        width: u32,
        mode: i32,
        output: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let expected = rows as usize * width as usize * if mode == 2 { 2 } else { 1 };
        ensure!(
            input.len() == expected,
            "activation shape mismatch: rows={rows} width={width} mode={mode} input={} expected={expected}",
            input.len()
        );
        ensure!(
            output.len()
                == if mode == 2 {
                    input.len() / 2
                } else {
                    input.len()
                },
            "activation output shape mismatch"
        );
        unsafe {
            self.stream
                .launch_builder(&self.ops.activation)
                .arg(input)
                .arg(output)
                .arg(&rows)
                .arg(&width)
                .arg(&mode)
                .launch(flat(input.len()))?;
        }
        Ok(())
    }

    fn normalize(&self, input: &CudaSlice<f32>, rows: u32, norm: &Norm) -> Result<CudaSlice<f32>> {
        let output = self.stream.alloc_zeros(input.len())?;
        let width = self.conformer.hidden;
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
                .launch(super::ops::grid(rows as usize, 256))?;
        }
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
        for (index, conv) in self.subsample.conv.iter().enumerate() {
            let next_time = time.div_ceil(2);
            let next_frequency = frequency / 2;
            let next_valid = current_valid.div_ceil(2);
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
                    .launch(flat(output.len()))?;
            }
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
            }
            hidden = self.activate(&hidden, spatial, 256, 0)?;
        }
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

    /// Input is `[frames,128]`; output is `[ceil(frames/8),1024]` plus the output mask.
    pub fn subsample(&self, input: &[f32], valid: usize) -> Result<(Vec<f32>, Vec<u8>)> {
        let output = self.subsample_device(input, valid)?;
        Ok((self.stream.clone_dtoh(&output.hidden)?, output.mask))
    }

    fn positions(&self, rows: usize) -> Result<Vec<f32>> {
        relative_positions(rows, self.conformer.hidden as usize)
    }

    fn layer_device(
        &self,
        index: usize,
        input: &CudaSlice<f32>,
        mask: &[u8],
        positions: &CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>> {
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
                .launch(LaunchConfig {
                    grid_dim: (rows, self.conformer.heads, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: (8 + rows) * 4,
                })?;
        }
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
                .launch(flat(convolved.len()))?;
        }
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
        }
        let normalized: CudaSlice<f32> = self.stream.alloc_zeros(hidden.len())?;
        unsafe {
            self.stream
                .launch_builder(&self.ops.batch_norm)
                .arg(&convolved)
                .arg(&block.convolution.batch_weight)
                .arg(&block.convolution.batch_bias)
                .arg(&block.convolution.running_mean)
                .arg(&block.convolution.running_var)
                .arg(&normalized)
                .arg(&rows)
                .arg(&self.conformer.hidden)
                .arg(&1e-5f32)
                .launch(flat(normalized.len()))?;
        }
        let activated = self.activate(&normalized, rows, self.conformer.hidden, 1)?;
        let delta = self.linear(&activated, rows, &block.convolution.pointwise2)?;
        self.residual(&mut hidden, &delta, 1.0)?;
        feed_forward(&mut hidden, &block.ff2, 0.5)?;
        let output = self.normalize(&hidden, rows, &block.output_norm)?;
        Ok(output)
    }

    /// Input and output are `[frames,1024]` in native F32.
    pub fn layer(&self, index: usize, input: &[f32], mask: &[u8]) -> Result<Vec<f32>> {
        let positions = self.stream.clone_htod(&self.positions(mask.len())?)?;
        let input = self.stream.clone_htod(input)?;
        let output = self.layer_device(index, &input, mask, &positions)?;
        Ok(self.stream.clone_dtoh(&output)?)
    }

    /// Input is `[frames,128]`; output is `[ceil(frames/8),1024]`, its mask and valid row count.
    pub fn encode(&self, input: &[f32], valid: usize) -> Result<(Vec<f32>, Vec<u8>, usize)> {
        #[cfg(test)]
        self.encode_calls.fetch_add(1, Ordering::Relaxed);
        let subsampled = self.subsample_device(input, valid)?;
        let positions = self
            .stream
            .clone_htod(&self.positions(subsampled.mask.len())?)?;
        let mut hidden = subsampled.hidden;
        for index in 0..self.blocks.len() {
            hidden = self.layer_device(index, &hidden, &subsampled.mask, &positions)?;
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

impl ParakeetTdt {
    pub fn load(directory: &Path, device: &Device) -> Result<Self> {
        let spec = ParakeetSpec::read(directory)?;
        let encoder = ParakeetEncoder::load(directory, device)?;
        let checkpoint = MappedCheckpoint::open(directory)?;
        let stream = Arc::clone(&encoder.stream);
        let hidden = spec.decoder_hidden as usize;
        let vocab = spec.vocab as usize;
        let gate = 4 * hidden;
        let joint_width = vocab + spec.durations.len();
        let embedding = tensor(
            &checkpoint,
            &stream,
            "decoder.embedding.weight",
            &[vocab, hidden],
        )?;
        let mut layers = Vec::with_capacity(spec.decoder_layers as usize);
        for index in 0..spec.decoder_layers {
            layers.push(LstmLayer {
                input: Linear {
                    weight: tensor(
                        &checkpoint,
                        &stream,
                        &format!("decoder.lstm.weight_ih_l{index}"),
                        &[gate, hidden],
                    )?,
                    bias: None,
                    output: 4 * spec.decoder_hidden,
                    input: spec.decoder_hidden,
                },
                recurrent: Linear {
                    weight: tensor(
                        &checkpoint,
                        &stream,
                        &format!("decoder.lstm.weight_hh_l{index}"),
                        &[gate, hidden],
                    )?,
                    bias: None,
                    output: 4 * spec.decoder_hidden,
                    input: spec.decoder_hidden,
                },
                bias_ih: tensor(
                    &checkpoint,
                    &stream,
                    &format!("decoder.lstm.bias_ih_l{index}"),
                    &[gate],
                )?,
                bias_hh: tensor(
                    &checkpoint,
                    &stream,
                    &format!("decoder.lstm.bias_hh_l{index}"),
                    &[gate],
                )?,
                hidden: stream.alloc_zeros(hidden)?,
                cell: stream.alloc_zeros(hidden)?,
                input_gates: stream.alloc_zeros(gate)?,
                recurrent_gates: stream.alloc_zeros(gate)?,
            });
        }
        let decoder_projector = linear(
            &checkpoint,
            &stream,
            "decoder.decoder_projector",
            spec.decoder_hidden,
            spec.decoder_hidden,
            true,
        )?;
        let encoder_projector = linear(
            &checkpoint,
            &stream,
            "encoder_projector",
            spec.decoder_hidden,
            spec.hidden,
            true,
        )?;
        let joint = linear(
            &checkpoint,
            &stream,
            "joint.head",
            u32::try_from(joint_width)?,
            spec.decoder_hidden,
            true,
        )?;
        let input = stream.alloc_zeros(hidden)?;
        let cache = stream.alloc_zeros(hidden)?;
        let joint_frame = stream.alloc_zeros(hidden)?;
        let joint_sum = stream.alloc_zeros(hidden)?;
        let joint_activated = stream.alloc_zeros(hidden)?;
        let joint_logits = stream.alloc_zeros(joint_width)?;
        stream.synchronize()?;
        Ok(Self {
            encoder,
            spec,
            embedding,
            lstm: layers,
            decoder_projector,
            encoder_projector,
            joint,
            input,
            cache,
            joint_frame,
            joint_sum,
            joint_activated,
            joint_logits,
            initialized: false,
            #[cfg(test)]
            lstm_calls: std::cell::Cell::new(0),
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        for layer in &mut self.lstm {
            self.encoder.stream.memset_zeros(&mut layer.hidden)?;
            self.encoder.stream.memset_zeros(&mut layer.cell)?;
            self.encoder.stream.memset_zeros(&mut layer.input_gates)?;
            self.encoder
                .stream
                .memset_zeros(&mut layer.recurrent_gates)?;
        }
        self.encoder.stream.memset_zeros(&mut self.input)?;
        self.encoder.stream.memset_zeros(&mut self.cache)?;
        self.encoder.stream.memset_zeros(&mut self.joint_frame)?;
        self.encoder.stream.memset_zeros(&mut self.joint_sum)?;
        self.encoder
            .stream
            .memset_zeros(&mut self.joint_activated)?;
        self.encoder.stream.memset_zeros(&mut self.joint_logits)?;
        self.initialized = false;
        #[cfg(test)]
        self.lstm_calls.set(0);
        Ok(())
    }

    #[cfg(test)]
    pub fn set_prediction_state(&mut self, hidden: &[f32], cell: &[f32]) -> Result<()> {
        let width = self.spec.decoder_hidden as usize;
        let layers = self.spec.decoder_layers as usize;
        ensure!(
            hidden.len() == width * layers && cell.len() == width * layers,
            "prediction state extent differs from decoder geometry"
        );
        for (index, layer) in self.lstm.iter_mut().enumerate() {
            self.encoder.stream.memcpy_htod(
                &hidden[index * width..(index + 1) * width],
                &mut layer.hidden,
            )?;
            self.encoder
                .stream
                .memcpy_htod(&cell[index * width..(index + 1) * width], &mut layer.cell)?;
        }
        self.initialized = true;
        Ok(())
    }

    pub fn predict(&mut self, token: u32) -> Result<Vec<f32>> {
        ensure!(
            token < self.spec.vocab,
            "prediction token outside vocabulary"
        );
        if self.initialized && token == self.spec.blank {
            return Ok(self.encoder.stream.clone_dtoh(&self.cache)?);
        }
        let width = self.spec.decoder_hidden as usize;
        let start = token as usize * width;
        self.encoder
            .stream
            .memcpy_dtod(&self.embedding.slice(start..start + width), &mut self.input)?;
        for index in 0..self.lstm.len() {
            let (previous, current) = self.lstm.split_at_mut(index);
            let layer = &mut current[0];
            let input = if index == 0 {
                &self.input
            } else {
                &previous[index - 1].hidden
            };
            self.encoder
                .linear_into(input, 1, &layer.input, &mut layer.input_gates)?;
            self.encoder.linear_into(
                &layer.hidden,
                1,
                &layer.recurrent,
                &mut layer.recurrent_gates,
            )?;
            unsafe {
                self.encoder
                    .stream
                    .launch_builder(&self.encoder.ops.lstm)
                    .arg(&layer.input_gates)
                    .arg(&layer.recurrent_gates)
                    .arg(&layer.bias_ih)
                    .arg(&layer.bias_hh)
                    .arg(&layer.cell)
                    .arg(&layer.hidden)
                    .arg(&self.spec.decoder_hidden)
                    .launch(flat(width))?;
            }
            #[cfg(test)]
            self.lstm_calls.set(self.lstm_calls.get() + 1);
        }
        let last = self.lstm.last().context("prediction LSTM stack empty")?;
        self.encoder
            .linear_into(&last.hidden, 1, &self.decoder_projector, &mut self.cache)?;
        self.initialized = true;
        Ok(self.encoder.stream.clone_dtoh(&self.cache)?)
    }

    pub fn prediction_state(&self) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let mut hidden = Vec::with_capacity(
            self.spec.decoder_layers as usize * self.spec.decoder_hidden as usize,
        );
        let mut cell = Vec::with_capacity(
            self.spec.decoder_layers as usize * self.spec.decoder_hidden as usize,
        );
        for layer in &self.lstm {
            hidden.extend(self.encoder.stream.clone_dtoh(&layer.hidden)?);
            cell.extend(self.encoder.stream.clone_dtoh(&layer.cell)?);
        }
        Ok((self.encoder.stream.clone_dtoh(&self.cache)?, hidden, cell))
    }

    pub fn project_encoder(&self, input: &[f32]) -> Result<Vec<f32>> {
        let width = self.spec.hidden as usize;
        ensure!(
            !input.is_empty()
                && input.len().is_multiple_of(width)
                && input.iter().all(|value| value.is_finite()),
            "invalid encoder projection input"
        );
        let input = self.encoder.stream.clone_htod(input)?;
        let output = self.encoder.linear(
            &input,
            u32::try_from(input.len() / width)?,
            &self.encoder_projector,
        )?;
        Ok(self.encoder.stream.clone_dtoh(&output)?)
    }

    pub fn joint(&mut self, encoder_frame: &[f32]) -> Result<Vec<f32>> {
        let width = self.spec.decoder_hidden as usize;
        ensure!(
            encoder_frame.len() == width && encoder_frame.iter().all(|value| value.is_finite()),
            "joint encoder frame extent differs from decoder hidden size"
        );
        self.encoder
            .stream
            .memcpy_htod(encoder_frame, &mut self.joint_frame)?;
        self.encoder
            .stream
            .memcpy_dtod(&self.cache, &mut self.joint_sum)?;
        self.encoder
            .residual(&mut self.joint_sum, &self.joint_frame, 1.0)?;
        self.encoder.activate_into(
            &self.joint_sum,
            1,
            self.spec.decoder_hidden,
            0,
            &mut self.joint_activated,
        )?;
        self.encoder.linear_into(
            &self.joint_activated,
            1,
            &self.joint,
            &mut self.joint_logits,
        )?;
        Ok(self.encoder.stream.clone_dtoh(&self.joint_logits)?)
    }

    #[cfg(test)]
    pub fn lstm_call_count(&self) -> usize {
        self.lstm_calls.get()
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
