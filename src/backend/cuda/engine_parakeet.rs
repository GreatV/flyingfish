//! Native-F32 Parakeet FastConformer encoder.
use crate::{
    backend::{
        cuda::{Device, cubin},
        gpu::{
            CudaFunction, CudaGraph, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
        },
    },
    checkpoint::MappedCheckpoint,
    parakeet::{FastConformer, ParakeetSpec, conv_out, position_rows_fit},
};
use anyhow::{Context, Result, ensure};
use cudarc::cublas::{CudaBlas, sys as cublas_sys};

/// Pinned f32 GEMM algorithm for the lane stack; AlgoChecked at 14/28/42/56 rows.
const LANE_GEMM_ALGO: cublas_sys::cublasGemmAlgo_t =
    cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_ALGO0;
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

pub(crate) struct Linear {
    pub(crate) weight: CudaSlice<f32>,
    pub(crate) bias: Option<CudaSlice<f32>>,
    pub(crate) output: u32,
    pub(crate) input: u32,
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

pub(crate) struct Ops {
    pub(crate) conv2d: CudaFunction,
    pub(crate) mask_time: CudaFunction,
    pub(crate) pack: CudaFunction,
    depthwise: CudaFunction,
    batch_norm: CudaFunction,
    pub(crate) residual: CudaFunction,
    relative_attention: CudaFunction,
    pub(crate) lstm: CudaFunction,
    pub(crate) activation: CudaFunction,
    pub(crate) bias: CudaFunction,
    layer_norm: CudaFunction,
    pub(crate) argmax: CudaFunction,
    row_scale: CudaFunction,
}

pub struct ParakeetEncoder {
    conformer: FastConformer,
    pub(crate) stream: Arc<CudaStream>,
    subsample: Subsample,
    subsample_linear: Linear,
    blas: CudaBlas,
    blocks: Vec<Block>,
    pub(crate) ops: Ops,
    #[cfg(test)]
    encode_calls: AtomicU64,
}

pub(crate) struct LstmLayer {
    pub(crate) input: Linear,
    pub(crate) recurrent: Linear,
    pub(crate) bias_ih: CudaSlice<f32>,
    pub(crate) bias_hh: CudaSlice<f32>,
    pub(crate) hidden: CudaSlice<f32>,
    pub(crate) cell: CudaSlice<f32>,
    pub(crate) input_gates: CudaSlice<f32>,
    pub(crate) recurrent_gates: CudaSlice<f32>,
}

pub struct TdtTrajectory {
    pub tokens: Vec<u32>,
    pub durations: Vec<u32>,
    pub frames: Vec<usize>,
    pub capacity: usize,
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
    selection: CudaSlice<u32>,
    step_index: CudaSlice<i32>,
    gather: CudaFunction,
    token_graph: Option<CudaGraph>,
    blank_graph: Option<CudaGraph>,
    projected: Option<CudaSlice<f32>>,
    initialized: bool,
    #[cfg(test)]
    eager_steps: std::cell::Cell<bool>,
    #[cfg(test)]
    graph_captures: std::cell::Cell<u64>,
    #[cfg(test)]
    graph_launches: std::cell::Cell<u64>,
    #[cfg(test)]
    lstm_calls: std::cell::Cell<usize>,
}

struct SubsampleOutput {
    hidden: CudaSlice<f32>,
    mask: Vec<u8>,
    valid: usize,
}

pub(crate) fn tensor(
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

pub(crate) fn linear(
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

fn norm(
    checkpoint: &MappedCheckpoint,
    stream: &Arc<CudaStream>,
    prefix: &str,
    hidden: u32,
) -> Result<Norm> {
    let size = [hidden as usize];
    Ok(Norm {
        weight: tensor(checkpoint, stream, &format!("{prefix}.weight"), &size)?,
        bias: tensor(checkpoint, stream, &format!("{prefix}.bias"), &size)?,
    })
}

fn flat(count: usize) -> LaunchConfig {
    super::ops::flat(count)
}

fn duration_step(blank: bool, duration: u32) -> u32 {
    if blank && duration == 0 { 1 } else { duration }
}

fn decode_capacity(max_symbols: usize, physical_rows: usize) -> Result<usize> {
    ensure!(physical_rows > 0, "decode requires encoder rows");
    max_symbols
        .checked_mul(physical_rows)
        .context("decode capacity overflow")
}

fn decode_admit(generated: usize, capacity: usize) -> Result<()> {
    ensure!(
        generated < capacity,
        "decode capacity exceeded: generated={generated} capacity={capacity}"
    );
    Ok(())
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
    query_offset: i64,
    kv_offset: i64,
    right_chunks: u32,
    valid_keys: u32,
    /// Speaker lanes: q/k/v/relative/output carry `lanes` planes; the mask is shared.
    lanes: u32,
    /// Sliding left window in rows (0 = chunk-aligned left_chunks semantics).
    left_slide: u32,
}

#[cfg(test)]
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
    query_offset: i64,
    kv_offset: i64,
    cached: u32,
    #[cfg(test)]
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

fn stream_window(
    cumulative_before: u32,
    query_rows: u32,
    #[cfg(test)] omit: StreamOmit,
) -> Result<StreamWindow> {
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
        query_offset: i64::from(cumulative_before),
        kv_offset: i64::from(kv_offset),
        cached,
        #[cfg(test)]
        omit,
    })
}

fn attend_span(window: &StreamWindow, query_rows: u32) -> (u32, u32) {
    #[cfg(test)]
    if window.omit == StreamOmit::Keys {
        return (query_rows, u32::try_from(window.query_offset).unwrap_or(0));
    }
    #[cfg(not(test))]
    let _ = query_rows;
    (
        window.key_rows,
        u32::try_from(window.kv_offset).unwrap_or(0),
    )
}

fn copy_cached_keys(window: &StreamWindow) -> bool {
    #[cfg(test)]
    {
        window.omit != StreamOmit::Keys
    }
    #[cfg(not(test))]
    {
        let _ = window;
        true
    }
}

fn cached_key_dest(window: &StreamWindow, cached_elems: usize) -> usize {
    #[cfg(test)]
    if window.omit == StreamOmit::Keys {
        return 0;
    }
    #[cfg(not(test))]
    let _ = window;
    cached_elems
}

fn conv_cache_omitted(window: &StreamWindow) -> bool {
    #[cfg(test)]
    {
        window.omit == StreamOmit::Conv
    }
    #[cfg(not(test))]
    {
        let _ = window;
        false
    }
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
    pub(crate) fn encode_call_count() -> u64 {
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
        Self::load_conformer_checkpoint(checkpoint, device, conformer)
    }

    /// Loads the EOU encoder from the stored-ZIP member of a `.nemo` tar archive.
    pub fn load_eou(nemo: &Path, device: &Device) -> Result<Self> {
        let spec = crate::eou::EouSpec::read(nemo)?;
        let checkpoint = MappedCheckpoint::open_nemo(nemo)?;
        Self::load_conformer_checkpoint(checkpoint, device, &spec.conformer())
    }

    /// Loads the multitalker encoder from the stored-ZIP member of a `.nemo` tar archive.
    pub fn load_multitalker(nemo: &Path, device: &Device) -> Result<Self> {
        let spec = crate::multitalker::MultitalkerSpec::read(nemo)?;
        let checkpoint = MappedCheckpoint::open_nemo(nemo)?;
        Self::load_conformer_checkpoint(checkpoint, device, &spec.conformer())
    }

    fn load_conformer_checkpoint(
        checkpoint: MappedCheckpoint,
        device: &Device,
        conformer: &FastConformer,
    ) -> Result<Self> {
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
        let sample = cubin::module(&device.ctx, "sample")?;
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
            argmax: sample.load_function("argmax_f32")?,
            row_scale: speech.load_function("speech_row_scale_f32")?,
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
            &format!("{root}.{}", conformer.linear_name),
            conformer.hidden,
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
                    norm: norm(
                        &checkpoint,
                        &stream,
                        &format!("{prefix}.norm_{branch}"),
                        conformer.hidden,
                    )?,
                    up: linear(
                        &checkpoint,
                        &stream,
                        &format!("{prefix}.{branch}.linear1"),
                        conformer.hidden * 4,
                        conformer.hidden,
                        attention_bias,
                    )?,
                    down: linear(
                        &checkpoint,
                        &stream,
                        &format!("{prefix}.{branch}.linear2"),
                        conformer.hidden,
                        conformer.hidden * 4,
                        attention_bias,
                    )?,
                })
            };
            let attention = |name: &str, bias: bool| -> Result<Linear> {
                linear(
                    &checkpoint,
                    &stream,
                    &format!("{prefix}.self_attn.{name}"),
                    conformer.hidden,
                    conformer.hidden,
                    bias,
                )
            };
            blocks.push(Block {
                ff1: feed_forward("feed_forward1")?,
                ff2: feed_forward("feed_forward2")?,
                attention: Attention {
                    q: attention(conformer.attention_names[0], attention_bias)?,
                    k: attention(conformer.attention_names[1], attention_bias)?,
                    v: attention(conformer.attention_names[2], attention_bias)?,
                    o: attention(conformer.attention_names[3], attention_bias)?,
                    relative: attention(conformer.attention_names[4], false)?,
                    bias_u: tensor(
                        &checkpoint,
                        &stream,
                        &format!("{}.self_attn.{}", prefix, conformer.attention_names[5]),
                        &[
                            conformer.heads as usize,
                            conformer.hidden as usize / conformer.heads as usize,
                        ],
                    )?,
                    bias_v: tensor(
                        &checkpoint,
                        &stream,
                        &format!("{}.self_attn.{}", prefix, conformer.attention_names[6]),
                        &[
                            conformer.heads as usize,
                            conformer.hidden as usize / conformer.heads as usize,
                        ],
                    )?,
                },
                convolution: Convolution {
                    norm: norm(
                        &checkpoint,
                        &stream,
                        &format!("{prefix}.norm_conv"),
                        conformer.hidden,
                    )?,
                    pointwise1: linear_shaped(
                        &checkpoint,
                        &stream,
                        &format!("{prefix}.conv.pointwise_conv1"),
                        conformer.hidden * 2,
                        conformer.hidden,
                        convolution_bias,
                        &[conformer.hidden as usize * 2, conformer.hidden as usize, 1],
                    )?,
                    depthwise: tensor(
                        &checkpoint,
                        &stream,
                        &format!("{prefix}.conv.depthwise_conv.weight"),
                        &[conformer.hidden as usize, 1, conformer.conv_kernel as usize],
                    )?,
                    depthwise_bias: if convolution_bias {
                        Some(tensor(
                            &checkpoint,
                            &stream,
                            &format!("{prefix}.conv.depthwise_conv.bias"),
                            &[conformer.hidden as usize],
                        )?)
                    } else {
                        None
                    },
                    conv_weight: tensor(
                        &checkpoint,
                        &stream,
                        &format!("{prefix}.conv.{}.weight", conformer.conv_norm_name),
                        &[conformer.hidden as usize],
                    )?,
                    conv_bias: tensor(
                        &checkpoint,
                        &stream,
                        &format!("{prefix}.conv.{}.bias", conformer.conv_norm_name),
                        &[conformer.hidden as usize],
                    )?,
                    running_mean: if conformer.conv_batch_norm {
                        Some(tensor(
                            &checkpoint,
                            &stream,
                            &format!("{prefix}.conv.{}.running_mean", conformer.conv_norm_name),
                            &[conformer.hidden as usize],
                        )?)
                    } else {
                        None
                    },
                    running_var: if conformer.conv_batch_norm {
                        Some(tensor(
                            &checkpoint,
                            &stream,
                            &format!("{prefix}.conv.{}.running_var", conformer.conv_norm_name),
                            &[conformer.hidden as usize],
                        )?)
                    } else {
                        None
                    },
                    pointwise2: linear_shaped(
                        &checkpoint,
                        &stream,
                        &format!("{prefix}.conv.pointwise_conv2"),
                        conformer.hidden,
                        conformer.hidden,
                        convolution_bias,
                        &[conformer.hidden as usize, conformer.hidden as usize, 1],
                    )?,
                },
                attention_norm: norm(
                    &checkpoint,
                    &stream,
                    &format!("{prefix}.norm_self_att"),
                    conformer.hidden,
                )?,
                output_norm: norm(
                    &checkpoint,
                    &stream,
                    &format!("{prefix}.norm_out"),
                    conformer.hidden,
                )?,
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

    pub(crate) fn linear(
        &self,
        input: &CudaSlice<f32>,
        rows: u32,
        weight: &Linear,
    ) -> Result<CudaSlice<f32>> {
        let mut output = self
            .stream
            .alloc_zeros(rows as usize * weight.output as usize)?;
        self.linear_into(input, rows, weight, &mut output)?;
        Ok(output)
    }

    pub(crate) fn lane_linear(
        &self,
        input: &CudaSlice<f32>,
        rows: u32,
        canonical: u32,
        weight: &Linear,
    ) -> Result<CudaSlice<f32>> {
        let mut output = self
            .stream
            .alloc_zeros(rows as usize * weight.output as usize)?;
        self.lane_linear_into(input, rows, canonical, weight, &mut output)?;
        Ok(output)
    }

    pub(crate) fn linear_into(
        &self,
        input: &CudaSlice<f32>,
        rows: u32,
        weight: &Linear,
        output: &mut CudaSlice<f32>,
    ) -> Result<()> {
        self.linear_into_algo(
            input,
            rows,
            weight,
            output,
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    }

    /// Row-invariant GEMM for the multitalker lane stack: every lane count
    /// runs the identical GEMM shape — rows below the 4-lane canonical count
    /// are zero-padded to it — so a lane's rows are bitwise-identical whether
    /// it runs alone or stacked. The pad is performance-neutral: these GEMMs
    /// are weight-traffic-bound at every lane count.
    pub(crate) fn lane_linear_into(
        &self,
        input: &CudaSlice<f32>,
        rows: u32,
        canonical: u32,
        weight: &Linear,
        output: &mut CudaSlice<f32>,
    ) -> Result<()> {
        ensure!(
            rows > 0 && canonical >= rows && canonical.is_multiple_of(4),
            "lane GEMM rows {rows} vs canonical {canonical}"
        );
        if canonical == rows {
            self.linear_into_algo(input, rows, weight, output, LANE_GEMM_ALGO)?;
            return Ok(());
        }
        let mut padded_in = self
            .stream
            .alloc_zeros(canonical as usize * weight.input as usize)?;
        self.copy_device(input, 0, &mut padded_in, 0, input.len())?;
        let mut padded_out = self
            .stream
            .alloc_zeros(canonical as usize * weight.output as usize)?;
        self.linear_into_algo(
            &padded_in,
            canonical,
            weight,
            &mut padded_out,
            LANE_GEMM_ALGO,
        )?;
        self.copy_device(
            &padded_out,
            0,
            output,
            0,
            rows as usize * weight.output as usize,
        )?;
        Ok(())
    }

    fn linear_into_algo(
        &self,
        input: &CudaSlice<f32>,
        rows: u32,
        weight: &Linear,
        output: &mut CudaSlice<f32>,
        algo: cublas_sys::cublasGemmAlgo_t,
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
                algo,
            )
            .result()?;
        }
        note_launch!();
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
            note_launch!();
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

    pub(crate) fn activate_into(
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
        note_launch!();
        Ok(())
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

    pub(crate) fn residual(
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

    /// Host-side row count of the pre-encode for a `[frames, 128]` window: the
    /// three conv stages' time shrink, mirroring `subsample_device`.
    pub(crate) fn multitalker_subsample_rows(&self, frames: usize) -> Result<usize> {
        let left = self.conformer.pad_left;
        let right = self.conformer.pad_right;
        conv_out(
            conv_out(conv_out(frames, left, right)?, left, right)?,
            left,
            right,
        )
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
                    .arg(&1u32)
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
                let activated = self.activate(&expanded, rows, self.conformer.hidden * 4, 1)?;
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
            valid_keys: rows,
            lanes: 1,
            left_slide: 0,
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
                .arg(&1u32)
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
        let lanes = launch.lanes as usize;
        self.check_right(launch.right_chunks)?;
        ensure!(
            lanes >= 1
                && key_rows >= rows
                && launch.k.len() == lanes * key_rows * hidden
                && launch.v.len() == lanes * key_rows * hidden
                && launch.q.len() == lanes * rows * hidden
                && launch.output.len() == lanes * rows * hidden
                && launch.relative.len() >= lanes * (rows + key_rows - 1) * hidden
                && launch.mask.len() >= key_rows
                && launch.valid_keys <= key_rows as u32
                && (self.conformer.chunk_size != 0
                    || (launch.query_offset == 0
                        && launch.kv_offset == 0
                        && launch.key_rows == launch.rows)),
            "attention launch shape rows={rows} key_rows={key_rows} lanes={lanes}"
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
                .arg(&(self.conformer.hidden / self.conformer.heads))
                .arg(&{
                    let head_dim = (self.conformer.hidden / self.conformer.heads) as f32;
                    1.0f32 / head_dim.sqrt()
                })
                .arg(&self.conformer.chunk_size)
                .arg(&self.conformer.left_chunks)
                .arg(&launch.right_chunks)
                .arg(&launch.key_rows)
                .arg(&launch.query_offset)
                .arg(&launch.kv_offset)
                .arg(&launch.valid_keys)
                .arg(&launch.lanes)
                .arg(&launch.left_slide)
                .launch(LaunchConfig {
                    grid_dim: (launch.rows, self.conformer.heads, launch.lanes),
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
        #[cfg(test)]
        {
            self.encode_chunk_omitting(session, mel, StreamOmit::None)
        }
        #[cfg(not(test))]
        {
            let (output, caches) = self.run_chunk(session, mel)?;
            self.commit_chunk(session, caches)?;
            Ok(output)
        }
    }

    #[cfg(test)]
    pub fn encode_chunk_omitting(
        &self,
        session: &mut NemotronStream,
        mel: &[f32],
        omit: StreamOmit,
    ) -> Result<StreamChunk> {
        ENCODE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        ATTENTION_QUERY_ROWS.store(0, std::sync::atomic::Ordering::Relaxed);
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
        #[cfg(test)] omit: StreamOmit,
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
        #[cfg(test)]
        let window = stream_window(session.cumulative, 4, omit)?;
        #[cfg(not(test))]
        let window = stream_window(session.cumulative, 4)?;
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
        #[cfg(test)]
        if omit == StreamOmit::Keys {
            ensure!(session.key_frames > 0, "omit keys requires stored keys");
        }
        #[cfg(test)]
        if omit == StreamOmit::Conv {
            ensure!(
                session.conv.iter().all(|frame| frame.is_some()),
                "omit conv requires a stored frame"
            );
        }
        let (mut hidden, conv) = self.subsample_stream(mel, session, &window)?;
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
        window: &StreamWindow,
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
            } else if conv_cache_omitted(window) {
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
                    .arg(&1u32)
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
        let (attend_rows, attend_offset) = attend_span(window, rows);
        let table = session
            .tables
            .get(table_index(attend_rows)?)
            .context("streaming position table")?;
        let relative_rows = 2 * attend_rows - 1;
        let relative = self.linear(table, relative_rows, &block.attention.relative)?;
        let cached = window.cached as usize;
        ensure!(
            session.keys[index].len() == cached * width
                && session.values[index].len() == cached * width,
            "layer {index} key cache length"
        );
        let mut cat_k = self.stream.alloc_zeros(attend_rows as usize * width)?;
        let mut cat_v = self.stream.alloc_zeros(attend_rows as usize * width)?;
        let dest = cached_key_dest(window, cached * width);
        if copy_cached_keys(window) {
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
            kv_offset: i64::from(attend_offset),
            right_chunks: 0,
            valid_keys: attend_rows,
            lanes: 1,
            left_slide: 0,
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
        let launch_rows = if session.chunks == 0 || conv_cache_omitted(window) {
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
                1,
            )?;
        } else {
            self.launch_depthwise(
                &activated,
                &block.convolution.depthwise,
                &session.mask,
                &convolved,
                launch_rows,
                1,
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
        lanes: u32,
    ) -> Result<()> {
        ensure!(
            input.len() == lanes as usize * rows as usize * self.conformer.hidden as usize
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
                .arg(&lanes)
                .launch(flat(output.len()))?;
        }
        note_launch!();
        Ok(())
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
        let selection = stream.alloc_zeros(2)?;
        let step_index = stream.alloc_zeros::<i32>(2)?;
        let gather =
            cubin::module(&device.ctx, "vision_ops")?.load_function("vision_row_gather_f32")?;
        stream.synchronize()?;
        device.finish_upload()?;
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
            selection,
            step_index,
            gather,
            token_graph: None,
            blank_graph: None,
            projected: None,
            initialized: false,
            #[cfg(test)]
            eager_steps: std::cell::Cell::new(false),
            #[cfg(test)]
            graph_captures: std::cell::Cell::new(0),
            #[cfg(test)]
            graph_launches: std::cell::Cell::new(0),
            #[cfg(test)]
            lstm_calls: std::cell::Cell::new(0),
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        self.token_graph = None;
        self.blank_graph = None;
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
        self.encoder.stream.memset_zeros(&mut self.selection)?;
        self.encoder.stream.memset_zeros(&mut self.step_index)?;
        self.projected = None;
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

    #[cfg(test)]
    pub fn set_joint_state(
        &mut self,
        hidden: &[f32],
        cell: &[f32],
        prediction_output: &[f32],
    ) -> Result<()> {
        self.set_prediction_state(hidden, cell)?;
        let width = self.spec.decoder_hidden as usize;
        ensure!(
            prediction_output.len() == width
                && prediction_output.iter().all(|value| value.is_finite()),
            "prediction output extent differs from decoder hidden size"
        );
        self.encoder
            .stream
            .memcpy_htod(prediction_output, &mut self.cache)?;
        Ok(())
    }

    fn predict_cached(&mut self, token: u32) -> Result<()> {
        ensure!(
            token < self.spec.vocab,
            "prediction token outside vocabulary"
        );
        if self.initialized && token == self.spec.blank {
            return Ok(());
        }
        let width = self.spec.decoder_hidden as usize;
        let start = token as usize * width;
        self.encoder
            .stream
            .memcpy_dtod(&self.embedding.slice(start..start + width), &mut self.input)?;
        self.predict_from_input()?;
        self.initialized = true;
        #[cfg(test)]
        self.lstm_calls.set(self.lstm_calls.get() + self.lstm.len());
        Ok(())
    }

    fn predict_from_input(&mut self) -> Result<()> {
        let width = self.spec.decoder_hidden as usize;
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
        }
        let last = self.lstm.last().context("prediction LSTM stack empty")?;
        self.encoder
            .linear_into(&last.hidden, 1, &self.decoder_projector, &mut self.cache)?;
        Ok(())
    }

    fn gather_row(
        stream: &Arc<CudaStream>,
        gather: &CudaFunction,
        table: &CudaSlice<f32>,
        index: &CudaSlice<i32>,
        which: usize,
        output: &mut CudaSlice<f32>,
        in_rows: u32,
    ) -> Result<()> {
        let width = u32::try_from(output.len()).context("tdt gather width")?;
        ensure!(
            which < index.len()
                && table.len() == in_rows as usize * output.len()
                && output.len() == width as usize,
            "tdt gather shape"
        );
        let rows = 1u32;
        let view = index.slice(which..which + 1);
        unsafe {
            stream
                .launch_builder(gather)
                .arg(table)
                .arg(&view)
                .arg(&*output)
                .arg(&rows)
                .arg(&in_rows)
                .arg(&width)
                .launch(flat(output.len()))?;
        }
        Ok(())
    }

    fn gather_embedding(&mut self) -> Result<()> {
        Self::gather_row(
            &self.encoder.stream,
            &self.gather,
            &self.embedding,
            &self.step_index,
            0,
            &mut self.input,
            self.spec.vocab,
        )
    }

    fn write_indices(&mut self, token: u32, frame: usize) -> Result<()> {
        let token = i32::try_from(token).context("tdt token index")?;
        let frame = i32::try_from(frame).context("tdt frame index")?;
        let indices = [token, frame];
        self.encoder
            .stream
            .memcpy_htod(indices.as_slice(), &mut self.step_index)?;
        Ok(())
    }

    pub fn predict(&mut self, token: u32) -> Result<Vec<f32>> {
        self.predict_cached(token)?;
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

    fn joint_device(&mut self, physical_rows: usize) -> Result<()> {
        let width = self.spec.decoder_hidden;
        let in_rows = u32::try_from(physical_rows).context("tdt projected rows")?;
        {
            let projected = self
                .projected
                .as_ref()
                .context("decode projector missing")?;
            ensure!(
                projected.len() == physical_rows * width as usize,
                "decode projected extent differs from decoder hidden size"
            );
            Self::gather_row(
                &self.encoder.stream,
                &self.gather,
                projected,
                &self.step_index,
                1,
                &mut self.joint_frame,
                in_rows,
            )?;
        }
        self.encoder
            .stream
            .memcpy_dtod(&self.cache, &mut self.joint_sum)?;
        self.encoder
            .residual(&mut self.joint_sum, &self.joint_frame, 1.0)?;
        self.encoder
            .activate_into(&self.joint_sum, 1, width, 0, &mut self.joint_activated)?;
        self.encoder.linear_into(
            &self.joint_activated,
            1,
            &self.joint,
            &mut self.joint_logits,
        )?;
        self.launch_argmax()
    }

    fn launch_argmax(&mut self) -> Result<()> {
        let vocab = self.spec.vocab as usize;
        let joint_width = vocab + self.spec.durations.len();
        let token_logits = self.joint_logits.slice(..vocab);
        let duration_logits = self.joint_logits.slice(vocab..joint_width);
        unsafe {
            self.encoder
                .stream
                .launch_builder(&self.encoder.ops.argmax)
                .arg(&token_logits)
                .arg(&mut self.selection.slice_mut(0..1))
                .arg(&(token_logits.len() as i32))
                .launch(super::ops::grid(1, 1024))?;
            self.encoder
                .stream
                .launch_builder(&self.encoder.ops.argmax)
                .arg(&duration_logits)
                .arg(&mut self.selection.slice_mut(1..2))
                .arg(&(duration_logits.len() as i32))
                .launch(super::ops::grid(1, 1024))?;
        }
        Ok(())
    }

    fn read_selection(&self) -> Result<(u32, u32)> {
        let selected = self.encoder.stream.clone_dtoh(&self.selection)?;
        Ok((selected[0], selected[1]))
    }

    fn token_body(&mut self, physical_rows: usize) -> Result<()> {
        self.gather_embedding()?;
        self.predict_from_input()?;
        self.joint_device(physical_rows)
    }

    fn blank_body(&mut self, physical_rows: usize) -> Result<()> {
        self.joint_device(physical_rows)
    }

    fn capture_step(&mut self, token_step: bool, physical_rows: usize) -> Result<()> {
        self.encoder.stream.begin_capture(
            cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )?;
        let recorded = if token_step {
            self.token_body(physical_rows)
        } else {
            self.blank_body(physical_rows)
        };
        let captured = self
            .encoder
            .stream
            .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
        recorded.context("tdt step graph body")?;
        let graph = captured?.context("empty tdt step graph")?;
        graph.upload().context("tdt step graph upload")?;
        self.encoder.stream.synchronize()?;
        if token_step {
            self.token_graph = Some(graph);
        } else {
            self.blank_graph = Some(graph);
        }
        #[cfg(test)]
        self.graph_captures.set(self.graph_captures.get() + 1);
        Ok(())
    }

    fn replay_step(&mut self, token_step: bool, #[cfg(test)] physical_rows: usize) -> Result<()> {
        #[cfg(test)]
        if self.eager_steps.get() {
            return if token_step {
                self.token_body(physical_rows)
            } else {
                self.blank_body(physical_rows)
            };
        }
        let launched = if token_step {
            self.token_graph
                .as_ref()
                .context("tdt token graph missing")?
                .launch()
        } else {
            self.blank_graph
                .as_ref()
                .context("tdt blank graph missing")?
                .launch()
        };
        launched.context("tdt step graph launch")?;
        #[cfg(test)]
        self.graph_launches.set(self.graph_launches.get() + 1);
        Ok(())
    }

    fn decode_with(
        &mut self,
        input: &[f32],
        valid: usize,
        mut on_step: impl FnMut(&Self),
    ) -> Result<TdtTrajectory> {
        self.reset()?;
        #[cfg(test)]
        self.encoder.reset_encode_calls();
        let (encoded, _, valid_rows) = self.encoder.encode(input, valid)?;
        let projected = self.project_encoder(&encoded)?;
        let width = self.spec.decoder_hidden as usize;
        ensure!(
            !projected.is_empty() && projected.len().is_multiple_of(width),
            "decode projected extent differs from decoder hidden size"
        );
        let physical_rows = projected.len() / width;
        ensure!(
            valid_rows > 0 && valid_rows <= physical_rows,
            "decode valid encoder rows outside projected extent"
        );
        let capacity = decode_capacity(self.spec.max_symbols as usize, physical_rows)?;
        self.projected = Some(self.encoder.stream.clone_htod(&projected)?);
        self.encoder.stream.synchronize()?;
        #[cfg(test)]
        let eager = self.eager_steps.get();
        #[cfg(not(test))]
        let eager = false;
        if !eager {
            self.capture_step(true, physical_rows)?;
            self.capture_step(false, physical_rows)?;
        }
        let mut trajectory = TdtTrajectory {
            tokens: Vec::with_capacity(capacity),
            durations: Vec::with_capacity(capacity),
            frames: Vec::with_capacity(capacity),
            capacity,
        };
        trajectory.tokens.push(self.spec.blank);
        trajectory.durations.push(0);
        trajectory.frames.push(0);
        let mut token = self.spec.blank;
        let mut frame = 0usize;
        while frame < valid_rows {
            let token_step = !(self.initialized && token == self.spec.blank);
            self.write_indices(token, frame)?;
            self.replay_step(
                token_step,
                #[cfg(test)]
                physical_rows,
            )?;
            if token_step {
                self.initialized = true;
                #[cfg(test)]
                self.lstm_calls.set(self.lstm_calls.get() + self.lstm.len());
            }
            on_step(self);
            let (selected, raw_duration) = self.read_selection()?;
            decode_admit(trajectory.tokens.len(), capacity).with_context(|| {
                format!(
                    "decode capacity state: frame={frame} token={selected} initialized={}",
                    self.initialized
                )
            })?;
            let duration = duration_step(selected == self.spec.blank, raw_duration);
            trajectory.tokens.push(selected);
            trajectory.durations.push(duration);
            trajectory.frames.push(frame);
            let next_frame = frame
                .checked_add(duration as usize)
                .context("decode frame overflow")?;
            if selected == self.spec.eos || next_frame >= valid_rows {
                break;
            }
            token = selected;
            frame = next_frame;
        }
        ensure!(
            trajectory.tokens.len() == trajectory.durations.len()
                && trajectory.tokens.len() == trajectory.frames.len(),
            "decode trajectory lengths differ"
        );
        Ok(trajectory)
    }

    pub fn decode(&mut self, input: &[f32], valid: usize) -> Result<TdtTrajectory> {
        self.decode_with(input, valid, |_| {})
    }

    #[cfg(test)]
    pub fn decode_trace(
        &mut self,
        input: &[f32],
        valid: usize,
    ) -> Result<(TdtTrajectory, Vec<Vec<f32>>)> {
        let mut logits = Vec::new();
        let trajectory = self.decode_with(input, valid, |this| {
            logits.push(
                this.encoder
                    .stream
                    .clone_dtoh(&this.joint_logits)
                    .expect("joint logits readback"),
            );
        })?;
        Ok((trajectory, logits))
    }

    #[cfg(test)]
    pub fn encoder_call_count(&self) -> u64 {
        self.encoder.encode_calls()
    }

    #[cfg(test)]
    pub fn lstm_call_count(&self) -> usize {
        self.lstm_calls.get()
    }

    #[cfg(test)]
    pub fn set_eager_steps(&self, eager: bool) {
        self.eager_steps.set(eager);
    }

    #[cfg(test)]
    pub fn graph_captures(&self) -> u64 {
        self.graph_captures.get()
    }

    #[cfg(test)]
    pub fn graph_launches(&self) -> u64 {
        self.graph_launches.get()
    }
}

/// EOU cache-aware streaming session: one encoder row per step under the
/// chunked_limited [70, 1] window, with hidden-row attention caches and
/// depthwise conv caches per layer.
pub struct EouStream {
    hidden_rows: Vec<CudaSlice<f32>>,
    depthwise: Vec<CudaSlice<f32>>,
    pub(crate) cache_len: usize,
    mask: CudaSlice<u8>,
}

pub struct EouStep {
    /// The two encoder query rows of this step, `[2, hidden]`.
    pub rows: Vec<f32>,
}

impl ParakeetEncoder {
    pub fn eou_stream(&self) -> Result<EouStream> {
        let conformer = &self.conformer;
        ensure!(
            conformer.chunk_size == 2
                && conformer.left_chunks == 35
                && conformer.layers == 17
                && conformer.hidden == 512
                && conformer.heads == 8
                && conformer.conv_kernel == 9
                && conformer.depthwise_left == 8
                && !conformer.conv_batch_norm
                && !conformer.convolution_bias
                && self.blocks.len() == 17,
            "EOU streaming session requires the published EOU encoder"
        );
        let mut hidden_rows = Vec::with_capacity(17);
        let mut depthwise = Vec::with_capacity(17);
        for _ in 0..17 {
            hidden_rows.push(self.stream.alloc_zeros(0)?);
            depthwise.push(self.stream.alloc_zeros(8 * 512)?);
        }
        Ok(EouStream {
            hidden_rows,
            depthwise,
            cache_len: 0,
            mask: self.stream.clone_htod(&vec![1u8; 72])?,
        })
    }

    /// One streaming step: `window` is the mel window (9 frames at a fresh
    /// session start, 25 frames afterwards = 9 lookback + 16 chunk frames; a
    /// post-<EOU> step keeps the steady-state window), returning the two
    /// query rows of this step.
    fn eou_layer_step(
        &self,
        index: usize,
        input: &CudaSlice<f32>,
        session: &EouStream,
        hidden_cache: &CudaSlice<f32>,
        valid: usize,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>)> {
        let block = self.blocks.get(index).context("EOU layer outside range")?;
        let rows = 2u32;
        let width = self.conformer.hidden as usize;
        ensure!(
            input.len() == rows as usize * width,
            "EOU streaming layer input"
        );
        let feed_forward =
            |hidden: &mut CudaSlice<f32>, ff: &FeedForward, scale: f32| -> Result<()> {
                let normalized = self.normalize(hidden, rows, &ff.norm)?;
                let expanded = self.linear(&normalized, rows, &ff.up)?;
                let activated = self.activate(&expanded, rows, width as u32 * 4, 1)?;
                let delta = self.linear(&activated, rows, &ff.down)?;
                self.residual(hidden, &delta, scale)
            };
        let mut hidden = input.try_clone()?;
        feed_forward(&mut hidden, &block.ff1, 0.5)?;
        let attention_input = self.normalize(&hidden, rows, &block.attention_norm)?;
        let cached = session.cache_len;
        let key_rows = (cached + rows as usize) as u32;
        let table = self
            .stream
            .clone_htod(&relative_positions(key_rows as usize, width)?)?;
        let relative = self.linear(&table, 2 * key_rows - 1, &block.attention.relative)?;
        let mut keys_input = self.stream.alloc_zeros(key_rows as usize * width)?;
        self.copy_device(hidden_cache, 0, &mut keys_input, 0, cached * width)?;
        self.copy_device(
            &attention_input,
            0,
            &mut keys_input,
            cached * width,
            rows as usize * width,
        )?;
        let q_full = self.linear(&keys_input, key_rows, &block.attention.q)?;
        let k = self.linear(&keys_input, key_rows, &block.attention.k)?;
        let v = self.linear(&keys_input, key_rows, &block.attention.v)?;
        let core_full = self.stream.alloc_zeros(key_rows as usize * width)?;
        let valid_keys = cached as u32 + valid as u32;
        self.attend(AttentionLaunch {
            q: &q_full,
            k: &k,
            v: &v,
            relative: &relative,
            bias_u: &block.attention.bias_u,
            bias_v: &block.attention.bias_v,
            mask: &session.mask,
            output: &core_full,
            rows: key_rows,
            key_rows,
            query_offset: 0,
            kv_offset: 0,
            right_chunks: 1,
            valid_keys,
            lanes: 1,
            left_slide: 0,
        })?;
        let mut core: CudaSlice<f32> = self.stream.alloc_zeros(rows as usize * width)?;
        self.copy_device(
            &core_full,
            cached * width,
            &mut core,
            0,
            rows as usize * width,
        )?;
        let delta = self.linear(&core, rows, &block.attention.o)?;
        self.residual(&mut hidden, &delta, 1.0)?;
        let conv_normalized = self.normalize(&hidden, rows, &block.convolution.norm)?;
        let expanded = self.linear(&conv_normalized, rows, &block.convolution.pointwise1)?;
        let mut activated = self.activate(&expanded, rows, self.conformer.hidden, 2)?;
        if valid < rows as usize {
            let mut zeros = self.stream.alloc_zeros(rows as usize * width)?;
            self.copy_device(&activated, 0, &mut zeros, 0, valid * width)?;
            activated = zeros;
        }
        let launch_rows = if cached == 0 { rows } else { 8 + rows };
        let convolved = self.stream.alloc_zeros(launch_rows as usize * width)?;
        if launch_rows == 8 + rows {
            let mut joined = self.stream.alloc_zeros(launch_rows as usize * width)?;
            self.copy_device(&session.depthwise[index], 0, &mut joined, 0, 8 * width)?;
            self.copy_device(&activated, 0, &mut joined, 8 * width, rows as usize * width)?;
            self.launch_depthwise(
                &joined,
                &block.convolution.depthwise,
                &session.mask,
                &convolved,
                launch_rows,
                1,
            )?;
        } else {
            self.launch_depthwise(
                &activated,
                &block.convolution.depthwise,
                &session.mask,
                &convolved,
                launch_rows,
                1,
            )?;
        }
        let kept = if launch_rows == 8 + rows {
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
        let normalized = self.normalize_pair(
            &kept,
            rows,
            &block.convolution.conv_weight,
            &block.convolution.conv_bias,
        )?;
        let conv_activated = self.activate(&normalized, rows, self.conformer.hidden, 1)?;
        let delta = self.linear(&conv_activated, rows, &block.convolution.pointwise2)?;
        self.residual(&mut hidden, &delta, 1.0)?;
        feed_forward(&mut hidden, &block.ff2, 0.5)?;
        let output = self.normalize(&hidden, rows, &block.output_norm)?;
        Ok((output, attention_input, activated))
    }

    /// Runs one streaming step over the mel window; the returned rows are the
    /// two query rows `[2, hidden]` (the second is the new row).
    pub fn eou_encode_step(
        &self,
        session: &mut EouStream,
        window: &[f32],
        valid: usize,
    ) -> Result<EouStep> {
        let width = self.conformer.hidden as usize;
        ensure!(
            window.len() == 9 * 128 || window.len() == 25 * 128,
            "EOU streaming window frames {}",
            window.len() / 128
        );
        ensure!(
            valid > 0 || session.cache_len > 0,
            "EOU streaming step has no valid rows and no cache"
        );
        ensure!(
            window.iter().all(|value| value.is_finite()),
            "EOU streaming window is non-finite"
        );
        let frames = window.len() / 128;
        let subsampled = self.subsample_device(window, frames)?;
        let raw_rows = subsampled.hidden.len() / width;
        let drop = (frames - 9) / 8;
        ensure!(
            raw_rows == drop + 2,
            "EOU streaming pre-encode rows {raw_rows} with drop {drop}"
        );
        let mut query = self.stream.alloc_zeros(2 * width)?;
        self.copy_device(&subsampled.hidden, drop * width, &mut query, 0, 2 * width)?;
        let mut hidden = query;
        let mut next_attention_rows = Vec::with_capacity(17);
        let mut next_conv_rows = Vec::with_capacity(17);
        for index in 0..self.blocks.len() {
            let (output, attention_input, conv_input) =
                self.eou_layer_step(index, &hidden, session, &session.hidden_rows[index], valid)?;
            hidden = output;
            next_attention_rows.push(attention_input);
            next_conv_rows.push(conv_input);
        }
        let rows = self.stream.clone_dtoh(&hidden)?;
        let keep = (session.cache_len + 1).min(70);
        for index in 0..self.blocks.len() {
            let mut cache = self.stream.alloc_zeros(keep * width)?;
            let combined = session.cache_len + 1;
            let skip = combined - keep;
            self.copy_device(
                &session.hidden_rows[index],
                skip * width,
                &mut cache,
                0,
                (session.cache_len - skip) * width,
            )?;
            self.copy_device(
                &next_attention_rows[index],
                0,
                &mut cache,
                (keep - 1) * width,
                width,
            )?;
            session.hidden_rows[index] = cache;
            let mut depthwise = self.stream.alloc_zeros(8 * width)?;
            if session.cache_len == 0 {
                self.copy_device(&next_conv_rows[index], 0, &mut depthwise, 7 * width, width)?;
            } else {
                self.copy_device(
                    &session.depthwise[index],
                    width,
                    &mut depthwise,
                    0,
                    7 * width,
                )?;
                self.copy_device(&next_conv_rows[index], 0, &mut depthwise, 7 * width, width)?;
            }
            session.depthwise[index] = depthwise;
        }
        session.cache_len = keep;
        Ok(EouStep { rows })
    }

    /// Resets the streaming session after an <EOU>: both caches restart; the
    /// window cadence continues from the stream position.
    pub fn eou_stream_reset(&self, session: &mut EouStream) -> Result<()> {
        let width = self.conformer.hidden as usize;
        for index in 0..session.hidden_rows.len() {
            session.hidden_rows[index] = self.stream.alloc_zeros(0)?;
            session.depthwise[index] = self.stream.alloc_zeros(8 * width)?;
        }
        session.cache_len = 0;
        Ok(())
    }

    /// Multitalker cache-aware streaming session: one step per published buffer
    /// chunk (105 mel frames first, then 9 pre-encode cache frames + 112 new),
    /// a variable number of rows per step on absolute chunk-14 boundaries, all
    /// active speaker lanes stacked as rows of the same launches. Attention and
    /// depthwise caches are lane-major `[lanes, rows, hidden]`.
    pub fn multitalker_stream(&self, lanes: u32) -> Result<MultitalkerStream> {
        let conformer = &self.conformer;
        ensure!(
            conformer.chunk_size == 14
                && conformer.left_chunks == 5
                && conformer.layers == 24
                && conformer.hidden == 1024
                && conformer.heads == 8
                && conformer.conv_kernel == 9
                && conformer.depthwise_left == 8
                && !conformer.conv_batch_norm
                && !conformer.convolution_bias
                && self.blocks.len() == 24
                && (1..=4).contains(&lanes),
            "multitalker streaming session requires the published encoder and 1..=4 lanes"
        );
        let width = conformer.hidden as usize;
        // Cache tensors use the fixed 98-row stride per lane: lanes pause and
        // rejoin at different row counts, so per-lane extents are host-side.
        let stride = lanes as usize * 98 * width;
        let mut keys = Vec::with_capacity(24);
        let mut values = Vec::with_capacity(24);
        let mut depthwise = Vec::with_capacity(24);
        for _ in 0..24 {
            keys.push(self.stream.alloc_zeros(stride)?);
            values.push(self.stream.alloc_zeros(stride)?);
            depthwise.push(self.stream.alloc_zeros(lanes as usize * 8 * width)?);
        }
        Ok(MultitalkerStream {
            keys,
            values,
            depthwise,
            mask: self.stream.clone_htod(&vec![1u8; 112])?,
            lanes,
            cumulative: 0,
            lane_cumulative: [0; 4],
            cold_conv_pad: false,
            steps: 0,
            #[cfg(test)]
            trace: Vec::new(),
            #[cfg(test)]
            block_trace: Vec::new(),
        })
    }

    /// One multitalker streaming step. `windows` holds the per-lane mel chunk
    /// `[frames, 128]` (masked mode: lane-masked features; kernel mode: the raw
    /// window for every lane). `masks.spk` holds the step's `[lanes, 14]` diar
    /// frames, end-aligned to the rows with a 1.0 left pad exactly as
    /// solve_length_mismatch does; `masks.bg` arrives at `[lanes, rows]`,
    /// sliced from the step start exactly as the reference captures it.
    /// Returns the step's per-lane valid rows and the layer-0 cells for the gate.
    pub(crate) fn multitanker_encode_step(
        &self,
        session: &mut MultitalkerStream,
        windows: &[Vec<f32>],
        masks: &MultitalkerMasks<'_>,
        kernels: &MultitalkerKernels<'_>,
        active: &[usize],
        mode: MultitalkerStepMode,
    ) -> Result<MultitalkerStep> {
        let MultitalkerStepMode {
            keep_all,
            shared_pre_encode,
        } = mode;
        let width = self.conformer.hidden as usize;
        let lanes = session.lanes as usize;
        ensure!(
            windows.len() == lanes
                && masks.spk.len() == lanes * 14
                && windows.iter().all(|window| {
                    window.len().is_multiple_of(128) && window.iter().all(|value| value.is_finite())
                }),
            "multitalker streaming step inputs"
        );
        let frames = windows[0].len() / 128;
        ensure!(
            windows.iter().all(|window| window.len() == frames * 128),
            "multitalker streaming lanes disagree on chunk length"
        );
        // Kernel mode passes the same raw window for every lane: the
        // published loop pre-encodes the session chunk once (a batch of one)
        // and stacks the rows per lane; the conv stack is batch-shape
        // sensitive at the 4e-4 level, so the shared path replicates one
        // pre-encode instead of running the lane stack through it.
        let rows;
        let mut hidden;
        if shared_pre_encode {
            let output = self.subsample_device(&windows[0], frames)?;
            rows = output.hidden.len() / width;
            ensure!(rows >= 1, "multitalker streaming pre-encode rows");
            hidden = self.stream.alloc_zeros(lanes * rows * width)?;
            for lane in 0..lanes {
                self.copy_device(
                    &output.hidden,
                    0,
                    &mut hidden,
                    lane * rows * width,
                    rows * width,
                )?;
            }
        } else {
            let mut subsampled = Vec::with_capacity(lanes);
            for window in windows {
                let output = self.subsample_device(window, frames)?;
                subsampled.push(output);
            }
            rows = subsampled[0].hidden.len() / width;
            ensure!(
                rows >= 1
                    && subsampled
                        .iter()
                        .all(|output| output.hidden.len() == rows * width),
                "multitalker streaming pre-encode rows"
            );
            hidden = self.stream.alloc_zeros(lanes * rows * width)?;
            for (lane, output) in subsampled.iter().enumerate() {
                self.copy_device(
                    &output.hidden,
                    0,
                    &mut hidden,
                    lane * rows * width,
                    rows * width,
                )?;
            }
        }
        let layer0_in = self.stream.clone_dtoh(&hidden)?;
        let spk_rows = Self::align_masks(masks.spk, lanes, rows, 1.0);
        ensure!(
            masks.bg.len() == lanes * rows,
            "multitalker bg mask rows {} vs step rows {rows}",
            masks.bg.len() / lanes.max(1)
        );
        let bg_rows = masks.bg.to_vec();
        self.multitalker_speaker_ff(
            &mut hidden,
            (lanes * rows) as u32,
            (4 * rows) as u32,
            &spk_rows,
            &bg_rows,
            kernels,
        )?;
        let layer0_post = self.stream.clone_dtoh(&hidden)?;

        // One attention call per outer step over all its rows (the reference's
        // captured form: N queries over [cache | new N]); the sliding window
        // and within-chunk right edge live in the kernel's left_slide/chunk
        // arithmetic. Mid-stream steps emit their first 14 rows; the final
        // step emits them all.
        let mut live = [false; 4];
        for lane in active {
            ensure!(
                *lane < lanes,
                "active lane {lane} outside the stack of {lanes} (active {active:?}, step {})",
                session.steps
            );
            live[*lane] = true;
        }
        let first_step = session.steps == 0;
        // The shared launch requires every encoded lane to fill the grid's
        // sliding left context with its own rows. A live lane whose history
        // is shorter (not yet started, or started late) runs detached below
        // in a sub-session over its own row grid.
        let grid_cache = {
            let probe = self.multitalker_span(session.cumulative, rows as u32)?;
            session
                .cumulative
                .saturating_sub(probe.kv_offset.max(0) as u32) as usize
        };
        let mut main = [false; 4];
        let mut detached: Vec<(u32, Vec<usize>)> = Vec::new();
        for lane in 0..lanes {
            let count = session.lane_cumulative[lane];
            if !live[lane] {
                ensure!(
                    count == 0 || count >= 70,
                    "lane {lane} pauses at {count} rows; only pauses at >= 70 are handled"
                );
                continue;
            }
            if count as usize >= grid_cache || first_step {
                main[lane] = true;
            } else if let Some(group) = detached.iter_mut().find(|(seed, _)| *seed == count) {
                group.1.push(lane);
            } else {
                detached.push((count, vec![lane]));
            }
        }
        let mut span = self.multitalker_span(session.cumulative, rows as u32)?;
        span.right_chunks = u32::from(keep_all);
        let table = relative_positions(span.key_rows as usize, width)?;
        let table_dev = self.stream.clone_htod(&table)?;
        let mut step = hidden;
        let mut caches = Vec::with_capacity(self.blocks.len());
        #[cfg(test)]
        let mut traced: Vec<Vec<f32>> = Vec::with_capacity(self.blocks.len() + 1);
        #[cfg(test)]
        traced.push(self.stream.clone_dtoh(&step)?);
        for index in 0..self.blocks.len() {
            let (next, cache) =
                self.multitalker_layer_step(index, &step, session, &table_dev, &span, &main)?;
            step = next;
            #[cfg(test)]
            traced.push(self.stream.clone_dtoh(&step)?);
            caches.push(cache);
        }
        #[cfg(test)]
        {
            session.trace = traced;
        }
        for (index, cache) in caches.into_iter().enumerate() {
            session.keys[index] = cache.key;
            session.values[index] = cache.value;
            session.depthwise[index] = cache.depthwise;
        }
        session.cumulative += rows as u32;
        for (lane, is_main) in main.iter().enumerate() {
            if *is_main {
                session.lane_cumulative[lane] += rows as u32;
            }
        }
        let out_rows = if keep_all { rows } else { rows.min(14) };
        let mut output = self.stream.alloc_zeros(lanes * out_rows * width)?;
        for lane in 0..lanes {
            self.copy_device(
                &step,
                lane * rows * width,
                &mut output,
                lane * out_rows * width,
                out_rows * width,
            )?;
        }
        let rows_out = self.stream.clone_dtoh(&output)?;
        session.steps += 1;
        // Detached lanes (late starts and young histories) run through a
        // sub-session over their own row grid — a fresh cold start for seed 0,
        // or a seeded continuation carrying their existing rows — and their
        // caches and outputs merge back into this step's result.
        let mut merged_rows = rows_out.clone();
        let mut merged_in = layer0_in.clone();
        let mut merged_post = layer0_post.clone();
        let stride = 98usize;
        for (seed, group) in &detached {
            let mut fresh = self.multitalker_stream(group.len() as u32)?;
            fresh.cold_conv_pad = *seed == 0 && !first_step;
            if *seed > 0 {
                fresh.cumulative = *seed;
                fresh.steps = (*seed / 14).max(1);
                for lane_index in fresh.lane_cumulative.iter_mut() {
                    *lane_index = *seed;
                }
                let carry = (*seed as usize).min(stride);
                for index in 0..self.blocks.len() {
                    for (slot, lane) in group.iter().enumerate() {
                        self.copy_device(
                            &session.keys[index],
                            lane * stride * width,
                            &mut fresh.keys[index],
                            slot * stride * width,
                            carry * width,
                        )?;
                        self.copy_device(
                            &session.values[index],
                            lane * stride * width,
                            &mut fresh.values[index],
                            slot * stride * width,
                            carry * width,
                        )?;
                        self.copy_device(
                            &session.depthwise[index],
                            lane * 8 * width,
                            &mut fresh.depthwise[index],
                            slot * 8 * width,
                            8 * width,
                        )?;
                    }
                }
            }
            let mut group_windows = Vec::with_capacity(group.len());
            let mut group_spk = Vec::with_capacity(group.len() * 14);
            let mut group_bg = Vec::with_capacity(rows * group.len());
            for lane in group {
                group_windows.push(windows[*lane].clone());
                group_spk.extend_from_slice(&masks.spk[*lane * 14..(*lane + 1) * 14]);
                group_bg.extend_from_slice(&masks.bg[*lane * rows..(*lane + 1) * rows]);
            }
            let group_masks = MultitalkerMasks {
                spk: &group_spk,
                bg: &group_bg,
            };
            let group_kernels = MultitalkerKernels {
                speaker_up: kernels.speaker_up,
                speaker_down: kernels.speaker_down,
                background_up: kernels.background_up,
                background_down: kernels.background_down,
            };
            let slots: Vec<usize> = (0..group.len()).collect();
            let group_step = self.multitanker_encode_step(
                &mut fresh,
                &group_windows,
                &group_masks,
                &group_kernels,
                &slots,
                MultitalkerStepMode {
                    keep_all,
                    shared_pre_encode,
                },
            )?;
            #[cfg(test)]
            {
                session.trace = fresh.trace.clone();
                session.block_trace = fresh.block_trace.clone();
            }
            for index in 0..self.blocks.len() {
                for (slot, lane) in group.iter().enumerate() {
                    self.copy_device(
                        &fresh.keys[index],
                        slot * stride * width,
                        &mut session.keys[index],
                        lane * stride * width,
                        stride * width,
                    )?;
                    self.copy_device(
                        &fresh.values[index],
                        slot * stride * width,
                        &mut session.values[index],
                        lane * stride * width,
                        stride * width,
                    )?;
                    self.copy_device(
                        &fresh.depthwise[index],
                        slot * 8 * width,
                        &mut session.depthwise[index],
                        lane * 8 * width,
                        8 * width,
                    )?;
                }
            }
            for (slot, lane) in group.iter().enumerate() {
                session.lane_cumulative[*lane] += rows as u32;
                merged_rows[*lane * out_rows * width..(*lane + 1) * out_rows * width]
                    .copy_from_slice(
                        &group_step.rows[slot * out_rows * width..(slot + 1) * out_rows * width],
                    );
                merged_in[*lane * rows * width..(*lane + 1) * rows * width].copy_from_slice(
                    &group_step.layer0_in[slot * rows * width..(slot + 1) * rows * width],
                );
                merged_post[*lane * rows * width..(*lane + 1) * rows * width].copy_from_slice(
                    &group_step.layer0_post[slot * rows * width..(slot + 1) * rows * width],
                );
            }
        }
        Ok(MultitalkerStep {
            rows: merged_rows,
            layer0_in: merged_in,
            layer0_post: merged_post,
        })
    }

    /// Diar masks cover the chunk's last rows: steps with more rows than the 14
    /// diar frames left-pad with the mask default (1.0 spk / 0.0 bg); shorter
    /// final steps keep the last rows of the mask, as solve_length_mismatch does.
    fn align_masks(masks: &[f32], lanes: usize, rows: usize, default: f32) -> Vec<f32> {
        let mut aligned = vec![default; lanes * rows];
        let covered = rows.min(14);
        for lane in 0..lanes {
            for row in 0..covered {
                aligned[lane * rows + rows - covered + row] = masks[lane * 14 + 14 - covered + row];
            }
        }
        aligned
    }

    /// Key span for a step's queries on absolute chunk-14 boundaries: five left
    /// chunks plus every chunk the new rows touch (a step may straddle two).
    fn multitalker_span(&self, cumulative: u32, rows: u32) -> Result<MultitalkerWindow> {
        ensure!(rows >= 1, "multitalker step rows");
        // att_left is a sliding 70-row left context (the reference's fixed
        // 70-slot cache); the layout covers the 70 rows before the chunk.
        // A cold start carries zero-filled slots before absolute row zero.
        let first_chunk = cumulative / 14;
        let begin = i64::from(first_chunk) * 14 - 70;
        let zeros = begin.min(0).unsigned_abs() as u32;
        let kv_offset = begin;
        let span_cache = cumulative - begin.max(0) as u32;
        let key_rows = zeros + span_cache + rows;
        ensure!(
            key_rows >= rows && key_rows <= 98,
            "multitalker key span {key_rows}"
        );
        Ok(MultitalkerWindow {
            key_rows,
            kv_offset,
            zeros,
            query_offset: i64::from(cumulative),
            rows,
            right_chunks: 0,
            left_slide: 70,
        })
    }

    /// The layer-0 speaker kernels: x += ff(x * spk); x += ff(x * bg).
    fn multitalker_speaker_ff(
        &self,
        hidden: &mut CudaSlice<f32>,
        rows: u32,
        canonical: u32,
        spk: &[f32],
        bg: &[f32],
        kernels: &MultitalkerKernels<'_>,
    ) -> Result<()> {
        let width = self.conformer.hidden;
        let apply = |hidden: &mut CudaSlice<f32>,
                     masks: &[f32],
                     up: &Linear,
                     down: &Linear|
         -> Result<()> {
            let masks_dev = self.stream.clone_htod(masks)?;
            let scaled = self.stream.alloc_zeros(hidden.len())?;
            unsafe {
                self.stream
                    .launch_builder(&self.ops.row_scale)
                    .arg(&*hidden)
                    .arg(&masks_dev)
                    .arg(&scaled)
                    .arg(&(hidden.len() as u64))
                    .arg(&width)
                    .launch(flat(hidden.len()))?;
            }
            let expanded = self.lane_linear(&scaled, rows, canonical, up)?;
            let activated = self.activate(&expanded, rows, width, 0)?;
            let delta = self.lane_linear(&activated, rows, canonical, down)?;
            self.residual(hidden, &delta, 1.0)
        };
        apply(hidden, spk, kernels.speaker_up, kernels.speaker_down)?;
        apply(hidden, bg, kernels.background_up, kernels.background_down)?;
        Ok(())
    }

    fn multitalker_layer_step(
        &self,
        index: usize,
        input: &CudaSlice<f32>,
        session: &mut MultitalkerStream,
        table_dev: &CudaSlice<f32>,
        span: &MultitalkerWindow,
        live: &[bool; 4],
    ) -> Result<(CudaSlice<f32>, MultitalkerLayerStep)> {
        let rows = span.rows;
        let block = self
            .blocks
            .get(index)
            .context("encoder layer outside range")?;
        let width = self.conformer.hidden as usize;
        let lanes = session.lanes as usize;
        let rows_us = span.rows as usize;
        let stacked = (lanes * rows_us) as u32;
        let canonical = (4 * rows_us) as u32;
        let table_rows = 2 * span.key_rows - 1;
        // The kernel consumes the layer's linear_pos-projected table.
        let projected = self.linear(table_dev, table_rows, &block.attention.relative)?;
        let mut relative = self
            .stream
            .alloc_zeros(lanes * table_rows as usize * width)?;
        for lane in 0..lanes {
            self.copy_device(
                &projected,
                0,
                &mut relative,
                lane * table_rows as usize * width,
                table_rows as usize * width,
            )?;
        }
        ensure!(
            input.len() == lanes * rows_us * width,
            "multitalker layer input"
        );
        let feed_forward =
            |hidden: &mut CudaSlice<f32>, ff: &FeedForward, scale: f32| -> Result<()> {
                let normalized = self.normalize(hidden, stacked, &ff.norm)?;
                let expanded = self.lane_linear(&normalized, stacked, canonical, &ff.up)?;
                let activated = self.activate(&expanded, stacked, width as u32 * 4, 1)?;
                let delta = self.lane_linear(&activated, stacked, canonical, &ff.down)?;
                self.residual(hidden, &delta, scale)
            };
        let mut hidden = input.try_clone()?;
        feed_forward(&mut hidden, &block.ff1, 0.5)?;
        #[cfg(test)]
        session.block_trace.push(self.stream.clone_dtoh(&hidden)?);
        let normalized = self.normalize(&hidden, stacked, &block.attention_norm)?;
        let q = self.lane_linear(&normalized, stacked, canonical, &block.attention.q)?;
        let k = self.lane_linear(&normalized, stacked, canonical, &block.attention.k)?;
        let v = self.lane_linear(&normalized, stacked, canonical, &block.attention.v)?;
        let key_rows = span.key_rows as usize;
        // The span covers whole chunks, including rows that do not exist yet;
        // attention is bounded by the real row count.
        let span_cache = session
            .cumulative
            .saturating_sub(span.kv_offset.max(0) as u32) as usize;
        let zeros = span.zeros as usize;
        ensure!(
            zeros + span_cache + rows_us == key_rows,
            "multitalker layout zeros {zeros} cache {span_cache} rows {rows_us} keys {key_rows}"
        );
        let stride = 98usize;
        ensure!(
            session.keys[index].len() == lanes * stride * width
                && session.values[index].len() == lanes * stride * width
                && (0..lanes).all(|lane| {
                    let count = session.lane_cumulative[lane].min(98) as usize;
                    count >= span_cache || count == 0 || !live[lane]
                }),
            "multitalker layer {index} cache extent"
        );
        let mut cat_k = self.stream.alloc_zeros(lanes * key_rows * width)?;
        let mut cat_v = self.stream.alloc_zeros(lanes * key_rows * width)?;
        for (lane, is_live) in live.iter().enumerate() {
            let cached = session.lane_cumulative[lane].min(98) as usize;
            let lane_k = lane * key_rows * width;
            if cached > 0 && *is_live {
                let cache_start = (lane * stride + cached - span_cache) * width;
                self.copy_device(
                    &session.keys[index],
                    cache_start,
                    &mut cat_k,
                    lane_k + zeros * width,
                    span_cache * width,
                )?;
                self.copy_device(
                    &session.values[index],
                    cache_start,
                    &mut cat_v,
                    lane_k + zeros * width,
                    span_cache * width,
                )?;
            }
            self.copy_device(
                &k,
                lane * rows_us * width,
                &mut cat_k,
                lane_k + (zeros + span_cache) * width,
                rows_us * width,
            )?;
            self.copy_device(
                &v,
                lane * rows_us * width,
                &mut cat_v,
                lane_k + (zeros + span_cache) * width,
                rows_us * width,
            )?;
        }
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
            key_rows: span.key_rows,
            query_offset: span.query_offset,
            kv_offset: span.kv_offset,
            right_chunks: span.right_chunks,
            valid_keys: span.key_rows,
            lanes: session.lanes,
            left_slide: span.left_slide,
        })?;
        let delta = self.lane_linear(&core, stacked, canonical, &block.attention.o)?;
        self.residual(&mut hidden, &delta, 1.0)?;
        #[cfg(test)]
        session.block_trace.push(self.stream.clone_dtoh(&hidden)?);
        let normalized = self.normalize(&hidden, stacked, &block.convolution.norm)?;
        let expanded = self.lane_linear(
            &normalized,
            stacked,
            canonical,
            &block.convolution.pointwise1,
        )?;
        let activated = self.activate(&expanded, stacked, width as u32, 2)?;
        let launch_rows = if session.cumulative == 0 && !session.cold_conv_pad {
            rows
        } else {
            rows + 8
        };
        ensure!(
            session.depthwise[index].len() == lanes * 8 * width,
            "multitalker layer {index} depthwise cache"
        );
        let joined = if launch_rows == rows + 8 {
            let mut joined = self
                .stream
                .alloc_zeros(lanes * launch_rows as usize * width)?;
            for lane in 0..lanes {
                self.copy_device(
                    &session.depthwise[index],
                    lane * 8 * width,
                    &mut joined,
                    lane * launch_rows as usize * width,
                    8 * width,
                )?;
                self.copy_device(
                    &activated,
                    lane * rows_us * width,
                    &mut joined,
                    (lane * launch_rows as usize + 8) * width,
                    rows_us * width,
                )?;
            }
            joined
        } else {
            activated
        };
        let convolved = self.stream.alloc_zeros(joined.len())?;
        self.launch_depthwise(
            &joined,
            &block.convolution.depthwise,
            &session.mask,
            &convolved,
            launch_rows,
            session.lanes,
        )?;
        let kept = {
            let mut kept = self.stream.alloc_zeros(hidden.len())?;
            let offset = if launch_rows == rows + 8 { 8 } else { 0 };
            for lane in 0..lanes {
                self.copy_device(
                    &convolved,
                    (lane * launch_rows as usize + offset) * width,
                    &mut kept,
                    lane * rows_us * width,
                    rows_us * width,
                )?;
            }
            kept
        };
        let mut next_depth = self.stream.alloc_zeros(lanes * 8 * width)?;
        {
            let source_rows = launch_rows as usize;
            let available = source_rows.min(8);
            for (lane, is_live) in live.iter().enumerate() {
                if !is_live {
                    self.copy_device(
                        &session.depthwise[index],
                        lane * 8 * width,
                        &mut next_depth,
                        lane * 8 * width,
                        8 * width,
                    )?;
                    continue;
                }
                self.copy_device(
                    &joined,
                    (lane * source_rows + source_rows - available) * width,
                    &mut next_depth,
                    (lane * 8 + 8 - available) * width,
                    available * width,
                )?;
            }
        }
        ensure!(
            block.convolution.running_mean.is_none() && block.convolution.running_var.is_none(),
            "multitalker convolution has batch-norm statistics"
        );
        let normalized = self.normalize_pair(
            &kept,
            stacked,
            &block.convolution.conv_weight,
            &block.convolution.conv_bias,
        )?;
        let activated = self.activate(&normalized, stacked, width as u32, 1)?;
        let delta = self.lane_linear(
            &activated,
            stacked,
            canonical,
            &block.convolution.pointwise2,
        )?;
        self.residual(&mut hidden, &delta, 1.0)?;
        #[cfg(test)]
        session.block_trace.push(self.stream.clone_dtoh(&hidden)?);
        feed_forward(&mut hidden, &block.ff2, 0.5)?;
        let output = self.normalize(&hidden, stacked, &block.output_norm)?;
        let stride = 98usize;
        let mut next_k = self.stream.alloc_zeros(lanes * stride * width)?;
        let mut next_v = self.stream.alloc_zeros(lanes * stride * width)?;
        for (lane, is_live) in live.iter().enumerate() {
            if !is_live {
                self.copy_device(
                    &session.keys[index],
                    lane * stride * width,
                    &mut next_k,
                    lane * stride * width,
                    stride * width,
                )?;
                self.copy_device(
                    &session.values[index],
                    lane * stride * width,
                    &mut next_v,
                    lane * stride * width,
                    stride * width,
                )?;
                continue;
            }
            let cached = session.lane_cumulative[lane].min(98) as usize;
            let keep = (session.lane_cumulative[lane] as usize + rows_us).min(stride);
            let drop = cached + rows_us - keep;
            if keep > rows_us {
                self.copy_device(
                    &session.keys[index],
                    (lane * stride + drop) * width,
                    &mut next_k,
                    lane * stride * width,
                    (keep - rows_us) * width,
                )?;
                self.copy_device(
                    &session.values[index],
                    (lane * stride + drop) * width,
                    &mut next_v,
                    lane * stride * width,
                    (keep - rows_us) * width,
                )?;
            }
            self.copy_device(
                &k,
                lane * rows_us * width,
                &mut next_k,
                (lane * stride + keep - rows_us) * width,
                rows_us * width,
            )?;
            self.copy_device(
                &v,
                lane * rows_us * width,
                &mut next_v,
                (lane * stride + keep - rows_us) * width,
                rows_us * width,
            )?;
        }
        Ok((
            output,
            MultitalkerLayerStep {
                key: next_k,
                value: next_v,
                depthwise: next_depth,
            },
        ))
    }
}

/// Multitalker streaming session state.
pub struct MultitalkerStream {
    keys: Vec<CudaSlice<f32>>,
    values: Vec<CudaSlice<f32>>,
    depthwise: Vec<CudaSlice<f32>>,
    mask: CudaSlice<u8>,
    lanes: u32,
    cumulative: u32,
    /// Per-lane encoded rows; an inactive lane's caches freeze at its own
    /// count until the reference's gating re-admits it.
    lane_cumulative: [u32; 4],
    /// A cold start on a mid-stream window convolves with the zero-filled
    /// 8-row left cache the reference initializes every lane with.
    cold_conv_pad: bool,
    steps: u32,
    #[cfg(test)]
    pub(crate) trace: Vec<Vec<f32>>,
    #[cfg(test)]
    pub(crate) block_trace: Vec<Vec<f32>>,
}

impl MultitalkerStream {
    pub fn lanes(&self) -> u32 {
        self.lanes
    }

    pub fn cumulative(&self) -> u32 {
        self.cumulative
    }

    pub fn steps(&self) -> u32 {
        self.steps
    }
}

struct MultitalkerWindow {
    key_rows: u32,
    kv_offset: i64,
    // Zero-filled layout slots before absolute row zero at a cold start.
    zeros: u32,
    query_offset: i64,
    rows: u32,
    // The final keep_all step opens the window over every real key.
    right_chunks: u32,
    // Sliding left window in rows (the reference's att_left as rows).
    left_slide: u32,
}

/// Per-layer streaming caches produced by one multitalker layer step.
struct MultitalkerLayerStep {
    key: CudaSlice<f32>,
    value: CudaSlice<f32>,
    depthwise: CudaSlice<f32>,
}
/// Per-call mode flags for the multitalker encode step: keep-all rows
/// (kernel mode) versus the 14-row emission window, and whether the
/// pre-encode input is shared across the lane stack.
pub(crate) struct MultitalkerStepMode {
    pub(crate) keep_all: bool,
    pub(crate) shared_pre_encode: bool,
}

/// Per-step diar masks `[lanes, 14]`.
pub(crate) struct MultitalkerMasks<'a> {
    pub spk: &'a [f32],
    pub bg: &'a [f32],
}

/// Layer-0 speaker and background kernel weights.
pub(crate) struct MultitalkerKernels<'a> {
    pub speaker_up: &'a Linear,
    pub speaker_down: &'a Linear,
    pub background_up: &'a Linear,
    pub background_down: &'a Linear,
}

/// One multitalker streaming step's outputs.
pub struct MultitalkerStep {
    /// Valid output rows `[lanes, out_rows, hidden]`.
    pub rows: Vec<f32>,
    /// Layer-0 input before the speaker kernels `[lanes, rows, hidden]`.
    pub layer0_in: Vec<f32>,
    /// Layer-0 input after the speaker kernels `[lanes, rows, hidden]`.
    pub layer0_post: Vec<f32>,
}

impl ParakeetEncoder {
    /// Runs one layer over an injected input with a fresh step-0 session for
    /// the common-input bisection (no caches exist at step 0).
    #[cfg(test)]
    pub(crate) fn multitalker_inject_layer(
        &self,
        index: usize,
        input: &CudaSlice<f32>,
        session: &mut MultitalkerStream,
    ) -> Result<CudaSlice<f32>> {
        let width = self.conformer.hidden as usize;
        let span = self.multitalker_span(0, 14)?;
        let table = relative_positions(span.key_rows as usize, width)?;
        let table_dev = self.stream.clone_htod(&table)?;
        let (output, _) = self.multitalker_layer_step(
            index,
            input,
            session,
            &table_dev,
            &span,
            &[true, true, true, true],
        )?;
        Ok(output)
    }

    /// Common-input probes for the layer-0 submodules over the captured
    /// (already normed) submodule inputs; each returns the module output
    /// before the layer applies its residual.
    #[cfg(test)]
    pub(crate) fn multitalker_inject_ff1(
        &self,
        index: usize,
        input: &CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>> {
        let block = self.blocks.get(index).context("layer")?;
        let rows = (input.len() / self.conformer.hidden as usize) as u32;
        let expanded = self.linear(input, rows, &block.ff1.up)?;
        let activated = self.activate(&expanded, rows, self.conformer.hidden * 4, 1)?;
        self.linear(&activated, rows, &block.ff1.down)
    }

    #[cfg(test)]
    pub(crate) fn multitalker_inject_attn(
        &self,
        index: usize,
        input: &CudaSlice<f32>,
        session: &mut MultitalkerStream,
    ) -> Result<CudaSlice<f32>> {
        let block = self.blocks.get(index).context("layer")?;
        let width = self.conformer.hidden as usize;
        let lanes = session.lanes as usize;
        let rows = (input.len() / width / lanes) as u32;
        let stacked = rows * session.lanes;
        let q = self.linear(input, stacked, &block.attention.q)?;
        let k = self.linear(input, stacked, &block.attention.k)?;
        let v = self.linear(input, stacked, &block.attention.v)?;
        let span = self.multitalker_span(0, rows)?;
        let key_rows = span.key_rows as usize;
        let zeros = span.zeros as usize;
        let rows_us = rows as usize;
        let mut cat_k = self.stream.alloc_zeros(lanes * key_rows * width)?;
        let mut cat_v = self.stream.alloc_zeros(lanes * key_rows * width)?;
        for lane in 0..lanes {
            let lane_k = lane * key_rows * width;
            self.copy_device(
                &k,
                lane * rows_us * width,
                &mut cat_k,
                lane_k + zeros * width,
                rows_us * width,
            )?;
            self.copy_device(
                &v,
                lane * rows_us * width,
                &mut cat_v,
                lane_k + zeros * width,
                rows_us * width,
            )?;
        }
        let table = relative_positions(span.key_rows as usize, width)?;
        let table_rows = 2 * span.key_rows - 1;
        let table_dev = self.stream.clone_htod(&table)?;
        // The kernel consumes the linear_pos-projected table.
        let projected = self.linear(&table_dev, table_rows, &block.attention.relative)?;
        let lanes = session.lanes as usize;
        let mut relative = self
            .stream
            .alloc_zeros(lanes * (2 * span.key_rows as usize - 1) * width)?;
        for lane in 0..lanes {
            self.copy_device(
                &projected,
                0,
                &mut relative,
                lane * table.len(),
                table.len(),
            )?;
        }
        let core = self.stream.alloc_zeros(input.len())?;
        #[cfg(test)]
        {
            let qh = self.stream.clone_dtoh(&q)?;
            let kh = self.stream.clone_dtoh(&k)?;
            let vh = self.stream.clone_dtoh(&v)?;
            let rh = self.stream.clone_dtoh(&table_dev)?;
            let uh = self.stream.clone_dtoh(&block.attention.bias_u)?;
            let ch = self.stream.clone_dtoh(&core)?;
            println!(
                "ATTN_DBG q={:?} k={:?} v={:?} rel0={:?} bu={:?}",
                &qh[..4],
                &kh[..4],
                &vh[..4],
                &rh[..4],
                &uh[..4]
            );
            let _ = ch;
        }
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
            key_rows: span.key_rows,
            query_offset: span.query_offset,
            kv_offset: span.kv_offset,
            right_chunks: 0,
            valid_keys: span.key_rows,
            lanes: session.lanes,
            left_slide: 0,
        })?;
        self.linear(&core, stacked, &block.attention.o)
    }

    #[cfg(test)]
    pub(crate) fn multitalker_inject_conv(
        &self,
        index: usize,
        input: &CudaSlice<f32>,
        session: &mut MultitalkerStream,
    ) -> Result<CudaSlice<f32>> {
        let block = self.blocks.get(index).context("layer")?;
        let width = self.conformer.hidden as usize;
        let rows = (input.len() / width / session.lanes as usize) as u32;
        let stacked = rows * session.lanes;
        let expanded = self.linear(input, stacked, &block.convolution.pointwise1)?;
        let activated = self.activate(&expanded, stacked, self.conformer.hidden, 2)?;
        let convolved = self.stream.alloc_zeros(activated.len())?;
        self.launch_depthwise(
            &activated,
            &block.convolution.depthwise,
            &session.mask,
            &convolved,
            rows,
            session.lanes,
        )?;
        let normalized = self.normalize_pair(
            &convolved,
            stacked,
            &block.convolution.conv_weight,
            &block.convolution.conv_bias,
        )?;
        let activated = self.activate(&normalized, stacked, self.conformer.hidden, 1)?;
        self.linear(&activated, stacked, &block.convolution.pointwise2)
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
        let first = stream_window(0, 4, StreamOmit::None)?;
        ensure!(
            first.key_rows == 4
                && first.query_offset == 0
                && first.kv_offset == 0
                && first.cached == 0,
            "first window"
        );
        let chunk14 = stream_window(56, 4, StreamOmit::None)?;
        ensure!(
            chunk14.key_rows == 60
                && chunk14.query_offset == 56
                && chunk14.kv_offset == 0
                && chunk14.cached == 56,
            "chunk 14 window"
        );
        let chunk15 = stream_window(60, 4, StreamOmit::None)?;
        ensure!(
            chunk15.key_rows == 60
                && chunk15.query_offset == 60
                && chunk15.kv_offset == 4
                && chunk15.cached == 56,
            "chunk 15 window"
        );
        let later = stream_window(5000, 4, StreamOmit::None)?;
        ensure!(
            later.key_rows == 60 && later.cached == 56 && later.kv_offset == 5000 - 56,
            "cumulative 5000 changed the resident length"
        );
        ensure!(
            stream_window(0, 3, StreamOmit::None).is_err(),
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

    #[test]
    fn omitted_keys_select_the_query_table() -> Result<()> {
        let omitted = stream_window(60, 4, StreamOmit::Keys)?;
        let (attend_rows, attend_offset) = attend_span(&omitted, 4);
        let self_row = 3usize * 2;
        let query_table = relative_positions(attend_rows as usize, 2)?;
        let wide_table = relative_positions(omitted.key_rows as usize, 2)?;
        ensure!(
            omitted.key_rows == 60
                && attend_rows == 4
                && i64::from(attend_offset) == omitted.query_offset
                && table_index(attend_rows)? == 0
                && (query_table[self_row] - 0.0).abs() < 1e-6
                && (query_table[self_row + 1] - 1.0).abs() < 1e-6
                && (wide_table[self_row] - 56f32.sin()).abs() < 1e-6
                && (wide_table[self_row + 1] - 56f32.cos()).abs() < 1e-6,
            "omitted keys select the 4-row table"
        );
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
    #[test]
    fn tdt_duration_and_capacity_rules() -> Result<()> {
        assert_eq!(duration_step(true, 0), 1);
        assert_eq!(duration_step(false, 0), 0);
        assert_eq!(duration_step(true, 3), 3);
        assert_eq!(duration_step(false, 4), 4);
        assert_eq!(decode_capacity(10, 37)?, 370);
        decode_admit(369, 370)?;
        assert!(decode_admit(370, 370).is_err());
        assert!(decode_capacity(usize::MAX, 2).is_err());
        assert!(decode_capacity(10, 0).is_err());
        println!(
            "PASS TDT_CAPACITY blank_zero=1 nonblank_zero=0 capacity=370 exact_fit=true overflow_rejected=true per_frame_counter=false"
        );
        Ok(())
    }
}
