//! F32 audio-tokenizer decoder. Rows are time-major. Codes are `[frames, 16]` in `0..=2047`.
use super::{
    Device,
    blas::linear_f32_on,
    contracts::{TensorMut, view},
    cubin,
    flash::{F32Attention, F32Mask},
    ops,
};
use crate::backend::gpu::{
    CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use anyhow::{Context, Result, ensure};
use cudarc::cublas::CudaBlas;
use safetensors::{Dtype, SafeTensors};
use std::{path::Path, sync::Arc};

const MAX_FRAMES: usize = 256;
const FRAME_SAMPLES: usize = 1920;
const BOOKS: usize = 16;
const CODE_MAX: i32 = 2047;
const VOCAB: i32 = 2048;
const CODE_DIM: usize = 256;
const QUANT_DIM: usize = 512;
const LATENT: usize = 1024;
const HIDDEN: usize = 512;
const HEADS: i32 = 16;
const HEAD_DIM: i32 = 64;
const MLP: usize = 1024;
const WINDOW: i32 = 72;
const LAYERS: usize = 8;
const Q_WIDTH: usize = (HEADS as usize) * (HEAD_DIM as usize);
const PEAK: usize = MAX_FRAMES * FRAME_SAMPLES * 96;
const RMS_EPS: f32 = 1e-5;
const LN_EPS: f32 = 1e-6;

pub fn check_codes(codes: &[i32]) -> Result<usize> {
    ensure!(!codes.is_empty(), "codes length");
    ensure!(codes.len().is_multiple_of(BOOKS), "codes length");
    let frames = codes.len() / BOOKS;
    ensure!((1..=MAX_FRAMES).contains(&frames), "codes length");
    for (index, code) in codes.iter().enumerate() {
        ensure!((0..=CODE_MAX).contains(code), "code {code} at {index}");
    }
    Ok(frames)
}

#[cfg(test)]
fn slide_keeps(q: i32, k: i32, window: i32) -> bool {
    k <= q && k > q - window
}

struct Kernels {
    rms: CudaFunction,
    bias: CudaFunction,
    gather: CudaFunction,
    causal: CudaFunction,
    trans: CudaFunction,
    rope: CudaFunction,
    rotate: CudaFunction,
    add: CudaFunction,
    snake: CudaFunction,
    norm: CudaFunction,
    gelu: CudaFunction,
    scale: CudaFunction,
    silu: CudaFunction,
    clamp: CudaFunction,
}

struct Snake {
    alpha: CudaSlice<f32>,
    beta: CudaSlice<f32>,
    channels: i32,
}

struct Causal {
    weight: CudaSlice<f32>,
    bias: CudaSlice<f32>,
    in_dim: i32,
    out_dim: i32,
    kernel: i32,
    dilation: i32,
    groups: i32,
}

struct Trans {
    weight: CudaSlice<f32>,
    bias: CudaSlice<f32>,
    in_dim: i32,
    out_dim: i32,
    kernel: i32,
    stride: i32,
    trimmed: bool,
}

struct Layer {
    rms1: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    o: CudaSlice<f32>,
    scale_attn: CudaSlice<f32>,
    rms2: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    up: CudaSlice<f32>,
    down: CudaSlice<f32>,
    scale_mlp: CudaSlice<f32>,
}

struct Unit {
    act1: Snake,
    conv7: Causal,
    act2: Snake,
    conv1: Causal,
}

struct DecBlock {
    act: Snake,
    up: Trans,
    units: [Unit; 3],
    stride: i32,
}

struct ConvNext {
    dw: Causal,
    norm_w: CudaSlice<f32>,
    norm_b: CudaSlice<f32>,
    pw1: CudaSlice<f32>,
    pw1_b: CudaSlice<f32>,
    pw2: CudaSlice<f32>,
    pw2_b: CudaSlice<f32>,
    gamma: CudaSlice<f32>,
}

struct Work {
    codes: CudaSlice<i32>,
    e0: CudaSlice<f32>,
    e1: CudaSlice<f32>,
    s0: CudaSlice<f32>,
    s1: CudaSlice<f32>,
    wide: CudaSlice<f32>,
    h0: CudaSlice<f32>,
    h1: CudaSlice<f32>,
    norm: CudaSlice<f32>,
    proj: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    upw: CudaSlice<f32>,
    cos: CudaSlice<f32>,
    sin: CudaSlice<f32>,
    a: CudaSlice<f32>,
    b: CudaSlice<f32>,
    c: CudaSlice<f32>,
    d: CudaSlice<f32>,
}

struct Op<'a> {
    stream: &'a Arc<CudaStream>,
    blas: &'a CudaBlas,
    kernels: &'a Kernels,
    attention: &'a F32Attention,
    count: &'a mut u64,
}

impl Op<'_> {
    fn tick(&mut self) {
        *self.count += 1;
    }
}

pub struct Tokenizer {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    kernels: Kernels,
    attention: F32Attention,
    books: Vec<CudaSlice<f32>>,
    quant_first: Causal,
    quant_rest: Causal,
    pre_conv: Causal,
    input_w: CudaSlice<f32>,
    input_b: CudaSlice<f32>,
    layers: Vec<Layer>,
    final_rms: CudaSlice<f32>,
    output_w: CudaSlice<f32>,
    output_b: CudaSlice<f32>,
    inv: CudaSlice<f32>,
    ups: [Trans; 2],
    nexts: [ConvNext; 2],
    dec0: Causal,
    blocks: [DecBlock; 4],
    tail_act: Snake,
    tail_conv: Causal,
    work: Work,
    weight_bytes: u64,
    launches: u64,
}

impl Tokenizer {
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    pub fn launches(&self) -> u64 {
        self.launches
    }

    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let bytes = std::fs::read(path).context("tokenizer weights")?;
        let st = SafeTensors::deserialize(&bytes).context("tokenizer safetensors")?;
        let stream = Arc::clone(&device.stream);
        let mut weight_bytes = 0u64;
        let common = cubin::module(&device.ctx, "q38_common")?;
        let vision = cubin::module(&device.ctx, "vision_ops")?;
        let act = cubin::module(&device.ctx, "act")?;
        let kernel = |name: &str| {
            common
                .load_function(name)
                .with_context(|| format!("kernel {name}"))
        };
        let vision_kernel = |name: &str| {
            vision
                .load_function(name)
                .with_context(|| format!("kernel {name}"))
        };
        let kernels = Kernels {
            rms: vision_kernel("vision_rms_scale_f32")?,
            bias: vision_kernel("vision_bias_f32")?,
            gather: vision_kernel("vision_row_gather_f32")?,
            causal: kernel("causal_conv_rows_f32")?,
            trans: kernel("conv_transpose_rows_f32")?,
            rope: kernel("rope_cos_sin_f32")?,
            rotate: kernel("rope_rotate_half_f32")?,
            add: kernel("add_scale_f32")?,
            snake: kernel("snake_beta_rows_f32")?,
            norm: kernel("q38_layer_norm_f32")?,
            gelu: kernel("q38_gelu_f32")?,
            scale: kernel("channel_scale_residual_f32")?,
            silu: act
                .load_function("silu_mul_split_f32")
                .context("kernel silu_mul_split_f32")?,
            clamp: kernel("clamp_f32")?,
        };
        let geometry = [
            u32::try_from(HEADS)?,
            u32::try_from(MAX_FRAMES)?,
            u32::try_from(MAX_FRAMES)?,
            u32::try_from(HEAD_DIM)?,
            u32::try_from(HEAD_DIM)?,
        ];
        let attention = F32Attention::new(
            Arc::clone(&stream),
            geometry,
            F32Attention::required_bytes(geometry)?,
        )?;
        let mut books = Vec::with_capacity(BOOKS);
        books.push(codebook(
            &st,
            &stream,
            &mut weight_bytes,
            "decoder.quantizer.rvq_first.vq.layers.0._codebook",
        )?);
        for layer in 0..15 {
            books.push(codebook(
                &st,
                &stream,
                &mut weight_bytes,
                &format!("decoder.quantizer.rvq_rest.vq.layers.{layer}._codebook"),
            )?);
        }
        let quant_first = causal_conv(
            &st,
            &stream,
            &mut weight_bytes,
            "decoder.quantizer.rvq_first.output_proj",
            CausalSpec {
                out_dim: QUANT_DIM,
                in_dim: CODE_DIM,
                kernel: 1,
                dilation: 1,
                groups: 1,
                bias: false,
            },
        )?;
        let quant_rest = causal_conv(
            &st,
            &stream,
            &mut weight_bytes,
            "decoder.quantizer.rvq_rest.output_proj",
            CausalSpec {
                out_dim: QUANT_DIM,
                in_dim: CODE_DIM,
                kernel: 1,
                dilation: 1,
                groups: 1,
                bias: false,
            },
        )?;
        let pre_conv = causal_conv(
            &st,
            &stream,
            &mut weight_bytes,
            "decoder.pre_conv.conv",
            CausalSpec {
                out_dim: LATENT,
                in_dim: QUANT_DIM,
                kernel: 3,
                dilation: 1,
                groups: 1,
                bias: true,
            },
        )?;
        let input_w = f32_tensor(
            &st,
            &stream,
            &mut weight_bytes,
            "decoder.pre_transformer.input_proj.weight",
            &[HIDDEN, LATENT],
        )?;
        let input_b = f32_tensor(
            &st,
            &stream,
            &mut weight_bytes,
            "decoder.pre_transformer.input_proj.bias",
            &[HIDDEN],
        )?;
        let mut layers = Vec::with_capacity(LAYERS);
        for index in 0..LAYERS {
            layers.push(load_layer(&st, &stream, &mut weight_bytes, index)?);
        }
        let final_rms = f32_tensor(
            &st,
            &stream,
            &mut weight_bytes,
            "decoder.pre_transformer.norm.weight",
            &[HIDDEN],
        )?;
        let output_w = f32_tensor(
            &st,
            &stream,
            &mut weight_bytes,
            "decoder.pre_transformer.output_proj.weight",
            &[LATENT, HIDDEN],
        )?;
        let output_b = f32_tensor(
            &st,
            &stream,
            &mut weight_bytes,
            "decoder.pre_transformer.output_proj.bias",
            &[LATENT],
        )?;
        let inv_host = rope_inv();
        ensure!(
            inv_host.iter().all(|value| value.is_finite()),
            "non-finite rope table"
        );
        let inv = stream.clone_htod(&inv_host).context("upload rope table")?;
        let mut ups = Vec::with_capacity(2);
        let mut nexts = Vec::with_capacity(2);
        for index in 0..2 {
            ups.push(trans_conv(
                &st,
                &stream,
                &mut weight_bytes,
                &format!("decoder.upsample.{index}.0.conv"),
                TransSpec {
                    in_dim: LATENT,
                    out_dim: LATENT,
                    kernel: 2,
                    stride: 2,
                    trimmed: false,
                },
            )?);
            nexts.push(load_next(
                &st,
                &stream,
                &mut weight_bytes,
                &format!("decoder.upsample.{index}.1"),
            )?);
        }
        let dec0 = causal_conv(
            &st,
            &stream,
            &mut weight_bytes,
            "decoder.decoder.0.conv",
            CausalSpec {
                out_dim: 1536,
                in_dim: LATENT,
                kernel: 7,
                dilation: 1,
                groups: 1,
                bias: true,
            },
        )?;
        let specs = [(1536, 768, 8), (768, 384, 5), (384, 192, 4), (192, 96, 3)];
        let mut blocks = Vec::with_capacity(4);
        for (index, (in_dim, out_dim, rate)) in specs.iter().copied().enumerate() {
            blocks.push(load_block(
                &st,
                &stream,
                &mut weight_bytes,
                index + 1,
                in_dim,
                out_dim,
                rate,
            )?);
        }
        let tail_act = snake(&st, &stream, &mut weight_bytes, "decoder.decoder.5", 96)?;
        let tail_conv = causal_conv(
            &st,
            &stream,
            &mut weight_bytes,
            "decoder.decoder.6.conv",
            CausalSpec {
                out_dim: 1,
                in_dim: 96,
                kernel: 7,
                dilation: 1,
                groups: 1,
                bias: true,
            },
        )?;
        let zeros = |n: usize| stream.alloc_zeros::<f32>(n).context("activation buffer");
        let work = Work {
            codes: stream
                .alloc_zeros::<i32>(BOOKS * MAX_FRAMES)
                .context("code buffer")?,
            e0: zeros(MAX_FRAMES * CODE_DIM)?,
            e1: zeros(MAX_FRAMES * CODE_DIM)?,
            s0: zeros(MAX_FRAMES * QUANT_DIM)?,
            s1: zeros(MAX_FRAMES * QUANT_DIM)?,
            wide: zeros(MAX_FRAMES * LATENT)?,
            h0: zeros(MAX_FRAMES * HIDDEN)?,
            h1: zeros(MAX_FRAMES * HIDDEN)?,
            norm: zeros(MAX_FRAMES * HIDDEN)?,
            proj: zeros(MAX_FRAMES * HIDDEN)?,
            q: zeros(MAX_FRAMES * Q_WIDTH)?,
            k: zeros(MAX_FRAMES * Q_WIDTH)?,
            v: zeros(MAX_FRAMES * Q_WIDTH)?,
            attn: zeros(MAX_FRAMES * Q_WIDTH)?,
            gate: zeros(MAX_FRAMES * MLP)?,
            upw: zeros(MAX_FRAMES * MLP)?,
            cos: zeros(MAX_FRAMES * HEAD_DIM as usize)?,
            sin: zeros(MAX_FRAMES * HEAD_DIM as usize)?,
            a: zeros(PEAK)?,
            b: zeros(PEAK)?,
            c: zeros(PEAK)?,
            d: zeros(PEAK)?,
        };
        let blas = CudaBlas::new(Arc::clone(&stream)).context("cublas")?;
        stream.synchronize().context("tokenizer load")?;
        let [up0, up1] = ups.try_into().ok().context("upsample count")?;
        let [next0, next1] = nexts.try_into().ok().context("convnext count")?;
        let [b0, b1, b2, b3] = blocks.try_into().ok().context("decoder block count")?;
        Ok(Self {
            stream,
            blas,
            kernels,
            attention,
            books,
            quant_first,
            quant_rest,
            pre_conv,
            input_w,
            input_b,
            layers,
            final_rms,
            output_w,
            output_b,
            inv,
            ups: [up0, up1],
            nexts: [next0, next1],
            dec0,
            blocks: [b0, b1, b2, b3],
            tail_act,
            tail_conv,
            work,
            weight_bytes,
            launches: 0,
        })
    }

    pub fn decode(&mut self, codes: &[i32]) -> Result<Vec<f32>> {
        let frames_usize = check_codes(codes)?;
        let frames = i32::try_from(frames_usize).context("frames")?;
        let mut packed = vec![0i32; BOOKS * MAX_FRAMES];
        for frame in 0..frames_usize {
            for book in 0..BOOKS {
                packed[book * MAX_FRAMES + frame] = codes[frame * BOOKS + book];
            }
        }
        self.stream
            .memcpy_htod(&packed, &mut self.work.codes)
            .context("upload codes")?;
        let launches = forward(self, frames)?;
        self.launches = launches;
        let samples = usize::try_from(frames).unwrap_or(0) * FRAME_SAMPLES;
        ensure!(samples == frames_usize * FRAME_SAMPLES, "samples");
        self.stream.synchronize().context("decode synchronize")?;
        let mut host = vec![0f32; samples];
        self.stream
            .memcpy_dtoh(&self.work.a.slice(..samples), &mut host)
            .context("download waveform")?;
        self.stream.synchronize().context("download synchronize")?;
        ensure!(
            host.iter().all(|value| value.is_finite()),
            "non-finite waveform"
        );
        ensure!(
            host.iter().all(|value| (-1.0..=1.0).contains(value)),
            "waveform outside [-1, 1]"
        );
        Ok(host)
    }
}

fn rope_inv() -> [f32; 32] {
    let mut inv = [0f32; 32];
    for (index, slot) in inv.iter_mut().enumerate() {
        *slot = 1.0 / 10000f32.powf((2 * index) as f32 / 64.0);
    }
    inv
}

fn read_f32(st: &SafeTensors, key: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let view = st.tensor(key).with_context(|| format!("missing {key}"))?;
    let count = shape.iter().product::<usize>();
    ensure!(view.dtype() == Dtype::F32, "{key} dtype");
    ensure!(view.shape() == shape, "{key} shape");
    let (chunks, rest) = view.data().as_chunks::<4>();
    ensure!(rest.is_empty() && chunks.len() == count, "{key} bytes");
    let mut values = Vec::with_capacity(count);
    for chunk in chunks {
        let value = f32::from_le_bytes(*chunk);
        ensure!(value.is_finite(), "non-finite {key}");
        values.push(value);
    }
    Ok(values)
}

fn f32_tensor(
    st: &SafeTensors,
    stream: &Arc<CudaStream>,
    bytes: &mut u64,
    key: &str,
    shape: &[usize],
) -> Result<CudaSlice<f32>> {
    let values = read_f32(st, key, shape)?;
    *bytes += (values.len() * 4) as u64;
    stream
        .clone_htod(&values)
        .with_context(|| format!("upload {key}"))
}

fn codebook(
    st: &SafeTensors,
    stream: &Arc<CudaStream>,
    bytes: &mut u64,
    prefix: &str,
) -> Result<CudaSlice<f32>> {
    let sum = read_f32(
        st,
        &format!("{prefix}.embedding_sum"),
        &[VOCAB as usize, CODE_DIM],
    )?;
    let usage = read_f32(st, &format!("{prefix}.cluster_usage"), &[VOCAB as usize])?;
    *bytes += ((sum.len() + usage.len()) * 4) as u64;
    let mut table = vec![0f32; sum.len()];
    for (row, used) in usage.iter().enumerate() {
        let denom = used.max(1e-5);
        for dim in 0..CODE_DIM {
            let value = sum[row * CODE_DIM + dim] / denom;
            ensure!(value.is_finite(), "non-finite {prefix}");
            table[row * CODE_DIM + dim] = value;
        }
    }
    stream
        .clone_htod(&table)
        .with_context(|| format!("upload {prefix}"))
}

fn pack_causal(published: &[f32], out: usize, in_pg: usize, kernel: usize) -> Vec<f32> {
    let mut device = vec![0f32; published.len()];
    for o in 0..out {
        for c in 0..in_pg {
            for k in 0..kernel {
                device[(k * in_pg + c) * out + o] = published[(o * in_pg + c) * kernel + k];
            }
        }
    }
    device
}

struct CausalSpec {
    out_dim: usize,
    in_dim: usize,
    kernel: usize,
    dilation: usize,
    groups: usize,
    bias: bool,
}

struct TransSpec {
    in_dim: usize,
    out_dim: usize,
    kernel: usize,
    stride: usize,
    trimmed: bool,
}

fn causal_conv(
    st: &SafeTensors,
    stream: &Arc<CudaStream>,
    bytes: &mut u64,
    prefix: &str,
    spec: CausalSpec,
) -> Result<Causal> {
    ensure!(
        spec.groups > 0
            && spec.in_dim.is_multiple_of(spec.groups)
            && spec.out_dim.is_multiple_of(spec.groups),
        "{prefix} groups"
    );
    let in_pg = spec.in_dim / spec.groups;
    let published = read_f32(
        st,
        &format!("{prefix}.weight"),
        &[spec.out_dim, in_pg, spec.kernel],
    )?;
    *bytes += (published.len() * 4) as u64;
    let packed = pack_causal(&published, spec.out_dim, in_pg, spec.kernel);
    let weight = stream
        .clone_htod(&packed)
        .with_context(|| format!("upload {prefix}.weight"))?;
    let bias = if spec.bias {
        f32_tensor(
            st,
            stream,
            bytes,
            &format!("{prefix}.bias"),
            &[spec.out_dim],
        )?
    } else {
        stream
            .alloc_zeros::<f32>(spec.out_dim)
            .with_context(|| format!("zero bias {prefix}"))?
    };
    Ok(Causal {
        weight,
        bias,
        in_dim: i32::try_from(spec.in_dim)?,
        out_dim: i32::try_from(spec.out_dim)?,
        kernel: i32::try_from(spec.kernel)?,
        dilation: i32::try_from(spec.dilation)?,
        groups: i32::try_from(spec.groups)?,
    })
}

fn trans_conv(
    st: &SafeTensors,
    stream: &Arc<CudaStream>,
    bytes: &mut u64,
    prefix: &str,
    spec: TransSpec,
) -> Result<Trans> {
    if spec.trimmed {
        ensure!(spec.kernel == 2 * spec.stride, "{prefix} trim geometry");
    } else {
        ensure!(spec.kernel == spec.stride, "{prefix} transpose geometry");
    }
    Ok(Trans {
        weight: f32_tensor(
            st,
            stream,
            bytes,
            &format!("{prefix}.weight"),
            &[spec.in_dim, spec.out_dim, spec.kernel],
        )?,
        bias: f32_tensor(
            st,
            stream,
            bytes,
            &format!("{prefix}.bias"),
            &[spec.out_dim],
        )?,
        in_dim: i32::try_from(spec.in_dim)?,
        out_dim: i32::try_from(spec.out_dim)?,
        kernel: i32::try_from(spec.kernel)?,
        stride: i32::try_from(spec.stride)?,
        trimmed: spec.trimmed,
    })
}

fn snake(
    st: &SafeTensors,
    stream: &Arc<CudaStream>,
    bytes: &mut u64,
    prefix: &str,
    channels: usize,
) -> Result<Snake> {
    Ok(Snake {
        alpha: f32_tensor(st, stream, bytes, &format!("{prefix}.alpha"), &[channels])?,
        beta: f32_tensor(st, stream, bytes, &format!("{prefix}.beta"), &[channels])?,
        channels: i32::try_from(channels)?,
    })
}

fn load_layer(
    st: &SafeTensors,
    stream: &Arc<CudaStream>,
    bytes: &mut u64,
    index: usize,
) -> Result<Layer> {
    let p = format!("decoder.pre_transformer.layers.{index}");
    let mut w =
        |name: &str, shape: &[usize]| f32_tensor(st, stream, bytes, &format!("{p}.{name}"), shape);
    Ok(Layer {
        rms1: w("input_layernorm.weight", &[HIDDEN])?,
        q: w("self_attn.q_proj.weight", &[Q_WIDTH, HIDDEN])?,
        k: w("self_attn.k_proj.weight", &[Q_WIDTH, HIDDEN])?,
        v: w("self_attn.v_proj.weight", &[Q_WIDTH, HIDDEN])?,
        o: w("self_attn.o_proj.weight", &[HIDDEN, Q_WIDTH])?,
        scale_attn: w("self_attn_layer_scale.scale", &[HIDDEN])?,
        rms2: w("post_attention_layernorm.weight", &[HIDDEN])?,
        gate: w("mlp.gate_proj.weight", &[MLP, HIDDEN])?,
        up: w("mlp.up_proj.weight", &[MLP, HIDDEN])?,
        down: w("mlp.down_proj.weight", &[HIDDEN, MLP])?,
        scale_mlp: w("mlp_layer_scale.scale", &[HIDDEN])?,
    })
}

fn load_next(
    st: &SafeTensors,
    stream: &Arc<CudaStream>,
    bytes: &mut u64,
    prefix: &str,
) -> Result<ConvNext> {
    Ok(ConvNext {
        dw: causal_conv(
            st,
            stream,
            bytes,
            &format!("{prefix}.dwconv.conv"),
            CausalSpec {
                out_dim: LATENT,
                in_dim: LATENT,
                kernel: 7,
                dilation: 1,
                groups: LATENT,
                bias: true,
            },
        )?,
        norm_w: f32_tensor(
            st,
            stream,
            bytes,
            &format!("{prefix}.norm.weight"),
            &[LATENT],
        )?,
        norm_b: f32_tensor(st, stream, bytes, &format!("{prefix}.norm.bias"), &[LATENT])?,
        pw1: f32_tensor(
            st,
            stream,
            bytes,
            &format!("{prefix}.pwconv1.weight"),
            &[LATENT * 4, LATENT],
        )?,
        pw1_b: f32_tensor(
            st,
            stream,
            bytes,
            &format!("{prefix}.pwconv1.bias"),
            &[LATENT * 4],
        )?,
        pw2: f32_tensor(
            st,
            stream,
            bytes,
            &format!("{prefix}.pwconv2.weight"),
            &[LATENT, LATENT * 4],
        )?,
        pw2_b: f32_tensor(
            st,
            stream,
            bytes,
            &format!("{prefix}.pwconv2.bias"),
            &[LATENT],
        )?,
        gamma: f32_tensor(st, stream, bytes, &format!("{prefix}.gamma"), &[LATENT])?,
    })
}

fn load_unit(
    st: &SafeTensors,
    stream: &Arc<CudaStream>,
    bytes: &mut u64,
    prefix: &str,
    channels: usize,
    dilation: usize,
) -> Result<Unit> {
    Ok(Unit {
        act1: snake(st, stream, bytes, &format!("{prefix}.act1"), channels)?,
        conv7: causal_conv(
            st,
            stream,
            bytes,
            &format!("{prefix}.conv1.conv"),
            CausalSpec {
                out_dim: channels,
                in_dim: channels,
                kernel: 7,
                dilation,
                groups: 1,
                bias: true,
            },
        )?,
        act2: snake(st, stream, bytes, &format!("{prefix}.act2"), channels)?,
        conv1: causal_conv(
            st,
            stream,
            bytes,
            &format!("{prefix}.conv2.conv"),
            CausalSpec {
                out_dim: channels,
                in_dim: channels,
                kernel: 1,
                dilation: 1,
                groups: 1,
                bias: true,
            },
        )?,
    })
}

fn load_block(
    st: &SafeTensors,
    stream: &Arc<CudaStream>,
    bytes: &mut u64,
    index: usize,
    in_dim: usize,
    out_dim: usize,
    rate: usize,
) -> Result<DecBlock> {
    let prefix = format!("decoder.decoder.{index}.block");
    Ok(DecBlock {
        act: snake(st, stream, bytes, &format!("{prefix}.0"), in_dim)?,
        up: trans_conv(
            st,
            stream,
            bytes,
            &format!("{prefix}.1.conv"),
            TransSpec {
                in_dim,
                out_dim,
                kernel: 2 * rate,
                stride: rate,
                trimmed: true,
            },
        )?,
        units: [
            load_unit(st, stream, bytes, &format!("{prefix}.2"), out_dim, 1)?,
            load_unit(st, stream, bytes, &format!("{prefix}.3"), out_dim, 3)?,
            load_unit(st, stream, bytes, &format!("{prefix}.4"), out_dim, 9)?,
        ],
        stride: i32::try_from(rate)?,
    })
}

fn dev<T>(slice: &CudaSlice<T>, stream: &CudaStream) -> u64
where
    T: cudarc::driver::DeviceRepr,
{
    slice.device_ptr(stream).0
}

fn fits(buf: &CudaSlice<f32>, rows: usize, cols: usize) -> Result<()> {
    let n = rows.checked_mul(cols).context("activation size")?;
    ensure!(n <= buf.len() && rows > 0 && cols > 0, "activation size");
    Ok(())
}

fn gemm(
    op: &mut Op<'_>,
    weight: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    y: &CudaSlice<f32>,
    rows: usize,
    out_dim: usize,
    in_dim: usize,
) -> Result<()> {
    fits(weight, out_dim, in_dim)?;
    fits(x, rows, in_dim)?;
    fits(y, rows, out_dim)?;
    let stream = op.stream;
    let blas = op.blas;
    let w = view(
        weight,
        stream,
        0,
        [u32::try_from(out_dim)?, u32::try_from(in_dim)?],
        [in_dim as u64, 1],
    )?;
    let input = view(
        x,
        stream,
        0,
        [u32::try_from(rows)?, u32::try_from(in_dim)?],
        [in_dim as u64, 1],
    )?;
    let mut output = TensorMut(view(
        y,
        stream,
        0,
        [u32::try_from(rows)?, u32::try_from(out_dim)?],
        [out_dim as u64, 1],
    )?);
    linear_f32_on(blas, w, input, &mut output)?;
    op.tick();
    Ok(())
}

fn row_bias(
    op: &mut Op<'_>,
    y: &CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    rows: usize,
    width: usize,
) -> Result<()> {
    fits(y, rows, width)?;
    ensure!(bias.len() >= width, "bias width");
    let rows_u = u32::try_from(rows)?;
    let width_u = u32::try_from(width)?;
    let count = rows_u.checked_mul(width_u).context("bias count")? as usize;
    let yp = dev(y, op.stream);
    let bp = dev(bias, op.stream);
    unsafe {
        op.stream
            .launch_builder(&op.kernels.bias)
            .arg(&yp)
            .arg(&bp)
            .arg(&yp)
            .arg(&rows_u)
            .arg(&width_u)
            .launch(ops::flat(count))?;
    }
    op.tick();
    Ok(())
}

fn gemm_bias(
    op: &mut Op<'_>,
    weight: &CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    y: &CudaSlice<f32>,
    rows: usize,
    dims: (usize, usize),
) -> Result<()> {
    let (out_dim, in_dim) = dims;
    gemm(op, weight, x, y, rows, out_dim, in_dim)?;
    row_bias(op, y, bias, rows, out_dim)
}

fn rms(
    op: &mut Op<'_>,
    x: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    y: &CudaSlice<f32>,
    rows: i32,
    width: i32,
) -> Result<()> {
    fits(x, rows as usize, width as usize)?;
    fits(y, rows as usize, width as usize)?;
    ensure!(weight.len() >= width as usize, "rms width");
    let xp = dev(x, op.stream);
    let wp = dev(weight, op.stream);
    let yp = dev(y, op.stream);
    let width_u = u32::try_from(width)?;
    let scale_stride = 1u32;
    let bf16 = 0u64;
    unsafe {
        op.stream
            .launch_builder(&op.kernels.rms)
            .arg(&xp)
            .arg(&wp)
            .arg(&scale_stride)
            .arg(&yp)
            .arg(&bf16)
            .arg(&width_u)
            .arg(&RMS_EPS)
            .launch(ops::grid(rows as usize, 256))?;
    }
    op.tick();
    Ok(())
}

fn gather(
    op: &mut Op<'_>,
    table: &CudaSlice<f32>,
    codes: u64,
    y: &CudaSlice<f32>,
    rows: i32,
) -> Result<()> {
    fits(y, rows as usize, CODE_DIM)?;
    ensure!(table.len() == (VOCAB as usize) * CODE_DIM, "codebook table");
    let tp = dev(table, op.stream);
    let yp = dev(y, op.stream);
    let count = (rows as usize) * CODE_DIM;
    let n = u32::try_from(rows)?;
    let vocab = u32::try_from(VOCAB)?;
    let width = u32::try_from(CODE_DIM)?;
    unsafe {
        op.stream
            .launch_builder(&op.kernels.gather)
            .arg(&tp)
            .arg(&codes)
            .arg(&yp)
            .arg(&n)
            .arg(&vocab)
            .arg(&width)
            .launch(ops::flat(count))?;
    }
    op.tick();
    Ok(())
}

fn add_rows(op: &mut Op<'_>, y: &CudaSlice<f32>, x: &CudaSlice<f32>, rows: i32) -> Result<()> {
    fits(y, rows as usize, CODE_DIM)?;
    fits(x, rows as usize, CODE_DIM)?;
    let count = i64::try_from((rows as usize) * CODE_DIM)?;
    let yp = dev(y, op.stream);
    let xp = dev(x, op.stream);
    let scale = 1.0f32;
    unsafe {
        op.stream
            .launch_builder(&op.kernels.add)
            .arg(&yp)
            .arg(&xp)
            .arg(&count)
            .arg(&scale)
            .launch(ops::flat(count as usize))?;
    }
    op.tick();
    Ok(())
}

fn causal(op: &mut Op<'_>, conv: &Causal, x: u64, y: u64, rows: i32, residual: u64) -> Result<()> {
    ensure!(rows > 0, "causal rows");
    let y_blocks = u32::try_from(conv.out_dim)?.div_ceil(256);
    let wp = dev(&conv.weight, op.stream);
    let bp = dev(&conv.bias, op.stream);
    let grid = LaunchConfig {
        grid_dim: (u32::try_from(rows)?, y_blocks, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        op.stream
            .launch_builder(&op.kernels.causal)
            .arg(&x)
            .arg(&wp)
            .arg(&bp)
            .arg(&residual)
            .arg(&y)
            .arg(&rows)
            .arg(&conv.in_dim)
            .arg(&conv.out_dim)
            .arg(&conv.kernel)
            .arg(&conv.dilation)
            .arg(&conv.groups)
            .launch(grid)?;
    }
    op.tick();
    Ok(())
}

fn transpose(op: &mut Op<'_>, conv: &Trans, x: u64, y: u64, rows: i32) -> Result<()> {
    ensure!(rows > 0, "transpose rows");
    let out_rows = rows.checked_mul(conv.stride).context("transpose rows")?;
    let wp = dev(&conv.weight, op.stream);
    let bp = dev(&conv.bias, op.stream);
    let grid_x = u32::try_from(out_rows)?.clamp(1, 65535);
    let trim = i32::from(conv.trimmed);
    if conv.trimmed {
        ensure!(conv.kernel == 2 * conv.stride, "trim tap");
    } else {
        ensure!(conv.kernel == conv.stride, "transpose tap");
    }
    let pad = 0i32;
    unsafe {
        op.stream
            .launch_builder(&op.kernels.trans)
            .arg(&x)
            .arg(&wp)
            .arg(&bp)
            .arg(&y)
            .arg(&rows)
            .arg(&conv.in_dim)
            .arg(&conv.out_dim)
            .arg(&conv.kernel)
            .arg(&conv.stride)
            .arg(&pad)
            .arg(&trim)
            .launch(ops::grid(grid_x as usize, 256))?;
    }
    op.tick();
    Ok(())
}

fn rope(
    op: &mut Op<'_>,
    inv: &CudaSlice<f32>,
    cos: &CudaSlice<f32>,
    sin: &CudaSlice<f32>,
    rows: i32,
) -> Result<()> {
    fits(cos, rows as usize, HEAD_DIM as usize)?;
    fits(sin, rows as usize, HEAD_DIM as usize)?;
    let ip = dev(inv, op.stream);
    let cp = dev(cos, op.stream);
    let sp = dev(sin, op.stream);
    let n = (rows as usize) * (HEAD_DIM as usize / 2);
    unsafe {
        op.stream
            .launch_builder(&op.kernels.rope)
            .arg(&ip)
            .arg(&cp)
            .arg(&sp)
            .arg(&rows)
            .arg(&HEAD_DIM)
            .launch(ops::flat(n))?;
    }
    op.tick();
    Ok(())
}

fn rotate_half(
    op: &mut Op<'_>,
    x: &CudaSlice<f32>,
    cos: &CudaSlice<f32>,
    sin: &CudaSlice<f32>,
    rows: i32,
) -> Result<()> {
    fits(x, rows as usize, Q_WIDTH)?;
    fits(cos, rows as usize, HEAD_DIM as usize)?;
    fits(sin, rows as usize, HEAD_DIM as usize)?;
    let xp = dev(x, op.stream);
    let cp = dev(cos, op.stream);
    let sp = dev(sin, op.stream);
    let pairs = (rows as usize) * (HEADS as usize);
    unsafe {
        op.stream
            .launch_builder(&op.kernels.rotate)
            .arg(&xp)
            .arg(&cp)
            .arg(&sp)
            .arg(&rows)
            .arg(&HEADS)
            .arg(&HEAD_DIM)
            .launch(ops::flat(pairs))?;
    }
    op.tick();
    Ok(())
}

fn attend(
    op: &mut Op<'_>,
    q: &CudaSlice<f32>,
    k: &CudaSlice<f32>,
    v: &CudaSlice<f32>,
    rope: (&CudaSlice<f32>, &CudaSlice<f32>),
    y: &CudaSlice<f32>,
    rows: i32,
) -> Result<()> {
    let (cos, sin) = rope;
    fits(q, rows as usize, Q_WIDTH)?;
    fits(k, rows as usize, Q_WIDTH)?;
    fits(v, rows as usize, Q_WIDTH)?;
    fits(y, rows as usize, Q_WIDTH)?;
    rotate_half(op, q, cos, sin, rows)?;
    rotate_half(op, k, cos, sin, rows)?;
    let queries = u32::try_from(rows)?;
    let shape = [u32::try_from(HEADS)?, queries, u32::try_from(HEAD_DIM)?];
    let stride = [u64::from(HEAD_DIM as u32), u64::try_from(Q_WIDTH)?, 1];
    let qv = view(q, op.stream, 0, shape, stride)?;
    let kv = view(k, op.stream, 0, shape, stride)?;
    let vv = view(v, op.stream, 0, shape, stride)?;
    let out = view(y, op.stream, 0, shape, stride)?;
    let mask = F32Mask {
        causal: true,
        valid_keys: queries,
        query_base: 0,
        window: u32::try_from(WINDOW)?,
    };
    let launches = op.attention.issued_launches(queries, queries, &mask)?;
    op.attention
        .run_cublas(op.blas, qv, kv, vv, &mut TensorMut(out), mask)?;
    *op.count += launches;
    Ok(())
}

fn snake_act(op: &mut Op<'_>, act: &Snake, x: u64, y: u64, rows: i32) -> Result<()> {
    let count = i64::from(i32::try_from(rows as usize * act.channels as usize)?);
    let ap = dev(&act.alpha, op.stream);
    let bp = dev(&act.beta, op.stream);
    unsafe {
        op.stream
            .launch_builder(&op.kernels.snake)
            .arg(&x)
            .arg(&ap)
            .arg(&bp)
            .arg(&y)
            .arg(&count)
            .arg(&act.channels)
            .launch(ops::flat(count as usize))?;
    }
    op.tick();
    Ok(())
}

fn layernorm(
    op: &mut Op<'_>,
    x: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    y: &CudaSlice<f32>,
    rows: u32,
    width: u32,
) -> Result<()> {
    fits(x, rows as usize, width as usize)?;
    fits(y, rows as usize, width as usize)?;
    let xp = dev(x, op.stream);
    let wp = dev(weight, op.stream);
    let bp = dev(bias, op.stream);
    let yp = dev(y, op.stream);
    let one = 1u32;
    let stride = u64::from(width);
    unsafe {
        op.stream
            .launch_builder(&op.kernels.norm)
            .arg(&xp)
            .arg(&wp)
            .arg(&bp)
            .arg(&yp)
            .arg(&rows)
            .arg(&one)
            .arg(&width)
            .arg(&stride)
            .arg(&stride)
            .arg(&1u64)
            .arg(&stride)
            .arg(&stride)
            .arg(&1u64)
            .arg(&LN_EPS)
            .launch(ops::grid(rows as usize, 256))?;
    }
    op.tick();
    Ok(())
}

fn gelu(op: &mut Op<'_>, x: &CudaSlice<f32>, rows: usize, width: usize) -> Result<()> {
    fits(x, rows, width)?;
    let xp = dev(x, op.stream);
    let rows_u = u32::try_from(rows)?;
    let width_u = u32::try_from(width)?;
    let stride = u64::from(width_u);
    let mode = 0i32;
    unsafe {
        op.stream
            .launch_builder(&op.kernels.gelu)
            .arg(&xp)
            .arg(&xp)
            .arg(&rows_u)
            .arg(&1u32)
            .arg(&width_u)
            .arg(&stride)
            .arg(&stride)
            .arg(&1u64)
            .arg(&stride)
            .arg(&stride)
            .arg(&1u64)
            .arg(&mode)
            .launch(ops::flat(rows * width))?;
    }
    op.tick();
    Ok(())
}

fn silu_mul(
    op: &mut Op<'_>,
    gate: &CudaSlice<f32>,
    up: &CudaSlice<f32>,
    y: &CudaSlice<f32>,
    rows: usize,
    width: usize,
) -> Result<()> {
    fits(gate, rows, width)?;
    fits(up, rows, width)?;
    fits(y, rows, width)?;
    let count = i32::try_from(rows * width)?;
    let gp = dev(gate, op.stream);
    let up_p = dev(up, op.stream);
    let yp = dev(y, op.stream);
    let mode = 0i32;
    unsafe {
        op.stream
            .launch_builder(&op.kernels.silu)
            .arg(&gp)
            .arg(&up_p)
            .arg(&yp)
            .arg(&count)
            .arg(&mode)
            .launch(ops::grid(
                (count as usize).div_ceil(256).clamp(1, 65535),
                256,
            ))?;
    }
    op.tick();
    Ok(())
}

fn scale_residual(
    op: &mut Op<'_>,
    residual: &CudaSlice<f32>,
    branch: &CudaSlice<f32>,
    scale: &CudaSlice<f32>,
    y: &CudaSlice<f32>,
    rows: usize,
    width: usize,
) -> Result<()> {
    fits(residual, rows, width)?;
    fits(branch, rows, width)?;
    fits(y, rows, width)?;
    ensure!(scale.len() >= width, "scale width");
    let rp = dev(residual, op.stream);
    let bp = dev(branch, op.stream);
    let sp = dev(scale, op.stream);
    let yp = dev(y, op.stream);
    let rows_u = u32::try_from(rows)?;
    let width_u = u32::try_from(width)?;
    let count = rows * width;
    unsafe {
        op.stream
            .launch_builder(&op.kernels.scale)
            .arg(&rp)
            .arg(&bp)
            .arg(&sp)
            .arg(&yp)
            .arg(&rows_u)
            .arg(&width_u)
            .launch(ops::grid(count.div_ceil(256).clamp(1, 65535), 256))?;
    }
    op.tick();
    Ok(())
}

fn clamp_wave(op: &mut Op<'_>, x: &CudaSlice<f32>, count: i64) -> Result<()> {
    ensure!(count > 0 && (count as usize) <= x.len(), "clamp count");
    let xp = dev(x, op.stream);
    unsafe {
        op.stream
            .launch_builder(&op.kernels.clamp)
            .arg(&xp)
            .arg(&count)
            .arg(&-1.0f32)
            .arg(&1.0f32)
            .launch(ops::flat(count as usize))?;
    }
    op.tick();
    Ok(())
}

fn code_ptr(codes: &CudaSlice<i32>, stream: &CudaStream, book: usize) -> u64 {
    dev(codes, stream) + (book * MAX_FRAMES * 4) as u64
}

fn forward(tok: &Tokenizer, frames: i32) -> Result<u64> {
    let mut count = 0u64;
    let mut op = Op {
        stream: &tok.stream,
        blas: &tok.blas,
        kernels: &tok.kernels,
        attention: &tok.attention,
        count: &mut count,
    };
    let w = &tok.work;
    let rows = frames as usize;
    let e0 = dev(&w.e0, op.stream);
    let e1 = dev(&w.e1, op.stream);
    let s0 = dev(&w.s0, op.stream);
    let s1 = dev(&w.s1, op.stream);
    let wide = dev(&w.wide, op.stream);
    let code0 = code_ptr(&w.codes, op.stream, 0);
    gather(&mut op, &tok.books[0], code0, &w.e0, frames)?;
    causal(&mut op, &tok.quant_first, e0, s0, frames, 0)?;
    let code1 = code_ptr(&w.codes, op.stream, 1);
    gather(&mut op, &tok.books[1], code1, &w.e1, frames)?;
    for book in 2..BOOKS {
        let code = code_ptr(&w.codes, op.stream, book);
        gather(&mut op, &tok.books[book], code, &w.e0, frames)?;
        add_rows(&mut op, &w.e1, &w.e0, frames)?;
    }
    causal(&mut op, &tok.quant_rest, e1, s1, frames, s0)?;
    causal(&mut op, &tok.pre_conv, s1, wide, frames, 0)?;
    rope(&mut op, &tok.inv, &w.cos, &w.sin, frames)?;
    gemm_bias(
        &mut op,
        &tok.input_w,
        &tok.input_b,
        &w.wide,
        &w.h0,
        rows,
        (HIDDEN, LATENT),
    )?;
    for layer in &tok.layers {
        rms(&mut op, &w.h0, &layer.rms1, &w.norm, frames, HIDDEN as i32)?;
        gemm(&mut op, &layer.q, &w.norm, &w.q, rows, Q_WIDTH, HIDDEN)?;
        gemm(&mut op, &layer.k, &w.norm, &w.k, rows, Q_WIDTH, HIDDEN)?;
        gemm(&mut op, &layer.v, &w.norm, &w.v, rows, Q_WIDTH, HIDDEN)?;
        attend(&mut op, &w.q, &w.k, &w.v, (&w.cos, &w.sin), &w.attn, frames)?;
        gemm(&mut op, &layer.o, &w.attn, &w.proj, rows, HIDDEN, Q_WIDTH)?;
        scale_residual(
            &mut op,
            &w.h0,
            &w.proj,
            &layer.scale_attn,
            &w.h1,
            rows,
            HIDDEN,
        )?;
        rms(&mut op, &w.h1, &layer.rms2, &w.norm, frames, HIDDEN as i32)?;
        gemm(&mut op, &layer.gate, &w.norm, &w.gate, rows, MLP, HIDDEN)?;
        gemm(&mut op, &layer.up, &w.norm, &w.upw, rows, MLP, HIDDEN)?;
        silu_mul(&mut op, &w.gate, &w.upw, &w.attn, rows, MLP)?;
        gemm(&mut op, &layer.down, &w.attn, &w.proj, rows, HIDDEN, MLP)?;
        scale_residual(
            &mut op,
            &w.h1,
            &w.proj,
            &layer.scale_mlp,
            &w.h0,
            rows,
            HIDDEN,
        )?;
    }
    rms(
        &mut op,
        &w.h0,
        &tok.final_rms,
        &w.norm,
        frames,
        HIDDEN as i32,
    )?;
    gemm_bias(
        &mut op,
        &tok.output_w,
        &tok.output_b,
        &w.norm,
        &w.wide,
        rows,
        (LATENT, HIDDEN),
    )?;
    let mut cur = frames;
    let wide = dev(&w.wide, &tok.stream);
    let a = dev(&w.a, &tok.stream);
    transpose(&mut op, &tok.ups[0], wide, a, cur)?;
    cur = cur.checked_mul(2).context("upsample rows")?;
    convnext(&mut op, &tok.nexts[0], w, cur)?;
    let c = dev(&w.c, &tok.stream);
    let a = dev(&w.a, &tok.stream);
    transpose(&mut op, &tok.ups[1], c, a, cur)?;
    cur = cur.checked_mul(2).context("upsample rows")?;
    convnext(&mut op, &tok.nexts[1], w, cur)?;
    let c = dev(&w.c, &tok.stream);
    let a = dev(&w.a, &tok.stream);
    causal(&mut op, &tok.dec0, c, a, cur, 0)?;
    for block in &tok.blocks {
        let out_rows = cur.checked_mul(block.stride).context("decoder rows")?;
        let a = dev(&w.a, &tok.stream);
        let b = dev(&w.b, &tok.stream);
        let c = dev(&w.c, &tok.stream);
        snake_act(&mut op, &block.act, a, b, cur)?;
        transpose(&mut op, &block.up, b, c, cur)?;
        unit(&mut op, &block.units[0], c, b, a, out_rows)?;
        unit(&mut op, &block.units[1], a, b, c, out_rows)?;
        unit(&mut op, &block.units[2], c, b, a, out_rows)?;
        cur = out_rows;
    }
    ensure!(
        cur as usize == (frames as usize) * FRAME_SAMPLES,
        "waveform length"
    );
    let a = dev(&w.a, &tok.stream);
    let b = dev(&w.b, &tok.stream);
    snake_act(&mut op, &tok.tail_act, a, b, cur)?;
    causal(&mut op, &tok.tail_conv, b, a, cur, 0)?;
    clamp_wave(&mut op, &w.a, i64::from(cur))?;
    Ok(count)
}

fn convnext(op: &mut Op<'_>, block: &ConvNext, w: &Work, rows: i32) -> Result<()> {
    let rows_u = rows as usize;
    let a = dev(&w.a, op.stream);
    let b = dev(&w.b, op.stream);
    causal(op, &block.dw, a, b, rows, 0)?;
    layernorm(
        op,
        &w.b,
        &block.norm_w,
        &block.norm_b,
        &w.c,
        rows as u32,
        LATENT as u32,
    )?;
    gemm_bias(
        op,
        &block.pw1,
        &block.pw1_b,
        &w.c,
        &w.d,
        rows_u,
        (LATENT * 4, LATENT),
    )?;
    gelu(op, &w.d, rows_u, LATENT * 4)?;
    gemm_bias(
        op,
        &block.pw2,
        &block.pw2_b,
        &w.d,
        &w.b,
        rows_u,
        (LATENT, LATENT * 4),
    )?;
    scale_residual(op, &w.a, &w.b, &block.gamma, &w.c, rows_u, LATENT)?;
    Ok(())
}

fn unit(
    op: &mut Op<'_>,
    unit: &Unit,
    input: u64,
    scratch: u64,
    output: u64,
    rows: i32,
) -> Result<()> {
    snake_act(op, &unit.act1, input, scratch, rows)?;
    causal(op, &unit.conv7, scratch, output, rows, 0)?;
    snake_act(op, &unit.act2, output, scratch, rows)?;
    causal(op, &unit.conv1, scratch, output, rows, input)?;
    Ok(())
}

#[cfg(test)]
mod gate {
    use super::{BOOKS, CODE_MAX, FRAME_SAMPLES, MAX_FRAMES, check_codes, slide_keeps};
    use anyhow::{Context, Result, bail, ensure};
    use safetensors::{Dtype, SafeTensors};
    use serde_json::Value;
    use std::path::PathBuf;

    const SHORT_MAX: f64 = f64::from_bits(0x3f5220e000000000);
    const SHORT_RMSE: f64 = f64::from_bits(0x3f2532090a48e54f);
    const SYN_MAX: f64 = f64::from_bits(0x3f88683c00000000);
    const SYN_RMSE: f64 = f64::from_bits(0x3f3cc9433704f29f);
    const REAL_MAX: f64 = f64::from_bits(0x3f447a6000000000);
    const REAL_RMSE: f64 = f64::from_bits(0x3f045a02280fdf2d);

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum GateStatus {
        Pass,
        Defect,
        ProtocolFail,
    }

    impl GateStatus {
        fn label(self) -> &'static str {
            match self {
                Self::Pass => "PASS",
                Self::Defect => "DEFECT",
                Self::ProtocolFail => "PROTOCOL_FAIL",
            }
        }
    }

    fn difference(actual: &[f32], expected: &[f32]) -> Result<(f64, f64)> {
        ensure!(!actual.is_empty(), "Runner FAIL empty tensor");
        ensure!(actual.len() == expected.len(), "Runner FAIL element count");
        ensure!(
            actual.iter().all(|value| value.is_finite()),
            "Runner FAIL non-finite candidate"
        );
        ensure!(
            expected.iter().all(|value| value.is_finite()),
            "Runner FAIL non-finite reference"
        );
        let mut max_abs = 0.0f64;
        let mut squares = 0.0f64;
        for (&got, &want) in actual.iter().zip(expected) {
            let error = (f64::from(got) - f64::from(want)).abs();
            max_abs = max_abs.max(error);
            squares += error * error;
        }
        let n = u32::try_from(actual.len()).context("Runner FAIL element count")?;
        let rmse = (squares / f64::from(n)).sqrt();
        ensure!(
            max_abs.is_finite() && rmse.is_finite(),
            "Runner FAIL non-finite metric"
        );
        Ok((max_abs, rmse))
    }

    fn emit(
        name: &str,
        plant: &str,
        detect: &str,
        max_abs: f64,
        rmse: f64,
        tau: (f64, f64),
        status: GateStatus,
    ) {
        let (tau_max, tau_rmse) = tau;
        let bar_max = 10.0 * tau_max;
        let bar_rmse = 10.0 * tau_rmse;
        println!(
            "METRIC tensor={name} plant={plant} bar_source=cells.{name}.tau max_abs={max_abs:?} rmse={rmse:?} tau_max_abs={tau_max:?} tau_rmse={tau_rmse:?} bar_max_abs={bar_max:?} bar_rmse={bar_rmse:?} detect={detect} status={}",
            status.label()
        );
    }

    fn score_nominal(
        name: &str,
        actual: &[f32],
        expected: &[f32],
        tau_max: f64,
        tau_rmse: f64,
    ) -> Result<GateStatus> {
        let (max_abs, rmse) = difference(actual, expected)?;
        let status = if max_abs <= tau_max && rmse <= tau_rmse {
            GateStatus::Pass
        } else {
            GateStatus::Defect
        };
        emit(
            name,
            "nominal",
            "nominal",
            max_abs,
            rmse,
            (tau_max, tau_rmse),
            status,
        );
        Ok(status)
    }

    fn score_control(
        name: &str,
        actual: &[f32],
        nominal: &[f32],
        tau_max: f64,
        tau_rmse: f64,
    ) -> Result<GateStatus> {
        let (max_abs, rmse) = difference(actual, nominal)?;
        let status = if max_abs >= 10.0 * tau_max && rmse >= 10.0 * tau_rmse {
            GateStatus::Pass
        } else {
            GateStatus::ProtocolFail
        };
        emit(
            name,
            "control",
            "both",
            max_abs,
            rmse,
            (tau_max, tau_rmse),
            status,
        );
        Ok(status)
    }

    fn score_causal(
        name: &str,
        actual: &[f32],
        nominal: &[f32],
        tau_max: f64,
        tau_rmse: f64,
    ) -> Result<GateStatus> {
        ensure!(
            actual.len() >= FRAME_SAMPLES && nominal.len() >= FRAME_SAMPLES,
            "Runner FAIL causal length"
        );
        let (max_abs, rmse) = difference(&actual[..FRAME_SAMPLES], &nominal[..FRAME_SAMPLES])?;
        let status = if max_abs <= tau_max && rmse <= tau_rmse {
            GateStatus::Pass
        } else {
            GateStatus::ProtocolFail
        };
        emit(
            name,
            "causal",
            "inside",
            max_abs,
            rmse,
            (tau_max, tau_rmse),
            status,
        );
        Ok(status)
    }

    fn record(status: GateStatus, defect: &mut bool, protocol: &mut bool) {
        match status {
            GateStatus::Pass => {}
            GateStatus::Defect => *defect = true,
            GateStatus::ProtocolFail => *protocol = true,
        }
    }

    fn finish_gate(defect: bool, protocol: bool) -> Result<()> {
        match (defect, protocol) {
            (false, false) => Ok(()),
            (true, false) => bail!("candidate DEFECT"),
            (false, true) => bail!("protocol FAIL"),
            (true, true) => bail!("candidate DEFECT\nprotocol FAIL"),
        }
    }

    fn same_bytes(left: &[f32], right: &[f32]) -> bool {
        left.len() == right.len()
            && left
                .iter()
                .zip(right)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }

    fn env_file(name: &str) -> Result<PathBuf> {
        let value = std::env::var(name).with_context(|| format!("missing {name}"))?;
        let path = PathBuf::from(value);
        ensure!(path.is_file(), "{name} is not a file");
        Ok(path)
    }

    fn lock_cell(
        floors: &Value,
        name: &str,
        samples: u64,
        frames: u64,
        max_bits: u64,
        rmse_bits: u64,
    ) -> Result<(f64, f64)> {
        ensure!(
            floors["cells"][name]["samples"].as_u64() == Some(samples),
            "{name} samples"
        );
        ensure!(
            floors["cells"][name]["frames"].as_u64() == Some(frames),
            "{name} frames"
        );
        let max_abs = floors["cells"][name]["tau"]["max_abs"]
            .as_f64()
            .with_context(|| format!("missing {name} max_abs"))?;
        let rmse = floors["cells"][name]["tau"]["rmse"]
            .as_f64()
            .with_context(|| format!("missing {name} rmse"))?;
        ensure!(max_abs.to_bits() == max_bits, "{name} max_abs bits");
        ensure!(rmse.to_bits() == rmse_bits, "{name} rmse bits");
        Ok((max_abs, rmse))
    }

    fn host_f32(path: &std::path::Path, key: &str, len: usize) -> Result<Vec<f32>> {
        let bytes = std::fs::read(path).context("reference wav")?;
        let st = SafeTensors::deserialize(&bytes).context("reference wav")?;
        let view = st.tensor(key).with_context(|| format!("missing {key}"))?;
        ensure!(view.dtype() == Dtype::F32, "{key} dtype");
        ensure!(view.shape() == [len], "{key} shape");
        let (chunks, rest) = view.data().as_chunks::<4>();
        ensure!(rest.is_empty() && chunks.len() == len, "{key} bytes");
        let mut values = Vec::with_capacity(len);
        for chunk in chunks {
            let value = f32::from_le_bytes(*chunk);
            ensure!(value.is_finite(), "Runner FAIL non-finite reference {key}");
            values.push(value);
        }
        Ok(values)
    }

    fn host_codes(path: &std::path::Path, frames: usize) -> Result<Vec<i32>> {
        let bytes = std::fs::read(path).context("codes")?;
        let st = SafeTensors::deserialize(&bytes).context("codes")?;
        let view = st.tensor("stored_codes").context("missing stored_codes")?;
        ensure!(view.dtype() == Dtype::I32, "stored_codes dtype");
        ensure!(view.shape() == [frames, BOOKS], "stored_codes shape");
        let (chunks, rest) = view.data().as_chunks::<4>();
        ensure!(
            rest.is_empty() && chunks.len() == frames * BOOKS,
            "stored_codes bytes"
        );
        let codes: Vec<i32> = chunks
            .iter()
            .map(|chunk| i32::from_le_bytes(*chunk))
            .collect();
        check_codes(&codes)?;
        Ok(codes)
    }

    fn synthetic(frames: usize) -> Vec<i32> {
        let mut codes = Vec::with_capacity(frames * BOOKS);
        for frame in 0..frames {
            for book in 0..BOOKS {
                let value = (frame * 17 + book * 13 + 3) % 2047;
                codes.push(i32::try_from(value).expect("synthetic code"));
            }
        }
        codes
    }

    fn plant(codes: &[i32], frame: usize) -> Result<Vec<i32>> {
        let mut out = codes.to_vec();
        let index = frame * BOOKS;
        let next = out[index].checked_add(1).context("plant code")?;
        ensure!(next <= CODE_MAX, "plant code");
        out[index] = next;
        Ok(out)
    }

    struct Decoded {
        samples: Vec<f32>,
        launches: u64,
    }

    fn decoded(tok: &mut super::Tokenizer, codes: &[i32], expect: usize) -> Result<Decoded> {
        let samples = tok.decode(codes)?;
        ensure!(samples.len() == expect, "Runner FAIL waveform length");
        Ok(Decoded {
            samples,
            launches: tok.launches(),
        })
    }

    #[test]
    fn codes_reject_out_of_range_and_bad_length() -> Result<()> {
        let mut row = vec![0i32; BOOKS];
        check_codes(&row)?;
        row[0] = CODE_MAX;
        check_codes(&row)?;
        row[3] = CODE_MAX + 1;
        ensure!(check_codes(&row).is_err(), "code 2048");
        row[3] = -1;
        ensure!(check_codes(&row).is_err(), "code -1");
        ensure!(check_codes(&[]).is_err(), "empty codes");
        ensure!(check_codes(&[0i32; BOOKS - 1]).is_err(), "short codes");
        let cap = vec![0i32; MAX_FRAMES * BOOKS];
        ensure!(check_codes(&cap)? == MAX_FRAMES, "max frames");
        let over = vec![0i32; (MAX_FRAMES + 1) * BOOKS];
        ensure!(check_codes(&over).is_err(), "frames above cap");
        Ok(())
    }

    #[test]
    fn sliding_window_keeps_the_causal_band() {
        let window = 72;
        assert!(slide_keeps(71, 0, window));
        assert!(!slide_keeps(72, 0, window));
        assert!(!slide_keeps(79, 0, window));
        assert!(slide_keeps(79, 8, window));
        assert!(!slide_keeps(79, 7, window));
        assert!(slide_keeps(3, 0, window));
        assert!(!slide_keeps(5, 6, window));
    }

    #[test]
    fn nominal_defect_and_causal_miss_use_separate_statuses() -> Result<()> {
        let got = [1.0f32];
        let zero = [0.0f32];
        let mut defect = false;
        let mut protocol = false;
        record(
            score_nominal("probe", &got, &zero, 0.0, 0.0)?,
            &mut defect,
            &mut protocol,
        );
        match finish_gate(defect, protocol) {
            Ok(()) => bail!("nominal defect"),
            Err(err) => ensure!(err.to_string() == "candidate DEFECT", "candidate DEFECT"),
        }
        defect = false;
        protocol = false;
        record(
            score_control("probe", &got, &zero, 1.0, 1.0)?,
            &mut defect,
            &mut protocol,
        );
        match finish_gate(defect, protocol) {
            Ok(()) => bail!("control miss"),
            Err(err) => ensure!(err.to_string() == "protocol FAIL", "protocol FAIL"),
        }
        let inside = [0.0f32; FRAME_SAMPLES];
        let mut outside = inside;
        outside[0] = 1.0;
        defect = false;
        protocol = false;
        record(
            score_causal("probe", &outside, &inside, 1e-6, 1e-7)?,
            &mut defect,
            &mut protocol,
        );
        ensure!(!defect && protocol, "causal miss status");
        match finish_gate(defect, protocol) {
            Ok(()) => bail!("causal miss"),
            Err(err) => ensure!(err.to_string() == "protocol FAIL", "protocol FAIL"),
        }
        defect = false;
        protocol = false;
        record(
            score_causal("probe", &inside, &inside, 1e-6, 1e-7)?,
            &mut defect,
            &mut protocol,
        );
        finish_gate(defect, protocol)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires GPU lease, frozen floors, and external fixtures"]
    fn n8_m2_candidate() -> Result<()> {
        let floors_path = env_file("N8_M2_FLOORS")?;
        let weights = env_file("N8_M2_WEIGHTS")?;
        let wavs = env_file("N8_M2_REF_WAV")?;
        let short_path = env_file("N8_M2_SHORT_CODES")?;
        let real_path = env_file("N8_M2_REAL_CODES")?;
        let clone_a = env_file("N8_M2_CLONE_A")?;
        let clone_b = env_file("N8_M2_CLONE_B")?;
        let clone_report = env_file("N8_M2_CLONE_REPORT")?;
        let floors: Value = serde_json::from_slice(&std::fs::read(&floors_path).context("floors")?)
            .context("floors")?;
        ensure!(floors["schema"].as_str() == Some("n8-m2-wav-v1"), "schema");
        ensure!(
            floors["candidate_run"].as_bool() == Some(false),
            "candidate_run"
        );
        let (short_max, short_rmse) = lock_cell(
            &floors,
            "short",
            7680,
            4,
            SHORT_MAX.to_bits(),
            SHORT_RMSE.to_bits(),
        )?;
        let (syn_max, syn_rmse) = lock_cell(
            &floors,
            "syn",
            153600,
            80,
            SYN_MAX.to_bits(),
            SYN_RMSE.to_bits(),
        )?;
        let (real_max, real_rmse) = lock_cell(
            &floors,
            "real",
            153600,
            80,
            REAL_MAX.to_bits(),
            REAL_RMSE.to_bits(),
        )?;
        let report: Value =
            serde_json::from_slice(&std::fs::read(&clone_report).context("clone report")?)
                .context("clone report")?;
        ensure!(
            report["schema"].as_str() == Some("n8-m2-clone-v1"),
            "Runner FAIL clone schema"
        );
        ensure!(
            report["candidate_run"].as_bool() == Some(false),
            "Runner FAIL clone candidate_run"
        );
        ensure!(
            report["encoder_calls"].as_u64() == Some(2),
            "Runner FAIL encoder_calls"
        );
        ensure!(
            report["codes_differ"].as_bool() == Some(true),
            "Runner FAIL codes_differ"
        );
        let clone_frames = usize::try_from(
            report["frames"]
                .as_u64()
                .context("Runner FAIL clone frames")?,
        )?;
        ensure!(
            (1..=MAX_FRAMES).contains(&clone_frames),
            "Runner FAIL clone frames"
        );
        ensure!(
            report["samples"].as_u64() == Some((clone_frames * FRAME_SAMPLES) as u64),
            "Runner FAIL clone samples"
        );
        let short_codes = host_codes(&short_path, 4)?;
        let real_codes = host_codes(&real_path, 80)?;
        let syn_codes = synthetic(80);
        check_codes(&syn_codes)?;
        let clone_left = host_codes(&clone_a, clone_frames)?;
        let clone_right = host_codes(&clone_b, clone_frames)?;
        ensure!(clone_left != clone_right, "Runner FAIL clone codes");
        let short_ref = host_f32(&wavs, "short_nominal", 7680)?;
        let syn_ref = host_f32(&wavs, "syn_nominal", 153600)?;
        let real_ref = host_f32(&wavs, "real_nominal", 153600)?;
        let device = super::super::Device::new(0)?;
        let mut tok = super::Tokenizer::load(&weights, &device)?;
        let short_n = decoded(&mut tok, &short_codes, 7680)?;
        let short_c = decoded(&mut tok, &plant(&short_codes, 0)?, 7680)?;
        let short_k = decoded(&mut tok, &plant(&short_codes, 3)?, 7680)?;
        let syn_n = decoded(&mut tok, &syn_codes, 153600)?;
        let syn_c = decoded(&mut tok, &plant(&syn_codes, 0)?, 153600)?;
        let syn_k = decoded(&mut tok, &plant(&syn_codes, 79)?, 153600)?;
        let real_n = decoded(&mut tok, &real_codes, 153600)?;
        let real_c = decoded(&mut tok, &plant(&real_codes, 0)?, 153600)?;
        let real_k = decoded(&mut tok, &plant(&real_codes, 79)?, 153600)?;
        let clone_samples = clone_frames * FRAME_SAMPLES;
        let left = decoded(&mut tok, &clone_left, clone_samples)?;
        let right = decoded(&mut tok, &clone_right, clone_samples)?;
        let launches = real_n.launches;
        for count in [
            short_n.launches,
            short_c.launches,
            short_k.launches,
            syn_n.launches,
            syn_c.launches,
            syn_k.launches,
            real_n.launches,
            real_c.launches,
            real_k.launches,
            left.launches,
            right.launches,
        ] {
            ensure!(count == launches, "Runner FAIL launch count");
        }
        println!("weight_bytes={}", tok.weight_bytes());
        println!("launches={launches}");
        println!("graphs=0");
        println!("host_download_bytes={}", real_n.samples.len() * 4);
        println!("mask=sliding_window_causal window=72");
        let mut defect = false;
        let mut protocol = false;
        record(
            score_nominal("short", &short_n.samples, &short_ref, short_max, short_rmse)?,
            &mut defect,
            &mut protocol,
        );
        record(
            score_control("short", &short_c.samples, &short_ref, short_max, short_rmse)?,
            &mut defect,
            &mut protocol,
        );
        record(
            score_causal("short", &short_k.samples, &short_ref, short_max, short_rmse)?,
            &mut defect,
            &mut protocol,
        );
        record(
            score_nominal("syn", &syn_n.samples, &syn_ref, syn_max, syn_rmse)?,
            &mut defect,
            &mut protocol,
        );
        record(
            score_control("syn", &syn_c.samples, &syn_ref, syn_max, syn_rmse)?,
            &mut defect,
            &mut protocol,
        );
        record(
            score_causal("syn", &syn_k.samples, &syn_ref, syn_max, syn_rmse)?,
            &mut defect,
            &mut protocol,
        );
        record(
            score_nominal("real", &real_n.samples, &real_ref, real_max, real_rmse)?,
            &mut defect,
            &mut protocol,
        );
        record(
            score_control("real", &real_c.samples, &real_ref, real_max, real_rmse)?,
            &mut defect,
            &mut protocol,
        );
        record(
            score_causal("real", &real_k.samples, &real_ref, real_max, real_rmse)?,
            &mut defect,
            &mut protocol,
        );
        let texts_equal = same_bytes(&short_n.samples, &real_n.samples);
        if texts_equal {
            protocol = true;
        }
        println!("TWO_TEXT equal={texts_equal}");
        let clone_equal = same_bytes(&left.samples, &right.samples);
        if clone_equal || left.samples.len() != right.samples.len() {
            protocol = true;
        }
        println!("CLONE frames={clone_frames} bytes_equal={clone_equal} encoder_calls=2");
        listen("short.wav", &short_n.samples)?;
        listen("real.wav", &real_n.samples)?;
        listen("clone-a.wav", &left.samples)?;
        listen("clone-b.wav", &right.samples)?;
        finish_gate(defect, protocol)
    }

    fn listen(name: &str, samples: &[f32]) -> Result<()> {
        let Ok(dir) = std::env::var("N8_M2_LISTEN_DIR") else {
            return Ok(());
        };
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).context("N8_M2_LISTEN_DIR")?;
        let data_bytes = u32::try_from(samples.len().checked_mul(2).context("listen length")?)
            .context("listen length")?;
        let mut data = Vec::with_capacity(44 + data_bytes as usize);
        let rate = 24_000u32;
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        data.extend_from_slice(b"WAVEfmt ");
        data.extend_from_slice(&16u32.to_le_bytes());
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&rate.to_le_bytes());
        data.extend_from_slice(&(rate * 2).to_le_bytes());
        data.extend_from_slice(&2u16.to_le_bytes());
        data.extend_from_slice(&16u16.to_le_bytes());
        data.extend_from_slice(b"data");
        data.extend_from_slice(&data_bytes.to_le_bytes());
        for sample in samples {
            let pcm = (f64::from(sample.clamp(-1.0, 1.0)) * 32767.0).round() as i32;
            data.extend_from_slice(&(pcm.clamp(-32767, 32767) as i16).to_le_bytes());
        }
        std::fs::write(dir.join(name), data).with_context(|| format!("listen {name}"))?;
        println!("LISTEN file={name} samples={}", samples.len());
        Ok(())
    }
}
