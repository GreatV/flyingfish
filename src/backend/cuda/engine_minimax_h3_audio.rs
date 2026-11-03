//! MiniMax-H3 audio VAE decoder: dec_in_proj and the BigVGAN decoder, F32, on time-major rows `[T, C]` per stereo
//! channel.
use super::{Source, pointer};
use crate::backend::cuda::{
    Device,
    blas::Blas,
    contracts::{TensorMut, view},
    cubin,
    ops::flat,
};
use crate::backend::gpu::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use crate::checkpoint::TensorManifest;
use crate::minimax_h3::{AUDIO_RESAMPLE_RATIO, AUDIO_RESAMPLE_TAPS, AudioVaeSpec};
use anyhow::{Context, Result, ensure};
use std::{path::Path, sync::Arc};

/// A 1D convolution `[out, input, kernel]` at `dilation` with zero padding `dilation * (kernel - 1) / 2`; the weight
/// is the published one (weight-norm folded at load when the module is weight-normed).
struct Conv {
    weight: CudaSlice<f32>,
    bias: Option<CudaSlice<f32>>,
    out: usize,
    input: usize,
    kernel: usize,
    dilation: usize,
}

/// A weight-normed transposed 1D convolution, folded weight `[input, out * kernel]`.
struct ConvTranspose {
    weight: CudaSlice<f32>,
    bias: CudaSlice<f32>,
    input: usize,
    out: usize,
    kernel: usize,
    stride: usize,
}

/// Alias-free SnakeBeta: log-space `alpha`, `beta` `[channels]` and the stored up / down Kaiser filters.
struct Activation {
    alpha: CudaSlice<f32>,
    beta: CudaSlice<f32>,
    up: CudaSlice<f32>,
    down: CudaSlice<f32>,
}

/// `AMPBlock1`: per dilation a (dilated conv, dilation-1 conv) pair, each conv preceded by its own activation.
struct AmpBlock {
    convs: Vec<(Conv, Conv)>,
    activations: Vec<(Activation, Activation)>,
}

struct Level {
    up: ConvTranspose,
    blocks: Vec<AmpBlock>,
}

struct AudioKernels {
    fold: CudaFunction,
    pad: CudaFunction,
    im2col: CudaFunction,
    transpose: CudaFunction,
    conv_transpose: CudaFunction,
    snake: CudaFunction,
    up: CudaFunction,
    down: CudaFunction,
    add_scale: CudaFunction,
    clamp: CudaFunction,
    bias_residual: CudaFunction,
}

/// Loads published F32 audio decoder tensors; weight-normed weights are folded `g * v / ||v||` per dim-0 row into
/// device memory once.
struct Loader<'a, 'b> {
    s: &'b Arc<CudaStream>,
    source: Source<'a>,
    fold: &'b CudaFunction,
    bytes: usize,
}

impl Loader<'_, '_> {
    fn f32(&mut self, name: &str, elements: usize) -> Result<CudaSlice<f32>> {
        self.bytes += 4 * elements;
        self.source.tensor::<f32>(self.s, name, elements)
    }

    /// `base.weight_v` `[rows, rest]` folded with `base.weight_g` `[rows]`.
    fn normed(&mut self, base: &str, rows: usize, rest: usize) -> Result<CudaSlice<f32>> {
        let g = self.f32(&format!("{base}.weight_g"), rows)?;
        let v = self.f32(&format!("{base}.weight_v"), rows * rest)?;
        let out = self.s.alloc_zeros::<f32>(rows * rest)?;
        unsafe {
            self.s
                .launch_builder(self.fold)
                .arg(&pointer(&g, self.s))
                .arg(&pointer(&v, self.s))
                .arg(&pointer(&out, self.s))
                .arg(&(rows as i64))
                .arg(&(rest as i64))
                .launch(LaunchConfig {
                    grid_dim: (u32::try_from(rows)?, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(out)
    }

    fn conv(
        &mut self,
        base: &str,
        (out, input, kernel, dilation): (usize, usize, usize, usize),
        bias: bool,
    ) -> Result<Conv> {
        Ok(Conv {
            weight: self.normed(base, out, input * kernel)?,
            bias: if bias {
                Some(self.f32(&format!("{base}.bias"), out)?)
            } else {
                None
            },
            out,
            input,
            kernel,
            dilation,
        })
    }

    fn activation(&mut self, base: &str, channels: usize) -> Result<Activation> {
        Ok(Activation {
            alpha: self.f32(&format!("{base}.act.alpha"), channels)?,
            beta: self.f32(&format!("{base}.act.beta"), channels)?,
            up: self.f32(&format!("{base}.upsample.filter"), AUDIO_RESAMPLE_TAPS)?,
            down: self.f32(
                &format!("{base}.downsample.lowpass.filter"),
                AUDIO_RESAMPLE_TAPS,
            )?,
        })
    }
}

/// The audio VAE decoder side, F32 and resident.
pub(crate) struct AudioDecoder {
    spec: AudioVaeSpec,
    in_proj: Conv,
    conv_pre: Conv,
    levels: Vec<Level>,
    post: Activation,
    conv_post: Conv,
    kernels: AudioKernels,
    /// Activations resample around SnakeBeta (the definition); off only for the gate's control.
    alias_free: bool,
}

/// Per-channel scratch rows of one decode, each sized for the widest `[T, C]` (or im2col) extent of the decoder: `a`
/// holds a level's input, `b` / `c` an AMP block's running state, `t` a dilated conv's output, `acc` the block sum.
struct AudioWork {
    a: CudaSlice<f32>,
    b: CudaSlice<f32>,
    c: CudaSlice<f32>,
    t: CudaSlice<f32>,
    acc: CudaSlice<f32>,
    act: CudaSlice<f32>,
    up: CudaSlice<f32>,
    padded: CudaSlice<f32>,
    columns: CudaSlice<f32>,
}

impl AudioDecoder {
    pub(crate) fn new(
        d: &Device,
        spec: &AudioVaeSpec,
        audio_dir: &Path,
        manifest: &TensorManifest,
    ) -> Result<Self> {
        let s = &d.stream;
        let (free, total) = d.ctx.mem_get_info()?;
        let q38 = cubin::module(s.context(), "q38_common")?;
        let vision = cubin::module(s.context(), "vision_ops")?;
        let kernels = AudioKernels {
            fold: q38.load_function("weight_norm_fold_f32")?,
            pad: q38.load_function("pad_rows_n_f32")?,
            im2col: q38.load_function("conv_im2col_f32")?,
            transpose: q38.load_function("transpose_f32")?,
            conv_transpose: q38.load_function("conv_transpose_rows_f32")?,
            snake: q38.load_function("snake_beta_rows_f32")?,
            up: q38.load_function("fir_upsample_rows_f32")?,
            down: q38.load_function("fir_downsample_rows_f32")?,
            add_scale: q38.load_function("add_scale_f32")?,
            clamp: q38.load_function("clamp_f32")?,
            bias_residual: vision.load_function("bias_residual_f32")?,
        };
        let mut load = Loader {
            s,
            source: Source::new(audio_dir, manifest),
            fold: &kernels.fold,
            bytes: 0,
        };
        let (latent, lc) = (spec.latent_dim, spec.latent_channels);
        let in_proj = Conv {
            weight: load.f32("dec_in_proj.weight", latent * lc)?,
            bias: Some(load.f32("dec_in_proj.bias", latent)?),
            out: latent,
            input: lc,
            kernel: 1,
            dilation: 1,
        };
        let conv_pre = load.conv("decoder.conv_pre", (spec.decoder_dim, latent, 7, 1), true)?;
        let blocks = spec.resblock_kernels.len();
        let mut levels = Vec::with_capacity(spec.decoder_rates.len());
        let mut channels = spec.decoder_dim;
        for (i, (&rate, &kernel)) in spec
            .decoder_rates
            .iter()
            .zip(&spec.decoder_kernels)
            .enumerate()
        {
            let half = channels / 2;
            let base = format!("decoder.ups.{i}.0");
            let up = ConvTranspose {
                weight: load.normed(&base, channels, half * kernel)?,
                bias: load.f32(&format!("{base}.bias"), half)?,
                input: channels,
                out: half,
                kernel,
                stride: rate,
            };
            channels = half;
            let mut amp = Vec::with_capacity(blocks);
            for (j, (&k, dilations)) in spec
                .resblock_kernels
                .iter()
                .zip(&spec.resblock_dilations)
                .enumerate()
            {
                let block = format!("decoder.resblocks.{}", i * blocks + j);
                let mut convs = Vec::with_capacity(dilations.len());
                let mut activations = Vec::with_capacity(dilations.len());
                for (m, &dilation) in dilations.iter().enumerate() {
                    let geometry = (channels, channels, k, dilation);
                    convs.push((
                        load.conv(&format!("{block}.convs1.{m}"), geometry, true)?,
                        load.conv(
                            &format!("{block}.convs2.{m}"),
                            (channels, channels, k, 1),
                            true,
                        )?,
                    ));
                    activations.push((
                        load.activation(&format!("{block}.activations.{}", 2 * m), channels)?,
                        load.activation(&format!("{block}.activations.{}", 2 * m + 1), channels)?,
                    ));
                }
                amp.push(AmpBlock { convs, activations });
            }
            levels.push(Level { up, blocks: amp });
        }
        let post = load.activation("decoder.activation_post", channels)?;
        let conv_post = load.conv("decoder.conv_post", (1, channels, 7, 1), false)?;
        let bytes = load.bytes;
        drop(load);
        s.synchronize()?;
        eprintln!(
            "{}",
            serde_json::json!({"h3_audio_decoder_placement": {"mode": "resident", "published_bytes": bytes,
                "free_before_bytes": free, "free_after_bytes": d.ctx.mem_get_info()?.0, "total_bytes": total}})
        );
        Ok(Self {
            spec: spec.clone(),
            in_proj,
            conv_pre,
            levels,
            post,
            conv_post,
            kernels,
            alias_free: true,
        })
    }

    /// Turns the activations' up / down resampling off (SnakeBeta alone), the gate's control.
    #[cfg(test)]
    pub(crate) fn without_resampling(mut self) -> Self {
        self.alias_free = false;
        self
    }

    /// Row extents `[T, C]` of every activation and the im2col width of every convolution for `frames` latent frames.
    fn extents(&self, frames: usize) -> (usize, usize, usize) {
        let mut rows = frames;
        let mut widest = rows * self.in_proj.out.max(self.in_proj.input);
        let mut columns = rows * self.conv_pre.input * self.conv_pre.kernel;
        let mut padded = (rows + self.conv_pre.kernel) * self.conv_pre.input;
        for level in &self.levels {
            rows *= level.up.stride;
            widest = widest.max(rows * level.up.out);
            for block in &level.blocks {
                for (c1, c2) in &block.convs {
                    for c in [c1, c2] {
                        columns = columns.max(rows * c.input * c.kernel);
                        padded = padded.max((rows + c.dilation * (c.kernel - 1)) * c.input);
                    }
                }
            }
        }
        let c = &self.conv_post;
        columns = columns.max(rows * c.input * c.kernel);
        padded = padded.max((rows + c.kernel) * c.input);
        (widest, columns, padded)
    }

    fn work(&self, s: &Arc<CudaStream>, frames: usize) -> Result<AudioWork> {
        let (widest, columns, padded) = self.extents(frames);
        let buffer = |n: usize| s.alloc_zeros::<f32>(n);
        Ok(AudioWork {
            a: buffer(widest)?,
            b: buffer(widest)?,
            c: buffer(widest)?,
            t: buffer(widest)?,
            acc: buffer(widest)?,
            act: buffer(widest)?,
            up: buffer(AUDIO_RESAMPLE_RATIO * widest)?,
            padded: buffer(padded)?,
            columns: buffer(columns)?,
        })
    }

    fn bias(
        &self,
        s: &Arc<CudaStream>,
        y: &CudaSlice<f32>,
        (bias, residual): (&CudaSlice<f32>, Option<&CudaSlice<f32>>),
        (count, channels): (usize, usize),
    ) -> Result<()> {
        let residual = residual.map_or(0, |r| pointer(r, s));
        unsafe {
            s.launch_builder(&self.kernels.bias_residual)
                .arg(&pointer(y, s))
                .arg(&pointer(bias, s))
                .arg(&residual)
                .arg(&(count as u64))
                .arg(&u32::try_from(channels)?)
                .arg(&1u64)
                .launch(flat(count))?;
        }
        Ok(())
    }

    /// `y [rows, out] = conv(x [rows, input]) + bias (+ residual)`: zero pad, im2col, one F32 GEMM against the folded
    /// `[out, input * kernel]` weight.
    fn conv(
        &self,
        (s, blas, w): (&Arc<CudaStream>, &Blas, &AudioWork),
        c: &Conv,
        (x, rows): (&CudaSlice<f32>, usize),
        y: &CudaSlice<f32>,
        residual: Option<&CudaSlice<f32>>,
    ) -> Result<()> {
        let k = c.input * c.kernel;
        let rows_u = u32::try_from(rows)?;
        let weight = view(
            &c.weight,
            s,
            0,
            [u32::try_from(c.out)?, u32::try_from(k)?],
            [k as u64, 1],
        )?;
        let a = if c.kernel == 1 {
            x
        } else {
            let pad = c.dilation * (c.kernel - 1) / 2;
            let padded = (rows + 2 * pad) * c.input;
            ensure!(
                padded <= w.padded.len() && rows * k <= w.columns.len(),
                "conv1d {rows}x{} k{} d{} exceeds its workspace",
                c.input,
                c.kernel,
                c.dilation
            );
            unsafe {
                s.launch_builder(&self.kernels.pad)
                    .arg(&pointer(x, s))
                    .arg(&pointer(&w.padded, s))
                    .arg(&(rows as i64))
                    .arg(&(pad as i64))
                    .arg(&(c.input as i64))
                    .launch(flat(padded))?;
                s.launch_builder(&self.kernels.im2col)
                    .arg(&pointer(&w.padded, s))
                    .arg(&pointer(&w.columns, s))
                    .arg(&(rows as i64))
                    .arg(&i32::try_from(c.input)?)
                    .arg(&i32::try_from(c.kernel)?)
                    .arg(&i32::try_from(c.dilation)?)
                    .launch(flat(rows * k))?;
            }
            &w.columns
        };
        blas.linear_f32(
            weight,
            view(a, s, 0, [rows_u, u32::try_from(k)?], [k as u64, 1])?,
            &mut TensorMut(view(
                y,
                s,
                0,
                [rows_u, u32::try_from(c.out)?],
                [c.out as u64, 1],
            )?),
        )?;
        match &c.bias {
            Some(bias) => self.bias(s, y, (bias, residual), (rows * c.out, c.out)),
            None => {
                ensure!(residual.is_none(), "residual on a bias-less convolution");
                Ok(())
            }
        }
    }

    /// `y [rows, C] = activation(x [rows, C])`: 2x Kaiser upsample, SnakeBeta, Kaiser lowpass downsample.
    fn activation(
        &self,
        (s, w): (&Arc<CudaStream>, &AudioWork),
        a: &Activation,
        (x, rows, channels): (&CudaSlice<f32>, usize, usize),
        y: &CudaSlice<f32>,
    ) -> Result<()> {
        let (ratio, taps) = (AUDIO_RESAMPLE_RATIO, AUDIO_RESAMPLE_TAPS);
        let snake = |x: &CudaSlice<f32>, y: &CudaSlice<f32>, count: usize| -> Result<()> {
            unsafe {
                s.launch_builder(&self.kernels.snake)
                    .arg(&pointer(x, s))
                    .arg(&pointer(&a.alpha, s))
                    .arg(&pointer(&a.beta, s))
                    .arg(&pointer(y, s))
                    .arg(&(count as i64))
                    .arg(&i32::try_from(channels)?)
                    .launch(flat(count))?;
            }
            Ok(())
        };
        if !self.alias_free {
            return snake(x, y, rows * channels);
        }
        let pad = taps / ratio - 1;
        let crop = pad * ratio + (taps - ratio) / 2;
        let crop_right = pad * ratio + (taps - ratio).div_ceil(2);
        ensure!(
            (rows + 2 * pad - 1) * ratio + taps - crop - crop_right == rows * ratio
                && rows * ratio * channels <= w.up.len(),
            "activation resampling of {rows} rows does not return {} rows",
            rows * ratio
        );
        let (left, right) = (taps / 2 - usize::from(taps % 2 == 0), taps / 2);
        let up_rows = rows * ratio;
        ensure!(
            (up_rows + left + right - taps) / ratio + 1 == rows,
            "lowpass of {up_rows} rows does not return {rows} rows"
        );
        unsafe {
            s.launch_builder(&self.kernels.up)
                .arg(&pointer(x, s))
                .arg(&pointer(&a.up, s))
                .arg(&pointer(&w.up, s))
                .arg(&i32::try_from(rows)?)
                .arg(&i32::try_from(channels)?)
                .arg(&i32::try_from(taps)?)
                .arg(&i32::try_from(ratio)?)
                .arg(&i32::try_from(pad)?)
                .arg(&i32::try_from(crop)?)
                .arg(&(ratio as f32))
                .launch(flat(up_rows * channels))?;
        }
        snake(&w.up, &w.up, up_rows * channels)?;
        unsafe {
            s.launch_builder(&self.kernels.down)
                .arg(&pointer(&w.up, s))
                .arg(&pointer(&a.down, s))
                .arg(&pointer(y, s))
                .arg(&i32::try_from(up_rows)?)
                .arg(&i32::try_from(channels)?)
                .arg(&i32::try_from(taps)?)
                .arg(&i32::try_from(ratio)?)
                .arg(&i32::try_from(left)?)
                .arg(&i32::try_from(rows)?)
                .launch(flat(rows * channels))?;
        }
        Ok(())
    }

    /// Decodes `latents` (`[channels, latent_channels, frames]` F32, the definition's denormalised decoder input; the
    /// stereo channels decode independently) into `[channels, 1, frames * hop]` F32 clamped to [-1, 1].
    pub(crate) fn decode(
        &self,
        (s, blas): (&Arc<CudaStream>, &Blas),
        latents: &CudaSlice<f32>,
        (channels, frames): (usize, usize),
    ) -> Result<CudaSlice<f32>> {
        let lc = self.spec.latent_channels;
        ensure!(
            latents.len() == channels * lc * frames && frames > 0,
            "audio latents hold {} values, not {channels} x {lc} x {frames}",
            latents.len()
        );
        let samples = frames * self.spec.hop();
        let mut out = s.alloc_zeros::<f32>(channels * samples)?;
        let mut w = self.work(s, frames)?;
        eprintln!(
            "{}",
            serde_json::json!({"h3_audio_decode": {"channels": channels, "frames": frames, "samples": samples,
                "alias_free": self.alias_free, "free_bytes": s.context().mem_get_info()?.0}})
        );
        for b in 0..channels {
            let source =
                pointer(latents, s) + (b * lc * frames * std::mem::size_of::<f32>()) as u64;
            unsafe {
                s.launch_builder(&self.kernels.transpose)
                    .arg(&source)
                    .arg(&pointer(&w.t, s))
                    .arg(&(lc as i64))
                    .arg(&(frames as i64))
                    .launch(flat(lc * frames))?;
            }
            let mut rows = frames;
            self.conv((s, blas, &w), &self.in_proj, (&w.t, rows), &w.b, None)?;
            self.conv((s, blas, &w), &self.conv_pre, (&w.b, rows), &w.a, None)?;
            for level in &self.levels {
                let up = &level.up;
                unsafe {
                    s.launch_builder(&self.kernels.conv_transpose)
                        .arg(&pointer(&w.a, s))
                        .arg(&pointer(&up.weight, s))
                        .arg(&pointer(&up.bias, s))
                        .arg(&pointer(&w.b, s))
                        .arg(&i32::try_from(rows)?)
                        .arg(&i32::try_from(up.input)?)
                        .arg(&i32::try_from(up.out)?)
                        .arg(&i32::try_from(up.kernel)?)
                        .arg(&i32::try_from(up.stride)?)
                        .arg(&i32::try_from((up.kernel - up.stride) / 2)?)
                        .launch(LaunchConfig {
                            grid_dim: (u32::try_from(rows * up.stride)?, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }
                rows *= up.stride;
                let c = up.out;
                let count = rows * c;
                std::mem::swap(&mut w.a, &mut w.b);
                let scale = (level.blocks.len() as f32).recip();
                for (j, block) in level.blocks.iter().enumerate() {
                    let mut current = &w.a;
                    for ((c1, c2), (a1, a2)) in block.convs.iter().zip(&block.activations) {
                        self.activation((s, &w), a1, (current, rows, c), &w.act)?;
                        self.conv((s, blas, &w), c1, (&w.act, rows), &w.t, None)?;
                        self.activation((s, &w), a2, (&w.t, rows, c), &w.act)?;
                        let target = if std::ptr::eq(current, &w.b) {
                            &w.c
                        } else {
                            &w.b
                        };
                        self.conv((s, blas, &w), c2, (&w.act, rows), target, Some(current))?;
                        current = target;
                    }
                    if j == 0 {
                        s.memcpy_dtod(&current.slice(..count), &mut w.acc.slice_mut(..count))?;
                    } else {
                        let factor = if j + 1 == level.blocks.len() {
                            scale
                        } else {
                            1.0
                        };
                        unsafe {
                            s.launch_builder(&self.kernels.add_scale)
                                .arg(&pointer(&w.acc, s))
                                .arg(&pointer(current, s))
                                .arg(&(count as i64))
                                .arg(&factor)
                                .launch(flat(count))?;
                        }
                    }
                }
                std::mem::swap(&mut w.a, &mut w.acc);
            }
            let c = self
                .levels
                .last()
                .context("audio decoder without levels")?
                .up
                .out;
            self.activation((s, &w), &self.post, (&w.a, rows, c), &w.act)?;
            ensure!(
                rows == samples,
                "decoded {rows} samples, expected {samples}"
            );
            self.conv((s, blas, &w), &self.conv_post, (&w.act, rows), &w.b, None)?;
            unsafe {
                s.launch_builder(&self.kernels.clamp)
                    .arg(&pointer(&w.b, s))
                    .arg(&(samples as i64))
                    .arg(&-1.0f32)
                    .arg(&1.0f32)
                    .launch(flat(samples))?;
            }
            s.memcpy_dtod(
                &w.b.slice(..samples),
                &mut out.slice_mut(b * samples..(b + 1) * samples),
            )?;
        }
        Ok(out)
    }
}
