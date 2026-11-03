//! MiniMax-H3 video VAE decoder: the non-causal ViT decoder over spatial tiles and temporal chunks of a latent video.
use super::{BlockKernels, NormRows, Source, matrix, pointer, rows_view, vector};
use crate::backend::cuda::{
    Device,
    blas::Blas,
    contracts::TensorMut,
    cubin,
    flash::{Flash, Score},
    linear::LinearBiasMode,
    ops::{self, flat},
    verification::Shape,
};
use crate::backend::gpu::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use crate::checkpoint::TensorManifest;
use crate::minimax_h3::{DecodePlan, VaeSpec, VaeTiles};
use anyhow::{Result, ensure};
use half::bf16;
use std::{path::Path, sync::Arc};

const MODE: LinearBiasMode = LinearBiasMode::BeforeOutputRound;
/// Threads per q/k head-norm row of `rms_norm_bf16`.
const QK_NORM_THREADS: u32 = 32;
/// Rotary axes of the decoder's position grid: (frame, row, column).
const ROPE_AXES: usize = 3;

/// The parameters the model definition holds in BF16 under `torch_dtype=bfloat16`; every other parameter is F32
/// (`_keep_in_fp32_modules`: encoder, quant_conv, norm1, norm2, norm_out, scale1, scale2).
pub(crate) fn bf16_parameter(name: &str) -> bool {
    !(name.starts_with("encoder.")
        || name.starts_with("quant_conv.")
        || name.starts_with("decoder.norm_out.")
        || name.ends_with(".norm1.weight")
        || name.ends_with(".norm2.weight")
        || name.ends_with(".scale1")
        || name.ends_with(".scale2"))
}

/// A biased BF16 projection `[out, input]` cast from the published F32 tensor at load.
struct Linear {
    weight: CudaSlice<bf16>,
    bias: CudaSlice<bf16>,
    out: usize,
    input: usize,
}

impl Linear {
    fn prepare(&self, s: &Arc<CudaStream>, blas: &mut Blas, rows: usize) -> Result<()> {
        blas.prepare_biased(
            u32::try_from(rows)?,
            u32::try_from(self.out)?,
            u32::try_from(self.input)?,
            vector(&self.bias, s)?,
            MODE,
        )
    }

    fn run(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        x: &CudaSlice<bf16>,
        y: &CudaSlice<bf16>,
        rows: usize,
    ) -> Result<()> {
        blas.linear_biased(
            matrix(&self.weight, s, self.out, self.input)?,
            vector(&self.bias, s)?,
            rows_view(x, s, 0, rows, self.input, self.input)?,
            &mut TensorMut(rows_view(y, s, 0, rows, self.out, self.out)?),
            MODE,
        )
    }
}

struct DecoderBlock {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    ff_in: Linear,
    ff_out: Linear,
    norm1: CudaSlice<f32>,
    norm2: CudaSlice<f32>,
    scale1: CudaSlice<f32>,
    scale2: CudaSlice<f32>,
}

struct VaeKernels {
    cast: CudaFunction,
    upcast: CudaFunction,
    rms_weight: CudaFunction,
    scaled_residual: CudaFunction,
    layer_norm: CudaFunction,
    unpatchify: CudaFunction,
    stitch: CudaFunction,
    frame_blend: CudaFunction,
    latent_rows: CudaFunction,
}

/// Loads published F32 tensors, casting the BF16 set through one F32 device staging buffer.
struct Loader<'a, 'b> {
    s: &'b Arc<CudaStream>,
    source: Source<'a>,
    staging: CudaSlice<f32>,
    cast: &'b CudaFunction,
    bytes: usize,
}

impl Loader<'_, '_> {
    fn f32(&mut self, name: &str, elements: usize) -> Result<CudaSlice<f32>> {
        ensure!(!bf16_parameter(name), "{name} is in the BF16 set");
        self.bytes += 4 * elements;
        self.source.tensor::<f32>(self.s, name, elements)
    }

    fn bf16(&mut self, name: &str, elements: usize) -> Result<CudaSlice<bf16>> {
        ensure!(bf16_parameter(name), "{name} is in the F32 set");
        ensure!(
            elements <= self.staging.len(),
            "{name}: {elements} elements exceed the staging buffer"
        );
        Source::into::<f32>(
            &mut self.source,
            self.s,
            name,
            &mut self.staging.slice_mut(..elements),
        )?;
        let out = self.s.alloc_zeros::<bf16>(elements)?;
        unsafe {
            self.s
                .launch_builder(self.cast)
                .arg(&pointer(&self.staging, self.s))
                .arg(&pointer(&out, self.s))
                .arg(&(elements as u64))
                .arg(&0f32)
                .arg(&1f32)
                .launch(flat(elements))?;
        }
        self.bytes += 2 * elements;
        Ok(out)
    }

    fn linear(&mut self, base: &str, out: usize, input: usize) -> Result<Linear> {
        Ok(Linear {
            weight: self.bf16(&format!("{base}.weight"), out * input)?,
            bias: self.bf16(&format!("{base}.bias"), out)?,
            out,
            input,
        })
    }
}

/// The video VAE decoder side: post_quant_conv, proj_in, register tokens, the transformer blocks, norm_out and
/// proj_out, resident on the device in the model definition's dtype map.
pub(crate) struct VideoDecoder {
    spec: VaeSpec,
    post_quant: Linear,
    proj_in: Linear,
    registers: CudaSlice<bf16>,
    norm_out: (CudaSlice<f32>, CudaSlice<f32>),
    proj_out: Linear,
    blocks: Vec<DecoderBlock>,
    kernels: VaeKernels,
    block_kernels: BlockKernels,
}

impl VideoDecoder {
    pub(crate) fn new(
        d: &Device,
        spec: &VaeSpec,
        vae_dir: &Path,
        manifest: &TensorManifest,
    ) -> Result<Self> {
        let s = &d.stream;
        let (free, total) = d.ctx.mem_get_info()?;
        let vision = cubin::module(s.context(), "vision_ops")?;
        let kernels = VaeKernels {
            cast: vision.load_function("vision_affine_cast_bf16")?,
            upcast: vision.load_function("vision_upcast_f32")?,
            rms_weight: vision.load_function("rms_norm_weight_f32")?,
            scaled_residual: vision.load_function("scaled_residual_f32")?,
            layer_norm: cubin::module(s.context(), "q38_common")?
                .load_function("q38_layer_norm_f32")?,
            unpatchify: vision.load_function("unpatchify_bf16")?,
            stitch: vision.load_function("tile_stitch_bf16")?,
            frame_blend: vision.load_function("frame_blend_bf16")?,
            latent_rows: vision.load_function("latent_rows_bf16")?,
        };
        let (dim, latent, ffn) = (spec.dim, spec.latent_channels, spec.ffn);
        let largest = (2 * ffn * dim).max(spec.patch_width() * dim);
        let mut load = Loader {
            s,
            source: Source::new(vae_dir, manifest),
            staging: s.alloc_zeros::<f32>(largest)?,
            cast: &kernels.cast,
            bytes: 0,
        };
        let post_quant = load.linear("post_quant_conv", latent, latent)?;
        let proj_in = load.linear("decoder.proj_in", dim, latent)?;
        let registers = load.bf16("decoder.register_tokens", spec.registers * dim)?;
        let norm_out = (
            load.f32("decoder.norm_out.weight", dim)?,
            load.f32("decoder.norm_out.bias", dim)?,
        );
        let proj_out = load.linear("decoder.proj_out", spec.patch_width(), dim)?;
        let mut blocks = Vec::with_capacity(spec.layers);
        for layer in 0..spec.layers {
            let b = format!("decoder.transformer_blocks.{layer}");
            blocks.push(DecoderBlock {
                q: load.linear(&format!("{b}.attn.to_q"), dim, dim)?,
                k: load.linear(&format!("{b}.attn.to_k"), dim, dim)?,
                v: load.linear(&format!("{b}.attn.to_v"), dim, dim)?,
                out: load.linear(&format!("{b}.attn.to_out.0"), dim, dim)?,
                ff_in: load.linear(&format!("{b}.ff.net.0.proj"), 2 * ffn, dim)?,
                ff_out: load.linear(&format!("{b}.ff.net.2"), dim, ffn)?,
                norm1: load.f32(&format!("{b}.norm1.weight"), dim)?,
                norm2: load.f32(&format!("{b}.norm2.weight"), dim)?,
                scale1: load.f32(&format!("{b}.scale1"), dim)?,
                scale2: load.f32(&format!("{b}.scale2"), dim)?,
            });
        }
        let loaded = load.bytes;
        drop(load);
        s.synchronize()?;
        eprintln!(
            "{}",
            serde_json::json!({"h3_vae_decoder_placement": {"mode": "resident", "weight_bytes": loaded,
                "free_before_bytes": free, "free_after_bytes": d.ctx.mem_get_info()?.0, "total_bytes": total}})
        );
        Ok(Self {
            spec: spec.clone(),
            post_quant,
            proj_in,
            registers,
            norm_out,
            proj_out,
            blocks,
            kernels,
            block_kernels: BlockKernels::new(s.context())?,
        })
    }

    fn linears(&self) -> impl Iterator<Item = &Linear> {
        [&self.proj_out].into_iter().chain(
            self.blocks
                .iter()
                .flat_map(|b| [&b.q, &b.k, &b.v, &b.out, &b.ff_in, &b.ff_out]),
        )
    }

    fn block(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        b: &DecoderBlock,
        w: &mut TileWork,
    ) -> Result<()> {
        let spec = &self.spec;
        let (t, dim) = (w.tokens, spec.dim);
        self.rms_weight(s, &w.x, &b.norm1, &w.normed, t)?;
        b.q.run(s, blas, &w.normed, &w.q, t)?;
        b.k.run(s, blas, &w.normed, &w.k, t)?;
        b.v.run(s, blas, &w.normed, &w.v, t)?;
        for (input, output) in [(&w.q, &w.q_n), (&w.k, &w.k_n)] {
            self.block_kernels.rms_norm(
                s,
                NormRows {
                    input: pointer(input, s),
                    weight: pointer(&w.ones, s),
                    output: pointer(output, s),
                    rows: t * spec.heads,
                    dim: spec.head_dim,
                    threads: QK_NORM_THREADS,
                    eps: spec.norm_eps,
                },
            )?;
        }
        let t_i = i32::try_from(t)?;
        let elements = t * 2 * dim;
        let rotary = &self.block_kernels.rotary;
        unsafe {
            s.launch_builder(rotary)
                .arg(&pointer(&w.q_n, s))
                .arg(&pointer(&w.k_n, s))
                .arg(&pointer(&w.v, s))
                .arg(&pointer(&w.k_cache, s))
                .arg(&pointer(&w.v_cache, s))
                .arg(&pointer(&w.rope, s))
                .arg(&0i32)
                .arg(&t_i)
                .arg(&(spec.heads as i32))
                .arg(&(spec.heads as i32))
                .arg(&(spec.head_dim as i32))
                .arg(&(spec.rope_dims as i32))
                .arg(&t_i)
                .launch(flat(elements))?;
        }
        w.flash.run(
            &w.q_n,
            dim,
            &w.k_cache,
            &w.v_cache,
            &mut w.attn,
            Shape {
                rows: t,
                start: 0,
                q_heads: spec.heads,
                kv_heads: spec.heads,
                capacity: t,
                head_dim: spec.head_dim,
                causal: false,
                window_left: None,
            },
        )?;
        b.out.run(s, blas, &w.attn, &w.o, t)?;
        self.scaled_residual(s, &w.x, &w.o, &b.scale1, t)?;
        self.rms_weight(s, &w.x, &b.norm2, &w.normed, t)?;
        b.ff_in.run(s, blas, &w.normed, &w.gate_up, t)?;
        let (gate_up, gated) = (pointer(&w.gate_up, s), pointer(&w.gated, s));
        let cfg = ops::silu_mul_grid(t, spec.ffn, gate_up, gated)?;
        unsafe {
            s.launch_builder(&self.block_kernels.swiglu)
                .arg(&gate_up)
                .arg(&gated)
                .arg(&t_i)
                .arg(&i32::try_from(spec.ffn)?)
                .arg(&3i32)
                .launch(cfg)?;
        }
        b.ff_out.run(s, blas, &w.gated, &w.o, t)?;
        self.scaled_residual(s, &w.x, &w.o, &b.scale2, t)
    }

    fn rms_weight(
        &self,
        s: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        out: &CudaSlice<bf16>,
        rows: usize,
    ) -> Result<()> {
        unsafe {
            s.launch_builder(&self.kernels.rms_weight)
                .arg(&pointer(x, s))
                .arg(&pointer(weight, s))
                .arg(&pointer(out, s))
                .arg(&u32::try_from(self.spec.dim)?)
                .arg(&self.spec.norm_eps)
                .launch(LaunchConfig {
                    grid_dim: (u32::try_from(rows)?, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    fn scaled_residual(
        &self,
        s: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        delta: &CudaSlice<bf16>,
        scale: &CudaSlice<f32>,
        rows: usize,
    ) -> Result<()> {
        let count = rows * self.spec.dim;
        let width = u32::try_from(self.spec.dim)?;
        unsafe {
            s.launch_builder(&self.kernels.scaled_residual)
                .arg(&pointer(x, s))
                .arg(&pointer(delta, s))
                .arg(&pointer(scale, s))
                .arg(&(count as u64))
                .arg(&width)
                .launch(flat(count))?;
        }
        Ok(())
    }

    /// Decodes one spatial tile of one chunk into raw tile slot `slot` of `raw` (`[slots, channels, frames, h, w]`).
    fn tile(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        w: &mut TileWork,
        z: &LatentVideo<'_>,
        origin: (usize, usize, usize),
        (raw, slot): (&CudaSlice<bf16>, usize),
    ) -> Result<()> {
        let spec = &self.spec;
        let (n, t, dim) = (w.patches, w.tokens, spec.dim);
        let (first_frame, first_row, first_col) = origin;
        let count = n * spec.latent_channels;
        unsafe {
            s.launch_builder(&self.kernels.latent_rows)
                .arg(&pointer(z.data, s))
                .arg(&pointer(&w.latent, s))
                .arg(&(spec.latent_channels as u32))
                .arg(&(z.frames as u32))
                .arg(&(z.height as u32))
                .arg(&(z.width as u32))
                .arg(&(first_frame as u32))
                .arg(&(first_row as u32))
                .arg(&(first_col as u32))
                .arg(&(w.frames as u32))
                .arg(&(w.height as u32))
                .arg(&(w.width as u32))
                .launch(flat(count))?;
        }
        self.post_quant.run(s, blas, &w.latent, &w.latent_q, n)?;
        self.proj_in.run(s, blas, &w.latent_q, &w.hidden, n)?;
        s.memcpy_dtod(
            &self.registers,
            &mut w.hidden.slice_mut(n * dim..(n + spec.registers) * dim),
        )?;
        let elements = u32::try_from(t * dim)?;
        unsafe {
            s.launch_builder(&self.kernels.upcast)
                .arg(&pointer(&w.hidden, s))
                .arg(&pointer(&w.x, s))
                .arg(&elements)
                .launch(flat(t * dim))?;
        }
        for b in &self.blocks {
            self.block(s, blas, b, w)?;
        }
        let d = dim as u64;
        unsafe {
            s.launch_builder(&self.kernels.layer_norm)
                .arg(&pointer(&w.x, s))
                .arg(&pointer(&self.norm_out.0, s))
                .arg(&pointer(&self.norm_out.1, s))
                .arg(&pointer(&w.y, s))
                .arg(&u32::try_from(t)?)
                .arg(&1u32)
                .arg(&u32::try_from(dim)?)
                .arg(&d)
                .arg(&0u64)
                .arg(&1u64)
                .arg(&d)
                .arg(&0u64)
                .arg(&1u64)
                .arg(&spec.norm_eps)
                .launch(LaunchConfig {
                    grid_dim: (u32::try_from(t)?, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        unsafe {
            s.launch_builder(&self.kernels.cast)
                .arg(&pointer(&w.y, s))
                .arg(&pointer(&w.normed, s))
                .arg(&((t * dim) as u64))
                .arg(&0f32)
                .arg(&1f32)
                .launch(flat(t * dim))?;
        }
        self.proj_out.run(s, blas, &w.normed, &w.patch_rows, t)?;
        let tile_size = w.pixels();
        let target = pointer(raw, s) + (slot * tile_size * std::mem::size_of::<bf16>()) as u64;
        ensure!(
            (slot + 1) * tile_size <= raw.len(),
            "raw tile slot {slot} out of range"
        );
        unsafe {
            s.launch_builder(&self.kernels.unpatchify)
                .arg(&pointer(&w.patch_rows, s))
                .arg(&target)
                .arg(&(w.frames as u32))
                .arg(&(w.height as u32))
                .arg(&(w.width as u32))
                .arg(&(spec.out_channels as u32))
                .arg(&(spec.temporal_ratio as u32))
                .arg(&(spec.spatial_ratio as u32))
                .launch(flat(tile_size))?;
        }
        Ok(())
    }

    /// Decodes `z` (`[latent_channels, frames, height, width]` F32) through `tiles` into `out`
    /// (`[out_channels, frames, height * ratio, width * ratio]` BF16, `plan.chunks * kept + tail` frames, of which the
    /// first `plan.frames` are the video).
    pub(crate) fn decode(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        w: &mut TileWork,
        tiles: &VaeTiles,
        z: LatentVideo<'_>,
        out: &mut CudaSlice<bf16>,
    ) -> Result<DecodePlan> {
        let spec = &self.spec;
        let r = spec.spatial_ratio;
        let plan = DecodePlan::new(spec, z.frames)?;
        ensure!(
            plan.pad == 0,
            "padded latent lengths ({} pad frames) are not bound",
            plan.pad
        );
        let (chunk, overlap) = (spec.chunk_latents(), spec.overlap_latents());
        ensure!(
            w.frames == chunk + overlap
                && w.height * r == tiles.height
                && w.width * r == tiles.width,
            "tile work {}x{}x{} does not match the tile grid {}x{} px",
            w.frames,
            w.height,
            w.width,
            tiles.height,
            tiles.width
        );
        let (height, width) = (z.height * r, z.width * r);
        let plane = height * width;
        let clip_frames = w.frames * spec.temporal_ratio;
        let (pre, fade) = (spec.frame_pre_padding(), spec.frame_overlap());
        let lead = chunk * spec.temporal_ratio;
        let kept = lead - pre;
        let tail = clip_frames - lead - pre;
        let total = plan.chunks * kept + if spec.token_drop > 0 { tail } else { 0 };
        let channels = spec.out_channels;
        ensure!(
            out.len() == channels * total * plane,
            "decode output holds {} elements, not {}",
            out.len(),
            channels * total * plane
        );
        let (rows, cols) = (&tiles.rows, &tiles.cols);
        let grid: Vec<u32> = rows
            .0
            .iter()
            .chain(&rows.1)
            .chain(&cols.0)
            .chain(&cols.1)
            .map(|&v| u32::try_from(v))
            .collect::<Result<_, _>>()?;
        let grid = s.clone_htod(&grid)?;
        let slots = tiles.count();
        let raw = s.alloc_zeros::<bf16>(slots * w.pixels())?;
        let canvas = s.alloc_zeros::<bf16>(channels * clip_frames * plane)?;
        let mut faded = s.alloc_zeros::<bf16>(channels * tail * plane)?;
        eprintln!(
            "{}",
            serde_json::json!({"h3_vae_decode": {"chunks": plan.chunks, "tiles": slots, "tokens_per_tile": w.tokens,
                "frames": plan.frames, "raw_tile_bytes": 2 * raw.len(), "canvas_bytes": 2 * canvas.len(),
                "output_bytes": 2 * out.len(), "free_bytes": s.context().mem_get_info()?.0}})
        );
        let copy_frames = |s: &Arc<CudaStream>,
                           from: &CudaSlice<bf16>,
                           from_frames: usize,
                           from_first: usize,
                           to: &mut CudaSlice<bf16>,
                           to_frames: usize,
                           to_first: usize,
                           count: usize|
         -> Result<()> {
            for c in 0..channels {
                let a = (c * from_frames + from_first) * plane;
                let b = (c * to_frames + to_first) * plane;
                s.memcpy_dtod(
                    &from.slice(a..a + count * plane),
                    &mut to.slice_mut(b..b + count * plane),
                )?;
            }
            Ok(())
        };
        for i in 0..plan.chunks {
            for (ti, &row) in rows.0.iter().enumerate() {
                for (tj, &col) in cols.0.iter().enumerate() {
                    self.tile(
                        s,
                        blas,
                        w,
                        &z,
                        (i * chunk, row / r, col / r),
                        (&raw, ti * cols.0.len() + tj),
                    )?;
                }
            }
            let planes = channels * clip_frames;
            unsafe {
                s.launch_builder(&self.kernels.stitch)
                    .arg(&pointer(&raw, s))
                    .arg(&pointer(&canvas, s))
                    .arg(&pointer(&grid, s))
                    .arg(&(rows.0.len() as u32))
                    .arg(&(cols.0.len() as u32))
                    .arg(&(planes as u32))
                    .arg(&(tiles.height as u32))
                    .arg(&(tiles.width as u32))
                    .arg(&(height as u32))
                    .arg(&(width as u32))
                    .launch(flat(planes * plane))?;
            }
            let dst = i * kept;
            if i > 0 && fade > 0 {
                unsafe {
                    s.launch_builder(&self.kernels.frame_blend)
                        .arg(&pointer(&faded, s))
                        .arg(&pointer(&canvas, s))
                        .arg(&pointer(out, s))
                        .arg(&(channels as u32))
                        .arg(&(fade as u32))
                        .arg(&(plane as u64))
                        .arg(&(tail as u32))
                        .arg(&0u32)
                        .arg(&(clip_frames as u32))
                        .arg(&(pre as u32))
                        .arg(&(total as u32))
                        .arg(&(dst as u32))
                        .launch(flat(channels * fade * plane))?;
                }
                copy_frames(
                    s,
                    &canvas,
                    clip_frames,
                    pre + fade,
                    out,
                    total,
                    dst + fade,
                    kept - fade,
                )?;
            } else {
                copy_frames(s, &canvas, clip_frames, pre, out, total, dst, kept)?;
            }
            if spec.token_drop > 0 {
                copy_frames(
                    s,
                    &canvas,
                    clip_frames,
                    lead + pre,
                    &mut faded,
                    tail,
                    0,
                    tail,
                )?;
            }
        }
        if spec.token_drop > 0 {
            copy_frames(s, &faded, tail, 0, out, total, plan.chunks * kept, tail)?;
        }
        Ok(plan)
    }
}

/// A device latent video `[latent_channels, frames, height, width]` F32.
pub(crate) struct LatentVideo<'a> {
    pub data: &'a CudaSlice<f32>,
    pub frames: usize,
    pub height: usize,
    pub width: usize,
}

/// Activations of one decoder tile forward: `frames x height x width` latent patches plus the register and cls tokens.
pub(crate) struct TileWork {
    frames: usize,
    height: usize,
    width: usize,
    patches: usize,
    tokens: usize,
    latent: CudaSlice<bf16>,
    latent_q: CudaSlice<bf16>,
    hidden: CudaSlice<bf16>,
    x: CudaSlice<f32>,
    y: CudaSlice<f32>,
    normed: CudaSlice<bf16>,
    q: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    q_n: CudaSlice<bf16>,
    k_n: CudaSlice<bf16>,
    k_cache: CudaSlice<bf16>,
    v_cache: CudaSlice<bf16>,
    attn: CudaSlice<bf16>,
    o: CudaSlice<bf16>,
    gate_up: CudaSlice<bf16>,
    gated: CudaSlice<bf16>,
    patch_rows: CudaSlice<bf16>,
    rope: CudaSlice<bf16>,
    ones: CudaSlice<bf16>,
    flash: Flash,
    spec: VaeSpec,
}

impl TileWork {
    pub(crate) fn new(
        s: &Arc<CudaStream>,
        spec: &VaeSpec,
        (frames, height, width): (usize, usize, usize),
    ) -> Result<Self> {
        let patches = frames * height * width;
        let tokens = patches + spec.registers + 1;
        let dim = spec.dim;
        let rows = |n: usize| s.alloc_zeros::<bf16>(n);
        Ok(Self {
            frames,
            height,
            width,
            patches,
            tokens,
            latent: rows(patches * spec.latent_channels)?,
            latent_q: rows(patches * spec.latent_channels)?,
            hidden: rows(tokens * dim)?,
            x: s.alloc_zeros::<f32>(tokens * dim)?,
            y: s.alloc_zeros::<f32>(tokens * dim)?,
            normed: rows(tokens * dim)?,
            q: rows(tokens * dim)?,
            k: rows(tokens * dim)?,
            v: rows(tokens * dim)?,
            q_n: rows(tokens * dim)?,
            k_n: rows(tokens * dim)?,
            k_cache: rows(tokens * dim)?,
            v_cache: rows(tokens * dim)?,
            attn: rows(tokens * dim)?,
            o: rows(tokens * dim)?,
            gate_up: rows(tokens * 2 * spec.ffn)?,
            gated: rows(tokens * spec.ffn)?,
            patch_rows: rows(tokens * spec.patch_width())?,
            rope: s.clone_htod(&rope_table(spec, (frames, height, width)))?,
            ones: s.clone_htod(&vec![bf16::ONE; spec.head_dim])?,
            flash: Flash::new_with_score(
                s.clone(),
                tokens,
                spec.heads,
                spec.heads,
                spec.head_dim,
                Score {
                    scale: (spec.head_dim as f32).sqrt().recip(),
                    window_left: None,
                },
                tokens,
            )?,
            spec: spec.clone(),
        })
    }

    /// Binds the biased GEMM plans of every decoder projection at this tile's row counts.
    pub(crate) fn prepare(
        &self,
        s: &Arc<CudaStream>,
        blas: &mut Blas,
        decoder: &VideoDecoder,
    ) -> Result<()> {
        decoder.post_quant.prepare(s, blas, self.patches)?;
        decoder.proj_in.prepare(s, blas, self.patches)?;
        for linear in decoder.linears() {
            linear.prepare(s, blas, self.tokens)?;
        }
        Ok(())
    }

    /// Pixels of one decoded tile: `out_channels x frames * temporal x height * spatial x width * spatial`.
    fn pixels(&self) -> usize {
        let r = self.spec.spatial_ratio;
        self.spec.out_channels
            * self.frames
            * self.spec.temporal_ratio
            * self.height
            * r
            * self.width
            * r
    }
}

/// BF16 rotary table `[tokens, rope_dims]` (`[cos | sin]` of `rope_dims / 2` angles) of a `frames x height x width`
/// patch grid followed by the register and cls tokens at angle 0. Per axis the coordinate is
/// `2 ((i + 0.5) / size) - 1`, the angle `2 pi x coordinate x theta^(-k * 2 * axes / rope_dims)` in F32, axes in
/// (frame, row, column) order.
pub(crate) fn rope_table(
    spec: &VaeSpec,
    (frames, height, width): (usize, usize, usize),
) -> Vec<bf16> {
    let half = spec.rope_dims / 2;
    let freqs = half / ROPE_AXES;
    let step = (2 * ROPE_AXES) as f32 / spec.rope_dims as f32;
    let inv: Vec<f32> = (0..freqs)
        .map(|k| 1.0f32 / spec.rope_theta.powf(k as f32 * step))
        .collect();
    let coordinate = |i: usize, size: usize| 2.0f32 * ((i as f32 + 0.5) / size as f32) - 1.0;
    let turn = (2.0 * std::f64::consts::PI) as f32;
    let tokens = frames * height * width + spec.registers + 1;
    let mut table = vec![bf16::ZERO; tokens * spec.rope_dims];
    for (row, chunk) in table.chunks_mut(spec.rope_dims).enumerate() {
        if row >= frames * height * width {
            chunk[..half].fill(bf16::ONE);
            continue;
        }
        let position = [row / (height * width), row / width % height, row % width];
        let sizes = [frames, height, width];
        for axis in 0..ROPE_AXES {
            let x = coordinate(position[axis], sizes[axis]);
            for (k, inv) in inv.iter().enumerate() {
                let angle = turn * x * inv;
                chunk[axis * freqs + k] = bf16::from_f32(angle.cos());
                chunk[half + axis * freqs + k] = bf16::from_f32(angle.sin());
            }
        }
    }
    table
}
