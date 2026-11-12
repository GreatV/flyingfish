//! MiniMax-H3 image/video VAEs: tiled ViT decoders and the causal CNN keyframe encoder.
use super::{BlockKernels, NormRows, Source, matrix, pointer, rows_view, vector};
use crate::backend::cuda::{
    Device,
    blas::Blas,
    contracts::{Tensor, TensorMut, view},
    cubin,
    flash::{Flash, Score},
    linear::LinearBiasMode,
    ops::{self, flat},
    verification::Shape,
};
use crate::backend::gpu::{
    CudaFunction, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg,
};
use crate::checkpoint::TensorManifest;
use crate::minimax_h3::{DecodePlan, VAE_PIXEL_MEAN, VAE_PIXEL_STD, VAE_TILE, VaeSpec, VaeTiles};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::{path::Path, sync::Arc};

struct ImageKernels {
    bias: CudaFunction,
    rms: CudaFunction,
    residual: CudaFunction,
    norm: CudaFunction,
    softmax: CudaFunction,
    silu: CudaFunction,
    gather: CudaFunction,
    append: CudaFunction,
    pack: CudaFunction,
    unpack: CudaFunction,
    patches: CudaFunction,
    stitch: CudaFunction,
    pixels: CudaFunction,
}

impl ImageKernels {
    fn new(s: &Arc<CudaStream>) -> Result<Self> {
        let vision = cubin::module(s.context(), "vision_ops")?;
        let common = cubin::module(s.context(), "q38_common")?;
        Ok(Self {
            bias: vision.load_function("vision_bias_f32")?,
            rms: vision.load_function("vision_rms_scale_f32")?,
            residual: common.load_function("channel_scale_residual_f32")?,
            norm: common.load_function("q38_layer_norm_f32")?,
            softmax: common.load_function("softmax_rows_f32")?,
            silu: cubin::module(s.context(), "act")?.load_function("silu_mul_f32")?,
            gather: vision.load_function("tile_rows_f32")?,
            append: vision.load_function("append_register_rows_f32")?,
            pack: vision.load_function("rotary_pack_qkv_f32")?,
            unpack: vision.load_function("head_rows_f32")?,
            patches: vision.load_function("unpatchify_f32")?,
            stitch: vision.load_function("tile_stitch_f32")?,
            pixels: vision.load_function("planar_affine_rows_f32")?,
        })
    }
}

struct ImageLimits {
    budget: usize,
    l2: usize,
    sms: usize,
    grid: usize,
}

#[derive(Debug)]
struct ImagePlan {
    linear: usize,
    attention: usize,
    patches: usize,
    tokens: usize,
    grid: usize,
    bytes: usize,
}

impl ImagePlan {
    fn derive(
        spec: &VaeSpec,
        tiles: &VaeTiles,
        canvas: [usize; 2],
        limits: ImageLimits,
    ) -> Result<Self> {
        let slots = tiles.count();
        let patches = tiles.height / spec.spatial_ratio * (tiles.width / spec.spatial_ratio);
        let tokens = patches + spec.registers + 1;
        let s = 4u128 * tokens as u128 * spec.dim as u128;
        let p = 4u128 * spec.heads as u128 * (tokens as u128).pow(2);
        let per_tile = 4u128 * tokens as u128 * (7 * spec.dim + 3 * spec.ffn) as u128
            + 4u128 * tokens as u128 * spec.patch_width() as u128
            + 8u128 * patches as u128 * spec.latent_channels as u128;
        let pixels = canvas[0] as u128 * canvas[1] as u128;
        let grid_words = 2 * (tiles.rows.0.len() + tiles.cols.0.len()) - 2;
        let fixed = 4u128
            * slots as u128
            * spec.out_channels as u128
            * tiles.height as u128
            * tiles.width as u128
            + 8 * pixels * spec.out_channels as u128
            + 4u128 * (grid_words + tokens * spec.rope_dims + spec.head_dim) as u128;
        ensure!(
            slots > 0 && patches > 0 && limits.sms > 0 && limits.grid > 0,
            "image VAE setup has an empty shape or device limit"
        );
        ensure!(
            pixels * spec.out_channels as u128 <= u32::MAX as u128,
            "image VAE canvas exceeds the F32 layout index range"
        );
        ensure!(
            fixed + per_tile + p <= limits.budget as u128,
            "image VAE serial tile requires {} bytes; workspace budget {}",
            fixed + per_tile + p,
            limits.budget
        );
        let cache_cap = (limits.l2 as u128 / (4 * s + p)).max(1) as usize;
        let wave = limits.sms.div_ceil(spec.heads);
        let cache_cap = if cache_cap >= wave {
            cache_cap / wave * wave
        } else {
            cache_cap
        };
        let score_cap = ((limits.budget as u128 - fixed - per_tile) / p) as usize;
        let mut attention = cache_cap.min(slots).min(score_cap).max(1);
        let launch_cap = (u32::MAX as usize / tokens / (2 * spec.ffn))
            .min(i32::MAX as usize / tokens)
            .min(limits.grid / tokens / spec.heads)
            .min(limits.grid.saturating_mul(256) / tokens / (2 * spec.ffn));
        loop {
            let available = limits.budget as u128 - fixed - attention as u128 * p;
            let mut linear = (available / per_tile) as usize;
            linear = linear.min(slots).min(launch_cap);
            ensure!(
                linear > 0,
                "image VAE has no tile batch within device launch limits"
            );
            if linear < attention {
                attention = linear;
                continue;
            }
            if linear < slots {
                linear = linear / attention * attention;
            }
            return Ok(Self {
                linear,
                attention,
                patches,
                tokens,
                grid: limits.grid,
                bytes: usize::try_from(fixed + linear as u128 * per_tile + attention as u128 * p)?,
            });
        }
    }

    fn flat(&self, count: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (count.div_ceil(256).min(self.grid) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

struct ImageWork {
    plan: ImagePlan,
    tiles: VaeTiles,
    canvas: [usize; 2],
    grid: CudaSlice<u32>,
    rope: CudaSlice<f32>,
    ones: CudaSlice<f32>,
    x: CudaSlice<f32>,
    norm: CudaSlice<f32>,
    o: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    ff: CudaSlice<f32>,
    gated: CudaSlice<f32>,
    latent: CudaSlice<f32>,
    latent_q: CudaSlice<f32>,
    patch_rows: CudaSlice<f32>,
    scores: CudaSlice<f32>,
    raw: CudaSlice<f32>,
    planar: CudaSlice<f32>,
    rgb: CudaSlice<f32>,
}

type ImageRows<'a> = (&'a CudaSlice<f32>, usize, usize, usize);

struct ImageDecoder {
    spec: VaeSpec,
    post_quant: Linear<f32>,
    proj_in: Linear<f32>,
    registers: CudaSlice<f32>,
    norm_out: (CudaSlice<f32>, CudaSlice<f32>),
    proj_out: Linear<f32>,
    blocks: Vec<DecoderBlock<f32>>,
    latent_affine: (CudaSlice<f32>, CudaSlice<f32>),
    pixel_affine: (CudaSlice<f32>, CudaSlice<f32>),
    kernels: ImageKernels,
}

impl Linear<f32> {
    fn load(load: &mut LoadF32<'_>, base: &str, out: usize, input: usize) -> Result<Self> {
        Ok(Self {
            weight: load(&format!("{base}.weight"), out * input)?,
            bias: load(&format!("{base}.bias"), out)?,
            out,
            input,
        })
    }

    fn run(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        bias: &CudaFunction,
        x: Tensor<f32, 2>,
        y: Tensor<f32, 2>,
    ) -> Result<()> {
        blas.linear_f32(
            matrix(&self.weight, s, self.out, self.input)?,
            x,
            &mut TensorMut(y),
        )?;
        unsafe {
            s.launch_builder(bias)
                .arg(&y.ptr)
                .arg(&pointer(&self.bias, s))
                .arg(&y.ptr)
                .arg(&y.shape[0])
                .arg(&y.shape[1])
                .launch(flat(y.shape[0] as usize * self.out))?;
        }
        Ok(())
    }
}

impl ImageDecoder {
    fn new(
        device: &Device,
        spec: VaeSpec,
        component: &Path,
        manifest: &TensorManifest,
    ) -> Result<Self> {
        let s = &device.stream;
        ensure!(
            spec.out_channels == VAE_PIXEL_STD.len(),
            "image VAE pixel affine requires three channels"
        );
        let kernels = ImageKernels::new(s)?;
        let expected: std::collections::BTreeSet<_> = manifest
            .entries
            .keys()
            .filter(|name| name.starts_with("decoder.") || name.starts_with("post_quant_conv."))
            .cloned()
            .collect();
        let weight_bytes: usize = expected
            .iter()
            .map(|name| manifest.entries[name].bytes)
            .sum();
        let (free, total) = device.ctx.mem_get_info()?;
        ensure!(
            weight_bytes <= device.workspace_budget(free, free, total / 16)?,
            "image VAE decoder weights need {weight_bytes} bytes"
        );
        let mut source = Source::new(component, manifest);
        let mut consumed = std::collections::BTreeSet::new();
        let mut load = |name: &str, count: usize| {
            ensure!(
                consumed.insert(name.to_owned()),
                "image VAE duplicate tensor {name}"
            );
            source.tensor::<f32>(s, name, count)
        };
        let (dim, latent, ffn) = (spec.dim, spec.latent_channels, spec.ffn);
        let post_quant = Linear::<f32>::load(&mut load, "post_quant_conv", latent, latent)?;
        let proj_in = Linear::<f32>::load(&mut load, "decoder.proj_in", dim, latent)?;
        let registers = load("decoder.register_tokens", spec.registers * dim)?;
        let norm_out = (
            load("decoder.norm_out.weight", dim)?,
            load("decoder.norm_out.bias", dim)?,
        );
        let proj_out = Linear::<f32>::load(&mut load, "decoder.proj_out", spec.patch_width(), dim)?;
        let mut blocks = Vec::with_capacity(spec.layers);
        for layer in 0..spec.layers {
            let b = format!("decoder.transformer_blocks.{layer}");
            blocks.push(DecoderBlock {
                q: Linear::<f32>::load(&mut load, &format!("{b}.attn.to_q"), dim, dim)?,
                k: Linear::<f32>::load(&mut load, &format!("{b}.attn.to_k"), dim, dim)?,
                v: Linear::<f32>::load(&mut load, &format!("{b}.attn.to_v"), dim, dim)?,
                out: Linear::<f32>::load(&mut load, &format!("{b}.attn.to_out.0"), dim, dim)?,
                ff_in: Linear::<f32>::load(&mut load, &format!("{b}.ff.net.0.proj"), 2 * ffn, dim)?,
                ff_out: Linear::<f32>::load(&mut load, &format!("{b}.ff.net.2"), dim, ffn)?,
                norm1: load(&format!("{b}.norm1.weight"), dim)?,
                norm2: load(&format!("{b}.norm2.weight"), dim)?,
                scale1: load(&format!("{b}.scale1"), dim)?,
                scale2: load(&format!("{b}.scale2"), dim)?,
            });
        }
        ensure!(
            consumed == expected,
            "image VAE partial loader does not cover the decoder tensor set"
        );
        eprintln!(
            "{}",
            serde_json::json!({"image_vae_decoder_load":{"tensors":consumed.len(),"weight_bytes":weight_bytes,"dtype":"F32"}})
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
            latent_affine: (
                s.clone_htod(&spec.latents_std)?,
                s.clone_htod(&spec.latents_mean)?,
            ),
            pixel_affine: (
                s.clone_htod(&VAE_PIXEL_STD.map(|x| x as f32))?,
                s.clone_htod(&VAE_PIXEL_MEAN.map(|x| x as f32))?,
            ),
        })
    }

    fn work(&self, device: &Device, canvas: [usize; 2], budget: usize) -> Result<ImageWork> {
        use cudarc::driver::sys::CUdevice_attribute as A;
        let spec = &self.spec;
        ensure!(
            canvas
                .into_iter()
                .all(|n| n > 0 && n <= u32::MAX as usize && n.is_multiple_of(spec.spatial_ratio)),
            "image VAE invalid canvas {canvas:?}"
        );
        ensure!(
            canvas[0] as u128 * canvas[1] as u128 * spec.out_channels as u128 <= u32::MAX as u128,
            "image VAE canvas exceeds the F32 layout index range"
        );
        let tiles = VaeTiles::released(spec, canvas[0], canvas[1])?;
        let plan = ImagePlan::derive(
            spec,
            &tiles,
            canvas,
            ImageLimits {
                budget,
                l2: usize::try_from(device.ctx.attribute(A::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)?)?,
                sms: usize::try_from(
                    device
                        .ctx
                        .attribute(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
                )?,
                grid: usize::try_from(
                    device
                        .ctx
                        .attribute(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
                )?,
            },
        )?;
        let s = &device.stream;
        let n = plan.linear * plan.tokens * spec.dim;
        let latent = plan.linear * plan.patches * spec.latent_channels;
        let grid: Vec<u32> = tiles
            .rows
            .0
            .iter()
            .chain(&tiles.rows.1)
            .chain(&tiles.cols.0)
            .chain(&tiles.cols.1)
            .map(|&v| u32::try_from(v))
            .collect::<Result<_, _>>()?;
        let rope = rotary_table(
            spec,
            (
                1,
                tiles.height / spec.spatial_ratio,
                tiles.width / spec.spatial_ratio,
            ),
            2,
            |x| x,
        );
        let raw = tiles.count() * spec.out_channels * tiles.height * tiles.width;
        let pixels = spec.out_channels * canvas[0] * canvas[1];
        // Every active element is written before it is read.
        let alloc = |count| unsafe { s.alloc::<f32>(count) };
        let work = ImageWork {
            grid: s.clone_htod(&grid)?,
            rope: s.clone_htod(&rope)?,
            ones: s.clone_htod(&vec![1f32; spec.head_dim])?,
            x: alloc(n)?,
            norm: alloc(n)?,
            o: alloc(n)?,
            q: alloc(n)?,
            k: alloc(n)?,
            v: alloc(n)?,
            attn: alloc(n)?,
            ff: alloc(n * 2 * spec.ffn / spec.dim)?,
            gated: alloc(n * spec.ffn / spec.dim)?,
            latent: alloc(latent)?,
            latent_q: alloc(latent)?,
            patch_rows: alloc(plan.linear * plan.tokens * spec.patch_width())?,
            scores: alloc(plan.attention * spec.heads * plan.tokens * plan.tokens)?,
            raw: alloc(raw)?,
            planar: alloc(pixels)?,
            rgb: alloc(pixels)?,
            plan,
            tiles,
            canvas,
        };
        let actual = [
            &work.rope,
            &work.ones,
            &work.x,
            &work.norm,
            &work.o,
            &work.q,
            &work.k,
            &work.v,
            &work.attn,
            &work.ff,
            &work.gated,
            &work.latent,
            &work.latent_q,
            &work.patch_rows,
            &work.scores,
            &work.raw,
            &work.planar,
            &work.rgb,
        ]
        .iter()
        .map(|b| 4 * b.len())
        .sum::<usize>()
            + 4 * work.grid.len();
        ensure!(
            actual == work.plan.bytes,
            "image VAE work differs from counted memory plan {actual}/{}",
            work.plan.bytes
        );
        eprintln!(
            "{}",
            serde_json::json!({"image_vae_plan":{"canvas":canvas,"tiles":work.tiles.count(),
            "tokens":work.plan.tokens,"linear_batch":work.plan.linear,"attention_batch":work.plan.attention,
            "bytes":actual,"workspace_budget":budget,"silu_grid":flat(work.plan.linear*work.plan.tokens*spec.ffn).grid_dim.0}})
        );
        Ok(work)
    }

    fn rms(
        &self,
        s: &Arc<CudaStream>,
        input: Tensor<f32, 2>,
        weight: &CudaSlice<f32>,
        output: u64,
    ) -> Result<()> {
        unsafe {
            s.launch_builder(&self.kernels.rms)
                .arg(&input.ptr)
                .arg(&pointer(weight, s))
                .arg(&1u32)
                .arg(&output)
                .arg(&0u64)
                .arg(&input.shape[1])
                .arg(&self.spec.norm_eps)
                .launch(ops::grid(input.shape[0] as usize, 256))?;
        }
        Ok(())
    }

    fn residual(
        &self,
        s: &Arc<CudaStream>,
        w: &ImageWork,
        branch: &CudaSlice<f32>,
        scale: &CudaSlice<f32>,
        rows: usize,
    ) -> Result<()> {
        unsafe {
            s.launch_builder(&self.kernels.residual)
                .arg(&pointer(&w.x, s))
                .arg(&pointer(branch, s))
                .arg(&pointer(scale, s))
                .arg(&pointer(&w.x, s))
                .arg(&u32::try_from(rows)?)
                .arg(&u32::try_from(self.spec.dim)?)
                .launch(w.plan.flat(rows * self.spec.dim))?;
        }
        Ok(())
    }

    fn attention(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        w: &ImageWork,
        active: usize,
    ) -> Result<()> {
        let spec = &self.spec;
        let (t, dim) = (w.plan.tokens, spec.dim);
        let shape = |batch, rows, width| -> Result<[u32; 3]> {
            Ok([
                u32::try_from(batch)?,
                u32::try_from(rows)?,
                u32::try_from(width)?,
            ])
        };
        for first in (0..active).step_by(w.plan.attention) {
            let batch = (active - first).min(w.plan.attention) * spec.heads;
            let offset = first * t * dim;
            let heads = |buffer: &CudaSlice<f32>| {
                view(
                    buffer,
                    s,
                    offset,
                    shape(batch, t, spec.head_dim)?,
                    [(t * spec.head_dim) as u64, spec.head_dim as u64, 1],
                )
            };
            let scores = view(
                &w.scores,
                s,
                0,
                shape(batch, t, t)?,
                [(t * t) as u64, t as u64, 1],
            )?;
            blas.batched_f32(
                heads(&w.q)?,
                heads(&w.k)?,
                &mut TensorMut(scores),
                [false, true],
                (spec.head_dim as f32).sqrt().recip(),
            )?;
            unsafe {
                s.launch_builder(&self.kernels.softmax)
                    .arg(&scores.ptr)
                    .arg(&u32::try_from(batch * t)?)
                    .arg(&u32::try_from(t)?)
                    .arg(&(t as u64))
                    .launch(ops::grid(batch * t, 256))?;
            }
            blas.batched_f32(
                scores,
                heads(&w.v)?,
                &mut TensorMut(heads(&w.attn)?),
                [false, false],
                1.0,
            )?;
        }
        unsafe {
            s.launch_builder(&self.kernels.unpack)
                .arg(&pointer(&w.attn, s))
                .arg(&pointer(&w.o, s))
                .arg(&u32::try_from(active)?)
                .arg(&u32::try_from(t)?)
                .arg(&u32::try_from(spec.heads)?)
                .arg(&u32::try_from(spec.head_dim)?)
                .launch(w.plan.flat(active * t * dim))?;
        }
        Ok(())
    }

    fn block(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        block: &DecoderBlock<f32>,
        w: &ImageWork,
        group: (usize, usize, usize),
        observe: &mut impl FnMut(&str, usize, usize, ImageRows<'_>) -> Result<()>,
    ) -> Result<()> {
        let spec = &self.spec;
        let (first, active, layer) = group;
        let rows = active * w.plan.tokens;
        let n = rows * spec.dim;
        let states = |data, offset| rows_view(data, s, offset, rows, spec.dim, spec.dim);
        self.rms(s, states(&w.x, 0)?, &block.norm1, pointer(&w.norm, s))?;
        for (linear, offset) in [(&block.q, 0), (&block.k, n), (&block.v, 2 * n)] {
            linear.run(
                s,
                blas,
                &self.kernels.bias,
                states(&w.norm, 0)?,
                states(&w.ff, offset)?,
            )?;
        }
        for (offset, name) in [(0, "q0"), (n, "k0")] {
            let heads = rows_view(
                &w.ff,
                s,
                offset,
                rows * spec.heads,
                spec.head_dim,
                spec.head_dim,
            )?;
            self.rms(s, heads, &w.ones, heads.ptr)?;
            if layer == 0 {
                observe(
                    name,
                    first,
                    active,
                    (&w.ff, offset, w.plan.tokens, spec.dim),
                )?;
            }
        }
        let q = pointer(&w.ff, s);
        unsafe {
            s.launch_builder(&self.kernels.pack)
                .arg(&q)
                .arg(&(q + (4 * n) as u64))
                .arg(&(q + (8 * n) as u64))
                .arg(&pointer(&w.rope, s))
                .arg(&pointer(&w.q, s))
                .arg(&pointer(&w.k, s))
                .arg(&pointer(&w.v, s))
                .arg(&u32::try_from(active)?)
                .arg(&u32::try_from(w.plan.tokens)?)
                .arg(&u32::try_from(spec.heads)?)
                .arg(&u32::try_from(spec.head_dim)?)
                .arg(&u32::try_from(spec.rope_dims)?)
                .launch(w.plan.flat(n / 2))?;
        }
        self.attention(s, blas, w, active)?;
        block.out.run(
            s,
            blas,
            &self.kernels.bias,
            states(&w.o, 0)?,
            states(&w.ff, 0)?,
        )?;
        self.residual(s, w, &w.ff, &block.scale1, rows)?;
        self.rms(s, states(&w.x, 0)?, &block.norm2, pointer(&w.norm, s))?;
        block.ff_in.run(
            s,
            blas,
            &self.kernels.bias,
            states(&w.norm, 0)?,
            rows_view(&w.ff, s, 0, rows, 2 * spec.ffn, 2 * spec.ffn)?,
        )?;
        let silu_grid = flat(rows * spec.ffn);
        unsafe {
            s.launch_builder(&self.kernels.silu)
                .arg(&pointer(&w.ff, s))
                .arg(&pointer(&w.gated, s))
                .arg(&i64::try_from(rows)?)
                .arg(&i32::try_from(spec.ffn)?)
                .launch(silu_grid)?;
        }
        block.ff_out.run(
            s,
            blas,
            &self.kernels.bias,
            rows_view(&w.gated, s, 0, rows, spec.ffn, spec.ffn)?,
            states(&w.o, 0)?,
        )?;
        self.residual(s, w, &w.o, &block.scale2, rows)?;
        if layer == 0 || layer + 1 == spec.layers {
            observe(
                if layer == 0 { "block0" } else { "block_last" },
                first,
                active,
                (&w.x, 0, w.plan.tokens, spec.dim),
            )?;
        }
        Ok(())
    }

    fn run(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        w: ImageWork,
        input: &CudaSlice<f32>,
        observe: impl FnMut(&str, usize, usize, ImageRows<'_>) -> Result<()>,
    ) -> Result<CudaSlice<f32>> {
        self.launch(s, blas, &w, input, observe)?;
        s.synchronize()?;
        Ok(w.rgb)
    }

    fn launch(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        w: &ImageWork,
        input: &CudaSlice<f32>,
        mut observe: impl FnMut(&str, usize, usize, ImageRows<'_>) -> Result<()>,
    ) -> Result<()> {
        let spec = &self.spec;
        let (h, width) = (
            w.canvas[0] / spec.spatial_ratio,
            w.canvas[1] / spec.spatial_ratio,
        );
        ensure!(
            input.len() == spec.latent_channels * h * width,
            "image VAE normalized latent extent mismatch"
        );
        let p = &w.plan;
        for first in (0..w.tiles.count()).step_by(p.linear) {
            let active = (w.tiles.count() - first).min(p.linear);
            let rows = active * p.tokens;
            unsafe {
                s.launch_builder(&self.kernels.gather)
                    .arg(&pointer(input, s))
                    .arg(&pointer(&self.latent_affine.0, s))
                    .arg(&pointer(&self.latent_affine.1, s))
                    .arg(&pointer(&w.grid, s))
                    .arg(&pointer(&w.latent, s))
                    .arg(&u32::try_from(spec.latent_channels)?)
                    .arg(&u32::try_from(h)?)
                    .arg(&u32::try_from(width)?)
                    .arg(&u32::try_from(w.tiles.rows.0.len())?)
                    .arg(&u32::try_from(w.tiles.cols.0.len())?)
                    .arg(&u32::try_from(spec.spatial_ratio)?)
                    .arg(&u32::try_from(w.tiles.height / spec.spatial_ratio)?)
                    .arg(&u32::try_from(w.tiles.width / spec.spatial_ratio)?)
                    .arg(&u32::try_from(first)?)
                    .arg(&u32::try_from(active)?)
                    .launch(p.flat(active * p.patches * spec.latent_channels))?;
            }
            let latent = |buffer| {
                rows_view(
                    buffer,
                    s,
                    0,
                    active * p.patches,
                    spec.latent_channels,
                    spec.latent_channels,
                )
            };
            self.post_quant.run(
                s,
                blas,
                &self.kernels.bias,
                latent(&w.latent)?,
                latent(&w.latent_q)?,
            )?;
            self.proj_in.run(
                s,
                blas,
                &self.kernels.bias,
                latent(&w.latent_q)?,
                rows_view(&w.o, s, 0, active * p.patches, spec.dim, spec.dim)?,
            )?;
            observe("proj_in", first, active, (&w.o, 0, p.patches, spec.dim))?;
            unsafe {
                s.launch_builder(&self.kernels.append)
                    .arg(&pointer(&w.o, s))
                    .arg(&pointer(&self.registers, s))
                    .arg(&pointer(&w.x, s))
                    .arg(&u32::try_from(active)?)
                    .arg(&u32::try_from(p.patches)?)
                    .arg(&u32::try_from(spec.registers)?)
                    .arg(&u32::try_from(spec.dim)?)
                    .launch(p.flat(rows * spec.dim))?;
            }
            for (layer, block) in self.blocks.iter().enumerate() {
                self.block(s, blas, block, w, (first, active, layer), &mut observe)?;
            }
            let dim = spec.dim as u64;
            unsafe {
                s.launch_builder(&self.kernels.norm)
                    .arg(&pointer(&w.x, s))
                    .arg(&pointer(&self.norm_out.0, s))
                    .arg(&pointer(&self.norm_out.1, s))
                    .arg(&pointer(&w.norm, s))
                    .arg(&u32::try_from(rows)?)
                    .arg(&1u32)
                    .arg(&u32::try_from(spec.dim)?)
                    .arg(&dim)
                    .arg(&0u64)
                    .arg(&1u64)
                    .arg(&dim)
                    .arg(&0u64)
                    .arg(&1u64)
                    .arg(&spec.norm_eps)
                    .launch(ops::grid(rows, 256))?;
            }
            observe("norm_out", first, active, (&w.norm, 0, p.tokens, spec.dim))?;
            self.proj_out.run(
                s,
                blas,
                &self.kernels.bias,
                rows_view(&w.norm, s, 0, rows, spec.dim, spec.dim)?,
                rows_view(
                    &w.patch_rows,
                    s,
                    0,
                    rows,
                    spec.patch_width(),
                    spec.patch_width(),
                )?,
            )?;
            let pixels = spec.out_channels * w.tiles.height * w.tiles.width;
            for tile in 0..active {
                let source =
                    pointer(&w.patch_rows, s) + (4 * tile * p.tokens * spec.patch_width()) as u64;
                let target = pointer(&w.raw, s) + (4 * (first + tile) * pixels) as u64;
                unsafe {
                    s.launch_builder(&self.kernels.patches)
                        .arg(&source)
                        .arg(&target)
                        .arg(&1u32)
                        .arg(&u32::try_from(w.tiles.height / spec.spatial_ratio)?)
                        .arg(&u32::try_from(w.tiles.width / spec.spatial_ratio)?)
                        .arg(&u32::try_from(spec.out_channels)?)
                        .arg(&1u32)
                        .arg(&u32::try_from(spec.spatial_ratio)?)
                        .launch(p.flat(pixels))?;
                }
            }
        }
        unsafe {
            s.launch_builder(&self.kernels.stitch)
                .arg(&pointer(&w.raw, s))
                .arg(&pointer(&w.planar, s))
                .arg(&pointer(&w.grid, s))
                .arg(&u32::try_from(w.tiles.rows.0.len())?)
                .arg(&u32::try_from(w.tiles.cols.0.len())?)
                .arg(&u32::try_from(spec.out_channels)?)
                .arg(&u32::try_from(w.tiles.height)?)
                .arg(&u32::try_from(w.tiles.width)?)
                .arg(&u32::try_from(w.canvas[0])?)
                .arg(&u32::try_from(w.canvas[1])?)
                .launch(p.flat(w.planar.len()))?;
            s.launch_builder(&self.kernels.pixels)
                .arg(&pointer(&w.planar, s))
                .arg(&pointer(&self.pixel_affine.0, s))
                .arg(&pointer(&self.pixel_affine.1, s))
                .arg(&pointer(&w.rgb, s))
                .arg(&u32::try_from(spec.out_channels)?)
                .arg(&u64::try_from(w.canvas[0] * w.canvas[1])?)
                .launch(p.flat(w.rgb.len()))?;
        }
        Ok(())
    }
}

/// Normalized F32 `[24, latent_h, latent_w]` to F32 `[height, width, 3]`, before clamp and quantization.
pub(super) fn decode_image_device(
    device: &Device,
    blas: &Blas,
    model: &Path,
    latent: &CudaSlice<f32>,
    extent: [usize; 2],
) -> Result<CudaSlice<f32>> {
    let component = model.join("vae_diffusers");
    let config = component.join("config.json");
    let spec = crate::minimax_h3::ImageVaeSpec::parse(&serde_json::from_slice(
        &std::fs::read(&config).with_context(|| format!("read {}", config.display()))?,
    )?)?;
    let checkpoint =
        crate::checkpoint::MappedCheckpoint::open_component(&component, "diffusion_pytorch_model")?;
    let manifest = checkpoint.minimax_h3_image_vae_manifest(&spec)?;
    let canvas = [
        extent[0]
            .checked_mul(spec.spatial_ratio)
            .context("image VAE canvas overflow")?,
        extent[1]
            .checked_mul(spec.spatial_ratio)
            .context("image VAE canvas overflow")?,
    ];
    let elements = extent[0]
        .checked_mul(extent[1])
        .and_then(|n| n.checked_mul(spec.latent_channels))
        .context("image VAE input extent overflow")?;
    ensure!(
        extent.into_iter().all(|n| n > 0) && latent.len() == elements,
        "image VAE normalized latent extent mismatch"
    );
    let decoder = ImageDecoder::new(device, spec.single_frame(), &component, &manifest)?;
    let (free, total) = device.ctx.mem_get_info()?;
    let budget = device.workspace_budget(free, free, total / 16)?;
    let work = decoder.work(device, canvas, budget)?;
    decoder.run(&device.stream, blas, work, latent, |_, _, _, _| Ok(()))
}

/// F32 RGB `[height,width,3]` and F32 posterior noise `[1,24,height/16,width/16]` to cropped normalized latents.
pub(super) fn encode_image_device(
    device: &Device,
    blas: &Blas,
    model: &Path,
    rgb: &CudaSlice<f32>,
    extent: [usize; 2],
    epsilon: &CudaSlice<f32>,
) -> Result<(CudaSlice<f32>, [usize; 2])> {
    let component = model.join("vae_diffusers");
    let published = crate::minimax_h3::ImageVaeSpec::parse(&serde_json::from_slice(
        &std::fs::read(component.join("config.json"))?,
    )?)?;
    let spec = published.single_frame();
    let shape = image_encode_shape(&spec, extent, rgb.len(), epsilon.len())?;
    let checkpoint =
        crate::checkpoint::MappedCheckpoint::open_component(&component, "diffusion_pytorch_model")?;
    let manifest = checkpoint.minimax_h3_image_vae_manifest(&published)?;
    let encoder = KeyframeEncoder::load(device, &spec, &component, &manifest, true)?;
    let tiles = VaeTiles::released(&spec, shape.0[0], shape.0[1])?;
    let (free, total) = device.ctx.mem_get_info()?;
    let budget = device.workspace_budget(free, free, total / 16)?;
    let future = image_encode_future(&spec, &tiles, shape.0, shape.1)?;
    let mut work = EncodeWork::with_budget(
        &device.stream,
        &encoder,
        (tiles.height, tiles.width),
        budget
            .checked_sub(future)
            .context("image encoder budget holds no frame buffers")?,
    )?;
    let [_, _, latents] = encoder.encode_frame(
        (&device.stream, blas),
        &mut work,
        &tiles,
        (rgb, extent[0], extent[1]),
        epsilon,
        shape.0,
    )?;
    Ok((latents, shape.1))
}

fn image_encode_shape(
    spec: &VaeSpec,
    extent: [usize; 2],
    rgb: usize,
    noise: usize,
) -> Result<([usize; 2], [usize; 2])> {
    let r = spec.spatial_ratio;
    ensure!(
        extent.into_iter().all(|n| n <= u32::MAX as usize),
        "image encoder extent exceeds kernel indices"
    );
    let input = extent[0]
        .checked_mul(extent[1])
        .and_then(|n| n.checked_mul(spec.in_channels))
        .context("image encoder RGB extent overflow")?;
    let latent = [extent[0] / r, extent[1] / r];
    let expected = latent[0]
        .checked_mul(latent[1])
        .and_then(|n| n.checked_mul(spec.latent_channels))
        .context("image encoder posterior extent overflow")?;
    let output = latent.map(|n| n / 2 * 2);
    ensure!(
        output.into_iter().all(|n| n > 0) && input == rgb && expected == noise,
        "image encoder RGB/posterior shapes differ: {extent:?}, RGB{rgb}/{input}, epsilon{noise}/{expected}"
    );
    Ok((latent.map(|n| n * r), output))
}

fn image_encode_future(
    spec: &VaeSpec,
    tiles: &VaeTiles,
    canvas: [usize; 2],
    output: [usize; 2],
) -> Result<usize> {
    let c = spec.latent_channels as u128;
    let plane = (canvas[0] / spec.spatial_ratio) as u128 * (canvas[1] / spec.spatial_ratio) as u128;
    let tile_plane =
        (tiles.height / spec.spatial_ratio) as u128 * (tiles.width / spec.spatial_ratio) as u128;
    let grid = (2 * (tiles.rows.0.len() + tiles.cols.0.len()) - 2) as u128;
    Ok(usize::try_from(
        4 * (grid
            + tiles.count() as u128 * 2 * c * tile_plane
            + 4 * c * plane
            + c * output[0] as u128 * output[1] as u128),
    )?)
}

#[cfg(test)]
mod image_tests {
    use super::*;
    use crate::backend::cuda::{closure::Closure, ops::Ops};
    use crate::checkpoint::MappedCheckpoint;
    use safetensors::{Dtype, SafeTensors};
    use std::{collections::BTreeMap, fs, path::PathBuf, rc::Rc};

    fn geometry() -> Result<VaeSpec> {
        Ok(
            crate::minimax_h3::ImageVaeSpec::parse(&serde_json::from_str(include_str!(
                "../../../tests/data/minimax-h3-image-vae-config.json"
            ))?)?
            .single_frame(),
        )
    }

    #[test]
    fn image_encoder_shapes_keep_noise_before_patch_crop() -> Result<()> {
        let spec = geometry()?;
        assert_eq!(
            image_encode_shape(&spec, [270, 286], 270 * 286 * 3, 24 * 16 * 17)?,
            ([256, 272], [16, 16])
        );
        assert!(image_encode_shape(&spec, [270, 286], 270 * 286 * 3, 24 * 16 * 16).is_err());
        assert_eq!(
            image_encode_shape(&spec, [32, 32], 32 * 32 * 3, 24 * 2 * 2)?,
            ([32, 32], [2, 2])
        );
        assert!(image_encode_shape(&spec, [31, 32], 31 * 32 * 3, 24 * 2).is_err());
        assert!(image_encode_shape(&spec, [usize::MAX, 32], 0, 0).is_err());
        Ok(())
    }

    #[test]
    fn image_workspace_respects_memory_and_launch_limits() -> Result<()> {
        let spec = geometry()?;
        let tiles = VaeTiles::released(&spec, 4096, 4096)?;
        let limits = |budget, grid| ImageLimits {
            budget,
            l2: 75_497_472,
            sms: 128,
            grid,
        };
        let plan = ImagePlan::derive(
            &spec,
            &tiles,
            [4096, 4096],
            limits(16 << 30, i32::MAX as usize),
        )?;
        assert_eq!((plan.linear, plan.attention, plan.tokens), (441, 8, 261));
        assert_eq!(
            flat(plan.linear * plan.tokens * spec.ffn).grid_dim.0,
            1_841_616
        );
        let serial_bytes = 775_026_440;
        assert!(
            ImagePlan::derive(
                &spec,
                &tiles,
                [4096, 4096],
                limits(serial_bytes - 1, i32::MAX as usize)
            )
            .is_err()
        );
        let serial = ImagePlan::derive(
            &spec,
            &tiles,
            [4096, 4096],
            limits(serial_bytes, i32::MAX as usize),
        )?;
        assert_eq!(
            (serial.linear, serial.attention, serial.bytes),
            (1, 1, serial_bytes)
        );
        assert!(ImagePlan::derive(&spec, &tiles, [4096, 4096], limits(16 << 30, 1)).is_err());
        let bounded = ImagePlan::derive(&spec, &tiles, [4096, 4096], limits(16 << 30, 65_535))?;
        assert!(
            bounded.linear < plan.linear
                && flat(bounded.linear * bounded.tokens * spec.ffn).grid_dim.0 <= 65_535
        );
        Ok(())
    }

    fn tensor(file: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
        let value = file.tensor(name)?;
        ensure!(
            value.dtype() == Dtype::F32,
            "RUNNER_FAIL {name} fixture dtype"
        );
        Ok(value
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }

    fn score(
        label: &str,
        actual: &[f32],
        reference: &[f32],
        threshold: &serde_json::Value,
    ) -> Result<bool> {
        ensure!(
            !actual.is_empty() && actual.len() == reference.len(),
            "RUNNER_FAIL {label} lengths"
        );
        ensure!(
            actual.iter().chain(reference).all(|x| x.is_finite()),
            "RUNNER_FAIL {label} nonfinite"
        );
        let (mut max_abs, mut squares) = (0f64, 0f64);
        for (&a, &b) in actual.iter().zip(reference) {
            let delta = f64::from(a) - f64::from(b);
            max_abs = max_abs.max(delta.abs());
            squares += delta * delta;
        }
        let rmse = (squares / actual.len() as f64).sqrt();
        let max_tau = threshold["max_abs"]
            .as_f64()
            .context("RUNNER_FAIL missing max_abs tau")?;
        let rmse_tau = threshold["rmse"]
            .as_f64()
            .context("RUNNER_FAIL missing rmse tau")?;
        ensure!(
            max_tau.is_finite() && rmse_tau.is_finite() && max_tau > 0.0 && rmse_tau > 0.0,
            "RUNNER_FAIL invalid taus"
        );
        let defect = max_abs > max_tau || rmse > rmse_tau;
        println!(
            "IMAGE_VAE_CELL {label} max_abs={max_abs:.10e} rmse={rmse:.10e} tau_max={max_tau:.10e} tau_rmse={rmse_tau:.10e} status={}",
            if defect { "DEFECT" } else { "PASS" }
        );
        Ok(defect)
    }

    fn path(key: &str) -> Result<PathBuf> {
        Ok(PathBuf::from(
            std::env::var_os(key).with_context(|| format!("{key} missing"))?,
        ))
    }

    fn input(root: &Path, name: &str, shape: [usize; 2]) -> Result<Vec<f32>> {
        let bytes = fs::read(root.join(format!("inputs/decode-{name}.safetensors")))?;
        let file = SafeTensors::deserialize(&bytes)?;
        ensure!(
            file.tensor("normalized_latent")?.shape() == [1, 24, shape[0] / 16, shape[1] / 16],
            "RUNNER_FAIL {name} input shape"
        );
        tensor(&file, "normalized_latent")
    }

    fn save_taps(
        path: &Path,
        taps: &BTreeMap<String, Vec<f32>>,
        reference: &SafeTensors<'_>,
    ) -> Result<()> {
        let bytes: BTreeMap<_, Vec<u8>> = taps
            .iter()
            .filter(|(name, _)| name.as_str() != "rgb")
            .map(|(name, values)| {
                (
                    name.as_str(),
                    values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect(),
                )
            })
            .collect();
        let views: BTreeMap<_, _> = bytes
            .iter()
            .map(|(name, data)| {
                Ok((
                    *name,
                    safetensors::tensor::TensorView::new(
                        Dtype::F32,
                        reference.tensor(name)?.shape().to_vec(),
                        data,
                    )?,
                ))
            })
            .collect::<Result<_>>()?;
        safetensors::tensor::serialize_to_file(views, None, path)?;
        Ok(())
    }

    fn reference_gate(root: &Path, model: &Path, runtime: &Path, capture: &Path) -> Result<()> {
        let spec = geometry()?;
        let component = model.join("vae_diffusers");
        let published = crate::minimax_h3::ImageVaeSpec::parse(&serde_json::from_slice(
            &fs::read(component.join("config.json"))?,
        )?)?;
        ensure!(
            published.single_frame() == spec,
            "RUNNER_FAIL published VAE differs from the frozen geometry"
        );
        let checkpoint = MappedCheckpoint::open_component(&component, "diffusion_pytorch_model")?;
        let manifest = checkpoint.minimax_h3_image_vae_manifest(&published)?;
        ensure!(
            manifest.entries.len() == 511,
            "RUNNER_FAIL full VAE tensor census"
        );
        let frozen: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("FROZEN_DECODER_FLOORS.json"))?)?;
        ensure!(
            frozen["formula"] == "max(2*worst_valid_pair,1e-6)",
            "RUNNER_FAIL decoder floor formula"
        );
        let device = Device::new(0)?;
        let sm = cubin::capability(&device.ctx)?;
        ensure!(
            [86, 89].contains(&sm),
            "RUNNER_FAIL VAE fixture has no sm{sm} arm"
        );
        let s = &device.stream;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let blas = Blas::new(
            &device,
            &ops,
            runtime,
            Rc::new(Closure::new(true, "image VAE gate".into())),
        )?;
        let mut decoder = ImageDecoder::new(&device, spec.clone(), &component, &manifest)?;
        let mut defects = 0;
        let mut cells = 0;
        for (name, canvas) in [
            ("small", [128, 192]),
            ("overlap", [272, 304]),
            ("2048", [2048, 2048]),
            ("4096", [4096, 4096]),
        ] {
            let z = s.clone_htod(&input(root, name, canvas)?)?;
            let expected_bytes = fs::read(root.join(format!(
                "reference-sm{sm}/output/decode-{name}-canonical0.safetensors"
            )))?;
            let expected = SafeTensors::deserialize(&expected_bytes)?;
            for low in [false, true] {
                let (free, total) = device.ctx.mem_get_info()?;
                let live = device.workspace_budget(free, free, total / 16)?;
                let budget = if low {
                    let tiles = VaeTiles::released(&spec, canvas[0], canvas[1])?;
                    let tokens = tiles.height / spec.spatial_ratio
                        * (tiles.width / spec.spatial_ratio)
                        + spec.registers
                        + 1;
                    let (p, score) = (tokens * spec.dim * 4, tokens * tokens * spec.heads * 4);
                    let base = tiles.count() * spec.out_channels * tiles.height * tiles.width * 4
                        + canvas[0] * canvas[1] * spec.out_channels * 8
                        + 4 * (2 * (tiles.rows.0.len() + tiles.cols.0.len()) - 2
                            + tokens * spec.rope_dims
                            + spec.head_dim);
                    let per_tile = 19 * p
                        + 4 * tokens * spec.patch_width()
                        + 8 * (tokens - spec.registers - 1) * spec.latent_channels;
                    live.min(base + per_tile + score)
                } else {
                    live
                };
                let work = decoder.work(&device, canvas, budget)?;
                if low {
                    ensure!(
                        work.plan.linear == 1 && work.plan.attention == 1,
                        "RUNNER_FAIL serial budget did not force serial tiles"
                    );
                }
                let selected: std::collections::BTreeSet<_> =
                    [0, work.tiles.count() / 2, work.tiles.count() - 1]
                        .into_iter()
                        .collect();
                let mut taps = BTreeMap::new();
                decoder.launch(
                    s,
                    &blas,
                    &work,
                    &z,
                    |role, first, active, (data, offset, rows, width)| {
                        let role = if role == "block_last" {
                            "block23"
                        } else {
                            role
                        };
                        for &tile in &selected {
                            if (first..first + active).contains(&tile) {
                                let start = offset + (tile - first) * rows * width;
                                let value =
                                    s.clone_dtoh(&data.slice(start..start + rows * width))?;
                                ensure!(
                                    taps.insert(format!("{role}.tile{tile}"), value).is_none(),
                                    "RUNNER_FAIL duplicate tap"
                                );
                            }
                        }
                        Ok(())
                    },
                )?;
                taps.insert("rgb".into(), s.clone_dtoh(&work.rgb)?);
                ensure!(
                    taps.len() == expected.names().len(),
                    "RUNNER_FAIL missing boundary tap"
                );
                for (key, actual) in &taps {
                    defects += usize::from(score(
                        &format!("case={name} low={low} boundary={key}"),
                        actual,
                        &tensor(&expected, key)?,
                        &frozen["cases"][name][key],
                    )?);
                    cells += 1;
                }
                save_taps(
                    &capture.join(format!("{name}-{low}-{}.safetensors", std::process::id())),
                    &taps,
                    &expected,
                )?;
                if name == "2048" && !low {
                    s.begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                    let launched = decoder.launch(s, &blas, &work, &z, |_, _, _, _| Ok(()));
                    let graph = s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
                    launched?;
                    graph?
                        .context("RUNNER_FAIL image VAE graph empty")?
                        .launch()?;
                    let actual = s.clone_dtoh(&work.rgb)?;
                    let bits = actual
                        .iter()
                        .zip(&taps["rgb"])
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    ensure!(
                        actual.iter().all(|v| v.is_finite()),
                        "RUNNER_FAIL image VAE graph nonfinite"
                    );
                    println!(
                        "IMAGE_VAE_GRAPH case=2048 linear={} attention={} tail={} bit_differences={bits} status={}",
                        work.plan.linear,
                        work.plan.attention,
                        work.tiles.count() % work.plan.attention,
                        if bits == 0 { "PASS" } else { "DEFECT" }
                    );
                    defects += usize::from(bits != 0);
                }
            }
        }
        let mut mean = spec.latents_mean.clone();
        mean.rotate_right(1);
        let mut std = spec.latents_std.clone();
        std.rotate_right(1);
        decoder.latent_affine = (s.clone_htod(&std)?, s.clone_htod(&mean)?);
        let z = s.clone_htod(&input(root, "2048", [2048, 2048])?)?;
        let (free, total) = device.ctx.mem_get_info()?;
        let work = decoder.work(
            &device,
            [2048, 2048],
            device.workspace_budget(free, free, total / 16)?,
        )?;
        let control = decoder.run(s, &blas, work, &z, |_, _, _, _| Ok(()))?;
        let bytes = fs::read(root.join(format!(
            "reference-sm{sm}/output/decode-2048-canonical0.safetensors"
        )))?;
        let reference = SafeTensors::deserialize(&bytes)?;
        let detected = score(
            "control=shift24",
            &s.clone_dtoh(&control)?,
            &tensor(&reference, "rgb")?,
            &frozen["cases"]["2048"]["rgb"],
        )?;
        println!(
            "IMAGE_VAE_CONTROL shift24_detected={detected} status={}",
            if detected { "PASS" } else { "FAIL" }
        );
        ensure!(detected, "RUNNER_FAIL image VAE control not detected");
        println!(
            "{} IMAGE_VAE_DECODER cells={cells} graph_cells=1 control=FAIL_AS_REQUIRED defects={defects}",
            if defects == 0 { "PASS" } else { "FAIL" }
        );
        ensure!(
            defects == 0,
            "image VAE decoder has {defects} finite numerical defects"
        );
        Ok(())
    }

    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted, FF_MODEL, FF_H3_S0 frozen references, FF_H3_CAPTURE output"]
    fn image_decoder_matches_the_frozen_reference() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let root = path("FF_H3_S0")?;
        let model = path("FF_MODEL")?;
        let capture = path("FF_H3_CAPTURE")?;
        fs::create_dir_all(&capture)?;
        let runtime = std::env::temp_dir().join(format!("ff-image-vae-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let result = reference_gate(&root, &model, &runtime, &capture);
        fs::remove_dir_all(&runtime)?;
        result
    }

    fn encoder_layouts(device: &Device, encoder: &KeyframeEncoder) -> Result<usize> {
        let s = &device.stream;
        let rgb: Vec<f32> = (0..60).map(|i| ((i * 37) % 251) as f32 / 8.0).collect();
        let pixels = s.clone_htod(&rgb)?;
        let norm = s.clone_htod(&vec![-12345.5f32; 18 + 64])?;
        let plane = 16 * 17;
        let count = 24 * plane;
        let moments = s.clone_htod(&vec![0f32; 2 * count])?;
        let noise: Vec<f32> = (0..count)
            .map(|i| ((i * 37) % 251) as f32 / 137.0 - 1.0)
            .collect();
        let eps = s.clone_htod(&noise)?;
        let mean = s.clone_htod(&vec![-12345.5f32; count + 64])?;
        let std = s.clone_htod(&vec![-12345.5f32; count + 64])?;
        let latent = s.clone_htod(&vec![-12345.5f32; 24 * 256 + 64])?;
        let launch = || -> Result<()> {
            unsafe {
                s.launch_builder(&encoder.kernels.pixels)
                    .arg(&pointer(&pixels, s))
                    .arg(&pointer(&encoder.pixel_table, s))
                    .arg(&(pointer(&norm, s) + 128))
                    .arg(&3u32)
                    .arg(&4u32)
                    .arg(&5u32)
                    .arg(&1u32)
                    .arg(&1u32)
                    .arg(&2u32)
                    .arg(&3u32)
                    .launch(flat(18))?;
                s.launch_builder(&encoder.kernels.posterior)
                    .arg(&pointer(&moments, s))
                    .arg(&pointer(&eps, s))
                    .arg(&pointer(&encoder.latent_table, s))
                    .arg(&(pointer(&mean, s) + 128))
                    .arg(&(pointer(&std, s) + 128))
                    .arg(&(pointer(&latent, s) + 128))
                    .arg(&24u32)
                    .arg(&16u32)
                    .arg(&17u32)
                    .arg(&16u32)
                    .arg(&16u32)
                    .launch(flat(count))?;
            }
            Ok(())
        };
        launch()?;
        let eager = [&norm, &mean, &std, &latent]
            .into_iter()
            .map(|b| s.clone_dtoh(b))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for values in &eager {
            ensure!(
                values.iter().all(|v| v.is_finite()),
                "RUNNER_FAIL encoder layout nonfinite"
            );
            ensure!(
                values[..32]
                    .iter()
                    .chain(&values[values.len() - 32..])
                    .all(|v| *v == -12345.5),
                "RUNNER_FAIL encoder layout guard overwritten"
            );
        }
        let mut fractional_bits = 0usize;
        for y in 0..2 {
            for x in 0..3 {
                for c in 0..3 {
                    let v = rgb[((y + 1) * 5 + x + 1) * 3 + c] * (1f32 / 255.0);
                    let expected = (v - VAE_PIXEL_MEAN[c] as f32) / VAE_PIXEL_STD[c] as f32;
                    fractional_bits += (eager[0][32 + (y * 3 + x) * 3 + c].to_bits()
                        ^ expected.to_bits())
                    .count_ones() as usize;
                }
            }
        }
        let moment_errors = eager[1][32..32 + count]
            .iter()
            .filter(|v| **v != 0.0)
            .count()
            + eager[2][32..32 + count]
                .iter()
                .filter(|v| **v != 1.0)
                .count();
        let mut crop_bits = 0usize;
        for c in 0..24 {
            for y in 0..16 {
                for x in 0..16 {
                    let rounded = half::f16::from_f32(noise[(c * 16 + y) * 17 + x]).to_f32();
                    let expected =
                        (rounded - encoder.spec.latents_mean[c]) / encoder.spec.latents_std[c];
                    crop_bits += (eager[3][32 + (c * 16 + y) * 16 + x].to_bits()
                        ^ expected.to_bits())
                    .count_ones() as usize;
                }
            }
        }
        s.begin_capture(
            cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )?;
        let launched = launch();
        let graph = s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
        launched?;
        graph?
            .context("RUNNER_FAIL encoder layout graph empty")?
            .launch()?;
        let mut graph_bits = 0usize;
        for (buffer, before) in [&norm, &mean, &std, &latent].into_iter().zip(eager) {
            let after = s.clone_dtoh(buffer)?;
            ensure!(
                after.iter().all(|v| v.is_finite()),
                "RUNNER_FAIL encoder layout graph nonfinite"
            );
            graph_bits += after
                .iter()
                .zip(before)
                .map(|(a, b)| (a.to_bits() ^ b.to_bits()).count_ones() as usize)
                .sum::<usize>();
        }
        let defects = [fractional_bits, moment_errors, crop_bits, graph_bits]
            .into_iter()
            .filter(|v| *v != 0)
            .count();
        println!(
            "{} IMAGE_VAE_ENCODER_LAYOUTS exports=2 fractional_rgb_bits={fractional_bits} mean_std_bad_values={moment_errors} posterior_crop_bits={crop_bits} graph_eager_bits={graph_bits} guards=true",
            if defects == 0 { "PASS" } else { "DEFECT" }
        );
        Ok(defects)
    }

    fn encoder_reference_gate(
        root: &Path,
        model: &Path,
        runtime: &Path,
        capture: &Path,
    ) -> Result<()> {
        use cudarc::driver::sys::CUdevice_attribute as A;
        let spec = geometry()?;
        let component = model.join("vae_diffusers");
        let published = crate::minimax_h3::ImageVaeSpec::parse(&serde_json::from_slice(
            &fs::read(component.join("config.json"))?,
        )?)?;
        ensure!(
            published.single_frame() == spec,
            "RUNNER_FAIL encoder published geometry differs"
        );
        let checkpoint = MappedCheckpoint::open_component(&component, "diffusion_pytorch_model")?;
        let manifest = checkpoint.minimax_h3_image_vae_manifest(&published)?;
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        device.finish_upload()?;
        let blas = Blas::new(
            &device,
            &ops,
            runtime,
            Rc::new(Closure::new(true, "image VAE encoder gate".into())),
        )?;
        let encoder = KeyframeEncoder::load(&device, &spec, &component, &manifest, true)?;
        let s = &device.stream;
        let sm = device
            .ctx
            .attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?
            * 10
            + device
                .ctx
                .attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        ensure!(
            [86, 89].contains(&sm),
            "RUNNER_FAIL encoder reference missing for sm{sm}"
        );
        let frozen: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("FROZEN_ENCODER_FLOORS.json"))?)?;
        let mut defects = encoder_layouts(&device, &encoder)?;
        let mut cells = 0;
        for (name, extent) in [
            ("edit0", [1536, 1024]),
            ("edit1", [1536, 1024]),
            ("crop", [270, 286]),
            ("landscape0", [1024, 1536]),
            ("landscape1", [1024, 1536]),
        ] {
            let input_bytes =
                fs::read(root.join(format!("encoder-inputs/encode-{name}.safetensors")))?;
            let input = SafeTensors::deserialize(&input_bytes)?;
            ensure!(
                input.tensor("rgb")?.shape() == [extent[0], extent[1], 3],
                "RUNNER_FAIL encoder RGB fixture shape"
            );
            let rgb = tensor(&input, "rgb")?;
            let noise = tensor(&input, "epsilon")?;
            let shape = image_encode_shape(&spec, extent, rgb.len(), noise.len())?;
            let pixels = s.clone_htod(&rgb)?;
            let eps = s.clone_htod(&noise)?;
            let tiles = VaeTiles::released(&spec, shape.0[0], shape.0[1])?;
            let expected_bytes = fs::read(root.join(format!(
                "reference-sm{sm}-cudnn/output/encode-{name}-canonical0.safetensors"
            )))?;
            let expected = SafeTensors::deserialize(&expected_bytes)?;
            for low in [false, true] {
                let (free, total) = device.ctx.mem_get_info()?;
                let available = device
                    .workspace_budget(free, free, total / 16)?
                    .checked_sub(image_encode_future(&spec, &tiles, shape.0, shape.1)?)
                    .context("RUNNER_FAIL encoder future memory budget")?;
                let budget = if low {
                    available.min(167_772_160 + (64 << 20))
                } else {
                    available
                };
                let mut work =
                    EncodeWork::with_budget(s, &encoder, (tiles.height, tiles.width), budget)?;
                ensure!(
                    !low || work.columns.len() * 4 <= 64 << 20,
                    "RUNNER_FAIL low encoder budget not forced"
                );
                ensure!(
                    EncodeWork::with_budget(
                        s,
                        &encoder,
                        (tiles.height, tiles.width),
                        5 * work.x.len() * 4 - 1
                    )
                    .is_err(),
                    "RUNNER_FAIL encoder accepted insufficient activation budget"
                );
                let outputs = encoder.encode_frame(
                    (s, &blas),
                    &mut work,
                    &tiles,
                    (&pixels, extent[0], extent[1]),
                    &eps,
                    shape.0,
                )?;
                let mut taps = BTreeMap::new();
                for (role, buffer) in ["mean", "std", "normalized_latent"]
                    .into_iter()
                    .zip(&outputs)
                {
                    let actual = s.clone_dtoh(buffer)?;
                    defects += usize::from(score(
                        &format!("encoder case={name} low={low} boundary={role}"),
                        &actual,
                        &tensor(&expected, role)?,
                        &frozen["cases"][name][role],
                    )?);
                    cells += 1;
                    taps.insert(role.to_owned(), actual);
                }
                save_taps(
                    &capture.join(format!(
                        "encoder-{name}-{low}-{}.safetensors",
                        std::process::id()
                    )),
                    &taps,
                    &expected,
                )?;
                println!(
                    "IMAGE_VAE_ENCODER_PLAN case={name} low={low} tiles={} column_bytes={} output={:?}",
                    tiles.count(),
                    4 * work.columns.len(),
                    shape.1
                );
            }
            if name == "edit0" {
                let eps = s.clone_htod(&vec![0f32; noise.len()])?;
                let (free, total) = device.ctx.mem_get_info()?;
                let budget = device
                    .workspace_budget(free, free, total / 16)?
                    .checked_sub(image_encode_future(&spec, &tiles, shape.0, shape.1)?)
                    .context("RUNNER_FAIL control budget")?;
                let mut work =
                    EncodeWork::with_budget(s, &encoder, (tiles.height, tiles.width), budget)?;
                let [_, _, out] = encoder.encode_frame(
                    (s, &blas),
                    &mut work,
                    &tiles,
                    (&pixels, extent[0], extent[1]),
                    &eps,
                    shape.0,
                )?;
                let detected = score(
                    "encoder control=posterior_mode",
                    &s.clone_dtoh(&out)?,
                    &tensor(&expected, "normalized_latent")?,
                    &frozen["cases"][name]["normalized_latent"],
                )?;
                ensure!(detected, "RUNNER_FAIL posterior-mode control not detected");
                println!("IMAGE_VAE_ENCODER_CONTROL posterior_mode_detected=true status=PASS");
            }
        }
        println!(
            "{} IMAGE_VAE_ENCODER cells={cells} modes=2 control=FAIL_AS_REQUIRED defects={defects}",
            if defects == 0 { "PASS" } else { "FAIL" }
        );
        ensure!(
            defects == 0,
            "image VAE encoder has {defects} finite numerical defects"
        );
        Ok(())
    }

    #[test]
    #[ignore = "GPU; FF_Q38_GPU_WINDOW=granted, FF_MODEL, FF_H3_S0 encoder references, FF_H3_CAPTURE output"]
    fn image_encoder_matches_the_frozen_reference() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let root = path("FF_H3_S0")?;
        let model = path("FF_MODEL")?;
        let capture = path("FF_H3_CAPTURE")?;
        fs::create_dir_all(&capture)?;
        let runtime =
            std::env::temp_dir().join(format!("ff-image-vae-encoder-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let result = encoder_reference_gate(&root, &model, &runtime, &capture);
        fs::remove_dir_all(&runtime)?;
        result
    }
}

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

pub(crate) fn planar_bf16_to_rgb_u8(
    device: &Device,
    kernel: &CudaFunction,
    input: &impl DevicePtr<bf16>,
    output: &impl DevicePtr<u8>,
    (frames, allocated_frames, pixels): (usize, usize, usize),
    (mean, std): ([f32; 3], [f32; 3]),
) -> Result<()> {
    ensure!(
        frames > 0 && frames <= allocated_frames && pixels > 0,
        "RGB extent frames={frames} allocated_frames={allocated_frames} pixels={pixels}"
    );
    let plane = allocated_frames
        .checked_mul(pixels)
        .context("RGB plane overflow")?;
    let elements = frames
        .checked_mul(pixels)
        .context("RGB pixel extent overflow")?;
    let input_elements = plane.checked_mul(3).context("RGB input extent overflow")?;
    let output_elements = elements
        .checked_mul(3)
        .context("RGB output extent overflow")?;
    let input_bytes = input_elements
        .checked_mul(2)
        .context("RGB input byte extent overflow")?;
    ensure!(
        input.len() >= input_elements && output.len() >= output_elements,
        "RGB needs input {input_elements}/{} and output {output_elements}/{} elements",
        input.len(),
        output.len()
    );
    ensure!(
        mean.into_iter().chain(std).all(f32::is_finite),
        "RGB affine parameters are nonfinite"
    );
    use cudarc::driver::sys::CUdevice_attribute as A;
    let active = kernel.occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?;
    let sms = u32::try_from(
        device
            .ctx
            .attribute(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
    )?;
    let grid_limit = u32::try_from(
        device
            .ctx
            .attribute(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
    )?;
    let resident = active
        .checked_mul(sms)
        .context("RGB resident grid overflow")?;
    ensure!(resident > 0 && grid_limit > 0, "RGB has no resident blocks");
    let grid = u32::try_from(
        elements
            .div_ceil(256)
            .min(resident.min(grid_limit) as usize),
    )?;
    let s = &device.stream;
    let source = input.device_ptr(s).0;
    let target = output.device_ptr(s).0;
    let source_end = source
        .checked_add(input_bytes as u64)
        .context("RGB source address overflow")?;
    let target_end = target
        .checked_add(output_elements as u64)
        .context("RGB target address overflow")?;
    ensure!(
        source_end <= target || target_end <= source,
        "RGB input/output overlap"
    );
    unsafe {
        s.launch_builder(kernel)
            .arg(&source)
            .arg(&target)
            .arg(&(allocated_frames as u64))
            .arg(&(frames as u64))
            .arg(&(pixels as u64))
            .arg(&mean[0])
            .arg(&mean[1])
            .arg(&mean[2])
            .arg(&std[0])
            .arg(&std[1])
            .arg(&std[2])
            .launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    if crate::backend::diagnostics_enabled() {
        eprintln!(
            "{}",
            serde_json::json!({"h3_rgb_launch":{"frames":frames,"allocated_frames":allocated_frames,
            "pixels":pixels,"input_bytes":input_bytes,"output_bytes":output_elements,
            "grid":grid,"threads":256,"active_blocks_per_sm":active,"sms":sms}})
        );
    }
    Ok(())
}

/// A biased BF16 projection `[out, input]` cast from the published F32 tensor at load.
struct Linear<T: crate::backend::gpu::DeviceRepr = bf16> {
    weight: CudaSlice<T>,
    bias: CudaSlice<T>,
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
        x: Tensor<bf16, 2>,
        y: Tensor<bf16, 2>,
    ) -> Result<()> {
        blas.linear_biased(
            matrix(&self.weight, s, self.out, self.input)?,
            vector(&self.bias, s)?,
            x,
            &mut TensorMut(y),
            MODE,
        )
    }
}

struct DecoderBlock<T: crate::backend::gpu::DeviceRepr = bf16> {
    q: Linear<T>,
    k: Linear<T>,
    v: Linear<T>,
    out: Linear<T>,
    ff_in: Linear<T>,
    ff_out: Linear<T>,
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
    rgb: CudaFunction,
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
        Source::fill::<f32>(
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

#[derive(Clone)]
struct TileBatch {
    batch: usize,
    work: usize,
    workspace: usize,
    peak: usize,
    cost: f64,
    launches: usize,
}

impl TileBatch {
    fn choose(choices: &[Self], budget: usize) -> Result<&Self> {
        ensure!(
            choices.iter().any(|c| c.batch == 1 && c.peak <= budget),
            "minimum serial tile work does not fit workspace budget {budget}"
        );
        choices
            .iter()
            .filter(|c| c.peak <= budget)
            .min_by(|a, b| {
                a.cost
                    .total_cmp(&b.cost)
                    .then(a.launches.cmp(&b.launches))
                    .then(a.peak.cmp(&b.peak))
            })
            .context("no legal tile batch")
    }
}

#[cfg(test)]
fn tile_bit_differences<T: super::Element + Copy>(
    s: &Arc<CudaStream>,
    name: &str,
    serial: &CudaSlice<T>,
    grouped: &CudaSlice<T>,
    (tile, rows, columns, heads): (usize, usize, usize, usize),
) -> Result<usize> {
    let capacity = grouped.len() / (columns * heads);
    let mut differences = 0;
    for head in 0..heads {
        let a = head * rows * columns;
        let b = (head * capacity + tile * rows) * columns;
        let expected = s.clone_dtoh(&serial.slice(a..a + rows * columns))?;
        let actual = s.clone_dtoh(&grouped.slice(b..b + rows * columns))?;
        let bytes = rows * columns * std::mem::size_of::<T>();
        // Element implementations are BF16 and F32 scalars.
        let (expected, actual) = unsafe {
            (
                std::slice::from_raw_parts(expected.as_ptr().cast::<u8>(), bytes),
                std::slice::from_raw_parts(actual.as_ptr().cast::<u8>(), bytes),
            )
        };
        ensure!(
            T::finite(expected) && T::finite(actual),
            "RUNNER_FAIL {name} nonfinite tile={tile}"
        );
        differences += expected
            .chunks_exact(std::mem::size_of::<T>())
            .zip(actual.chunks_exact(std::mem::size_of::<T>()))
            .filter(|(a, b)| a != b)
            .count();
    }
    println!(
        "H3_VAE_GROUP_OP name={name} tile={tile} bit_differences={differences} status={}",
        if differences == 0 { "PASS" } else { "DEFECT" }
    );
    Ok(usize::from(differences != 0))
}

#[test]
#[cfg(test)]
fn tile_batch_budget_keeps_serial_minimum() -> Result<()> {
    let choices = [
        TileBatch {
            batch: 1,
            work: 210_349_280,
            workspace: 0,
            peak: 210_349_280,
            cost: 2.0,
            launches: 28,
        },
        TileBatch {
            batch: 3,
            work: 630_933_552,
            workspace: 0,
            peak: 630_933_552,
            cost: 1.0,
            launches: 10,
        },
    ];
    assert_eq!(TileBatch::choose(&choices, 210_349_280)?.batch, 1);
    assert!(TileBatch::choose(&choices, 210_349_279).is_err());
    assert_eq!(TileBatch::choose(&choices, 630_933_552)?.batch, 3);
    assert!(TileBatch::choose(&choices[1..], 630_933_552).is_err());
    Ok(())
}

impl VideoDecoder {
    pub(crate) fn rgb(
        &self,
        device: &Device,
        decoded: &CudaSlice<bf16>,
        shape: (usize, usize, usize),
    ) -> Result<CudaSlice<u8>> {
        ensure!(
            self.spec.out_channels == 3,
            "RGB conversion requires three decoder channels"
        );
        ensure!(
            shape.0 > 0 && shape.0 <= shape.1 && shape.2 > 0,
            "invalid RGB conversion shape {shape:?}"
        );
        let bytes = shape
            .0
            .checked_mul(shape.2)
            .and_then(|n| n.checked_mul(3))
            .context("RGB output byte extent overflow")?;
        device.stream.synchronize()?;
        let (free, total) = device.ctx.mem_get_info()?;
        let reserve = total / 16;
        let budget = device.workspace_budget(free, free, reserve)?;
        ensure!(
            bytes <= budget,
            "RGB output {bytes} B exceeds conversion budget {budget} B"
        );
        // Every output byte is written by planar_bf16_to_rgb_u8 before transfer.
        let output = unsafe { device.stream.alloc::<u8>(bytes)? };
        planar_bf16_to_rgb_u8(
            device,
            &self.kernels.rgb,
            decoded,
            &output,
            shape,
            (
                VAE_PIXEL_MEAN.map(|v| v as f32),
                VAE_PIXEL_STD.map(|v| v as f32),
            ),
        )?;
        if crate::backend::diagnostics_enabled() {
            eprintln!(
                "{}",
                serde_json::json!({"h3_rgb_memory":{"free_before_output_bytes":free,
                "total_bytes":total,"safety_reserve_bytes":reserve,"output_bytes":bytes,"workspace_budget_bytes":budget}})
            );
        }
        Ok(output)
    }

    #[cfg(test)]
    pub(crate) fn batch_identity_gate(
        &self,
        device: &Device,
        blas: &mut Blas,
        tiles: &VaeTiles,
        mut inputs: impl Iterator<Item = (Vec<f32>, [usize; 3])>,
        frozen: &serde_json::Value,
        capture: &Path,
    ) -> Result<()> {
        let first = inputs.next().context("RUNNER_FAIL group inputs empty")?;
        let spec = &self.spec;
        let s = &device.stream;
        let r = spec.spatial_ratio;
        let shape = (
            spec.chunk_latents() + spec.overlap_latents(),
            tiles.height / r,
            tiles.width / r,
        );
        let [frames, height, width] = first.1;
        let mut grouped = self.work(device, blas, tiles, (frames, height, width))?;
        ensure!(
            frozen["bit_difference_limit"] == 0
                && frozen["arm_results"]["standalone-A"]["setup"]["chosen"]["batch"]
                    == grouped.batch,
            "RUNNER_FAIL derived group differs from frozen maximum"
        );
        let slots = tiles.count();
        let mut serial = TileWork::new(s, spec, shape, (1, slots))?;
        serial.prepare(s, blas, self)?;
        let mut lower = TileWork::new(s, spec, shape, (3, slots))?;
        lower.prepare(s, blas, self)?;
        self.group_op_gate(s, blas, &mut serial, &mut grouped)?;
        let origins: Vec<_> = tiles
            .rows
            .0
            .iter()
            .flat_map(|&row| tiles.cols.0.iter().map(move |&col| (row / r, col / r)))
            .collect();
        ensure!(
            origins.len() == 28,
            "RUNNER_FAIL group gate requires production spatial grid"
        );
        let mut expected_raw = s.alloc_zeros::<bf16>(slots * serial.pixels())?;
        let mut actual_raw = s.alloc_zeros::<bf16>(expected_raw.len())?;
        let mut previous: Option<Vec<bf16>> = None;
        let mut defects = 0;
        let mut passes = 0;
        let mut video_cells = 0;
        let mut video_defects = 0;
        for (input, [frames, height, width]) in std::iter::once(first).chain(inputs) {
            ensure!(
                input.len() == spec.latent_channels * frames * height * width
                    && input.iter().all(|x| x.is_finite()),
                "RUNNER_FAIL group latent input layout/nonfinite"
            );
            let input = s.clone_htod(&input)?;
            let z = LatentVideo {
                data: &input,
                frames,
                height,
                width,
            };
            s.memset_zeros(&mut expected_raw)?;
            for (slot, origin) in origins.iter().enumerate() {
                self.tile(
                    s,
                    blas,
                    &mut serial,
                    &z,
                    (std::slice::from_ref(origin), 0),
                    (&expected_raw, slot),
                )?;
            }
            let expected = s.clone_dtoh(&expected_raw)?;
            ensure!(
                expected.iter().all(|x| x.is_finite()),
                "RUNNER_FAIL serial group output nonfinite"
            );
            if let Some(previous) = &previous {
                ensure!(
                    previous
                        .iter()
                        .zip(&expected)
                        .any(|(a, b)| a.to_bits() != b.to_bits()),
                    "RUNNER_FAIL repeated group inputs have identical serial outputs"
                );
            }
            for work in [&mut grouped, &mut lower] {
                s.memset_zeros(&mut actual_raw)?;
                for (group, positions) in origins.chunks(work.batch).enumerate() {
                    self.tile(
                        s,
                        blas,
                        work,
                        &z,
                        (positions, 0),
                        (&actual_raw, group * work.batch),
                    )?;
                }
                let actual = s.clone_dtoh(&actual_raw)?;
                ensure!(
                    actual.iter().all(|x| x.is_finite()),
                    "RUNNER_FAIL grouped raw output nonfinite"
                );
                let bits = actual
                    .iter()
                    .zip(&expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                defects += usize::from(bits != 0);
                println!(
                    "H3_VAE_GROUP_FORWARD input={passes} batch={} tail={} mode=eager tiles={slots} raw_bit_differences={bits} status={}",
                    work.batch,
                    work.tail,
                    if bits == 0 { "PASS" } else { "DEFECT" }
                );
            }
            let plan = DecodePlan::new(spec, frames)?;
            let pre = spec.frame_pre_padding();
            let lead = spec.chunk_latents() * spec.temporal_ratio;
            let kept = lead - pre;
            let tail = shape.0 * spec.temporal_ratio - lead - pre;
            let total = plan.chunks * kept + if spec.token_drop > 0 { tail } else { 0 };
            let plane = height * r * width * r;
            let directory = capture.join(format!("video-{passes}"));
            std::fs::create_dir(&directory)?;
            for work in [&mut serial, &mut grouped, &mut lower] {
                let batch = work.batch;
                let mut output = s.alloc_zeros::<bf16>(spec.out_channels * total * plane)?;
                self.decode_observed(
                    (s, blas),
                    work,
                    tiles,
                    LatentVideo {
                        data: &input,
                        frames,
                        height,
                        width,
                    },
                    &mut output,
                    |stage, chunk, buffer| {
                        let values = if stage == "emitted" {
                            let mut values = Vec::with_capacity(spec.out_channels * kept * plane);
                            for channel in 0..spec.out_channels {
                                let first = (channel * total + chunk * kept) * plane;
                                values.extend(s.clone_dtoh(
                                    &buffer.slice(first..first + kept * plane),
                                )?);
                            }
                            values
                        } else {
                            s.clone_dtoh(buffer)?
                        };
                        ensure!(
                            values.iter().all(|x| x.is_finite()),
                            "RUNNER_FAIL video {stage} nonfinite input={passes} batch={batch} chunk={chunk}"
                        );
                        // BF16 scalars occupy two bytes.
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                values.as_ptr().cast::<u8>(),
                                values.len() * 2,
                            )
                        };
                        let path = directory.join(format!("{stage}-{chunk}.bf16"));
                        if batch == 1 {
                            std::fs::write(path, bytes)?;
                        } else {
                            let expected = std::fs::read(path)?;
                            ensure!(
                                expected.len() == bytes.len(),
                                "RUNNER_FAIL video {stage} snapshot extent"
                            );
                            let bits = bytes
                                .as_chunks::<2>()
                                .0
                                .iter()
                                .zip(expected.as_chunks::<2>().0)
                                .filter(|(a, b)| a != b)
                                .count();
                            video_cells += 1;
                            video_defects += usize::from(bits != 0);
                            println!(
                                "H3_VAE_VIDEO_IDENTITY input={passes} batch={batch} chunk={chunk} stage={stage} elements={} bit_differences={bits} status={}",
                                values.len(),
                                if bits == 0 { "PASS" } else { "DEFECT" }
                            );
                        }
                        Ok(())
                    },
                )?;
                println!(
                    "H3_VAE_VIDEO_FORWARD input={passes} batch={batch} tail={} chunks={} frames={total} tiles={slots}",
                    work.tail, plan.chunks
                );
            }
            std::fs::remove_dir_all(directory)?;
            previous = Some(expected);
            passes += 1;
        }
        ensure!(passes >= 2, "RUNNER_FAIL repeated group inputs missing");
        println!(
            "H3_VAE_GROUP_RESULT inputs={passes} batches=[{},3] bit_limit=0 defects={defects}",
            grouped.batch
        );
        ensure!(defects == 0, "H3_VAE_GROUP_FORWARD DEFECT cells={defects}");
        println!("H3_VAE_GROUP_FORWARD PASS repeated_inputs={passes} bit_limit=0");
        println!(
            "H3_VAE_VIDEO_RESULT inputs={passes} cells={video_cells} bit_limit=0 defects={video_defects}"
        );
        ensure!(
            video_defects == 0,
            "H3_VAE_VIDEO DEFECT cells={video_defects}"
        );
        Ok(())
    }

    #[cfg(test)]
    fn group_op_gate(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        serial: &mut TileWork,
        grouped: &mut TileWork,
    ) -> Result<()> {
        let spec = &self.spec;
        let (t, dim) = (serial.tokens, spec.dim);
        let input: Vec<f32> = (0..grouped.rows * dim)
            .map(|i| {
                let (row, col) = (i / dim, i % dim);
                (((row % 251) * 17 + (col % 257) * 13 + (row / t % 269) * 19) % 1009) as f32 / 512.0
                    - 1.0
            })
            .collect();
        let delta: Vec<bf16> = input
            .iter()
            .enumerate()
            .map(|(i, &x)| bf16::from_f32(x * 0.25 + ((i * 23 % 263) as f32 - 131.0) / 512.0))
            .collect();
        s.memcpy_htod(&input, &mut grouped.x)?;
        s.memcpy_htod(&delta, &mut grouped.o)?;
        let block = &self.blocks[0];
        self.rms_weight(s, &grouped.x, &block.norm1, &grouped.normed, grouped.rows)?;
        self.scaled_residual(s, &grouped.x, &grouped.o, &block.scale1, grouped.rows)?;
        let mut defects = 0;
        let mut cells = 0;
        for tile in 0..grouped.batch {
            let first = tile * t * dim;
            s.memcpy_htod(&input[first..first + t * dim], &mut serial.x)?;
            s.memcpy_htod(&delta[first..first + t * dim], &mut serial.o)?;
            self.rms_weight(s, &serial.x, &block.norm1, &serial.normed, t)?;
            self.scaled_residual(s, &serial.x, &serial.o, &block.scale1, t)?;
            defects += tile_bit_differences(
                s,
                "weighted-rms",
                &serial.normed,
                &grouped.normed,
                (tile, t, dim, 1),
            )?;
            defects += tile_bit_differences(
                s,
                "scaled-residual",
                &serial.x,
                &grouped.x,
                (tile, t, dim, 1),
            )?;
            cells += 2;
        }
        s.memcpy_htod(&input, &mut grouped.x)?;
        self.block(s, blas, block, grouped)?;
        for tile in 0..grouped.batch {
            let first = tile * t * dim;
            s.memcpy_htod(&input[first..first + t * dim], &mut serial.x)?;
            self.block(s, blas, block, serial)?;
            for (name, a, b, columns, heads) in [
                ("q", &serial.q, &grouped.q, dim, 1),
                ("k", &serial.k, &grouped.k, dim, 1),
                ("v", &serial.v, &grouped.v, dim, 1),
                ("rope-q", &serial.q_n, &grouped.q_n, dim, 1),
                ("head-norm-k", &serial.k_n, &grouped.k_n, dim, 1),
                (
                    "rope-k-cache",
                    &serial.k_cache,
                    &grouped.k_cache,
                    spec.head_dim,
                    spec.heads,
                ),
                (
                    "v-cache",
                    &serial.v_cache,
                    &grouped.v_cache,
                    spec.head_dim,
                    spec.heads,
                ),
                ("attention", &serial.attn, &grouped.attn, dim, 1),
                ("norm2", &serial.normed, &grouped.normed, dim, 1),
                ("ff-in", &serial.gate_up, &grouped.gate_up, 2 * spec.ffn, 1),
                ("silu", &serial.gated, &grouped.gated, spec.ffn, 1),
                ("ff-out", &serial.o, &grouped.o, dim, 1),
            ] {
                defects += tile_bit_differences(s, name, a, b, (tile, t, columns, heads))?;
                cells += 1;
            }
            defects +=
                tile_bit_differences(s, "residual2", &serial.x, &grouped.x, (tile, t, dim, 1))?;
            defects += tile_bit_differences(
                s,
                "packed-lse",
                &serial.flash.lse,
                &grouped.flash.lse,
                (tile, t, spec.heads, 1),
            )?;
            cells += 2;
        }
        self.output_rows(s, blas, grouped)?;
        for tile in 0..grouped.batch {
            let first = tile * t * dim;
            s.memcpy_htod(&input[first..first + t * dim], &mut serial.x)?;
            self.block(s, blas, block, serial)?;
            self.output_rows(s, blas, serial)?;
            defects +=
                tile_bit_differences(s, "norm-out-f32", &serial.y, &grouped.y, (tile, t, dim, 1))?;
            defects += tile_bit_differences(
                s,
                "norm-out-bf16",
                &serial.normed,
                &grouped.normed,
                (tile, t, dim, 1),
            )?;
            defects += tile_bit_differences(
                s,
                "proj-out",
                &serial.patch_rows,
                &grouped.patch_rows,
                (tile, t, spec.patch_width(), 1),
            )?;
            cells += 3;
        }
        println!("H3_VAE_GROUP_OPS cells={cells} bit_limit=0 defects={defects}");
        ensure!(defects == 0, "H3_VAE_GROUP_OPS DEFECT cells={defects}");
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn biased_row_gate(
        &self,
        device: &Device,
        blas: &mut Blas,
        frozen: &serde_json::Value,
    ) -> Result<()> {
        let spec = &self.spec;
        let side = VAE_TILE / spec.spatial_ratio;
        let canonical =
            (spec.chunk_latents() + spec.overlap_latents()) * side * side + spec.registers + 1;
        let rows: Vec<usize> = serde_json::from_value(frozen["row_gate_counts"].clone())?;
        ensure!(
            canonical == 1797
                && rows == [1797, 3594, 5391, 50316]
                && frozen["bit_difference_limit"] == 0
                && frozen["row_gate_owners"] == self.linears().count(),
            "RUNNER_FAIL biased row freeze contract mismatch"
        );
        let mut defects = 0;
        let mut controls = 0;
        for (owner, linear) in self.linears().enumerate() {
            for &rows in &rows {
                let (bad, control) =
                    self.biased_row_cell(device, blas, (owner, linear), (rows, canonical), frozen)?;
                defects += bad;
                controls += control;
            }
        }
        ensure!(
            controls == 1,
            "RUNNER_FAIL biased row control not discriminating"
        );
        println!(
            "H3_VAE_BIASED_ROWS owners={} rows={rows:?} modes=2 bit_limit=0 defects={defects} control=FAIL_AS_REQUIRED",
            self.linears().count()
        );
        ensure!(defects == 0, "H3_VAE_BIASED_ROWS DEFECT cells={defects}");
        println!("H3_VAE_BIASED_ROWS PASS owners=217 cells=1736 bit_limit=0");
        Ok(())
    }

    #[cfg(test)]
    fn biased_row_cell(
        &self,
        device: &Device,
        blas: &mut Blas,
        (owner, linear): (usize, &Linear),
        (rows, canonical): (usize, usize),
        frozen: &serde_json::Value,
    ) -> Result<(usize, usize)> {
        use crate::backend::setup::LinearShape;
        let s = &device.stream;
        let shape = LinearShape {
            rows,
            output: linear.out,
            input: linear.input,
        };
        let algorithm = blas.prepare_biased_family(shape, canonical, vector(&linear.bias, s)?)?;
        let identity = serde_json::to_value(algorithm.data)?;
        for arm in ["standalone-A", "standalone-P", "pipeline-A", "pipeline-P"] {
            let expected = &frozen["arm_results"][arm]["canonical_bias_owners"][owner.to_string()];
            ensure!(
                expected["opaque_algorithm"] == identity
                    && expected["output"] == linear.out
                    && expected["input"] == linear.input,
                "RUNNER_FAIL biased row frozen algorithm mismatch owner={owner} rows={rows} arm={arm}"
            );
        }
        let guard = [bf16::from_f32(-123.5); 8];
        let mut input = vec![guard[0]; (rows + 1) * linear.input + 16];
        let boundary = [0x3f7f, 0x3f80, 0x3f81, 0xbf7f, 0xbf80, 0xbf81, 0x3b80];
        for (i, x) in input[8..8 + (rows + 1) * linear.input]
            .iter_mut()
            .enumerate()
        {
            let (row, col) = (i / linear.input, i % linear.input);
            *x = if (row + col).is_multiple_of(263) {
                bf16::from_bits(boundary[(row + col / 263) % boundary.len()])
            } else {
                let signal = ((row % 251) * 17
                    + (col % 257) * 13
                    + (row / canonical % 269) * 19
                    + owner * 7)
                    % 509;
                bf16::from_f32((signal as f32 - 254.0) / 256.0)
            };
        }
        let input = s.clone_htod(&input)?;
        let mut serial = s.alloc_zeros::<bf16>(rows * linear.out + 16)?;
        let mut grouped = s.alloc_zeros::<bf16>(serial.len())?;
        for out in [&mut serial, &mut grouped] {
            let end = out.len() - 8;
            s.memcpy_htod(&guard, &mut out.slice_mut(..8))?;
            s.memcpy_htod(&guard, &mut out.slice_mut(end..))?;
        }
        ensure!(
            rows_view(&input, s, 9, rows, linear.input, linear.input).is_err()
                && rows_view(&grouped, s, 24, rows, linear.out, linear.out).is_err(),
            "RUNNER_FAIL biased row alignment/capacity guard accepted invalid view"
        );
        let call = |offset: usize, count: usize, destination: &CudaSlice<bf16>, shifted: usize| {
            blas.linear_biased(
                matrix(&linear.weight, s, linear.out, linear.input)?,
                vector(&linear.bias, s)?,
                rows_view(
                    &input,
                    s,
                    8 + (offset + shifted) * linear.input,
                    count,
                    linear.input,
                    linear.input,
                )?,
                &mut TensorMut(rows_view(
                    destination,
                    s,
                    8 + offset * linear.out,
                    count,
                    linear.out,
                    linear.out,
                )?),
                MODE,
            )
        };
        for first in (0..rows).step_by(canonical) {
            call(first, canonical, &serial, 0)?;
        }
        let expected = s.clone_dtoh(&serial)?;
        ensure!(
            expected.iter().all(|x| x.is_finite()),
            "RUNNER_FAIL serial biased output nonfinite"
        );
        let mut defects = 0;
        let mut control = 0;
        for graph in [false, true] {
            let end = grouped.len() - 8;
            s.memset_zeros(&mut grouped.slice_mut(8..end))?;
            if graph {
                s.begin_capture(
                    cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )?;
            }
            let launched = call(0, rows, &grouped, 0);
            if graph {
                let captured = s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
                launched?;
                captured?
                    .context("RUNNER_FAIL biased row graph empty")?
                    .launch()?;
            } else {
                launched?;
            }
            let actual = s.clone_dtoh(&grouped)?;
            let last = actual.len() - 8;
            ensure!(
                actual[..8] == guard
                    && actual[last..] == guard
                    && expected[..8] == guard
                    && expected[last..] == guard
                    && actual.iter().all(|x| x.is_finite()),
                "RUNNER_FAIL biased row guard/nonfinite owner={owner} rows={rows} graph={graph}"
            );
            let bits = expected[8..last]
                .iter()
                .zip(&actual[8..last])
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            defects += usize::from(bits != 0);
            println!(
                "H3_VAE_BIASED_ROW owner={owner} rows={rows} graph={graph} bit_differences={bits} status={}",
                if bits == 0 { "PASS" } else { "DEFECT" }
            );
            if owner == 0 && rows == 3 * canonical && !graph {
                call(0, rows, &grouped, 1)?;
                let wrong = s.clone_dtoh(&grouped)?;
                let differences = expected[8..last]
                    .iter()
                    .zip(&wrong[8..last])
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                ensure!(
                    differences > 0,
                    "RUNNER_FAIL biased row wrong-view control passed"
                );
                println!(
                    "H3_VAE_BIASED_ROW_CONTROL shifted_input_rows=1 bit_differences={differences} status=FAIL_AS_REQUIRED"
                );
                control = 1;
            }
        }
        Ok((defects, control))
    }

    #[cfg(test)]
    pub(crate) fn batch_setup(
        &self,
        device: &Device,
        blas: &mut Blas,
        tiles: &VaeTiles,
        shape: (usize, usize, usize),
    ) -> Result<()> {
        self.batch_plan(device, blas, tiles, shape)?;
        Ok(())
    }

    fn batch_plan(
        &self,
        device: &Device,
        blas: &mut Blas,
        tiles: &VaeTiles,
        (frames, height, width): (usize, usize, usize),
    ) -> Result<TileBatch> {
        let spec = &self.spec;
        let shape = (
            spec.chunk_latents() + spec.overlap_latents(),
            tiles.height / spec.spatial_ratio,
            tiles.width / spec.spatial_ratio,
        );
        let patches = shape.0 * shape.1 * shape.2;
        let tokens = patches + spec.registers + 1;
        let plan = DecodePlan::new(spec, frames)?;
        ensure!(
            plan.pad == 0,
            "batch setup requires unpadded temporal input"
        );
        let kept = spec.chunk_latents() * spec.temporal_ratio - spec.frame_pre_padding();
        let tail_frames = shape.0 * spec.temporal_ratio
            - spec.chunk_latents() * spec.temporal_ratio
            - spec.frame_pre_padding();
        let output_frames = plan.chunks * kept + tail_frames;
        let pixels = height * width * spec.spatial_ratio * spec.spatial_ratio;
        let output = 2 * spec.out_channels * output_frames * pixels;
        let latent = 4 * spec.latent_channels * frames * height * width;
        let stitch = decode_bytes(
            spec,
            tiles,
            (height * spec.spatial_ratio, width * spec.spatial_ratio),
        );
        let grid =
            4 * (tiles.rows.0.len() + tiles.rows.1.len() + tiles.cols.0.len() + tiles.cols.1.len());
        device.stream.synchronize()?;
        let (free, total) = device.ctx.mem_get_info()?;
        let reserve = total / 16;
        let mandatory = output + latent + stitch + grid;
        let budget = device.workspace_budget(free, free, mandatory + reserve)?;
        let workspace_before = blas.workspace_bytes();
        self.post_quant.prepare(&device.stream, blas, patches)?;
        self.proj_in.prepare(&device.stream, blas, patches)?;
        for linear in self.linears() {
            linear.prepare(&device.stream, blas, tokens)?;
            #[cfg(test)]
            let owner = self
                .linears()
                .position(|item| std::ptr::eq(item, linear))
                .context("bias owner missing")?;
            #[cfg(test)]
            eprintln!(
                "{}",
                serde_json::json!({"h3_vae_bias_owner": {
                    "owner":owner,"bias_pointer":pointer(&linear.bias,&device.stream),
                    "output":linear.out,"input":linear.input,"canonical_rows":tokens
                }})
            );
        }
        let slots = tiles.count();
        use cudarc::driver::sys::CUdevice_attribute as A;
        let grid_x = device
            .ctx
            .attribute(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)? as usize;
        let grid_z = device
            .ctx
            .attribute(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)? as usize;
        let mut choices = Vec::new();
        #[cfg(test)]
        let mut records = Vec::new();
        for batch in 1..=slots {
            if batch > grid_z
                || tokens.div_ceil(64) > grid_x
                || !tokens
                    .checked_mul(batch)
                    .and_then(|r| r.checked_mul(2 * spec.ffn))
                    .is_some_and(|n| n <= i32::MAX as usize)
            {
                if crate::backend::diagnostics_enabled() {
                    eprintln!(
                        "H3_VAE_BATCH_REJECT batch={batch} reason=kernel_index_or_grid_limit"
                    );
                }
                continue;
            }
            let tail = slots % batch;
            let groups = slots.div_ceil(batch);
            let work = TileWork::batch_bytes(spec, shape, batch, slots)?;
            let query = (|| -> Result<(f64, usize)> {
                let mut cost = 0.0;
                let mut workspace = 0;
                for linear in self.linears() {
                    #[cfg(test)]
                    let owner = self
                        .linears()
                        .position(|item| std::ptr::eq(item, linear))
                        .context("bias owner missing")?;
                    for (active, count) in [(batch, slots / batch), (tail, 1)] {
                        if active == 0 || count == 0 {
                            continue;
                        }
                        #[cfg(test)]
                        eprintln!(
                            "{}",
                            serde_json::json!({"h3_vae_batch_query": {
                                "batch":batch,"active":active,"owner":owner,"multiplicity":count
                            }})
                        );
                        let rows = active * tokens;
                        let (multiplier, bytes) = blas.biased_family_cost(
                            crate::backend::setup::LinearShape {
                                rows,
                                output: linear.out,
                                input: linear.input,
                            },
                            tokens,
                            vector(&linear.bias, &device.stream)?,
                        )?;
                        cost += 2.0
                            * rows as f64
                            * linear.out as f64
                            * linear.input as f64
                            * multiplier
                            * count as f64;
                        workspace = workspace.max(bytes);
                    }
                }
                ensure!(
                    cost.is_finite() && cost > 0.0,
                    "invalid tile tensor cost {cost}"
                );
                Ok((cost, workspace))
            })();
            let (cost, workspace) = match query {
                Ok(value) => value,
                Err(error) if error.downcast_ref::<cudarc::cublaslt::result::CublasError>()
                    .is_some_and(|e| e.0 == cudarc::cublaslt::sys::cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED) => {
                    if crate::backend::diagnostics_enabled() {
                        eprintln!("H3_VAE_BATCH_REJECT batch={batch} reason={error:#}");
                    }
                    continue;
                }
                Err(error) => return Err(error.context(format!("tile batch {batch}"))),
            };
            let replacement = if workspace > workspace_before {
                workspace
            } else {
                0
            };
            let peak = work
                .checked_add(replacement)
                .context("tile batch peak overflow")?;
            let launches = plan.chunks * (544 * groups + 4 * slots) + 2 * plan.chunks - 1;
            #[cfg(test)]
            {
                let record = serde_json::json!({"batch":batch,"tail":tail,"groups_per_chunk":groups,
                "full_rows":batch*tokens,"tail_rows":tail*tokens,"work_bytes":work,
                "max_workspace_bytes":workspace,"workspace_replacement_bytes":replacement,
                "peak_new_allocation_bytes":peak,"tensor_cost":cost,"launches":launches,
                "within_budget":peak<=budget});
                eprintln!("{}", serde_json::json!({"h3_vae_batch_candidate":record}));
                records.push(record);
            }
            choices.push(TileBatch {
                batch,
                peak,
                cost,
                launches,
                work,
                workspace,
            });
        }
        let chosen = TileBatch::choose(&choices, budget)?;
        #[cfg(test)]
        {
            let result = serde_json::json!({"tokens_per_tile":tokens,"patches_per_tile":patches,
            "slots":slots,"temporal_chunks":plan.chunks,"output_frames":output_frames,
            "free_before_work_bytes":free,"total_bytes":total,"safety_reserve_bytes":reserve,
            "future_output_bytes":output,"future_latent_bytes":latent,"future_stitch_bytes":stitch,
            "future_grid_bytes":grid,"mandatory_future_bytes":mandatory,"workspace_budget_bytes":budget,
            "workspace_before_bytes":workspace_before,"canonical_biased_owners":self.linears().count(),
            "chosen":records.iter().find(|x| x["batch"] == chosen.batch),"candidates":records,
            "numerical_candidate_executed":false,"model_cells_scored":0});
            eprintln!("{}", serde_json::json!({"h3_vae_batch_setup":result}));
        }
        if crate::backend::diagnostics_enabled() {
            eprintln!(
                "{}",
                serde_json::json!({"h3_vae_tile_plan": {
                    "batch":chosen.batch,"tail":slots%chosen.batch,"work_bytes":chosen.work,
                    "workspace_bytes":chosen.workspace,"peak_new_bytes":chosen.peak,
                "budget_bytes":budget,"free_before_work_bytes":free,"slots":slots,
                "future_fixed_bytes":mandatory,"safety_reserve_bytes":reserve,
                "retained_workspace_bytes":workspace_before,"total_bytes":total,
                "full_rows":chosen.batch*tokens,"tail_rows":(slots%chosen.batch)*tokens,
                "predicted_launches":chosen.launches
                }})
            );
        }
        Ok(chosen.clone())
    }

    pub(crate) fn work(
        &self,
        device: &Device,
        blas: &mut Blas,
        tiles: &VaeTiles,
        shape: (usize, usize, usize),
    ) -> Result<TileWork> {
        let plan = self.batch_plan(device, blas, tiles, shape)?;
        blas.bind_workspace(plan.workspace)?;
        let spec = &self.spec;
        let work = TileWork::new(
            &device.stream,
            spec,
            (
                spec.chunk_latents() + spec.overlap_latents(),
                tiles.height / spec.spatial_ratio,
                tiles.width / spec.spatial_ratio,
            ),
            (plan.batch, tiles.count()),
        )?;
        work.prepare(&device.stream, blas, self)?;
        ensure!(
            work.allocated_bytes() == plan.work,
            "tile work differs from selected memory plan"
        );
        Ok(work)
    }

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
            rms_weight: vision.load_function("vision_rms_scale_f32")?,
            scaled_residual: vision.load_function("scaled_residual_f32")?,
            layer_norm: cubin::module(s.context(), "q38_common")?
                .load_function("q38_layer_norm_f32")?,
            unpatchify: vision.load_function("unpatchify_bf16")?,
            stitch: vision.load_function("tile_stitch_bf16")?,
            frame_blend: vision.load_function("frame_blend_bf16")?,
            latent_rows: vision.load_function("latent_rows_bf16")?,
            rgb: vision.load_function("planar_bf16_to_rgb_u8")?,
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
        if crate::backend::diagnostics_enabled() {
            eprintln!(
                "{}",
                serde_json::json!({"h3_vae_decoder_placement": {"mode": "resident", "weight_bytes": loaded,
                    "free_before_bytes": free, "free_after_bytes": d.ctx.mem_get_info()?.0, "total_bytes": total}})
            );
        }
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
        let (t, dim) = (w.rows, spec.dim);
        self.rms_weight(s, &w.x, &b.norm1, &w.normed, t)?;
        for (linear, output) in [(&b.q, &w.q), (&b.k, &w.k), (&b.v, &w.v)] {
            linear.run(
                s,
                blas,
                rows_view(&w.normed, s, 0, t, dim, dim)?,
                rows_view(output, s, 0, t, dim, dim)?,
            )?;
        }
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
        b.out.run(
            s,
            blas,
            rows_view(&w.attn, s, 0, t, dim, dim)?,
            rows_view(&w.o, s, 0, t, dim, dim)?,
        )?;
        self.scaled_residual(s, &w.x, &w.o, &b.scale1, t)?;
        self.rms_weight(s, &w.x, &b.norm2, &w.normed, t)?;
        b.ff_in.run(
            s,
            blas,
            rows_view(&w.normed, s, 0, t, dim, dim)?,
            rows_view(&w.gate_up, s, 0, t, 2 * spec.ffn, 2 * spec.ffn)?,
        )?;
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
        b.ff_out.run(
            s,
            blas,
            rows_view(&w.gated, s, 0, t, spec.ffn, spec.ffn)?,
            rows_view(&w.o, s, 0, t, dim, dim)?,
        )?;
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
                .arg(&1u32)
                .arg(&0u64)
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

    fn output_rows(&self, s: &Arc<CudaStream>, blas: &Blas, w: &TileWork) -> Result<()> {
        let spec = &self.spec;
        let (t, dim) = (w.rows, spec.dim);
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
        self.proj_out.run(
            s,
            blas,
            rows_view(&w.normed, s, 0, t, dim, dim)?,
            rows_view(
                &w.patch_rows,
                s,
                0,
                t,
                spec.patch_width(),
                spec.patch_width(),
            )?,
        )?;
        Ok(())
    }

    /// Consecutive spatial tiles write consecutive raw slots `[slots, channels, frames, h, w]`.
    fn tile(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        w: &mut TileWork,
        z: &LatentVideo<'_>,
        (origins, first_frame): (&[(usize, usize)], usize),
        (raw, slot): (&CudaSlice<bf16>, usize),
    ) -> Result<()> {
        let spec = &self.spec;
        let (n, dim) = (w.patches, spec.dim);
        let active = origins.len();
        ensure!(
            active > 0 && active <= w.batch && (active == w.batch || active == w.tail),
            "spatial group does not match its full/tail bindings"
        );
        ensure!(
            slot.checked_add(active)
                .and_then(|n| n.checked_mul(w.pixels()))
                .is_some_and(|n| n <= raw.len()),
            "raw spatial group exceeds slot capacity"
        );
        let t = active * w.tokens;
        w.rows = t;
        if active != w.batch {
            std::mem::swap(
                &mut w.flash,
                w.tail_flash.as_mut().context("tail attention missing")?,
            );
        }
        let count = n * spec.latent_channels;
        for (tile, &(first_row, first_col)) in origins.iter().enumerate() {
            ensure!(
                first_frame
                    .checked_add(w.frames)
                    .is_some_and(|n| n <= z.frames)
                    && first_row
                        .checked_add(w.height)
                        .is_some_and(|n| n <= z.height)
                    && first_col.checked_add(w.width).is_some_and(|n| n <= z.width),
                "latent spatial group origin exceeds input capacity"
            );
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
            self.post_quant.run(
                s,
                blas,
                rows_view(
                    &w.latent,
                    s,
                    0,
                    n,
                    spec.latent_channels,
                    spec.latent_channels,
                )?,
                rows_view(
                    &w.latent_q,
                    s,
                    0,
                    n,
                    spec.latent_channels,
                    spec.latent_channels,
                )?,
            )?;
            let offset = tile * w.tokens * dim;
            self.proj_in.run(
                s,
                blas,
                rows_view(
                    &w.latent_q,
                    s,
                    0,
                    n,
                    spec.latent_channels,
                    spec.latent_channels,
                )?,
                rows_view(&w.hidden, s, offset, n, dim, dim)?,
            )?;
            s.memcpy_dtod(
                &self.registers,
                &mut w
                    .hidden
                    .slice_mut(offset + n * dim..offset + (n + spec.registers) * dim),
            )?;
        }
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
        self.output_rows(s, blas, w)?;
        let tile_size = w.pixels();
        for tile in 0..active {
            let slot = slot + tile;
            let target = pointer(raw, s) + (slot * tile_size * std::mem::size_of::<bf16>()) as u64;
            let source = rows_view(
                &w.patch_rows,
                s,
                tile * w.tokens * spec.patch_width(),
                w.tokens,
                spec.patch_width(),
                spec.patch_width(),
            )?
            .ptr;
            ensure!(
                (slot + 1) * tile_size <= raw.len(),
                "raw tile slot {slot} out of range"
            );
            unsafe {
                s.launch_builder(&self.kernels.unpatchify)
                    .arg(&source)
                    .arg(&target)
                    .arg(&(w.frames as u32))
                    .arg(&(w.height as u32))
                    .arg(&(w.width as u32))
                    .arg(&(spec.out_channels as u32))
                    .arg(&(spec.temporal_ratio as u32))
                    .arg(&(spec.spatial_ratio as u32))
                    .launch(flat(tile_size))?;
            }
        }
        if active != w.batch {
            std::mem::swap(
                &mut w.flash,
                w.tail_flash.as_mut().context("tail attention missing")?,
            );
        }
        w.rows = w.batch * w.tokens;
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
        self.decode_observed((s, blas), w, tiles, z, out, |_, _, _| Ok(()))
    }

    fn decode_observed(
        &self,
        (s, blas): (&Arc<CudaStream>, &Blas),
        w: &mut TileWork,
        tiles: &VaeTiles,
        z: LatentVideo<'_>,
        out: &mut CudaSlice<bf16>,
        mut observe: impl FnMut(&str, usize, &CudaSlice<bf16>) -> Result<()>,
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
        ensure!(
            2 * (raw.len() + canvas.len() + faded.len())
                == decode_bytes(spec, tiles, (height, width)),
            "decode buffers differ from decode_bytes"
        );
        if crate::backend::diagnostics_enabled() {
            eprintln!(
                "{}",
                serde_json::json!({"h3_vae_decode": {"chunks": plan.chunks, "tiles": slots, "tokens_per_tile": w.tokens,
                    "frames": plan.frames, "raw_tile_bytes": 2 * raw.len(), "canvas_bytes": 2 * canvas.len(),
                    "output_bytes": 2 * out.len(), "free_bytes": s.context().mem_get_info()?.0}})
            );
        }
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
        let origins: Vec<_> = rows
            .0
            .iter()
            .flat_map(|&row| cols.0.iter().map(move |&col| (row / r, col / r)))
            .collect();
        for i in 0..plan.chunks {
            for (group, origins) in origins.chunks(w.batch).enumerate() {
                self.tile(
                    s,
                    blas,
                    w,
                    &z,
                    (origins, i * chunk),
                    (&raw, group * w.batch),
                )?;
            }
            observe("raw", i, &raw)?;
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
            observe("canvas", i, &canvas)?;
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
            observe("emitted", i, out)?;
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
            observe("tail", i, &faded)?;
        }
        if spec.token_drop > 0 {
            copy_frames(s, &faded, tail, 0, out, total, plan.chunks * kept, tail)?;
        }
        observe("final", plan.chunks, out)?;
        Ok(plan)
    }
}

/// Device bytes `VideoDecoder::decode` allocates besides its output for a `height x width` px video through `tiles`:
/// raw tiles of one clip, the stitched clip canvas and the cross-faded tail, BF16.
pub(crate) fn decode_bytes(
    spec: &VaeSpec,
    tiles: &VaeTiles,
    (height, width): (usize, usize),
) -> usize {
    let clip_frames = (spec.chunk_latents() + spec.overlap_latents()) * spec.temporal_ratio;
    let lead = spec.chunk_latents() * spec.temporal_ratio;
    let tail = clip_frames - lead - spec.frame_pre_padding();
    let channels = spec.out_channels;
    let tile = channels * clip_frames * tiles.height * tiles.width;
    2 * (tiles.count() * tile + channels * (clip_frames + tail) * height * width)
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
    rows: usize,
    batch: usize,
    tail: usize,
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
    tail_flash: Option<Flash>,
    spec: VaeSpec,
}

impl TileWork {
    pub(crate) fn batch_bytes(
        spec: &VaeSpec,
        (frames, height, width): (usize, usize, usize),
        batch: usize,
        slots: usize,
    ) -> Result<usize> {
        ensure!(batch > 0 && batch <= slots, "invalid tile batch capacity");
        ensure!(
            [frames, height, width].iter().all(|&n| n > 0),
            "zero tile work dimension"
        );
        let product = |values: &[usize]| -> Result<usize> {
            values.iter().try_fold(1usize, |a, &b| {
                a.checked_mul(b).context("tile work overflow")
            })
        };
        let sum = |values: &[usize]| -> Result<usize> {
            values.iter().try_fold(0usize, |a, &b| {
                a.checked_add(b).context("tile work overflow")
            })
        };
        let patches = product(&[frames, height, width])?;
        let tokens = sum(&[patches, spec.registers, 1])?;
        let bf16s = sum(&[
            product(&[2, patches, spec.latent_channels])?,
            product(&[11, tokens, spec.dim])?,
            product(&[3, tokens, spec.ffn])?,
            product(&[tokens, spec.patch_width()])?,
            product(&[tokens, spec.rope_dims])?,
            spec.head_dim,
        ])?;
        let serial = sum(&[
            product(&[2, bf16s])?,
            product(&[
                4,
                sum(&[
                    product(&[2, tokens, spec.dim])?,
                    product(&[spec.heads, tokens])?,
                ])?,
            ])?,
        ])?;
        let shared = sum(&[
            product(&[4, patches, spec.latent_channels])?,
            product(&[2, spec.head_dim])?,
        ])?;
        let tail = slots % batch;
        sum(&[
            product(&[
                batch,
                serial.checked_sub(shared).context("tile shared size")?,
            ])?,
            shared,
            if batch > 1 {
                product(&[4, batch.checked_add(1).context("offset count")?])?
            } else {
                0
            },
            product(&[tail, 4, spec.heads, tokens])?,
            if tail > 1 {
                product(&[4, tail.checked_add(1).context("offset count")?])?
            } else {
                0
            },
        ])
    }

    /// Device bytes `new` allocates for a `frames x height x width` tile: BF16 latent rows, activations, FFN, patch
    /// rows, rotary table and unit vector; the F32 residual pair; the attention LSE `[heads, tokens]` F32.
    pub(crate) fn bytes(spec: &VaeSpec, (frames, height, width): (usize, usize, usize)) -> usize {
        let patches = frames * height * width;
        let tokens = patches + spec.registers + 1;
        let bf16s = 2 * patches * spec.latent_channels
            + 11 * tokens * spec.dim
            + 3 * tokens * spec.ffn
            + tokens * spec.patch_width()
            + tokens * spec.rope_dims
            + spec.head_dim;
        2 * bf16s + 4 * (2 * tokens * spec.dim + spec.heads * tokens)
    }

    pub(crate) fn new(
        s: &Arc<CudaStream>,
        spec: &VaeSpec,
        (frames, height, width): (usize, usize, usize),
        (batch, slots): (usize, usize),
    ) -> Result<Self> {
        let expected = Self::batch_bytes(spec, (frames, height, width), batch, slots)?;
        let patches = frames * height * width;
        let tokens = patches + spec.registers + 1;
        let active_rows = tokens.checked_mul(batch).context("tile rows overflow")?;
        ensure!(
            active_rows
                .checked_mul(2 * spec.ffn)
                .is_some_and(|n| n <= i32::MAX as usize),
            "tile FFN rows exceed kernel indexing range"
        );
        let tail = slots % batch;
        let dim = spec.dim;
        let rows = |n: usize| s.alloc_zeros::<bf16>(n);
        let flash = |batch: usize| -> Result<Flash> {
            let count = batch * tokens;
            let mut flash = Flash::new_with_score(
                s.clone(),
                count,
                spec.heads,
                spec.heads,
                spec.head_dim,
                Score {
                    scale: (spec.head_dim as f32).sqrt().recip(),
                    window_left: None,
                    window_right: None,
                },
                count,
            )?;
            if batch > 1 {
                let offsets = (0..=batch)
                    .map(|i| u32::try_from(i * tokens))
                    .collect::<Result<Vec<_>, _>>()?;
                flash.bind_bf16_windows(&offsets)?;
            }
            Ok(flash)
        };
        let work = Self {
            frames,
            height,
            width,
            patches,
            tokens,
            rows: active_rows,
            batch,
            tail,
            latent: rows(patches * spec.latent_channels)?,
            latent_q: rows(patches * spec.latent_channels)?,
            hidden: rows(active_rows * dim)?,
            x: s.alloc_zeros::<f32>(active_rows * dim)?,
            y: s.alloc_zeros::<f32>(active_rows * dim)?,
            normed: rows(active_rows * dim)?,
            q: rows(active_rows * dim)?,
            k: rows(active_rows * dim)?,
            v: rows(active_rows * dim)?,
            q_n: rows(active_rows * dim)?,
            k_n: rows(active_rows * dim)?,
            k_cache: rows(active_rows * dim)?,
            v_cache: rows(active_rows * dim)?,
            attn: rows(active_rows * dim)?,
            o: rows(active_rows * dim)?,
            gate_up: rows(active_rows * 2 * spec.ffn)?,
            gated: rows(active_rows * spec.ffn)?,
            patch_rows: rows(active_rows * spec.patch_width())?,
            rope: s.clone_htod(&rope_table(spec, (frames, height, width)).repeat(batch))?,
            ones: s.clone_htod(&vec![bf16::ONE; spec.head_dim])?,
            flash: flash(batch)?,
            tail_flash: if tail > 0 { Some(flash(tail)?) } else { None },
            spec: spec.clone(),
        };
        let allocated = work.allocated_bytes();
        ensure!(
            allocated == expected,
            "tile work allocated {allocated} B, planned {expected}"
        );
        Ok(work)
    }

    fn allocated_bytes(&self) -> usize {
        let bf16s: usize = [
            &self.latent,
            &self.latent_q,
            &self.hidden,
            &self.normed,
            &self.q,
            &self.k,
            &self.v,
            &self.q_n,
            &self.k_n,
            &self.k_cache,
            &self.v_cache,
            &self.attn,
            &self.o,
            &self.gate_up,
            &self.gated,
            &self.patch_rows,
            &self.rope,
            &self.ones,
        ]
        .iter()
        .map(|b| b.len())
        .sum();
        2 * bf16s
            + 4 * (self.x.len() + self.y.len())
            + self.flash.allocated_bytes()
            + self.tail_flash.as_ref().map_or(0, Flash::allocated_bytes)
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
            for active in [self.batch, self.tail] {
                if active > 1 {
                    blas.prepare_biased_family(
                        crate::backend::setup::LinearShape {
                            rows: active * self.tokens,
                            output: linear.out,
                            input: linear.input,
                        },
                        self.tokens,
                        vector(&linear.bias, s)?,
                    )?;
                }
            }
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
pub(crate) fn rope_table(spec: &VaeSpec, shape: (usize, usize, usize)) -> Vec<bf16> {
    rotary_table(spec, shape, ROPE_AXES, bf16::from_f32)
}

fn rotary_table<T: Clone>(
    spec: &VaeSpec,
    (frames, height, width): (usize, usize, usize),
    axes: usize,
    cast: impl Fn(f32) -> T,
) -> Vec<T> {
    let half = spec.rope_dims / 2;
    let freqs = half / axes;
    let step = (2 * axes) as f32 / spec.rope_dims as f32;
    let inv: Vec<f32> = (0..freqs)
        .map(|k| 1.0f32 / spec.rope_theta.powf(k as f32 * step))
        .collect();
    let coordinate = |i: usize, size: usize| 2.0f32 * ((i as f32 + 0.5) / size as f32) - 1.0;
    let turn = (2.0 * std::f64::consts::PI) as f32;
    let tokens = frames * height * width + spec.registers + 1;
    let mut table = vec![cast(0.0); tokens * spec.rope_dims];
    for (row, chunk) in table.chunks_mut(spec.rope_dims).enumerate() {
        if row >= frames * height * width {
            chunk[..half].fill(cast(1.0));
            continue;
        }
        let position = [row / (height * width), row / width % height, row % width];
        let sizes = [frames, height, width];
        for axis in 0..axes {
            let source = 3 - axes + axis;
            let x = coordinate(position[source], sizes[source]);
            for (k, inv) in inv.iter().enumerate() {
                let angle = turn * x * inv;
                chunk[axis * freqs + k] = cast(angle.cos());
                chunk[half + axis * freqs + k] = cast(angle.sin());
            }
        }
    }
    table
}

/// A causal 3D convolution `[out, input, k, k, k]` with bias, published F32, applied to one frame.
struct Conv {
    weight: CudaSlice<f32>,
    bias: CudaSlice<f32>,
    out: usize,
    input: usize,
    kernel: usize,
}

/// Uploads one published F32 tensor of the F32 set by name and element count.
type LoadF32<'a> = dyn FnMut(&str, usize) -> Result<CudaSlice<f32>> + 'a;

impl Conv {
    fn load(
        load: &mut LoadF32<'_>,
        base: &str,
        (out, input, kernel): (usize, usize, usize),
    ) -> Result<Self> {
        Ok(Self {
            weight: load(&format!("{base}.weight"), out * input * kernel.pow(3))?,
            bias: load(&format!("{base}.bias"), out)?,
            out,
            input,
            kernel,
        })
    }

    /// GEMM reduction width: `input x k^3` im2col columns (the input rows themselves for `k = 1`).
    fn columns(&self) -> usize {
        self.input * self.kernel.pow(3)
    }

    /// Output extent of `size` input pixels: `size` at stride 1 (pad `k / 2` each side); at stride 2 the input carries
    /// one reflected row/column after it and no pad before, `(size + 1 - k) / 2 + 1`.
    fn extent(&self, size: usize, stride: usize) -> usize {
        if stride == 1 {
            size
        } else {
            (size + 1 - self.kernel) / stride + 1
        }
    }
}

struct GroupNorm {
    weight: CudaSlice<f32>,
    bias: CudaSlice<f32>,
}

impl GroupNorm {
    fn load(load: &mut LoadF32<'_>, base: &str, channels: usize) -> Result<Self> {
        Ok(Self {
            weight: load(&format!("{base}.weight"), channels)?,
            bias: load(&format!("{base}.bias"), channels)?,
        })
    }
}

struct Resnet {
    norm1: GroupNorm,
    conv1: Conv,
    norm2: GroupNorm,
    conv2: Conv,
    shortcut: Option<Conv>,
}

/// One encoder level: its resnets, then the stride-2 downsampling convolution when the level downsamples.
struct EncoderLevel {
    resnets: Vec<Resnet>,
    down: Option<Conv>,
}

struct EncoderKernels {
    im2col: CudaFunction,
    group_norm: CudaFunction,
    bias_residual: CudaFunction,
    pixels: CudaFunction,
    stitch: CudaFunction,
    posterior: CudaFunction,
}

/// Image/video encoder and quant_conv weights, F32 and resident, for single frames.
pub(crate) struct KeyframeEncoder {
    spec: VaeSpec,
    image: bool,
    conv_in: Conv,
    levels: Vec<EncoderLevel>,
    norm_out: GroupNorm,
    conv_out: Conv,
    quant: Conv,
    /// `[pixel mean | pixel std]` and `[latents_mean | latents_std]`, F32.
    pixel_table: CudaSlice<f32>,
    latent_table: CudaSlice<f32>,
    kernels: EncoderKernels,
}

/// F32 mean/std at the posterior extent and normalized latents at the output extent, channel-major.
pub(crate) type KeyframeLatents = [CudaSlice<f32>; 3];

impl KeyframeEncoder {
    pub(crate) fn new(
        d: &Device,
        spec: &VaeSpec,
        vae_dir: &Path,
        manifest: &TensorManifest,
    ) -> Result<Self> {
        Self::load(d, spec, vae_dir, manifest, false)
    }

    fn load(
        d: &Device,
        spec: &VaeSpec,
        vae_dir: &Path,
        manifest: &TensorManifest,
        image: bool,
    ) -> Result<Self> {
        let s = &d.stream;
        let (free, total) = d.ctx.mem_get_info()?;
        let expected: std::collections::BTreeSet<_> = manifest
            .entries
            .keys()
            .filter(|name| name.starts_with("encoder.") || name.starts_with("quant_conv."))
            .cloned()
            .collect();
        if image {
            let required: usize = expected
                .iter()
                .map(|name| manifest.entries[name].bytes)
                .sum();
            ensure!(
                required <= d.workspace_budget(free, free, total / 16)?,
                "image VAE encoder weights need {required} bytes"
            );
        }
        ensure!(
            spec.in_channels == VAE_PIXEL_MEAN.len(),
            "{} input channels, pixel normalisation has {}",
            spec.in_channels,
            VAE_PIXEL_MEAN.len()
        );
        let vision = cubin::module(s.context(), "vision_ops")?;
        let kernels = EncoderKernels {
            im2col: vision.load_function("causal_frame_im2col_f32")?,
            group_norm: vision.load_function("group_norm_silu_f32")?,
            bias_residual: vision.load_function("bias_residual_f32")?,
            pixels: vision.load_function(if image {
                "normalize_hwc_window_f32"
            } else {
                "normalize_u8_window_f32"
            })?,
            stitch: vision.load_function("tile_stitch_f32")?,
            posterior: vision.load_function(if image {
                "gaussian_posterior_crop_f32"
            } else {
                "gaussian_posterior_f32"
            })?,
        };
        let mut source = Source::new(vae_dir, manifest);
        let mut bytes = 0usize;
        let mut consumed = std::collections::BTreeSet::new();
        let load: &mut LoadF32<'_> = &mut |name: &str, elements: usize| {
            ensure!(!bf16_parameter(name), "{name} is in the BF16 set");
            ensure!(
                consumed.insert(name.to_owned()),
                "duplicate encoder tensor {name}"
            );
            bytes += 4 * elements;
            source.tensor::<f32>(s, name, elements)
        };
        let first = spec.block_out_channels[0];
        let conv_in = Conv::load(load, "encoder.conv_in", (first, spec.in_channels, 3))?;
        let mut levels = Vec::with_capacity(spec.block_out_channels.len());
        let mut input = first;
        for (i, &out) in spec.block_out_channels.iter().enumerate() {
            let mut resnets = Vec::with_capacity(spec.layers_per_block);
            for j in 0..spec.layers_per_block {
                let base = format!("encoder.down_blocks.{i}.resnets.{j}");
                let channels = if j == 0 { input } else { out };
                resnets.push(Resnet {
                    norm1: GroupNorm::load(load, &format!("{base}.norm1"), channels)?,
                    conv1: Conv::load(load, &format!("{base}.conv1"), (out, channels, 3))?,
                    norm2: GroupNorm::load(load, &format!("{base}.norm2"), out)?,
                    conv2: Conv::load(load, &format!("{base}.conv2"), (out, out, 3))?,
                    shortcut: if channels == out {
                        None
                    } else {
                        Some(Conv::load(
                            load,
                            &format!("{base}.conv_shortcut"),
                            (out, channels, 1),
                        )?)
                    },
                });
            }
            let down = if spec.spatial_factors[i] * spec.temporal_factors[i] > 1 {
                let base = format!("encoder.down_blocks.{i}.downsamplers.0.conv");
                Some(Conv::load(load, &base, (out, out, 3))?)
            } else {
                None
            };
            levels.push(EncoderLevel { resnets, down });
            input = out;
        }
        let latent2 = 2 * spec.latent_channels;
        let norm_out = GroupNorm::load(load, "encoder.norm_out", input)?;
        let conv_out = Conv::load(load, "encoder.conv_out", (latent2, input, 3))?;
        let quant = Conv::load(load, "quant_conv", (latent2, latent2, 1))?;
        ensure!(
            consumed == expected,
            "encoder partial loader tensor set differs"
        );
        let table =
            |a: &[f64], b: &[f64]| -> Vec<f32> { a.iter().chain(b).map(|&v| v as f32).collect() };
        let pixel_table = s.clone_htod(&table(&VAE_PIXEL_MEAN, &VAE_PIXEL_STD))?;
        let latent_table = s.clone_htod(
            &spec
                .latents_mean
                .iter()
                .chain(&spec.latents_std)
                .copied()
                .collect::<Vec<f32>>(),
        )?;
        s.synchronize()?;
        if crate::backend::diagnostics_enabled() {
            eprintln!(
                "{}",
                serde_json::json!({"h3_vae_encoder_placement": {"mode": "resident", "weight_bytes": bytes,
                    "tensors":consumed.len(),"image":image,
                    "free_before_bytes": free, "free_after_bytes": d.ctx.mem_get_info()?.0, "total_bytes": total}})
            );
        }
        Ok(Self {
            spec: spec.clone(),
            image,
            conv_in,
            levels,
            norm_out,
            conv_out,
            quant,
            pixel_table,
            latent_table,
            kernels,
        })
    }

    /// Every convolution with its stride, in execution order, with the input extent it sees for a `height x width`
    /// tile.
    fn convolutions(&self, (height, width): (usize, usize)) -> Vec<(&Conv, usize, usize, usize)> {
        let (mut h, mut w) = (height, width);
        let mut out = vec![(&self.conv_in, 1, h, w)];
        for level in &self.levels {
            for r in &level.resnets {
                out.push((&r.conv1, 1, h, w));
                out.push((&r.conv2, 1, h, w));
            }
            if let Some(down) = &level.down {
                out.push((down, 2, h, w));
                (h, w) = (down.extent(h, 2), down.extent(w, 2));
            }
        }
        out.push((&self.conv_out, 1, h, w));
        out
    }

    fn group_norm(
        &self,
        s: &Arc<CudaStream>,
        norm: &GroupNorm,
        x: &CudaSlice<f32>,
        y: &CudaSlice<f32>,
        (pixels, channels): (usize, usize),
    ) -> Result<()> {
        let groups = u32::try_from(self.spec.groups)?;
        unsafe {
            s.launch_builder(&self.kernels.group_norm)
                .arg(&pointer(x, s))
                .arg(&pointer(&norm.weight, s))
                .arg(&pointer(&norm.bias, s))
                .arg(&pointer(y, s))
                .arg(&(pixels as u64))
                .arg(&u32::try_from(channels)?)
                .arg(&groups)
                .arg(&self.spec.encoder_eps)
                .arg(&0u32)
                .arg(&0u32)
                .launch(LaunchConfig {
                    grid_dim: (groups, 1, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    fn bias_residual(
        &self,
        s: &Arc<CudaStream>,
        (y, offset, count): (&CudaSlice<f32>, usize, usize),
        bias: &CudaSlice<f32>,
        residual: Option<&CudaSlice<f32>>,
        (channels, inner): (usize, usize),
    ) -> Result<()> {
        let target = pointer(y, s) + (offset * std::mem::size_of::<f32>()) as u64;
        let residual = residual.map_or(0, |r| pointer(r, s));
        unsafe {
            s.launch_builder(&self.kernels.bias_residual)
                .arg(&target)
                .arg(&pointer(bias, s))
                .arg(&residual)
                .arg(&(count as u64))
                .arg(&u32::try_from(channels)?)
                .arg(&(inner as u64))
                .launch(flat(count))?;
        }
        Ok(())
    }

    /// `y [out_h * out_w, out] = conv(x [h * w, input]) + bias (+ residual)`: im2col row chunks that fit `columns`,
    /// each one F32 GEMM against the published `[out, input * k^3]` weight. Returns the output extent.
    fn conv(
        &self,
        (s, blas, columns): (&Arc<CudaStream>, &Blas, &CudaSlice<f32>),
        c: &Conv,
        (x, height, width, stride): (&CudaSlice<f32>, usize, usize, usize),
        y: &CudaSlice<f32>,
        residual: Option<&CudaSlice<f32>>,
    ) -> Result<(usize, usize)> {
        let (out_h, out_w) = (c.extent(height, stride), c.extent(width, stride));
        let pixels = out_h * out_w;
        let k = c.columns();
        ensure!(
            pixels * c.out <= y.len() && height * width * c.input <= x.len(),
            "convolution {height}x{width}x{} -> {out_h}x{out_w}x{} exceeds its buffers",
            c.input,
            c.out
        );
        let weight = view(
            &c.weight,
            s,
            0,
            [u32::try_from(c.out)?, u32::try_from(k)?],
            [k as u64, 1],
        )?;
        if c.kernel == 1 {
            ensure!(stride == 1, "strided 1x1x1 convolution");
            blas.linear_f32(
                weight,
                view(
                    x,
                    s,
                    0,
                    [u32::try_from(pixels)?, u32::try_from(k)?],
                    [k as u64, 1],
                )?,
                &mut TensorMut(view(
                    y,
                    s,
                    0,
                    [u32::try_from(pixels)?, u32::try_from(c.out)?],
                    [c.out as u64, 1],
                )?),
            )?;
        } else {
            let chunk = (columns.len() / k) / 4 * 4;
            ensure!(chunk > 0, "im2col workspace holds no row of {k} columns");
            let pad = if stride == 1 { c.kernel / 2 } else { 0 };
            for first in (0..pixels).step_by(chunk) {
                let rows = chunk.min(pixels - first);
                unsafe {
                    s.launch_builder(&self.kernels.im2col)
                        .arg(&pointer(x, s))
                        .arg(&pointer(columns, s))
                        .arg(&u32::try_from(c.input)?)
                        .arg(&u32::try_from(height)?)
                        .arg(&u32::try_from(width)?)
                        .arg(&u32::try_from(out_w)?)
                        .arg(&u32::try_from(c.kernel)?)
                        .arg(&u32::try_from(c.kernel)?)
                        .arg(&u32::try_from(stride)?)
                        .arg(&u32::try_from(pad)?)
                        .arg(&(first as u64))
                        .arg(&u32::try_from(rows)?)
                        .arg(&0u32)
                        .launch(flat(rows * k))?;
                }
                blas.linear_f32(
                    weight,
                    view(
                        columns,
                        s,
                        0,
                        [u32::try_from(rows)?, u32::try_from(k)?],
                        [k as u64, 1],
                    )?,
                    &mut TensorMut(view(
                        y,
                        s,
                        first * c.out,
                        [u32::try_from(rows)?, u32::try_from(c.out)?],
                        [c.out as u64, 1],
                    )?),
                )?;
            }
        }
        self.bias_residual(s, (y, 0, pixels * c.out), &c.bias, residual, (c.out, 1))?;
        Ok((out_h, out_w))
    }

    /// Encodes the tile at pixel `origin` of `frame` (`[in_channels, height, width]` u8) into moments slot `slot` of
    /// `moments` (`[slots, 2 * latent_channels, tile_h / r, tile_w / r]`, channel-major).
    fn tile<T: DeviceRepr>(
        &self,
        (s, blas): (&Arc<CudaStream>, &Blas),
        w: &mut EncodeWork,
        (frame, height, width): (&CudaSlice<T>, usize, usize),
        origin: (usize, usize),
        (moments, slot): (&CudaSlice<f32>, usize),
    ) -> Result<()> {
        let channels = self.spec.in_channels;
        let (mut h, mut wd) = (w.height, w.width);
        unsafe {
            s.launch_builder(&self.kernels.pixels)
                .arg(&pointer(frame, s))
                .arg(&pointer(&self.pixel_table, s))
                .arg(&pointer(&w.x, s))
                .arg(&u32::try_from(channels)?)
                .arg(&u32::try_from(height)?)
                .arg(&u32::try_from(width)?)
                .arg(&u32::try_from(origin.0)?)
                .arg(&u32::try_from(origin.1)?)
                .arg(&u32::try_from(h)?)
                .arg(&u32::try_from(wd)?)
                .launch(flat(h * wd * channels))?;
        }
        let ctx = (s, blas, &w.columns);
        self.conv(ctx, &self.conv_in, (&w.x, h, wd, 1), &w.y, None)?;
        std::mem::swap(&mut w.x, &mut w.y);
        for level in &self.levels {
            for r in &level.resnets {
                let px = h * wd;
                self.group_norm(s, &r.norm1, &w.x, &w.normed, (px, r.conv1.input))?;
                let ctx = (s, blas, &w.columns);
                self.conv(ctx, &r.conv1, (&w.normed, h, wd, 1), &w.hidden, None)?;
                self.group_norm(s, &r.norm2, &w.hidden, &w.normed, (px, r.conv1.out))?;
                let residual = match &r.shortcut {
                    Some(shortcut) => {
                        self.conv(ctx, shortcut, (&w.x, h, wd, 1), &w.shortcut, None)?;
                        &w.shortcut
                    }
                    None => &w.x,
                };
                self.conv(ctx, &r.conv2, (&w.normed, h, wd, 1), &w.y, Some(residual))?;
                std::mem::swap(&mut w.x, &mut w.y);
            }
            if let Some(down) = &level.down {
                let ctx = (s, blas, &w.columns);
                (h, wd) = self.conv(ctx, down, (&w.x, h, wd, 2), &w.y, None)?;
                std::mem::swap(&mut w.x, &mut w.y);
            }
        }
        let px = h * wd;
        self.group_norm(
            s,
            &self.norm_out,
            &w.x,
            &w.normed,
            (px, self.conv_out.input),
        )?;
        let ctx = (s, blas, &w.columns);
        self.conv(ctx, &self.conv_out, (&w.normed, h, wd, 1), &w.y, None)?;
        let q = &self.quant;
        let size = q.out * px;
        ensure!(
            (slot + 1) * size <= moments.len(),
            "moments slot {slot} out of range"
        );
        blas.linear_f32(
            view(
                &w.y,
                s,
                0,
                [u32::try_from(px)?, u32::try_from(q.input)?],
                [q.input as u64, 1],
            )?,
            view(
                &q.weight,
                s,
                0,
                [u32::try_from(q.out)?, u32::try_from(q.input)?],
                [q.input as u64, 1],
            )?,
            &mut TensorMut(view(
                moments,
                s,
                slot * size,
                [u32::try_from(q.out)?, u32::try_from(px)?],
                [px as u64, 1],
            )?),
        )?;
        self.bias_residual(s, (moments, slot * size, size), &q.bias, None, (q.out, px))
    }

    /// Encodes one u8 frame `[in_channels, height, width]` through `tiles`: posterior moments of every tile, stitched
    /// in latent space, then mean, std and normalised latents of the sample `mean + std * eps` rounded to F16, with
    /// `eps` `[latent_channels, height / r, width / r]` F32.
    pub(crate) fn encode(
        &self,
        (s, blas): (&Arc<CudaStream>, &Blas),
        w: &mut EncodeWork,
        tiles: &VaeTiles,
        (frame, height, width): (&CudaSlice<u8>, usize, usize),
        eps: &CudaSlice<f32>,
    ) -> Result<KeyframeLatents> {
        self.encode_frame(
            (s, blas),
            w,
            tiles,
            (frame, height, width),
            eps,
            [height, width],
        )
    }

    fn encode_frame<T: DeviceRepr>(
        &self,
        (s, blas): (&Arc<CudaStream>, &Blas),
        w: &mut EncodeWork,
        tiles: &VaeTiles,
        (frame, source_h, source_w): (&CudaSlice<T>, usize, usize),
        eps: &CudaSlice<f32>,
        [height, width]: [usize; 2],
    ) -> Result<KeyframeLatents> {
        let spec = &self.spec;
        let r = spec.spatial_ratio;
        ensure!(
            w.height == tiles.height
                && w.width == tiles.width
                && height.is_multiple_of(r)
                && width.is_multiple_of(r)
                && height <= source_h
                && width <= source_w
                && frame.len() == spec.in_channels * source_h * source_w,
            "encode work {}x{} / frame {height}x{width} do not match the tile grid {}x{}",
            w.height,
            w.width,
            tiles.height,
            tiles.width
        );
        let (lh, lw) = (height / r, width / r);
        let plane = lh * lw;
        let latent = spec.latent_channels;
        ensure!(
            eps.len() == latent * plane,
            "posterior noise holds {} values, not {}",
            eps.len(),
            latent * plane
        );
        let (rows, cols) = (&tiles.rows, &tiles.cols);
        let grid: Vec<u32> = rows
            .0
            .iter()
            .chain(&rows.1)
            .chain(&cols.0)
            .chain(&cols.1)
            .map(|&v| u32::try_from(v / r))
            .collect::<Result<_, _>>()?;
        let grid = s.clone_htod(&grid)?;
        let (th, tw) = (tiles.height / r, tiles.width / r);
        let moments = s.alloc_zeros::<f32>(tiles.count() * 2 * latent * th * tw)?;
        for (ti, &row) in rows.0.iter().enumerate() {
            for (tj, &col) in cols.0.iter().enumerate() {
                self.tile(
                    (s, blas),
                    w,
                    (frame, source_h, source_w),
                    (row, col),
                    (&moments, ti * cols.0.len() + tj),
                )?;
            }
        }
        let canvas = s.alloc_zeros::<f32>(2 * latent * plane)?;
        unsafe {
            s.launch_builder(&self.kernels.stitch)
                .arg(&pointer(&moments, s))
                .arg(&pointer(&canvas, s))
                .arg(&pointer(&grid, s))
                .arg(&(rows.0.len() as u32))
                .arg(&(cols.0.len() as u32))
                .arg(&u32::try_from(2 * latent)?)
                .arg(&u32::try_from(th)?)
                .arg(&u32::try_from(tw)?)
                .arg(&u32::try_from(lh)?)
                .arg(&u32::try_from(lw)?)
                .launch(flat(2 * latent * plane))?;
        }
        let out = [
            s.alloc_zeros::<f32>(latent * plane)?,
            s.alloc_zeros::<f32>(latent * plane)?,
            s.alloc_zeros::<f32>(
                latent
                    * if self.image {
                        (lh / 2 * 2) * (lw / 2 * 2)
                    } else {
                        plane
                    },
            )?,
        ];
        unsafe {
            let pointers = [
                pointer(&canvas, s),
                pointer(eps, s),
                pointer(&self.latent_table, s),
                pointer(&out[0], s),
                pointer(&out[1], s),
                pointer(&out[2], s),
            ];
            let channels = u32::try_from(latent)?;
            let dimensions = [
                u32::try_from(lh)?,
                u32::try_from(lw)?,
                u32::try_from(lh / 2 * 2)?,
                u32::try_from(lw / 2 * 2)?,
            ];
            let plane = plane as u64;
            let mut launch = s.launch_builder(&self.kernels.posterior);
            for p in &pointers {
                launch.arg(p);
            }
            launch.arg(&channels);
            if self.image {
                for d in &dimensions {
                    launch.arg(d);
                }
            } else {
                launch.arg(&plane);
            }
            launch.launch(flat(latent * plane as usize))?;
        }
        Ok(out)
    }
}

/// Activations of one encoder tile forward over `height x width` pixels, pixel-major `[pixels, channels]` F32, and
/// the im2col workspace: the largest convolution of a released `VAE_TILE` tile in one chunk, larger frames in row
/// chunks.
pub(crate) struct EncodeWork {
    height: usize,
    width: usize,
    x: CudaSlice<f32>,
    y: CudaSlice<f32>,
    normed: CudaSlice<f32>,
    hidden: CudaSlice<f32>,
    shortcut: CudaSlice<f32>,
    columns: CudaSlice<f32>,
}

impl EncodeWork {
    pub(crate) fn new(
        s: &Arc<CudaStream>,
        encoder: &KeyframeEncoder,
        (height, width): (usize, usize),
    ) -> Result<Self> {
        Self::with_budget(s, encoder, (height, width), usize::MAX)
    }

    fn with_budget(
        s: &Arc<CudaStream>,
        encoder: &KeyframeEncoder,
        (height, width): (usize, usize),
        budget: usize,
    ) -> Result<Self> {
        let activations = encoder
            .convolutions((height, width))
            .iter()
            .map(|(c, stride, h, w)| {
                (h * w * c.input).max(c.extent(*h, *stride) * c.extent(*w, *stride) * c.out)
            })
            .max()
            .context("encoder without convolutions")?;
        let columns = |(h, w): (usize, usize)| -> Result<usize> {
            encoder
                .convolutions((h, w))
                .iter()
                .filter(|(c, ..)| c.kernel > 1)
                .map(|(c, stride, h, w)| {
                    c.extent(*h, *stride) * c.extent(*w, *stride) * c.columns()
                })
                .max()
                .context("encoder without spatial convolutions")
        };
        let fixed = 5 * activations * 4;
        let available = budget
            .checked_sub(fixed)
            .context("encoder budget holds no activation set")?;
        let workspace = columns((height, width))?
            .min(columns((VAE_TILE, VAE_TILE))?)
            .min(available / 4);
        let minimum = encoder
            .convolutions((height, width))
            .iter()
            .filter(|(c, ..)| c.kernel > 1)
            .map(|(c, ..)| 4 * c.columns())
            .max()
            .context("encoder without spatial convolutions")?;
        ensure!(
            workspace >= minimum,
            "encoder im2col budget needs at least {} bytes",
            4 * minimum
        );
        eprintln!(
            "{}",
            serde_json::json!({"vae_encoder_work":{
            "activation_bytes":fixed,"column_bytes":4 * workspace,"minimum_column_bytes":4 * minimum,
            "budget":budget,"tile":[height,width]}})
        );
        let buffer = || s.alloc_zeros::<f32>(activations);
        Ok(Self {
            height,
            width,
            x: buffer()?,
            y: buffer()?,
            normed: buffer()?,
            hidden: buffer()?,
            shortcut: buffer()?,
            columns: s.alloc_zeros::<f32>(workspace)?,
        })
    }
}
