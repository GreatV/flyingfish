//! MiniMax-H3 image/video VAEs: tiled ViT decoders and the causal CNN keyframe encoder.
use super::{Source, pointer, rows_view, vector};
use crate::backend::cuda::{
    Device,
    blas::Blas,
    contracts::{Tensor, TensorMut, view},
    cubin,
    ops::{self, flat},
};
use crate::backend::gpu::{
    CudaFunction, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg,
};
use crate::checkpoint::TensorManifest;
use crate::minimax_h3::{DecodePlan, VAE_PIXEL_MEAN, VAE_PIXEL_STD, VAE_TILE, VaeSpec, VaeTiles};
use anyhow::{Context, Result, ensure};
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
        let pixels = canvas[0] as u128 * canvas[1] as u128;
        let grid_words = 2 * (tiles.rows.0.len() + tiles.cols.0.len()) - 2;
        let fixed = 4u128
            * slots as u128
            * spec.out_channels as u128
            * tiles.height as u128
            * tiles.width as u128
            + 8 * pixels * spec.out_channels as u128
            + 4u128 * (grid_words + tokens * spec.rope_dims + spec.head_dim) as u128;
        Self::derive_counts(spec, slots, patches, fixed, limits)
    }

    fn derive_counts(
        spec: &VaeSpec,
        slots: usize,
        patches: usize,
        fixed: u128,
        limits: ImageLimits,
    ) -> Result<Self> {
        let tokens = patches + spec.registers + 1;
        let s = 4u128 * tokens as u128 * spec.dim as u128;
        let p = 4u128 * spec.heads as u128 * (tokens as u128).pow(2);
        let per_tile = 4u128 * tokens as u128 * (7 * spec.dim + 3 * spec.ffn) as u128
            + 4u128 * tokens as u128 * spec.patch_width() as u128
            + 8u128 * patches as u128 * spec.latent_channels as u128;
        ensure!(
            slots > 0 && patches > 0 && limits.sms > 0 && limits.grid > 0,
            "image VAE setup has an empty shape or device limit"
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

pub(crate) struct DecoderWork {
    plan: ImagePlan,
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
}

impl DecoderWork {
    fn new(s: &Arc<CudaStream>, spec: &VaeSpec, plan: ImagePlan, rope: Vec<f32>) -> Result<Self> {
        let n = plan.linear * plan.tokens * spec.dim;
        let latent = plan.linear * plan.patches * spec.latent_channels;
        // Every active element is written before it is read.
        let alloc = |count| unsafe { s.alloc::<f32>(count) };
        Ok(Self {
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
            plan,
        })
    }
}

struct ImageWork {
    core: DecoderWork,
    tiles: VaeTiles,
    canvas: [usize; 2],
    grid: CudaSlice<u32>,
    raw: CudaSlice<f32>,
    planar: CudaSlice<f32>,
    rgb: CudaSlice<f32>,
}

impl std::ops::Deref for ImageWork {
    type Target = DecoderWork;
    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

type ImageRows<'a> = (&'a CudaSlice<f32>, usize, usize, usize);

pub(crate) struct ImageDecoder {
    spec: VaeSpec,
    post_quant: Linear,
    proj_in: Linear,
    registers: Tensor<f32, 1>,
    norm_out: (Tensor<f32, 1>, Tensor<f32, 1>),
    proj_out: Linear,
    blocks: Vec<DecoderBlock>,
    latent_affine: Option<(CudaSlice<f32>, CudaSlice<f32>)>,
    pixel_affine: Option<(CudaSlice<f32>, CudaSlice<f32>)>,
    kernels: ImageKernels,
    _weights: Vec<CudaSlice<f32>>,
    dense: Option<std::cell::RefCell<crate::backend::cuda::weights::DenseStreaming>>,
}

impl Linear {
    fn load(
        load: &mut impl FnMut(&str, usize) -> Result<Tensor<f32, 1>>,
        base: &str,
        out: usize,
        input: usize,
    ) -> Result<Self> {
        Ok(Self {
            weight: {
                let data = load(&format!("{base}.weight"), out * input)?;
                unsafe {
                    Tensor::new(
                        data.ptr,
                        [out as u32, input as u32],
                        [input as u64, 1],
                        data.span_bytes,
                    )?
                }
            },
            bias: load(&format!("{base}.bias"), out)?,
            out,
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
        blas.linear_f32(self.weight, x, &mut TensorMut(y))?;
        unsafe {
            s.launch_builder(bias)
                .arg(&y.ptr)
                .arg(&self.bias.ptr)
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
        Self::load(device, spec, component, manifest, None)
    }

    fn load(
        device: &Device,
        spec: VaeSpec,
        component: &Path,
        manifest: &TensorManifest,
        placement: Option<(usize, super::Placement)>,
    ) -> Result<Self> {
        let image = placement.is_none();
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
        let mut source = Source::new(component, manifest);
        let mut weights = Vec::new();
        let mut consumed = std::collections::BTreeSet::new();
        let mut order = Vec::new();
        let mut sizes = Vec::new();
        for layer in 0..spec.layers {
            let prefix = format!("decoder.transformer_blocks.{layer}.");
            let names: Vec<_> = expected
                .iter()
                .filter(|n| n.starts_with(&prefix))
                .cloned()
                .collect();
            ensure!(
                names.len() == 16,
                "VAE block {layer} tensor census {}",
                names.len()
            );
            sizes.push(
                names
                    .iter()
                    .map(|n| manifest.entries[n].bytes)
                    .sum::<usize>(),
            );
            order.push(names);
        }
        let fixed = weight_bytes - sizes.iter().sum::<usize>();
        let plan = if let Some((future, mode)) = placement {
            let mut budget =
                crate::prefill::DensePlacement::device_budget(free, total, fixed, 0, future)?;
            if mode != super::Placement::Derived {
                let slots = 2 * sizes.iter().max().context("VAE has no blocks")?;
                ensure!(slots <= budget, "VAE streamed slots exceed derived budget");
                budget = slots;
            }
            let host = if sizes.iter().sum::<usize>() > budget {
                let available = usize::try_from(crate::backend::cuda::calibrate::host_memory(
                    Path::new("/"),
                )?)?;
                available
                    .checked_sub(4 << 30)
                    .context("VAE host reserve exceeds available RAM")?
            } else {
                0
            };
            let plan = crate::prefill::DensePlacement::new(&sizes, Some(&sizes), budget, host)?;
            ensure!(
                plan.disk_bytes == 0,
                "VAE pinned host plan exceeds available RAM"
            );
            if mode != super::Placement::Derived {
                ensure!(
                    plan.resident_bytes == 0,
                    "forced streamed VAE retained blocks"
                );
            }
            if crate::backend::diagnostics_enabled() {
                eprintln!(
                    "{}",
                    serde_json::json!({"vae_f32_placement":{
                    "weight_bytes":weight_bytes,"fixed_bytes":fixed,"resident_block_bytes":plan.resident_bytes,
                    "slot_bytes":2*plan.slot_bytes,"pinned_bytes":plan.host_bytes,"future_bytes":future,
                    "free_bytes":free,"total_bytes":total,"reserve_bytes":total/16,"dtype":"F32"}})
                );
            }
            Some(plan)
        } else {
            ensure!(
                weight_bytes <= device.workspace_budget(free, free, total / 16)?,
                "image VAE decoder weights need {weight_bytes} bytes"
            );
            None
        };
        let mut dense = if let Some(plan) = plan {
            if plan.slot_bytes > 0 {
                Some(crate::backend::cuda::weights::DenseStreaming::new(
                    component,
                    manifest,
                    plan,
                    device,
                    Some(&order),
                    0,
                )?)
            } else {
                None
            }
        } else {
            None
        };
        if let Some(streaming) = &mut dense {
            for (layer, names) in order.iter().enumerate() {
                if let Some(host) = streaming.host_layer(layer)? {
                    let mut offset = 0;
                    for name in names {
                        let entry = &manifest.entries[name];
                        source.reader.read_into(
                            name,
                            entry,
                            &mut host[offset..offset + entry.bytes],
                        )?;
                        offset += entry.bytes;
                    }
                    ensure!(offset == host.len(), "VAE pinned block byte count");
                }
            }
        }
        let mut load = |name: &str, count: usize| -> Result<Tensor<f32, 1>> {
            ensure!(
                consumed.insert(name.to_owned()),
                "VAE duplicate tensor {name}"
            );
            let entry = manifest
                .entries
                .get(name)
                .with_context(|| format!("VAE tensor {name} missing"))?;
            ensure!(
                entry.dtype == safetensors::Dtype::F32
                    && entry.quant.is_none()
                    && entry.bytes == 4 * count,
                "VAE tensor {name} is not published F32"
            );
            if let Some(streaming) = &dense {
                for (layer, names) in order.iter().enumerate() {
                    if streaming.plan.slots[layer].is_some() {
                        let mut offset = 0;
                        for key in names {
                            if key == name {
                                let ptr = streaming.pointer(layer, offset, entry.bytes)?;
                                return unsafe {
                                    Tensor::new(
                                        ptr,
                                        [u32::try_from(count)?],
                                        [1],
                                        entry.bytes as u64,
                                    )
                                };
                            }
                            offset += manifest.entries[key].bytes;
                        }
                    }
                }
            }
            let data = source.tensor::<f32>(s, name, count)?;
            let tensor = vector(&data, s)?;
            weights.push(data);
            Ok(tensor)
        };
        let (dim, latent, ffn) = (spec.dim, spec.latent_channels, spec.ffn);
        let post_quant = Linear::load(&mut load, "post_quant_conv", latent, latent)?;
        let proj_in = Linear::load(&mut load, "decoder.proj_in", dim, latent)?;
        let registers = load("decoder.register_tokens", spec.registers * dim)?;
        let norm_out = (
            load("decoder.norm_out.weight", dim)?,
            load("decoder.norm_out.bias", dim)?,
        );
        let proj_out = Linear::load(&mut load, "decoder.proj_out", spec.patch_width(), dim)?;
        let mut blocks = Vec::with_capacity(spec.layers);
        for layer in 0..spec.layers {
            let b = format!("decoder.transformer_blocks.{layer}");
            blocks.push(DecoderBlock {
                q: Linear::load(&mut load, &format!("{b}.attn.to_q"), dim, dim)?,
                k: Linear::load(&mut load, &format!("{b}.attn.to_k"), dim, dim)?,
                v: Linear::load(&mut load, &format!("{b}.attn.to_v"), dim, dim)?,
                out: Linear::load(&mut load, &format!("{b}.attn.to_out.0"), dim, dim)?,
                ff_in: Linear::load(&mut load, &format!("{b}.ff.net.0.proj"), 2 * ffn, dim)?,
                ff_out: Linear::load(&mut load, &format!("{b}.ff.net.2"), dim, ffn)?,
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
        if crate::backend::diagnostics_enabled() {
            eprintln!(
                "{}",
                serde_json::json!({"image_vae_decoder_load":{"tensors":consumed.len(),"weight_bytes":weight_bytes,"dtype":"F32"}})
            );
        }
        Ok(Self {
            spec: spec.clone(),
            _weights: weights,
            dense: dense.map(std::cell::RefCell::new),
            post_quant,
            proj_in,
            registers,
            norm_out,
            proj_out,
            blocks,
            kernels,
            latent_affine: if image {
                Some((
                    s.clone_htod(&spec.latents_std)?,
                    s.clone_htod(&spec.latents_mean)?,
                ))
            } else {
                None
            },
            pixel_affine: if image {
                Some((
                    s.clone_htod(&VAE_PIXEL_STD.map(|x| x as f32))?,
                    s.clone_htod(&VAE_PIXEL_MEAN.map(|x| x as f32))?,
                ))
            } else {
                None
            },
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
            core: DecoderWork::new(s, spec, plan, rope)?,
            raw: alloc(raw)?,
            planar: alloc(pixels)?,
            rgb: alloc(pixels)?,
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
        if crate::backend::diagnostics_enabled() {
            eprintln!(
                "{}",
                serde_json::json!({"image_vae_plan":{"canvas":canvas,"tiles":work.tiles.count(),
                "tokens":work.plan.tokens,"linear_batch":work.plan.linear,"attention_batch":work.plan.attention,
                "bytes":actual,"workspace_budget":budget,"silu_grid":flat(work.plan.linear*work.plan.tokens*spec.ffn).grid_dim.0}})
            );
        }
        Ok(work)
    }

    fn rms(
        &self,
        s: &Arc<CudaStream>,
        input: Tensor<f32, 2>,
        weight: Tensor<f32, 1>,
        output: u64,
    ) -> Result<()> {
        unsafe {
            s.launch_builder(&self.kernels.rms)
                .arg(&input.ptr)
                .arg(&weight.ptr)
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
        w: &DecoderWork,
        branch: &CudaSlice<f32>,
        scale: Tensor<f32, 1>,
        rows: usize,
    ) -> Result<()> {
        unsafe {
            s.launch_builder(&self.kernels.residual)
                .arg(&pointer(&w.x, s))
                .arg(&pointer(branch, s))
                .arg(&scale.ptr)
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
        w: &DecoderWork,
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
        block: &DecoderBlock,
        w: &DecoderWork,
        group: (usize, usize, usize),
        observe: &mut impl FnMut(&str, usize, usize, ImageRows<'_>) -> Result<()>,
    ) -> Result<()> {
        let spec = &self.spec;
        let (first, active, layer) = group;
        let rows = active * w.plan.tokens;
        let n = rows * spec.dim;
        let states = |data, offset| rows_view(data, s, offset, rows, spec.dim, spec.dim);
        self.rms(s, states(&w.x, 0)?, block.norm1, pointer(&w.norm, s))?;
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
            self.rms(s, heads, vector(&w.ones, s)?, heads.ptr)?;
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
        self.residual(s, w, &w.ff, block.scale1, rows)?;
        self.rms(s, states(&w.x, 0)?, block.norm2, pointer(&w.norm, s))?;
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
        self.residual(s, w, &w.o, block.scale2, rows)?;
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

    fn output_rows(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        w: &DecoderWork,
        rows: usize,
    ) -> Result<()> {
        let spec = &self.spec;
        let dim = spec.dim as u64;
        unsafe {
            s.launch_builder(&self.kernels.norm)
                .arg(&pointer(&w.x, s))
                .arg(&self.norm_out.0.ptr)
                .arg(&self.norm_out.1.ptr)
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
        let latent_affine = self
            .latent_affine
            .as_ref()
            .context("image VAE latent affine missing")?;
        let pixel_affine = self
            .pixel_affine
            .as_ref()
            .context("image VAE pixel affine missing")?;
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
                    .arg(&pointer(&latent_affine.0, s))
                    .arg(&pointer(&latent_affine.1, s))
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
                    .arg(&self.registers.ptr)
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
            self.output_rows(s, blas, w, rows)?;
            observe("norm_out", first, active, (&w.norm, 0, p.tokens, spec.dim))?;
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
                .arg(&pointer(&pixel_affine.0, s))
                .arg(&pointer(&pixel_affine.1, s))
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

pub(crate) fn planar_f32_to_rgb_u8(
    device: &Device,
    kernel: &CudaFunction,
    input: &impl DevicePtr<f32>,
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
        .checked_mul(4)
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

struct Linear {
    weight: Tensor<f32, 2>,
    bias: Tensor<f32, 1>,
    out: usize,
}

struct DecoderBlock {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    ff_in: Linear,
    ff_out: Linear,
    norm1: Tensor<f32, 1>,
    norm2: Tensor<f32, 1>,
    scale1: Tensor<f32, 1>,
    scale2: Tensor<f32, 1>,
}

struct VaeKernels {
    frame_blend: CudaFunction,
    latent_rows: CudaFunction,
    rgb: CudaFunction,
    stitch: CudaFunction,
}

pub(crate) struct VideoDecoder {
    core: ImageDecoder,
    kernels: VaeKernels,
    #[cfg(test)]
    prefix_only: bool,
}

impl std::ops::Deref for VideoDecoder {
    type Target = ImageDecoder;
    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl VideoDecoder {
    pub(crate) fn new(
        d: &Device,
        spec: &VaeSpec,
        dir: &Path,
        manifest: &TensorManifest,
        placement: (usize, super::Placement),
    ) -> Result<Self> {
        let vision = cubin::module(&d.ctx, "vision_ops")?;
        Ok(Self {
            core: ImageDecoder::load(d, spec.clone(), dir, manifest, Some(placement))?,
            kernels: VaeKernels {
                frame_blend: vision.load_function("frame_blend_f32")?,
                latent_rows: vision.load_function("latent_rows_f32")?,
                rgb: vision.load_function("planar_f32_to_rgb_u8")?,
                stitch: vision.load_function("tile_stitch_f32")?,
            },
            #[cfg(test)]
            prefix_only: false,
        })
    }

    pub(crate) fn rgb(
        &self,
        d: &Device,
        decoded: &CudaSlice<f32>,
        shape: (usize, usize, usize),
    ) -> Result<CudaSlice<u8>> {
        let count = shape
            .0
            .checked_mul(shape.2)
            .and_then(|n| n.checked_mul(3))
            .context("RGB output overflow")?;
        let (free, total) = d.ctx.mem_get_info()?;
        ensure!(
            count <= d.workspace_budget(free, free, total / 16)?,
            "RGB output exceeds memory budget"
        );
        let output = unsafe { d.stream.alloc::<u8>(count)? };
        planar_f32_to_rgb_u8(
            d,
            &self.kernels.rgb,
            decoded,
            &output,
            shape,
            (
                VAE_PIXEL_MEAN.map(|v| v as f32),
                VAE_PIXEL_STD.map(|v| v as f32),
            ),
        )?;
        Ok(output)
    }

    pub(crate) fn work(
        &self,
        d: &Device,
        _blas: &mut Blas,
        tiles: &VaeTiles,
        shape: (usize, usize, usize),
    ) -> Result<TileWork> {
        use cudarc::driver::sys::CUdevice_attribute as A;
        let spec = &self.spec;
        let tile = (
            spec.chunk_latents() + spec.overlap_latents(),
            tiles.height / spec.spatial_ratio,
            tiles.width / spec.spatial_ratio,
        );
        let plan = DecodePlan::new(spec, shape.0)?;
        let total_frames = plan.chunks
            * (spec.chunk_latents() * spec.temporal_ratio - spec.frame_pre_padding())
            + tile.0 * spec.temporal_ratio
            - spec.chunk_latents() * spec.temporal_ratio
            - spec.frame_pre_padding();
        let canvas = (shape.1 * spec.spatial_ratio, shape.2 * spec.spatial_ratio);
        let mandatory = decode_bytes(spec, tiles, canvas)
            + 4 * spec.out_channels * total_frames * canvas.0 * canvas.1
            + 4 * spec.latent_channels * shape.0 * shape.1 * shape.2;
        let (free, total) = d.ctx.mem_get_info()?;
        let budget = crate::prefill::DensePlacement::device_budget(free, total, mandatory, 0, 0)?;
        let patches = tile.0 * tile.1 * tile.2;
        let tokens = patches + spec.registers + 1;
        let fixed = 4u128 * (tokens * spec.rope_dims + spec.head_dim) as u128;
        let plan = ImagePlan::derive_counts(
            spec,
            tiles.count(),
            patches,
            fixed,
            ImageLimits {
                budget,
                l2: usize::try_from(d.ctx.attribute(A::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)?)?,
                sms: usize::try_from(
                    d.ctx
                        .attribute(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
                )?,
                grid: usize::try_from(d.ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?)?,
            },
        )?;
        if crate::backend::diagnostics_enabled() {
            eprintln!(
                "{}",
                serde_json::json!({"video_vae_f32_plan":{
            "linear_batch":plan.linear,"attention_batch":plan.attention,"work_bytes":plan.bytes,"future_bytes":mandatory,"budget":budget}})
            );
        }
        TileWork::with_plan(&d.stream, spec, tile, plan)
    }

    #[cfg(test)]
    pub(crate) fn batch_setup(
        &self,
        d: &Device,
        blas: &mut Blas,
        tiles: &VaeTiles,
        shape: (usize, usize, usize),
    ) -> Result<()> {
        let work = self.work(d, blas, tiles, shape)?;
        println!(
            "PASS VIDEO_F32_SETUP linear={} attention={} bytes={}",
            work.batch,
            work.plan.attention,
            work.allocated_bytes()
        );
        Ok(())
    }

    fn tile(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        w: &mut TileWork,
        z: &LatentVideo<'_>,
        (origins, first_frame): (&[(usize, usize)], usize),
        (raw, slot): (&CudaSlice<f32>, usize),
    ) -> Result<()> {
        let spec = &self.spec;
        let active = origins.len();
        ensure!(active > 0 && active <= w.batch, "video tile batch capacity");
        ensure!(
            (slot + active) * w.pixels() <= raw.len(),
            "video raw tile capacity"
        );
        let p = w.patches;
        for (tile, &(row, col)) in origins.iter().enumerate() {
            ensure!(
                first_frame + w.frames <= z.frames
                    && row + w.height <= z.height
                    && col + w.width <= z.width,
                "video tile input capacity"
            );
            let target = pointer(&w.latent, s) + (4 * tile * p * spec.latent_channels) as u64;
            unsafe {
                s.launch_builder(&self.kernels.latent_rows)
                    .arg(&pointer(z.data, s))
                    .arg(&target)
                    .arg(&(spec.latent_channels as u32))
                    .arg(&(z.frames as u32))
                    .arg(&(z.height as u32))
                    .arg(&(z.width as u32))
                    .arg(&(first_frame as u32))
                    .arg(&(row as u32))
                    .arg(&(col as u32))
                    .arg(&(w.frames as u32))
                    .arg(&(w.height as u32))
                    .arg(&(w.width as u32))
                    .launch(flat(p * spec.latent_channels))?;
            }
        }
        let latent = |data| {
            rows_view(
                data,
                s,
                0,
                active * p,
                spec.latent_channels,
                spec.latent_channels,
            )
        };
        self.post_quant.run(
            s,
            blas,
            &self.core.kernels.bias,
            latent(&w.latent)?,
            latent(&w.latent_q)?,
        )?;
        self.proj_in.run(
            s,
            blas,
            &self.core.kernels.bias,
            latent(&w.latent_q)?,
            rows_view(&w.o, s, 0, active * p, spec.dim, spec.dim)?,
        )?;
        #[cfg(test)]
        if self.prefix_only {
            return Ok(());
        }
        let rows = active * w.tokens;
        unsafe {
            s.launch_builder(&self.core.kernels.append)
                .arg(&pointer(&w.o, s))
                .arg(&self.registers.ptr)
                .arg(&pointer(&w.x, s))
                .arg(&(active as u32))
                .arg(&(p as u32))
                .arg(&(spec.registers as u32))
                .arg(&(spec.dim as u32))
                .launch(w.plan.flat(rows * spec.dim))?;
        }
        for (layer, block) in self.blocks.iter().enumerate() {
            if let Some(dense) = &self.dense {
                dense.borrow_mut().begin(layer, s)?;
            }
            self.core.block(
                s,
                blas,
                block,
                &w.core,
                (slot, active, layer),
                &mut |_, _, _, _| Ok(()),
            )?;
            if let Some(dense) = &self.dense {
                dense.borrow_mut().end(layer, s)?;
            }
        }
        self.core.output_rows(s, blas, &w.core, rows)?;
        for tile in 0..active {
            let source =
                pointer(&w.patch_rows, s) + (4 * tile * w.tokens * spec.patch_width()) as u64;
            let target = pointer(raw, s) + (4 * (slot + tile) * w.pixels()) as u64;
            unsafe {
                s.launch_builder(&self.core.kernels.patches)
                    .arg(&source)
                    .arg(&target)
                    .arg(&(w.frames as u32))
                    .arg(&(w.height as u32))
                    .arg(&(w.width as u32))
                    .arg(&(spec.out_channels as u32))
                    .arg(&(spec.temporal_ratio as u32))
                    .arg(&(spec.spatial_ratio as u32))
                    .launch(w.plan.flat(w.pixels()))?;
            }
        }
        Ok(())
    }

    pub(crate) fn decode(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        w: &mut TileWork,
        tiles: &VaeTiles,
        z: LatentVideo<'_>,
        out: &mut CudaSlice<f32>,
    ) -> Result<DecodePlan> {
        self.decode_observed((s, blas), w, tiles, z, out, |_, _, _| Ok(()))
    }

    fn decode_observed(
        &self,
        (s, blas): (&Arc<CudaStream>, &Blas),
        w: &mut TileWork,
        tiles: &VaeTiles,
        z: LatentVideo<'_>,
        out: &mut CudaSlice<f32>,
        mut observe: impl FnMut(&str, usize, &CudaSlice<f32>) -> Result<()>,
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
        let raw = s.alloc_zeros::<f32>(slots * w.pixels())?;
        let canvas = s.alloc_zeros::<f32>(channels * clip_frames * plane)?;
        let mut faded = s.alloc_zeros::<f32>(channels * tail * plane)?;
        ensure!(
            4 * (raw.len() + canvas.len() + faded.len())
                == decode_bytes(spec, tiles, (height, width)),
            "decode buffers differ from decode_bytes"
        );
        if crate::backend::diagnostics_enabled() {
            eprintln!(
                "{}",
                serde_json::json!({"h3_vae_decode": {"chunks": plan.chunks, "tiles": slots, "tokens_per_tile": w.tokens,
                    "frames": plan.frames, "raw_tile_bytes": 4 * raw.len(), "canvas_bytes": 4 * canvas.len(),
                    "output_bytes": 4 * out.len(), "free_bytes": s.context().mem_get_info()?.0}})
            );
        }
        let copy_frames = |s: &Arc<CudaStream>,
                           from: &CudaSlice<f32>,
                           from_frames: usize,
                           from_first: usize,
                           to: &mut CudaSlice<f32>,
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
/// raw tiles of one clip, the stitched clip canvas and the cross-faded tail, F32.
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
    4 * (tiles.count() * tile + channels * (clip_frames + tail) * height * width)
}

/// A device latent video `[latent_channels, frames, height, width]` F32.
pub(crate) struct LatentVideo<'a> {
    pub data: &'a CudaSlice<f32>,
    pub frames: usize,
    pub height: usize,
    pub width: usize,
}

pub(crate) struct TileWork {
    core: DecoderWork,
    frames: usize,
    height: usize,
    width: usize,
    patches: usize,
    tokens: usize,
    batch: usize,
    spec: VaeSpec,
}

impl std::ops::Deref for TileWork {
    type Target = DecoderWork;
    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl TileWork {
    pub(crate) fn batch_bytes(
        spec: &VaeSpec,
        shape: (usize, usize, usize),
        batch: usize,
        slots: usize,
    ) -> Result<usize> {
        ensure!(batch > 0 && batch <= slots, "invalid video tile batch");
        let (f, h, w) = shape;
        ensure!(f > 0 && h > 0 && w > 0, "empty video tile");
        let p = (f as u128)
            .checked_mul(h as u128)
            .and_then(|n| n.checked_mul(w as u128))
            .context("video tile overflow")?;
        ensure!(
            p <= u32::MAX as u128,
            "video patch count exceeds layout capacity"
        );
        let t = p + spec.registers as u128 + 1;
        let per = 4 * t * (7 * spec.dim + 3 * spec.ffn + spec.patch_width()) as u128
            + 8 * p * spec.latent_channels as u128;
        let scores = 4 * spec.heads as u128 * t * t;
        Ok(usize::try_from(
            batch as u128 * per + scores + 4 * (t * spec.rope_dims as u128 + spec.head_dim as u128),
        )?)
    }
    pub(crate) fn bytes(spec: &VaeSpec, shape: (usize, usize, usize)) -> usize {
        Self::batch_bytes(spec, shape, 1, 1).expect("validated video tile extent")
    }
    #[cfg(test)]
    pub(crate) fn new(
        s: &Arc<CudaStream>,
        spec: &VaeSpec,
        shape: (usize, usize, usize),
        (batch, slots): (usize, usize),
    ) -> Result<Self> {
        let bytes = Self::batch_bytes(spec, shape, batch, slots)?;
        let patches = shape.0 * shape.1 * shape.2;
        Self::with_plan(
            s,
            spec,
            shape,
            ImagePlan {
                linear: batch,
                attention: 1,
                patches,
                tokens: patches + spec.registers + 1,
                grid: i32::MAX as usize,
                bytes,
            },
        )
    }
    fn with_plan(
        s: &Arc<CudaStream>,
        spec: &VaeSpec,
        shape: (usize, usize, usize),
        plan: ImagePlan,
    ) -> Result<Self> {
        let (frames, height, width) = shape;
        let (patches, tokens, batch) = (plan.patches, plan.tokens, plan.linear);
        let work = Self {
            core: DecoderWork::new(s, spec, plan, rotary_table(spec, shape, 3, |x| x))?,
            frames,
            height,
            width,
            patches,
            tokens,
            batch,
            spec: spec.clone(),
        };
        ensure!(
            work.allocated_bytes() == work.plan.bytes,
            "video work differs from counted plan"
        );
        Ok(work)
    }
    fn allocated_bytes(&self) -> usize {
        [
            &self.rope,
            &self.ones,
            &self.x,
            &self.norm,
            &self.o,
            &self.q,
            &self.k,
            &self.v,
            &self.attn,
            &self.ff,
            &self.gated,
            &self.latent,
            &self.latent_q,
            &self.patch_rows,
            &self.scores,
        ]
        .iter()
        .map(|v| 4 * v.len())
        .sum()
    }
    fn pixels(&self) -> usize {
        self.spec.out_channels
            * self.frames
            * self.spec.temporal_ratio
            * self.height
            * self.spec.spatial_ratio
            * self.width
            * self.spec.spatial_ratio
    }
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
        if crate::backend::diagnostics_enabled() {
            eprintln!(
                "{}",
                serde_json::json!({"vae_encoder_work":{
                "activation_bytes":fixed,"column_bytes":4 * workspace,"minimum_column_bytes":4 * minimum,
                "budget":budget,"tile":[height,width]}})
            );
        }
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
        let (max_abs, rmse) = crate::verify::max_abs_rmse(label, actual, reference)?;
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

    #[test]
    #[ignore = "GPU; FF_MODEL, longest video capacity and forced F32 streaming"]
    fn video_decoder_longest_capacity_gpu() -> Result<()> {
        let model = path("FF_MODEL")?;
        let spec = VaeSpec::read(&model)?;
        let component = model.join("vae");
        let manifest = MappedCheckpoint::open_component(&component, "diffusion_pytorch_model")?
            .minimax_h3_vae_manifest(&spec)?;
        let runtime =
            std::env::temp_dir().join(format!("ff-video-capacity-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        let result = (|| -> Result<()> {
            let d = Device::new(0)?;
            let s = &d.stream;
            let ops = Ops::new(&d.ctx)?;
            let mut blas = Blas::new(
                &d,
                &ops,
                &runtime,
                Rc::new(Closure::new(false, "F32 video capacity".into())),
            )?;
            let (height, width) = (576, 1856);
            let r = spec.spatial_ratio;
            let tiles = VaeTiles::released(&spec, height, width)?;
            let tile = (
                spec.chunk_latents() + spec.overlap_latents(),
                tiles.height / r,
                tiles.width / r,
            );
            for (frames, latent_frames, caller) in
                [(345, 102, "MiniMax-H3"), (362, 107, "HyperFlow")]
            {
                let plan = DecodePlan::new(&spec, latent_frames)?;
                ensure!(
                    plan.frames == frames && plan.pad == 0,
                    "RUNNER_FAIL longest frame layout"
                );
                let total = plan.chunks
                    * (spec.chunk_latents() * spec.temporal_ratio - spec.frame_pre_padding())
                    + tile.0 * spec.temporal_ratio
                    - spec.chunk_latents() * spec.temporal_ratio
                    - spec.frame_pre_padding();
                let count = spec.latent_channels * latent_frames * (height / r) * (width / r);
                let output_count = spec.out_channels * total * height * width;
                let scratch = decode_bytes(&spec, &tiles, (height, width));
                let future =
                    TileWork::bytes(&spec, tile) + scratch + 4 * (count + output_count) + (1 << 30);
                for placement in [
                    super::super::Placement::Derived,
                    super::super::Placement::Streamed,
                ] {
                    let mut decoder =
                        VideoDecoder::new(&d, &spec, &component, &manifest, (future, placement))?;
                    decoder.prefix_only = true;
                    let mut work = decoder.work(
                        &d,
                        &mut blas,
                        &tiles,
                        (latent_frames, height / r, width / r),
                    )?;
                    ensure!(
                        work.allocated_bytes() == work.plan.bytes,
                        "RUNNER_FAIL capacity workspace count"
                    );
                    let input: Vec<_> = (0..count)
                        .map(|i| ((i * 73 % 65521) as f32 - 32760.0) / 8192.0)
                        .collect();
                    let input_device = s.clone_htod(&input)?;
                    let output = s.alloc_zeros::<f32>(output_count)?;
                    let raw = s.alloc_zeros::<f32>(tiles.count() * work.pixels())?;
                    let canvas = s.alloc_zeros::<f32>(
                        spec.out_channels * tile.0 * spec.temporal_ratio * height * width,
                    )?;
                    let tail = tile.0 * spec.temporal_ratio
                        - spec.chunk_latents() * spec.temporal_ratio
                        - spec.frame_pre_padding();
                    let faded = s.alloc_zeros::<f32>(spec.out_channels * tail * height * width)?;
                    ensure!(
                        4 * (raw.len() + canvas.len() + faded.len()) == scratch,
                        "RUNNER_FAIL longest stitch count"
                    );
                    let first_frame = (plan.chunks - 1) * spec.chunk_latents();
                    let row = *tiles.rows.0.last().context("no tile rows")? / r;
                    let col = *tiles.cols.0.last().context("no tile columns")? / r;
                    decoder.tile(
                        s,
                        &blas,
                        &mut work,
                        &LatentVideo {
                            data: &input_device,
                            frames: latent_frames,
                            height: height / r,
                            width: width / r,
                        },
                        (&[(row, col)], first_frame),
                        (&raw, tiles.count() - 1),
                    )?;
                    let actual =
                        s.clone_dtoh(&work.latent.slice(..work.patches * spec.latent_channels))?;
                    let mut differing = 0;
                    for (i, value) in actual.iter().enumerate() {
                        let c = i % spec.latent_channels;
                        let token = i / spec.latent_channels;
                        let x = token % tile.2;
                        let y = token / tile.2 % tile.1;
                        let t = token / (tile.1 * tile.2);
                        let index =
                            ((c * latent_frames + first_frame + t) * (height / r) + row + y)
                                * (width / r)
                                + col
                                + x;
                        differing += usize::from(value.to_bits() != input[index].to_bits());
                    }
                    ensure!(
                        differing == 0,
                        "DEFECT longest last-tile F32 gather {differing}"
                    );
                    let active = 4
                        * (input_device.len()
                            + output.len()
                            + raw.len()
                            + canvas.len()
                            + faded.len())
                        + work.allocated_bytes();
                    println!(
                        "PASS VIDEO_F32_CAPACITY caller={caller} frames={frames} latent=[1,24,{latent_frames},{},{}] canvas=[{height},{width}] placement={placement:?} linear={} attention={} activation_bytes={active} last_tile_differences={differing} transformer_forwards=0",
                        height / r,
                        width / r,
                        work.batch,
                        work.plan.attention
                    );
                    drop((faded, canvas, raw, output, input_device, work, decoder));
                    s.synchronize()?;
                }
            }
            Ok(())
        })();
        let cleanup = fs::remove_dir_all(&runtime);
        result?;
        cleanup?;
        Ok(())
    }

    #[test]
    #[ignore = "GPU; FF_MODEL, FF_H3_S0 prefix prediction and FF_H3_CAPTURE"]
    fn video_decoder_prefix_capture_gpu() -> Result<()> {
        let (model, root, capture) = (path("FF_MODEL")?, path("FF_H3_S0")?, path("FF_H3_CAPTURE")?);
        let prediction: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("PREDICTION.json"))?)?;
        ensure!(
            prediction["rule"] == "r7-prefix-rounds-v1"
                && prediction["native_arrays_read"] == false,
            "RUNNER_FAIL prefix prediction declaration"
        );
        let spec = VaeSpec::read(&model)?;
        let shape = (
            spec.chunk_latents() + spec.overlap_latents(),
            VAE_TILE / spec.spatial_ratio,
            VAE_TILE / spec.spatial_ratio,
        );
        ensure!(
            shape == (7, 16, 16) && spec.latent_channels == 24 && spec.dim == 2048,
            "RUNNER_FAIL prefix production shape"
        );
        let bytes = fs::read(root.join("input.safetensors"))?;
        let input = SafeTensors::deserialize(&bytes)?;
        ensure!(
            input.tensor("value")?.shape() == [1, 24, 37, 48, 84],
            "RUNNER_FAIL prefix input shape"
        );
        let values = tensor(&input, "value")?;
        let runtime = std::env::temp_dir().join(format!("ff-video-prefix-{}", std::process::id()));
        fs::create_dir(&runtime)?;
        fs::create_dir_all(&capture)?;
        let result = (|| -> Result<()> {
            let device = Device::new(0)?;
            let s = &device.stream;
            let ops = Ops::new(&device.ctx)?;
            let blas = Blas::new(
                &device,
                &ops,
                &runtime,
                Rc::new(Closure::new(false, "video decoder prefix capture".into())),
            )?;
            let component = model.join("vae");
            let manifest = MappedCheckpoint::open_component(&component, "diffusion_pytorch_model")?
                .minimax_h3_vae_manifest(&spec)?;
            let mut decoder = VideoDecoder::new(
                &device,
                &spec,
                &component,
                &manifest,
                (
                    TileWork::bytes(&spec, shape),
                    super::super::Placement::Derived,
                ),
            )?;
            decoder.prefix_only = true;
            let mut work = TileWork::new(s, &spec, shape, (1, 1))?;
            let input = s.clone_htod(&values)?;
            let raw = s.alloc_zeros::<f32>(work.pixels())?;
            decoder.tile(
                s,
                &blas,
                &mut work,
                &LatentVideo {
                    data: &input,
                    frames: 37,
                    height: 48,
                    width: 84,
                },
                (&[(0, 0)], 0),
                (&raw, 0),
            )?;
            let packed = s.clone_dtoh(&work.latent)?;
            let bytes = fs::read(root.join("f32/packed.safetensors"))?;
            let expected = tensor(&SafeTensors::deserialize(&bytes)?, "value")?;
            ensure!(
                packed.len() == expected.len()
                    && packed
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                "RUNNER_FAIL packed F32 input mismatch"
            );
            let shifted = packed
                .iter()
                .enumerate()
                .any(|(i, a)| a.to_bits() != expected[(i + 24) % expected.len()].to_bits());
            ensure!(shifted, "RUNNER_FAIL prefix row-shift control missed");
            for (name, shape, values) in [
                ("packed", vec![work.patches, 24], packed),
                (
                    "post_quant",
                    vec![work.patches, 24],
                    s.clone_dtoh(&work.latent_q)?,
                ),
                (
                    "proj_in",
                    vec![work.patches, 2048],
                    s.clone_dtoh(&work.o.slice(..work.patches * 2048))?,
                ),
            ] {
                let file = capture.join(name).with_extension("safetensors");
                ensure!(!file.exists(), "RUNNER_FAIL prefix capture exists");
                let mut trace = crate::trace::Trace::default();
                trace.set_device(&device.info.name)?;
                trace.add("value".into(), shape, values)?;
                trace.save(&file)?;
            }
            println!(
                "PASS VIDEO_PREFIX_CAPTURE boundaries=3 packed_exact=true row_shift_control=true transformer_forwards=0 input_bytes={} work_bytes={} raw_bytes={} blas_workspace_bytes={}",
                4 * values.len(),
                work.allocated_bytes(),
                4 * work.pixels(),
                blas.workspace_bytes()
            );
            Ok(())
        })();
        let cleanup = fs::remove_dir_all(&runtime);
        result?;
        cleanup?;
        Ok(())
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
        decoder.latent_affine = Some((s.clone_htod(&std)?, s.clone_htod(&mean)?));
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
