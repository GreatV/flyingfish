use super::vision_runtime::{EncodedImages, VisionIo, stage_view};
use crate::backend::cuda::{
    Device,
    blas::Blas,
    contracts::{RowDomain, SeqRef, TensorMut},
    cubin,
    linear::{ClippedLinear, LinearBiasMode},
    ops::{CommonOpsRef, RmsMode, RmsOptions, flat, grid},
    weights::WeightCatalog,
};
use crate::backend::gpu::{
    CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use crate::{
    model_spec::{AudioConfig, ModelSpec},
    vision_input::{MultimodalInput, PositionPlan},
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::sync::Arc;

struct FeedForward {
    pre: CudaSlice<bf16>,
    post: CudaSlice<bf16>,
    up: ClippedLinear,
    down: ClippedLinear,
}

struct Block {
    ff: [FeedForward; 2],
    pre_attention: CudaSlice<bf16>,
    post_attention: CudaSlice<bf16>,
    final_norm: CudaSlice<bf16>,
    projections: [ClippedLinear; 4],
    relative: CudaSlice<bf16>,
    scale: CudaSlice<bf16>,
    conv_pre: CudaSlice<bf16>,
    conv_norm: CudaSlice<bf16>,
    conv_start: ClippedLinear,
    conv_end: ClippedLinear,
    depthwise: CudaSlice<bf16>,
}

struct AudioOps {
    patch: CudaFunction,
    activation: CudaFunction,
    depthwise: CudaFunction,
    attention: CudaFunction,
    branch: CudaFunction,
    residual: CudaFunction,
    clamp: CudaFunction,
    layer_norm: CudaFunction,
    gather: CudaFunction,
}

pub(super) struct AudioRuntime {
    config: AudioConfig,
    width: u32,
    subsample: [(ClippedLinear, CudaSlice<bf16>); 2],
    input: ClippedLinear,
    blocks: Vec<Block>,
    output: ClippedLinear,
    output_bias: CudaSlice<bf16>,
    projection: ClippedLinear,
    ones: CudaSlice<bf16>,
    zeros: CudaSlice<bf16>,
    clip: [CudaSlice<bf16>; 2],
    common: CommonOpsRef,
    ops: AudioOps,
    io: Arc<VisionIo>,
}

struct Clip {
    input: CudaSlice<bf16>,
    masks: [CudaSlice<u8>; 3],
    indices: CudaSlice<i32>,
    meta: CudaSlice<i32>,
    frames: u32,
    feature_start: usize,
    valid: u32,
}

pub(super) struct AudioRequest {
    clips: Vec<Clip>,
    patch: CudaSlice<bf16>,
    subsample: [CudaSlice<bf16>; 2],
    hidden: CudaSlice<bf16>,
    norm: CudaSlice<bf16>,
    delta: CudaSlice<bf16>,
    qkv: [CudaSlice<bf16>; 3],
    core: CudaSlice<bf16>,
    expanded: CudaSlice<bf16>,
    activation: CudaSlice<bf16>,
    clipped: CudaSlice<bf16>,
    projected: CudaSlice<bf16>,
    embed_norm: CudaSlice<bf16>,
    features: CudaSlice<bf16>,
}

impl AudioRuntime {
    pub(super) fn from_catalog(
        model: &ModelSpec,
        catalog: &mut WeightCatalog,
        device: &Device,
        blas: &mut Blas,
        common: CommonOpsRef,
    ) -> Result<Self> {
        let a = model.audio.clone().context("audio configuration missing")?;
        let h = a.hidden;
        let s = &device.stream;
        let module = cubin::module(&device.ctx, "vision_ops")?;
        let common_module = cubin::module(&device.ctx, "q38_common")?;
        let ops = AudioOps {
            patch: module.load_function("masked_strided_patch_bf16")?,
            activation: module.load_function("activation_bf16")?,
            depthwise: module.load_function("causal_depthwise_conv_bf16")?,
            attention: cubin::module(&device.ctx, "attention")?
                .load_function("relative_window_attention_d128")?,
            branch: common_module.load_function("rms_scaled_branch_add")?,
            residual: cubin::module(&device.ctx, "q38_vision_io")?
                .load_function("q38_vision_residual")?,
            clamp: module.load_function("vision_clamp_bf16")?,
            layer_norm: common_module.load_function("q38_layer_norm")?,
            gather: module.load_function("vision_row_gather")?,
        };
        let p = "model.audio_tower";
        let mut subsample = Vec::new();
        for (index, input, output) in [
            (0, 1, a.subsample_channels[0]),
            (1, a.subsample_channels[0], a.subsample_channels[1]),
        ] {
            let name = format!("{p}.subsample_conv_projection.layer{index}");
            subsample.push((
                ClippedLinear {
                    weight: catalog.take_bf16(
                        &format!("{name}.conv.weight"),
                        &[output as usize, input as usize, 3, 3],
                    )?,
                    shape: [output, input * 9],
                    bounds: None,
                },
                catalog.take_bf16(&format!("{name}.norm.weight"), &[output as usize])?,
            ));
        }
        let relative_rows = (a.chunk + a.context_left - 1) / 2 + 1;
        let relative_input = (0..relative_rows)
            .flat_map(|row| {
                (0..h).map(move |d| {
                    let exponent =
                        (d % (h / 2)) as f32 * -(10000_f64.ln() / f64::from(h / 2 - 1)) as f32;
                    let phase = (relative_rows - 1 - row) as f32 * exponent.exp();
                    bf16::from_f32(if d < h / 2 { phase.sin() } else { phase.cos() })
                })
            })
            .collect::<Vec<_>>();
        let relative_input = s.clone_htod(&relative_input)?;
        let scratch = s.alloc_zeros(relative_rows as usize * h as usize)?;
        let mut blocks = Vec::new();
        for index in 0..a.layers {
            let prefix = format!("{p}.layers.{index}");
            let norm = |catalog: &mut WeightCatalog, name: &str| {
                catalog.take_bf16(&format!("{prefix}.{name}.weight"), &[h as usize])
            };
            let linear = |catalog: &mut WeightCatalog, name: &str, shape| {
                let name = format!("{prefix}.{name}");
                ClippedLinear::take(
                    catalog,
                    &format!("{name}.linear.weight"),
                    shape,
                    Some(&name),
                )
            };
            let mut ff = Vec::new();
            for i in 1..=2 {
                ff.push(FeedForward {
                    pre: norm(catalog, &format!("feed_forward{i}.pre_layer_norm"))?,
                    post: norm(catalog, &format!("feed_forward{i}.post_layer_norm"))?,
                    up: linear(catalog, &format!("feed_forward{i}.ffw_layer_1"), [h * 4, h])?,
                    down: linear(catalog, &format!("feed_forward{i}.ffw_layer_2"), [h, h * 4])?,
                });
            }
            let relative_weight = ClippedLinear::take(
                catalog,
                &format!("{prefix}.self_attn.relative_k_proj.weight"),
                [h, h],
                None,
            )?;
            relative_weight.prepare(blas, relative_rows)?;
            let relative = s.alloc_zeros(relative_rows as usize * h as usize)?;
            relative_weight.run(
                s,
                blas,
                &ops.clamp,
                stage_view(&relative_input, s, [relative_rows, h])?,
                TensorMut(stage_view(&scratch, s, [relative_rows, h])?),
                &mut TensorMut(stage_view(&relative, s, [relative_rows, h])?),
            )?;
            let scale_parameter = catalog.take_bf16(
                &format!("{prefix}.self_attn.per_dim_scale"),
                &[(h / a.heads) as usize],
            )?;
            let scale = s.alloc_zeros((h / a.heads) as usize)?;
            unsafe {
                s.launch_builder(&ops.activation)
                    .arg(&scale_parameter)
                    .arg(&scale)
                    .arg(&1u32)
                    .arg(&(h / a.heads))
                    .arg(&3i32)
                    .launch(flat((h / a.heads) as usize))?;
            }
            blocks.push(Block {
                ff: ff
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("audio FFN count"))?,
                pre_attention: norm(catalog, "norm_pre_attn")?,
                post_attention: norm(catalog, "norm_post_attn")?,
                final_norm: norm(catalog, "norm_out")?,
                projections: [
                    linear(catalog, "self_attn.q_proj", [h, h])?,
                    linear(catalog, "self_attn.k_proj", [h, h])?,
                    linear(catalog, "self_attn.v_proj", [h, h])?,
                    linear(catalog, "self_attn.post", [h, h])?,
                ],
                relative,
                scale,
                conv_pre: norm(catalog, "lconv1d.pre_layer_norm")?,
                conv_norm: norm(catalog, "lconv1d.conv_norm")?,
                conv_start: linear(catalog, "lconv1d.linear_start", [h * 2, h])?,
                conv_end: linear(catalog, "lconv1d.linear_end", [h, h])?,
                depthwise: catalog.take_bf16(
                    &format!("{prefix}.lconv1d.depthwise_conv1d.weight"),
                    &[h as usize, 1, a.conv_kernel as usize],
                )?,
            });
            s.synchronize()?;
        }
        let output = a.output;
        Ok(Self {
            input: ClippedLinear::take(
                catalog,
                &format!("{p}.subsample_conv_projection.input_proj_linear.weight"),
                [h, h],
                None,
            )?,
            output: ClippedLinear::take(
                catalog,
                &format!("{p}.output_proj.weight"),
                [output, h],
                None,
            )?,
            output_bias: catalog.take_bf16(&format!("{p}.output_proj.bias"), &[output as usize])?,
            projection: ClippedLinear::take(
                catalog,
                "model.embed_audio.embedding_projection.weight",
                [model.hidden, output],
                None,
            )?,
            ones: s.clone_htod(&vec![bf16::ONE; output as usize])?,
            zeros: s.alloc_zeros(a.subsample_channels[0].max(a.subsample_channels[1]) as usize)?,
            clip: [
                s.clone_htod(&[bf16::from_f32(-a.gradient_clip)])?,
                s.clone_htod(&[bf16::from_f32(a.gradient_clip)])?,
            ],
            config: a,
            width: model.hidden,
            subsample: subsample
                .try_into()
                .map_err(|_| anyhow::anyhow!("audio subsampler count"))?,
            blocks,
            common,
            ops,
            io: Arc::new(VisionIo::new(device, model.hidden)?),
        })
    }

    pub(super) fn request_bytes(
        model: &ModelSpec,
        input: &MultimodalInput<'_>,
        capacity: usize,
    ) -> Result<usize> {
        input.validate(model, capacity)?;
        let a = model
            .audio
            .as_ref()
            .context("audio configuration missing")?;
        let frames = input
            .audio
            .iter()
            .map(|clip| clip.mask.len())
            .max()
            .context("audio request has no clips")?;
        let t1 = frames.div_ceil(2);
        let rows = t1.div_ceil(2);
        let c0 = a.subsample_channels[0] as usize;
        let c1 = a.subsample_channels[1] as usize;
        let patch = (t1 * 64 * 9).max(rows * 32 * c0 * 9);
        let working = patch
            + t1 * 64 * c0
            + rows * 32 * c1
            + rows
                * (11 * a.hidden as usize
                    + 8 * a.hidden as usize
                    + 2 * a.output as usize
                    + model.hidden as usize);
        let mut bytes = working.checked_mul(2).context("audio workspace overflow")?;
        for clip in input.audio {
            let frames = clip.mask.len();
            let resident =
                clip.features.len() * 2 + frames + frames.div_ceil(2) + frames.div_ceil(4) * 5 + 32;
            bytes = bytes
                .checked_add(resident)
                .context("audio request overflow")?;
        }
        Ok(bytes)
    }

    pub(super) fn empty_features(
        &self,
        device: &Device,
        positions: PositionPlan,
    ) -> Result<EncodedImages> {
        let features = device.stream.alloc_zeros(
            positions
                .feature_rows()
                .checked_mul(self.width as usize)
                .context("media feature size overflow")?,
        )?;
        EncodedImages::new(positions, features, self.io.clone())
    }

    pub(super) fn prepare(
        &self,
        device: &Device,
        blas: &mut Blas,
        input: &MultimodalInput<'_>,
        positions: &PositionPlan,
    ) -> Result<AudioRequest> {
        let s = &device.stream;
        let mut clips = Vec::new();
        for (clip, slot) in input.audio.iter().zip(positions.audio_slots()) {
            let frames = u32::try_from(clip.mask.len())?;
            let rows = frames.div_ceil(4);
            let masks = [1, 2, 4].map(|stride| {
                clip.mask
                    .iter()
                    .step_by(stride)
                    .map(|&v| u8::from(v))
                    .collect::<Vec<_>>()
            });
            let indices = masks[2]
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| (v != 0).then_some(i as i32))
                .collect::<Vec<_>>();
            let valid = indices.len() as u32;
            ensure!(
                slot.features.len() == valid as usize,
                "audio slot extent mismatch"
            );
            clips.push(Clip {
                input: s.clone_htod(
                    &clip
                        .features
                        .iter()
                        .copied()
                        .map(bf16::from_f32)
                        .collect::<Vec<_>>(),
                )?,
                masks: [
                    s.clone_htod(&masks[0])?,
                    s.clone_htod(&masks[1])?,
                    s.clone_htod(&masks[2])?,
                ],
                indices: s.clone_htod(&indices)?,
                meta: s.clone_htod(&[0, rows as i32, rows as i32, rows as i32, 0, 0, 0, 0])?,
                frames,
                feature_start: slot.features.start,
                valid,
            });
            self.subsample[0].0.prepare(blas, frames.div_ceil(2) * 64)?;
            self.subsample[1].0.prepare(blas, rows * 32)?;
            self.input.prepare(blas, rows)?;
            self.projection.prepare(blas, rows)?;
            blas.prepare_biased(
                rows,
                self.config.output,
                self.config.hidden,
                stage_view(&self.output_bias, s, [self.config.output])?,
                LinearBiasMode::BeforeOutputRound,
            )?;
            for block in &self.blocks {
                for ff in &block.ff {
                    ff.up.prepare(blas, rows)?;
                    ff.down.prepare(blas, rows)?;
                }
                for projection in &block.projections {
                    projection.prepare(blas, rows)?;
                }
                block.conv_start.prepare(blas, rows)?;
                block.conv_end.prepare(blas, rows)?;
            }
        }
        let frames = clips
            .iter()
            .map(|c| c.frames as usize)
            .max()
            .context("no audio clips")?;
        let t1 = frames.div_ceil(2);
        let rows = t1.div_ceil(2);
        let h = self.config.hidden as usize;
        let output = self.config.output as usize;
        let c0 = self.config.subsample_channels[0] as usize;
        let c1 = self.config.subsample_channels[1] as usize;
        Ok(AudioRequest {
            clips,
            patch: s.alloc_zeros((t1 * 64 * 9).max(rows * 32 * c0 * 9))?,
            subsample: [s.alloc_zeros(t1 * 64 * c0)?, s.alloc_zeros(rows * 32 * c1)?],
            hidden: s.alloc_zeros(rows * h)?,
            norm: s.alloc_zeros(rows * h)?,
            delta: s.alloc_zeros(rows * h)?,
            qkv: [
                s.alloc_zeros(rows * h)?,
                s.alloc_zeros(rows * h)?,
                s.alloc_zeros(rows * h)?,
            ],
            core: s.alloc_zeros(rows * h)?,
            expanded: s.alloc_zeros(rows * 4 * h)?,
            activation: s.alloc_zeros(rows * 4 * h)?,
            clipped: s.alloc_zeros(rows * 4 * h)?,
            projected: s.alloc_zeros(rows * output)?,
            embed_norm: s.alloc_zeros(rows * output)?,
            features: s.alloc_zeros(rows * self.width as usize)?,
        })
    }

    fn activate(
        &self,
        s: &Arc<CudaStream>,
        input: &CudaSlice<bf16>,
        output: &CudaSlice<bf16>,
        rows: u32,
        width: u32,
        mode: i32,
    ) -> Result<()> {
        unsafe {
            s.launch_builder(&self.ops.activation)
                .arg(input)
                .arg(output)
                .arg(&rows)
                .arg(&width)
                .arg(&mode)
                .launch(flat(rows as usize * width as usize))?;
        }
        Ok(())
    }

    fn clamp(
        &self,
        s: &Arc<CudaStream>,
        input: &CudaSlice<bf16>,
        output: &CudaSlice<bf16>,
        rows: u32,
    ) -> Result<()> {
        let count = u64::from(rows) * u64::from(self.config.hidden);
        unsafe {
            s.launch_builder(&self.ops.clamp)
                .arg(input)
                .arg(output)
                .arg(&count)
                .arg(&self.clip[0])
                .arg(&self.clip[1])
                .launch(flat(count as usize))?;
        }
        Ok(())
    }

    fn norm(
        &self,
        s: &Arc<CudaStream>,
        domain: RowDomain,
        input: &CudaSlice<bf16>,
        output: &CudaSlice<bf16>,
        weight: &CudaSlice<bf16>,
        width: u32,
    ) -> Result<()> {
        let rows = domain.row_capacity;
        self.common.norm.rms_with(
            s,
            domain,
            stage_view(input, s, [rows, 1, width])?,
            stage_view(weight, s, [width])?,
            &mut TensorMut(stage_view(output, s, [rows, 1, width])?),
            RmsOptions {
                mode: RmsMode::Unrounded,
                eps: self.config.rms_eps,
            },
        )
    }

    fn branch(
        &self,
        s: &Arc<CudaStream>,
        work: &AudioRequest,
        weight: &CudaSlice<bf16>,
        rows: u32,
        scale: f32,
    ) -> Result<()> {
        unsafe {
            s.launch_builder(&self.ops.branch)
                .arg(&work.hidden)
                .arg(&work.delta)
                .arg(weight)
                .arg(&self.config.hidden)
                .arg(&self.config.rms_eps)
                .arg(&self.config.gradient_clip)
                .arg(&scale)
                .launch(grid(rows as usize, 256))?;
        }
        Ok(())
    }

    fn feed_forward(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        work: &AudioRequest,
        ff: &FeedForward,
        domain: RowDomain,
    ) -> Result<()> {
        let rows = domain.row_capacity;
        let h = self.config.hidden;
        self.clamp(s, &work.hidden, &work.delta, rows)?;
        self.norm(s, domain, &work.delta, &work.norm, &ff.pre, h)?;
        self.linear(s, blas, work, &ff.up, (&work.norm, &work.expanded), rows)?;
        self.activate(s, &work.expanded, &work.activation, rows, h * 4, 1)?;
        self.linear(
            s,
            blas,
            work,
            &ff.down,
            (&work.activation, &work.delta),
            rows,
        )?;
        self.branch(s, work, &ff.post, rows, self.config.residual_weight)
    }

    fn linear(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        work: &AudioRequest,
        weight: &ClippedLinear,
        buffers: (&CudaSlice<bf16>, &CudaSlice<bf16>),
        rows: u32,
    ) -> Result<()> {
        let (input, output) = buffers;
        weight.run(
            s,
            blas,
            &self.ops.clamp,
            stage_view(input, s, [rows, weight.shape[1]])?,
            TensorMut(stage_view(&work.clipped, s, [rows, weight.shape[1]])?),
            &mut TensorMut(stage_view(output, s, [rows, weight.shape[0]])?),
        )
    }

    pub(super) fn encode(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        work: AudioRequest,
        features: &CudaSlice<bf16>,
        mut trace: Option<&mut crate::trace::Trace>,
    ) -> Result<()> {
        let a = &self.config;
        let h = a.hidden;
        for (clip_index, clip) in work.clips.iter().enumerate() {
            let rows = clip.frames.div_ceil(4);
            let domain = RowDomain {
                meta: SeqRef {
                    ptr: clip.meta.device_ptr(s).0,
                },
                row_base: 0,
                row_capacity: rows,
            };
            for i in 0..2 {
                let (time, frequency, channels, input) = if i == 0 {
                    (clip.frames, 128u32, 1, &clip.input)
                } else {
                    (
                        clip.frames.div_ceil(2),
                        64,
                        a.subsample_channels[0],
                        &work.subsample[0],
                    )
                };
                let out_rows = time.div_ceil(2) * frequency.div_ceil(2);
                let width = a.subsample_channels[i];
                unsafe {
                    s.launch_builder(&self.ops.patch)
                        .arg(input)
                        .arg(&clip.masks[i])
                        .arg(&work.patch)
                        .arg(&time)
                        .arg(&frequency)
                        .arg(&channels)
                        .launch(flat(out_rows as usize * channels as usize * 9))?;
                }
                self.subsample[i].0.run(
                    s,
                    blas,
                    &self.ops.clamp,
                    stage_view(&work.patch, s, [out_rows, channels * 9])?,
                    TensorMut(stage_view(&work.patch, s, [out_rows, channels * 9])?),
                    &mut TensorMut(stage_view(&work.subsample[i], s, [out_rows, width])?),
                )?;
                unsafe {
                    s.launch_builder(&self.ops.layer_norm)
                        .arg(&work.subsample[i])
                        .arg(&self.subsample[i].1)
                        .arg(&self.zeros)
                        .arg(&work.subsample[i])
                        .arg(&out_rows)
                        .arg(&1u32)
                        .arg(&width)
                        .arg(&u64::from(width))
                        .arg(&u64::from(width))
                        .arg(&1u64)
                        .arg(&u64::from(width))
                        .arg(&u64::from(width))
                        .arg(&1u64)
                        .arg(&a.rms_eps)
                        .launch(grid(out_rows as usize, 256))?;
                }
                self.activate(
                    s,
                    &work.subsample[i],
                    &work.subsample[i],
                    out_rows,
                    width,
                    0,
                )?;
                Self::trace(
                    s,
                    &mut trace,
                    &format!("audio.{clip_index}.subsample{i}"),
                    &work.subsample[i],
                    out_rows,
                    width,
                )?;
            }
            self.linear(
                s,
                blas,
                &work,
                &self.input,
                (&work.subsample[1], &work.hidden),
                rows,
            )?;
            for (index, block) in self.blocks.iter().enumerate() {
                self.feed_forward(s, blas, &work, &block.ff[0], domain)?;
                self.clamp(s, &work.hidden, &work.delta, rows)?;
                self.norm(s, domain, &work.delta, &work.norm, &block.pre_attention, h)?;
                for (projection, output) in block.projections[..3].iter().zip(&work.qkv) {
                    self.linear(s, blas, &work, projection, (&work.norm, output), rows)?;
                }
                let q_scale = (128_f64.powf(-0.5) / std::f64::consts::LN_2) as f32;
                let k_scale = (1_f64.exp().ln_1p() / std::f64::consts::LN_2) as f32;
                unsafe {
                    s.launch_builder(&self.ops.attention)
                        .arg(&work.qkv[0])
                        .arg(&work.qkv[1])
                        .arg(&work.qkv[2])
                        .arg(&block.relative)
                        .arg(&block.scale)
                        .arg(&clip.masks[2])
                        .arg(&work.core)
                        .arg(&rows)
                        .arg(&a.heads)
                        .arg(&a.chunk)
                        .arg(&a.context_left)
                        .arg(&q_scale)
                        .arg(&k_scale)
                        .arg(&a.logit_cap)
                        .arg(&a.invalid_logit)
                        .launch(LaunchConfig {
                            grid_dim: (rows, a.heads, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }
                self.linear(
                    s,
                    blas,
                    &work,
                    &block.projections[3],
                    (&work.core, &work.delta),
                    rows,
                )?;
                self.branch(s, &work, &block.post_attention, rows, 1.0)?;
                self.norm(s, domain, &work.hidden, &work.norm, &block.conv_pre, h)?;
                self.linear(
                    s,
                    blas,
                    &work,
                    &block.conv_start,
                    (&work.norm, &work.expanded),
                    rows,
                )?;
                self.activate(s, &work.expanded, &work.activation, rows, h, 2)?;
                unsafe {
                    s.launch_builder(&self.ops.depthwise)
                        .arg(&work.activation)
                        .arg(&block.depthwise)
                        .arg(&work.core)
                        .arg(&rows)
                        .arg(&h)
                        .arg(&a.conv_kernel)
                        .launch(flat(rows as usize * h as usize))?;
                }
                self.clamp(s, &work.core, &work.delta, rows)?;
                self.norm(s, domain, &work.delta, &work.norm, &block.conv_norm, h)?;
                self.activate(s, &work.norm, &work.activation, rows, h, 1)?;
                self.linear(
                    s,
                    blas,
                    &work,
                    &block.conv_end,
                    (&work.activation, &work.delta),
                    rows,
                )?;
                unsafe {
                    s.launch_builder(&self.ops.residual)
                        .arg(&work.hidden)
                        .arg(&work.delta)
                        .arg(&(u64::from(rows) * u64::from(h)))
                        .launch(flat(rows as usize * h as usize))?;
                }
                self.feed_forward(s, blas, &work, &block.ff[1], domain)?;
                self.clamp(s, &work.hidden, &work.delta, rows)?;
                self.norm(s, domain, &work.delta, &work.hidden, &block.final_norm, h)?;
                if index == 0 {
                    Self::trace(
                        s,
                        &mut trace,
                        &format!("audio.{clip_index}.layer0"),
                        &work.hidden,
                        rows,
                        h,
                    )?;
                }
            }
            blas.linear_biased(
                stage_view(&self.output.weight, s, self.output.shape)?,
                stage_view(&self.output_bias, s, [a.output])?,
                stage_view(&work.hidden, s, [rows, h])?,
                &mut TensorMut(stage_view(&work.projected, s, [rows, a.output])?),
                LinearBiasMode::BeforeOutputRound,
            )?;
            Self::trace(
                s,
                &mut trace,
                &format!("audio.{clip_index}.tower"),
                &work.projected,
                rows,
                a.output,
            )?;
            self.norm(
                s,
                domain,
                &work.projected,
                &work.embed_norm,
                &self.ones,
                a.output,
            )?;
            self.linear(
                s,
                blas,
                &work,
                &self.projection,
                (&work.embed_norm, &work.features),
                rows,
            )?;
            Self::trace(
                s,
                &mut trace,
                &format!("audio.{clip_index}.features"),
                &work.features,
                rows,
                self.width,
            )?;
            let start = clip
                .feature_start
                .checked_mul(self.width as usize)
                .context("audio feature offset overflow")?;
            let end = start
                .checked_add(clip.valid as usize * self.width as usize)
                .context("audio feature extent overflow")?;
            ensure!(
                end <= features.len(),
                "audio feature output exceeds allocation"
            );
            unsafe {
                s.launch_builder(&self.ops.gather)
                    .arg(&work.features)
                    .arg(&clip.indices)
                    .arg(&features.slice(start..end))
                    .arg(&clip.valid)
                    .arg(&rows)
                    .arg(&self.width)
                    .launch(flat(clip.valid as usize * self.width as usize))?;
            }
        }
        s.synchronize()?;
        Ok(())
    }

    fn trace(
        s: &Arc<CudaStream>,
        trace: &mut Option<&mut crate::trace::Trace>,
        name: &str,
        data: &CudaSlice<bf16>,
        rows: u32,
        width: u32,
    ) -> Result<()> {
        if let Some(trace) = trace.as_mut() {
            let values = s.clone_dtoh(&data.slice(..rows as usize * width as usize))?;
            trace.add(
                name.into(),
                vec![rows as usize, width as usize],
                values.into_iter().map(bf16::to_f32).collect(),
            )?;
        }
        Ok(())
    }
}
