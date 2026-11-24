//! MiniMax-H3 i2va generation on one GPU: conditioner, keyframe encoder, transformer chain, then the video and audio
//! decoders after the transformer and conditioner are released. Stages run in that order and each stage's weights are
//! dropped before the next stage's placement is derived from the then-free device memory.
use super::super::vision_runtime::LayerNormTower;
use super::audio::AudioDecoder;
use super::vae::{
    EncodeWork, ImageDecodeStage, ImageEncodeSetup, ImageEncodeStage, KeyframeEncoder, LatentVideo,
    TileWork, VideoDecoder, decode_bytes, decoder_layer_sizes, decoder_weight_bytes,
    image_vae_manifest,
};
use super::{
    AdalnTable, BlockContext, BlockKernels, Conditioner, ConditionerInput, ConditionerPrompt, Dit,
    ImageDit, Placement, Source, conditioner_embeds, conditioner_rope, embedding_rows,
    image_block_spec, image_runs, seed_dense_supply_workers, stored_allocation_bytes,
    stored_allocation_granularity,
};
use crate::backend::cuda::{
    Device, blas::Blas, closure::Closure, cubin, ops::Ops, weights::WeightCatalog,
};
use crate::backend::gpu::CudaSlice;
use crate::checkpoint::{LoadPolicy, MappedCheckpoint, TensorManifest};
use crate::minimax_h3::{
    AdalnPlan, AudioVaeSpec, Geometry, H3Output, H3Request, H3Spec, ImageLatents, ImageLayout,
    ImageRequest, ImageRunOptions, ImageSchedule, ImageSpec, KEYFRAME_NOISE_AUG, PackedLayout,
    RowTimesteps, Schedule, TEXT_ENCODER_LAYER, VaeSpec, VaeTiles, hyperflow_schedules,
    hyperflow_steps, patchify_latents_into, row_timestep_plan, unpatchify_latents,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::{
    path::{Path, PathBuf},
    rc::Rc,
};

/// Generated media and, in test builds, the denormalised decoder inputs and the video decoder output (the PCM is the
/// audio decoder output) that the splice gate scores.
pub(crate) struct Generated {
    pub output: H3Output,
    #[cfg(all(test, feature = "cuda"))]
    conditioner: Vec<bf16>,
    #[cfg(all(test, feature = "cuda"))]
    anchors: Vec<Vec<f32>>,
    #[cfg(test)]
    pub video_latents: (Vec<usize>, Vec<f32>),
    #[cfg(test)]
    pub audio_latents: (Vec<usize>, Vec<f32>),
    #[cfg(test)]
    pub video_decoded: (Vec<usize>, Vec<f32>),
    #[cfg(test)]
    replay: Option<Replay>,
}

enum NoiseDraws {
    Seed([(u64, u64); 4]),
    #[cfg(test)]
    Recorded(ReplayInput),
}

#[cfg(test)]
struct ReplayInput {
    case: &'static str,
    gate: Rc<std::cell::RefCell<CompositionGate>>,
    draws: [Vec<f32>; 4],
    conditioner: Vec<bf16>,
    anchors: Vec<Vec<f32>>,
    teacher_inputs: Vec<(Vec<f32>, Vec<f32>)>,
}

impl NoiseDraws {
    fn draw(
        &self,
        device: &Device,
        normal: &crate::backend::gpu::CudaFunction,
        index: usize,
        count: usize,
    ) -> Result<CudaSlice<f32>> {
        match self {
            Self::Seed(draws) => {
                super::draw_normals(&device.stream, normal, draws[index], count, 256)
            }
            #[cfg(test)]
            Self::Recorded(input) => {
                let values = &input.draws[index];
                ensure!(
                    values.len() == count && values.iter().all(|v| v.is_finite()),
                    "RUNNER_FAIL recorded draw{index} needs{count} finite values, got{}",
                    values.len()
                );
                Ok(device.stream.clone_htod(values)?)
            }
        }
    }
}

#[cfg(test)]
struct Replay {
    initial: (Vec<f32>, Vec<f32>),
    refiner: Vec<bf16>,
    steps: Vec<super::SampleTrace>,
    chunks: (usize, usize),
}

/// Conditioner stage: Qwen3-VL decoder layers `0..TEXT_ENCODER_LAYER` (released on return) map the prompt to the
/// transformer's BF16 text rows `[rows, hidden]`. A prompt with image tokens also loads the vision tower.
fn condition(
    device: &Device,
    blas: &mut Blas,
    model: &Path,
    request: &H3Request,
    runtime: &Path,
) -> Result<Vec<bf16>> {
    let encoder = model.join("text_encoder");
    let spec = crate::model_spec::ModelSpec::read(&encoder)?;
    let image_token = spec
        .vision
        .as_ref()
        .context("text encoder without vision")?
        .image_token;
    condition_prompt(
        device,
        blas,
        model,
        &video_conditioner_prompt(request, image_token),
        runtime,
    )
}

fn video_conditioner_prompt(request: &H3Request, image_token: u32) -> ConditionerPrompt {
    ConditionerPrompt {
        modality: request
            .token_ids
            .iter()
            .map(|&id| u8::from(id == image_token))
            .collect(),
        tokens: request.token_ids.clone(),
        grids: request.grids.clone(),
        pixels: request.pixel_values.clone(),
    }
}

fn condition_prompt(
    device: &Device,
    blas: &mut Blas,
    model: &Path,
    prompt: &ConditionerPrompt,
    runtime: &Path,
) -> Result<Vec<bf16>> {
    use crate::backend::cuda::ops::CommonOpsRef;
    let encoder = model.join("text_encoder");
    let spec = crate::model_spec::ModelSpec::read(&encoder)?;
    let image_token = spec
        .vision
        .as_ref()
        .context("text encoder without vision")?
        .image_token;
    let manifest = MappedCheckpoint::open_component(&encoder, "model")?
        .minimax_h3_conditioner_manifest(&spec, TEXT_ENCODER_LAYER)?;
    seed_dense_supply_workers(&encoder, &manifest, runtime)?;
    let hidden = spec.hidden as usize;
    let fp16 = device.info.arch.fp16_stored_half();
    let table = conditioner_table(&encoder, &manifest, prompt, image_token, hidden, fp16)?;
    let runs = image_runs(&prompt.tokens, image_token);
    ensure!(
        runs.is_empty() == prompt.grids.is_empty()
            && prompt.grids.is_empty() == prompt.pixels.is_empty(),
        "image runs, grids and pixels disagree"
    );
    let count = prompt.tokens.len();
    let (mut conditioner, merged, encoded) = if runs.is_empty() {
        (
            Conditioner::new(
                device,
                &spec,
                (TEXT_ENCODER_LAYER, count, 0),
                Source::new(&encoder, &manifest, fp16),
                Placement::Derived,
                1,
            )?,
            Vec::new(),
            None,
        )
    } else {
        let mut catalog =
            WeightCatalog::load_with_policy(&encoder, &spec, device, LoadPolicy::VisionOnly)?;
        let tower = super::super::vision_runtime::LayerNormTower::from_catalog(
            &spec,
            &mut catalog,
            device,
            CommonOpsRef::new(&device.ctx)?,
        )?;
        ensure!(
            catalog.tensors.is_empty(),
            "vision binding left unloaded owners"
        );
        device.finish_upload()?;
        let input = crate::vision_input::MultimodalInput {
            input_ids: &prompt.tokens,
            modality_ids: &prompt.modality,
            pixel_values: &prompt.pixels,
            grids: &prompt.grids,
            audio: &[],
        };
        let media =
            super::super::vision_runtime::LayerNormTower::request_bytes(&spec, &input, count)?;
        let conditioner = Conditioner::new(
            device,
            &spec,
            (TEXT_ENCODER_LAYER, count, media),
            Source::new(&encoder, &manifest, fp16),
            Placement::Derived,
            1,
        )?;
        let prepared = tower.prepare(device, blas, &input, count, 0)?;
        let encoded = tower.encode(&device.stream, blas, prepared, None)?;
        #[cfg(test)]
        println!("HYPERFLOW_VISION_ENCODER_CALL rows={count} calls=1");
        let merged = device.stream.clone_dtoh(&encoded.features)?;
        (conditioner, merged, Some(encoded))
    };
    conditioner.prepare(blas, count)?;
    finish_conditioner(
        (device, blas),
        &mut conditioner,
        (&spec, prompt, image_token),
        &table,
        (&merged, &runs),
        encoded
            .as_ref()
            .map_or(&[], |item| item.deepstack.as_slice()),
    )
}

fn conditioner_table(
    encoder: &Path,
    manifest: &TensorManifest,
    prompt: &ConditionerPrompt,
    image_token: u32,
    hidden: usize,
    fp16: bool,
) -> Result<std::collections::BTreeMap<u32, Vec<bf16>>> {
    let ids: Vec<u32> = prompt
        .tokens
        .iter()
        .copied()
        .filter(|&id| id != image_token)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let rows = embedding_rows(
        &mut Source::new(encoder, manifest, fp16),
        "model.language_model.embed_tokens.weight",
        hidden,
        &ids,
        fp16,
    )?;
    Ok(ids
        .iter()
        .zip(rows.chunks(hidden))
        .map(|(&id, row)| (id, row.to_vec()))
        .collect())
}

fn finish_conditioner(
    (device, blas): (&Device, &Blas),
    conditioner: &mut Conditioner,
    (spec, prompt, image_token): (&crate::model_spec::ModelSpec, &ConditionerPrompt, u32),
    table: &std::collections::BTreeMap<u32, Vec<bf16>>,
    (merged, runs): (&[bf16], &[(usize, usize)]),
    deepstack: &[CudaSlice<bf16>],
) -> Result<Vec<bf16>> {
    let hidden = spec.hidden as usize;
    let embeds = conditioner_embeds(prompt, image_token, table, merged, hidden)?;
    let rope = conditioner_rope(spec, prompt)?;
    let mut taps = conditioner.forward(
        &device.stream,
        blas,
        &ConditionerInput {
            embeds: &embeds,
            deepstack,
            runs,
            rope: &rope,
        },
        &[TEXT_ENCODER_LAYER],
    )?;
    taps.pop().context("conditioner returned no tap")
}

/// Each ordered keyframe maps to normalized `[latent_channels, height / r, width / r]` latents.
fn keyframes(
    device: &Device,
    blas: &Blas,
    model: &Path,
    request: &H3Request,
    eps: Option<&CudaSlice<f32>>,
) -> Result<Vec<Vec<f32>>> {
    if request.keyframes.is_empty() {
        return Ok(Vec::new());
    }
    let eps = eps.context("keyframes require posterior noise")?;
    let spec = VaeSpec::read(model)?;
    let vae_dir = model.join("vae");
    let manifest = MappedCheckpoint::open_component(&vae_dir, "diffusion_pytorch_model")?
        .minimax_h3_vae_manifest(&spec)?;
    let mut stage = VideoEncodeStage::new(
        device,
        (&spec, &vae_dir, &manifest),
        [request.height, request.width],
    )?;
    stage.run(device, blas, request, eps)
}

struct VideoEncodeStage {
    encoder: KeyframeEncoder,
    tiles: VaeTiles,
    work: EncodeWork,
}

impl VideoEncodeStage {
    fn new(
        device: &Device,
        (spec, dir, manifest): (&VaeSpec, &Path, &TensorManifest),
        [height, width]: [usize; 2],
    ) -> Result<Self> {
        let encoder = KeyframeEncoder::new(device, spec, dir, manifest)?;
        let tiles = VaeTiles::released(spec, height, width)?;
        let work = EncodeWork::new(&device.stream, &encoder, (tiles.height, tiles.width))?;
        Ok(Self {
            encoder,
            tiles,
            work,
        })
    }

    fn run(
        &mut self,
        device: &Device,
        blas: &Blas,
        request: &H3Request,
        eps: &CudaSlice<f32>,
    ) -> Result<Vec<Vec<f32>>> {
        let s = &device.stream;
        let mut anchors = Vec::with_capacity(request.keyframes.len());
        for (_, pixels) in &request.keyframes {
            let frame = s.clone_htod(pixels)?;
            let [_, _, latents] = self.encoder.encode(
                (s, blas),
                &mut self.work,
                &self.tiles,
                (&frame, request.height, request.width),
                eps,
            )?;
            #[cfg(test)]
            println!("H3_VIDEO_KEYFRAME_ENCODER_CALL calls=1");
            anchors.push(s.clone_dtoh(&latents)?);
        }
        Ok(anchors)
    }
}

/// Resident BF16 and F32 video weights plus the published F32 audio weights.
fn decoder_bytes(video: &TensorManifest, audio: &TensorManifest) -> usize {
    decoder_weight_bytes(video)
        + audio
            .entries
            .values()
            .filter(|e| e.load)
            .map(|e| e.bytes)
            .sum::<usize>()
}

/// Packed video rows become NCTHW; channel-major stereo rows become NCL. Both outputs are denormalized F32.
fn decoder_latents(
    (spec, vae, audio_spec): (&H3Spec, &VaeSpec, &AudioVaeSpec),
    (geometry, layout): (&Geometry, &PackedLayout),
    (video, audio): (&[f32], &[f32]),
) -> Result<(Vec<f32>, Vec<f32>)> {
    let width = spec.video_row_width();
    ensure!(
        video.len() == layout.video_indices.len() * width
            && audio.len() == layout.audio_indices.len() * spec.audio_in_channels
            && spec.in_channels == vae.latent_channels
            && spec.audio_in_channels == audio_spec.latent_channels,
        "decoder handoff differs from packed video/audio extents"
    );
    let generated = &video[layout.condition_video_rows * width..];
    let mut video_latents = unpatchify_latents(
        generated,
        (
            vae.latent_channels,
            geometry.latent_frames,
            geometry.latent_height,
            geometry.latent_width,
        ),
        spec.patch,
    );
    let plane = geometry.latent_frames * geometry.latent_height * geometry.latent_width;
    for (c, chunk) in video_latents.chunks_mut(plane).enumerate() {
        for v in chunk {
            *v = *v * vae.latents_std[c] + vae.latents_mean[c];
        }
    }
    let (ac, latents) = (audio_spec.latent_channels, geometry.audio_latents);
    let mut audio_latents = vec![0f32; audio.len()];
    for ch in 0..2 {
        for t in 0..latents {
            for c in 0..ac {
                audio_latents[(ch * ac + c) * latents + t] = audio[(ch * latents + t) * ac + c]
                    * audio_spec.latents_std[c]
                    + audio_spec.latents_mean[c];
            }
        }
    }
    Ok((video_latents, audio_latents))
}

/// Runs `request` through the native path with the transformer LoRA at `lora`, cuBLAS calibration in `runtime`.
/// `gguf` replaces the transformer safetensors with one community GGUF file; the conditioner and keyframe VAE
/// stay on the published safetensors.
pub(crate) fn run(
    model: &Path,
    lora: &Path,
    request: &H3Request,
    (index, runtime): (usize, &Path),
    gguf: Option<&Path>,
) -> Result<Generated> {
    run_mode::<false>(
        (model, &model.join("transformer")),
        Some(lora),
        request,
        (index, runtime),
        gguf,
        NoiseDraws::Seed(crate::minimax_h3::noise_draws(request.seed)),
    )
}

pub(crate) fn run_hyperflow(
    (model, transformer): (&Path, &Path),
    request: &H3Request,
    device: (usize, &Path),
) -> Result<Generated> {
    run_mode::<false>(
        (model, transformer),
        None,
        request,
        device,
        None,
        NoiseDraws::Seed(crate::minimax_h3::noise_draws(request.seed)),
    )
}

#[cfg(test)]
pub(crate) fn decoder_setup(
    model: &Path,
    lora: &Path,
    request: &H3Request,
    runtime: &Path,
) -> Result<Generated> {
    run_mode::<true>(
        (model, &model.join("transformer")),
        Some(lora),
        request,
        (0, runtime),
        None,
        NoiseDraws::Seed(crate::minimax_h3::noise_draws(request.seed)),
    )
}

struct VideoTransformer {
    dit: Dit,
    table: AdalnTable,
    adaln: AdalnPlan,
    video_grid: Schedule,
    audio_grid: Schedule,
    scale: f32,
}

fn prepare_transformer(
    (device, blas): (&Device, &mut Blas),
    (transformer, lora, gguf): (&Path, Option<&Path>, Option<&Path>),
    (spec, layout): (&H3Spec, &PackedLayout),
    (evaluations, shifts): (usize, (f64, f64)),
    choose: Option<&mut super::VideoPlacementChooser<'_>>,
) -> Result<VideoTransformer> {
    let hyperflow = spec.is_hyperflow()?;
    #[cfg(feature = "cuda")]
    ensure!(
        !hyperflow || spec.int8_convrot.is_some(),
        "CUDA HyperFlow requires fl2va_int8_convrot"
    );
    let manifest = match gguf {
        Some(path) => {
            let gguf = crate::gguf::GgufFile::open(path)?;
            MappedCheckpoint::minimax_h3_gguf_manifest(spec, &gguf)?
        }
        None => MappedCheckpoint::open_component(transformer, "diffusion_pytorch_model")?
            .minimax_h3_manifest(spec)?,
    };
    let source_dir = match gguf {
        Some(path) => path.parent().context("GGUF directory")?.to_path_buf(),
        None => transformer.to_path_buf(),
    };
    let adapter = lora
        .map(|lora| -> Result<_> {
            let dir = lora.parent().context("LoRA directory")?;
            let stem = lora
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_suffix(".safetensors"))
                .context("LoRA file stem")?;
            let (manifest, adapter) =
                MappedCheckpoint::open_component(dir, stem)?.minimax_h3_lora_manifest(spec)?;
            Ok((dir, manifest, adapter))
        })
        .transpose()?;
    let (rank, scale) = adapter
        .as_ref()
        .map_or((0, 0.0), |(_, _, a)| (a.rank, a.scale));
    let (video_grid, audio_grid) = if hyperflow {
        ensure!(
            shifts == (12.0, 3.0),
            "HyperFlow requires video/audio shifts12/3"
        );
        hyperflow_schedules(evaluations)?
    } else {
        (
            Schedule::new(evaluations + 1, shifts.0)?,
            Schedule::new(evaluations + 1, shifts.1)?,
        )
    };
    let plan = if hyperflow {
        hyperflow_steps(evaluations)?
            .into_iter()
            .map(|step| RowTimesteps::hyperflow(layout, &step))
            .collect()
    } else {
        row_timestep_plan(layout, &video_grid, &audio_grid)?
    };
    let adaln_plan = AdalnPlan::build(layout, &plan)?;
    let table = AdalnTable::build(device, blas, &source_dir, &manifest, spec, &adaln_plan)?;
    let base = Source::new(&source_dir, &manifest, device.info.arch.fp16_stored_half());
    let lora = adapter
        .as_ref()
        .map(|(dir, manifest, _)| Source::new(dir, manifest, device.info.arch.fp16_stored_half()));
    let dit = Dit::with_placement(
        device,
        blas,
        spec,
        (rank, layout, &table),
        base,
        lora,
        (
            Placement::Derived,
            crate::prefill::placement_visits(plan.len(), 1)?,
            choose,
        ),
    )?;
    Ok(VideoTransformer {
        dit,
        table,
        adaln: adaln_plan,
        video_grid,
        audio_grid,
        scale,
    })
}

struct VideoNoise {
    posterior_eps: Option<CudaSlice<f32>>,
    anchor: Vec<f32>,
    video: Vec<f32>,
    audio: CudaSlice<f32>,
}

fn video_noise(
    (device, normal): (&Device, &crate::backend::gpu::CudaFunction),
    (spec, vae): (&H3Spec, &VaeSpec),
    (geometry, layout): (&Geometry, &PackedLayout),
    request: &H3Request,
    draws: &NoiseDraws,
) -> Result<VideoNoise> {
    let s = &device.stream;
    let (r, lc) = (vae.spatial_ratio, vae.latent_channels);
    let (lh, lw) = (request.height / r, request.width / r);
    let plane = lh * lw;
    let latent_frames = geometry.latent_frames;
    let (anchor_count, video_count) = (lc * plane, lc * latent_frames * plane);
    let audio_count = layout.audio_indices.len() * spec.audio_in_channels;

    let condition_count = anchor_count
        .checked_mul(request.keyframes.len())
        .context("conditioning noise extent overflow")?;
    let posterior_eps = if request.keyframes.is_empty() {
        None
    } else {
        Some(draws.draw(device, normal, 0, anchor_count)?)
    };
    let anchor_noise = if condition_count == 0 {
        Vec::new()
    } else {
        let draw = draws.draw(device, normal, 1, condition_count)?;
        s.clone_dtoh(&draw)?
    };
    let video_noise = {
        let draw = draws.draw(device, normal, 2, video_count)?;
        s.clone_dtoh(&draw)?
    };
    let audio = draws.draw(device, normal, 3, audio_count)?;

    Ok(VideoNoise {
        posterior_eps,
        anchor: anchor_noise,
        video: video_noise,
        audio,
    })
}

fn video_rows(
    (spec, vae): (&H3Spec, &VaeSpec),
    (geometry, layout): (&Geometry, &PackedLayout),
    request: &H3Request,
    noise: &VideoNoise,
    (anchors, text): (&[Vec<f32>], &[bf16]),
) -> Result<Vec<f32>> {
    let (r, lc) = (vae.spatial_ratio, vae.latent_channels);
    let (lh, lw) = (request.height / r, request.width / r);
    let plane = lh * lw;
    let latent_frames = geometry.latent_frames;
    let (anchor_count, video_count) = (lc * plane, lc * latent_frames * plane);
    let audio_count = layout.audio_indices.len() * spec.audio_in_channels;

    let VideoNoise {
        anchor: anchor_noise,
        video: video_noise,
        audio,
        ..
    } = noise;
    let condition_count = anchor_count
        .checked_mul(request.keyframes.len())
        .context("conditioning noise extent overflow")?;
    let patch = spec.patch;
    let condition_rows = {
        let t = KEYFRAME_NOISE_AUG as f32;
        let mixed: Vec<f32> = anchors
            .iter()
            .flatten()
            .zip(anchor_noise)
            .map(|(&x, &n)| t * x + (1.0 - t) * n)
            .collect();
        let mut rows = Vec::with_capacity(condition_count);
        for anchor in mixed.chunks_exact(anchor_count) {
            patchify_latents_into(anchor, (lc, 1, lh, lw), [1, patch[1], patch[2]], &mut rows);
        }
        rows
    };
    let width = spec.video_row_width();
    ensure!(
        anchors.len() == request.keyframes.len()
            && anchors.iter().all(|anchor| anchor.len() == anchor_count)
            && condition_rows.len() == layout.condition_video_rows * width
            && video_noise.len() == video_count
            && (layout.video_indices.len() - layout.condition_video_rows) * width == video_count
            && audio.len() == audio_count
            && request.text_tags.len() == text.len() / spec.text_dim,
        "native draws do not match the packed layout: anchor {}, video {}, audio {}, text rows {}",
        anchor_noise.len(),
        video_noise.len(),
        audio.len(),
        request.text_tags.len()
    );
    let mut video_rows = condition_rows;
    patchify_latents_into(
        video_noise,
        (lc, latent_frames, lh, lw),
        patch,
        &mut video_rows,
    );
    Ok(video_rows)
}

fn run_mode<const SETUP_ONLY: bool>(
    (model, transformer): (&Path, &Path),
    lora: Option<&Path>,
    request: &H3Request,
    (index, runtime): (usize, &Path),
    gguf: Option<&Path>,
    draws: NoiseDraws,
) -> Result<Generated> {
    let device = Device::new(index)?;
    let ops = Ops::new(&device.ctx)?;
    let s = &device.stream;
    let mut blas = super::open_blas(
        &device,
        &ops,
        runtime,
        Rc::new(Closure::new(false, "MiniMax-H3 generate".into())),
    )?;
    let config = transformer.join("config.json");
    let spec = H3Spec::parse(&serde_json::from_slice(
        &std::fs::read(&config).with_context(|| format!("read {}", config.display()))?,
    )?)?;
    let hyperflow = spec.is_hyperflow()?;
    #[cfg(feature = "cuda")]
    ensure!(
        !hyperflow || spec.int8_convrot.is_some(),
        "CUDA HyperFlow requires fl2va_int8_convrot"
    );
    ensure!(
        hyperflow == lora.is_none() && (!hyperflow || gguf.is_none()),
        "HyperFlow requires stored fl2va weights without LoRA or GGUF; MiniMax-H3 requires LoRA"
    );
    let vae = VaeSpec::read(model)?;
    let audio_spec = AudioVaeSpec::read(model)?;
    let (geometry, layout) = request.layout(&spec)?;
    let normal = cubin::module(&device.ctx, "sample")?.load_function("philox_normal_f32")?;
    let noise = video_noise(
        (&device, &normal),
        (&spec, &vae),
        (&geometry, &layout),
        request,
        &draws,
    )?;
    let (text, anchors) = match &draws {
        #[cfg(test)]
        NoiseDraws::Recorded(input) => {
            let mut text = input.conditioner.clone();
            if device.info.arch.fp16_stored_half() {
                crate::tensors::bf16_table_to_runtime_half("video_text", &mut text)?;
            }
            (text, input.anchors.clone())
        }
        NoiseDraws::Seed(_) => (
            condition(&device, &mut blas, model, request, runtime)?,
            keyframes(&device, &blas, model, request, noise.posterior_eps.as_ref())?,
        ),
    };
    #[cfg(test)]
    let mut replay = match &draws {
        NoiseDraws::Recorded(_) => Some(Replay {
            initial: (Vec::new(), Vec::new()),
            refiner: Vec::new(),
            steps: Vec::new(),
            chunks: (0, 0),
        }),
        NoiseDraws::Seed(_) => None,
    };

    let video_rows = video_rows(
        (&spec, &vae),
        (&geometry, &layout),
        request,
        &noise,
        (&anchors, &text),
    )?;
    let audio = &noise.audio;
    #[cfg(test)]
    if let Some(replay) = &mut replay {
        replay.initial = (video_rows.clone(), s.clone_dtoh(audio)?);
    }

    let (video, audio, free_before, mut hoist) = {
        let VideoTransformer {
            mut dit,
            table,
            adaln: adaln_plan,
            video_grid,
            audio_grid,
            scale,
        } = prepare_transformer(
            (&device, &mut blas),
            (transformer, lora, gguf),
            (&spec, &layout),
            (request.evaluations, request.shifts),
            None,
        )?;
        #[cfg(test)]
        if let Some(replay) = &mut replay {
            replay.chunks = (dit.work.chunk, dit.text_work.chunk);
            dit.trace = Some(Vec::new());
            if let NoiseDraws::Recorded(input) = &draws {
                dit.trace_gate = Some((input.case, Rc::clone(&input.gate)));
            }
        }
        let kernels = BlockKernels::new(&device)?;
        let context = BlockContext {
            s,
            blas: &blas,
            kernels: &kernels,
            spec: &spec,
            lora_scale: scale,
        };
        let video = s.clone_htod(&video_rows)?;
        let text = s.clone_htod(&text)?;
        let hoist = {
            let disk_tier = !dit.hoist_ready()?;
            super::vae::hoist_video_vae(model, disk_tier, &device.info.arch)?
        };
        dit.sample(
            &context,
            (&table, &adaln_plan),
            (&video_grid, &audio_grid),
            (&video, audio),
            &text,
        )?;
        #[cfg(test)]
        if let Some(replay) = &mut replay {
            replay.steps = dit
                .trace
                .take()
                .context("RUNNER_FAIL sample trace missing")?;
            replay.refiner = dit
                .refiner_trace
                .take()
                .context("RUNNER_FAIL refiner trace missing")?;
        }
        let (video, audio) = (s.clone_dtoh(&video)?, s.clone_dtoh(audio)?);
        (video, audio, device.ctx.mem_get_info()?.0, hoist)
    };
    s.synchronize()?;
    ensure!(
        video.iter().chain(&audio).all(|v| v.is_finite()),
        "non-finite final latents"
    );

    let (video_latents, audio_latents) = decoder_latents(
        (&spec, &vae, &audio_spec),
        (&geometry, &layout),
        (&video, &audio),
    )?;
    let generated = decode::<SETUP_ONLY, false>(
        &device,
        &mut blas,
        (model, request, hoist.as_mut()),
        (&geometry, &vae, &audio_spec),
        (video_latents, audio_latents),
        (free_before, super::Placement::Derived),
    )?;
    #[cfg(test)]
    let generated = Generated {
        #[cfg(feature = "cuda")]
        conditioner: text,
        #[cfg(feature = "cuda")]
        anchors,
        replay,
        ..generated
    };
    Ok(generated)
}

pub(super) fn video_latent_entry(values: &[f32], fp16: bool) -> Result<Vec<bf16>> {
    let mut rounded: Vec<bf16> = values.iter().copied().map(bf16::from_f32).collect();
    ensure!(
        rounded.iter().all(|x| x.is_finite()),
        "video latent input is nonfinite after BF16 entry cast"
    );
    if fp16 {
        crate::tensors::bf16_table_to_runtime_half("video_latent", &mut rounded)?;
    }
    Ok(rounded)
}

#[cfg(test)]
#[test]
fn sm75_vae_latent_entry_stores_fp16_one() {
    let fp16 = video_latent_entry(&[1.0, 0.5, -1.0], true).unwrap();
    assert_eq!(fp16[0].to_bits(), 0x3c00);
    assert_eq!(fp16[1].to_bits(), 0x3800);
    assert_eq!(fp16[2].to_bits(), 0xbc00);
    let kept = video_latent_entry(&[1.0], false).unwrap();
    assert_eq!(kept[0].to_bits(), 0x3f80);
}

fn video_decode_device(
    (device, blas): (&Device, &Blas),
    (decoder, work, tiles): (&VideoDecoder, &mut TileWork, &VaeTiles),
    (geometry, vae, total_frames): (&Geometry, &VaeSpec, usize),
    video_latents: &[f32],
) -> Result<CudaSlice<bf16>> {
    let s = &device.stream;
    let mut decoded =
        s.alloc_zeros::<bf16>(vae.out_channels * total_frames * geometry.height * geometry.width)?;
    let rounded = video_latent_entry(video_latents, device.info.arch.fp16_stored_half())?;
    let latents = s.clone_htod(&rounded)?;
    drop(rounded);
    let plan = decoder.decode(
        s,
        blas,
        work,
        tiles,
        LatentVideo {
            data: &latents,
            frames: geometry.latent_frames,
            height: geometry.height / vae.spatial_ratio,
            width: geometry.width / vae.spatial_ratio,
        },
        &mut decoded,
    )?;
    ensure!(
        plan.frames == geometry.frames,
        "decoded {} frames, resolved geometry has {}",
        plan.frames,
        geometry.frames
    );
    Ok(decoded)
}

fn decode<const SETUP_ONLY: bool, const SERIAL: bool>(
    device: &Device,
    blas: &mut Blas,
    (model, request, hoist): (&Path, &H3Request, Option<&mut super::vae::VaeHoist>),
    (geometry, vae, audio_spec): (&Geometry, &VaeSpec, &AudioVaeSpec),
    (video_latents, audio_latents): (Vec<f32>, Vec<f32>),
    (free_before, placement): (usize, super::Placement),
) -> Result<Generated> {
    let s = &device.stream;
    let r = vae.spatial_ratio;
    #[cfg(test)]
    let lc = vae.latent_channels;
    let (lh, lw) = (request.height / r, request.width / r);
    let latent_frames = geometry.latent_frames;
    let ac = audio_spec.latent_channels;
    let latents = audio_latents.len() / (2 * ac);

    let vae_dir = model.join("vae");
    let vae_manifest = MappedCheckpoint::open_component(&vae_dir, "diffusion_pytorch_model")?
        .minimax_h3_vae_manifest(vae)?;
    let audio_dir = model.join("audio_vae");
    let audio_manifest = MappedCheckpoint::open_component(&audio_dir, "diffusion_pytorch_model")?
        .minimax_h3_audio_vae_manifest(audio_spec)?;
    let tiles = VaeTiles::released(vae, request.height, request.width)?;
    let clip = vae.chunk_latents() + vae.overlap_latents();
    let tile = (clip, tiles.height / r, tiles.width / r);
    let decode_plan = crate::minimax_h3::DecodePlan::new(vae, latent_frames)?;
    let (lead, pre) = (
        vae.chunk_latents() * vae.temporal_ratio,
        vae.frame_pre_padding(),
    );
    let total_frames = decode_plan.chunks * (lead - pre) + clip * vae.temporal_ratio - lead - pre;
    let pixels = request.height * request.width;
    let weights = decoder_bytes(&vae_manifest, &audio_manifest);
    let work_bytes = TileWork::bytes(vae, tile)
        + decode_bytes(vae, &tiles, (request.height, request.width))
        + 2 * vae.out_channels * total_frames * pixels
        + 2 * video_latents.len()
        + 4 * audio_latents.len()
        + AudioDecoder::work_bytes(audio_spec, (2, latents))?;
    let planned = weights + work_bytes;
    let (free_after, total) = device.ctx.mem_get_info()?;
    if crate::backend::diagnostics_enabled() {
        eprintln!(
            "{}",
            serde_json::json!({"h3_release": {"free_before_release_bytes": free_before,
                "free_after_release_bytes": free_after,
                "decoder_weight_bytes": weights, "decode_work_bytes": work_bytes,
                "planned_bytes": planned, "total_bytes": total}})
        );
    }
    let decoder_weights = decoder_weight_bytes(&vae_manifest);
    let future =
        work_bytes + weights - decoder_weights + 3 * decode_plan.frames * pixels + (1 << 30);
    let decoder = VideoDecoder::new(
        device,
        vae,
        &vae_dir,
        &vae_manifest,
        (future, placement),
        hoist,
    )?;
    #[cfg(test)]
    if SETUP_ONLY {
        decoder.batch_setup(device, blas, &tiles, (latent_frames, lh, lw))?;
        anyhow::bail!("H3_VAE_SETUP_COMPLETE");
    }
    let mut work = decoder.work(device, &tiles, (latent_frames, lh, lw))?;
    #[cfg(test)]
    if SERIAL {
        drop(work);
        work = TileWork::new(s, vae, tile, (1, tiles.count()))?;
    }
    decoder.bind(blas, &work)?;
    let decoded = video_decode_device(
        (device, blas),
        (&decoder, &mut work, &tiles),
        (geometry, vae, total_frames),
        &video_latents,
    )?;
    let frames = geometry.frames;
    drop(work);
    #[cfg(test)]
    let video_decoded = s
        .clone_dtoh(&decoded)?
        .into_iter()
        .map(bf16::to_f32)
        .collect();
    let rgb = decoder.rgb(device, &decoded, (frames, total_frames, pixels))?;
    let video_u8 = s.clone_dtoh(&rgb)?;
    drop((rgb, decoded, decoder));
    let audio_decoder = AudioDecoder::new(device, audio_spec, &audio_dir, &audio_manifest)?;
    let audio_device = s.clone_htod(&audio_latents)?;
    let pcm = s.clone_dtoh(&audio_decoder.decode((s, blas), &audio_device, (2, latents))?)?;

    Ok(Generated {
        output: H3Output {
            frames,
            height: request.height,
            width: request.width,
            video: video_u8,
            samples: pcm.len() / 2,
            sampling_rate: audio_spec.sampling_rate,
            audio: pcm,
        },
        #[cfg(all(test, feature = "cuda"))]
        conditioner: Vec::new(),
        #[cfg(all(test, feature = "cuda"))]
        anchors: Vec::new(),
        #[cfg(test)]
        video_latents: (vec![1, lc, latent_frames, lh, lw], video_latents),
        #[cfg(test)]
        audio_latents: (vec![2, ac, latents], audio_latents),
        #[cfg(test)]
        video_decoded: (
            vec![
                1,
                vae.out_channels,
                total_frames,
                request.height,
                request.width,
            ],
            video_decoded,
        ),
        #[cfg(test)]
        replay: None,
    })
}

/// Greedy expansion. The published sampler is not used. `aspect_ratio` is required and ignored.
fn expand_prompt(
    dir: &Path,
    prompt: &str,
    options: &crate::minimax_h3::ImageRunOptions,
) -> Result<String> {
    let system = std::fs::read_to_string(dir.join("prompt/t2i/system_prompt.txt"))
        .context("expander system prompt")?;
    let task = std::fs::read_to_string(dir.join("prompt/t2i/task_prompt.txt"))
        .context("expander task prompt")?;
    let rendered = crate::serve::http::render_image_messages(
        dir,
        &system,
        &format!("{task}\n\ninput_text:\n{prompt}"),
    )?;
    let ids = crate::serve::detok::Shared::load(dir)?.encode_chat(&rendered)?;
    let spec = crate::model_spec::ModelSpec::read(dir)?;
    let room = ids
        .len()
        .checked_add(crate::minimax_h3::IMAGE_MAX_SEQUENCE_LENGTH)
        .context("expander capacity overflows")?;
    ensure!(
        room <= spec.max_position as usize,
        "expander prompt needs {room} positions, the model has {}",
        spec.max_position
    );
    let settings = crate::backend::Settings {
        tree_builder: crate::backend::TreeBuilder::Waves,
        spec_budget: crate::backend::SpecBudget::Chain,
        spec_graph: false,
        draft_model: None,
        prefix_cache: false,
        kv_format: crate::kv::KvFormat::Bf16,
        runtime_dir: Some(options.runtime_dir.clone()),
        device_budget_gib: None,
    };
    let mut model = crate::model::Model::load(dir, options.device, room, None, settings)?;
    let generated = crate::model::generate_plain(
        &mut model,
        &ids,
        crate::minimax_h3::IMAGE_MAX_SEQUENCE_LENGTH,
        false,
        false,
        None,
        &mut |_| true,
    )?;
    drop(model);
    let text = crate::tokenizer::decode(dir, &generated.ids)?;
    let parts: Vec<&str> = text.split("</think>").collect();
    ensure!(
        parts.len() == 2,
        "expander output is not one thinking block"
    );
    let value: serde_json::Value =
        serde_json::from_str(parts[1].trim()).context("expander output is not JSON")?;
    let obj = value
        .as_object()
        .context("expander output is not an object")?;
    ensure!(
        obj.len() == 2 && obj.contains_key("expanded_caption") && obj.contains_key("aspect_ratio"),
        "expander output keys"
    );
    let caption = obj
        .get("expanded_caption")
        .and_then(|v| v.as_str())
        .context("expander expanded_caption")?;
    ensure!(
        obj.get("aspect_ratio").and_then(|v| v.as_str()).is_some(),
        "expander aspect_ratio"
    );
    ensure!(!caption.is_empty(), "expander caption is empty");
    Ok(caption.to_string())
}

pub(super) fn image_condition(
    model: &Path,
    options: &crate::minimax_h3::ImageRunOptions,
    prompt: &ConditionerPrompt,
) -> Result<Vec<bf16>> {
    let device = Device::new(options.device)?;
    let ops = Ops::new(&device.ctx)?;
    let mut blas = super::open_blas(
        &device,
        &ops,
        &options.runtime_dir,
        Rc::new(Closure::new(true, "MiniMax-H3-Image conditioner".into())),
    )?;
    condition_prompt(&device, &mut blas, model, prompt, &options.runtime_dir)
}

fn image_noise(
    options: &crate::minimax_h3::ImageRunOptions,
    seed: u64,
    count: usize,
) -> Result<Vec<f32>> {
    let device = Device::new(options.device)?;
    let normal = cubin::module(&device.ctx, "sample")?.load_function("philox_normal_f32")?;
    image_noise_draws(&device, &normal, seed, count)
}

fn image_noise_draws(
    device: &Device,
    normal: &crate::backend::gpu::CudaFunction,
    seed: u64,
    count: usize,
) -> Result<Vec<f32>> {
    let draw = super::draw_normals(
        &device.stream,
        normal,
        (seed, crate::minimax_h3::STREAM_IMAGE_NOISE),
        count,
        256,
    )?;
    let out = device.stream.clone_dtoh(&draw)?;
    ensure!(
        out.len() == count && out.iter().all(|v| v.is_finite()),
        "non-finite image noise"
    );
    Ok(out)
}

fn fitted_rgb(
    image: &crate::serve::image::RgbImage,
    width: usize,
    height: usize,
) -> Result<crate::serve::image::RgbImage> {
    let pixels = if width == image.width && height == image.height {
        image.pixels.clone()
    } else {
        crate::serve::image::resize_lanczos(
            &image.pixels,
            [image.height, image.width, 3],
            [height, width],
            crate::serve::image::ResizeRounding::F64,
        )?
    };
    Ok(crate::serve::image::RgbImage {
        width,
        height,
        pixels,
    })
}

fn vision_rgb(
    spec: &crate::minimax_h3::ImageSpec,
    image: &crate::serve::image::RgbImage,
) -> Result<crate::serve::image::RgbImage> {
    let (width, height) =
        crate::minimax_h3::image_vision_reference_size(spec, image.width, image.height)?;
    fitted_rgb(image, width, height)
}

/// F32 RGB HWC in `[0, 255]` at the VAE reference size, and the uncropped posterior-epsilon length.
fn vae_rgb(
    spec: &crate::minimax_h3::ImageSpec,
    image: &crate::serve::image::RgbImage,
) -> Result<(Vec<f32>, [usize; 2], usize)> {
    let (width, height) =
        crate::minimax_h3::image_vae_reference_size(spec, image.width, image.height)?;
    let fitted = fitted_rgb(image, width, height)?;
    let rgb: Vec<f32> = fitted
        .pixels
        .iter()
        .map(|&pixel| f32::from(pixel))
        .collect();
    let ratio = crate::minimax_h3::VAE_SPATIAL_RATIO;
    let posterior = (height / ratio)
        .checked_mul(width / ratio)
        .and_then(|plane| plane.checked_mul(spec.in_channels))
        .context("reference posterior extent overflows")?;
    Ok((rgb, [height, width], posterior))
}

fn image_presentation(
    model: &Path,
    spec: &crate::minimax_h3::ImageSpec,
    prompt: &str,
    references: &[crate::minimax_h3::ImageReference],
) -> Result<(Vec<u32>, Vec<u8>, ConditionerPrompt)> {
    let encoder = model.join("text_encoder");
    let text_spec = crate::model_spec::ModelSpec::read(&encoder)?;
    let vision = text_spec
        .vision
        .as_ref()
        .context("text encoder without vision")?;
    let tokenizer = crate::serve::detok::Shared::load(&encoder)?;
    let prompt_ids = tokenizer.encode_chat(prompt)?;
    let modality_of = |ids: &[u32]| -> Vec<u8> {
        ids.iter()
            .map(|&id| u8::from(id == vision.image_token))
            .collect()
    };
    if references.is_empty() {
        let (ids, tags) = crate::minimax_h3::image_text_rows(&prompt_ids)?;
        return Ok((
            ids.clone(),
            tags,
            ConditionerPrompt {
                modality: modality_of(&ids),
                tokens: ids,
                grids: Vec::new(),
                pixels: Vec::new(),
            },
        ));
    }
    let processor = crate::serve::image::Processor::load(&encoder, Some(&tokenizer))?
        .context("text encoder has no image processor")?;
    let start = tokenizer.token_id("<|vision_start|>")?;
    let pad = tokenizer.token_id("<|image_pad|>")?;
    let end = tokenizer.token_id("<|vision_end|>")?;
    ensure!(
        start == vision.start_token && pad == vision.image_token && end == vision.end_token,
        "vision special tokens"
    );
    let mut labels = Vec::with_capacity(references.len());
    let mut blocks = Vec::with_capacity(references.len());
    let mut grids = Vec::with_capacity(references.len());
    let mut pixels = Vec::new();
    for (index, reference) in references.iter().enumerate() {
        labels.push(tokenizer.encode_chat(&crate::minimax_h3::image_picture_label(index))?);
        let packed = processor.prepare_rgb(&vision_rgb(spec, &reference.image)?)?;
        ensure!(
            packed.grids.len() == 1,
            "vision processor returned no single grid"
        );
        let grid = packed.grids[0];
        let tokens = grid.feature_rows(vision.merge)?;
        let mut block = Vec::with_capacity(tokens + 2);
        block.push(start);
        block.extend(std::iter::repeat_n(pad, tokens));
        block.push(end);
        blocks.push(block);
        grids.push(grid);
        pixels.extend(packed.pixel_values);
    }
    let label_rows: Vec<&[u32]> = labels.iter().map(Vec::as_slice).collect();
    let block_rows: Vec<&[u32]> = blocks.iter().map(Vec::as_slice).collect();
    let (ids, tags) = crate::minimax_h3::image_edit_rows(&label_rows, &block_rows, &prompt_ids)?;
    Ok((
        ids.clone(),
        tags,
        ConditionerPrompt {
            modality: modality_of(&ids),
            tokens: ids,
            grids,
            pixels,
        },
    ))
}

#[derive(PartialEq)]
struct ImageShape {
    canvas: [usize; 2],
    latent: [usize; 2],
    output: [usize; 2],
    reference: Option<[usize; 2]>,
    grids: Vec<crate::vision_input::ImageGrid>,
    tags: Vec<u8>,
}

impl ImageShape {
    fn check(&self, actual: &Self) -> Result<()> {
        ensure!(
            self == actual,
            "image request differs from admitted canvas {:?}, output {:?}, reference {:?}, text rows {}, and grids {:?}",
            self.canvas,
            self.output,
            self.reference,
            self.tags.len(),
            self.grids
        );
        Ok(())
    }

    fn layout(&self, spec: &ImageSpec) -> Result<ImageLayout> {
        let reference = self
            .reference
            .map(|[height, width]| {
                crate::minimax_h3::image_reference_latent_extent(spec, height, width)
            })
            .transpose()?;
        crate::minimax_h3::image_layout(
            spec,
            &self.tags,
            reference.as_slice(),
            self.latent[0],
            self.latent[1],
        )
    }
}

struct ImageInput {
    shape: ImageShape,
    prompt: ConditionerPrompt,
}

impl ImageInput {
    fn read(model: &Path, spec: &ImageSpec, request: &ImageRequest) -> Result<Self> {
        ensure!(
            !request.prompt.is_empty() && request.references.len() <= 1,
            "resident image requests require a prompt and at most one reference"
        );
        ensure!(
            request.expander.is_none(),
            "prompt expander is outside the resident workload"
        );
        let (height, width) = crate::minimax_h3::image_canvas(
            spec,
            request.height,
            request.width,
            request.min_generation_area,
        )?;
        let (lh, lw) = crate::minimax_h3::image_latent_extent(spec, height, width)?;
        let reference = request
            .references
            .first()
            .map(|reference| {
                crate::minimax_h3::image_vae_reference_size(
                    spec,
                    reference.image.width,
                    reference.image.height,
                )
                .map(|(width, height)| [height, width])
            })
            .transpose()?;
        let (_, tags, prompt) =
            image_presentation(model, spec, &request.prompt, &request.references)?;
        Ok(Self {
            shape: ImageShape {
                canvas: [height, width],
                latent: [lh, lw],
                output: [request.height, request.width],
                reference,
                grids: prompt.grids.clone(),
                tags,
            },
            prompt,
        })
    }
}

struct ConditionerStage {
    dir: PathBuf,
    spec: crate::model_spec::ModelSpec,
    manifest: TensorManifest,
    vision: Option<LayerNormTower>,
    conditioner: Conditioner,
    blas: Blas,
    image_token: u32,
}

fn conditioner_vision_input(
    prompt: &ConditionerPrompt,
) -> crate::vision_input::MultimodalInput<'_> {
    crate::vision_input::MultimodalInput {
        input_ids: &prompt.tokens,
        modality_ids: &prompt.modality,
        pixel_values: &prompt.pixels,
        grids: &prompt.grids,
        audio: &[],
    }
}

impl ConditionerStage {
    fn run(&mut self, device: &Device, prompt: &ConditionerPrompt) -> Result<Vec<bf16>> {
        let runs = image_runs(&prompt.tokens, self.image_token);
        ensure!(
            runs.is_empty() == self.vision.is_none(),
            "resident vision input changed"
        );
        let fp16 = device.info.arch.fp16_stored_half();
        let table = conditioner_table(
            &self.dir,
            &self.manifest,
            prompt,
            self.image_token,
            self.spec.hidden as usize,
            fp16,
        )?;
        let encoded = self
            .vision
            .as_ref()
            .map(|tower| -> Result<_> {
                let input = conditioner_vision_input(prompt);
                let prepared =
                    tower.prepare(device, &mut self.blas, &input, prompt.tokens.len(), 0)?;
                let encoded = tower.encode(&device.stream, &self.blas, prepared, None)?;
                #[cfg(test)]
                println!(
                    "HYPERFLOW_VISION_ENCODER_CALL rows={} calls=1",
                    prompt.tokens.len()
                );
                Ok(encoded)
            })
            .transpose()?;
        let merged = encoded
            .as_ref()
            .map(|encoded| device.stream.clone_dtoh(&encoded.features))
            .transpose()?
            .unwrap_or_default();
        finish_conditioner(
            (device, &self.blas),
            &mut self.conditioner,
            (&self.spec, prompt, self.image_token),
            &table,
            (&merged, &runs),
            encoded
                .as_ref()
                .map_or(&[], |item| item.deepstack.as_slice()),
        )
    }
}

#[cfg(feature = "cuda")]
struct VideoShape {
    geometry: Geometry,
    tags: Vec<u8>,
    grids: Vec<crate::vision_input::ImageGrid>,
    anchors: Vec<crate::minimax_h3::KeyframeAnchor>,
    shifts: (f64, f64),
    evaluations: usize,
    pixels: usize,
}

#[cfg(feature = "cuda")]
impl VideoShape {
    fn new(geometry: Geometry, request: &H3Request) -> Self {
        Self {
            geometry,
            tags: request.text_tags.clone(),
            grids: request.grids.clone(),
            anchors: request
                .keyframes
                .iter()
                .map(|(anchor, _)| *anchor)
                .collect(),
            shifts: request.shifts,
            evaluations: request.evaluations,
            pixels: request.pixel_values.len(),
        }
    }

    fn check(&self, spec: &H3Spec, request: &H3Request) -> Result<()> {
        ensure!(
            request.geometry(spec)? == self.geometry
                && request.text_tags == self.tags
                && request.grids == self.grids
                && request
                    .keyframes
                    .iter()
                    .map(|(anchor, _)| *anchor)
                    .eq(self.anchors.iter().copied())
                && request.shifts == self.shifts
                && request.evaluations == self.evaluations
                && request.pixel_values.len() == self.pixels,
            "video request differs from the resident canvas, packed layout, or schedule"
        );
        Ok(())
    }
}

#[cfg(feature = "cuda")]
struct VideoDecodeGeometry {
    tiles: VaeTiles,
    tile: (usize, usize, usize),
    total_frames: usize,
    chunks: usize,
    transient: usize,
    rgb_bytes: usize,
}

#[cfg(feature = "cuda")]
impl VideoDecodeGeometry {
    fn new(spec: &VaeSpec, geometry: &Geometry) -> Result<Self> {
        let (h, w) = (geometry.height, geometry.width);
        let tiles = VaeTiles::released(spec, h, w)?;
        let clip = spec.chunk_latents() + spec.overlap_latents();
        let r = spec.spatial_ratio;
        let tile = (clip, tiles.height / r, tiles.width / r);
        let plan = crate::minimax_h3::DecodePlan::new(spec, geometry.latent_frames)?;
        let (lead, pre) = (
            spec.chunk_latents() * spec.temporal_ratio,
            spec.frame_pre_padding(),
        );
        let total_frames = plan.chunks * (lead - pre) + clip * spec.temporal_ratio - lead - pre;
        let pixels = h * w;
        let decoded = 2 * spec.out_channels * total_frames * pixels;
        let latent = 2 * spec.latent_channels * geometry.latent_frames * (h / r) * (w / r);
        let grid = 4 * (2 * (tiles.rows.0.len() + tiles.cols.0.len()) - 2);
        let tile_scratch = decode_bytes(spec, &tiles, (h, w)) + latent + grid;
        let rgb_bytes = plan
            .frames
            .checked_mul(pixels)
            .and_then(|n| n.checked_mul(3))
            .context("resident video RGB extent overflow")?;
        Ok(Self {
            tiles,
            tile,
            total_frames,
            chunks: plan.chunks,
            transient: decoded + tile_scratch,
            rgb_bytes,
        })
    }
}

#[cfg(feature = "cuda")]
struct DecoderVisitCost {
    weight_bytes: usize,
    weight_seconds: f64,
    launches: usize,
    launch_seconds: f64,
    handoff_seconds: f64,
}

#[cfg(feature = "cuda")]
impl DecoderVisitCost {
    fn measure(
        device: &Device,
        normal: &crate::backend::gpu::CudaFunction,
        layers: usize,
        weight_bytes: usize,
    ) -> Result<Self> {
        use cudarc::driver::PushKernelArg;
        use std::time::Instant;
        let stream = &device.stream;
        let out = stream.alloc_zeros::<f32>(1)?;
        let ready = device.ctx.new_event(None)?;
        let retired = device.ctx.new_event(None)?;
        let mut launches = [0.0; 5];
        let mut handoffs = [0.0; 5];
        for trial in 0..=launches.len() {
            stream.synchronize()?;
            let start = Instant::now();
            for _ in 0..128 {
                unsafe {
                    stream
                        .launch_builder(normal)
                        .arg(&0u64)
                        .arg(&0u64)
                        .arg(&out)
                        .arg(&0i64)
                        .launch(crate::backend::cuda::ops::grid(1, 1))?;
                }
            }
            stream.synchronize()?;
            let launch = start.elapsed().as_secs_f64() / 128.0;
            let start = Instant::now();
            for _ in 0..128 {
                ready.record(&device.upload)?;
                stream.wait(&ready)?;
                retired.record(stream)?;
                device.upload.wait(&retired)?;
            }
            device.upload.synchronize()?;
            if trial > 0 {
                launches[trial - 1] = launch;
                handoffs[trial - 1] = start.elapsed().as_secs_f64() / 256.0;
            }
        }
        launches.sort_by(f64::total_cmp);
        handoffs.sort_by(f64::total_cmp);
        Ok(Self {
            weight_bytes,
            weight_seconds: weight_bytes as f64 / device.info.memory_bytes_per_second,
            // Six outer submissions and six GEMMs plus five kernels per decoder block.
            launches: 6 + 11 * layers,
            launch_seconds: launches[2],
            handoff_seconds: handoffs[2],
        })
    }

    fn price_groups(
        &self,
        (vae, geometry, cap): (&VaeSpec, &VideoDecodeGeometry, usize),
        mut stages: [crate::prefill::DenseStage<'_>; 3],
        (stage_and_work, budget, host, page): (usize, usize, usize, usize),
        rates: (f64, f64),
        ring_workers: usize,
    ) -> Result<(
        usize,
        usize,
        crate::prefill::PricedDense,
        Vec<serde_json::Value>,
    )> {
        use crate::prefill::{PricedDense, try_price_dense_stages};
        let slots = geometry.tiles.count();
        let mut best = None;
        let mut alternatives = Vec::new();
        for batch in (1..=cap).rev() {
            let groups = slots.div_ceil(batch);
            // Within a visit count the smallest workspace leaves the most space for weights.
            if batch < cap.saturating_sub(1) && batch > slots.div_ceil(groups) {
                continue;
            }
            let charged = TileWork::charged_bytes(
                vae,
                geometry.tile,
                batch,
                slots,
                page,
                geometry.rgb_bytes,
            )?;
            if charged > budget {
                continue;
            }
            let visits = geometry.chunks * groups;
            stages[2].visits = visits;
            let stage_limit = stage_and_work - charged;
            let Some(priced) = try_price_dense_stages(
                stages,
                (stage_limit, host, page),
                rates,
                self.streamed_block_seconds(),
                ring_workers,
            )?
            else {
                continue;
            };
            let tier_seconds = priced
                .decoder_streamed_seconds
                .into_iter()
                .chain(priced.decoder_resident_seconds)
                .min_by(f64::total_cmp)
                .context("decoder placement lacks a price")?;
            let seconds = tier_seconds + visits as f64 * self.group_seconds();
            let streamed = priced.plans[2].slots.iter().flatten().count();
            let demand = |disk_only| {
                priced
                    .plans
                    .iter()
                    .zip(stages)
                    .map(|(plan, stage)| {
                        stage.visits
                            * (plan.disk_bytes + if disk_only { 0 } else { plan.host_bytes })
                    })
                    .sum::<usize>()
            };
            alternatives.push(serde_json::json!({
                "batch": batch, "visits": visits, "work_charge": charged, "stage_limit": stage_limit,
                "source_bytes": demand(false), "disk_bytes": demand(true),
                "decoder_weight_bytes": visits * self.weight_bytes,
                "decoder_launches": visits * (self.launches + 12 * streamed),
                "decoder_stream_handoffs": visits * 2 * streamed, "seconds": seconds,
            }));
            if best
                .as_ref()
                .is_none_or(|(previous, ..): &(f64, usize, usize, PricedDense)| seconds < *previous)
            {
                best = Some((seconds, batch, stage_limit, priced));
            }
        }
        let (_, batch, device_budget, priced) = best.with_context(|| format!(
            "resident video has no decoder group and dense placement within device={stage_and_work} host={host} bytes"
        ))?;
        Ok((batch, device_budget, priced, alternatives))
    }

    fn group_seconds(&self) -> f64 {
        self.weight_seconds + self.launches as f64 * self.launch_seconds
    }

    fn streamed_block_seconds(&self) -> f64 {
        // Six BF16 projections narrow one weight and bias each. Ready and retired cross streams.
        12.0 * self.launch_seconds + 2.0 * self.handoff_seconds
    }
}

#[cfg(feature = "cuda")]
fn pre_fault_bytes(len: usize) -> Vec<u8> {
    let mut buffer = vec![0u8; len];
    if len == 0 {
        return buffer;
    }
    let page = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
        .ok()
        .filter(|&size| size > 0)
        .unwrap_or(4096);
    let pages = len.div_ceil(page);
    let workers = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(pages);
    let touch = |slice: &mut [u8]| {
        for chunk in slice.chunks_mut(page) {
            // SAFETY: chunk is non-empty and exclusively borrowed; volatile write forces page faults on zeroed pages.
            unsafe {
                std::ptr::write_volatile(chunk.as_mut_ptr(), 0);
            }
        }
    };
    if workers <= 1 {
        touch(&mut buffer);
        return buffer;
    }
    let base = pages / workers;
    let extra = pages % workers;
    std::thread::scope(|scope| {
        let mut rest = buffer.as_mut_slice();
        for index in 0..workers {
            let count = base + usize::from(index < extra);
            let bytes = (count * page).min(rest.len());
            let (slice, next) = rest.split_at_mut(bytes);
            rest = next;
            scope.spawn(move || touch(slice));
        }
    });
    buffer
}

#[cfg(feature = "cuda")]
fn readback_pinned_chunks(
    upload: &std::sync::Arc<crate::backend::gpu::CudaStream>,
    src_ptr: u64,
    dst: &mut [u8],
    staging: &mut [u8],
    done: &[crate::backend::gpu::CudaEvent; 2],
) -> Result<()> {
    if dst.is_empty() {
        return Ok(());
    }
    upload.context().bind_to_thread()?;
    let result: Result<()> = (|| {
        if staging.len() >= dst.len() {
            // SAFETY: staging and src_ptr..src_ptr+dst.len() stay live until done[0] synchronizes.
            unsafe {
                crate::backend::cuda::copy::dtoh_raw(
                    staging.as_mut_ptr().cast(),
                    src_ptr,
                    dst.len(),
                    upload,
                )?;
            }
            done[0].record(upload)?;
            done[0].synchronize()?;
            dst.copy_from_slice(&staging[..dst.len()]);
            return Ok(());
        }
        let chunk = staging.len() / 2;
        ensure!(
            chunk > 0,
            "RGB pinned staging requires at least 2 bytes for chunked readback"
        );
        let (slot0, rest) = staging.split_at_mut(chunk);
        let slots = [slot0, &mut rest[..chunk]];
        let mut pending: Option<(usize, usize, usize)> = None;
        let mut slot_idx = 0usize;
        let mut offset = 0usize;
        while offset < dst.len() {
            let len = chunk.min(dst.len() - offset);
            // SAFETY: slots[slot_idx] was synchronized and copied before reuse; src_ptr stays live until return.
            unsafe {
                crate::backend::cuda::copy::dtoh_raw(
                    slots[slot_idx].as_mut_ptr().cast(),
                    src_ptr + offset as u64,
                    len,
                    upload,
                )?;
            }
            done[slot_idx].record(upload)?;
            if let Some((prev_slot, prev_offset, prev_len)) =
                pending.replace((slot_idx, offset, len))
            {
                done[prev_slot].synchronize()?;
                dst[prev_offset..prev_offset + prev_len]
                    .copy_from_slice(&slots[prev_slot][..prev_len]);
            }
            slot_idx ^= 1;
            offset += len;
        }
        if let Some((prev_slot, prev_offset, prev_len)) = pending {
            done[prev_slot].synchronize()?;
            dst[prev_offset..prev_offset + prev_len].copy_from_slice(&slots[prev_slot][..prev_len]);
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = upload.synchronize();
    }
    result
}

#[cfg(feature = "cuda")]
struct VideoDecodeStage {
    geometry: VideoDecodeGeometry,
    video: VideoDecoder,
    audio: AudioDecoder,
    work: TileWork,
    rgb_staging: crate::backend::gpu::PinnedHostSlice<u8>,
    rgb_ready: crate::backend::gpu::CudaEvent,
    rgb_done: [crate::backend::gpu::CudaEvent; 2],
}

#[cfg(feature = "cuda")]
impl VideoDecodeStage {
    fn render(
        &mut self,
        (device, blas): (&Device, &Blas),
        (geometry, vae, audio_spec): (&Geometry, &VaeSpec, &AudioVaeSpec),
        (video, audio): (&[f32], &[f32]),
    ) -> Result<H3Output> {
        use crate::backend::gpu::DevicePtr;
        let decoded = video_decode_device(
            (device, blas),
            (&self.video, &mut self.work, &self.geometry.tiles),
            (geometry, vae, self.geometry.total_frames),
            video,
        )?;
        let s = &device.stream;
        let rgb = self.work.rgb_output(self.geometry.rgb_bytes)?;
        self.video.rgb_into(
            device,
            &decoded,
            &rgb,
            (
                geometry.frames,
                self.geometry.total_frames,
                geometry.height * geometry.width,
            ),
        )?;
        self.rgb_ready.record(s)?;
        // Stream-ordered release returns decoded VRAM before audio_decoder allocates on `s`.
        drop(decoded);
        let (rgb_ptr, _rgb_guard) = rgb.device_ptr(&device.upload);
        device.upload.wait(&self.rgb_ready)?;
        let mut video = pre_fault_bytes(self.geometry.rgb_bytes);
        let staging = self.rgb_staging.as_mut_slice()?;
        let done = &self.rgb_done;
        let (readback, audio) = std::thread::scope(|scope| {
            let d2h = scope.spawn(|| {
                readback_pinned_chunks(&device.upload, rgb_ptr, &mut video, staging, done)
            });
            let audio = (|| -> Result<Vec<f32>> {
                let input = s.clone_htod(audio)?;
                let decoded_audio =
                    self.audio
                        .decode((s, blas), &input, (2, geometry.audio_latents))?;
                Ok(s.clone_dtoh(&decoded_audio)?)
            })();
            (
                d2h.join().expect("RGB pinned readback thread panicked"),
                audio,
            )
        });
        drop(_rgb_guard);
        readback?;
        let audio = audio?;
        Ok(H3Output {
            frames: geometry.frames,
            height: geometry.height,
            width: geometry.width,
            video,
            samples: audio.len() / 2,
            sampling_rate: audio_spec.sampling_rate,
            audio,
        })
    }
}

#[cfg(feature = "cuda")]
/// Video requests sharing one packed geometry, schedule and set of loaded weights.
pub struct VideoSession {
    device: Device,
    shape: VideoShape,
    layout: PackedLayout,
    spec: H3Spec,
    vae: VaeSpec,
    audio: AudioVaeSpec,
    normal: crate::backend::gpu::CudaFunction,
    text: ConditionerStage,
    encode: Option<(VideoEncodeStage, Blas)>,
    transform: VideoTransformer,
    sample_blas: Blas,
    kernels: BlockKernels,
    decode: VideoDecodeStage,
}

#[cfg(feature = "cuda")]
impl VideoSession {
    pub fn open(
        (model, companion): (&Path, Option<&Path>),
        (lora, gguf): (Option<&Path>, Option<&Path>),
        options: &crate::RunOptions,
        requests: impl IntoIterator<Item = Result<H3Request>>,
    ) -> Result<Self> {
        use crate::backend::cuda::ops::CommonOpsRef;
        use crate::prefill::{DensePlacement, DenseStage, PricedDense};
        let companion_model = companion.unwrap_or(model);
        let transformer = model.join(if companion.is_some() {
            "fl2va_int8_convrot"
        } else {
            "transformer"
        });
        let config = transformer.join("config.json");
        let spec = H3Spec::parse(&serde_json::from_slice(
            &std::fs::read(&config).with_context(|| format!("read {}", config.display()))?,
        )?)?;
        let hyperflow = spec.is_hyperflow()?;
        ensure!(
            hyperflow == companion.is_some()
                && hyperflow == lora.is_none()
                && (!hyperflow || gguf.is_none()),
            "HyperFlow requires its companion model and stored fl2va weights; MiniMax-H3 requires LoRA"
        );
        let vae = VaeSpec::read(companion_model)?;
        let audio = AudioVaeSpec::read(companion_model)?;
        let text_dir = companion_model.join("text_encoder");
        let text_spec = crate::model_spec::ModelSpec::read(&text_dir)?;
        let image_token = text_spec
            .vision
            .as_ref()
            .context("text encoder without vision")?
            .image_token;
        let mut requests = requests.into_iter();
        let first = requests
            .next()
            .context("resident video workload is empty")??;
        let (geometry, layout) = first.layout(&spec)?;
        let shape = VideoShape::new(geometry, &first);
        let evaluations = first.evaluations;
        let shifts = first.shifts;
        let count = first.token_ids.len();
        let vision_bytes = if shape.anchors.is_empty() {
            0
        } else {
            let prompt = video_conditioner_prompt(&first, image_token);
            LayerNormTower::request_bytes(&text_spec, &conditioner_vision_input(&prompt), count)?
        };
        drop(first);
        let mut request_count = 1;
        for request in requests {
            shape.check(&spec, &request?)?;
            request_count += 1;
        }
        let device = Device::new(options.device)?;
        let page = stored_allocation_granularity(&device)?;
        let ops = Ops::new(&device.ctx)?;
        let blas = |name: &str| {
            Blas::new(
                &device,
                &ops,
                &options.runtime_dir,
                Rc::new(Closure::new(false, name.into())),
            )
        };
        let normal = cubin::module(&device.ctx, "sample")?.load_function("philox_normal_f32")?;
        let text_manifest = MappedCheckpoint::open_component(&text_dir, "model")?
            .minimax_h3_conditioner_manifest(&text_spec, TEXT_ENCODER_LAYER)?;
        let vision = if shape.anchors.is_empty() {
            None
        } else {
            let mut catalog = WeightCatalog::load_with_policy(
                &text_dir,
                &text_spec,
                &device,
                LoadPolicy::VisionOnly,
            )?;
            let tower = LayerNormTower::from_catalog(
                &text_spec,
                &mut catalog,
                &device,
                CommonOpsRef::new(&device.ctx)?,
            )?;
            ensure!(
                catalog.tensors.is_empty(),
                "vision binding left unloaded owners"
            );
            device.finish_upload()?;
            Some(tower)
        };
        let vae_dir = companion_model.join("vae");
        let vae_manifest = MappedCheckpoint::open_component(&vae_dir, "diffusion_pytorch_model")?
            .minimax_h3_vae_manifest(&vae)?;
        let encode = if shape.anchors.is_empty() {
            None
        } else {
            Some((
                VideoEncodeStage::new(
                    &device,
                    (&vae, &vae_dir, &vae_manifest),
                    [geometry.height, geometry.width],
                )?,
                blas("H3 keyframe encoder")?,
            ))
        };
        let audio_dir = companion_model.join("audio_vae");
        let audio_manifest =
            MappedCheckpoint::open_component(&audio_dir, "diffusion_pytorch_model")?
                .minimax_h3_audio_vae_manifest(&audio)?;
        let audio_decoder = AudioDecoder::new(&device, &audio, &audio_dir, &audio_manifest)?;
        let decode_geometry = VideoDecodeGeometry::new(&vae, &geometry)?;
        let rgb_staging_bytes = decode_geometry
            .rgb_bytes
            .min(crate::backend::cuda::weights::UPLOAD_CHUNK_BYTES);
        let slots = decode_geometry.tiles.count();
        let cap = VideoDecoder::work_batch_cap(&device, &vae, decode_geometry.tile, slots)?;
        let one_tile = TileWork::charged_bytes(
            &vae,
            decode_geometry.tile,
            1,
            slots,
            page,
            decode_geometry.rgb_bytes,
        )?;
        let audio_input = 4 * layout.audio_indices.len() * spec.audio_in_channels;
        let posterior = 4 * vae.latent_channels * geometry.latent_height * geometry.latent_width;
        let anchor = encode.as_ref().map_or(0, |(stage, _)| {
            let tiles = &stage.tiles;
            let grid = 4 * (2 * (tiles.rows.0.len() + tiles.cols.0.len()) - 2);
            3 * geometry.height * geometry.width
                + 8 * vae.latent_channels
                    * tiles.count()
                    * (tiles.height / vae.spatial_ratio)
                    * (tiles.width / vae.spatial_ratio)
                + 5 * posterior
                + grid
        });
        let noise = audio_input + if encode.is_some() { posterior } else { 0 };
        let samples = 4 * layout.video_indices.len() * spec.video_row_width()
            + 2 * layout.text_indices.len() * spec.text_dim
            + 8 * layout.token_tags.len();
        let transient = stored_allocation_bytes(
            (noise + samples.max(vision_bytes).max(anchor))
                .max(decode_geometry.transient)
                .max(audio_input + AudioDecoder::work_bytes(&audio, (2, geometry.audio_latents))?),
            page,
        )?;
        let (text_layers, text_largest) =
            Conditioner::layer_sizes(&text_manifest, TEXT_ENCODER_LAYER)?;
        let text_fixed =
            Conditioner::resident_fixed_bytes(&device, &text_spec, (TEXT_ENCODER_LAYER, count))?;
        let ([decoder_layers, decoder_source], decoder_fixed, decoder_largest) =
            decoder_layer_sizes(&vae_manifest, vae.layers);
        let decoder_staging = *decoder_source
            .iter()
            .max()
            .context("video decoder lacks blocks")?;
        let fixed = stored_allocation_bytes(text_fixed, page)?
            + stored_allocation_bytes(decoder_fixed, page)?;
        let (disk_probe_bytes, disk, ring_workers) =
            seed_dense_supply_workers(&text_dir, &text_manifest, &options.runtime_dir)?;
        let pcie = crate::backend::cuda::calibrate::pinned_h2d_rate(&device)?;
        let visit = DecoderVisitCost::measure(
            &device,
            &normal,
            vae.layers,
            decoder_weight_bytes(&vae_manifest),
        )?;
        let text_blas = blas("H3 conditioner")?;
        let mut sample_blas = blas("MiniMax-H3 generate")?;
        let mut later = None;
        let transform = prepare_transformer(
            (&device, &mut sample_blas),
            (&transformer, lora, gguf),
            (&spec, &layout),
            (evaluations, shifts),
            Some(&mut |sizing, charged, owned, bind_ffn| {
                let (dit_layers, dit_source): (Vec<_>, Vec<_>) = charged
                    .iter()
                    .copied()
                    .zip(sizing.source_layers.iter().copied())
                    .filter(|&(_, source)| source != 0)
                    .unzip();
                let fixed = fixed + stored_allocation_bytes(owned, page)?;
                let stages = [
                    DenseStage {
                        device: &text_layers,
                        source: &text_layers,
                        visits: 1,
                        load_scratch: text_largest,
                        device_staging: 0,
                    },
                    DenseStage {
                        device: &dit_layers,
                        source: &dit_source,
                        visits: evaluations,
                        load_scratch: sizing.largest,
                        device_staging: 0,
                    },
                    DenseStage {
                        device: &decoder_layers,
                        source: &decoder_source,
                        visits: decode_geometry.chunks,
                        load_scratch: decoder_largest,
                        device_staging: decoder_staging,
                    },
                ];
                let floors = stages.iter().try_fold(fixed + transient, |sum, stage| {
                    Ok::<_, anyhow::Error>(sum + stage.minimum_device_bytes(page)?)
                })?;
                bind_ffn(floors + stored_allocation_bytes(one_tile, page)?).with_context(|| {
                    format!(
                        "resident video FFN with {one_tile} bytes reserved for one decoder tile"
                    )
                })?;
                device.ctx.synchronize()?;
                let (free, total) = device.ctx.mem_get_info()?;
                let remaining = DensePlacement::device_budget(free, total, floors, 0, 0)?;
                let budget = remaining - remaining % page;
                let stage_and_work =
                    DensePlacement::device_budget(free, total, fixed, 0, transient)?;
                let available = usize::try_from(crate::backend::cuda::calibrate::host_memory(
                    Path::new("/"),
                )?)?;
                let host = available
                    .checked_sub(4 << 30)
                    .and_then(|bytes| bytes.checked_sub(rgb_staging_bytes))
                    .context("resident video host reserve")?;
                let (batch, device_budget, priced, alternatives) = visit.price_groups(
                    (&vae, &decode_geometry, cap),
                    stages,
                    (stage_and_work, budget, host, page),
                    (pcie, disk),
                    ring_workers,
                )?;
                let decoder_visits = decode_geometry.chunks * slots.div_ceil(batch);
                let bytes = TileWork::charged_bytes(
                    &vae,
                    decode_geometry.tile,
                    batch,
                    slots,
                    1,
                    decode_geometry.rgb_bytes,
                )?;
                let work = TileWork::with_output(
                    &device.stream,
                    &vae,
                    decode_geometry.tile,
                    (batch, slots),
                    (3, decode_geometry.rgb_bytes),
                )?;
                device.ctx.synchronize()?;
                let (free, total) = device.ctx.mem_get_info()?;
                let live_budget = DensePlacement::device_budget(free, total, fixed, 0, transient)?;
                ensure!(
                    priced.device_bytes.iter().sum::<usize>() <= live_budget,
                    "resident video stage placement exceeds the live budget {live_budget} after allocating {batch} tiles"
                );
                let PricedDense {
                    plans: [text, dit, decoder],
                    device_bytes,
                    decoder_resident_seconds,
                    decoder_streamed_seconds,
                } = priced;
                let demand = |plan: &DensePlacement, visits: usize| {
                    serde_json::json!({
                        "source_bytes": visits * (plan.host_bytes + plan.disk_bytes),
                        "disk_bytes": visits * plan.disk_bytes,
                    })
                };
                eprintln!(
                    "{}",
                    serde_json::json!({"h3_video_resident_plan": {
                        "device_budget": device_budget, "host_budget": host, "transient_bytes": transient,
                        "shared_ffn_rgb_bytes": decode_geometry.rgb_bytes, "rgb_d2h_staging_bytes": rgb_staging_bytes,
                        "decoder_group_prices": alternatives,
                        "decoder_launch_seconds": visit.launch_seconds, "decoder_handoff_seconds": visit.handoff_seconds,
                        "device_memory_bytes_per_second": device.info.memory_bytes_per_second,
                        "pcie_bytes_per_second": pcie, "disk_bytes_per_second": disk, "disk_probe_bytes": disk_probe_bytes,
                        "device_allocation_granularity": page, "per_stage_device_bytes": device_bytes,
                        "conditioner": &text, "dit": &dit, "decoder": &decoder,
                        "request_demand": {
                            "conditioner": demand(&text, 1), "dit": demand(&dit, evaluations),
                            "decoder": demand(&decoder, decoder_visits),
                        },
                        "decoder_tile_batch": batch, "decoder_tile_cap": cap, "decoder_work_budget": budget, "decoder_work_bytes": bytes,
                        "decoder_resident_seconds": decoder_resident_seconds, "decoder_streamed_seconds": decoder_streamed_seconds,
                        "dit_visits": evaluations, "decoder_visits": decoder_visits,
                    }})
                );
                later = Some((text, decoder, work));
                Ok(dit)
            }),
        )?;
        let (text_plan, decode_plan, work) =
            later.context("resident video placement was not bound")?;
        let conditioner = Conditioner::with_plan(
            &device,
            &text_spec,
            (TEXT_ENCODER_LAYER, count, transient),
            Source::new(
                &text_dir,
                &text_manifest,
                device.info.arch.fp16_stored_half(),
            ),
            (Placement::Derived, Some(text_plan)),
            request_count,
        )?;
        conditioner.prepare(&text_blas, count)?;
        let video = VideoDecoder::with_plan(&device, (&vae, &vae_dir, &vae_manifest), decode_plan)?;
        video.bind_groups(&mut sample_blas, &work, cap)?;
        let kernels = BlockKernels::new(&device)?;
        let rgb_staging = crate::backend::cuda::copy::pinned_zeros(&device.ctx, rgb_staging_bytes)?;
        let untimed = Some(crate::backend::gpu::EVENT_DISABLE_TIMING);
        let rgb_ready = device.ctx.new_event(untimed)?;
        let rgb_done = [
            device.ctx.new_event(untimed)?,
            device.ctx.new_event(untimed)?,
        ];
        device.finish_upload()?;
        let (free, total) = device.ctx.mem_get_info()?;
        ensure!(
            free >= total / 16 + transient,
            "resident video needs {} free bytes for transient work and reserve, found {free}",
            total / 16 + transient
        );
        Ok(Self {
            device,
            shape,
            layout,
            spec,
            vae,
            audio,
            normal,
            transform,
            sample_blas,
            kernels,
            encode,
            text: ConditionerStage {
                dir: text_dir,
                spec: text_spec,
                manifest: text_manifest,
                vision,
                conditioner,
                blas: text_blas,
                image_token,
            },
            decode: VideoDecodeStage {
                geometry: decode_geometry,
                video,
                audio: audio_decoder,
                work,
                rgb_staging,
                rgb_ready,
                rgb_done,
            },
        })
    }

    pub fn render(&mut self, request: &H3Request) -> Result<H3Output> {
        self.render_request(
            request,
            #[cfg(test)]
            None,
        )
    }

    #[cfg(test)]
    pub(super) fn render_with_reference(
        &mut self,
        request: &H3Request,
        reference: &Generated,
    ) -> Result<H3Output> {
        self.render_request(request, Some(reference))
    }

    fn render_request(
        &mut self,
        request: &H3Request,
        #[cfg(test)] reference: Option<&Generated>,
    ) -> Result<H3Output> {
        self.shape.check(&self.spec, request)?;
        let geometry = &self.shape.geometry;
        let (video, audio) = {
            let s = &self.device.stream;
            let draws = NoiseDraws::Seed(crate::minimax_h3::noise_draws(request.seed));
            let noise = video_noise(
                (&self.device, &self.normal),
                (&self.spec, &self.vae),
                (geometry, &self.layout),
                request,
                &draws,
            )?;
            let prompt = video_conditioner_prompt(request, self.text.image_token);
            let text = self.text.run(&self.device, &prompt)?;
            let anchors = self
                .encode
                .as_mut()
                .map(|(stage, blas)| {
                    stage.run(
                        &self.device,
                        blas,
                        request,
                        noise
                            .posterior_eps
                            .as_ref()
                            .context("keyframes require posterior noise")?,
                    )
                })
                .transpose()?
                .unwrap_or_default();
            #[cfg(test)]
            if let Some(reference) = reference {
                let text_exact = text
                    .iter()
                    .map(|x| x.to_bits())
                    .eq(reference.conditioner.iter().map(|x| x.to_bits()));
                let anchors_exact = anchors.len() == reference.anchors.len()
                    && anchors
                        .iter()
                        .zip(&reference.anchors)
                        .all(|(a, b)| same_f32_bits(a, b));
                println!(
                    "H3_VIDEO_RESIDENT_INPUT conditioner={text_exact} anchors={anchors_exact}"
                );
            }
            let rows = video_rows(
                (&self.spec, &self.vae),
                (geometry, &self.layout),
                request,
                &noise,
                (&anchors, &text),
            )?;
            let video = s.clone_htod(&rows)?;
            let text = s.clone_htod(&text)?;
            let transform = &mut self.transform;
            let c = BlockContext {
                s,
                blas: &self.sample_blas,
                kernels: &self.kernels,
                spec: &self.spec,
                lora_scale: transform.scale,
            };
            transform.dit.sample(
                &c,
                (&transform.table, &transform.adaln),
                (&transform.video_grid, &transform.audio_grid),
                (&video, &noise.audio),
                &text,
            )?;
            (s.clone_dtoh(&video)?, s.clone_dtoh(&noise.audio)?)
        };
        ensure!(
            video.iter().chain(&audio).all(|v| v.is_finite()),
            "non-finite final latents"
        );
        let (video, audio) = decoder_latents(
            (&self.spec, &self.vae, &self.audio),
            (geometry, &self.layout),
            (&video, &audio),
        )?;
        #[cfg(test)]
        if let Some(reference) = reference {
            println!(
                "H3_VIDEO_RESIDENT_LATENTS video={} audio={}",
                same_f32_bits(&video, &reference.video_latents.1),
                same_f32_bits(&audio, &reference.audio_latents.1)
            );
        }
        self.decode.render(
            (&self.device, &self.sample_blas),
            (geometry, &self.vae, &self.audio),
            (&video, &audio),
        )
    }
}

#[cfg(all(test, feature = "cuda"))]
pub(super) fn same_f32_bits(a: &[f32], b: &[f32]) -> bool {
    a.iter()
        .map(|x| x.to_bits())
        .eq(b.iter().map(|x| x.to_bits()))
}

/// Image requests sharing one packed geometry and one set of loaded weights.
pub struct ImageSession {
    model: PathBuf,
    image: ImageSpec,
    spec: H3Spec,
    shape: ImageShape,
    layout: ImageLayout,
    d: Device,
    normal: crate::backend::gpu::CudaFunction,
    text: ConditionerStage,
    dit: ImageDit,
    table: AdalnTable,
    adaln: AdalnPlan,
    schedule: ImageSchedule,
    kernels: BlockKernels,
    sample_blas: Blas,
    decode: ImageDecodeStage,
    decode_blas: Blas,
    encode: Option<(ImageEncodeStage, Blas)>,
}

impl ImageSession {
    pub fn open(
        model: &Path,
        options: &ImageRunOptions,
        requests: impl IntoIterator<Item = Result<ImageRequest>>,
    ) -> Result<Self> {
        use crate::backend::cuda::ops::CommonOpsRef;
        let image = ImageSpec::read(model)?;
        let dir = model.join("text_encoder");
        let text_spec = crate::model_spec::ModelSpec::read(&dir)?;
        let image_token = text_spec
            .vision
            .as_ref()
            .context("text encoder without vision")?
            .image_token;
        let mut requests = requests.into_iter();
        let (shape, count, media) = {
            let first = requests
                .next()
                .context("resident image workload is empty")??;
            let ImageInput { shape, prompt } = ImageInput::read(model, &image, &first)?;
            let count = prompt.tokens.len();
            let media = if shape.reference.is_some() {
                LayerNormTower::request_bytes(
                    &text_spec,
                    &conditioner_vision_input(&prompt),
                    count,
                )?
            } else {
                0
            };
            (shape, count, media)
        };
        for request in requests {
            let input = ImageInput::read(model, &image, &request?)?;
            shape.check(&input.shape)?;
        }
        let spec = image_block_spec(&image);
        let schedule = ImageSchedule::published()?;
        let layout = shape.layout(&image)?;
        let plan = crate::minimax_h3::image_row_plan(&layout, &schedule)?;
        let adaln = AdalnPlan::from_tags(&layout.token_tags, &plan)?;
        let d = Device::new(options.device)?;
        let device_page = stored_allocation_granularity(&d)?;
        let ops = Ops::new(&d.ctx)?;
        let blas = |name: &str| {
            super::open_blas(
                &d,
                &ops,
                &options.runtime_dir,
                Rc::new(Closure::new(true, name.into())),
            )
        };
        let normal = cubin::module(&d.ctx, "sample")?.load_function("philox_normal_f32")?;
        let text_manifest = MappedCheckpoint::open_component(&dir, "model")?
            .minimax_h3_conditioner_manifest(&text_spec, TEXT_ENCODER_LAYER)?;
        let vision = if shape.reference.is_some() {
            let mut catalog =
                WeightCatalog::load_with_policy(&dir, &text_spec, &d, LoadPolicy::VisionOnly)?;
            let tower = LayerNormTower::from_catalog(
                &text_spec,
                &mut catalog,
                &d,
                CommonOpsRef::new(&d.ctx)?,
            )?;
            ensure!(
                catalog.tensors.is_empty(),
                "vision binding left unloaded owners"
            );
            d.finish_upload()?;
            Some(tower)
        } else {
            None
        };
        let (conditioner_layers, conditioner_largest) =
            Conditioner::layer_sizes(&text_manifest, TEXT_ENCODER_LAYER)?;
        let conditioner_fixed =
            Conditioner::resident_fixed_bytes(&d, &text_spec, (TEXT_ENCODER_LAYER, count))?;
        let sample_input_bytes = layout.image_indices.len() * image.row_width() * 4
            + layout.text_indices.len() * spec.text_dim * 2
            + layout.rows() * 2 * size_of::<u32>();
        let latents_bytes = shape.latent[0] * shape.latent[1] * image.in_channels * 4;
        let temporary = media.max(sample_input_bytes).max(latents_bytes);
        let (vae_dir, vae_spec, vae_manifest) = image_vae_manifest(model)?;
        let vae = vae_spec.single_frame();
        let ([decoder_layers, decoder_source], decoder_fixed, decoder_largest) =
            decoder_layer_sizes(&vae_manifest, vae.layers);
        let decoder_staging = *decoder_source
            .iter()
            .max()
            .context("image VAE has no decoder layers")?;
        let (disk_probe_bytes, disk, ring_workers) =
            seed_dense_supply_workers(&dir, &text_manifest, &options.runtime_dir)?;
        let pcie = crate::backend::cuda::calibrate::pinned_h2d_rate(&d)?;
        let mut decode_blas = blas("image VAE")?;
        let text_blas = blas("MiniMax-H3-Image conditioner")?;
        let device_fixed = stored_allocation_bytes(conditioner_fixed, device_page)?
            .checked_add(stored_allocation_bytes(decoder_fixed, device_page)?)
            .context("resident image fixed weights overflow")?;
        let transformer = model.join("transformer_diffusers");
        let manifest = MappedCheckpoint::open_component(&transformer, "diffusion_pytorch_model")?
            .minimax_h3_image_manifest(&image)?;
        let mut sample_blas = blas("MiniMax-H3-Image")?;
        let table =
            AdalnTable::build(&d, &mut sample_blas, &transformer, &manifest, &spec, &adaln)?;
        let mut later = None;
        let dit = ImageDit::resident(
            &d,
            &mut sample_blas,
            (&spec, &image),
            &layout,
            (&transformer, &manifest),
            &mut |sizing, owned, bind_ffn| {
                use crate::prefill::{DensePlacement, DenseStage, PricedDense, price_dense_stages};
                let fixed = device_fixed
                    .checked_add(stored_allocation_bytes(owned, device_page)?)
                    .context("resident image fixed weights overflow")?;
                let mut stages = [
                    DenseStage {
                        device: &conditioner_layers,
                        source: &conditioner_layers,
                        visits: 1,
                        load_scratch: conditioner_largest,
                        device_staging: 0,
                    },
                    DenseStage {
                        device: &sizing.layers,
                        source: &sizing.source_layers,
                        visits: schedule.timesteps.len(),
                        load_scratch: sizing.largest,
                        device_staging: 0,
                    },
                    DenseStage {
                        device: &decoder_layers,
                        source: &decoder_source,
                        visits: 1,
                        load_scratch: decoder_largest,
                        device_staging: decoder_staging,
                    },
                ];
                let encoder = shape
                    .reference
                    .map(|extent| -> Result<_> {
                        let encoder_blas = blas("image VAE encoder")?;
                        let setup =
                            ImageEncodeSetup::new(&d, (&vae_dir, &vae, &vae_manifest), extent)?;
                        Ok((setup, encoder_blas))
                    })
                    .transpose()?;
                let (encoder_work, encoder_transient) =
                    encoder.as_ref().map_or((0, 0), |(setup, _)| {
                        (setup.minimum_work(), setup.transient_bytes)
                    });
                let temporary =
                    stored_allocation_bytes(temporary.max(encoder_transient), device_page)?;
                let fixed_and_transient = fixed
                    .checked_add(temporary)
                    .context("resident image fixed and transient reserve overflow")?;
                let pending = stages
                    .iter()
                    .try_fold(fixed_and_transient, |bytes, stage| {
                        bytes
                            .checked_add(stage.minimum_device_bytes(device_page)?)
                            .context("resident image tier floor overflow")
                    })?;
                let decoder_floor = stored_allocation_bytes(
                    ImageDecodeStage::minimum_work(&vae, shape.canvas)?,
                    device_page,
                )?;
                let decoder_pending = pending
                    .checked_add(decoder_floor)
                    .context("resident image decoder floor overflow")?;
                let ffn_pending = decoder_pending
                    .checked_add(stored_allocation_bytes(encoder_work, device_page)?)
                    .context("resident image encoder floor overflow")?;
                let ffn = bind_ffn(ffn_pending)?;
                let encode = encoder
                    .map(|(setup, blas)| -> Result<_> {
                        Ok((setup.bind(&d, decoder_pending)?, blas))
                    })
                    .transpose()?;
                let decoder_work = ImageDecodeStage::prepare_work(&d, &vae, shape.canvas, pending)?;
                let decoder_visits = decoder_work.visits();
                stages[2].visits = decoder_visits;
                d.ctx.synchronize()?;
                let (free, total) = d.ctx.mem_get_info()?;
                let host_available = usize::try_from(
                    crate::backend::cuda::calibrate::host_memory(Path::new("/"))?,
                )?;
                let host = host_available.saturating_sub(4 << 30);
                let device = DensePlacement::device_budget(free, total, fixed, 0, temporary)?;
                let PricedDense {
                    plans: [text, dit, decode],
                    device_bytes,
                    decoder_resident_seconds,
                    decoder_streamed_seconds,
                } = price_dense_stages(
                    stages,
                    (device, host, device_page),
                    (pcie, disk),
                    ring_workers,
                )?;
                let device_after = device_fixed + temporary + device_bytes[0] + device_bytes[2];
                eprintln!(
                    "{}",
                    serde_json::json!({"h3_image_resident_plan": {
                        "canvas": shape.canvas, "packed_rows": layout.rows(),
                        "device_free_before_placement": free, "device_budget": device, "host_available_bytes": host_available, "host_budget": host,
                        "transient_device_bytes": temporary, "decoder_fixed_gpu_bytes": decoder_fixed,
                        "pcie_bytes_per_second": pcie, "disk_bytes_per_second": disk, "disk_probe_bytes": disk_probe_bytes, "disk_ring_workers": ring_workers,
                        "device_allocation_granularity": device_page,
                        "conditioner_charged_device_bytes": device_bytes[0],
                        "dit_charged_device_bytes": device_bytes[1],
                        "decoder_charged_device_bytes": device_bytes[2],
                        "decoder_resident_seconds": decoder_resident_seconds, "decoder_streamed_seconds": decoder_streamed_seconds,
                        "conditioner": &text, "dit": &dit, "decoder": &decode,
                        "dit_visits": schedule.timesteps.len(), "decoder_visits": decoder_visits,
                    }})
                );
                later = Some((text, decode, decoder_work, encode, temporary));
                Ok((dit, device_after, ffn))
            },
        )?;
        let (text_plan, decode_plan, decoder_work, encode, temporary) =
            later.context("resident image placement was not bound")?;
        let conditioner = Conditioner::with_plan(
            &d,
            &text_spec,
            (TEXT_ENCODER_LAYER, count, temporary),
            Source::new(&dir, &text_manifest, d.info.arch.fp16_stored_half()),
            (Placement::Derived, Some(text_plan)),
            count,
        )?;
        conditioner.prepare(&text_blas, count)?;
        let decode = ImageDecodeStage::bind(
            &d,
            &mut decode_blas,
            (&vae_dir, &vae, &vae_manifest),
            decoder_work,
            decode_plan,
        )?;
        let kernels = BlockKernels::new(&d)?;
        d.finish_upload()?;
        let (free, total) = d.ctx.mem_get_info()?;
        let required = (total / 16)
            .checked_add(temporary)
            .context("resident reserve overflow")?;
        ensure!(
            free >= required,
            "resident image setup needs {required} free bytes for transient work and reserve, found {free}"
        );
        let text = ConditionerStage {
            dir,
            spec: text_spec,
            manifest: text_manifest,
            vision,
            conditioner,
            blas: text_blas,
            image_token,
        };
        Ok(Self {
            model: model.to_path_buf(),
            image,
            spec,
            shape,
            layout,
            d,
            normal,
            text,
            dit,
            table,
            adaln,
            schedule,
            kernels,
            sample_blas,
            decode,
            decode_blas,
            encode,
        })
    }

    pub fn render(&mut self, request: &ImageRequest) -> Result<ImageLatents> {
        let input = ImageInput::read(&self.model, &self.image, request)?;
        self.shape.check(&input.shape)?;
        image_render(
            request,
            &self.image.clone(),
            None,
            ImageStages::Resident(self, &input),
        )
        .map(|(image, _)| image)
    }
}

enum ImageStages<'a> {
    Single {
        model: &'a Path,
        options: &'a ImageRunOptions,
        prompt: String,
    },
    Resident(&'a mut ImageSession, &'a ImageInput),
}

impl ImageStages<'_> {
    fn noise(&self, seed: u64, count: usize) -> Result<Vec<f32>> {
        match self {
            Self::Single { options, .. } => image_noise(options, seed, count),
            Self::Resident(session, _) => {
                image_noise_draws(&session.d, &session.normal, seed, count)
            }
        }
    }

    fn encode(
        &mut self,
        extent: [usize; 2],
        rgb: &[f32],
        epsilon: &[f32],
    ) -> Result<(Vec<f32>, [usize; 2])> {
        match self {
            Self::Single { model, options, .. } => super::image_encode(
                model,
                (options.device, &options.runtime_dir),
                extent,
                rgb,
                epsilon,
            ),
            Self::Resident(session, _) => {
                let (stage, blas) = session
                    .encode
                    .as_mut()
                    .context("resident image encoder not admitted")?;
                stage.run(&session.d, blas, extent, rgb, epsilon)
            }
        }
    }

    fn condition(
        &mut self,
        spec: &ImageSpec,
        request: &ImageRequest,
        reference_extents: &[(usize, usize)],
        [lh, lw]: [usize; 2],
    ) -> Result<(ImageLayout, Vec<bf16>)> {
        let (layout, text) = match self {
            Self::Single {
                model,
                options,
                prompt,
            } => {
                let (ids, tags, prompt) =
                    image_presentation(model, spec, prompt, &request.references)?;
                let layout =
                    crate::minimax_h3::image_layout(spec, &tags, reference_extents, lh, lw)?;
                ensure!(
                    ids.len() == tags.len() && ids.len() == layout.text_indices.len(),
                    "image presentation rows"
                );
                let text = image_condition(model, options, &prompt)?;
                (layout, text)
            }
            Self::Resident(session, input) => {
                let text = session.text.run(&session.d, &input.prompt)?;
                (session.layout.clone(), text)
            }
        };
        ensure!(
            text.len() == layout.text_indices.len() * spec.text_dim,
            "conditioner row width"
        );
        Ok((layout, text))
    }

    fn sample(&mut self, layout: &ImageLayout, rows: &[f32], text: &[bf16]) -> Result<Vec<f32>> {
        match self {
            Self::Single { model, options, .. } => super::image_sample(
                model,
                layout,
                (options.device, &options.runtime_dir),
                rows,
                text,
            ),
            Self::Resident(session, _) => {
                let context = BlockContext {
                    s: &session.d.stream,
                    blas: &session.sample_blas,
                    kernels: &session.kernels,
                    spec: &session.spec,
                    lora_scale: 1.0,
                };
                session.dit.sample_host(
                    &context,
                    (&session.table, &session.adaln),
                    &session.schedule,
                    (rows, text),
                    (None, None),
                )
            }
        }
    }

    fn decode(&mut self, extent: [usize; 2], values: &[f32]) -> Result<Vec<f32>> {
        match self {
            Self::Single { model, options, .. } => super::image_decode(
                model,
                (options.device, &options.runtime_dir),
                extent,
                values,
            ),
            Self::Resident(session, _) => {
                session.decode.run(&session.d, &session.decode_blas, values)
            }
        }
    }
}

/// Normalised generated latent and the finished RGB8. One Philox draw on `STREAM_IMAGE_NOISE` is each
/// reference's posterior epsilon, the generated rows, then each reference's mix. `image_encode` returns before
/// the conditioner loads, and `image_sample` returns before `image_decode` loads the decoder. Reference rows are
/// `KEYFRAME_NOISE_AUG * latent + (1 - KEYFRAME_NOISE_AUG) * mix` on the cropped encoder latent. `rgb` is clamp,
/// truncation toward zero, then Lanczos-3 of the decoder canvas.
pub fn image_latents(
    model: &Path,
    options: &crate::minimax_h3::ImageRunOptions,
    request: &crate::minimax_h3::ImageRequest,
) -> Result<crate::minimax_h3::ImageLatents> {
    Ok(image_pass(model, options, request, None)?.0)
}

/// Production stages of [`image_latents`]. `supplied` is that Philox buffer when present. The second value is the conditioner.
pub(crate) fn image_pass(
    model: &Path,
    options: &crate::minimax_h3::ImageRunOptions,
    request: &crate::minimax_h3::ImageRequest,
    supplied: Option<&[f32]>,
) -> Result<(crate::minimax_h3::ImageLatents, Vec<bf16>)> {
    ensure!(!request.prompt.is_empty(), "image prompt is empty");
    let prompt = match &request.expander {
        Some(dir) => expand_prompt(dir, &request.prompt, options)?,
        None => request.prompt.clone(),
    };
    let spec = ImageSpec::read(model)?;
    image_render(
        request,
        &spec,
        supplied,
        ImageStages::Single {
            model,
            options,
            prompt,
        },
    )
}

fn image_render(
    request: &ImageRequest,
    spec: &ImageSpec,
    supplied: Option<&[f32]>,
    mut stages: ImageStages<'_>,
) -> Result<(ImageLatents, Vec<bf16>)> {
    let (canvas_h, canvas_w) = crate::minimax_h3::image_canvas(
        spec,
        request.height,
        request.width,
        request.min_generation_area,
    )?;
    let (latent_h, latent_w) = crate::minimax_h3::image_latent_extent(spec, canvas_h, canvas_w)?;
    let width = spec.row_width();
    let mut prepared = Vec::with_capacity(request.references.len());
    let mut noise_count = 0usize;
    for reference in &request.references {
        let (rgb, extent, posterior) = vae_rgb(spec, &reference.image)?;
        let (lat_h, lat_w) =
            crate::minimax_h3::image_reference_latent_extent(spec, extent[0], extent[1])?;
        let cropped = spec
            .in_channels
            .checked_mul(lat_h)
            .and_then(|n| n.checked_mul(lat_w))
            .context("reference latent extent overflows")?;
        noise_count = noise_count
            .checked_add(posterior)
            .context("image noise count overflows")?;
        prepared.push((rgb, extent, posterior, lat_h, lat_w, cropped));
    }
    let generated_rows = (latent_h / spec.patch[1]) * (latent_w / spec.patch[2]);
    let generated_at = noise_count;
    noise_count = noise_count
        .checked_add(generated_rows * width)
        .context("image noise count overflows")?;
    let mix_at = noise_count;
    for (_, _, _, _, _, cropped) in &prepared {
        noise_count = noise_count
            .checked_add(*cropped)
            .context("image noise count overflows")?;
    }
    let noise = match supplied {
        Some(values) => {
            ensure!(
                values.len() == noise_count && values.iter().all(|value| value.is_finite()),
                "image noise extent"
            );
            values.to_vec()
        }
        None => stages.noise(request.seed, noise_count)?,
    };
    let mut cursor = 0usize;
    let mut reference_rows = Vec::new();
    let mut reference_extents = Vec::with_capacity(prepared.len());
    let mut encoded = Vec::with_capacity(prepared.len());
    for (rgb, extent, posterior, lat_h, lat_w, cropped) in &prepared {
        let epsilon = &noise[cursor..cursor + posterior];
        cursor += posterior;
        let (latent, shape) = stages.encode(*extent, rgb, epsilon)?;
        ensure!(
            shape == [*lat_h, *lat_w] && latent.len() == *cropped,
            "reference latent extent"
        );
        encoded.push(latent);
        reference_extents.push((*lat_h, *lat_w));
    }
    ensure!(cursor == generated_at, "posterior draw extent");
    let mix_scale = crate::minimax_h3::KEYFRAME_NOISE_AUG as f32;
    let mut mix_cursor = mix_at;
    for (latent, (_, _, _, lat_h, lat_w, cropped)) in encoded.iter().zip(&prepared) {
        let mix = &noise[mix_cursor..mix_cursor + cropped];
        mix_cursor += cropped;
        let mixed: Vec<f32> = latent
            .iter()
            .zip(mix)
            .map(|(&value, &draw)| mix_scale * value + (1.0 - mix_scale) * draw)
            .collect();
        ensure!(
            mixed.iter().all(|value| value.is_finite()),
            "non-finite reference row"
        );
        reference_rows.extend(crate::minimax_h3::patchify_latents(
            &mixed,
            (spec.in_channels, 1, *lat_h, *lat_w),
            spec.patch,
        ));
    }
    ensure!(mix_cursor == noise.len(), "mix draw extent");
    let (layout, text) =
        stages.condition(spec, request, &reference_extents, [latent_h, latent_w])?;
    let generated = &noise[generated_at..mix_at];
    ensure!(
        reference_rows.len() == layout.reference_rows * width,
        "reference row count"
    );
    let mut rows = reference_rows.clone();
    rows.extend_from_slice(generated);
    let sampled = stages.sample(&layout, &rows, &text)?;
    let prefix = layout.reference_rows * width;
    ensure!(
        sampled.len() == rows.len() && sampled[..prefix] == reference_rows[..],
        "sample moved a reference row"
    );
    let values = crate::minimax_h3::unpatchify_latents(
        &sampled[prefix..],
        (spec.in_channels, 1, latent_h, latent_w),
        spec.patch,
    );
    ensure!(
        values.len() == spec.in_channels * latent_h * latent_w,
        "unpatchified latent extent"
    );
    let canvas = stages.decode([latent_h, latent_w], &values)?;
    let rgb = crate::minimax_h3::image_rgb8(
        &canvas,
        [canvas_h, canvas_w],
        [request.height, request.width],
    )?;
    Ok((
        crate::minimax_h3::ImageLatents {
            channels: spec.in_channels,
            height: latent_h,
            width: latent_w,
            values,
            canvas_height: canvas_h,
            canvas_width: canvas_w,
            rgb,
        },
        text,
    ))
}

#[cfg(test)]
#[derive(Clone, serde::Deserialize)]
struct CompositionCell {
    shape: Vec<usize>,
    tau_abs: f64,
    tau_rmse: f64,
    tau_relative_rmse: f64,
    gating: bool,
    gating_metrics: Vec<String>,
    reference_arms: Vec<String>,
}

#[cfg(test)]
#[derive(serde::Deserialize)]
struct ConditionerReading {
    cpu_row: Vec<f32>,
    max_abs: f64,
    rmse: f64,
    coordinate_abs: f64,
}

#[cfg(test)]
#[derive(serde::Deserialize)]
struct CompositionFloors {
    freeze_revision: String,
    formula: String,
    native_candidate_run: bool,
    native_candidate_used_in_floor: bool,
    r6_qualified: bool,
    r6_control_checks: usize,
    conditioner_partition: serde_json::Value,
    cells: std::collections::BTreeMap<String, CompositionCell>,
    conditioner_reading: ConditionerReading,
    stage_controls: std::collections::BTreeMap<String, bool>,
}

#[cfg(test)]
pub(super) struct CompositionGate {
    root: std::path::PathBuf,
    floors: CompositionFloors,
    gating: usize,
    measurements: usize,
    controls: usize,
    control_failed: bool,
    defects: Vec<String>,
    seen: std::collections::BTreeSet<String>,
    capture: Option<(std::path::PathBuf, String)>,
}

#[cfg(test)]
pub(super) fn composition_stamp() -> Result<String> {
    let output = std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()?;
    ensure!(output.status.success(), "Runner FAIL UTC clock command");
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

#[cfg(test)]
fn composition_values(root: &Path, relative: &str, shape: &[usize]) -> Result<Vec<f64>> {
    use safetensors::{Dtype, SafeTensors};
    let path = root.join(relative).with_extension("safetensors");
    let bytes = std::fs::read(&path).with_context(|| format!("open {}", path.display()))?;
    let file = SafeTensors::deserialize(&bytes)?;
    let value = file.tensor("value")?;
    ensure!(
        value.shape() == shape,
        "Runner FAIL {relative} shape {:?} != {shape:?}",
        value.shape()
    );
    let values: Vec<f64> = match value.dtype() {
        Dtype::BF16 => value
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f64::from(bf16::from_bits(u16::from_le_bytes(*b)).to_f32()))
            .collect(),
        Dtype::F32 => value
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f64::from(f32::from_le_bytes(*b)))
            .collect(),
        Dtype::F64 => value
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| f64::from_le_bytes(*b))
            .collect(),
        Dtype::I64 => value
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b) as f64)
            .collect(),
        Dtype::U8 => value.data().iter().map(|&v| f64::from(v)).collect(),
        other => anyhow::bail!("Runner FAIL {relative} dtype {other:?}"),
    };
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "Runner FAIL {relative} nonfinite fixture"
    );
    Ok(values)
}

#[cfg(test)]
impl CompositionGate {
    fn read(root: &Path) -> Result<Self> {
        let floors: CompositionFloors =
            serde_json::from_slice(&std::fs::read(root.join("FROZEN_FLOORS.json"))?)?;
        ensure!(
            floors.freeze_revision == "r6-conv"
                && floors.formula == "max(2 * worst_valid_reference_pair, 1e-6)"
                && !floors.native_candidate_run
                && !floors.native_candidate_used_in_floor
                && floors.r6_qualified
                && floors.r6_control_checks == 41,
            "Runner FAIL composition floor declaration"
        );
        for stage in [
            "conditioner",
            "anchor",
            "transformer_video",
            "transformer_audio",
            "video_decoder",
            "audio_decoder",
        ] {
            ensure!(
                floors.stage_controls.get(stage) == Some(&true),
                "Runner FAIL {stage} has no detected reference control"
            );
        }
        for (key, cell) in &floors.cells {
            ensure!(
                [cell.tau_abs, cell.tau_rmse, cell.tau_relative_rmse]
                    .into_iter()
                    .all(|v| v.is_finite() && v >= 1e-6),
                "Runner FAIL {key} invalid floor"
            );
            ensure!(
                (cell.gating_metrics == ["max_abs", "rmse"]
                    || cell.gating_metrics == ["max_abs", "rmse", "relative_rmse"])
                    && !cell.reference_arms.is_empty(),
                "Runner FAIL {key} scoring declaration"
            );
        }
        ensure!(
            !floors
                .cells
                .keys()
                .any(|key| key.starts_with("conditioner-effect/")),
            "Runner FAIL functional conditioner cells are not composition gates"
        );
        for case in ["text", "first", "last", "both"] {
            for part in ["bulk", "outlier"] {
                let key = format!("producer/{case}/conditioner/{part}");
                let cell = floors
                    .cells
                    .get(&key)
                    .context("conditioner subcell missing")?;
                ensure!(
                    cell.gating == (case == "text")
                        && cell.gating_metrics == ["max_abs", "rmse"]
                        && cell.reference_arms == ["arm-a", "arm-b", "arm-e"],
                    "Runner FAIL {key} conditioner status or references"
                );
            }
        }
        Ok(Self {
            root: root.to_path_buf(),
            floors,
            gating: 0,
            measurements: 0,
            controls: 0,
            control_failed: false,
            defects: Vec::new(),
            seen: std::collections::BTreeSet::new(),
            capture: None,
        })
    }

    fn complete(&self, case: &str, conv_only: bool) -> Result<()> {
        let expected: std::collections::BTreeSet<_> = self
            .floors
            .cells
            .iter()
            .filter(|(key, floor)| {
                let conv = (key.starts_with("producer/") && key.contains("/anchor-"))
                    || (key.starts_with("replay/") && key.contains("/decoded-"));
                floor.gating && key.split('/').nth(1) == Some(case) && (!conv_only || conv)
            })
            .map(|(key, _)| key.clone())
            .collect();
        ensure!(
            !expected.is_empty() && self.seen == expected,
            "Runner FAIL case={case} gating coverage missing={:?} extra={:?}",
            expected.difference(&self.seen).collect::<Vec<_>>(),
            self.seen.difference(&expected).collect::<Vec<_>>()
        );
        if conv_only {
            let cells = match case {
                "text" => 4,
                "first" | "last" => 5,
                "both" => 6,
                _ => anyhow::bail!("Runner FAIL unknown conv case {case}"),
            };
            ensure!(
                expected.len() == cells
                    && expected
                        .iter()
                        .all(|key| self.floors.cells[key].reference_arms
                            == if key.contains("/decoded-video-") {
                                ["arm-a", "arm-b", "arm-c"]
                            } else {
                                ["arm-a", "arm-b", "arm-f"]
                            })
                    && self.gating == cells * 3
                    && self.controls == 0
                    && !self.control_failed,
                "Runner FAIL case={case} conv cells, references or comparison count"
            );
            return Ok(());
        }
        let exact = expected
            .iter()
            .filter(|key| key.starts_with("replay/") && key.contains("/decoder-"))
            .count();
        let comparisons = expected
            .iter()
            .filter(|key| !(key.starts_with("replay/") && key.contains("/decoder-")))
            .map(|key| self.floors.cells[key].reference_arms.len())
            .sum::<usize>()
            + usize::from(case == "first" || case == "both");
        let controls = if case == "text" { 2 } else { 0 };
        ensure!(
            exact == 2
                && self.gating == comparisons
                && self.controls == controls
                && !self.control_failed,
            "Runner FAIL case={case} comparison count or control strength"
        );
        Ok(())
    }

    fn values(&self, relative: &str, shape: &[usize]) -> Result<Vec<f64>> {
        composition_values(&self.root, relative, shape)
    }

    fn f32(&self, relative: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let values: Vec<f32> = self
            .values(relative, shape)?
            .into_iter()
            .map(|v| v as f32)
            .collect();
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "Runner FAIL {relative} F32 overflow"
        );
        Ok(values)
    }

    fn exact<A: crate::verify::Scalar>(
        &self,
        relative: &str,
        candidate: &[A],
        shape: &[usize],
    ) -> Result<()> {
        let reference = self.values(relative, shape)?;
        ensure!(
            candidate.len() == reference.len()
                && candidate
                    .iter()
                    .zip(&reference)
                    .all(|(&a, &b)| a.to_f64().to_bits() == b.to_bits()),
            "Runner FAIL {relative} input bits differ"
        );
        println!(
            "HYPERFLOW_EXACT PASS input={relative} shape={shape:?} bits=0 utc={}",
            composition_stamp()?
        );
        Ok(())
    }

    fn bf16_video_floors(&mut self, case: &str, mode: &str) -> Result<()> {
        let file = format!("BF16_VIDEO_{}_FLOORS.json", mode.to_ascii_uppercase());
        let frozen: serde_json::Value =
            serde_json::from_slice(&std::fs::read(self.root.join(file)).with_context(|| {
                format!("Runner FAIL missing publisher BF16 video freeze {mode}")
            })?)?;
        ensure!(
            frozen["precision"] == "publisher-bf16"
                && frozen["formula"] == "max(2 * worst_valid_reference_pair, 1e-6)"
                && frozen["native_arrays_read"] == false
                && frozen["controls"][mode][case] == true,
            "Runner FAIL unqualified BF16 video floor {mode}/{case}"
        );
        for (role, shape) in [
            ("decoded-video-sample", vec![1, 3, 5, 768, 1344]),
            ("decoded-video-stats", vec![124, 3, 4]),
            ("decoded-video-seams", vec![3, 14, 96, 168]),
        ] {
            let key = format!("{mode}/{case}/{role}");
            let mut cell: CompositionCell =
                serde_json::from_value(frozen["cells"][&key].clone())
                    .with_context(|| format!("Runner FAIL absent BF16 floor {key}"))?;
            ensure!(
                cell.shape == shape
                    && cell.gating
                    && cell.gating_metrics == ["max_abs", "rmse"]
                    && cell.reference_arms == ["arm-a", "arm-b", "arm-c"]
                    && [cell.tau_abs, cell.tau_rmse]
                        .iter()
                        .all(|v| v.is_finite() && *v >= 1e-6),
                "Runner FAIL malformed BF16 floor {key}"
            );
            if mode == "free" {
                cell.gating = false;
            }
            self.floors.cells.insert(key, cell);
        }
        Ok(())
    }

    fn media(&mut self, case: &str, mode: &str, generated: &Generated) -> Result<()> {
        let (shape, decoded) = &generated.video_decoded;
        ensure!(
            shape.len() == 5
                && shape[0] == 1
                && shape[1] == 3
                && shape[2] >= 124
                && shape[3] == 768
                && shape[4] == 1344
                && decoded.len() == shape.iter().product::<usize>(),
            "Runner FAIL decoded video shape {shape:?}"
        );
        let pixels = 768 * 1344;
        let mut sample = Vec::with_capacity(3 * 5 * pixels);
        for channel in 0..3 {
            for frame in [0, 31, 62, 93, 123] {
                let start = (channel * shape[2] + frame) * pixels;
                sample.extend_from_slice(&decoded[start..start + pixels]);
            }
        }
        let mut statistics = Vec::with_capacity(124 * 3 * 4);
        for frame in 0..124 {
            for channel in 0..3 {
                let start = (channel * shape[2] + frame) * pixels;
                let plane = &decoded[start..start + pixels];
                ensure!(
                    plane.iter().all(|v| v.is_finite()),
                    "Runner FAIL decoder nonfinite frame={frame}"
                );
                let mean = plane.iter().map(|v| f64::from(*v)).sum::<f64>() / pixels as f64;
                let variance = plane
                    .iter()
                    .map(|v| (f64::from(*v) - mean).powi(2))
                    .sum::<f64>()
                    / pixels as f64;
                let lo = plane
                    .iter()
                    .map(|v| f64::from(*v))
                    .fold(f64::INFINITY, f64::min);
                let hi = plane
                    .iter()
                    .map(|v| f64::from(*v))
                    .fold(f64::NEG_INFINITY, f64::max);
                statistics.extend([mean, variance.sqrt(), lo, hi]);
            }
        }
        let source = if mode == "replay" { "replay-a" } else { "a" };
        let mut seams = Vec::with_capacity(3 * 14 * 96 * 168);
        for channel in 0..3 {
            for frame in [16, 17, 33, 34, 50, 51, 67, 68, 84, 85, 101, 102, 118, 119] {
                let start = (channel * shape[2] + frame) * pixels;
                for y in (0..768).step_by(8) {
                    seams.extend(
                        decoded[start + y * 1344..start + (y + 1) * 1344]
                            .iter()
                            .step_by(8)
                            .map(|&v| f64::from(v)),
                    );
                }
            }
        }
        for (role, values, extent) in [
            (
                "decoded-video-sample",
                sample.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
                vec![1, 3, 5, 768, 1344],
            ),
            ("decoded-video-stats", statistics, vec![124, 3, 4]),
            ("decoded-video-seams", seams, vec![3, 14, 96, 168]),
            (
                "decoded-audio",
                generated
                    .output
                    .audio
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect(),
                vec![2, 165600],
            ),
        ] {
            let relative = if role.starts_with("decoded-video-") {
                format!("bf16-video/{source}/{case}/{role}")
            } else {
                format!("{source}/{case}/{role}")
            };
            self.score(
                &format!("{mode}/{case}/{role}"),
                &relative,
                &values,
                &extent,
            )?;
        }
        Ok(())
    }

    fn begin_cell(&mut self, key: &str, shape: &[usize]) -> Result<()> {
        let floor = self
            .floors
            .cells
            .get(key)
            .with_context(|| format!("Runner FAIL missing floor {key}"))?;
        ensure!(
            floor.shape == shape,
            "Runner FAIL {key} candidate shape {shape:?} != {:?}",
            floor.shape
        );
        if floor.gating {
            ensure!(
                self.seen.insert(key.to_owned()),
                "Runner FAIL duplicate gating cell {key}"
            );
        }
        Ok(())
    }

    fn score_values<A: crate::verify::Scalar>(
        &mut self,
        key: &str,
        arm: &str,
        candidate: &[A],
        reference: &[f64],
    ) -> Result<()> {
        let floor = self
            .floors
            .cells
            .get(key)
            .context("Runner FAIL score cell")?;
        let m = crate::verify::metrics(key, candidate, reference)?;
        let pass = m.max_abs <= floor.tau_abs
            && m.rmse <= floor.tau_rmse
            && (!floor.gating_metrics.iter().any(|v| v == "relative_rmse")
                || m.rel_rmse <= floor.tau_relative_rmse);
        let result = if !floor.gating {
            "MEASUREMENT"
        } else if pass {
            "PASS"
        } else {
            "DEFECT"
        };
        println!(
            "HYPERFLOW_CELL {result} cell={key} reference={arm} max_abs={} rmse={} relative_rmse={} tau_abs={} tau_rmse={} tau_relative_rmse={} utc={}",
            m.max_abs,
            m.rmse,
            m.rel_rmse,
            floor.tau_abs,
            floor.tau_rmse,
            floor.tau_relative_rmse,
            composition_stamp()?
        );
        if floor.gating {
            self.gating += 1;
            if !pass {
                self.defects.push(format!("{key}/{arm}"));
            }
        } else {
            self.measurements += 1;
        }
        Ok(())
    }

    fn score<A: crate::verify::Scalar>(
        &mut self,
        key: &str,
        relative: &str,
        candidate: &[A],
        shape: &[usize],
    ) -> Result<()> {
        self.begin_cell(key, shape)?;
        if let Some((directory, device)) = &self.capture {
            let file = directory.join(key.replace('/', "-"));
            if key.ends_with("decoded-video-stats") {
                let path = file.with_extension("json");
                ensure!(!path.exists(), "Runner FAIL capture already exists");
                let values: Vec<_> = candidate.iter().map(|&v| v.to_f64()).collect();
                ensure!(
                    values.iter().all(|v| v.is_finite()),
                    "Runner FAIL capture nonfinite"
                );
                std::fs::write(
                    path,
                    serde_json::to_vec(&serde_json::json!({"shape":shape,"value":values}))?,
                )?;
            } else {
                let path = file.with_extension("safetensors");
                ensure!(!path.exists(), "Runner FAIL capture already exists");
                let mut trace = crate::trace::Trace::default();
                trace.set_device(device)?;
                trace.add(
                    "value".into(),
                    shape.to_vec(),
                    candidate.iter().map(|&v| v.to_f64() as f32).collect(),
                )?;
                trace.save(&path)?;
            }
        }
        for arm in self.floors.cells[key].reference_arms.clone() {
            let reference = self.values(&format!("{arm}/{relative}"), shape)?;
            self.score_values(key, &arm, candidate, &reference)?;
        }
        Ok(())
    }

    fn conditioner(&mut self, case: &str, values: &[bf16]) -> Result<()> {
        ensure!(
            values.len().is_multiple_of(5120),
            "Runner FAIL conditioner extent"
        );
        let rows = values.len() / 5120;
        let outliers: Vec<usize> =
            serde_json::from_value(self.floors.conditioner_partition["outlier_channels"].clone())?;
        ensure!(
            !outliers.is_empty()
                && outliers.len() < 5120
                && outliers.windows(2).all(|v| v[0] < v[1])
                && outliers.iter().all(|&v| v < 5120),
            "Runner FAIL conditioner partition"
        );
        for (part, is_outlier) in [("bulk", false), ("outlier", true)] {
            let channels: Vec<_> = (0..5120)
                .filter(|c| outliers.binary_search(c).is_ok() == is_outlier)
                .collect();
            let actual: Vec<_> = values
                .as_chunks::<5120>()
                .0
                .iter()
                .flat_map(|row| channels.iter().map(|&c| row[c]))
                .collect();
            let key = format!("producer/{case}/conditioner/{part}");
            self.begin_cell(&key, &[1, rows, channels.len()])?;
            for arm in self.floors.cells[&key].reference_arms.clone() {
                let full = self.values(
                    &format!("{arm}/producer-a/{case}/conditioner"),
                    &[1, rows, 5120],
                )?;
                let reference: Vec<_> = full
                    .as_chunks::<5120>()
                    .0
                    .iter()
                    .flat_map(|row| channels.iter().map(|&c| row[c]))
                    .collect();
                self.score_values(&key, &arm, &actual, &reference)?;
            }
        }
        if case == "both" {
            let refs = ["arm-a", "arm-b", "arm-e"].map(|arm| -> Result<f64> {
                Ok(self.values(
                    &format!("{arm}/producer-a/{case}/conditioner"),
                    &[1, rows, 5120],
                )?[1953 * 5120 + 731])
            });
            let [a, b, e] = refs;
            println!(
                "HYPERFLOW_NAMED_MEASUREMENT cell=producer/both/conditioner/row1953-column731 native={} reference_a={} reference_b={} reference_e={} gating=false",
                values[1953 * 5120 + 731],
                a?,
                b?,
                e?
            );
            self.measurements += 1;
        }
        Ok(())
    }

    pub(super) fn sample_step(
        &mut self,
        case: &str,
        i: usize,
        step: &super::SampleTrace,
    ) -> Result<()> {
        let common = format!("common/a/{case}");
        let video_rows = self.floors.cells[&format!("free/{case}/post-video-{i}")].shape[0];
        for (role, values, shape) in [
            ("post-video", &step.video, vec![video_rows, 96]),
            ("post-audio", &step.audio, vec![414, 32]),
            ("velocity-video", &step.video_velocity, vec![37296, 96]),
            ("velocity-audio", &step.audio_velocity, vec![414, 32]),
        ] {
            self.score(
                &format!("free/{case}/{role}-{i}"),
                &format!("a/{case}/{role}-{i}"),
                values,
                &shape,
            )?;
        }
        let prefix = (video_rows - 37296) * 96;
        let initial = self.values(&format!("{common}/enter-video-0"), &[video_rows, 96])?;
        ensure!(
            step.video[..prefix]
                .iter()
                .zip(&initial[..prefix])
                .all(|(&a, &b)| f64::from(a).to_bits() == b.to_bits()),
            "Runner FAIL conditioning prefix changed step{i}"
        );
        println!(
            "HYPERFLOW_FORWARD case={case} phase=free step={i} state=COMPLETE utc={}",
            composition_stamp()?
        );
        Ok(())
    }

    fn conditioner_reading(&mut self, values: &[bf16]) -> Result<()> {
        let row = values
            .get(20 * 5120..21 * 5120)
            .context("Runner FAIL conditioner row20")?;
        let rule = &self.floors.conditioner_reading;
        let m = crate::verify::metrics("conditioner-F32-row", row, &rule.cpu_row)?;
        let coordinate = f64::from(row[731].to_f32());
        let delta = (coordinate - f64::from(rule.cpu_row[731])).abs();
        let pass = m.max_abs <= rule.max_abs && m.rmse <= rule.rmse && delta <= rule.coordinate_abs;
        println!(
            "HYPERFLOW_CONDITIONER_READING {} row=20 column=731 native={coordinate} cpu_f32={} distance={delta} allowed_coordinate={} max_abs={} allowed_max_abs={} rmse={} allowed_rmse={} utc={}",
            if pass { "PASS" } else { "DEFECT" },
            rule.cpu_row[731],
            rule.coordinate_abs,
            m.max_abs,
            rule.max_abs,
            m.rmse,
            rule.rmse,
            composition_stamp()?
        );
        self.gating += 1;
        if !pass {
            self.defects.push("conditioner/F32-row".into());
        }
        Ok(())
    }

    fn handoff_control(&mut self, stream: &str, values: &[f32]) -> Result<()> {
        let swapped: Vec<_> = match stream {
            "audio" => {
                ensure!(
                    values.len() == 414 * 32,
                    "Runner FAIL teacher audio control extent"
                );
                values[207 * 32..]
                    .iter()
                    .chain(&values[..207 * 32])
                    .copied()
                    .collect()
            }
            "video" => {
                ensure!(
                    values.len() == 37296 * 96,
                    "Runner FAIL teacher video control extent"
                );
                values
                    .as_chunks::<96>()
                    .0
                    .iter()
                    .flat_map(|row| row[48..].iter().chain(&row[..48]))
                    .copied()
                    .collect()
            }
            _ => anyhow::bail!("Runner FAIL unknown teacher control stream {stream}"),
        };
        let key = format!("teacher/text/teacher-{stream}-3");
        let floor = self
            .floors
            .cells
            .get(&key)
            .context("Runner FAIL handoff control floor")?;
        let shape = if stream == "video" {
            [37296, 96]
        } else {
            [414, 32]
        };
        for arm in ["arm-a", "arm-b"] {
            let reference = self.values(&format!("{arm}/a/text/teacher-{stream}-3"), &shape)?;
            let m = crate::verify::metrics(&key, &swapped, &reference)?;
            let ratios = [
                m.max_abs / floor.tau_abs,
                m.rmse / floor.tau_rmse,
                m.rel_rmse / floor.tau_relative_rmse,
            ];
            let detected = ratios.into_iter().any(|r| r > 3.0);
            println!(
                "HYPERFLOW_CONTROL {} cell={key} reference={arm} channel_block_swap=true factor=3 ratio_abs={} ratio_rmse={} ratio_relative_rmse={} utc={}",
                if detected { "PASS" } else { "FAIL" },
                ratios[0],
                ratios[1],
                ratios[2],
                composition_stamp()?
            );
            self.control_failed |= !detected;
        }
        self.controls += 1;
        Ok(())
    }
}

#[cfg(test)]
#[test]
#[ignore = "GPU; FF_MODEL, FF_HYPERFLOW_MODEL, FF_HYPERFLOW_COMPOSITION, FF_HYPERFLOW_CASE and FF_H3_CAPTURE"]
fn hyperflow_anchor_and_decoders_match_frozen_references() -> Result<()> {
    conv_decoder_gate::<false>(true, super::Placement::Derived)
}

#[cfg(test)]
#[test]
#[ignore = "GPU; publisher BF16 video / F32 audio references and capture directory"]
fn minimax_video_decoder_matches_frozen_references() -> Result<()> {
    conv_decoder_gate::<false>(false, super::Placement::Derived)
}

#[cfg(test)]
#[test]
#[ignore = "GPU; publisher BF16 video / F32 audio references and capture directory"]
fn hyperflow_streamed_decoder_matches_frozen_references() -> Result<()> {
    conv_decoder_gate::<false>(true, super::Placement::Streamed)
}

#[cfg(test)]
#[test]
#[ignore = "GPU; publisher BF16 video / F32 audio references and capture directory"]
fn minimax_streamed_decoder_matches_frozen_references() -> Result<()> {
    conv_decoder_gate::<false>(false, super::Placement::Streamed)
}

#[cfg(test)]
#[test]
#[ignore = "GPU; publisher BF16 video / F32 audio references and capture directory"]
fn hyperflow_serial_decoder_matches_frozen_references() -> Result<()> {
    conv_decoder_gate::<true>(true, super::Placement::Derived)
}

#[cfg(test)]
fn conv_decoder_gate<const SERIAL: bool>(
    hyperflow: bool,
    placement: super::Placement,
) -> Result<()> {
    use crate::minimax_h3::{KeyframeAnchor, prepare_video};
    use crate::serve::image::RgbImage;
    let path = |key| {
        std::env::var_os(key)
            .map(std::path::PathBuf::from)
            .with_context(|| format!("{key} missing"))
    };
    let case = std::env::var("FF_HYPERFLOW_CASE").context("FF_HYPERFLOW_CASE missing")?;
    let model = path("FF_MODEL")?;
    let component = if hyperflow {
        path("FF_HYPERFLOW_MODEL")?
    } else {
        model.join("transformer")
    };
    let spec = H3Spec::parse(&serde_json::from_slice(&std::fs::read(
        component.join("config.json"),
    )?)?)?;
    ensure!(
        spec.is_hyperflow()? == hyperflow,
        "Runner FAIL HyperFlow component required"
    );
    let mut gate = CompositionGate::read(&path("FF_HYPERFLOW_COMPOSITION")?)?;
    gate.bf16_video_floors(&case, "replay")?;
    let out = path("FF_H3_CAPTURE")?;
    std::fs::create_dir_all(&out)?;
    let image = |height, width, offset| RgbImage {
        height,
        width,
        pixels: (0..height * width * 3)
            .map(|i| ((i * 73 + offset) % 251) as u8)
            .collect(),
    };
    let (first, last) = (image(273, 481, 17), image(389, 267, 83));
    let anchors = match case.as_str() {
        "text" => vec![],
        "first" => vec![(KeyframeAnchor::First, &first)],
        "last" => vec![(KeyframeAnchor::Last, &first)],
        "both" => vec![
            (KeyframeAnchor::First, &first),
            (KeyframeAnchor::Last, &last),
        ],
        _ => anyhow::bail!("Runner FAIL unknown conv case {case}"),
    };
    let request = prepare_video(
        &model,
        &spec,
        "A small animal walks beside a wooden bridge in warm daylight.",
        &anchors,
        ((124, 42), (6, (12.0, 3.0))),
    )?;
    let (geometry, _) = request.layout(&spec)?;
    let vae = VaeSpec::read(&model)?;
    let audio = AudioVaeSpec::read(&model)?;
    let video_shape = [
        1,
        vae.latent_channels,
        geometry.latent_frames,
        geometry.latent_height,
        geometry.latent_width,
    ];
    let anchor_shape = [
        1,
        vae.latent_channels,
        1,
        geometry.latent_height,
        geometry.latent_width,
    ];
    ensure!(
        video_shape == [1, 24, 37, 48, 84]
            && request.height == 768
            && request.width == 1344
            && geometry.frames == 124,
        "Runner FAIL conv production geometry"
    );
    for (i, (_, pixels)) in request.keyframes.iter().enumerate() {
        let values: Vec<_> = pixels.iter().map(|&v| f64::from(v)).collect();
        gate.exact(
            &format!("prepared/{case}/keyframe-{i}"),
            &values,
            &[3, 768, 1344],
        )?;
    }
    let runtime = std::env::temp_dir().join(format!("ff-hyperflow-conv-{}", std::process::id()));
    std::fs::create_dir(&runtime)?;
    let result = (|| -> Result<()> {
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        let mut blas = super::open_blas(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(false, "HyperFlow conv cells".into())),
        )?;
        gate.capture = Some((out.clone(), device.info.name.clone()));
        println!(
            "HYPERFLOW_CONV_CASE_START case={case} conditioner_forwards=0 dit_forwards=0 utc={}",
            composition_stamp()?
        );
        let eps = if request.keyframes.is_empty() {
            None
        } else {
            let values = gate.f32(&format!("common/a/{case}/posterior-eps-0"), &anchor_shape)?;
            if request.keyframes.len() == 2 {
                gate.exact(
                    &format!("common/a/{case}/posterior-eps-1"),
                    &values,
                    &anchor_shape,
                )?;
            }
            Some(device.stream.clone_htod(&values)?)
        };
        let produced = keyframes(&device, &blas, &model, &request, eps.as_ref())?;
        ensure!(
            produced.len() == anchors.len(),
            "Runner FAIL conv anchor count"
        );
        for (i, values) in produced.iter().enumerate() {
            gate.score(
                &format!("producer/{case}/anchor-{i}"),
                &format!("producer-a/{case}/anchor-{i}"),
                values,
                &anchor_shape,
            )?;
        }
        let video = gate.f32(
            &format!("arm-a/replay-a/{case}/decoder-video-input"),
            &video_shape,
        )?;
        let audio_input = gate.f32(
            &format!("arm-a/replay-a/{case}/decoder-audio-input"),
            &[2, 32, 207],
        )?;
        let generated = decode::<false, SERIAL>(
            &device,
            &mut blas,
            (&model, &request, None),
            (&geometry, &vae, &audio),
            (video, audio_input),
            (device.ctx.mem_get_info()?.0, placement),
        )?;
        gate.exact(
            &format!("arm-a/replay-a/{case}/decoder-video-input"),
            &generated.video_latents.1,
            &video_shape,
        )?;
        gate.exact(
            &format!("arm-a/replay-a/{case}/decoder-audio-input"),
            &generated.audio_latents.1,
            &[2, 32, 207],
        )?;
        gate.media(&case, "replay", &generated)?;
        gate.complete(&case, true)?;
        Ok(())
    })();
    let cleanup = std::fs::remove_dir_all(&runtime);
    result?;
    cleanup?;
    let pass = gate.defects.is_empty();
    println!(
        "HYPERFLOW_CONV_CASE_COMPLETE case={case} result={} gating_comparisons={} defects={} conditioner_forwards=0 dit_forwards=0 utc={}",
        if pass { "PASS" } else { "DEFECT" },
        gate.gating,
        gate.defects.len(),
        composition_stamp()?
    );
    ensure!(pass, "DEFECT conv cells {:?}", gate.defects);
    Ok(())
}

#[cfg(test)]
#[test]
#[ignore = "GPU; FF_MODEL, FF_HYPERFLOW_MODEL, FF_HYPERFLOW_COMPOSITION fixtures and FF_HYPERFLOW_CASE"]
fn hyperflow_composition_matches_frozen_references() -> Result<()> {
    use crate::minimax_h3::{KeyframeAnchor, prepare_video};
    use crate::serve::image::RgbImage;
    let path = |key| {
        std::env::var_os(key)
            .map(std::path::PathBuf::from)
            .with_context(|| format!("{key} missing"))
    };
    let selected = std::env::var("FF_HYPERFLOW_CASE").context("FF_HYPERFLOW_CASE missing")?;
    ensure!(
        matches!(selected.as_str(), "text" | "first" | "last" | "both"),
        "Runner FAIL unknown composition case {selected}"
    );
    let (model, component) = (path("FF_MODEL")?, path("FF_HYPERFLOW_MODEL")?);
    let spec = H3Spec::parse(&serde_json::from_slice(&std::fs::read(
        component.join("config.json"),
    )?)?)?;
    ensure!(
        spec.is_hyperflow()?,
        "setup needs the HyperFlow fl2va component"
    );
    let shared_gate = Rc::new(std::cell::RefCell::new(CompositionGate::read(&path(
        "FF_HYPERFLOW_COMPOSITION",
    )?)?));
    shared_gate
        .borrow_mut()
        .bf16_video_floors(&selected, "replay")?;
    // 2026-11-22 erratum: free/*/decoded-video-{sample,stats,seams} stay gating=false measurements against the frozen BF16 floors; VAE channel control ctrl_max_abs 4.38-6.17 is below free-trajectory tau_abs 8.00-8.76.
    shared_gate
        .borrow_mut()
        .bf16_video_floors(&selected, "free")?;
    let vae = VaeSpec::read(&model)?;
    let audio_spec = AudioVaeSpec::read(&model)?;
    let image = |height, width, offset| RgbImage {
        height,
        width,
        pixels: (0..height * width * 3)
            .map(|i| ((i * 73 + offset) % 251) as u8)
            .collect(),
    };
    let (first, last) = (image(273, 481, 17), image(389, 267, 83));
    let runtime =
        std::env::temp_dir().join(format!("ff-hyperflow-composition-{}", std::process::id()));
    std::fs::create_dir(&runtime)?;
    let result = (|| -> Result<bool> {
        let mut cases = Vec::new();
        for (case, anchors) in [
            ("text", vec![]),
            ("first", vec![(KeyframeAnchor::First, &first)]),
            ("last", vec![(KeyframeAnchor::Last, &first)]),
            (
                "both",
                vec![
                    (KeyframeAnchor::First, &first),
                    (KeyframeAnchor::Last, &last),
                ],
            ),
        ]
        .into_iter()
        .filter(|(case, _)| *case == selected)
        {
            let mut gate = shared_gate.borrow_mut();
            let request = prepare_video(
                &model,
                &spec,
                "A small animal walks beside a wooden bridge in warm daylight.",
                &anchors,
                ((124, 42), (6, (12.0, 3.0))),
            )?;
            let (geometry, layout) = request.layout(&spec)?;
            let common = format!("common/a/{case}");
            let anchor_count = request.keyframes.len();
            let (text_rows, video_rows, audio_rows) = (
                request.token_ids.len(),
                layout.video_indices.len(),
                layout.audio_indices.len(),
            );
            let video_shape = [
                1,
                vae.latent_channels,
                geometry.latent_frames,
                geometry.latent_height,
                geometry.latent_width,
            ];
            let anchor_shape = [
                1,
                vae.latent_channels,
                1,
                geometry.latent_height,
                geometry.latent_width,
            ];
            ensure!(
                video_shape == [1, 24, 37, 48, 84]
                    && audio_rows == 414
                    && spec.video_row_width() == 96,
                "Runner FAIL production geometry"
            );
            gate.exact(
                &format!("{common}/input_ids"),
                &request
                    .token_ids
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect::<Vec<_>>(),
                &[1, text_rows],
            )?;
            gate.exact(
                &format!("{common}/token_tags"),
                &layout
                    .token_tags
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect::<Vec<_>>(),
                &[layout.rows()],
            )?;
            gate.exact(
                &format!("{common}/position_ids"),
                &layout
                    .position_ids
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>(),
                &[layout.rows(), 3],
            )?;
            for (role, ids) in [
                ("video_indices", &layout.video_indices),
                ("audio_indices", &layout.audio_indices),
                ("text_indices", &layout.text_indices),
            ] {
                gate.exact(
                    &format!("{common}/{role}"),
                    &ids.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                    &[ids.len()],
                )?;
            }
            for (i, step) in hyperflow_steps(6)?.iter().enumerate() {
                let row = RowTimesteps::hyperflow(&layout, step);
                gate.exact(
                    &format!("{common}/t-{i}"),
                    &row.timesteps,
                    &[row.timesteps.len()],
                )?;
                gate.exact(
                    &format!("{common}/r-{i}"),
                    &row.endpoints,
                    &[row.endpoints.len()],
                )?;
                gate.exact(
                    &format!("{common}/time-indices-{i}"),
                    &row.indices
                        .iter()
                        .map(|&v| f64::from(v))
                        .collect::<Vec<_>>(),
                    &[layout.rows()],
                )?;
            }
            let input = {
                let load =
                    |role: &str, shape: &[usize]| gate.f32(&format!("{common}/{role}"), shape);
                let posterior = if anchor_count == 0 {
                    Vec::new()
                } else {
                    load("posterior-eps-0", &anchor_shape)?
                };
                if anchor_count == 2 {
                    gate.exact(
                        &format!("{common}/posterior-eps-1"),
                        &posterior,
                        &anchor_shape,
                    )?;
                }
                let mut anchor_draws = Vec::new();
                let mut reference_anchors = Vec::new();
                for i in 0..anchor_count {
                    anchor_draws.extend(load(&format!("noise-{i}"), &anchor_shape)?);
                    reference_anchors.push(load(&format!("anchor-{i}"), &anchor_shape)?);
                }
                let mut teachers = Vec::new();
                for i in 0..6 {
                    teachers.push((
                        load(&format!("enter-video-{i}"), &[video_rows, 96])?,
                        load(&format!("enter-audio-{i}"), &[audio_rows, 32])?,
                    ));
                }
                ReplayInput {
                    case,
                    gate: Rc::clone(&shared_gate),
                    draws: [
                        posterior,
                        anchor_draws,
                        load(&format!("noise-{anchor_count}"), &video_shape)?,
                        load(&format!("noise-{}", anchor_count + 1), &[audio_rows, 32])?,
                    ],
                    conditioner: load("conditioner", &[1, text_rows, 5120])?
                        .into_iter()
                        .map(bf16::from_f32)
                        .collect(),
                    anchors: reference_anchors,
                    teacher_inputs: teachers,
                }
            };
            println!(
                "HYPERFLOW_CASE_START case={case} phase=producer utc={}",
                composition_stamp()?
            );
            {
                let device = Device::new(0)?;
                let ops = Ops::new(&device.ctx)?;
                let mut blas = super::open_blas(
                    &device,
                    &ops,
                    &runtime,
                    Rc::new(Closure::new(false, "HyperFlow producers".into())),
                )?;
                let conditioner = condition(&device, &mut blas, &model, &request, &runtime)?;
                gate.conditioner(case, &conditioner)?;
                if case == "both" {
                    gate.conditioner_reading(&conditioner)?;
                }
                let posterior = if anchor_count == 0 {
                    None
                } else {
                    Some(device.stream.clone_htod(&input.draws[0])?)
                };
                let produced = keyframes(&device, &blas, &model, &request, posterior.as_ref())?;
                ensure!(
                    produced.len() == anchor_count,
                    "Runner FAIL producer anchor count"
                );
                for (i, anchor) in produced.iter().enumerate() {
                    gate.score(
                        &format!("producer/{case}/anchor-{i}"),
                        &format!("producer-a/{case}/anchor-{i}"),
                        anchor,
                        &anchor_shape,
                    )?;
                }
                if case == "first" {
                    let mut control = request.clone();
                    control.pixel_values =
                        gate.f32("arm-a/media-control/first/pixel_values", &[4032, 1536])?;
                    ensure!(
                        control.pixel_values != request.pixel_values,
                        "Runner FAIL media-control input identical"
                    );
                    let output = condition(&device, &mut blas, &model, &control, &runtime)?;
                    let distance =
                        crate::verify::metrics("conditioner-media-control", &output, &conditioner)?;
                    let reached = !distance.exact;
                    println!(
                        "HYPERFLOW_MEDIA {} ids_grids_equal=true pixels_different=true conditioner_different={reached} max_abs={} rmse={} utc={}",
                        if reached { "PASS" } else { "DEFECT" },
                        distance.max_abs,
                        distance.rmse,
                        composition_stamp()?
                    );
                    gate.gating += 1;
                    if !reached {
                        gate.defects.push("conditioner/media-control".into());
                    }
                }
            }
            drop(gate);
            cases.push((case, request, input));
        }
        for (case, request, input) in &cases {
            let mut gate = shared_gate.borrow_mut();
            let device = Device::new(0)?;
            let ops = Ops::new(&device.ctx)?;
            let mut blas = super::open_blas(
                &device,
                &ops,
                &runtime,
                Rc::new(Closure::new(false, "HyperFlow teacher".into())),
            )?;
            let (_, layout) = request.layout(&spec)?;
            let VideoTransformer {
                mut dit,
                table,
                adaln: plan,
                scale,
                ..
            } = prepare_transformer(
                (&device, &mut blas),
                (&component, None, None),
                (&spec, &layout),
                (request.evaluations, request.shifts),
                None,
            )?;
            ensure!(
                input.teacher_inputs.len() == plan.row_pairs.len(),
                "Runner FAIL teacher step count"
            );
            let kernels = BlockKernels::new(&device)?;
            let context = BlockContext {
                s: &device.stream,
                blas: &blas,
                kernels: &kernels,
                spec: &spec,
                lora_scale: scale,
            };
            let text = super::upload_runtime_half(
                &device.stream,
                device.info.arch.fp16_stored_half(),
                "video_text",
                &input.conditioner,
            )?;
            println!(
                "HYPERFLOW_CASE_START case={case} phase=teacher steps=6 utc={}",
                composition_stamp()?
            );
            for (i, (video, audio)) in input.teacher_inputs.iter().enumerate() {
                println!(
                    "HYPERFLOW_FORWARD case={case} phase=teacher step={i} state=START utc={}",
                    composition_stamp()?
                );
                let s = &device.stream;
                let video = s.clone_htod(video)?;
                let audio = s.clone_htod(audio)?;
                let row_pairs = s.clone_htod(&plan.row_pairs[i])?;
                let row_timesteps = s.clone_htod(&plan.row_timesteps[i])?;
                dit.step(
                    &context,
                    &table,
                    (
                        u32::try_from(plan.pairs.len())?,
                        u32::try_from(plan.timesteps.len())?,
                    ),
                    &super::StepInputs {
                        video: &video,
                        audio: &audio,
                        text: &text,
                        row_pairs: &row_pairs,
                        row_timesteps: &row_timesteps,
                    },
                )?;
                let video = s.clone_dtoh(&dit.video)?;
                let audio = s.clone_dtoh(&dit.audio)?;
                for (role, values, shape) in
                    [("video", &video, [37296, 96]), ("audio", &audio, [414, 32])]
                {
                    gate.score(
                        &format!("teacher/{case}/teacher-{role}-{i}"),
                        &format!("a/{case}/teacher-{role}-{i}"),
                        values,
                        &shape,
                    )?;
                }
                if *case == "text" && i == 3 {
                    gate.handoff_control("video", &video)?;
                    gate.handoff_control("audio", &audio)?;
                }
                println!(
                    "HYPERFLOW_FORWARD case={case} phase=teacher step={i} state=COMPLETE utc={}",
                    composition_stamp()?
                );
            }
        }
        for (case, request, _) in &cases {
            let mut gate = shared_gate.borrow_mut();
            println!(
                "HYPERFLOW_CASE_START case={case} phase=isolated_decoder video_dtype=BF16 audio_dtype=F32 utc={}",
                composition_stamp()?
            );
            let (geometry, layout) = request.layout(&spec)?;
            let video_shape = [
                1,
                vae.latent_channels,
                geometry.latent_frames,
                geometry.latent_height,
                geometry.latent_width,
            ];
            let video_rows = layout.video_indices.len();
            let common = format!("common/a/{case}");
            let decoded = {
                let device = Device::new(0)?;
                let ops = Ops::new(&device.ctx)?;
                let mut blas = super::open_blas(
                    &device,
                    &ops,
                    &runtime,
                    Rc::new(Closure::new(false, "HyperFlow isolated decoders".into())),
                )?;
                let last_video = gate.f32(&format!("{common}/post-video-5"), &[video_rows, 96])?;
                let last_audio = gate.f32(&format!("{common}/post-audio-5"), &[414, 32])?;
                let latents = decoder_latents(
                    (&spec, &vae, &audio_spec),
                    (&geometry, &layout),
                    (&last_video, &last_audio),
                )?;
                gate.exact(
                    &format!("arm-a/replay-a/{case}/decoder-video-input"),
                    &latents.0,
                    &video_shape,
                )?;
                gate.exact(
                    &format!("arm-a/replay-a/{case}/decoder-audio-input"),
                    &latents.1,
                    &[2, 32, 207],
                )?;
                for role in ["video", "audio"] {
                    gate.seen
                        .insert(format!("replay/{case}/decoder-{role}-input"));
                }
                decode::<false, false>(
                    &device,
                    &mut blas,
                    (&model, request, None),
                    (&geometry, &vae, &audio_spec),
                    latents,
                    (device.ctx.mem_get_info()?.0, super::Placement::Derived),
                )?
            };
            gate.media(case, "replay", &decoded)?;
        }
        for (case, request, input) in cases {
            println!(
                "HYPERFLOW_CASE_START case={case} phase=free_measurement steps=6 utc={}",
                composition_stamp()?
            );
            let (geometry, layout) = request.layout(&spec)?;
            let (video_rows, audio_rows, text_rows) = (
                layout.video_indices.len(),
                layout.audio_indices.len(),
                request.token_ids.len(),
            );
            let video_shape = [
                1,
                vae.latent_channels,
                geometry.latent_frames,
                geometry.latent_height,
                geometry.latent_width,
            ];
            let common = format!("common/a/{case}");
            let generated = run_mode::<false>(
                (&model, &component),
                None,
                &request,
                (0, &runtime),
                None,
                NoiseDraws::Recorded(input),
            )?;
            let replay = generated
                .replay
                .as_ref()
                .context("Runner FAIL composition replay missing")?;
            ensure!(
                replay.steps.len() == 6,
                "Runner FAIL composition sample count"
            );
            let mut gate = shared_gate.borrow_mut();
            gate.exact(
                &format!("{common}/enter-video-0"),
                &replay.initial.0,
                &[video_rows, 96],
            )?;
            gate.exact(
                &format!("{common}/enter-audio-0"),
                &replay.initial.1,
                &[audio_rows, 32],
            )?;
            gate.score(
                &format!("free/{case}/refiner"),
                &format!("a/{case}/refiner"),
                &replay.refiner,
                &[1, text_rows, spec.hidden],
            )?;
            gate.score(
                &format!("free/{case}/decoder-video-input"),
                &format!("a/{case}/decoder-video-input"),
                &generated.video_latents.1,
                &video_shape,
            )?;
            gate.score(
                &format!("free/{case}/decoder-audio-input"),
                &format!("a/{case}/decoder-audio-input"),
                &generated.audio_latents.1,
                &[2, 32, 207],
            )?;
            gate.media(case, "free", &generated)?;
            println!(
                "HYPERFLOW_CHAIN case={case} rows={} ffn_chunk={} text_chunk={} forwards=6 prefix_bits=0 utc={}",
                layout.rows(),
                replay.chunks.0,
                replay.chunks.1,
                composition_stamp()?
            );
        }
        let gate = shared_gate.borrow();
        gate.complete(&selected, false)?;
        println!(
            "{} HYPERFLOW_COMPOSITION case={selected} cases=1 sampler_steps=6 teacher_steps=6 gating_comparisons={} measurements={} controls={} defects={} utc={}",
            if gate.defects.is_empty() {
                "PASS"
            } else {
                "DEFECT"
            },
            gate.gating,
            gate.measurements,
            gate.controls,
            gate.defects.len(),
            composition_stamp()?
        );
        Ok(gate.defects.is_empty())
    })();
    let cleanup = std::fs::remove_dir_all(&runtime);
    let pass = result?;
    cleanup?;
    println!(
        "HYPERFLOW_CASE_COMPLETE case={selected} result={} utc={}",
        if pass { "PASS" } else { "DEFECT" },
        composition_stamp()?
    );
    ensure!(
        pass,
        "DEFECT composition cells {:?}",
        shared_gate.borrow().defects
    );
    Ok(())
}

#[cfg(test)]
#[test]
fn composition_case_completion_retains_defects_and_rejects_missing_cells() -> Result<()> {
    let keys = [
        "teacher/first/teacher-video-1",
        "replay/first/decoder-video-input",
        "replay/first/decoder-audio-input",
        "producer/first/conditioner/bulk",
    ];
    let floors = CompositionFloors {
        freeze_revision: "r6-conv".into(),
        formula: "max(2 * worst_valid_reference_pair, 1e-6)".into(),
        native_candidate_run: false,
        native_candidate_used_in_floor: false,
        r6_qualified: true,
        r6_control_checks: 41,
        conditioner_partition: serde_json::json!({"outlier_channels":[731]}),
        cells: keys
            .iter()
            .map(|key| {
                (
                    (*key).into(),
                    CompositionCell {
                        shape: vec![1],
                        tau_abs: 1e-6,
                        tau_rmse: 1e-6,
                        tau_relative_rmse: 1e-6,
                        gating: !key.starts_with("producer/"),
                        gating_metrics: vec!["max_abs".into(), "rmse".into()],
                        reference_arms: if key.starts_with("producer/") {
                            vec!["arm-a".into(), "arm-b".into(), "arm-e".into()]
                        } else {
                            vec!["arm-a".into(), "arm-b".into()]
                        },
                    },
                )
            })
            .collect(),
        conditioner_reading: ConditionerReading {
            cpu_row: Vec::new(),
            max_abs: 0.0,
            rmse: 0.0,
            coordinate_abs: 0.0,
        },
        stage_controls: std::collections::BTreeMap::new(),
    };
    let mut gate = CompositionGate {
        root: std::path::PathBuf::new(),
        floors,
        gating: 3,
        measurements: 0,
        controls: 0,
        control_failed: false,
        defects: vec![format!("{}/arm-a", keys[0])],
        seen: keys
            .into_iter()
            .filter(|key| !key.starts_with("producer/"))
            .map(str::to_owned)
            .collect(),
        capture: None,
    };
    gate.complete("first", false)?;
    assert!(!gate.defects.is_empty());
    gate.score_values(keys[3], "arm-e", &[100.0f32], &[0.0])?;
    assert_eq!(gate.measurements, 1);
    assert_eq!(gate.defects.len(), 1);
    gate.complete("first", false)?;
    gate.gating -= 1;
    assert!(gate.complete("first", false).is_err());
    gate.gating += 1;
    gate.seen.remove(keys[0]);
    assert!(gate.complete("first", false).is_err());
    let conv = [
        "producer/first/anchor-0",
        "replay/first/decoded-video-sample",
        "replay/first/decoded-video-stats",
        "replay/first/decoded-video-seams",
        "replay/first/decoded-audio",
    ];
    let cell = gate.floors.cells[keys[0]].clone();
    gate.floors.cells = conv
        .iter()
        .map(|key| {
            let mut cell = cell.clone();
            cell.reference_arms = vec![
                "arm-a".into(),
                "arm-b".into(),
                if key.contains("/decoded-video-") {
                    "arm-c".into()
                } else {
                    "arm-f".into()
                },
            ];
            ((*key).into(), cell)
        })
        .collect();
    gate.seen = conv.iter().map(|key| (*key).to_owned()).collect();
    gate.gating = 15;
    gate.complete("first", true)?;
    assert!(!gate.defects.is_empty());
    gate.seen.remove(conv[0]);
    assert!(gate.complete("first", true).is_err());
    gate.seen.insert(conv[0].into());
    gate.gating -= 1;
    assert!(gate.complete("first", true).is_err());

    let dir = std::env::temp_dir().join(format!("ff-bf16-video-floors-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let check_bf16_modes = (|| -> Result<()> {
        for (mode, filename) in [
            ("replay", "BF16_VIDEO_REPLAY_FLOORS.json"),
            ("free", "BF16_VIDEO_FREE_FLOORS.json"),
        ] {
            let mut cells = serde_json::Map::new();
            for (role, shape) in [
                (
                    "decoded-video-sample",
                    serde_json::json!([1, 3, 5, 768, 1344]),
                ),
                ("decoded-video-stats", serde_json::json!([124, 3, 4])),
                ("decoded-video-seams", serde_json::json!([3, 14, 96, 168])),
            ] {
                cells.insert(
                    format!("{mode}/first/{role}"),
                    serde_json::json!({
                        "shape": shape,
                        "tau_abs": 0.5,
                        "tau_rmse": 0.02,
                        "tau_relative_rmse": 1e-6,
                        "gating": true,
                        "gating_metrics": ["max_abs", "rmse"],
                        "reference_arms": ["arm-a", "arm-b", "arm-c"],
                    }),
                );
            }
            std::fs::write(
                dir.join(filename),
                serde_json::to_vec(&serde_json::json!({
                    "precision": "publisher-bf16",
                    "formula": "max(2 * worst_valid_reference_pair, 1e-6)",
                    "native_arrays_read": false,
                    "controls": {mode: {"first": true}},
                    "cells": cells,
                }))?,
            )?;
        }
        gate.root = dir.clone();
        gate.floors.cells.clear();
        gate.seen.clear();
        gate.gating = 0;
        gate.measurements = 0;
        gate.controls = 0;
        gate.defects.clear();
        for key in [
            "teacher/first/teacher-audio-4",
            "replay/first/decoder-video-input",
            "replay/first/decoder-audio-input",
        ] {
            gate.floors.cells.insert(key.into(), cell.clone());
            gate.seen.insert(key.into());
        }
        gate.bf16_video_floors("first", "replay")?;
        gate.bf16_video_floors("first", "free")?;
        for role in [
            "decoded-video-sample",
            "decoded-video-stats",
            "decoded-video-seams",
        ] {
            assert!(gate.floors.cells[&format!("replay/first/{role}")].gating);
            let free_shape = {
                let free_cell = &gate.floors.cells[&format!("free/first/{role}")];
                assert!(!free_cell.gating);
                assert_eq!(free_cell.tau_abs.to_bits(), 0.5f64.to_bits());
                free_cell.shape.clone()
            };
            gate.begin_cell(&format!("replay/first/{role}"), &free_shape)?;
            gate.begin_cell(&format!("free/first/{role}"), &free_shape)?;
            for arm in ["arm-a", "arm-b", "arm-c"] {
                gate.score_values(&format!("replay/first/{role}"), arm, &[0.0f32], &[0.0])?;
                gate.score_values(&format!("free/first/{role}"), arm, &[100.0f32], &[0.0])?;
            }
        }
        assert_eq!(gate.gating, 9);
        assert_eq!(gate.measurements, 9);
        assert!(gate.defects.is_empty());
        gate.gating += 3;
        gate.complete("first", false)?;
        gate.score_values(
            "replay/first/decoded-video-sample",
            "arm-a",
            &[100.0f32],
            &[0.0],
        )?;
        assert_eq!(
            gate.defects,
            vec!["replay/first/decoded-video-sample/arm-a"]
        );
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(&dir);
    check_bf16_modes?;
    Ok(())
}

#[cfg(test)]
#[derive(serde::Serialize, serde::Deserialize)]
struct ConditionerEffectFloor {
    tau_abs: f64,
    tau_rmse: f64,
}

#[cfg(test)]
fn conditioner_effect_floor(reference: [&[f32]; 3]) -> Result<ConditionerEffectFloor> {
    let mut floor = ConditionerEffectFloor {
        tau_abs: 1e-6,
        tau_rmse: 1e-6,
    };
    for (left, right) in [(0, 1), (0, 2), (1, 2)] {
        let m = crate::verify::metrics(
            "conditioner-effect-reference",
            reference[left],
            reference[right],
        )?;
        floor.tau_abs = floor.tau_abs.max(2.0 * m.max_abs);
        floor.tau_rmse = floor.tau_rmse.max(2.0 * m.rmse);
    }
    Ok(floor)
}

#[cfg(test)]
#[test]
fn conditioner_effect_freezes_reference_spread_and_requires_shift_detection() -> Result<()> {
    let a = [0.0, 1.0, 2.0, 3.0];
    let b = [0.0, 1.001, 2.0, 3.0];
    let e = [0.0, 0.999, 2.0, 3.0];
    let floor = conditioner_effect_floor([&a, &b, &e])?;
    let worst = crate::verify::metrics("reference-worst", &b, &e)?;
    assert_eq!(floor.tau_abs, 2.0 * worst.max_abs);
    assert_eq!(floor.tau_rmse, 2.0 * worst.rmse);
    let shifted = [3.0, 0.0, 1.0, 2.0];
    let control = crate::verify::metrics("shift", &shifted, &a)?;
    assert!(control.max_abs > 3.0 * floor.tau_abs || control.rmse > 3.0 * floor.tau_rmse);
    let omitted = crate::verify::metrics("omitted-shift", &a, &a)?;
    assert!(omitted.max_abs <= 3.0 * floor.tau_abs && omitted.rmse <= 3.0 * floor.tau_rmse);
    let outside = crate::verify::metrics("outside", &shifted, &e)?;
    assert!(outside.max_abs > floor.tau_abs && outside.rmse > floor.tau_rmse);
    Ok(())
}

#[cfg(test)]
fn conditioner_effect_measurements(
    case: &str,
    values: &[bf16],
    root: &Path,
    authorization: &serde_json::Value,
) -> Result<()> {
    let outliers: Vec<usize> =
        serde_json::from_value(authorization["conditioner_outlier_channels"].clone())?;
    ensure!(
        !outliers.is_empty()
            && outliers.len() < 5120
            && outliers.windows(2).all(|v| v[0] < v[1])
            && outliers.iter().all(|&v| v < 5120)
            && values.len().is_multiple_of(5120),
        "Runner FAIL conditioner measurement partition"
    );
    let rows = values.len() / 5120;
    for (part, is_outlier) in [("bulk", false), ("outlier", true)] {
        let channels: Vec<_> = (0..5120)
            .filter(|c| outliers.binary_search(c).is_ok() == is_outlier)
            .collect();
        let actual: Vec<_> = values
            .as_chunks::<5120>()
            .0
            .iter()
            .flat_map(|row| channels.iter().map(|&c| row[c]))
            .collect();
        let key = format!("producer/{case}/conditioner/{part}");
        let floor = &authorization["conditioner_measurement_floors"][&key];
        ensure!(
            floor["gating"] == false,
            "Runner FAIL image subcell is not a measurement"
        );
        let tau_abs = floor["tau_abs"].as_f64().context("measurement abs floor")?;
        let tau_rmse = floor["tau_rmse"]
            .as_f64()
            .context("measurement RMSE floor")?;
        for arm in ["arm-a", "arm-b", "arm-e"] {
            let full = composition_values(
                root,
                &format!("{arm}/producer-a/{case}/conditioner"),
                &[1, rows, 5120],
            )?;
            let expected: Vec<_> = full
                .as_chunks::<5120>()
                .0
                .iter()
                .flat_map(|row| channels.iter().map(|&c| row[c]))
                .collect();
            let m = crate::verify::metrics(&key, &actual, &expected)?;
            println!(
                "CONDITIONER_SUBCELL_MEASUREMENT cell={key} reference={arm} max_abs={} rmse={} tau_abs={tau_abs} tau_rmse={tau_rmse} gating=false",
                m.max_abs, m.rmse
            );
        }
    }
    Ok(())
}

#[cfg(test)]
#[test]
#[ignore = "GPU; merged MMA, explicit reference/native phase and composition fixtures"]
fn hyperflow_conditioner_effect_gpu() -> Result<()> {
    use crate::minimax_h3::{KeyframeAnchor, prepare_video};
    use crate::serve::image::RgbImage;
    let path = |key| {
        std::env::var_os(key)
            .map(std::path::PathBuf::from)
            .with_context(|| format!("{key} missing"))
    };
    let case = std::env::var("FF_HYPERFLOW_CASE").context("FF_HYPERFLOW_CASE missing")?;
    ensure!(
        matches!(case.as_str(), "first" | "last" | "both"),
        "Runner FAIL effect case"
    );
    let phase = std::env::var("FF_HYPERFLOW_EFFECT_PHASE").context("effect phase missing")?;
    ensure!(
        matches!(phase.as_str(), "reference" | "native"),
        "Runner FAIL effect phase"
    );
    let root = path("FF_HYPERFLOW_COMPOSITION")?;
    let out = path("FF_H3_CAPTURE")?;
    let model = path("FF_MODEL")?;
    let component = path("FF_HYPERFLOW_MODEL")?;
    let mma = std::env::var("FF_HYPERFLOW_MMA_COMMIT").context("merged MMA commit missing")?;
    ensure!(
        mma.len() == 40 && mma.bytes().all(|b| b.is_ascii_hexdigit()),
        "Runner FAIL MMA commit"
    );
    let ancestry = std::process::Command::new("git")
        .args([
            "-C",
            env!("CARGO_MANIFEST_DIR"),
            "merge-base",
            "--is-ancestor",
            &mma,
            "HEAD",
        ])
        .status()?;
    ensure!(
        ancestry.success(),
        "Runner FAIL MMA commit absent from source"
    );
    let spec = H3Spec::parse(&serde_json::from_slice(&std::fs::read(
        component.join("config.json"),
    )?)?)?;
    ensure!(spec.is_hyperflow()?, "Runner FAIL effect component");
    let image = |height, width, offset| RgbImage {
        height,
        width,
        pixels: (0..height * width * 3)
            .map(|i| ((i * 73 + offset) % 251) as u8)
            .collect(),
    };
    let (first, last) = (image(273, 481, 17), image(389, 267, 83));
    let anchors = match case.as_str() {
        "first" => vec![(KeyframeAnchor::First, &first)],
        "last" => vec![(KeyframeAnchor::Last, &first)],
        "both" => vec![
            (KeyframeAnchor::First, &first),
            (KeyframeAnchor::Last, &last),
        ],
        _ => unreachable!(),
    };
    let request = prepare_video(
        &model,
        &spec,
        "A small animal walks beside a wooden bridge in warm daylight.",
        &anchors,
        ((124, 42), (6, (12.0, 3.0))),
    )?;
    let (_, layout) = request.layout(&spec)?;
    let common = format!("common/a/{case}");
    let load = |directory: &Path, name: &str, shape: &[usize]| -> Result<Vec<f32>> {
        let result: Vec<_> = composition_values(directory, name, shape)?
            .into_iter()
            .map(|v| v as f32)
            .collect();
        ensure!(
            result.iter().all(|v| v.is_finite()),
            "Runner FAIL effect F32 input"
        );
        Ok(result)
    };
    let ids = composition_values(
        &root,
        &format!("{common}/input_ids"),
        &[1, request.token_ids.len()],
    )?;
    ensure!(
        ids.iter()
            .copied()
            .eq(request.token_ids.iter().map(|&v| f64::from(v))),
        "Runner FAIL effect ids"
    );
    let text_shape = [1, request.token_ids.len(), 5120];
    let reference = if phase == "native" {
        let directory = path("FF_H3_REPLAY")?;
        let authorization: serde_json::Value = serde_json::from_slice(&std::fs::read(
            directory.join("MEASUREMENT_AUTHORIZATION.json"),
        )?)?;
        ensure!(
            authorization["measurement_only"] == true
                && authorization["ruling"] == "2026-11-16T11:17:46Z"
                && authorization["reference_controls_passed"] == false
                && authorization["cases"] == serde_json::json!(["first", "last", "both"]),
            "Runner FAIL effect measurement authorization"
        );
        Some((directory.join(&case), authorization))
    } else {
        None
    };
    let runtime =
        std::env::temp_dir().join(format!("ff-conditioner-effect-{}", std::process::id()));
    std::fs::create_dir(&runtime)?;
    let result = (|| -> Result<bool> {
        let native_conditioner = if phase == "native" {
            let device = Device::new(0)?;
            let ops = Ops::new(&device.ctx)?;
            let mut blas = super::open_blas(
                &device,
                &ops,
                &runtime,
                Rc::new(Closure::new(false, "conditioner effect producer".into())),
            )?;
            Some(condition(&device, &mut blas, &model, &request, &runtime)?)
        } else {
            None
        };
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        let mut blas = super::open_blas(
            &device,
            &ops,
            &runtime,
            Rc::new(Closure::new(false, "conditioner effect".into())),
        )?;
        let save = |name: &str, shape: &[usize], values: Vec<f32>| -> Result<()> {
            let file = out.join(name).with_extension("safetensors");
            ensure!(!file.exists(), "Runner FAIL effect output already exists");
            let mut trace = crate::trace::Trace::default();
            trace.set_device(&device.info.name)?;
            trace.add("value".into(), shape.to_vec(), values)?;
            trace.save(&file)
        };
        let mut conditioning = Vec::new();
        if phase == "reference" {
            for arm in ["arm-a", "arm-b", "arm-e"] {
                let values = load(
                    &root,
                    &format!("{arm}/producer-a/{case}/conditioner"),
                    &text_shape,
                )?;
                ensure!(
                    values
                        .iter()
                        .all(|&v| bf16::from_f32(v).to_f32().to_bits() == v.to_bits()),
                    "Runner FAIL reference not BF16-exact"
                );
                conditioning.push((
                    arm,
                    values.into_iter().map(bf16::from_f32).collect::<Vec<_>>(),
                ));
            }
            let a = &conditioning[0].1;
            let shifted = a[a.len() - 5120..]
                .iter()
                .chain(&a[..a.len() - 5120])
                .copied()
                .collect();
            conditioning.push(("row-shift", shifted));
        } else {
            let values = native_conditioner.context("Runner FAIL missing native conditioning")?;
            let (_, authorization) = reference
                .as_ref()
                .context("Runner FAIL measurement authorization path")?;
            conditioner_effect_measurements(&case, &values, &root, authorization)?;
            save(
                "native-conditioner",
                &text_shape,
                values.iter().map(|v| v.to_f32()).collect(),
            )?;
            if case == "both" {
                println!(
                    "CONDITIONER_EFFECT_MEASUREMENT case=both row=1953 channel=731 native={} reference_range=140..161",
                    values[1953 * 5120 + 731]
                );
            }
            conditioning.push(("native", values));
        }
        let VideoTransformer {
            mut dit,
            table,
            adaln: plan,
            scale,
            ..
        } = prepare_transformer(
            (&device, &mut blas),
            (&component, None, None),
            (&spec, &layout),
            (request.evaluations, request.shifts),
            None,
        )?;
        let s = &device.stream;
        let kernels = BlockKernels::new(&device)?;
        let context = BlockContext {
            s,
            blas: &blas,
            kernels: &kernels,
            spec: &spec,
            lora_scale: scale,
        };
        let video = s.clone_htod(&load(
            &root,
            &format!("{common}/enter-video-0"),
            &[layout.video_indices.len(), 96],
        )?)?;
        let audio = s.clone_htod(&load(
            &root,
            &format!("{common}/enter-audio-0"),
            &[414, 32],
        )?)?;
        let row_pairs = s.clone_htod(&plan.row_pairs[0])?;
        let row_timesteps = s.clone_htod(&plan.row_timesteps[0])?;
        let mut velocities = std::collections::BTreeMap::new();
        for (name, values) in conditioning {
            let text = if name == "native" {
                s.clone_htod(&values)?
            } else {
                super::upload_runtime_half(
                    s,
                    device.info.arch.fp16_stored_half(),
                    "video_text",
                    &values,
                )?
            };
            dit.step(
                &context,
                &table,
                (
                    u32::try_from(plan.pairs.len())?,
                    u32::try_from(plan.timesteps.len())?,
                ),
                &super::StepInputs {
                    video: &video,
                    audio: &audio,
                    text: &text,
                    row_pairs: &row_pairs,
                    row_timesteps: &row_timesteps,
                },
            )?;
            let outputs = [s.clone_dtoh(&dit.video)?, s.clone_dtoh(&dit.audio)?];
            for (index, (role, shape)) in [("video", [37296, 96]), ("audio", [414, 32])]
                .into_iter()
                .enumerate()
            {
                save(&format!("{name}-{role}"), &shape, outputs[index].clone())?;
            }
            velocities.insert(name, outputs);
            println!(
                "CONDITIONER_EFFECT_FORWARD case={case} phase={phase} conditioning={name} step=0 complete=true utc={}",
                composition_stamp()?
            );
        }
        let mut defects = Vec::new();
        let mut floors = serde_json::Map::new();
        let mut controls = serde_json::Map::new();
        for (index, (role, shape)) in [("video", [37296, 96]), ("audio", [414, 32])]
            .into_iter()
            .enumerate()
        {
            if phase == "reference" {
                let samples =
                    ["arm-a", "arm-b", "arm-e"].map(|arm| velocities[arm][index].as_slice());
                let floor = conditioner_effect_floor(samples)?;
                let m = crate::verify::metrics(role, &velocities["row-shift"][index], samples[0])?;
                let ratios = [m.max_abs / floor.tau_abs, m.rmse / floor.tau_rmse];
                let detected = ratios.into_iter().any(|v| v > 3.0);
                println!(
                    "CONDITIONER_EFFECT_REFERENCE_CONTROL {} case={case} stream={role} factor=3 ratio_abs={} ratio_rmse={}",
                    if detected { "PASS" } else { "DEFECT" },
                    ratios[0],
                    ratios[1]
                );
                if !detected {
                    defects.push(role.to_owned());
                }
                floors.insert(role.into(), serde_json::to_value(floor)?);
                controls.insert(role.into(), serde_json::json!({"detected":detected,"factor":3,"ratio_abs":ratios[0],"ratio_rmse":ratios[1]}));
            } else {
                let (directory, authorization) = reference
                    .as_ref()
                    .context("Runner FAIL effect reference path")?;
                let frozen: serde_json::Value = serde_json::from_slice(&std::fs::read(
                    directory.join("REFERENCE_FLOORS.json"),
                )?)?;
                ensure!(
                    frozen["case"] == case
                        && frozen["formula"] == "max(2 * worst_valid_reference_pair, 1e-6)",
                    "Runner FAIL effect floor declaration"
                );
                let floor: ConditionerEffectFloor =
                    serde_json::from_value(frozen["cells"][role].clone())?;
                ensure!(
                    floor.tau_abs.is_finite()
                        && floor.tau_abs >= 1e-6
                        && floor.tau_rmse.is_finite()
                        && floor.tau_rmse >= 1e-6,
                    "Runner FAIL invalid effect floor"
                );
                let key = format!("conditioner-effect/{case}/{role}");
                let spread = &authorization["cells"][&key]["reference_spread"];
                let spread_abs = spread["max_abs"].as_f64().context("reference abs spread")?;
                let spread_rmse = spread["rmse"].as_f64().context("reference RMSE spread")?;
                ensure!(
                    spread_abs.is_finite()
                        && spread_abs > 0.0
                        && spread_rmse.is_finite()
                        && spread_rmse > 0.0
                        && floor.tau_abs == (2.0 * spread_abs).max(1e-6),
                    "Runner FAIL reference spread declaration"
                );
                for arm in ["arm-a", "arm-b", "arm-e"] {
                    let expected = load(directory, &format!("{arm}-{role}"), &shape)?;
                    let m = crate::verify::metrics(role, &velocities["native"][index], &expected)?;
                    println!(
                        "CONDITIONER_EFFECT_MEASUREMENT case={case} stream={role} reference={arm} max_abs={} rmse={} reference_spread_max_abs={spread_abs} reference_spread_rmse={spread_rmse} ratio_abs={} ratio_rmse={} gating=false",
                        m.max_abs,
                        m.rmse,
                        m.max_abs / spread_abs,
                        m.rmse / spread_rmse
                    );
                }
            }
        }
        if phase == "reference" {
            std::fs::write(
                out.join("REFERENCE_FLOORS.json"),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"case":case,"formula":"max(2 * worst_valid_reference_pair, 1e-6)","native_conditioning_run":false,"cells":floors}),
                )?,
            )?;
            std::fs::write(
                out.join("REFERENCE_CONTROLS.json"),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"case":case,"passed":defects.is_empty(),"streams":controls}),
                )?,
            )?;
        }
        std::fs::write(
            out.join("RESULT.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"case":case,"phase":phase,"result":if phase=="native"{"MEASUREMENT"}else if defects.is_empty(){"PASS"}else{"DEFECT"},"gating":false,"defects":defects}),
            )?,
        )?;
        Ok(defects.is_empty())
    })();
    let cleanup = std::fs::remove_dir_all(&runtime);
    let pass = result?;
    cleanup?;
    println!(
        "CONDITIONER_EFFECT_CASE_COMPLETE case={case} phase={phase} result={} utc={}",
        if phase == "native" {
            "MEASUREMENT"
        } else if pass {
            "PASS"
        } else {
            "DEFECT"
        },
        composition_stamp()?
    );
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
#[test]
fn video_placement_sizes_decoder_groups_from_remaining_memory() -> Result<()> {
    let spec = H3Spec::parse(&serde_json::from_str(include_str!(
        "../../../tests/data/minimax-h3-transformer-config.json"
    ))?)?;
    let vae = VaeSpec::parse(&serde_json::from_str(include_str!(
        "../../../tests/data/minimax-h3-vae-config.json"
    ))?)?;
    let geometry = Geometry::resolve(&spec, 768, 1344, 124)?;
    let decoder = VideoDecodeGeometry::new(&vae, &geometry)?;
    let slots = decoder.tiles.count();
    assert_eq!(slots, 28);
    assert_eq!(decoder.chunks, 7);
    assert_eq!(decoder.rgb_bytes, 383_975_424);
    let bytes = TileWork::batch_bytes(&vae, decoder.tile, slots, slots)?;
    assert_eq!(bytes, 4_442_630_708);
    let work = stored_allocation_bytes(bytes, 2 << 20)?;
    let select = |budget| VideoDecoder::work_batch(&vae, decoder.tile, slots, slots, budget);
    assert_eq!(select(work)?, (slots, bytes));
    let remaining = work - (11_013_382_144 - 10_806_362_112);
    let (batch, chosen) = select(remaining - remaining % (2 << 20))?;
    assert_eq!(batch, 26);
    assert!(stored_allocation_bytes(chosen, 2 << 20)? <= remaining);
    assert!(TileWork::batch_bytes(&vae, decoder.tile, batch + 1, slots)? > remaining);
    let one = TileWork::batch_bytes(&vae, decoder.tile, 1, slots)?;
    assert_eq!(select(one)?, (1, one));
    assert!(select(one - 1).is_err());
    let page = 2 << 20;
    let full = TileWork::charged_bytes(&vae, decoder.tile, slots, slots, page, decoder.rgb_bytes)?;
    assert_eq!(
        full,
        TileWork::charged_bytes(&vae, decoder.tile, slots, slots, page, 0)?
    );
    let ffn_bytes =
        4 * (decoder.tile.0 * decoder.tile.1 * decoder.tile.2 + vae.registers + 1) * vae.ffn;
    let odd_rgb = decoder.rgb_bytes + 1;
    assert_eq!(
        TileWork::charged_bytes(&vae, decoder.tile, 1, slots, 1, odd_rgb)? - one,
        2 * odd_rgb.div_ceil(2) - ffn_bytes
    );
    let visit = DecoderVisitCost {
        weight_bytes: 1 << 30,
        weight_seconds: 0.001,
        launches: 6 + 11 * vae.layers,
        launch_seconds: 0.000002,
        handoff_seconds: 0.000001,
    };
    let small = crate::prefill::DenseStage {
        device: &[16 << 20],
        source: &[16 << 20],
        visits: 1,
        load_scratch: 0,
        device_staging: 0,
    };
    let stages = [
        small,
        crate::prefill::DenseStage {
            device: &[64 << 20; 6],
            source: &[64 << 20; 6],
            visits: 6,
            ..small
        },
        small,
    ];
    let select_priced = |available| {
        visit.price_groups(
            (&vae, &decoder, slots),
            stages,
            (available, full, 1 << 30, page),
            (10e9, 1e9),
            2,
        )
    };
    let (batch, _, _, alternatives) = select_priced(full + (180 << 20))?;
    let price = |tiles| {
        alternatives
            .iter()
            .find(|p| p["batch"].as_u64() == Some(tiles))
            .and_then(|p| p["seconds"].as_f64())
            .context("missing decoder comparison")
    };
    assert!(price(27)? < price(28)?);
    assert!(price(batch as u64)? <= price(27)?);
    assert_eq!(select_priced(full + (1 << 30))?.0, 28);
    let total = 25_252_724_736;
    let budget = crate::prefill::DensePlacement::device_budget;
    let minimum = total / 16 + work + decoder.transient;
    assert_eq!(budget(minimum, total, 0, work, decoder.transient)?, 0);
    assert!(budget(minimum - 1, total, 0, work, decoder.transient).is_err());
    Ok(())
}

#[cfg(test)]
#[test]
fn packed_decoder_handoffs_preserve_video_patches_and_stereo_channels() -> Result<()> {
    use crate::minimax_h3::{KeyframeAnchor, TEXT_TAG};
    let spec = H3Spec::parse(&serde_json::from_str(include_str!(
        "../../../tests/data/minimax-h3-transformer-config.json"
    ))?)?;
    let mut vae = VaeSpec::parse(&serde_json::from_str(include_str!(
        "../../../tests/data/minimax-h3-vae-config.json"
    ))?)?;
    vae.latents_mean = (0..24).map(|c| c as f32 / 16.0).collect();
    vae.latents_std = (0..24).map(|c| 1.0 + c as f32 / 32.0).collect();
    let audio_spec = AudioVaeSpec {
        encoder_dim: 64,
        encoder_rates: vec![2, 4, 5, 5, 4],
        latent_dim: 64,
        latent_channels: 32,
        heads: 1,
        decoder_dim: 64,
        decoder_rates: vec![2, 4, 5, 5, 4],
        decoder_kernels: vec![4, 8, 10, 10, 8],
        resblock_kernels: vec![3],
        resblock_dilations: vec![vec![1]],
        sampling_rate: 32000,
        latents_mean: (0..32).map(|c| c as f32 / 8.0).collect(),
        latents_std: (0..32).map(|c| 1.0 + c as f32 / 16.0).collect(),
    };
    for (height, width, anchors) in [
        (32, 64, vec![]),
        (32, 64, vec![KeyframeAnchor::First]),
        (32, 64, vec![KeyframeAnchor::Last]),
        (768, 1344, vec![KeyframeAnchor::First, KeyframeAnchor::Last]),
    ] {
        let geometry = Geometry::resolve(&spec, height, width, 124)?;
        let layout = PackedLayout::build(&spec, &geometry, &[TEXT_TAG; 7], &anchors)?;
        let (h, w, frames) = (
            geometry.latent_height,
            geometry.latent_width,
            geometry.latent_frames,
        );
        let signal = |i: usize| ((i * 73 + 17) % 251) as f32 / 32.0 - 4.0;
        let count = spec.in_channels * frames * h * w;
        let input: Vec<f32> = (0..count).map(signal).collect();
        let prefix = layout.condition_video_rows * spec.video_row_width();
        let mut packed = vec![12345.0; prefix];
        patchify_latents_into(
            &input,
            (spec.in_channels, frames, h, w),
            spec.patch,
            &mut packed,
        );
        for (row, patch) in packed[prefix..].as_chunks::<96>().0.iter().enumerate() {
            let (f, spatial) = (row / (h / 2 * (w / 2)), row % (h / 2 * (w / 2)));
            let (y, x) = (spatial / (w / 2) * 2, spatial % (w / 2) * 2);
            for (column, &value) in patch.iter().enumerate() {
                let (c, p) = (column / 4, column % 4);
                assert_eq!(
                    value,
                    signal(((c * frames + f) * h + y + p / 2) * w + x + p % 2)
                );
            }
        }
        let audio: Vec<f32> = (0..2 * geometry.audio_latents * 32).map(signal).collect();
        let (video, pcm) = decoder_latents(
            (&spec, &vae, &audio_spec),
            (&geometry, &layout),
            (&packed, &audio),
        )?;
        assert_eq!(video.len(), input.len());
        let plane = frames * h * w;
        let mut channel_control = 0;
        for (i, &value) in video.iter().enumerate() {
            let c = i / plane;
            assert_eq!(value, signal(i) * vae.latents_std[c] + vae.latents_mean[c]);
            let wrong = (c + 1) % 24;
            channel_control +=
                usize::from(value != signal(i) * vae.latents_std[wrong] + vae.latents_mean[wrong]);
        }
        let mut stereo_control = 0;
        for (i, &value) in pcm.iter().enumerate() {
            let (channel, t) = (i / geometry.audio_latents, i % geometry.audio_latents);
            let (stereo, c) = (channel / 32, channel % 32);
            let index = (stereo * geometry.audio_latents + t) * 32 + c;
            assert_eq!(
                value,
                signal(index) * audio_spec.latents_std[c] + audio_spec.latents_mean[c]
            );
            let wrong = (t * 2 + stereo) * 32 + c;
            stereo_control += usize::from(
                value != signal(wrong) * audio_spec.latents_std[c] + audio_spec.latents_mean[c],
            );
        }
        assert!(channel_control > 0 && stereo_control > 0);
        assert!(
            decoder_latents(
                (&spec, &vae, &audio_spec),
                (&geometry, &layout),
                (&packed[..packed.len() - 1], &audio),
            )
            .is_err()
        );
    }
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
#[test]
#[ignore = "requires CUDA GPU"]
fn rgb_pinned_chunked_readback_matches_clone_dtoh() -> Result<()> {
    use crate::backend::gpu::DevicePtr;
    for len in [0usize, 1, 4095, 4096, 4097, 200_003] {
        let zeros = pre_fault_bytes(len);
        assert_eq!(zeros.len(), len);
        assert!(zeros.iter().all(|&b| b == 0));
    }
    let device = Device::new(0)?;
    let untimed = Some(crate::backend::gpu::EVENT_DISABLE_TIMING);
    let ready = device.ctx.new_event(untimed)?;
    let done = [
        device.ctx.new_event(untimed)?,
        device.ctx.new_event(untimed)?,
    ];
    let mut staging = crate::backend::cuda::copy::pinned_zeros::<u8>(&device.ctx, 65_536)?;
    for &len in &[0usize, 17_389, 131_072, 200_003] {
        let host: Vec<u8> = (0..len).map(|i| ((i * 73 + 19) % 251) as u8).collect();
        let gpu = if len == 0 {
            device.stream.alloc_zeros::<u8>(1)?
        } else {
            device.stream.clone_htod(&host)?
        };
        ready.record(&device.stream)?;
        let (ptr, _guard) = gpu.device_ptr(&device.upload);
        device.upload.wait(&ready)?;
        let mut out = pre_fault_bytes(len);
        readback_pinned_chunks(
            &device.upload,
            ptr,
            &mut out,
            staging.as_mut_slice()?,
            &done,
        )?;
        drop(_guard);
        let reference = if len == 0 {
            Vec::new()
        } else {
            device.stream.clone_dtoh(&gpu)?
        };
        assert_eq!(out, reference, "pinned readback mismatch for len={len}");
        if len == 200_003 {
            let mut shifted = pre_fault_bytes(len - 1);
            let (ptr_shift, _guard_shift) = gpu.device_ptr(&device.upload);
            readback_pinned_chunks(
                &device.upload,
                ptr_shift + 1,
                &mut shifted,
                staging.as_mut_slice()?,
                &done,
            )?;
            drop(_guard_shift);
            assert_ne!(shifted, reference[..len - 1], "offset mutant must fail");
        }
    }
    let mut tiny_staging = crate::backend::cuda::copy::pinned_zeros::<u8>(&device.ctx, 1)?;
    let pair = device.stream.clone_htod(&[1u8, 2u8])?;
    let (ptr, _guard) = pair.device_ptr(&device.upload);
    let mut two = [0u8; 2];
    assert!(
        readback_pinned_chunks(
            &device.upload,
            ptr,
            &mut two,
            tiny_staging.as_mut_slice()?,
            &done,
        )
        .is_err()
    );
    drop(_guard);
    Ok(())
}
