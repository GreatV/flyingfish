//! MiniMax-H3 i2va generation on one GPU: conditioner, keyframe encoder, transformer chain, then the video and audio
//! decoders after the transformer and conditioner are released. Stages run in that order and each stage's weights are
//! dropped before the next stage's placement is derived from the then-free device memory.
use super::audio::AudioDecoder;
use super::vae::{
    EncodeWork, KeyframeEncoder, LatentVideo, TileWork, VideoDecoder, bf16_parameter, decode_bytes,
};
use super::{
    AdalnTable, BlockContext, BlockKernels, Conditioner, ConditionerInput, ConditionerPrompt, Dit,
    Placement, Source, conditioner_embeds, conditioner_rope, embedding_rows, image_runs,
};
use crate::backend::cuda::{
    Device, blas::Blas, closure::Closure, cubin, ops::Ops, weights::WeightCatalog,
};
use crate::backend::gpu::CudaSlice;
use crate::checkpoint::{LoadPolicy, MappedCheckpoint, TensorManifest};
use crate::minimax_h3::{
    AdalnPlan, AudioVaeSpec, Geometry, H3Output, H3Request, H3Spec, KEYFRAME_NOISE_AUG,
    KeyframeAnchor, PackedLayout, Schedule, TEXT_ENCODER_LAYER, VaeSpec, VaeTiles,
    row_timestep_plan,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::{path::Path, rc::Rc};

/// Generated media and, in test builds, the denormalised decoder inputs and the video decoder output (the PCM is the
/// audio decoder output) that the splice gate scores.
pub(crate) struct Generated {
    pub output: H3Output,
    #[cfg(test)]
    pub video_latents: (Vec<usize>, Vec<f32>),
    #[cfg(test)]
    pub audio_latents: (Vec<usize>, Vec<f32>),
    #[cfg(test)]
    pub video_decoded: (Vec<usize>, Vec<bf16>),
}

/// Conditioner stage: Qwen3-VL decoder layers `0..TEXT_ENCODER_LAYER` (released on return) map the prompt to the
/// transformer's BF16 text rows `[rows, hidden]`. A prompt with image tokens also loads the vision tower.
fn condition(
    device: &Device,
    blas: &mut Blas,
    model: &Path,
    request: &H3Request,
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
        &ConditionerPrompt {
            modality: request
                .token_ids
                .iter()
                .map(|&id| u8::from(id == image_token))
                .collect(),
            tokens: request.token_ids.clone(),
            grids: request.grids.clone(),
            pixels: request.pixel_values.clone(),
        },
    )
}

fn condition_prompt(
    device: &Device,
    blas: &mut Blas,
    model: &Path,
    prompt: &ConditionerPrompt,
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
    let hidden = spec.hidden as usize;
    let ids: Vec<u32> = prompt
        .tokens
        .iter()
        .copied()
        .filter(|&id| id != image_token)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let rows = embedding_rows(
        &mut Source::new(&encoder, &manifest),
        "model.language_model.embed_tokens.weight",
        hidden,
        &ids,
    )?;
    let table = ids
        .iter()
        .zip(rows.chunks(hidden))
        .map(|(&id, row)| (id, row.to_vec()))
        .collect();
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
                Source::new(&encoder, &manifest),
                Placement::Derived,
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
            Source::new(&encoder, &manifest),
            Placement::Derived,
        )?;
        let prepared = tower.prepare(device, blas, &input, count, 0)?;
        let encoded = tower.encode(&device.stream, blas, prepared, None)?;
        let merged = device.stream.clone_dtoh(&encoded.features)?;
        (conditioner, merged, Some(encoded))
    };
    conditioner.prepare(blas, count)?;
    let embeds = conditioner_embeds(prompt, image_token, &table, &merged, hidden)?;
    let rope = conditioner_rope(&spec, prompt)?;
    let deepstack = encoded.as_ref().map(|item| item.deepstack.as_slice());
    let mut taps = conditioner.forward(
        &device.stream,
        blas,
        &ConditionerInput {
            embeds: &embeds,
            deepstack: deepstack.unwrap_or(&[]),
            runs: &runs,
            rope: &rope,
        },
        &[TEXT_ENCODER_LAYER],
    )?;
    taps.pop().context("conditioner returned no tap")
}

/// Keyframe stage: the video VAE encoder (released on return) maps the keyframe to its normalised latents
/// `[latent_channels, height / r, width / r]`.
fn keyframe(
    device: &Device,
    blas: &Blas,
    model: &Path,
    request: &H3Request,
    eps: &CudaSlice<f32>,
) -> Result<Vec<f32>> {
    let s = &device.stream;
    let spec = VaeSpec::read(model)?;
    let vae_dir = model.join("vae");
    let manifest = MappedCheckpoint::open_component(&vae_dir, "diffusion_pytorch_model")?
        .minimax_h3_vae_manifest(&spec)?;
    let encoder = KeyframeEncoder::new(device, &spec, &vae_dir, &manifest)?;
    let tiles = VaeTiles::released(&spec, request.height, request.width)?;
    let mut work = EncodeWork::new(s, &encoder, (tiles.height, tiles.width))?;
    let frame = s.clone_htod(&request.keyframe)?;
    let [_, _, latents] = encoder.encode(
        (s, blas),
        &mut work,
        &tiles,
        (&frame, request.height, request.width),
        eps,
    )?;
    Ok(s.clone_dtoh(&latents)?)
}

/// Bytes the decoders hold resident: the video decoder side of `video` (BF16 set at two bytes per element) and every
/// loaded audio decoder tensor.
fn decoder_bytes(video: &TensorManifest, audio: &TensorManifest) -> usize {
    let video_bytes: usize = video
        .entries
        .iter()
        .filter(|(name, e)| {
            e.load && !name.starts_with("encoder.") && !name.starts_with("quant_conv.")
        })
        .map(|(name, e)| {
            if bf16_parameter(name) {
                e.bytes / 2
            } else {
                e.bytes
            }
        })
        .sum();
    video_bytes
        + audio
            .entries
            .values()
            .filter(|e| e.load)
            .map(|e| e.bytes)
            .sum::<usize>()
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
    run_mode::<false>(model, lora, request, (index, runtime), gguf)
}

#[cfg(test)]
pub(crate) fn decoder_setup(
    model: &Path,
    lora: &Path,
    request: &H3Request,
    runtime: &Path,
) -> Result<Generated> {
    run_mode::<true>(model, lora, request, (0, runtime), None)
}

fn run_mode<const SETUP_ONLY: bool>(
    model: &Path,
    lora: &Path,
    request: &H3Request,
    (index, runtime): (usize, &Path),
    gguf: Option<&Path>,
) -> Result<Generated> {
    let device = Device::new(index)?;
    let ops = Ops::new(&device.ctx)?;
    let s = &device.stream;
    let mut blas = Blas::new(
        &device,
        &ops,
        runtime,
        Rc::new(Closure::new(false, "MiniMax-H3 generate".into())),
    )?;
    let spec = H3Spec::read(model)?;
    let vae = VaeSpec::read(model)?;
    let audio_spec = AudioVaeSpec::read(model)?;
    let geometry = Geometry::resolve(&spec, request.height, request.width, request.frames)?;
    let layout = PackedLayout::build(
        &spec,
        &geometry,
        &request.text_tags,
        &[KeyframeAnchor::First],
    )?;
    let (r, lc) = (vae.spatial_ratio, vae.latent_channels);
    let (lh, lw) = (request.height / r, request.width / r);
    let plane = lh * lw;
    let latent_frames = geometry.latent_frames;
    let (anchor_count, video_count) = (lc * plane, lc * latent_frames * plane);
    let audio_count = layout.audio_indices.len() * spec.audio_in_channels;

    let normal = cubin::module(&device.ctx, "sample")?.load_function("philox_normal_f32")?;
    let draws = crate::minimax_h3::noise_draws(request.seed);
    let posterior_eps = super::draw_normals(s, &normal, draws[0], anchor_count, 256)?;
    let anchor_noise = {
        let draw = super::draw_normals(s, &normal, draws[1], anchor_count, 256)?;
        s.clone_dtoh(&draw)?
    };
    let video_noise = {
        let draw = super::draw_normals(s, &normal, draws[2], video_count, 256)?;
        s.clone_dtoh(&draw)?
    };
    let audio = super::draw_normals(s, &normal, draws[3], audio_count, 256)?;

    let text = condition(&device, &mut blas, model, request)?;
    let anchor = keyframe(&device, &blas, model, request, &posterior_eps)?;

    let patch = spec.patch;
    let condition_rows = {
        let t = KEYFRAME_NOISE_AUG as f32;
        let mixed: Vec<f32> = anchor
            .iter()
            .zip(&anchor_noise)
            .map(|(&x, &n)| t * x + (1.0 - t) * n)
            .collect();
        crate::minimax_h3::patchify_latents(&mixed, (lc, 1, lh, lw), [1, patch[1], patch[2]])
    };
    let width = spec.video_row_width();
    ensure!(
        anchor.len() == anchor_count
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
    video_rows.extend(crate::minimax_h3::patchify_latents(
        &video_noise,
        (lc, latent_frames, lh, lw),
        patch,
    ));

    let transformer = model.join("transformer");
    let manifest = match gguf {
        Some(path) => {
            let gguf = crate::gguf::GgufFile::open(path)?;
            MappedCheckpoint::minimax_h3_gguf_manifest(&spec, &gguf)?
        }
        None => MappedCheckpoint::open_component(&transformer, "diffusion_pytorch_model")?
            .minimax_h3_manifest(&spec)?,
    };
    let source_dir = match gguf {
        Some(path) => path.parent().context("GGUF directory")?.to_path_buf(),
        None => transformer.clone(),
    };
    let lora_dir = lora.parent().context("LoRA directory")?;
    let stem = lora
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.strip_suffix(".safetensors"))
        .context("LoRA file stem")?;
    let (lora_manifest, adapter) =
        MappedCheckpoint::open_component(lora_dir, stem)?.minimax_h3_lora_manifest(&spec)?;
    let video_grid = Schedule::new(request.evaluations + 1, request.shifts.0)?;
    let audio_grid = Schedule::new(request.evaluations + 1, request.shifts.1)?;
    let plan = row_timestep_plan(&layout, &video_grid, &audio_grid)?;
    let adaln_plan = AdalnPlan::build(&layout, &plan);
    let (video, audio, free_before) = {
        let table = AdalnTable::build(
            &device,
            &mut blas,
            &source_dir,
            &manifest,
            &spec,
            &adaln_plan,
        )?;
        let mut dit = Dit::new(
            &device,
            &mut blas,
            &spec,
            (adapter.rank, &layout),
            Source::new(&source_dir, &manifest),
            Source::new(lora_dir, &lora_manifest),
            Placement::Derived,
        )?;
        let kernels = BlockKernels::new(&device.ctx)?;
        let context = BlockContext {
            s,
            blas: &blas,
            kernels: &kernels,
            spec: &spec,
            lora_scale: adapter.scale,
        };
        let video = s.clone_htod(&video_rows)?;
        let text = s.clone_htod(&text)?;
        dit.sample(
            &context,
            (&table, &adaln_plan),
            (&video_grid, &audio_grid),
            (&video, &audio),
            &text,
        )?;
        let (video, audio) = (s.clone_dtoh(&video)?, s.clone_dtoh(&audio)?);
        (video, audio, device.ctx.mem_get_info()?.0)
    };
    s.synchronize()?;
    ensure!(
        video.iter().chain(&audio).all(|v| v.is_finite()),
        "non-finite final latents"
    );

    let generated = &video[layout.condition_video_rows * width..];
    let mut video_latents =
        crate::minimax_h3::unpatchify_latents(generated, (lc, latent_frames, lh, lw), patch);
    let plane = latent_frames * lh * lw;
    for (c, chunk) in video_latents.chunks_mut(plane).enumerate() {
        for v in chunk {
            *v = *v * vae.latents_std[c] + vae.latents_mean[c];
        }
    }
    let ac = audio_spec.latent_channels;
    let latents = audio.len() / (2 * ac);
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

    let vae_dir = model.join("vae");
    let vae_manifest = MappedCheckpoint::open_component(&vae_dir, "diffusion_pytorch_model")?
        .minimax_h3_vae_manifest(&vae)?;
    let audio_dir = model.join("audio_vae");
    let audio_manifest = MappedCheckpoint::open_component(&audio_dir, "diffusion_pytorch_model")?
        .minimax_h3_audio_vae_manifest(&audio_spec)?;
    let tiles = VaeTiles::released(&vae, request.height, request.width)?;
    let clip = vae.chunk_latents() + vae.overlap_latents();
    let tile = (clip, tiles.height / r, tiles.width / r);
    let decode_plan = crate::minimax_h3::DecodePlan::new(&vae, latent_frames)?;
    let (lead, pre) = (
        vae.chunk_latents() * vae.temporal_ratio,
        vae.frame_pre_padding(),
    );
    let total_frames = decode_plan.chunks * (lead - pre) + clip * vae.temporal_ratio - lead - pre;
    let pixels = request.height * request.width;
    let weights = decoder_bytes(&vae_manifest, &audio_manifest);
    let work_bytes = TileWork::bytes(&vae, tile)
        + decode_bytes(&vae, &tiles, (request.height, request.width))
        + 2 * vae.out_channels * total_frames * pixels
        + 4 * (video_latents.len() + audio_latents.len())
        + AudioDecoder::work_bytes(&audio_spec, (2, latents))?;
    let planned = weights + work_bytes;
    let (free_after, total) = device.ctx.mem_get_info()?;
    eprintln!(
        "{}",
        serde_json::json!({"h3_release": {"free_before_release_bytes": free_before, "free_after_release_bytes": free_after,
            "decoder_weight_bytes": weights, "decode_work_bytes": work_bytes, "planned_bytes": planned,
            "total_bytes": total, "placement": "resident"}})
    );
    ensure!(
        planned < free_after,
        "decoders {weights} B plus decode work {work_bytes} B do not fit the {free_after} B free after releasing the transformer and conditioner"
    );
    let decoder = VideoDecoder::new(&device, &vae, &vae_dir, &vae_manifest)?;
    #[cfg(test)]
    if SETUP_ONLY {
        decoder.batch_setup(&device, &mut blas, &tiles, (latent_frames, lh, lw))?;
        anyhow::bail!("H3_VAE_SETUP_COMPLETE");
    }
    let mut work = decoder.work(&device, &mut blas, &tiles, (latent_frames, lh, lw))?;
    let mut decoded = s.alloc_zeros::<bf16>(vae.out_channels * total_frames * pixels)?;
    let latents_device = s.clone_htod(&video_latents)?;
    let decode_plan = decoder.decode(
        s,
        &blas,
        &mut work,
        &tiles,
        LatentVideo {
            data: &latents_device,
            frames: latent_frames,
            height: lh,
            width: lw,
        },
        &mut decoded,
    )?;
    let frames = decode_plan.frames;
    ensure!(
        frames == geometry.frames,
        "decoded {frames} frames, resolved geometry has {}",
        geometry.frames
    );
    drop((work, latents_device));
    #[cfg(test)]
    let video_decoded = s.clone_dtoh(&decoded)?;
    let rgb = decoder.rgb(&device, &decoded, (frames, total_frames, pixels))?;
    let video_u8 = s.clone_dtoh(&rgb)?;
    drop((rgb, decoded, decoder));
    let audio_decoder = AudioDecoder::new(&device, &audio_spec, &audio_dir, &audio_manifest)?;
    let audio_device = s.clone_htod(&audio_latents)?;
    let pcm = s.clone_dtoh(&audio_decoder.decode((s, &blas), &audio_device, (2, latents))?)?;

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

fn image_condition(
    model: &Path,
    options: &crate::minimax_h3::ImageRunOptions,
    prompt: &ConditionerPrompt,
) -> Result<Vec<bf16>> {
    let device = Device::new(options.device)?;
    let ops = Ops::new(&device.ctx)?;
    let mut blas = Blas::new(
        &device,
        &ops,
        &options.runtime_dir,
        Rc::new(Closure::new(true, "MiniMax-H3-Image conditioner".into())),
    )?;
    condition_prompt(&device, &mut blas, model, prompt)
}

fn image_noise(
    options: &crate::minimax_h3::ImageRunOptions,
    seed: u64,
    count: usize,
) -> Result<Vec<f32>> {
    let device = Device::new(options.device)?;
    let normal = cubin::module(&device.ctx, "sample")?.load_function("philox_normal_f32")?;
    let draw = super::draw_normals(
        &device.stream,
        &normal,
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
    let spec = crate::minimax_h3::ImageSpec::read(model)?;
    let (canvas_h, canvas_w) = crate::minimax_h3::image_canvas(
        &spec,
        request.height,
        request.width,
        request.min_generation_area,
    )?;
    let (latent_h, latent_w) = crate::minimax_h3::image_latent_extent(&spec, canvas_h, canvas_w)?;
    let width = spec.row_width();
    let mut prepared = Vec::with_capacity(request.references.len());
    let mut noise_count = 0usize;
    for reference in &request.references {
        let (rgb, extent, posterior) = vae_rgb(&spec, &reference.image)?;
        let (lat_h, lat_w) =
            crate::minimax_h3::image_reference_latent_extent(&spec, extent[0], extent[1])?;
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
        None => image_noise(options, request.seed, noise_count)?,
    };
    let mut cursor = 0usize;
    let mut reference_rows = Vec::new();
    let mut reference_extents = Vec::with_capacity(prepared.len());
    let mut encoded = Vec::with_capacity(prepared.len());
    for (rgb, extent, posterior, lat_h, lat_w, cropped) in &prepared {
        let epsilon = &noise[cursor..cursor + posterior];
        cursor += posterior;
        let (latent, shape) = super::image_encode(
            model,
            (options.device, &options.runtime_dir),
            *extent,
            rgb,
            epsilon,
        )?;
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
    let (ids, tags, prompt_rows) = image_presentation(model, &spec, &prompt, &request.references)?;
    let layout =
        crate::minimax_h3::image_layout(&spec, &tags, &reference_extents, latent_h, latent_w)?;
    ensure!(
        ids.len() == tags.len() && ids.len() == layout.text_indices.len(),
        "image presentation rows"
    );
    let text = image_condition(model, options, &prompt_rows)?;
    ensure!(
        text.len() == ids.len() * spec.text_dim,
        "conditioner row width"
    );
    let generated = &noise[generated_at..mix_at];
    ensure!(
        reference_rows.len() == layout.reference_rows * width,
        "reference row count"
    );
    let mut rows = reference_rows.clone();
    rows.extend_from_slice(generated);
    let sampled = super::image_sample(
        model,
        &layout,
        (options.device, &options.runtime_dir),
        &rows,
        &text,
    )?;
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
    let canvas = super::image_decode(
        model,
        (options.device, &options.runtime_dir),
        [latent_h, latent_w],
        &values,
    )?;
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
