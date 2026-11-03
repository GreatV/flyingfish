//! MiniMax-H3 i2va generation on one GPU: conditioner, keyframe encoder, transformer chain, then the video and audio
//! decoders after the transformer and conditioner are released. Stages run in that order and each stage's weights are
//! dropped before the next stage's placement is derived from the then-free device memory.
use super::audio::AudioDecoder;
use super::vae::{
    EncodeWork, KeyframeEncoder, LatentVideo, TileWork, VideoDecoder, bf16_parameter,
};
use super::{
    AdalnTable, BlockContext, BlockKernels, Conditioner, ConditionerInput, ConditionerPrompt, Dit,
    Placement, Source, conditioner_embeds, conditioner_rope, embedding_rows, image_runs,
};
use crate::backend::cuda::{
    Device, blas::Blas, closure::Closure, ops::Ops, weights::WeightCatalog,
};
use crate::checkpoint::{LoadPolicy, MappedCheckpoint, TensorManifest};
use crate::minimax_h3::{
    AdalnPlan, AudioVaeSpec, Geometry, H3Output, H3Request, H3Spec, KEYFRAME_NOISE_AUG,
    KeyframeAnchor, PackedLayout, Schedule, TEXT_ENCODER_LAYER, VAE_PIXEL_MEAN, VAE_PIXEL_STD,
    VaeSpec, VaeTiles, row_timestep_plan,
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

/// Rows `[(f / pt) * (h / ph) * (w / pw), channels * pt * ph * pw]` of `x` `[channels, frames, h, w]`: frame-major then
/// row-major, columns (channel, patch t, patch h, patch w).
fn patchify(
    x: &[f32],
    (channels, frames, h, w): (usize, usize, usize, usize),
    patch: [usize; 3],
) -> Vec<f32> {
    let [pt, ph, pw] = patch;
    let mut rows = Vec::with_capacity(x.len());
    for f in 0..frames / pt {
        for i in 0..h / ph {
            for j in 0..w / pw {
                for c in 0..channels {
                    for a in 0..pt {
                        for y in 0..ph {
                            for z in 0..pw {
                                rows.push(
                                    x[((c * frames + f * pt + a) * h + i * ph + y) * w
                                        + j * pw
                                        + z],
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    rows
}

/// Inverse of `patchify`.
fn unpatchify(
    rows: &[f32],
    (channels, frames, h, w): (usize, usize, usize, usize),
    patch: [usize; 3],
) -> Vec<f32> {
    let [pt, ph, pw] = patch;
    let mut x = vec![0f32; rows.len()];
    let mut next = rows.iter();
    for f in 0..frames / pt {
        for i in 0..h / ph {
            for j in 0..w / pw {
                for c in 0..channels {
                    for a in 0..pt {
                        for y in 0..ph {
                            for z in 0..pw {
                                x[((c * frames + f * pt + a) * h + i * ph + y) * w + j * pw + z] =
                                    *next.next().expect("rows cover the patch grid");
                            }
                        }
                    }
                }
            }
        }
    }
    x
}

/// Conditioner stage: the Qwen3-VL vision tower and decoder layers `0..TEXT_ENCODER_LAYER` (both released on
/// return) map the prompt to the transformer's BF16 text rows `[rows, hidden]`.
fn condition(
    device: &Device,
    blas: &mut Blas,
    model: &Path,
    request: &H3Request,
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
    let prompt = ConditionerPrompt {
        modality: request
            .token_ids
            .iter()
            .map(|&id| u8::from(id == image_token))
            .collect(),
        tokens: request.token_ids.clone(),
        grids: request.grids.clone(),
        pixels: request.pixel_values.clone(),
    };
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
    let count = prompt.tokens.len();
    let media = super::super::vision_runtime::LayerNormTower::request_bytes(&spec, &input, count)?;
    let mut conditioner = Conditioner::new(
        device,
        &spec,
        (TEXT_ENCODER_LAYER, count, media),
        Source::new(&encoder, &manifest),
        Placement::Derived,
    )?;
    let prepared = tower.prepare(device, blas, &input, count, 0)?;
    let encoded = tower.encode(&device.stream, blas, prepared, None)?;
    let merged = device.stream.clone_dtoh(&encoded.features)?;
    conditioner.prepare(blas, count)?;
    let embeds = conditioner_embeds(&prompt, image_token, &table, &merged, hidden)?;
    let runs = image_runs(&prompt.tokens, image_token);
    let rope = conditioner_rope(&spec, &prompt)?;
    let mut taps = conditioner.forward(
        &device.stream,
        blas,
        &ConditionerInput {
            embeds: &embeds,
            deepstack: &encoded.deepstack,
            runs: &runs,
            rope: &rope,
        },
        &[TEXT_ENCODER_LAYER],
    )?;
    taps.pop().context("conditioner returned no tap")
}

/// Keyframe stage: the video VAE encoder (released on return) maps the keyframe to its normalised latents
/// `[latent_channels, height / r, width / r]`.
fn keyframe(device: &Device, blas: &Blas, model: &Path, request: &H3Request) -> Result<Vec<f32>> {
    let s = &device.stream;
    let spec = VaeSpec::read(model)?;
    let vae_dir = model.join("vae");
    let manifest = MappedCheckpoint::open_component(&vae_dir, "diffusion_pytorch_model")?
        .minimax_h3_vae_manifest(&spec)?;
    let encoder = KeyframeEncoder::new(device, &spec, &vae_dir, &manifest)?;
    let tiles = VaeTiles::released(&spec, request.height, request.width)?;
    let mut work = EncodeWork::new(s, &encoder, (tiles.height, tiles.width))?;
    let frame = s.clone_htod(&request.keyframe)?;
    let eps = s.clone_htod(&request.posterior_eps)?;
    let [_, _, latents] = encoder.encode(
        (s, blas),
        &mut work,
        &tiles,
        (&frame, request.height, request.width),
        &eps,
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
pub(crate) fn run(
    model: &Path,
    lora: &Path,
    request: &H3Request,
    (index, runtime): (usize, &Path),
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
    let text = condition(&device, &mut blas, model, request)?;
    let anchor = keyframe(&device, &blas, model, request)?;

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
    let latent_frames = request.video_noise.len() / (lc * lh * lw);
    let patch = spec.patch;
    let condition_rows = {
        let t = KEYFRAME_NOISE_AUG as f32;
        let mixed: Vec<f32> = anchor
            .iter()
            .zip(&request.anchor_noise)
            .map(|(&x, &n)| t * x + (1.0 - t) * n)
            .collect();
        patchify(&mixed, (lc, 1, lh, lw), [1, patch[1], patch[2]])
    };
    let width = spec.video_row_width();
    ensure!(
        anchor.len() == request.anchor_noise.len()
            && condition_rows.len() == layout.condition_video_rows * width
            && request.video_noise.len() == lc * latent_frames * lh * lw
            && (layout.video_indices.len() - layout.condition_video_rows) * width
                == request.video_noise.len()
            && layout.audio_indices.len() * spec.audio_in_channels == request.audio_rows.len()
            && request.text_tags.len() == text.len() / spec.text_dim,
        "request does not match the packed layout: video noise {}, audio rows {}, text rows {}",
        request.video_noise.len(),
        request.audio_rows.len(),
        request.text_tags.len()
    );
    let mut video_rows = condition_rows;
    video_rows.extend(patchify(
        &request.video_noise,
        (lc, latent_frames, lh, lw),
        patch,
    ));

    let transformer = model.join("transformer");
    let checkpoint = MappedCheckpoint::open_component(&transformer, "diffusion_pytorch_model")?;
    let manifest = checkpoint.minimax_h3_manifest(&spec)?;
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
            &transformer,
            &checkpoint,
            &manifest,
            &spec,
            &adaln_plan,
        )?;
        let mut dit = Dit::new(
            &device,
            &mut blas,
            &spec,
            (adapter.rank, &layout),
            Source::new(&transformer, &manifest),
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
        let audio = s.clone_htod(&request.audio_rows)?;
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
    let mut video_latents = unpatchify(generated, (lc, latent_frames, lh, lw), patch);
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
    let (free_after, total) = device.ctx.mem_get_info()?;
    let planned = decoder_bytes(&vae_manifest, &audio_manifest);
    eprintln!(
        "{}",
        serde_json::json!({"h3_release": {"free_before_release_bytes": free_before, "free_after_release_bytes": free_after,
            "decoder_weight_bytes": planned, "total_bytes": total, "placement": "resident"}})
    );
    ensure!(
        planned < free_after,
        "decoder weights {planned} B do not fit the {free_after} B free after releasing the transformer and conditioner"
    );
    let decoder = VideoDecoder::new(&device, &vae, &vae_dir, &vae_manifest)?;
    let tiles = VaeTiles::released(&vae, request.height, request.width)?;
    let clip = vae.chunk_latents() + vae.overlap_latents();
    let mut work = TileWork::new(s, &vae, (clip, tiles.height / r, tiles.width / r))?;
    work.prepare(s, &mut blas, &decoder)?;
    let decode_plan = crate::minimax_h3::DecodePlan::new(&vae, latent_frames)?;
    let (lead, pre) = (
        vae.chunk_latents() * vae.temporal_ratio,
        vae.frame_pre_padding(),
    );
    let total_frames = decode_plan.chunks * (lead - pre) + clip * vae.temporal_ratio - lead - pre;
    let pixels = request.height * request.width;
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
        frames == request.frames,
        "decoded {frames} frames, requested {}",
        request.frames
    );
    let decoded = s.clone_dtoh(&decoded)?;
    drop((work, decoder, latents_device));
    let audio_decoder = AudioDecoder::new(&device, &audio_spec, &audio_dir, &audio_manifest)?;
    let audio_device = s.clone_htod(&audio_latents)?;
    let pcm = s.clone_dtoh(&audio_decoder.decode((s, &blas), &audio_device, (2, latents))?)?;

    let (mean, std) = (
        VAE_PIXEL_MEAN.map(|v| v as f32),
        VAE_PIXEL_STD.map(|v| v as f32),
    );
    let channels = vae.out_channels;
    let mut video_u8 = vec![0u8; frames * pixels * channels];
    for c in 0..channels {
        for f in 0..frames {
            let base = (c * total_frames + f) * pixels;
            for p in 0..pixels {
                let v = (decoded[base + p].to_f32() * std[c] + mean[c]).clamp(0.0, 1.0);
                video_u8[(f * pixels + p) * channels + c] = (v * 255.0).round_ties_even() as u8;
            }
        }
    }
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
            vec![1, channels, total_frames, request.height, request.width],
            decoded,
        ),
    })
}
