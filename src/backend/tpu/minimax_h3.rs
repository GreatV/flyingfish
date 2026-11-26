//! MiniMax-H3-Image end-to-end inference on TPU: Qwen3-VL conditioner, 50-block DiT, and 24-layer ViT VAE decoder.

use super::{
    TpuBuffer, TpuDevice,
    dit_block::{
        DitBlockStack, StackStepStats, TpuAdalnCache, compile_h3_dit_block,
        compute_all_adaln_tables, compute_norm_out_adaln_table, load_refiner_block_weights,
        read_bf16_ckpt, read_f32_ckpt, read_f32_narrow_bf16_ckpt, upload_block_weights,
    },
    hlo::{
        DitNormOutPddSpec, DitRefinerBlockHloSpec, Qwen3VlLayerHloSpec, VaeBlockHloSpec,
        VaePostHloSpec, VaePreHloSpec, emit_bf16_rms_norm, emit_h3_dit_context_embedder,
        emit_h3_dit_norm_out_pdd, emit_h3_dit_proj_in_pack, emit_h3_dit_refiner_block,
        emit_h3_image_vae_block, emit_h3_image_vae_post, emit_h3_image_vae_pre,
        emit_qwen3_vl_text_layer,
    },
    pjrt::HostSemantics,
    plan::{ModelFootprint, TpuPlacementPlan, TpuTopology},
};
use crate::{
    backend::setup::decoder_weight_is_raw_f32,
    checkpoint::MappedCheckpoint,
    minimax_h3::{
        AdalnPlan, ImageLatents, ImageLayout, ImageRequest, ImageRunOptions, ImageSchedule,
        ImageSpec, ImageVaeSpec, TEXT_ENCODER_LAYER, VAE_PIXEL_MEAN, VAE_PIXEL_STD, VaeTiles,
        image_canvas, image_latent_extent, image_layout, image_noise_draw, image_rgb8,
        image_row_plan, image_text_rows, mrope_table, rotary_rows, unpatchify_latents,
    },
    model_spec::{Architecture, ModelSpec},
    serve::detok::Shared as Tokenizer,
    vision_input::PositionPlan,
};
use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use safetensors::Dtype;
use serde::Serialize;
use std::{fs, path::Path, time::Instant};

const DIT_ACTIVE_BLOCK_BYTES_HEADROOM: u64 = 64 * 1024 * 1024;
const VAE_MAX_TILE_BATCH: usize = 16;

/// Optional overrides for discriminating gate controls and forced-tier verification.
#[derive(Clone, Debug, Default)]
pub struct DitSampleOptions {
    pub transpose_rope_hw: bool,
    pub force_pdd_head: Option<usize>,
    pub wrong_step0_timestep_index: Option<usize>,
    pub negate_euler_velocity: bool,
    pub stop_after_step0: bool,
    pub host_available_cap_bytes: Option<u64>,
}

/// Optional overrides for discriminating VAE gate controls.
#[derive(Clone, Debug, Default)]
pub struct VaeDecodeOptions {
    pub transpose_rope_hw: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ConditionerReport {
    pub prompt_rows: usize,
    pub hidden: usize,
    pub layers_run: usize,
    pub hbm_before_bytes: i64,
    pub hbm_after_drop_bytes: i64,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct DitSampleReport {
    pub total_rows: usize,
    pub text_rows: usize,
    pub image_rows: usize,
    pub head_chunk: usize,
    pub steps_run: usize,
    pub live_topology: TpuTopology,
    pub live_plan: TpuPlacementPlan,
    pub per_step_stats: Vec<StackStepStats>,
    #[serde(skip_serializing)]
    pub step0_velocity: Vec<f32>,
    pub hbm_before_bytes: i64,
    pub hbm_after_drop_bytes: i64,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct VaeDecodeReport {
    pub latent_extent: [usize; 2],
    pub canvas_extent: [usize; 2],
    pub tile_count: usize,
    pub tile_batch: usize,
    pub hbm_before_bytes: i64,
    pub hbm_after_drop_bytes: i64,
    pub elapsed_ms: f64,
}

/// Runs the Qwen3-VL text conditioner through `TEXT_ENCODER_LAYER` (50) on TPU, returning `[rows, 5120]` BF16.
pub fn run_conditioner_with_device(
    dev: &TpuDevice,
    model: &Path,
    token_ids: &[u32],
) -> Result<(Vec<bf16>, ConditionerReport)> {
    ensure!(!token_ids.is_empty(), "conditioner token_ids is empty");
    let t0 = Instant::now();
    let hbm_before_bytes = dev.memory_stats()?.bytes_in_use;
    let text_dir = model.join("text_encoder");
    let spec = ModelSpec::read(&text_dir)?;
    ensure!(
        spec.architecture == Architecture::Qwen3Vl && spec.layers.len() > TEXT_ENCODER_LAYER,
        "expected Qwen3-VL text_encoder with > {TEXT_ENCODER_LAYER} layers"
    );
    let mrope_section = spec
        .vision
        .as_ref()
        .and_then(|v| v.mrope_section)
        .context("Qwen3-VL mrope_section missing")?;

    let ckpt = MappedCheckpoint::open_component_shards(&text_dir, "model", |name| {
        conditioner_text_tensor_needed(name, TEXT_ENCODER_LAYER)
    })?;
    let _manifest = ckpt.minimax_h3_conditioner_text_manifest(&spec, TEXT_ENCODER_LAYER)?;

    let rows = token_ids.len();
    let hidden = spec.hidden as usize;
    let q_heads = spec.q_heads as usize;
    let kv_heads = spec.kv_heads as usize;
    let head_dim = spec.head_dim as usize;
    let intermediate = spec.intermediate as usize;

    let embeds = lookup_bf16_embeddings(&ckpt, token_ids, spec.vocab as usize, hidden)?;
    let pos_plan = PositionPlan::text(rows, rows)?;
    let rope_f32 = mrope_table(&pos_plan, mrope_section, spec.rope_theta, spec.head_dim)?;
    let rope_bf16: Vec<bf16> = rope_f32.into_iter().map(bf16::from_f32).collect();

    let head_chunk = if rows <= 512 { 8 } else { 4 };
    let layer_mlir = emit_qwen3_vl_text_layer(&Qwen3VlLayerHloSpec {
        rows,
        hidden,
        heads: q_heads,
        kv_heads,
        head_dim,
        intermediate,
        head_chunk,
        rms_norm_eps: spec.rms_eps,
    })?;
    let exec = dev.compile_mlir(&layer_mlir)?;
    let sem = HostSemantics::ImmutableOnlyDuringCall;
    let mut x = dev.upload_bf16(&embeds, &[rows as i64, hidden as i64], sem)?;
    let rope = dev.upload_bf16(&rope_bf16, &[rows as i64, head_dim as i64], sem)?;
    let out_bytes = rows * hidden * 2;

    let mut cur_weights = stage_qwen_layer_to_hbm(dev, &ckpt, &spec, 0)?;
    for layer in 0..TEXT_ENCODER_LAYER {
        let mut args: Vec<&TpuBuffer> = Vec::with_capacity(13);
        args.push(&x);
        args.push(&rope);
        for wb in &cur_weights {
            args.push(wb);
        }
        let (mut outs, ev) = exec.execute_async(&args, &[out_bytes])?;
        let next_weights = if layer + 1 < TEXT_ENCODER_LAYER {
            Some(stage_qwen_layer_to_hbm(dev, &ckpt, &spec, layer + 1)?)
        } else {
            None
        };
        ev.wait()
            .with_context(|| format!("Qwen3-VL layer {layer} execute"))?;
        x = outs.pop().context("Qwen3-VL layer returned zero outputs")?;
        drop(cur_weights);
        cur_weights = next_weights.unwrap_or_default();
    }
    let output = x.download_bf16()?;
    ensure!(
        output.iter().all(|v| v.is_finite()),
        "Qwen3-VL conditioner output contains non-finite value"
    );
    drop((x, rope, exec, ckpt));
    let hbm_after_drop_bytes = dev.memory_stats()?.bytes_in_use;
    Ok((
        output,
        ConditionerReport {
            prompt_rows: rows,
            hidden,
            layers_run: TEXT_ENCODER_LAYER,
            hbm_before_bytes,
            hbm_after_drop_bytes,
            elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
        },
    ))
}

/// Six published image evaluations. `rows` is `[image rows, 96]`, matching CUDA `image_sample`.
pub fn image_sample(
    model: &Path,
    layout: &ImageLayout,
    (device, _runtime): (usize, &Path),
    rows: &[f32],
    text: &[bf16],
) -> Result<Vec<f32>> {
    let dev = TpuDevice::open_default(device)?;
    let (sampled, _) = image_sample_with_device(
        &dev,
        model,
        layout,
        rows,
        text,
        &DitSampleOptions::default(),
    )?;
    Ok(sampled)
}

/// Six published image evaluations on an open [`TpuDevice`] with telemetry and gate control hooks.
pub fn image_sample_with_device(
    dev: &TpuDevice,
    model: &Path,
    layout: &ImageLayout,
    rows: &[f32],
    text: &[bf16],
    opts: &DitSampleOptions,
) -> Result<(Vec<f32>, DitSampleReport)> {
    let t0 = Instant::now();
    let hbm_before_bytes = dev.memory_stats()?.bytes_in_use;
    ensure!(
        layout.reference_rows == 0,
        "TPU image sampling does not yet support edit reference rows ({})",
        layout.reference_rows
    );
    let spec = ImageSpec::read(model)?;
    let schedule = ImageSchedule::published()?;
    let patch_dim = spec.row_width();
    let text_rows = layout.text_indices.len();
    let img_rows = layout.image_indices.len();
    let total_rows = layout.rows();
    ensure!(
        text_rows > 0 && img_rows > 0 && text_rows + img_rows == total_rows,
        "invalid image layout counts: text={text_rows} img={img_rows} total={total_rows}"
    );
    ensure!(
        layout
            .text_indices
            .iter()
            .enumerate()
            .all(|(i, &idx)| idx as usize == i)
            && layout
                .image_indices
                .iter()
                .enumerate()
                .all(|(i, &idx)| idx as usize == text_rows + i),
        "TPU image layout requires contiguous [text | image] row ordering"
    );
    ensure!(
        rows.len() == img_rows * patch_dim
            && text.len() == text_rows * spec.text_dim
            && rows.iter().all(|v| v.is_finite())
            && text.iter().all(|v| v.is_finite()),
        "image sample input extents or finiteness invalid"
    );
    if let Some(head) = opts.force_pdd_head {
        ensure!(head < spec.pdd_heads, "force_pdd_head {head} out of range");
    }
    if let Some(ts_step) = opts.wrong_step0_timestep_index {
        ensure!(
            ts_step < spec.pdd_heads,
            "wrong_step0_timestep_index {ts_step} out of range"
        );
    }

    let dir = model.join("transformer_diffusers");
    let ckpt = MappedCheckpoint::open_component(&dir, "diffusion_pytorch_model")?;
    let _manifest = ckpt.minimax_h3_image_manifest(&spec)?;

    let row_plan = image_row_plan(layout, &schedule)?;
    let adaln_plan = AdalnPlan::from_tags(&layout.token_tags, &row_plan)?;

    // 1. Run context_embedder + token_refiner once before the denoising loop.
    let refined_text_hbm = run_text_refiner_hbm(dev, &spec, &ckpt, text, text_rows)?;

    // 2. Setup-time schedule-keyed AdaLN tables for all 50 blocks + norm_out.
    let mut adaln_cache = TpuAdalnCache::default();
    adaln_cache.ensure_all_for_plan(&adaln_plan, |p| {
        compute_all_adaln_tables(dev, &spec, &ckpt, p)
    })?;
    let sem = HostSemantics::ImmutableOnlyDuringCall;
    let norm_out_adaln_hbm = dev.upload_bf16(
        adaln_cache.norm_out(),
        &[adaln_plan.timesteps.len() as i64, 2, spec.hidden as i64],
        sem,
    )?;

    // 3. Compile DiT executables and upload small persistent weights & step index tables.
    let head_chunk = if total_rows <= 1024 { 4 } else { 2 };
    let block_exec =
        compile_h3_dit_block(dev, &spec, total_rows, adaln_plan.pairs.len(), head_chunk)?;
    let proj_in_exec = dev.compile_mlir(&emit_h3_dit_proj_in_pack(
        text_rows,
        img_rows,
        patch_dim,
        spec.hidden,
    )?)?;
    let norm_out_pdd_exec = dev.compile_mlir(&emit_h3_dit_norm_out_pdd(&DitNormOutPddSpec {
        rows: total_rows,
        text_rows,
        hidden: spec.hidden,
        patch_dim,
        timesteps: adaln_plan.timesteps.len(),
        final_norm_eps: spec.final_norm_eps,
    })?)?;

    let proj_in_w = dev.upload_f32(
        &read_f32_ckpt(&ckpt, "proj_in.weight", &[spec.hidden, patch_dim])?,
        &[spec.hidden as i64, patch_dim as i64],
        sem,
    )?;
    let proj_in_b = dev.upload_f32(
        &read_f32_ckpt(&ckpt, "proj_in.bias", &[spec.hidden])?,
        &[spec.hidden as i64],
        sem,
    )?;
    let norm_out_w = dev.upload_bf16(
        &read_bf16_ckpt(&ckpt, "norm_out.norm.weight", &[spec.hidden])?,
        &[spec.hidden as i64],
        sem,
    )?;
    let mut pdd_w = Vec::with_capacity(spec.pdd_heads);
    let mut pdd_b = Vec::with_capacity(spec.pdd_heads);
    for h in 0..spec.pdd_heads {
        pdd_w.push(dev.upload_f32(
            &read_f32_ckpt(
                &ckpt,
                &format!("pdd_proj_out.{h}.weight"),
                &[patch_dim, spec.hidden],
            )?,
            &[patch_dim as i64, spec.hidden as i64],
            sem,
        )?);
        pdd_b.push(dev.upload_f32(
            &read_f32_ckpt(&ckpt, &format!("pdd_proj_out.{h}.bias"), &[patch_dim])?,
            &[patch_dim as i64],
            sem,
        )?);
    }

    let positions: Vec<[f64; 3]> = if opts.transpose_rope_hw {
        layout
            .position_ids
            .iter()
            .map(|&[t, h, w]| [t, w, h])
            .collect()
    } else {
        layout.position_ids.clone()
    };
    let rotary_dim = 6 * spec.rope_freq_dim;
    let rotary_data = rotary_rows(spec.rope_freq_dim, spec.rope_theta, &positions);
    let rotary_hbm = dev.upload_bf16(&rotary_data, &[total_rows as i64, rotary_dim as i64], sem)?;

    let mut step_row_pairs_hbm = Vec::with_capacity(spec.pdd_heads);
    let mut step_row_timesteps_hbm = Vec::with_capacity(spec.pdd_heads);
    for step in 0..spec.pdd_heads {
        let pairs_i32: Vec<i32> = adaln_plan.row_pairs[step]
            .iter()
            .map(|&v| i32::try_from(v).context("row_pair overflows i32"))
            .collect::<Result<Vec<_>>>()?;
        let ts_i32: Vec<i32> = adaln_plan.row_timesteps[step]
            .iter()
            .map(|&v| i32::try_from(v).context("row_timestep overflows i32"))
            .collect::<Result<Vec<_>>>()?;
        step_row_pairs_hbm.push(dev.upload_i32(&pairs_i32, &[total_rows as i64], sem)?);
        step_row_timesteps_hbm.push(dev.upload_i32(&ts_i32, &[total_rows as i64], sem)?);
    }

    // 4. Derive live placement plan from sensed HBM, compiled block memory, and host RAM.
    let mut live_topology = TpuTopology::detect(dev, None)?;
    if let Some(cap) = opts.host_available_cap_bytes {
        live_topology.host_available_bytes = live_topology.host_available_bytes.min(cap);
    }
    let active_block_bytes = active_dit_block_bytes(&spec);
    let compiled_block_peak = block_exec
        .compiled_memory_stats
        .as_ref()
        .map(|s| u64::try_from(s.peak_memory_in_bytes.max(0)).unwrap_or(0))
        .filter(|&v| v > active_block_bytes)
        .unwrap_or(active_block_bytes + 640 * 1024 * 1024);
    let adaln_tables_bytes = (spec.layers * adaln_plan.pairs.len() * 6 * spec.hidden * 2) as u64;
    let persistent_hbm_bytes = adaln_tables_bytes
        + DIT_ACTIVE_BLOCK_BYTES_HEADROOM
        + (total_rows * spec.hidden * 2 * 2) as u64;
    let footprint = ModelFootprint {
        persistent_hbm_bytes,
        compiled_block_peak_bytes: compiled_block_peak,
        compiled_block_weight_arg_bytes: active_block_bytes,
        block_weight_bytes: vec![active_block_bytes; spec.layers],
    };
    let live_plan = TpuPlacementPlan::derive(&live_topology, &footprint)?;
    let stack = DitBlockStack::load(
        dev,
        &spec,
        &ckpt,
        live_plan.clone(),
        adaln_cache.layers(),
        adaln_plan.pairs.len(),
    )?;

    // 5. Six-step denoising loop.
    let mut cur_latents = rows.to_vec();
    let mut step0_velocity = Vec::new();
    let steps_to_run = if opts.stop_after_step0 {
        1
    } else {
        spec.pdd_heads
    };
    let mut per_step_stats = Vec::with_capacity(steps_to_run);
    let packed_bytes = total_rows * spec.hidden * 2;
    let vel_bytes = img_rows * patch_dim * 4;

    for step in 0..steps_to_run {
        let lat_buf = dev.upload_f32(&cur_latents, &[img_rows as i64, patch_dim as i64], sem)?;
        let mut packed_outs = proj_in_exec.execute(
            &[&refined_text_hbm, &lat_buf, &proj_in_w, &proj_in_b],
            &[packed_bytes],
        )?;
        drop(lat_buf);
        let x = packed_outs
            .pop()
            .context("proj_in_pack returned zero outputs")?;

        let ts_slot = if step == 0 {
            opts.wrong_step0_timestep_index.unwrap_or(0)
        } else {
            step
        };
        let (x, stats) = stack.run_step(
            dev,
            &block_exec,
            &ckpt,
            x,
            (&step_row_pairs_hbm[ts_slot], &rotary_hbm, total_rows),
        )?;
        per_step_stats.push(stats);

        let head = opts.force_pdd_head.unwrap_or(step);
        let mut vel_outs = norm_out_pdd_exec.execute(
            &[
                &x,
                &norm_out_w,
                &norm_out_adaln_hbm,
                &step_row_timesteps_hbm[ts_slot],
                &pdd_w[head],
                &pdd_b[head],
            ],
            &[vel_bytes],
        )?;
        drop(x);
        let vel_buf = vel_outs
            .pop()
            .context("norm_out_pdd returned zero outputs")?;
        let vel = vel_buf.download_f32()?;
        drop(vel_buf);
        ensure!(
            vel.len() == cur_latents.len() && vel.iter().all(|v| v.is_finite()),
            "non-finite or mismatched velocity at denoising step {step}"
        );
        if step == 0 {
            step0_velocity = vel.clone();
        }

        let dt = schedule.sigmas[step] - schedule.sigmas[step + 1];
        let signed_dt = if opts.negate_euler_velocity { -dt } else { dt };
        for (x_i, &v_i) in cur_latents.iter_mut().zip(&vel) {
            *x_i += signed_dt * v_i;
            ensure!(x_i.is_finite(), "non-finite latent after Euler step {step}");
        }
    }

    drop((
        stack,
        step_row_pairs_hbm,
        step_row_timesteps_hbm,
        rotary_hbm,
        pdd_w,
        pdd_b,
        norm_out_w,
        proj_in_w,
        proj_in_b,
        norm_out_adaln_hbm,
        refined_text_hbm,
        norm_out_pdd_exec,
        proj_in_exec,
        block_exec,
        ckpt,
    ));
    let hbm_after_drop_bytes = dev.memory_stats()?.bytes_in_use;
    Ok((
        cur_latents,
        DitSampleReport {
            total_rows,
            text_rows,
            image_rows: img_rows,
            head_chunk,
            steps_run: steps_to_run,
            live_topology,
            live_plan,
            per_step_stats,
            step0_velocity,
            hbm_before_bytes,
            hbm_after_drop_bytes,
            elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
        },
    ))
}

/// Evaluates `norm_out` + F32 `pdd_proj_out.{head_index}` (`precision = [HIGHEST, HIGHEST]`) on TPU
/// from a pre-`norm_out` BF16 hidden state `[total_rows, hidden]`.
pub fn run_norm_out_pdd_with_device(
    dev: &TpuDevice,
    model: &Path,
    text_rows: usize,
    pre_norm_bf16: &[bf16],
    timestep_index: usize,
    head_index: usize,
) -> Result<Vec<f32>> {
    let spec = ImageSpec::read(model)?;
    let schedule = ImageSchedule::published()?;
    ensure!(
        timestep_index < schedule.timesteps.len() && head_index < spec.pdd_heads,
        "norm_out_pdd index out of range: ts={timestep_index} head={head_index}"
    );
    ensure!(
        pre_norm_bf16.len().is_multiple_of(spec.hidden),
        "pre_norm_bf16 len {} not divisible by hidden {}",
        pre_norm_bf16.len(),
        spec.hidden
    );
    let total_rows = pre_norm_bf16.len() / spec.hidden;
    ensure!(
        text_rows > 0 && total_rows > text_rows,
        "invalid norm_out_pdd row counts: text={text_rows} total={total_rows}"
    );
    let img_rows = total_rows - text_rows;
    let patch_dim = spec.row_width();

    let dir = model.join("transformer_diffusers");
    let ckpt = MappedCheckpoint::open_component(&dir, "diffusion_pytorch_model")?;
    let norm_out_adaln = compute_norm_out_adaln_table(dev, &spec, &ckpt, &schedule.timesteps)?;
    let sem = HostSemantics::ImmutableOnlyDuringCall;
    let exec = dev.compile_mlir(&emit_h3_dit_norm_out_pdd(&DitNormOutPddSpec {
        rows: total_rows,
        text_rows,
        hidden: spec.hidden,
        patch_dim,
        timesteps: schedule.timesteps.len(),
        final_norm_eps: spec.final_norm_eps,
    })?)?;
    let x_buf = dev.upload_bf16(pre_norm_bf16, &[total_rows as i64, spec.hidden as i64], sem)?;
    let norm_w = dev.upload_bf16(
        &read_bf16_ckpt(&ckpt, "norm_out.norm.weight", &[spec.hidden])?,
        &[spec.hidden as i64],
        sem,
    )?;
    let adaln_buf = dev.upload_bf16(
        &norm_out_adaln,
        &[schedule.timesteps.len() as i64, 2, spec.hidden as i64],
        sem,
    )?;
    let ts_i32 = vec![i32::try_from(timestep_index).context("ts_index overflows i32")?; total_rows];
    let ts_buf = dev.upload_i32(&ts_i32, &[total_rows as i64], sem)?;
    let w_out = dev.upload_f32(
        &read_f32_ckpt(
            &ckpt,
            &format!("pdd_proj_out.{head_index}.weight"),
            &[patch_dim, spec.hidden],
        )?,
        &[patch_dim as i64, spec.hidden as i64],
        sem,
    )?;
    let b_out = dev.upload_f32(
        &read_f32_ckpt(
            &ckpt,
            &format!("pdd_proj_out.{head_index}.bias"),
            &[patch_dim],
        )?,
        &[patch_dim as i64],
        sem,
    )?;
    let mut outs = exec.execute(
        &[&x_buf, &norm_w, &adaln_buf, &ts_buf, &w_out, &b_out],
        &[img_rows * patch_dim * 4],
    )?;
    let vel_buf = outs.pop().context("norm_out_pdd returned zero outputs")?;
    let vel = vel_buf.download_f32()?;
    drop((
        vel_buf, outs, x_buf, norm_w, adaln_buf, ts_buf, w_out, b_out, exec, ckpt,
    ));
    ensure!(
        vel.len() == img_rows * patch_dim && vel.iter().all(|v| v.is_finite()),
        "norm_out_pdd probe produced non-finite or mismatched output"
    );
    Ok(vel)
}

/// Decodes normalized F32 NCHW image latents `[24, latent_h, latent_w]` to F32 RGB HWC `[H, W, 3]`,
/// matching CUDA `image_decode`.
pub fn image_decode(
    model: &Path,
    (ordinal, _runtime): (usize, &Path),
    extent: [usize; 2],
    normalized: &[f32],
) -> Result<Vec<f32>> {
    let dev = TpuDevice::open_default(ordinal)?;
    let (canvas, _) = image_decode_with_device(
        &dev,
        model,
        extent,
        normalized,
        &VaeDecodeOptions::default(),
    )?;
    Ok(canvas)
}

/// Decodes normalized F32 NCHW image latents on an open [`TpuDevice`] with telemetry and control hooks.
pub fn image_decode_with_device(
    dev: &TpuDevice,
    model: &Path,
    extent: [usize; 2],
    normalized: &[f32],
    opts: &VaeDecodeOptions,
) -> Result<(Vec<f32>, VaeDecodeReport)> {
    let t0 = Instant::now();
    let hbm_before_bytes = dev.memory_stats()?.bytes_in_use;
    ensure!(
        normalized.iter().all(|v| v.is_finite()),
        "image VAE input is non-finite"
    );
    let vae_dir = model.join("vae_diffusers");
    let config_path = vae_dir.join("config.json");
    let vae_spec = ImageVaeSpec::parse(&serde_json::from_slice(
        &fs::read(&config_path).with_context(|| format!("read {}", config_path.display()))?,
    )?)?;
    let single = vae_spec.single_frame();
    let [latent_h, latent_w] = extent;
    let elems = vae_spec
        .latent_channels
        .checked_mul(latent_h)
        .and_then(|n| n.checked_mul(latent_w))
        .context("VAE latent extent overflow")?;
    ensure!(
        latent_h > 0 && latent_w > 0 && normalized.len() == elems,
        "image VAE latent extent mismatch: len={} expected={elems}",
        normalized.len()
    );
    let canvas_h = latent_h * vae_spec.spatial_ratio;
    let canvas_w = latent_w * vae_spec.spatial_ratio;
    let tiles = VaeTiles::released(&single, canvas_h, canvas_w)?;
    let tile_lh = tiles.height / vae_spec.spatial_ratio;
    let tile_lw = tiles.width / vae_spec.spatial_ratio;
    let patches = tile_lh * tile_lw;
    let tokens = patches + vae_spec.registers + 1;
    let tile_count = tiles.count();
    let tile_batch = tile_count.min(VAE_MAX_TILE_BATCH);

    let ckpt = MappedCheckpoint::open_component(&vae_dir, "diffusion_pytorch_model")?;
    let _manifest = ckpt.minimax_h3_image_vae_manifest(&vae_spec)?;

    let weights = VaeDecoderHbmWeights::load(dev, &ckpt, &vae_spec, single.patch_width())?;
    let rotary = vae_2d_rotary_table(&vae_spec, tile_lh, tile_lw, opts.transpose_rope_hw)?;
    let sem = HostSemantics::ImmutableOnlyDuringCall;
    let rotary_hbm = dev.upload_bf16(&rotary, &[tokens as i64, vae_spec.rope_dims as i64], sem)?;

    let pre_exec = dev.compile_mlir(&emit_h3_image_vae_pre(&VaePreHloSpec {
        batch: tile_batch,
        patches,
        latent_channels: vae_spec.latent_channels,
        dim: vae_spec.dim,
        registers: vae_spec.registers,
    })?)?;
    let block_exec = dev.compile_mlir(&emit_h3_image_vae_block(&VaeBlockHloSpec {
        batch: tile_batch,
        tokens,
        dim: vae_spec.dim,
        heads: vae_spec.heads,
        head_dim: vae_spec.head_dim,
        ffn: vae_spec.ffn,
        rotary_dim: vae_spec.rope_dims,
        norm_eps: vae_spec.norm_eps,
    })?)?;
    let post_exec = dev.compile_mlir(&emit_h3_image_vae_post(&VaePostHloSpec {
        batch: tile_batch,
        tokens,
        num_patches: patches,
        dim: vae_spec.dim,
        patch_width: single.patch_width(),
        norm_eps: vae_spec.norm_eps,
    })?)?;

    // Build tile origins in row-major order.
    let mut origins = Vec::with_capacity(tile_count);
    for &ry in &tiles.rows.0 {
        for &cx in &tiles.cols.0 {
            origins.push((ry / vae_spec.spatial_ratio, cx / vae_spec.spatial_ratio));
        }
    }

    let c_lat = vae_spec.latent_channels;
    let patch_w = single.patch_width();
    let stream_bytes = tile_batch * tokens * vae_spec.dim * 4;
    let post_bytes = tile_batch * patches * patch_w * 2;
    let tile_pixels = vae_spec.out_channels * tiles.height * tiles.width;
    let mut raw_tiles = vec![bf16::ZERO; tile_count * tile_pixels];

    for chunk_start in (0..tile_count).step_by(tile_batch) {
        let active = (tile_count - chunk_start).min(tile_batch);
        let mut z_batch = vec![bf16::ZERO; tile_batch * patches * c_lat];
        for b in 0..active {
            let (ly0, lx0) = origins[chunk_start + b];
            let base = b * patches * c_lat;
            for py in 0..tile_lh {
                for px in 0..tile_lw {
                    let tok_off = base + (py * tile_lw + px) * c_lat;
                    for c in 0..c_lat {
                        let raw = normalized[(c * latent_h + ly0 + py) * latent_w + (lx0 + px)];
                        let denorm = raw * vae_spec.latents_std[c] + vae_spec.latents_mean[c];
                        let val = bf16::from_f32(denorm);
                        ensure!(val.is_finite(), "VAE denormalized latent is non-finite");
                        z_batch[tok_off + c] = val;
                    }
                }
            }
        }
        let z_buf = dev.upload_bf16(
            &z_batch,
            &[tile_batch as i64, patches as i64, c_lat as i64],
            sem,
        )?;
        let mut pre_outs = pre_exec.execute(
            &[
                &z_buf,
                &weights.post_quant_w,
                &weights.post_quant_b,
                &weights.proj_in_w,
                &weights.proj_in_b,
                &weights.register_tokens,
            ],
            &[stream_bytes],
        )?;
        drop(z_buf);
        let mut x = pre_outs.pop().context("vae_pre returned zero outputs")?;

        for layer in &weights.blocks {
            let mut blk_outs = block_exec.execute(
                &[
                    &x,
                    &rotary_hbm,
                    &layer.norm1,
                    &layer.wq,
                    &layer.bq,
                    &layer.wk,
                    &layer.bk,
                    &layer.wv,
                    &layer.bv,
                    &layer.wout,
                    &layer.bout,
                    &layer.scale1,
                    &layer.norm2,
                    &layer.wff0,
                    &layer.bff0,
                    &layer.wff2,
                    &layer.bff2,
                    &layer.scale2,
                ],
                &[stream_bytes],
            )?;
            x = blk_outs.pop().context("vae_block returned zero outputs")?;
        }

        let mut post_outs = post_exec.execute(
            &[
                &x,
                &weights.norm_out_w,
                &weights.norm_out_b,
                &weights.proj_out_w,
                &weights.proj_out_b,
            ],
            &[post_bytes],
        )?;
        drop(x);
        let out_buf = post_outs.pop().context("vae_post returned zero outputs")?;
        let out_bf16 = out_buf.download_bf16()?;
        drop(out_buf);
        ensure!(
            out_bf16.iter().all(|v| v.is_finite()),
            "image VAE decoder produced non-finite value"
        );

        for b in 0..active {
            let src = &out_bf16[b * patches * patch_w..(b + 1) * patches * patch_w];
            let dst = &mut raw_tiles
                [(chunk_start + b) * tile_pixels..(chunk_start + b + 1) * tile_pixels];
            unpatchify_vae_tile_bf16(
                src,
                dst,
                (
                    tile_lh,
                    tile_lw,
                    vae_spec.out_channels,
                    vae_spec.spatial_ratio,
                ),
            );
        }
    }

    drop((weights, rotary_hbm, pre_exec, block_exec, post_exec, ckpt));
    let hbm_after_drop_bytes = dev.memory_stats()?.bytes_in_use;

    let stitched = stitch_tiles_bf16(
        &raw_tiles,
        &tiles,
        vae_spec.out_channels,
        [canvas_h, canvas_w],
    )?;
    let pixels = canvas_h * canvas_w;
    let mut hwc = vec![0.0f32; pixels * vae_spec.out_channels];
    for p in 0..pixels {
        for c in 0..vae_spec.out_channels {
            let v = stitched[c * pixels + p].to_f32() * (VAE_PIXEL_STD[c] as f32)
                + (VAE_PIXEL_MEAN[c] as f32);
            ensure!(v.is_finite(), "image VAE denormalized pixel is non-finite");
            hwc[p * vae_spec.out_channels + c] = v;
        }
    }
    Ok((
        hwc,
        VaeDecodeReport {
            latent_extent: extent,
            canvas_extent: [canvas_h, canvas_w],
            tile_count,
            tile_batch,
            hbm_before_bytes,
            hbm_after_drop_bytes,
            elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
        },
    ))
}

/// End-to-end MiniMax-H3-Image generation on TPU, matching CUDA `image_latents`.
pub fn image_latents(
    model: &Path,
    options: &ImageRunOptions,
    request: &ImageRequest,
) -> Result<ImageLatents> {
    let dev = TpuDevice::open_default(options.device)?;
    image_latents_with_device(&dev, model, request, None)
}

/// End-to-end MiniMax-H3-Image generation on an open [`TpuDevice`] with optional supplied initial noise.
pub fn image_latents_with_device(
    dev: &TpuDevice,
    model: &Path,
    request: &ImageRequest,
    supplied_noise: Option<&[f32]>,
) -> Result<ImageLatents> {
    ensure!(!request.prompt.is_empty(), "image prompt is empty");
    if request.expander.is_some() {
        bail!("TPU image generation does not yet support --expand");
    }
    if !request.references.is_empty() {
        bail!("TPU image generation does not yet support --edit reference images");
    }
    let spec = ImageSpec::read(model)?;
    let (canvas_h, canvas_w) = image_canvas(
        &spec,
        request.height,
        request.width,
        request.min_generation_area,
    )?;
    let (latent_h, latent_w) = image_latent_extent(&spec, canvas_h, canvas_w)?;
    let patch_dim = spec.row_width();
    let img_rows = (latent_h / spec.patch[1]) * (latent_w / spec.patch[2]);
    let noise_len = img_rows * patch_dim;
    let init_rows = match supplied_noise {
        Some(v) => {
            ensure!(
                v.len() == noise_len && v.iter().all(|x| x.is_finite()),
                "supplied noise len {} != {noise_len} or non-finite",
                v.len()
            );
            v.to_vec()
        }
        None => image_noise_draw(request.seed, noise_len),
    };

    let tokenizer = Tokenizer::load(&model.join("text_encoder"))?;
    let raw_ids = tokenizer.encode_chat(&request.prompt)?;
    let (token_ids, text_tags) = image_text_rows(&raw_ids)?;
    let layout = image_layout(&spec, &text_tags, &[], latent_h, latent_w)?;

    let (prompt_embeds, _) = run_conditioner_with_device(dev, model, &token_ids)?;
    let (sampled_rows, _) = image_sample_with_device(
        dev,
        model,
        &layout,
        &init_rows,
        &prompt_embeds,
        &DitSampleOptions::default(),
    )?;
    let values = unpatchify_latents(
        &sampled_rows,
        (spec.in_channels, 1, latent_h, latent_w),
        spec.patch,
    );
    let (canvas, _) = image_decode_with_device(
        dev,
        model,
        [latent_h, latent_w],
        &values,
        &VaeDecodeOptions::default(),
    )?;
    let rgb = image_rgb8(
        &canvas,
        [canvas_h, canvas_w],
        [request.height, request.width],
    )?;
    Ok(ImageLatents {
        channels: spec.in_channels,
        height: latent_h,
        width: latent_w,
        values,
        canvas_height: canvas_h,
        canvas_width: canvas_w,
        rgb,
    })
}

fn conditioner_text_tensor_needed(name: &str, layers: usize) -> bool {
    if name == "model.language_model.embed_tokens.weight" {
        return true;
    }
    name.strip_prefix("model.language_model.layers.")
        .and_then(|rest| rest.split('.').next())
        .and_then(|idx| idx.parse::<usize>().ok())
        .is_some_and(|idx| idx < layers)
}

fn lookup_bf16_embeddings(
    ckpt: &MappedCheckpoint,
    token_ids: &[u32],
    vocab: usize,
    hidden: usize,
) -> Result<Vec<bf16>> {
    let view = ckpt.tensor("model.language_model.embed_tokens.weight")?;
    ensure!(
        view.dtype() == Dtype::BF16 && view.shape() == [vocab, hidden],
        "embed_tokens dtype/shape mismatch: {:?} {:?}",
        view.dtype(),
        view.shape()
    );
    let bytes = view.data();
    let row_bytes = hidden * 2;
    let mut out = Vec::with_capacity(token_ids.len() * hidden);
    for &id in token_ids {
        let idx = id as usize;
        ensure!(idx < vocab, "token id {idx} >= vocab {vocab}");
        let start = idx * row_bytes;
        for &chunk in bytes[start..start + row_bytes].as_chunks::<2>().0 {
            let v = bf16::from_le_bytes(chunk);
            ensure!(v.is_finite(), "embed_tokens row {idx} has non-finite value");
            out.push(v);
        }
    }
    Ok(out)
}

fn stage_qwen_layer_to_hbm(
    dev: &TpuDevice,
    ckpt: &MappedCheckpoint,
    spec: &ModelSpec,
    layer: usize,
) -> Result<Vec<TpuBuffer>> {
    let d = spec.hidden as usize;
    let qw = spec.q_width as usize;
    let kvw = (spec.kv_heads * spec.head_dim) as usize;
    let hd = spec.head_dim as usize;
    let inter = spec.intermediate as usize;
    let p = format!("model.language_model.layers.{layer}");
    let specs: [(String, Vec<usize>); 11] = [
        (format!("{p}.input_layernorm.weight"), vec![d]),
        (format!("{p}.self_attn.q_proj.weight"), vec![qw, d]),
        (format!("{p}.self_attn.k_proj.weight"), vec![kvw, d]),
        (format!("{p}.self_attn.v_proj.weight"), vec![kvw, d]),
        (format!("{p}.self_attn.q_norm.weight"), vec![hd]),
        (format!("{p}.self_attn.k_norm.weight"), vec![hd]),
        (format!("{p}.self_attn.o_proj.weight"), vec![d, qw]),
        (format!("{p}.post_attention_layernorm.weight"), vec![d]),
        (format!("{p}.mlp.gate_proj.weight"), vec![inter, d]),
        (format!("{p}.mlp.up_proj.weight"), vec![inter, d]),
        (format!("{p}.mlp.down_proj.weight"), vec![d, inter]),
    ];
    let sem = HostSemantics::ImmutableOnlyDuringCall;
    let mut hbm_bufs = Vec::with_capacity(specs.len());
    for (name, shape) in specs {
        let data = read_bf16_ckpt(ckpt, &name, &shape)?;
        let dims: Vec<i64> = shape.iter().map(|&v| v as i64).collect();
        hbm_bufs.push(dev.upload_bf16(&data, &dims, sem)?);
    }
    Ok(hbm_bufs)
}

fn run_text_refiner_hbm(
    dev: &TpuDevice,
    spec: &ImageSpec,
    ckpt: &MappedCheckpoint,
    text: &[bf16],
    text_rows: usize,
) -> Result<TpuBuffer> {
    let sem = HostSemantics::ImmutableOnlyDuringCall;
    let ctx_exec = dev.compile_mlir(&emit_h3_dit_context_embedder(
        text_rows,
        spec.text_dim,
        spec.hidden,
    )?)?;
    let x_in = dev.upload_bf16(text, &[text_rows as i64, spec.text_dim as i64], sem)?;
    let cw = dev.upload_bf16(
        &read_bf16_ckpt(
            ckpt,
            "context_embedder.weight",
            &[spec.hidden, spec.text_dim],
        )?,
        &[spec.hidden as i64, spec.text_dim as i64],
        sem,
    )?;
    let cb = dev.upload_bf16(
        &read_bf16_ckpt(ckpt, "context_embedder.bias", &[spec.hidden])?,
        &[spec.hidden as i64],
        sem,
    )?;
    let out_bytes = text_rows * spec.hidden * 2;
    let mut outs = ctx_exec.execute(&[&x_in, &cw, &cb], &[out_bytes])?;
    drop((x_in, cw, cb, ctx_exec));
    let mut x = outs
        .pop()
        .context("context_embedder returned zero outputs")?;

    let ref_chunk = if text_rows <= 512 { 8 } else { 2 };
    let ref_exec = dev.compile_mlir(&emit_h3_dit_refiner_block(&DitRefinerBlockHloSpec {
        rows: text_rows,
        hidden: spec.hidden,
        heads: spec.heads,
        head_dim: spec.head_dim,
        ffn: spec.ffn,
        head_chunk: ref_chunk,
        norm_eps: spec.norm_eps,
        qk_norm_eps: spec.qk_norm_eps,
    })?)?;
    for layer in 0..spec.refiner_layers {
        let w = load_refiner_block_weights(ckpt, spec, layer)?;
        let bufs = upload_block_weights(dev, spec, &w, None)?;
        drop(w);
        let mut args: Vec<&TpuBuffer> = Vec::with_capacity(11);
        args.push(&x);
        for wb in &bufs {
            args.push(wb);
        }
        let mut r_outs = ref_exec.execute(&args, &[out_bytes])?;
        drop(bufs);
        x = r_outs
            .pop()
            .context("refiner block returned zero outputs")?;
    }
    drop(ref_exec);

    let norm_exec = dev.compile_mlir(&emit_bf16_rms_norm(
        text_rows,
        spec.hidden,
        spec.final_norm_eps,
    )?)?;
    let fn_w = dev.upload_bf16(
        &read_bf16_ckpt(ckpt, "token_refiner.final_norm.weight", &[spec.hidden])?,
        &[spec.hidden as i64],
        sem,
    )?;
    let mut n_outs = norm_exec.execute(&[&x, &fn_w], &[out_bytes])?;
    drop((x, fn_w, norm_exec));
    n_outs
        .pop()
        .context("token_refiner.final_norm returned zero outputs")
}

fn active_dit_block_bytes(spec: &ImageSpec) -> u64 {
    let d = spec.hidden as u64;
    let w = (spec.heads * spec.head_dim) as u64;
    let hd = spec.head_dim as u64;
    let f = spec.ffn as u64;
    (d + 3 * w * d + 2 * hd + d * w + d + 2 * f * d + d * f) * 2
}

struct VaeBlockHbmWeights {
    norm1: TpuBuffer,
    wq: TpuBuffer,
    bq: TpuBuffer,
    wk: TpuBuffer,
    bk: TpuBuffer,
    wv: TpuBuffer,
    bv: TpuBuffer,
    wout: TpuBuffer,
    bout: TpuBuffer,
    scale1: TpuBuffer,
    norm2: TpuBuffer,
    wff0: TpuBuffer,
    bff0: TpuBuffer,
    wff2: TpuBuffer,
    bff2: TpuBuffer,
    scale2: TpuBuffer,
}

struct VaeDecoderHbmWeights {
    post_quant_w: TpuBuffer,
    post_quant_b: TpuBuffer,
    proj_in_w: TpuBuffer,
    proj_in_b: TpuBuffer,
    register_tokens: TpuBuffer,
    blocks: Vec<VaeBlockHbmWeights>,
    norm_out_w: TpuBuffer,
    norm_out_b: TpuBuffer,
    proj_out_w: TpuBuffer,
    proj_out_b: TpuBuffer,
}

impl VaeDecoderHbmWeights {
    fn load(
        dev: &TpuDevice,
        ckpt: &MappedCheckpoint,
        spec: &ImageVaeSpec,
        patch_width: usize,
    ) -> Result<Self> {
        let sem = HostSemantics::ImmutableOnlyDuringCall;
        let c = spec.latent_channels;
        let d = spec.dim;
        let f = spec.ffn;
        let load_bf16 = |name: &str, file_shape: &[usize], tpu_dims: &[i64]| -> Result<TpuBuffer> {
            ensure!(
                !decoder_weight_is_raw_f32(name),
                "expected BF16-narrowed VAE weight, got raw F32 {name}"
            );
            let v = read_f32_narrow_bf16_ckpt(ckpt, name, file_shape)?;
            dev.upload_bf16(&v, tpu_dims, sem)
        };
        let load_f32 = |name: &str, shape: &[usize]| -> Result<TpuBuffer> {
            ensure!(
                decoder_weight_is_raw_f32(name),
                "expected raw F32 VAE weight, got narrowed {name}"
            );
            let v = read_f32_ckpt(ckpt, name, shape)?;
            let dims: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
            dev.upload_f32(&v, &dims, sem)
        };

        let post_quant_w = load_bf16(
            "post_quant_conv.weight",
            &[c, c, 1, 1],
            &[c as i64, c as i64],
        )?;
        let post_quant_b = load_bf16("post_quant_conv.bias", &[c], &[c as i64])?;
        let proj_in_w = load_bf16("decoder.proj_in.weight", &[d, c], &[d as i64, c as i64])?;
        let proj_in_b = load_bf16("decoder.proj_in.bias", &[d], &[d as i64])?;
        let register_tokens = load_bf16(
            "decoder.register_tokens",
            &[1, spec.registers, d],
            &[spec.registers as i64, d as i64],
        )?;

        let mut blocks = Vec::with_capacity(spec.layers);
        for layer in 0..spec.layers {
            let p = format!("decoder.transformer_blocks.{layer}");
            blocks.push(VaeBlockHbmWeights {
                norm1: load_f32(&format!("{p}.norm1.weight"), &[d])?,
                wq: load_bf16(
                    &format!("{p}.attn.to_q.weight"),
                    &[d, d],
                    &[d as i64, d as i64],
                )?,
                bq: load_bf16(&format!("{p}.attn.to_q.bias"), &[d], &[d as i64])?,
                wk: load_bf16(
                    &format!("{p}.attn.to_k.weight"),
                    &[d, d],
                    &[d as i64, d as i64],
                )?,
                bk: load_bf16(&format!("{p}.attn.to_k.bias"), &[d], &[d as i64])?,
                wv: load_bf16(
                    &format!("{p}.attn.to_v.weight"),
                    &[d, d],
                    &[d as i64, d as i64],
                )?,
                bv: load_bf16(&format!("{p}.attn.to_v.bias"), &[d], &[d as i64])?,
                wout: load_bf16(
                    &format!("{p}.attn.to_out.0.weight"),
                    &[d, d],
                    &[d as i64, d as i64],
                )?,
                bout: load_bf16(&format!("{p}.attn.to_out.0.bias"), &[d], &[d as i64])?,
                scale1: load_f32(&format!("{p}.scale1"), &[d])?,
                norm2: load_f32(&format!("{p}.norm2.weight"), &[d])?,
                wff0: load_bf16(
                    &format!("{p}.ff.net.0.proj.weight"),
                    &[2 * f, d],
                    &[(2 * f) as i64, d as i64],
                )?,
                bff0: load_bf16(
                    &format!("{p}.ff.net.0.proj.bias"),
                    &[2 * f],
                    &[(2 * f) as i64],
                )?,
                wff2: load_bf16(
                    &format!("{p}.ff.net.2.weight"),
                    &[d, f],
                    &[d as i64, f as i64],
                )?,
                bff2: load_bf16(&format!("{p}.ff.net.2.bias"), &[d], &[d as i64])?,
                scale2: load_f32(&format!("{p}.scale2"), &[d])?,
            });
        }
        let norm_out_w = load_f32("decoder.norm_out.weight", &[d])?;
        let norm_out_b = load_f32("decoder.norm_out.bias", &[d])?;
        let proj_out_w = load_bf16(
            "decoder.proj_out.weight",
            &[patch_width, d],
            &[patch_width as i64, d as i64],
        )?;
        let proj_out_b = load_bf16(
            "decoder.proj_out.bias",
            &[patch_width],
            &[patch_width as i64],
        )?;
        Ok(Self {
            post_quant_w,
            post_quant_b,
            proj_in_w,
            proj_in_b,
            register_tokens,
            blocks,
            norm_out_w,
            norm_out_b,
            proj_out_w,
            proj_out_b,
        })
    }
}

/// 2D ViT decoder rotary table `[tokens, rope_dims]` BF16: `[cos_h, cos_w, sin_h, sin_w]`.
pub(crate) fn vae_2d_rotary_table(
    spec: &ImageVaeSpec,
    tile_lh: usize,
    tile_lw: usize,
    transpose_hw: bool,
) -> Result<Vec<bf16>> {
    ensure!(
        spec.rope_dims.is_multiple_of(4) && tile_lh > 0 && tile_lw > 0,
        "invalid VAE 2D RoPE geometry"
    );
    let per_axis = spec.rope_dims / 4;
    let half = 2 * per_axis;
    let inv_freq: Vec<f32> = (0..per_axis)
        .map(|k| 1.0f32 / spec.rope_theta.powf((4 * k) as f32 / spec.rope_dims as f32))
        .collect();
    let tau = 2.0f32 * std::f32::consts::PI;
    let h_grid: Vec<f32> = (0..tile_lh)
        .map(|y| 2.0 * ((y as f32 + 0.5) / tile_lh as f32) - 1.0)
        .collect();
    let w_grid: Vec<f32> = (0..tile_lw)
        .map(|x| 2.0 * ((x as f32 + 0.5) / tile_lw as f32) - 1.0)
        .collect();

    let suffix = spec.registers + 1;
    let tokens = tile_lh * tile_lw + suffix;
    let mut table = Vec::with_capacity(tokens * spec.rope_dims);
    let mut angles = vec![0.0f32; half];
    for &hy in &h_grid {
        for &wx in &w_grid {
            let (a0, a1) = if transpose_hw { (wx, hy) } else { (hy, wx) };
            for k in 0..per_axis {
                angles[k] = tau * a0 * inv_freq[k];
                angles[per_axis + k] = tau * a1 * inv_freq[k];
            }
            table.extend(angles.iter().map(|a| bf16::from_f32(a.cos())));
            table.extend(angles.iter().map(|a| bf16::from_f32(a.sin())));
        }
    }
    for _ in 0..suffix {
        table.extend(std::iter::repeat_n(bf16::ONE, half));
        table.extend(std::iter::repeat_n(bf16::ZERO, half));
    }
    Ok(table)
}

/// Unpatchifies one decoded tile `[tile_lh * tile_lw, out_channels * ratio * ratio]` to planar `[out_channels, tile_h, tile_w]`.
pub(crate) fn unpatchify_vae_tile_bf16(
    src: &[bf16],
    dst: &mut [bf16],
    (tile_lh, tile_lw, channels, ratio): (usize, usize, usize, usize),
) {
    let tile_h = tile_lh * ratio;
    let tile_w = tile_lw * ratio;
    let plane = tile_h * tile_w;
    let patch_w = channels * ratio * ratio;
    debug_assert_eq!(src.len(), tile_lh * tile_lw * patch_w);
    debug_assert_eq!(dst.len(), channels * plane);
    for ly in 0..tile_lh {
        for lx in 0..tile_lw {
            let tok = &src[(ly * tile_lw + lx) * patch_w..];
            for c in 0..channels {
                for py in 0..ratio {
                    let y = ly * ratio + py;
                    let dst_row = c * plane + y * tile_w + lx * ratio;
                    let src_row = (c * ratio + py) * ratio;
                    dst[dst_row..dst_row + ratio].copy_from_slice(&tok[src_row..src_row + ratio]);
                }
            }
        }
    }
}

fn linear_blend_bf16(a: bf16, b: bf16, position: usize, extent: usize) -> bf16 {
    let p = bf16::from_f32(position as f32).to_f32();
    let w = bf16::from_f32(p / extent as f32);
    let inv = bf16::from_f32(1.0f32 - w.to_f32());
    let ta = bf16::from_f32(a.to_f32() * inv.to_f32());
    let tb = bf16::from_f32(b.to_f32() * w.to_f32());
    bf16::from_f32(ta.to_f32() + tb.to_f32())
}

/// Stitches raw planar BF16 tiles `[rows * cols, channels, tile_h, tile_w]` onto `[channels, canvas_h, canvas_w]`
/// with vertical blend against the raw tile above followed by horizontal blend against the raw tile to the left.
pub(crate) fn stitch_tiles_bf16(
    tiles_data: &[bf16],
    tiles: &VaeTiles,
    channels: usize,
    canvas_extent: [usize; 2],
) -> Result<Vec<bf16>> {
    let [canvas_h, canvas_w] = canvas_extent;
    let rows = tiles.rows.0.len();
    let cols = tiles.cols.0.len();
    let tile_h = tiles.height;
    let tile_w = tiles.width;
    let tile_plane = tile_h * tile_w;
    let tile_size = channels * tile_plane;
    ensure!(
        tiles_data.len() == rows * cols * tile_size,
        "stitch_tiles_bf16 tile buffer length mismatch"
    );
    let canvas_plane = canvas_h * canvas_w;
    let mut canvas = vec![bf16::ZERO; channels * canvas_plane];
    let row_starts = &tiles.rows.0;
    let row_overlaps = &tiles.rows.1;
    let col_starts = &tiles.cols.0;
    let col_overlaps = &tiles.cols.1;

    for y_out in 0..canvas_h {
        let mut r = 0usize;
        while r + 1 < rows && row_starts[r + 1] <= y_out {
            r += 1;
        }
        let y = y_out - row_starts[r];
        for x_out in 0..canvas_w {
            let mut c_tile = 0usize;
            while c_tile + 1 < cols && col_starts[c_tile + 1] <= x_out {
                c_tile += 1;
            }
            let x = x_out - col_starts[c_tile];
            let cur_tile = (r * cols + c_tile) * tile_size;
            for ch in 0..channels {
                let base = ch * tile_plane;
                let mut v = tiles_data[cur_tile + base + y * tile_w + x];
                if r > 0 && y < row_overlaps[r - 1] {
                    let above_y = y + tile_h - row_overlaps[r - 1];
                    let above_tile = ((r - 1) * cols + c_tile) * tile_size;
                    let a = tiles_data[above_tile + base + above_y * tile_w + x];
                    v = linear_blend_bf16(a, v, y, row_overlaps[r - 1]);
                }
                if c_tile > 0 && x < col_overlaps[c_tile - 1] {
                    let left_x = x + tile_w - col_overlaps[c_tile - 1];
                    let left_tile = (r * cols + c_tile - 1) * tile_size;
                    let a = tiles_data[left_tile + base + y * tile_w + left_x];
                    v = linear_blend_bf16(a, v, x, col_overlaps[c_tile - 1]);
                }
                canvas[ch * canvas_plane + y_out * canvas_w + x_out] = v;
            }
        }
    }
    Ok(canvas)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_vae_spec() -> ImageVaeSpec {
        ImageVaeSpec {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 24,
            block_out_channels: vec![128, 256, 256, 512, 512, 1024],
            layers_per_block: 2,
            spatial_factors: vec![2, 2, 2, 2, 1, 1],
            temporal_factors: vec![1, 2, 2, 1, 1, 1],
            spatial_ratio: 16,
            temporal_ratio: 4,
            dim: 1024,
            layers: 24,
            heads: 16,
            head_dim: 64,
            registers: 4,
            ffn: 4096,
            rope_theta: 100.0,
            rope_dims: 32,
            norm_eps: 1e-5,
            groups: 32,
            encoder_eps: 1e-6,
            latents_mean: vec![0.0; 24],
            latents_std: vec![1.0; 24],
        }
    }

    #[test]
    fn vae_rotary_table_matches_2d_grid_and_transpose_perturbs_off_diagonal() -> Result<()> {
        let spec = test_vae_spec();
        let normal = vae_2d_rotary_table(&spec, 16, 16, false)?;
        let swapped = vae_2d_rotary_table(&spec, 16, 16, true)?;
        let tokens = 16 * 16 + 5;
        assert_eq!(normal.len(), tokens * 32);
        assert_eq!(swapped.len(), tokens * 32);
        // Suffix register + cls tokens sit at (0, 0): cos=1, sin=0.
        for t in 256..tokens {
            let row = &normal[t * 32..(t + 1) * 32];
            assert!(row[..16].iter().all(|&v| v == bf16::ONE));
            assert!(row[16..].iter().all(|&v| v == bf16::ZERO));
        }
        // Diagonal token (0, 0) has h == w so swap leaves it unchanged; off-diagonal (0, 1) differs.
        assert_eq!(&normal[0..32], &swapped[0..32]);
        assert_ne!(&normal[32..64], &swapped[32..64]);
        Ok(())
    }

    #[test]
    fn unpatchify_and_stitch_match_vertical_then_horizontal_raw_blend() -> Result<()> {
        // 2x2 grid of 4x4 tiles with 2-px overlap -> 6x6 canvas, 1 channel.
        let tiles = VaeTiles {
            height: 4,
            width: 4,
            rows: (vec![0, 2], vec![2]),
            cols: (vec![0, 2], vec![2]),
        };
        let vals = [1.0f32, 3.0, 5.0, 9.0];
        let mut raw = Vec::new();
        for &v in &vals {
            raw.extend(std::iter::repeat_n(bf16::from_f32(v), 16));
        }
        let stitched = stitch_tiles_bf16(&raw, &tiles, 1, [6, 6])?;
        // Corner (y=2, x=2): tile(1,1)=9 blended vertically at y=0 (w=0) with tile(0,1)=3 -> 3,
        // then blended horizontally at x=0 (w=0) with raw tile(1,0)=5 -> 5.
        assert_eq!(stitched[2 * 6 + 2], bf16::from_f32(5.0));
        // (y=3, x=3): tile(1,1)=9 blended vertically at y=1 (w=0.5) with tile(0,1)=3 -> 6,
        // then blended horizontally at x=1 (w=0.5) with raw tile(1,0)=5 -> 5.5.
        assert_eq!(stitched[3 * 6 + 3], bf16::from_f32(5.5));
        // Unpatchify sanity check: 2x2 latent patches of 2x2 pixels, 2 channels.
        let mut patches = Vec::new();
        for ly in 0..2 {
            for lx in 0..2 {
                for c in 0..2 {
                    for py in 0..2 {
                        for px in 0..2 {
                            let y = ly * 2 + py;
                            let x = lx * 2 + px;
                            patches.push(bf16::from_f32((c * 100 + y * 10 + x) as f32));
                        }
                    }
                }
            }
        }
        let mut planar = vec![bf16::ZERO; 2 * 16];
        unpatchify_vae_tile_bf16(&patches, &mut planar, (2, 2, 2, 2));
        for c in 0..2 {
            for y in 0..4 {
                for x in 0..4 {
                    assert_eq!(
                        planar[c * 16 + y * 4 + x],
                        bf16::from_f32((c * 100 + y * 10 + x) as f32)
                    );
                }
            }
        }
        Ok(())
    }
}
