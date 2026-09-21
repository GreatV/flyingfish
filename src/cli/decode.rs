use super::H3DecodeCommand;
use super::checkpoint::resolve_component;
use super::device_parse::parse_device;
use super::output_hygiene::{
    create_new_directory, ensure_new_output, publish_png_frame_manifest,
    resolve_output_outside_model, write_telemetry,
};
use super::prompt::take_input;
use super::{ensure_optional_output_is_distinct, resolve_optional_new_output};
use anyhow::{Context, Result, bail};
use candle_core::safetensors;
use flyingfish::h3::audio_vae::{StreamedAudioVae, write_wav};
use flyingfish::h3::video_vae::StreamedVideoVae;
use flyingfish::runtime::artifact::ArtifactStaging;
use flyingfish::runtime::telemetry::TelemetryMonitor;
use std::time::Duration;

pub(super) fn run_decode_audio(command: H3DecodeCommand) -> Result<()> {
    let H3DecodeCommand::Audio {
        model,
        component,
        inputs,
        output,
        device,
        weights: weight_args,
        wav_format,
        telemetry_json,
    } = command
    else {
        bail!("internal CLI dispatch mismatch for decode-audio");
    };
    let component_dir = resolve_component(&model, &component)?;
    let output = resolve_output_outside_model(&output, &model)?;
    anyhow::ensure!(
        inputs.is_file(),
        "input safetensors does not exist: {}",
        inputs.display()
    );
    ensure_new_output(&output, "audio output")?;
    let output_staging = ArtifactStaging::new_for_path_producer(&output)
        .with_context(|| format!("failed to stage audio output {}", output.display()))?;
    let device = parse_device(&device)?;
    let telemetry_json = resolve_optional_new_output(telemetry_json, &model)?;
    ensure_optional_output_is_distinct(telemetry_json.as_deref(), &[&output])?;
    let telemetry = telemetry_json
        .as_ref()
        .map(|_| TelemetryMonitor::start(Some(device.clone()), Duration::from_millis(100)))
        .transpose()?;
    let mut values = safetensors::load(&inputs, &device)
        .with_context(|| format!("failed to load latents {}", inputs.display()))?;
    let audio_latents = take_input(&mut values, "audio_latents")?;
    drop(values);
    let decoder = StreamedAudioVae::open(
        component_dir,
        weight_args.weight_source,
        weight_args.cache_policy()?,
        device,
    )?;
    let waveform = decoder.decode(&audio_latents)?;
    write_wav(
        output_staging.producer_path(),
        &waveform,
        decoder.config().sampling_rate,
        wav_format,
    )?;
    output_staging.publish()?;
    println!(
        "decoded {} channels at {} Hz to {}",
        waveform.dim(0)?,
        decoder.config().sampling_rate,
        output.display()
    );
    if let (Some(path), Some(monitor)) = (telemetry_json.as_ref(), telemetry) {
        write_telemetry(path, monitor)?;
    }
    Ok(())
}

pub(super) fn run_decode_video(command: H3DecodeCommand) -> Result<()> {
    let H3DecodeCommand::Video {
        model,
        component,
        inputs,
        output_dir,
        device,
        weights: weight_args,
        attention_query_chunk_size,
        telemetry_json,
    } = command
    else {
        bail!("internal CLI dispatch mismatch for decode-video");
    };
    let component_dir = resolve_component(&model, &component)?;
    let output_dir = resolve_output_outside_model(&output_dir, &model)?;
    anyhow::ensure!(
        inputs.is_file(),
        "input safetensors does not exist: {}",
        inputs.display()
    );
    ensure_new_output(&output_dir, "output directory")?;
    let device = parse_device(&device)?;
    let telemetry_json = resolve_optional_new_output(telemetry_json, &model)?;
    ensure_optional_output_is_distinct(telemetry_json.as_deref(), &[&output_dir])?;
    let telemetry = telemetry_json
        .as_ref()
        .map(|_| TelemetryMonitor::start(Some(device.clone()), Duration::from_millis(100)))
        .transpose()?;
    let mut values = safetensors::load(&inputs, &device)
        .with_context(|| format!("failed to load latents {}", inputs.display()))?;
    let video_latents = take_input(&mut values, "video_latents")?;
    drop(values);
    let decoder = StreamedVideoVae::open(
        component_dir,
        weight_args.weight_source,
        weight_args.cache_policy()?,
        device,
        attention_query_chunk_size,
    )?;
    create_new_directory(&output_dir, "frame directory")?;
    let frames = decoder.decode_to_png_frames(&video_latents, &output_dir)?;
    publish_png_frame_manifest(&output_dir, frames)?;
    println!("decoded {frames} RGB frames to {}", output_dir.display());
    if let (Some(path), Some(monitor)) = (telemetry_json.as_ref(), telemetry) {
        write_telemetry(path, monitor)?;
    }
    Ok(())
}
