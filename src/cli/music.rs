use super::*;
mod batch;
use flyingfish::music::pipeline::{Music3, Options};
use flyingfish::runtime::artifact::ArtifactStaging;

#[derive(Debug, Subcommand)]
pub(super) enum MusicCommand {
    #[command(about = "Generate a batch using one reusable Music3 worker per CUDA device")]
    Batch(batch::Args),
    #[command(about = "Generate a stereo WAV with the native Music3 pipeline")]
    Generate {
        #[arg(long)]
        model: PathBuf,
        #[arg(
            long,
            required_unless_present = "prompt_file",
            conflicts_with = "prompt_file"
        )]
        prompt: Option<String>,
        #[arg(long, conflicts_with = "prompt")]
        prompt_file: Option<PathBuf>,
        #[arg(
            long,
            required_unless_present = "lyrics_file",
            conflicts_with = "lyrics_file"
        )]
        lyrics: Option<String>,
        #[arg(long, conflicts_with = "lyrics")]
        lyrics_file: Option<PathBuf>,
        #[arg(long, default_value_t = 8.)]
        duration: f64,
        #[arg(long, default_value_t = 30)]
        steps: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value_t = 32)]
        attention_query_chunk_size: usize,
        #[arg(long, default_value = "auto")]
        device: String,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, value_enum, default_value = "pcm16")]
        wav_format: WavSampleFormat,
        #[command(flatten)]
        weights: WeightCacheArgs,
        #[command(flatten)]
        device_cache: DeviceCacheArgs,
    },
}

pub(super) fn run(command: MusicCommand) -> Result<()> {
    let command = match command {
        MusicCommand::Batch(args) => return batch::run(args),
        command => command,
    };
    let MusicCommand::Generate {
        model,
        prompt,
        prompt_file,
        lyrics,
        lyrics_file,
        duration,
        steps,
        seed,
        attention_query_chunk_size,
        device,
        output,
        wav_format,
        weights,
        device_cache,
    } = command
    else {
        unreachable!()
    };
    let read = |text: Option<String>, path: Option<PathBuf>| -> Result<String> {
        match (text, path) {
            (Some(text), None) => Ok(text),
            (None, Some(path)) => Ok(std::fs::read_to_string(path)?),
            _ => bail!("supply either text or a text file"),
        }
    };
    let prompt = read(prompt, prompt_file)?;
    let lyrics = read(lyrics, lyrics_file)?;
    let output = resolve_output_outside_model(&output, &model)?;
    ensure_new_output(&output, "music WAV")?;
    let staging = ArtifactStaging::new_for_path_producer(&output)?;
    let options = Options {
        duration_seconds: duration,
        steps,
        seed,
        attention_query_chunk: attention_query_chunk_size,
    };
    let device = parse_device(&device)?;
    let mut model = Music3::open(
        &model,
        &device,
        weights.weight_source,
        weights.cache_policy()?,
        DeviceCache::disabled(),
    )?;
    let memory = model.request_memory(&prompt, &lyrics, &options)?;
    let demands = model.residency_demands(memory.frames, steps, &device)?;
    eprintln!(
        "Music3 known tensor peak: {} bytes; KV {} bytes, AR activations {} bytes, acoustic stages {} bytes; library/allocator margin remains separate",
        memory.known_device_reserve_bytes,
        memory.language_kv_peak_bytes,
        memory.autoregressive_activation_bytes,
        memory.acoustic.device_peak_bytes
    );
    model.configure_device_cache(super::decide_auto_residency_with_required_memory(
        &demands,
        &device,
        device_cache,
        if device.is_cpu() {
            0
        } else {
            memory.known_device_reserve_bytes
        },
    )?)?;
    let mut previous = String::new();
    let started = Instant::now();
    let result = model.generate(&prompt, &lyrics, &options, |stage, done, total| {
        if stage != previous || done == total || done.is_multiple_of(10) {
            eprintln!(
                "{stage}: {done}/{total} ({:.1}s)",
                started.elapsed().as_secs_f64()
            );
            previous = stage.to_string();
        }
    });
    for event in model.memory_releases() {
        eprintln!("Music3 memory release: {}", serde_json::to_string(event)?);
    }
    let result = result?;
    let waveform = result.waveform.squeeze(0)?;
    let peak = waveform.abs()?.max_all()?.to_scalar::<f32>()?;
    let rms = waveform.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
    anyhow::ensure!(
        peak.is_finite() && rms.is_finite(),
        "Music3 produced non-finite waveform values"
    );
    write_wav(
        staging.producer_path(),
        &waveform.unsqueeze(1)?,
        result.sample_rate,
        wav_format,
    )?;
    staging.publish()?;
    println!(
        "saved {}: stereo {} Hz, {:.3}s, {} frames, peak={peak:.6}, RMS={rms:.6}",
        output.display(),
        result.sample_rate,
        waveform.dim(1)? as f64 / f64::from(result.sample_rate),
        result.generated_frames
    );
    Ok(())
}
