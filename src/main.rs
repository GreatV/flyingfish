use anyhow::{Context, Result, bail, ensure};
use clap::{Args, Parser, Subcommand};
use flyingfish::backend;
use flyingfish::{
    bench as measure,
    config::Config,
    model::{Model, check_generation_capacity},
    tokenizer,
    trace::Trace,
};
#[cfg(feature = "cuda")]
use std::io::{Read, Write};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Parser)]
#[command(version)]
struct Cli {
    #[arg(long, env = "FLYINGFISH_MODEL")]
    model: PathBuf,
    #[arg(long, global = true, env = "FLYINGFISH_DRAFT_MODEL")]
    draft_model: Option<PathBuf>,
    #[arg(long, default_value_t = 0, global = true)]
    device: usize,
    #[arg(
        long,
        global = true,
        help = "Print placement, Lt and decode-plan diagnostics"
    )]
    verbose: bool,
    #[arg(long, default_value_t = 4096, global = true)]
    capacity: usize,
    #[arg(
        long,
        global = true,
        default_value_t = 958.0,
        help = "DRAM reference denominator in GB/s; default is an RTX 4090 reference value"
    )]
    dram_gb_s: f64,
    #[arg(
        long,
        global = true,
        default_value_t = 166.06,
        help = "GEMM reference denominator in TFLOP/s; default is an RTX 4090 reference value"
    )]
    gemm_tflops: f64,
    #[arg(
        long,
        global = true,
        help = "Prefill chunk override; default uses VRAM budget and tile shapes"
    )]
    chunk: Option<usize>,
    #[command(flatten)]
    kernels: backend::Settings,
    #[command(subcommand)]
    command: Command,
}

#[derive(Args)]
#[group(required = true, multiple = false)]
struct Input {
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long, value_delimiter = ',')]
    tokens: Option<Vec<u32>>,
    #[arg(long)]
    tokens_file: Option<PathBuf>,
}

impl Input {
    fn read(&self, model: &Path) -> Result<Vec<u32>> {
        if let Some(text) = &self.prompt {
            return tokenizer::encode(model, text);
        }
        if let Some(ids) = &self.tokens {
            return Ok(ids.clone());
        }
        let file = self.tokens_file.as_ref().context("input missing")?;
        let value: serde_json::Value = serde_json::from_slice(&fs::read(file)?)?;
        let ids = if value.is_array() {
            &value
        } else {
            value
                .get("input_ids")
                .context("tokens file requires input_ids")?
        };
        let ids = if ids.get(0).is_some_and(serde_json::Value::is_array) {
            &ids[0]
        } else {
            ids
        };
        serde_json::from_value(ids.clone()).context("input_ids must be an array of token ids")
    }
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum TranscriptionFormat {
    Txt,
    Json,
    Srt,
    Vtt,
}

impl TranscriptionFormat {
    #[cfg(feature = "cuda")]
    fn extension(self) -> &'static str {
        match self {
            Self::Txt => "txt",
            Self::Json => "json",
            Self::Srt => "srt",
            Self::Vtt => "vtt",
        }
    }
}

#[derive(Args)]
struct TranscribeArgs {
    input: PathBuf,
    #[arg(long, value_enum, default_value_t = TranscriptionFormat::Txt)]
    format: TranscriptionFormat,
    #[arg(long)]
    stream: bool,
    #[arg(long, default_value = "en")]
    source_lang: String,
    #[arg(long, default_value = "en")]
    target_lang: String,
    #[arg(long, conflicts_with = "no_pnc")]
    pnc: bool,
    #[arg(long, conflicts_with = "pnc")]
    no_pnc: bool,
    #[arg(
        long,
        help = "Diarization model directory selecting the multitalker path"
    )]
    diar_model: Option<PathBuf>,
    #[arg(
        long,
        requires = "diar_model",
        help = "Masked-ASR lane masking instead of the speaker-kernel default"
    )]
    masked: bool,
    #[arg(short = 'o', long = "output")]
    output: Option<PathBuf>,
}

#[derive(Args)]
struct DiarizeArgs {
    input: PathBuf,
    #[arg(long)]
    stream: bool,
    #[arg(long, default_value_t = flyingfish::diarization::DIARIZATION_THRESHOLD)]
    threshold: f64,
    #[arg(short = 'o', long = "output")]
    output: Option<PathBuf>,
}

#[derive(Args)]
#[group(required = true, multiple = false)]
struct EmbedInput {
    #[arg(long)]
    text: Option<String>,
    #[arg(long)]
    input: Option<PathBuf>,
}

#[derive(Args)]
struct EmbedArgs {
    #[command(flatten)]
    input: EmbedInput,
    #[arg(short = 'o', long = "output")]
    output: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Command {
    Calibrate,
    Generate {
        #[command(flatten)]
        input: Input,
        #[arg(long, default_value_t = 32)]
        max_new_tokens: usize,
        #[arg(long)]
        graph: bool,
        #[arg(long, value_parser = clap::value_parser!(u8).range(0..=4), help = "MTP depth; defaults to 4 when the checkpoint contains MTP weights for an architecture with MTP decode; 0 selects plain generation")]
        mtp: Option<u8>,
        #[arg(long)]
        dump: Option<PathBuf>,
        #[arg(
            long,
            requires = "dump",
            help = "Zero-based decoder layer for operator trace"
        )]
        trace_layer: Option<usize>,
        #[arg(
            long,
            help = "Processor-packed image input saved by the serve input trace"
        )]
        image_tensors: Option<PathBuf>,
    },
    Transcribe(TranscribeArgs),
    Diarize(DiarizeArgs),
    Embed(EmbedArgs),
    /// Image to 3D mesh.
    #[command(name = "to-3d")]
    To3d {
        #[arg(
            value_name = "IMAGE",
            help = "input PNG with alpha; the background must already be removed (RMBG-2.0 integration is pending)"
        )]
        image: PathBuf,
        #[arg(long, help = "DINOv3 encoder directory")]
        dino: Option<PathBuf>,
        #[arg(
            long,
            value_name = "IMAGE",
            help = "extra Pixal3D multiview RGBA input"
        )]
        view: Vec<PathBuf>,
        #[arg(
            long,
            value_name = "JSON",
            help = "camera JSON per view with camera_angle_x, distance and mesh_scale (MoGe-2 estimation is not integrated)"
        )]
        camera: Vec<PathBuf>,
        #[arg(long, help = "NAF feature extractor directory (Pixal3D)")]
        naf: Option<PathBuf>,
        #[arg(
            long,
            default_value_t = 0,
            help = "requested resolution; 0 selects the model default (Pixal3D 1536)"
        )]
        resolution: u32,
        #[arg(
            long,
            default_value_t = 0,
            help = "request seed for the stage noise draws"
        )]
        seed: u64,
        #[arg(
            short = 'o',
            long = "output",
            help = "output mesh (.ply or .obj); GLB, texture bake and render are out of scope"
        )]
        output: PathBuf,
    },
    Image {
        #[arg(long, help = "Image description in natural language")]
        prompt: String,
        #[arg(long, value_parser = parse_image_size, help = "Requested output size WIDTHxHEIGHT")]
        size: (usize, usize),
        #[arg(long, help = "Image noise seed (required; no documented default)")]
        seed: u64,
        #[arg(
            long,
            help = "Prompt-expander model directory; replaces the input prompt"
        )]
        expand: Option<PathBuf>,
        #[arg(long, action = clap::ArgAction::Append, help = "Upright RGB8 edit reference; repeat for multiple images")]
        edit: Vec<PathBuf>,
        #[arg(
            short = 'o',
            long = "output",
            help = "Output .png at the requested size"
        )]
        output: PathBuf,
        #[arg(long, help = "Print output size, seed and reference count as JSON")]
        json: bool,
    },
    Video {
        #[arg(long, help = "Video description in natural language")]
        prompt: String,
        #[arg(long, help = "Keyframe image (required for MiniMax-H3)")]
        image: Option<PathBuf>,
        #[arg(long, help = "Final keyframe for HyperFlow; alone or with --image")]
        last_image: Option<PathBuf>,
        #[arg(
            long,
            help = "MiniMax-H3 companion checkpoint for HyperFlow text encoder and VAEs"
        )]
        base_model: Option<PathBuf>,
        #[arg(
            long,
            help = "Requested frame count; rounds up to 17n+5 (136 -> 141), published duration 5-15 s at 24 fps"
        )]
        frames: usize,
        #[arg(long, help = "Noise seed (required; no documented default)")]
        seed: u64,
        #[arg(
            long,
            help = "Turbo LoRA safetensors, required for MiniMax-H3 and invalid for HyperFlow (e.g. minimax_h3_fl2v_turbo_8step_v1.1_768p_bf16.safetensors)"
        )]
        lora: Option<PathBuf>,
        #[arg(
            long,
            help = "Use a GGUF Q4_0 transformer instead of transformer safetensors"
        )]
        gguf: Option<PathBuf>,
        #[arg(
            short = 'o',
            long = "output",
            help = "Output .mp4 (requires ffmpeg on PATH) or .y4m with a sibling .wav"
        )]
        output: PathBuf,
        #[arg(
            long,
            help = "Print resolved output metadata, including the aligned frame count, as JSON"
        )]
        json: bool,
    },
    Speak {
        #[arg(value_name = "TEXT", help = "Text to synthesize, or - to read stdin")]
        text: String,
        #[arg(long, default_value_t = 42, help = "Sampling seed")]
        seed: u64,
        #[arg(
            short = 'o',
            long = "output",
            help = "Output mono 24 kHz WAV; stdout if omitted"
        )]
        output: Option<PathBuf>,
    },
    Music {
        #[arg(long, help = "Music description in natural language")]
        prompt: String,
        #[arg(long, help = "Lyrics with optional section tags, or - to read stdin")]
        lyrics: String,
        #[arg(long, help = "Sampling seed")]
        seed: u64,
        #[arg(long, default_value_t = 1500, help = "Maximum audio frames at 25 fps")]
        frames: usize,
        #[arg(
            short = 'o',
            long = "output",
            help = "Output stereo 44.1 kHz WAV; stdout if omitted"
        )]
        output: Option<PathBuf>,
    },
    Serve {
        #[arg(long, default_value_t = 8080)]
        port: u16,
        #[arg(long, value_parser = clap::value_parser!(u8).range(0..=4), help = "MTP depth; defaults to 4 when the checkpoint contains MTP weights for an architecture with MTP decode; 0 selects plain serving")]
        mtp: Option<u8>,
    },
    Bench {
        #[command(flatten)]
        input: Input,
        #[arg(long, default_value_t = 256)]
        steps: usize,
        #[arg(long, default_value_t = 32)]
        warmup: usize,
        #[arg(long, default_value_t = 3)]
        runs: usize,
        #[arg(long)]
        graph: bool,
    },
    ReplayDraft {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        thresholds: PathBuf,
    },
    ProfileSpec {
        #[command(flatten)]
        input: Input,
    },
    VerifySpec {
        #[command(flatten)]
        input: Input,
        #[arg(long, default_value_t = 64)]
        steps: usize,
        #[arg(long, requires = "prompt_name", required_unless_present = "strict")]
        logit_limits: Option<PathBuf>,
        #[arg(long, requires = "logit_limits")]
        prompt_name: Option<String>,
        #[arg(long, conflicts_with = "logit_limits")]
        strict: bool,
        #[arg(long)]
        reference: Option<PathBuf>,
    },
    BenchSpec {
        #[command(flatten)]
        input: Input,
        #[arg(long, default_value_t = 256)]
        steps: usize,
        #[arg(long, default_value_t = 3)]
        runs: usize,
    },
    CompareSpecGraph {
        #[command(flatten)]
        input: Input,
        #[arg(long, default_value_t = 64)]
        steps: usize,
    },
    CompareTreePadding {
        #[command(flatten)]
        input: Input,
        #[arg(long, default_value_t = 9)]
        nodes: usize,
    },
    CompareTreeBuilders {
        #[command(flatten)]
        input: Input,
        #[arg(long, default_value_t = 256)]
        steps: usize,
    },
    ReplayTree {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        dump: PathBuf,
        #[arg(long, value_parser=["fixture"])]
        commit_policy: String,
        #[arg(long)]
        reset_each_round: bool,
    },
    CheckTreeGreedy {
        #[command(flatten)]
        input: Input,
        #[arg(long, default_value_t = 256)]
        steps: usize,
        #[arg(long)]
        report: PathBuf,
        #[arg(long)]
        strict: bool,
        #[arg(long)]
        ignore_eos: bool,
    },
    Profile {
        #[command(flatten)]
        input: Input,
        #[arg(long, value_enum, default_value = "decode")]
        phase: measure::Phase,
        #[arg(long, default_value_t = 32)]
        warmup: usize,
        #[arg(long)]
        graph: bool,
    },
    Inspect,
    Verify {
        #[arg(long)]
        fixture: PathBuf,
        #[arg(long)]
        hidden_limits: PathBuf,
        #[arg(long)]
        logit_limits: PathBuf,
        #[arg(long)]
        graph: bool,
    },
    ComparePrefill {
        #[arg(long)]
        fixture: PathBuf,
        #[arg(long)]
        dump: PathBuf,
        #[arg(long)]
        hidden_limits: PathBuf,
        #[arg(long)]
        logit_limits: PathBuf,
    },
    CompareLog {
        #[arg(long)]
        fixture: PathBuf,
        #[arg(long)]
        log: PathBuf,
        #[arg(long)]
        hidden_limits: PathBuf,
        #[arg(long)]
        logit_limits: PathBuf,
    },
    Tokenize {
        #[arg(long)]
        prompt: String,
    },
}

fn output_path(path: &Path, model: &Path) -> Result<PathBuf> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent = parent
        .canonicalize()
        .context("dump parent directory must exist")?;
    let result = parent.join(path.file_name().context("dump filename missing")?);
    let resolved = if result.exists() {
        result.canonicalize()?
    } else {
        result.clone()
    };
    ensure!(
        !resolved.starts_with(model)
            && !resolved
                .components()
                .any(|p| matches!(p, std::path::Component::Normal(name) if name == "models")),
        "dump cannot be written inside readonly models directories"
    );
    ensure!(
        !result.exists(),
        "dump already exists: {}",
        result.display()
    );
    Ok(result)
}

fn generate_mtp(cli: &Cli, input: &Input, count: usize, graph: bool, depth: u8) -> Result<()> {
    ensure!(count > 0, "max-new-tokens must be positive");
    let ids = input.read(&cli.model)?;
    let policy = flyingfish::mtp::VerifyPolicy::default();
    let output = Model::generate_mtp(
        &cli.model,
        &ids,
        &flyingfish::spec::Options {
            device: cli.device,
            capacity: cli.capacity,
            chunk: cli.chunk,
            backend: cli.kernels.clone(),
            count,
            ignore_eos: false,
        },
        usize::from(depth),
        policy,
        graph,
    )?;
    let decode_steps = output.ids.len().saturating_sub(1);
    println!(
        "{}",
        serde_json::json!({
            "input_ids": ids,
            "generated_ids": output.ids,
            "text": tokenizer::decode(&cli.model, &output.ids)?,
            "load_ms": output.load_ms,
            "prefill_ms": output.prefill_ms,
            "decode_ms": output.decode_ms,
            "decode_steps": decode_steps,
            "decode_tok_s": if decode_steps > 0 { Some(decode_steps as f64 * 1000.0 / output.decode_ms) } else { None },
            "graph": graph,
            "mtp_depth": depth,
            "mode": "mtp",
            "mtp_verify": policy,
        })
    );
    Ok(())
}

fn generate(
    cli: &Cli,
    input: &Input,
    count: usize,
    graph: bool,
    dump: &Option<PathBuf>,
    trace_layer: Option<usize>,
    image_tensors: &Option<PathBuf>,
) -> Result<()> {
    ensure!(count > 0, "max-new-tokens must be positive");
    if let Some(layer) = trace_layer {
        let config = flyingfish::model_spec::ModelSpec::read(&cli.model)?;
        ensure!(
            layer < config.layers.len(),
            "trace layer {layer} is outside model"
        );
    }
    ensure!(
        !graph || dump.is_none(),
        "--graph and --dump cannot be combined"
    );
    let ids = input.read(&cli.model)?;
    let model_spec = match &image_tensors {
        Some(_) => Some(flyingfish::model_spec::ModelSpec::read(&cli.model)?),
        None => None,
    };
    let images = match (&image_tensors, &model_spec) {
        (Some(path), Some(spec)) => Some(flyingfish::serve::image::PreparedInput::read(
            spec, path, &ids,
        )?),
        _ => None,
    };
    if let (Some(images), Some(spec)) = (&images, &model_spec) {
        images.input(&ids).validate(spec, cli.capacity)?;
    }
    if let Some(draft) = &cli.draft_model {
        ensure!(
            trace_layer.is_none(),
            "DSpark generate does not support --trace-layer"
        );
        ensure!(
            !graph || cli.kernels.spec_graph,
            "--graph conflicts with --spec-graph false"
        );
        ensure!(
            dump.is_none() || !cli.kernels.spec_graph,
            "DSpark --dump requires --spec-graph false"
        );
        let draft = draft.canonicalize()?;
        eprintln!("draft_model={}", draft.display());
        let dump = dump
            .as_ref()
            .map(|p| output_path(p, &cli.model))
            .transpose()?;
        if let Some(path) = &dump {
            ensure!(
                !path.starts_with(&draft),
                "dump cannot be inside draft weights"
            );
        }
        return flyingfish::spec::generate(
            &cli.model,
            &draft,
            &ids,
            images.as_ref().map(|images| images.input(&ids)).as_ref(),
            &flyingfish::spec::Options {
                device: cli.device,
                capacity: cli.capacity,
                chunk: cli.chunk,
                backend: cli.kernels.clone(),
                count,
                ignore_eos: false,
            },
            dump.as_deref(),
        );
    }

    check_generation_capacity(ids.len(), count, cli.capacity)?;
    let dump = dump
        .as_ref()
        .map(|p| output_path(p, &cli.model))
        .transpose()?;
    let load_start = Instant::now();
    let mut m = Model::load(
        &cli.model,
        cli.device,
        cli.capacity,
        cli.chunk,
        cli.kernels.clone(),
    )?;
    if let Some(layer) = trace_layer {
        m.trace_layer(layer)?;
    }
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    let mut trace = dump.as_ref().map(|_| Trace::default());
    let output = if let Some(images) = &images {
        flyingfish::model::generate_multimodal(
            &mut m,
            &images.input(&ids),
            count,
            false,
            graph,
            trace.as_mut(),
            &mut |_| true,
        )?
    } else {
        flyingfish::model::generate_plain(
            &mut m,
            &ids,
            count,
            false,
            graph,
            trace.as_mut(),
            &mut |_| true,
        )?
    };
    if let (Some(t), Some(path)) = (&trace, &dump) {
        t.save(path)?;
    }
    println!(
        "{}",
        generation_report(
            &ids,
            &output,
            tokenizer::decode(&cli.model, &output.ids)?,
            load_ms,
            graph,
            &dump,
            trace_layer,
        )
    );
    Ok(())
}

fn generation_report(
    ids: &[u32],
    output: &flyingfish::model::GenerationOutput,
    text: String,
    load_ms: f64,
    graph: bool,
    dump: &Option<PathBuf>,
    trace_layer: Option<usize>,
) -> serde_json::Value {
    let decode_steps = output.ids.len() - 1;
    serde_json::json!({ "input_ids": ids, "generated_ids": output.ids,
        "text": text, "load_ms": load_ms, "prefill_ms": output.prefill_ms,
        "capture_ms": output.capture_ms, "decode_ms": output.decode_ms, "decode_steps": decode_steps,
        "decode_tok_s": if decode_steps > 0 { Some(decode_steps as f64 * 1000.0 / output.decode_ms) } else { None },
        "graph": graph, "dump": dump, "trace_layer": trace_layer.unwrap_or(0), "mode": "plain" })
}

fn bench(
    cli: &Cli,
    input: &Input,
    steps: usize,
    warmup: usize,
    runs: usize,
    graph: bool,
) -> Result<()> {
    let ids = input.read(&cli.model)?;
    measure::run(
        &cli.model,
        &ids,
        &measure::Settings {
            device: cli.device,
            capacity: cli.capacity,
            dram_gb_s: cli.dram_gb_s,
            gemm_tflops: cli.gemm_tflops,
            chunk: cli.chunk,
            kernels: cli.kernels.clone(),
            steps,
            warmup,
            runs,
            graph,
        },
    )
}

fn normalize_budget(cli: &mut Cli) {
    if cli.draft_model.is_none() {
        cli.kernels.spec_budget = backend::SpecBudget::Chain;
    }
}

fn resolve_mtp(requested: Option<u8>, component: bool, plain_selected: bool) -> Result<Option<u8>> {
    match requested {
        Some(0) => Ok(None),
        Some(depth) => {
            ensure!(depth <= 4, "MTP depth must be 0..=4");
            ensure!(
                !plain_selected,
                "MTP cannot combine --draft-model, --dump, --trace-layer or --image-tensors"
            );
            ensure!(
                component,
                "checkpoint has no MTP component with a bound decode path"
            );
            Ok(Some(depth))
        }
        None => Ok((component && !plain_selected).then_some(4)),
    }
}

fn checkpoint_mtp(dir: &Path, requested: Option<u8>, plain_selected: bool) -> Result<Option<u8>> {
    let component = if requested == Some(0) || (requested.is_none() && plain_selected) {
        false
    } else {
        flyingfish::checkpoint::has_mtp_component(dir)?
    };
    resolve_mtp(requested, component, plain_selected)
}

#[cfg(feature = "cuda")]
fn model_class(model: &Path) -> Result<String> {
    if model
        .extension()
        .is_some_and(|extension| extension == "nemo")
    {
        return Ok("nemo".into());
    }
    let config: serde_json::Value = serde_json::from_slice(&fs::read(model.join("config.json"))?)?;
    config["architectures"][0]
        .as_str()
        .map(str::to_string)
        .context("model architecture class is missing")
}

#[cfg(feature = "cuda")]
fn input_bytes(input: &Path) -> Result<Vec<u8>> {
    if input == Path::new("-") {
        let mut bytes = Vec::new();
        std::io::stdin().read_to_end(&mut bytes)?;
        Ok(bytes)
    } else {
        Ok(fs::read(input)?)
    }
}

#[cfg(feature = "cuda")]
fn write_bytes_output(
    output: Option<&Path>,
    model: &Path,
    extension: &str,
    bytes: &[u8],
) -> Result<()> {
    if let Some(output) = output {
        ensure!(
            output.extension().is_some_and(|value| value == extension),
            "output must end in .{extension}"
        );
        let output = output_path(output, model)?;
        if let Err(error) = fs::write(&output, bytes) {
            let _ = fs::remove_file(&output);
            return Err(error.into());
        }
        println!("{}", output.display());
    } else {
        print!("{}", std::str::from_utf8(bytes)?);
    }
    Ok(())
}

fn transcribe(cli: &Cli, args: &TranscribeArgs) -> Result<()> {
    let _ = (cli, args);
    #[cfg(not(feature = "cuda"))]
    bail!("transcribe requires CUDA");
    #[cfg(feature = "cuda")]
    flyingfish::backend::with_verbose(cli.verbose, || {
        let waveform = flyingfish::media::read_wav(input_bytes(&args.input)?.as_slice(), 16000)?;
        let class = model_class(&cli.model)?;
        if args.stream && class != "nemo" && class != "Nemotron3_5AsrForRNNT" {
            bail!(
                "streaming transcription is supported only for parakeet realtime EOU and Nemotron 3.5 ASR"
            );
        }
        let options = flyingfish::RunOptions {
            device: cli.device,
            runtime_dir: cli
                .kernels
                .runtime_dir
                .clone()
                .context("--runtime-dir is required for transcribe")?,
            verbose: cli.verbose,
        };
        let bytes = if let Some(diar_model) = &args.diar_model {
            ensure!(
                class == "nemo",
                "--diar-model selects the multitalker path; unsupported for architecture {class}"
            );
            let masks = flyingfish::diarization::speaker_masks(
                diar_model,
                &waveform,
                args.stream,
                flyingfish::diarization::DIARIZATION_THRESHOLD,
                options.clone(),
            )?;
            let result = flyingfish::multitalker::transcribe(
                &cli.model,
                &waveform,
                &masks,
                args.masked,
                options,
            )?;
            match args.format {
                TranscriptionFormat::Txt => {
                    let mut text = String::new();
                    for speaker in &result.speakers {
                        text.push_str(&format!(
                            "speaker_{} [{:.2}-{:.2}]\n{}\n",
                            speaker.speaker, speaker.start, speaker.end, speaker.text
                        ));
                    }
                    format!("{text}\n").into_bytes()
                }
                TranscriptionFormat::Json => {
                    format!("{}\n", serde_json::to_string(&result)?).into_bytes()
                }
                TranscriptionFormat::Srt | TranscriptionFormat::Vtt => {
                    let mut segments: Vec<flyingfish::media::TextSegment> = result
                        .speakers
                        .iter()
                        .flat_map(|speaker| {
                            speaker
                                .words
                                .iter()
                                .map(|word| flyingfish::media::TextSegment {
                                    start: word.start,
                                    end: word.end,
                                    text: format!("speaker_{}: {}", speaker.speaker, word.text),
                                })
                        })
                        .collect();
                    segments.sort_by(|a, b| {
                        a.start
                            .total_cmp(&b.start)
                            .then(a.end.total_cmp(&b.end))
                            .then_with(|| a.text.cmp(&b.text))
                    });
                    let mut bytes = Vec::new();
                    match args.format {
                        TranscriptionFormat::Srt => {
                            flyingfish::media::write_srt(&mut bytes, &segments)?
                        }
                        _ => flyingfish::media::write_vtt(&mut bytes, &segments)?,
                    }
                    bytes
                }
            }
        } else {
            let result = match class.as_str() {
                "ParakeetForTDT" => {
                    flyingfish::parakeet::transcribe(&cli.model, &waveform, options)?
                }
                "CanaryForConditionalGeneration" => flyingfish::canary::transcribe(
                    &cli.model,
                    &waveform,
                    &args.source_lang,
                    &args.target_lang,
                    !args.no_pnc,
                    options,
                )?,
                "nemo" => flyingfish::eou::transcribe(&cli.model, &waveform, args.stream, options)?,
                "Nemotron3_5AsrForRNNT" => {
                    flyingfish::nemotron::transcribe(&cli.model, &waveform, args.stream, options)?
                }
                _ => bail!("unsupported transcription architecture {class}"),
            };
            match args.format {
                TranscriptionFormat::Txt => format!("{}\n", result.text).into_bytes(),
                TranscriptionFormat::Json => {
                    format!("{}\n", serde_json::to_string(&result)?).into_bytes()
                }
                TranscriptionFormat::Srt => {
                    let mut bytes = Vec::new();
                    flyingfish::media::write_srt(&mut bytes, &result.segments)?;
                    bytes
                }
                TranscriptionFormat::Vtt => {
                    let mut bytes = Vec::new();
                    flyingfish::media::write_vtt(&mut bytes, &result.segments)?;
                    bytes
                }
            }
        };
        write_bytes_output(
            args.output.as_deref(),
            &cli.model,
            args.format.extension(),
            &bytes,
        )
    })
}

fn diarize(cli: &Cli, args: &DiarizeArgs) -> Result<()> {
    let _ = (cli, args);
    #[cfg(not(feature = "cuda"))]
    bail!("diarize requires CUDA");
    #[cfg(feature = "cuda")]
    flyingfish::backend::with_verbose(cli.verbose, || {
        let waveform = flyingfish::media::read_wav(input_bytes(&args.input)?.as_slice(), 16000)?;
        let class = model_class(&cli.model)?;
        ensure!(
            class == "Nemotron3DiarizationForAudioFrameClassification",
            "unsupported diarization architecture {class}"
        );
        let options = flyingfish::RunOptions {
            device: cli.device,
            runtime_dir: cli
                .kernels
                .runtime_dir
                .clone()
                .context("--runtime-dir is required for diarize")?,
            verbose: cli.verbose,
        };
        let segments = flyingfish::diarization::diarize(
            &cli.model,
            &waveform,
            args.stream,
            args.threshold,
            options,
        )?;
        let file_id = if args.input == Path::new("-") {
            "stdin".to_string()
        } else {
            args.input
                .file_stem()
                .context("input filename missing")?
                .to_string_lossy()
                .replace(char::is_whitespace, "_")
        };
        let mut bytes = Vec::new();
        flyingfish::media::write_rttm(&mut bytes, &file_id, &segments)?;
        write_bytes_output(args.output.as_deref(), &cli.model, "rttm", &bytes)
    })
}

fn embed(cli: &Cli, args: &EmbedArgs) -> Result<()> {
    let _ = (cli, args);
    #[cfg(not(feature = "cuda"))]
    bail!("embed requires CUDA");
    #[cfg(feature = "cuda")]
    flyingfish::backend::with_verbose(cli.verbose, || {
        let text = if let Some(text) = &args.input.text {
            text.clone()
        } else {
            String::from_utf8(input_bytes(
                args.input.input.as_deref().context("text input missing")?,
            )?)?
        };
        ensure!(
            flyingfish::model_spec::ModelSpec::read(&cli.model)?.architecture
                == flyingfish::model_spec::Architecture::EmbeddingGemma2,
            "embed requires EmbeddingGemma-2"
        );
        let options = flyingfish::RunOptions {
            device: cli.device,
            runtime_dir: cli
                .kernels
                .runtime_dir
                .clone()
                .context("--runtime-dir is required for embed")?,
            verbose: cli.verbose,
        };
        let values = flyingfish::model::embed_text(&cli.model, &text, options)?;
        let bytes = format!("{}\n", flyingfish::model::embedding_json(&values)?).into_bytes();
        write_bytes_output(args.output.as_deref(), &cli.model, "json", &bytes)
    })
}

fn parse_image_size(text: &str) -> Result<(usize, usize)> {
    let (width, height) = text
        .split_once('x')
        .context("image size must be WIDTHxHEIGHT")?;
    let width = width
        .parse::<usize>()
        .with_context(|| format!("invalid image width {width}"))?;
    let height = height
        .parse::<usize>()
        .with_context(|| format!("invalid image height {height}"))?;
    ensure!(width > 0 && height > 0, "image size must be positive");
    Ok((width, height))
}

#[cfg(any(feature = "cuda", test))]
fn png_output(path: &Path) -> Result<()> {
    ensure!(
        path.extension().is_some_and(|extension| extension == "png"),
        "image output must end in .png"
    );
    Ok(())
}

#[cfg(any(feature = "cuda", test))]
fn mp4_output(path: &Path) -> Result<bool> {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("mp4") => Ok(true),
        Some("y4m") => Ok(false),
        _ => bail!("video output must end in .mp4 or .y4m"),
    }
}

struct ImageCliRequest {
    prompt: String,
    width: usize,
    height: usize,
    seed: u64,
    expander: Option<PathBuf>,
    references: Vec<PathBuf>,
}

fn image(cli: &Cli, request: ImageCliRequest, output: &Path, json: bool) -> Result<()> {
    let _ = (
        cli,
        &request.prompt,
        &request.width,
        &request.height,
        &request.seed,
        &request.expander,
        &request.references,
        output,
        json,
    );
    #[cfg(not(feature = "cuda"))]
    bail!("image requires CUDA");
    #[cfg(feature = "cuda")]
    flyingfish::backend::with_verbose(cli.verbose, || {
        png_output(output)?;
        let output = output_path(output, &cli.model)?;
        let expander = request
            .expander
            .as_ref()
            .map(|path| {
                path.canonicalize()
                    .with_context(|| format!("resolve prompt expander {}", path.display()))
            })
            .transpose()?;
        let references = request
            .references
            .iter()
            .map(|path| {
                let path = path
                    .canonicalize()
                    .with_context(|| format!("resolve edit reference {}", path.display()))?;
                let bytes = fs::read(&path)
                    .with_context(|| format!("read edit reference {}", path.display()))?;
                Ok(flyingfish::minimax_h3::ImageReference {
                    image: flyingfish::serve::image::decode_image(&bytes, None)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut request = flyingfish::minimax_h3::ImageRequest::new(
            request.prompt,
            request.width,
            request.height,
            request.seed,
        );
        request.expander = expander;
        request.references = references;
        let options = flyingfish::minimax_h3::ImageRunOptions {
            device: cli.device,
            runtime_dir: cli
                .kernels
                .runtime_dir
                .clone()
                .context("--runtime-dir is required for image")?,
        };
        let generated = flyingfish::minimax_h3::image_latents(&cli.model, &options, &request)?;
        ensure!(
            generated.rgb.len() == request.width * request.height * 3,
            "image entry RGB extent"
        );
        if let Err(error) = flyingfish::media::write_png(
            fs::File::create(&output)?,
            request.width,
            request.height,
            &generated.rgb,
        ) {
            let _ = fs::remove_file(&output);
            return Err(error);
        }
        if json {
            println!(
                "{}",
                serde_json::json!({"image": {
                    "output": output.display().to_string(),
                    "width": request.width,
                    "height": request.height,
                    "seed": request.seed,
                    "references": request.references.len(),
                }})
            );
        } else {
            println!("{}", output.display());
        }
        Ok(())
    })
}

fn speak(cli: &Cli, text: &str, seed: u64, output: Option<&Path>) -> Result<()> {
    let _ = (cli, text, seed, output);
    #[cfg(not(feature = "cuda"))]
    bail!("speak requires CUDA");
    #[cfg(feature = "cuda")]
    flyingfish::backend::with_verbose(cli.verbose, || {
        let text = if text == "-" {
            let mut input = String::new();
            std::io::stdin().read_to_string(&mut input)?;
            input
        } else {
            text.to_owned()
        };
        let options = flyingfish::RunOptions {
            device: cli.device,
            runtime_dir: cli
                .kernels
                .runtime_dir
                .clone()
                .context("--runtime-dir is required for speak")?,
            verbose: cli.verbose,
        };
        let speech = flyingfish::breeze::speak(&cli.model, &text, seed, options)?;
        let mut bytes = Vec::new();
        flyingfish::media::write_wav(&mut bytes, speech.sampling_rate, 1, &speech.samples)?;
        if let Some(output) = output {
            ensure!(
                output.extension().is_some_and(|value| value == "wav"),
                "speak output must end in .wav"
            );
            let output = output_path(output, &cli.model)?;
            if let Err(error) = fs::write(&output, bytes) {
                let _ = fs::remove_file(&output);
                return Err(error.into());
            }
            println!("{}", output.display());
        } else {
            std::io::stdout().write_all(&bytes)?;
        }
        Ok(())
    })
}

fn music(
    cli: &Cli,
    prompt: &str,
    lyrics: &str,
    seed: u64,
    frames: usize,
    output: Option<&Path>,
) -> Result<()> {
    let _ = (cli, prompt, lyrics, seed, frames, output);
    #[cfg(not(feature = "cuda"))]
    bail!("music requires CUDA");
    #[cfg(feature = "cuda")]
    flyingfish::backend::with_verbose(cli.verbose, || {
        ensure!(frames > 0, "music frames must be positive");
        let lyrics = if lyrics == "-" {
            let mut input = String::new();
            std::io::stdin().read_to_string(&mut input)?;
            input
        } else {
            lyrics.to_owned()
        };
        let prompt_ids = flyingfish::music3::prompt_ids(&cli.model, prompt, &lyrics)?;
        let spec = flyingfish::music3::SynSpec::read(&cli.model)?;
        let runtime_dir = cli
            .kernels
            .runtime_dir
            .clone()
            .context("--runtime-dir is required for music")?;
        let result = flyingfish::music3::generate(
            &cli.model,
            &flyingfish::music3::Music3Request {
                prompt_ids,
                frames,
                seed,
                device: cli.device,
            },
            &runtime_dir,
        )?;
        let pcm = result.interleaved()?;
        let mut bytes = Vec::new();
        flyingfish::media::write_wav(&mut bytes, spec.output_rate as usize, 2, &pcm)?;
        if let Some(output) = output {
            ensure!(
                output.extension().is_some_and(|value| value == "wav"),
                "music output must end in .wav"
            );
            let output = output_path(output, &cli.model)?;
            if let Err(error) = fs::write(&output, bytes) {
                let _ = fs::remove_file(&output);
                return Err(error.into());
            }
            println!("{}", output.display());
        } else {
            std::io::stdout().write_all(&bytes)?;
        }
        Ok(())
    })
}

fn video(cli: &Cli) -> Result<()> {
    let _ = cli;
    #[cfg(not(feature = "cuda"))]
    bail!("video requires CUDA");
    #[cfg(feature = "cuda")]
    flyingfish::backend::with_verbose(cli.verbose, || {
        let Command::Video {
            prompt,
            image,
            last_image,
            base_model,
            frames,
            seed,
            lora,
            gguf,
            output,
            json,
        } = &cli.command
        else {
            bail!("video requires the video subcommand");
        };
        let hyperflow = flyingfish::minimax_h3::is_hyperflow_model(&cli.model)?;
        ensure!(
            hyperflow || (last_image.is_none() && base_model.is_none()),
            "--last-image and --base-model require HyperFlow"
        );
        ensure!(
            !hyperflow || (lora.is_none() && gguf.is_none()),
            "HyperFlow uses fused fl2va weights; --lora and --gguf are invalid"
        );
        let resolve = |path: &Path, role: &str| {
            path.canonicalize()
                .with_context(|| format!("resolve {role} {}", path.display()))
        };
        let base = if hyperflow {
            Some(resolve(
                base_model
                    .as_deref()
                    .context("--base-model is required for HyperFlow")?,
                "base model",
            )?)
        } else {
            None
        };
        let mp4 = mp4_output(output)?;
        let output = output_path(output, &cli.model)?;
        if let Some(base) = &base {
            output_path(&output, base)?;
        }
        let options = flyingfish::RunOptions {
            device: cli.device,
            runtime_dir: cli
                .kernels
                .runtime_dir
                .clone()
                .context("--runtime-dir is required for video")?,
            verbose: cli.verbose,
        };
        let generated = if let Some(base_model) = &base {
            let request = flyingfish::minimax_h3::HyperFlowH3Request {
                model: cli.model.clone(),
                base_model: base_model.clone(),
                first_frame: image
                    .as_ref()
                    .map(|p| resolve(p, "first image"))
                    .transpose()?,
                last_frame: last_image
                    .as_ref()
                    .map(|p| resolve(p, "last image"))
                    .transpose()?,
                prompt: prompt.clone(),
                frames: *frames,
                seed: *seed,
            };
            flyingfish::minimax_h3::run_hyperflow_h3_video(&request, options)
        } else {
            let request = flyingfish::minimax_h3::MiniMaxH3I2vaRequest {
                model: cli.model.clone(),
                lora: resolve(
                    lora.as_deref()
                        .context("--lora is required for MiniMax-H3")?,
                    "LoRA",
                )?,
                keyframe: resolve(
                    image
                        .as_deref()
                        .context("--image is required for MiniMax-H3 image-to-video+audio")?,
                    "keyframe",
                )?,
                gguf: gguf.as_ref().map(|p| resolve(p, "GGUF")).transpose()?,
                prompt: prompt.clone(),
                frames: *frames,
                seed: *seed,
            };
            flyingfish::minimax_h3::run_minimax_h3_i2va(&request, options)
        }?;
        let flyingfish::minimax_h3::H3Output {
            frames,
            height,
            width,
            video,
            samples,
            sampling_rate,
            audio,
        } = generated;
        ensure!(
            audio.len() == samples.checked_mul(2).context("audio extent overflow")?,
            "MiniMax-H3 audio extent"
        );
        let mut interleaved = Vec::with_capacity(audio.len());
        for index in 0..samples {
            interleaved.push(audio[index]);
            interleaved.push(audio[samples + index]);
        }
        let mut outputs = Vec::new();
        if mp4 {
            flyingfish::media::mux_mp4(
                &output,
                flyingfish::media::Y4mVideo {
                    width,
                    height,
                    fps: (flyingfish::minimax_h3::FPS, 1),
                    frames,
                    video,
                },
                Some((sampling_rate, 2, &interleaved)),
            )?;
            outputs.push(output);
        } else {
            let wav = output_path(&output.with_extension("wav"), &cli.model)?;
            if let Some(base) = &base {
                output_path(&wav, base)?;
            }
            let result = (|| -> Result<()> {
                flyingfish::media::write_y4m(
                    fs::File::create(&output)?,
                    width,
                    height,
                    (flyingfish::minimax_h3::FPS, 1),
                    frames,
                    &video,
                )?;
                flyingfish::media::write_wav(
                    fs::File::create(&wav)?,
                    sampling_rate,
                    2,
                    &interleaved,
                )?;
                Ok(())
            })();
            if let Err(error) = result {
                let _ = fs::remove_file(&output);
                let _ = fs::remove_file(&wav);
                return Err(error);
            }
            outputs.push(output);
            outputs.push(wav);
        }
        if *json {
            let outputs: Vec<_> = outputs
                .iter()
                .map(|path| path.display().to_string())
                .collect();
            println!(
                "{}",
                serde_json::json!({"video": {
                    "outputs": outputs,
                    "frames": frames,
                    "width": width,
                    "height": height,
                    "samples": samples,
                    "sampling_rate": sampling_rate,
                    "seed": seed,
                }})
            );
        } else {
            for path in outputs {
                println!("{}", path.display());
            }
        }
        Ok(())
    })
}

fn run() -> Result<()> {
    let mut cli = Cli::parse();
    normalize_budget(&mut cli);
    cli.model = cli
        .model
        .canonicalize()
        .context("resolve model directory")?;
    eprintln!("model={}", cli.model.display());
    match &cli.command {
        Command::Calibrate => {
            let settings = if let Some(draft) = &cli.draft_model {
                cli.kernels.with_draft(&draft.canonicalize()?)
            } else {
                cli.kernels.clone()
            };
            let _ = Model::calibrate(&cli.model, cli.device, cli.capacity, cli.chunk, settings)?;
            Ok(())
        }
        Command::Generate {
            input,
            max_new_tokens,
            graph,
            dump,
            trace_layer,
            image_tensors,
            mtp,
        } => {
            let plain_selected = cli.draft_model.is_some()
                || dump.is_some()
                || trace_layer.is_some()
                || image_tensors.is_some();
            if let Some(depth) = checkpoint_mtp(&cli.model, *mtp, plain_selected)? {
                generate_mtp(&cli, input, *max_new_tokens, *graph, depth)
            } else {
                generate(
                    &cli,
                    input,
                    *max_new_tokens,
                    *graph,
                    dump,
                    *trace_layer,
                    image_tensors,
                )
            }
        }
        Command::Transcribe(args) => transcribe(&cli, args),
        Command::Diarize(args) => diarize(&cli, args),
        Command::Embed(args) => embed(&cli, args),
        Command::Image {
            prompt,
            size,
            seed,
            expand,
            edit,
            output,
            json,
        } => {
            let request = ImageCliRequest {
                prompt: prompt.clone(),
                width: size.0,
                height: size.1,
                seed: *seed,
                expander: expand.clone(),
                references: edit.clone(),
            };
            image(&cli, request, output, *json)
        }
        #[cfg(feature = "cuda")]
        Command::To3d {
            image,
            dino,
            view,
            camera,
            naf,

            resolution,

            seed,
            output,
        } => {
            let runtime_dir = cli
                .kernels
                .runtime_dir
                .clone()
                .context("--runtime-dir is required for to-3d")?;
            // Model-class dispatch on the published configs: a diffusers-style
            // model_index.json is not to-3d input; the Trellis2
            // pipelines ship pipeline.json, and Pixal3D's single-view pipeline
            // carries the sparse_structure_decoder key TRELLIS.2 lacks.
            let index: Option<serde_json::Value> =
                match fs::read(cli.model.join("model_index.json")) {
                    Ok(bytes) => Some(serde_json::from_slice(&bytes)?),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                    Err(error) => return Err(error.into()),
                };
            let pipeline: Option<serde_json::Value> =
                match fs::read(cli.model.join("pipeline.json")) {
                    Ok(bytes) => Some(serde_json::from_slice(&bytes)?),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                    Err(error) => return Err(error.into()),
                };
            if let Some(index) = &index {
                bail!(
                    "unsupported model_index class {:?}: Hi3D is not enabled from to-3d yet",
                    index["_class_name"].as_str()
                );
            }
            enum Class {
                Trellis,
                Pixal { multiview: bool },
            }
            let class = if let Some(pipeline) = &pipeline {
                match pipeline["name"].as_str() {
                    Some("Trellis2ImageTo3DPipeline") => {
                        if pipeline["args"]["models"]["sparse_structure_decoder"].is_string() {
                            Class::Pixal {
                                multiview: !view.is_empty(),
                            }
                        } else {
                            Class::Trellis
                        }
                    }
                    Some("Pixal3DMVImageTo3DPipeline") => Class::Pixal { multiview: true },
                    other => bail!("unsupported pipeline class {other:?}"),
                }
            } else {
                bail!("model directory carries neither model_index.json nor pipeline.json");
            };
            let extension = output
                .extension()
                .and_then(|e| e.to_str())
                .map(str::to_owned);
            ensure!(
                matches!(extension.as_deref(), Some("ply") | Some("obj")),
                "output must be .ply or .obj, got {extension:?}"
            );
            let write = |mesh: &flyingfish::sparse::SparseMesh,
                         rgb: Option<&[u8]>,
                         extras: &[(&str, &[f32])],
                         resolution: u32|
             -> Result<()> {
                let view = flyingfish::media::MeshWrite {
                    positions: &mesh.vertices,
                    triangles: &mesh.faces,
                    rgb,
                    extras,
                };
                let mut buffered = std::io::BufWriter::new(fs::File::create(output)?);
                use std::io::Write as _;
                match extension.as_deref() {
                    Some("ply") => flyingfish::media::write_ply(&mut buffered, &view)?,
                    Some("obj") => flyingfish::media::write_obj(&mut buffered, &view)?,
                    _ => unreachable!(),
                }
                buffered.flush()?;
                println!(
                    "wrote {} vertices {} faces resolution {resolution} to {}",
                    mesh.vertices.len() / 3,
                    mesh.faces.len() / 3,
                    output.display()
                );
                Ok(())
            };
            match class {
                Class::Pixal { multiview } => {
                    let naf = naf
                        .as_ref()
                        .context("Pixal3D requires --naf (the NAF directory)")?;
                    let dino = dino
                        .as_ref()
                        .context("Pixal3D requires --dino (the DINOv3 directory)")?;
                    let mut paths = vec![image.clone()];
                    paths.extend(view.iter().cloned());
                    ensure!(
                        camera.len() == paths.len(),
                        "Pixal3D needs one --camera JSON per view ({} given for {} views)",
                        camera.len(),
                        paths.len()
                    );
                    let mut decoded = Vec::with_capacity(paths.len());
                    for path in &paths {
                        let (rgba, width, height) =
                            flyingfish::serve::image::decode_image_rgba(&fs::read(path)?, None)?;
                        decoded.push((rgba, width, height));
                    }
                    let mut views = Vec::with_capacity(paths.len());
                    for ((rgba, width, height), camera_path) in decoded.iter().zip(camera) {
                        let camera: serde_json::Value =
                            serde_json::from_slice(&fs::read(camera_path)?)?;
                        let fov = camera["camera_angle_x"]
                            .as_f64()
                            .context("camera_angle_x")?;
                        let scale = camera["mesh_scale"].as_f64().unwrap_or(1.0);
                        // The camera JSON may carry the publisher's
                        // world_to_camera extrinsic directly - a flat
                        // 12-float row-major 3x4, or the 4x4 nested form the
                        // fixture files store (top three rows taken). It is
                        // required for multiview, where a shared look-at
                        // would project every view from one direction;
                        // without it the single-view camera sits at
                        // (0, 0, distance) looking at the origin with up
                        // (0, 1, 0) (structured_latent_shape.py:135-138),
                        // the row-major
                        // [[1,0,0,0],[0,1,0,0],[0,0,1,distance]].
                        let world_to_camera: [f32; 12] = if let Some(matrix) =
                            camera["world_to_camera"].as_array()
                        {
                            let flat: Vec<f64> = if matrix.len() == 12 {
                                matrix
                                    .iter()
                                    .map(|value| value.as_f64().context("world_to_camera entry"))
                                    .collect::<Result<_>>()?
                            } else {
                                ensure!(
                                    matrix.len() == 4
                                        && matrix.iter().all(|row| row
                                            .as_array()
                                            .is_some_and(|row| row.len() == 4)),
                                    "world_to_camera must be 12 floats or a 4x4 matrix"
                                );
                                matrix
                                    .iter()
                                    .take(3)
                                    .flat_map(|row| row.as_array().expect("checked"))
                                    .map(|value| value.as_f64().context("world_to_camera entry"))
                                    .collect::<Result<_>>()?
                            };
                            let values: Vec<f32> =
                                flat.into_iter().map(|value| value as f32).collect();
                            values
                                .try_into()
                                .map_err(|_| anyhow::anyhow!("world_to_camera needs 12 entries"))?
                        } else if multiview {
                            bail!(
                                "multiview requires per-view world_to_camera in each camera JSON"
                            );
                        } else {
                            let distance = camera["distance"].as_f64().context("distance")?;
                            [
                                1.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                1.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                1.0,
                                distance as f32,
                            ]
                        };
                        views.push(flyingfish::backend::PixalView {
                            rgba,
                            dimensions: [*height, *width],
                            camera: flyingfish::backend::ProjectionCamera {
                                world_to_camera,
                                horizontal_fov: fov as f32,
                                mesh_scale: scale as f32,
                            },
                        });
                    }
                    let requested_resolution = match *resolution {
                        0 => 1536,
                        value => value,
                    };
                    let device = flyingfish::backend::BackendDevice::new(cli.device)?;
                    let output = flyingfish::backend::PixalSeededRequest {
                        model: &cli.model,
                        dino,
                        naf,
                        runtime: &runtime_dir,
                        views: &views,
                        multiview,
                        requested_resolution,
                    }
                    .run(*seed, &device)?;
                    let rgb: Vec<u8> = output
                        .material
                        .features
                        .chunks(output.material.channels)
                        .flat_map(|row| {
                            [row[0], row[1], row[2]]
                                .iter()
                                .map(|v| (v * 255.0).clamp(0.0, 255.0) as u8)
                                .collect::<Vec<u8>>()
                        })
                        .collect();
                    let metallic: Vec<f32> = output
                        .material
                        .features
                        .chunks(output.material.channels)
                        .map(|row| row[3])
                        .collect();
                    let roughness: Vec<f32> = output
                        .material
                        .features
                        .chunks(output.material.channels)
                        .map(|row| row[4])
                        .collect();
                    let alpha: Vec<f32> = output
                        .material
                        .features
                        .chunks(output.material.channels)
                        .map(|row| row[5])
                        .collect();
                    write(
                        &output.mesh,
                        Some(&rgb),
                        &[
                            ("metallic", &metallic),
                            ("roughness", &roughness),
                            ("alpha", &alpha),
                        ],
                        output.resolution,
                    )?;
                    Ok(())
                }
                Class::Trellis => {
                    ensure!(
                        view.is_empty() && camera.is_empty() && naf.is_none() && *resolution == 0,
                        "--view/--camera/--naf/--resolution serve Pixal3D"
                    );
                    let dino = dino
                        .as_ref()
                        .context("TRELLIS.2 requires --dino (the DINOv3 directory)")?;
                    let (rgba, width, height) =
                        flyingfish::serve::image::decode_image_rgba(&fs::read(image)?, None)?;
                    let request = flyingfish::backend::TrellisTo3d {
                        model: &cli.model,
                        dino,
                        runtime: &runtime_dir,
                        rgba: &rgba,
                        dimensions: [height, width],
                        seed: *seed,
                        device: cli.device,
                    };
                    let mesh = request.run()?;
                    let rgb: Vec<u8> = mesh
                        .material
                        .chunks(6)
                        .flat_map(|row| {
                            [row[0], row[1], row[2]]
                                .iter()
                                .map(|v| (v * 255.0).clamp(0.0, 255.0) as u8)
                                .collect::<Vec<u8>>()
                        })
                        .collect();
                    let metallic: Vec<f32> = mesh.material.chunks(6).map(|row| row[3]).collect();
                    let roughness: Vec<f32> = mesh.material.chunks(6).map(|row| row[4]).collect();
                    let alpha: Vec<f32> = mesh.material.chunks(6).map(|row| row[5]).collect();
                    let mesh_view = flyingfish::sparse::SparseMesh {
                        vertices: mesh.vertices,
                        faces: mesh.faces,
                    };
                    write(
                        &mesh_view,
                        Some(&rgb),
                        &[
                            ("metallic", &metallic),
                            ("roughness", &roughness),
                            ("alpha", &alpha),
                        ],
                        mesh.resolution,
                    )?;
                    Ok(())
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Command::To3d { .. } => bail!("to-3d requires CUDA"),
        Command::Video { .. } => video(&cli),
        Command::Speak { text, seed, output } => speak(&cli, text, *seed, output.as_deref()),
        Command::Music {
            prompt,
            lyrics,
            seed,
            frames,
            output,
        } => music(&cli, prompt, lyrics, *seed, *frames, output.as_deref()),
        Command::Serve { port, mtp } => {
            let draft = cli
                .draft_model
                .as_ref()
                .map(|path| path.canonicalize())
                .transpose()?;
            if let Some(draft) = &draft {
                eprintln!("draft_model={}", draft.display());
            }
            if let Some(depth) = checkpoint_mtp(&cli.model, *mtp, draft.is_some())? {
                return flyingfish::serve::run_mtp(
                    cli.model.clone(),
                    flyingfish::spec::Options {
                        device: cli.device,
                        capacity: cli.capacity,
                        chunk: cli.chunk,
                        backend: cli.kernels.clone(),
                        count: 256,
                        ignore_eos: false,
                    },
                    cli.model.clone(),
                    *port,
                    usize::from(depth),
                );
            }
            flyingfish::serve::run(
                cli.model.clone(),
                draft,
                flyingfish::spec::Options {
                    device: cli.device,
                    capacity: cli.capacity,
                    chunk: cli.chunk,
                    backend: cli.kernels.clone(),
                    count: 256,
                    ignore_eos: false,
                },
                cli.model.clone(),
                *port,
            )
        }
        Command::Bench {
            input,
            steps,
            warmup,
            runs,
            graph,
        } => bench(&cli, input, *steps, *warmup, *runs, *graph),
        Command::Profile {
            input,
            phase,
            warmup,
            graph,
        } => {
            let ids = input.read(&cli.model)?;
            measure::profile(
                &cli.model,
                &ids,
                &measure::Settings {
                    device: cli.device,
                    capacity: cli.capacity,
                    dram_gb_s: cli.dram_gb_s,
                    gemm_tflops: cli.gemm_tflops,
                    chunk: cli.chunk,
                    kernels: cli.kernels.clone(),
                    steps: 1,
                    warmup: *warmup,
                    runs: 1,
                    graph: *graph,
                },
                *phase,
            )
        }
        Command::ProfileSpec { input } => {
            let draft = cli
                .draft_model
                .as_ref()
                .context("profile-spec requires --draft-model")?
                .canonicalize()?;
            flyingfish::spec::profile(
                &cli.model,
                &draft,
                &input.read(&cli.model)?,
                &flyingfish::spec::Options {
                    device: cli.device,
                    capacity: cli.capacity,
                    chunk: cli.chunk,
                    backend: cli.kernels.clone(),
                    count: 9,
                    ignore_eos: true,
                },
            )
        }
        Command::ReplayDraft { plan, thresholds } => {
            let draft = cli
                .draft_model
                .as_ref()
                .context("replay-draft requires --draft-model")?
                .canonicalize()?;
            eprintln!("draft_model={}", draft.display());
            flyingfish::spec::replay(
                &cli.model,
                &draft,
                &flyingfish::spec::Options {
                    device: cli.device,
                    capacity: cli.capacity,
                    chunk: cli.chunk,
                    backend: cli.kernels.clone(),
                    count: 1,
                    ignore_eos: true,
                },
                plan,
                thresholds,
            )
        }
        Command::VerifySpec {
            input,
            steps,
            logit_limits,
            prompt_name,
            strict,
            reference,
        } => {
            let draft = cli
                .draft_model
                .as_ref()
                .context("verify-spec requires --draft-model")?
                .canonicalize()?;
            eprintln!("draft_model={}", draft.display());
            let limits: Vec<f64> = if *strict {
                eprintln!("spec comparison=strict exact; near-tie exceptions disabled");
                vec![0.0; *steps]
            } else {
                let limits: serde_json::Value = serde_json::from_slice(&fs::read(
                    logit_limits.as_ref().context("missing logit limits")?,
                )?)?;
                serde_json::from_value(
                    limits
                        .get(prompt_name.as_ref().context("missing prompt name")?)
                        .and_then(|v| v.get("max_abs"))
                        .context("prompt has no max_abs thresholds")?
                        .clone(),
                )?
            };
            flyingfish::spec::verify_greedy(
                &cli.model,
                &draft,
                &input.read(&cli.model)?,
                &flyingfish::spec::Options {
                    device: cli.device,
                    capacity: cli.capacity,
                    chunk: cli.chunk,
                    backend: cli.kernels.clone(),
                    count: *steps,
                    ignore_eos: true,
                },
                &limits,
                reference.as_deref(),
                match (logit_limits.as_deref(), reference.as_deref()) {
                    (Some(limits), Some(reference)) => Some(flyingfish::verify::Rules::read(
                        None,
                        limits,
                        reference,
                        Config::read(&cli.model)?.num_hidden_layers,
                    )?),
                    _ => None,
                }
                .as_ref(),
            )
        }
        Command::ReplayTree {
            plan,
            dump,
            commit_policy,
            reset_each_round,
        } => {
            ensure!(
                commit_policy == "fixture" && *reset_each_round,
                "replay-tree requires --commit-policy fixture --reset-each-round"
            );
            let draft = cli
                .draft_model
                .as_ref()
                .context("replay-tree requires --draft-model")?
                .canonicalize()?;
            eprintln!("draft_model={}", draft.display());
            let dump = output_path(dump, &cli.model)?;
            flyingfish::tree_replay::run(
                &cli.model,
                &draft,
                plan,
                &dump,
                &flyingfish::spec::Options {
                    device: cli.device,
                    capacity: cli.capacity,
                    chunk: cli.chunk,
                    backend: cli.kernels.clone(),
                    count: 64,
                    ignore_eos: true,
                },
            )
        }
        Command::CheckTreeGreedy {
            input,
            steps,
            report,
            strict,
            ignore_eos,
        } => {
            ensure!(
                *strict && *ignore_eos,
                "check-tree-greedy requires --strict --ignore-eos"
            );
            let draft = cli
                .draft_model
                .as_ref()
                .context("check-tree-greedy requires --draft-model")?
                .canonicalize()?;
            let report = output_path(report, &cli.model)?;
            flyingfish::spec::check_tree_greedy(
                &cli.model,
                &draft,
                &input.read(&cli.model)?,
                &flyingfish::spec::Options {
                    device: cli.device,
                    capacity: cli.capacity,
                    chunk: cli.chunk,
                    backend: cli.kernels.clone(),
                    count: *steps,
                    ignore_eos: *ignore_eos,
                },
                &report,
            )
        }
        Command::CompareTreePadding { input, nodes } => {
            let draft = cli
                .draft_model
                .as_ref()
                .context("compare-tree-padding requires --draft-model")?
                .canonicalize()?;
            flyingfish::spec::compare_padding(
                &cli.model,
                &draft,
                &input.read(&cli.model)?,
                &flyingfish::spec::Options {
                    device: cli.device,
                    capacity: cli.capacity,
                    chunk: cli.chunk,
                    backend: cli.kernels.clone(),
                    count: 64,
                    ignore_eos: true,
                },
                *nodes,
            )
        }
        Command::CompareSpecGraph { input, steps } => {
            let draft = cli
                .draft_model
                .as_ref()
                .context("compare-spec-graph requires --draft-model")?
                .canonicalize()?;
            eprintln!("draft_model={}", draft.display());
            flyingfish::spec::compare_graph(
                &cli.model,
                &draft,
                &input.read(&cli.model)?,
                &flyingfish::spec::Options {
                    device: cli.device,
                    capacity: cli.capacity,
                    chunk: cli.chunk,
                    backend: cli.kernels.clone(),
                    count: *steps,
                    ignore_eos: true,
                },
            )
        }
        Command::CompareTreeBuilders { input, steps } => {
            let draft = cli
                .draft_model
                .as_ref()
                .context("compare-tree-builders requires --draft-model")?
                .canonicalize()?;
            eprintln!("draft_model={}", draft.display());
            flyingfish::spec::compare_builders(
                &cli.model,
                &draft,
                &input.read(&cli.model)?,
                &flyingfish::spec::Options {
                    device: cli.device,
                    capacity: cli.capacity,
                    chunk: cli.chunk,
                    backend: cli.kernels.clone(),
                    count: *steps,
                    ignore_eos: true,
                },
            )
        }
        Command::BenchSpec { input, steps, runs } => {
            let draft = cli
                .draft_model
                .as_ref()
                .context("bench-spec requires --draft-model")?
                .canonicalize()?;
            eprintln!("draft_model={}", draft.display());
            flyingfish::spec::bench(
                &cli.model,
                &draft,
                &input.read(&cli.model)?,
                &flyingfish::spec::Options {
                    device: cli.device,
                    capacity: cli.capacity,
                    chunk: cli.chunk,
                    backend: cli.kernels.clone(),
                    count: *steps,
                    ignore_eos: true,
                },
                *runs,
            )
        }
        Command::Inspect => {
            let c = Config::read(&cli.model)?;
            println!("{c:#?}");
            Ok(())
        }
        Command::Verify {
            fixture,
            hidden_limits,
            logit_limits,
            graph,
        } => {
            let config = Config::read(&cli.model)?;
            let rules = flyingfish::verify::Rules::read(
                Some(hidden_limits),
                logit_limits,
                fixture,
                config.num_hidden_layers,
            )?;
            let mut model = Model::load(
                &cli.model,
                cli.device,
                cli.capacity,
                cli.chunk,
                cli.kernels.clone(),
            )?;
            flyingfish::verify::run(&mut model, fixture, &rules, *graph)
        }
        Command::ComparePrefill {
            fixture,
            dump,
            hidden_limits,
            logit_limits,
        } => {
            let config = Config::read(&cli.model)?;
            let rules = flyingfish::verify::Rules::read(
                Some(hidden_limits),
                logit_limits,
                fixture,
                config.num_hidden_layers,
            )?;
            flyingfish::verify::compare_prefill(
                fixture,
                dump,
                config.vocab_size,
                config.hidden_size,
                &rules,
            )
        }
        Command::CompareLog {
            fixture,
            log,
            hidden_limits,
            logit_limits,
        } => {
            let config = Config::read(&cli.model)?;
            let rules = flyingfish::verify::Rules::read(
                Some(hidden_limits),
                logit_limits,
                fixture,
                config.num_hidden_layers,
            )?;
            flyingfish::verify::compare_saved(fixture, log, &rules)
        }
        Command::Tokenize { prompt } => {
            println!(
                "{}",
                serde_json::to_string(&tokenizer::encode(&cli.model, prompt)?)?
            );
            Ok(())
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_parse_covers_size_expander_edits_and_png_output() -> Result<()> {
        let cli = Cli::try_parse_from([
            "flyingfish",
            "--model",
            "model",
            "--runtime-dir",
            "runtime",
            "--device",
            "1",
            "image",
            "--prompt",
            "a red cube",
            "--size",
            "1536x1024",
            "--seed",
            "42",
            "--expand",
            "pe",
            "--edit",
            "first.png",
            "--edit",
            "second.png",
            "-o",
            "out.png",
            "--json",
        ])?;
        assert_eq!(cli.device, 1);
        match cli.command {
            Command::Image {
                prompt,
                size,
                seed,
                expand,
                edit,
                output,
                json,
            } => {
                assert_eq!(prompt, "a red cube");
                assert_eq!(size, (1536, 1024));
                assert_eq!(seed, 42);
                assert_eq!(expand, Some(PathBuf::from("pe")));
                assert_eq!(
                    edit,
                    [PathBuf::from("first.png"), PathBuf::from("second.png")]
                );
                assert_eq!(output, PathBuf::from("out.png"));
                assert!(json);
            }
            _ => panic!("image subcommand was not selected"),
        }
        let help = {
            use clap::CommandFactory;
            Cli::command()
                .find_subcommand_mut("image")
                .expect("image subcommand")
                .render_help()
                .to_string()
        };
        assert!(help.contains("WIDTHxHEIGHT"), "{help}");
        assert!(help.contains("required; no documented default"), "{help}");
        assert!(help.contains("replaces the input prompt"), "{help}");
        assert!(help.contains("repeat for multiple images"), "{help}");
        assert!(help.contains("Output .png"), "{help}");
        assert!(help.contains("reference count as JSON"), "{help}");
        for invalid in ["0x1024", "1024", "1024X1024", "-1x2"] {
            assert!(parse_image_size(invalid).is_err(), "accepted {invalid}");
        }
        assert!(png_output(Path::new("out.jpg")).is_err());
        assert!(
            Cli::try_parse_from([
                "flyingfish",
                "--model",
                "model",
                "image",
                "--prompt",
                "a red cube"
            ])
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn video_help_documents_required_keyframe_and_frame_rounding() {
        use clap::CommandFactory;

        let mut command = Cli::command();
        let help = command
            .find_subcommand_mut("video")
            .expect("video subcommand")
            .render_help()
            .to_string();
        assert!(help.contains("required for MiniMax-H3"), "{help}");
        assert!(help.contains("17n+5"), "{help}");
        assert!(help.contains("5-15 s at 24 fps"), "{help}");
        assert!(help.contains("requires ffmpeg on PATH"), "{help}");
        assert!(help.contains("sibling .wav"), "{help}");
    }

    #[test]
    fn phase_one_commands_parse_audio_text_and_output_options() -> Result<()> {
        let cli = Cli::try_parse_from([
            "flyingfish",
            "--model",
            "model",
            "--runtime-dir",
            "runtime",
            "transcribe",
            "clip.wav",
            "--format",
            "json",
            "--stream",
            "--source-lang",
            "en",
            "--target-lang",
            "de",
            "--no-pnc",
            "-o",
            "transcript.json",
        ])?;
        match cli.command {
            Command::Transcribe(args) => {
                assert_eq!(args.input, PathBuf::from("clip.wav"));
                assert!(matches!(args.format, TranscriptionFormat::Json));
                assert!(args.stream);
                assert_eq!(args.source_lang, "en");
                assert_eq!(args.target_lang, "de");
                assert!(args.no_pnc);
                assert_eq!(args.output, Some(PathBuf::from("transcript.json")));
            }
            _ => panic!("transcribe subcommand was not selected"),
        }
        let cli = Cli::try_parse_from([
            "flyingfish",
            "--model",
            "model",
            "diarize",
            "meeting.wav",
            "--stream",
            "--threshold",
            "0.5",
            "-o",
            "meeting.rttm",
        ])?;
        match cli.command {
            Command::Diarize(args) => {
                assert_eq!(args.input, PathBuf::from("meeting.wav"));
                assert!(args.stream);
                assert_eq!(args.threshold, 0.5);
                assert_eq!(args.output, Some(PathBuf::from("meeting.rttm")));
            }
            _ => panic!("diarize subcommand was not selected"),
        }
        let cli = Cli::try_parse_from([
            "flyingfish",
            "--model",
            "model",
            "embed",
            "--text",
            "a red cube",
            "-o",
            "embedding.json",
        ])?;
        match cli.command {
            Command::Embed(args) => {
                assert_eq!(args.input.text.as_deref(), Some("a red cube"));
                assert_eq!(args.output, Some(PathBuf::from("embedding.json")));
            }
            _ => panic!("embed subcommand was not selected"),
        }
        Ok(())
    }

    #[test]
    fn video_parse_covers_i2va_media_and_output_options() -> Result<()> {
        let cli = Cli::try_parse_from([
            "flyingfish",
            "--model",
            "model",
            "--runtime-dir",
            "runtime",
            "--device",
            "1",
            "video",
            "--prompt",
            "a boat",
            "--image",
            "keyframe.png",
            "--frames",
            "121",
            "--seed",
            "42",
            "--lora",
            "turbo.safetensors",
            "--gguf",
            "transformer.gguf",
            "-o",
            "out.y4m",
            "--json",
        ])?;
        assert_eq!(cli.device, 1);
        assert_eq!(cli.kernels.runtime_dir, Some(PathBuf::from("runtime")));
        match cli.command {
            Command::To3d { .. } => {}
            Command::Video {
                prompt,
                image,
                last_image,
                base_model,
                frames,
                seed,
                lora,
                gguf,
                output,
                json,
            } => {
                assert_eq!(prompt, "a boat");
                assert_eq!(image, Some(PathBuf::from("keyframe.png")));
                assert_eq!(frames, 121);
                assert_eq!(seed, 42);
                assert_eq!(lora, Some(PathBuf::from("turbo.safetensors")));
                assert!(last_image.is_none() && base_model.is_none());
                assert_eq!(gguf, Some(PathBuf::from("transformer.gguf")));
                assert_eq!(output, PathBuf::from("out.y4m"));
                assert!(json);
            }
            _ => panic!("video subcommand was not selected"),
        }
        assert!(mp4_output(Path::new("out.mp4"))?);
        assert!(!mp4_output(Path::new("out.y4m"))?);
        assert!(mp4_output(Path::new("out.avi")).is_err());
        assert!(
            Cli::try_parse_from([
                "flyingfish",
                "--model",
                "model",
                "video",
                "--prompt",
                "a boat"
            ])
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn speak_parse_covers_text_stdin_seed_and_wav_output() -> Result<()> {
        let cli = Cli::try_parse_from([
            "flyingfish",
            "--model",
            "model",
            "--runtime-dir",
            "runtime",
            "speak",
            "-",
            "--seed",
            "7",
            "-o",
            "speech.wav",
        ])?;
        match cli.command {
            Command::Speak { text, seed, output } => {
                assert_eq!(text, "-");
                assert_eq!(seed, 7);
                assert_eq!(output, Some(PathBuf::from("speech.wav")));
            }
            _ => panic!("speak subcommand was not selected"),
        }
        let default = Cli::try_parse_from(["flyingfish", "--model", "model", "speak", "Hello"])?;
        assert!(matches!(
            default.command,
            Command::Speak {
                seed: 42,
                output: None,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn mtp_default_resolves_from_checkpoint_and_explicit_plain_modes() -> Result<()> {
        let args = [
            "flyingfish",
            "--model",
            "model",
            "generate",
            "--tokens",
            "1",
        ];
        assert!(matches!(
            Cli::try_parse_from(args)?.command,
            Command::Generate { mtp: None, .. }
        ));
        for depth in ["0", "1", "2", "3", "4"] {
            let cli = Cli::try_parse_from(args.into_iter().chain(["--mtp", depth]))?;
            assert!(
                matches!(cli.command, Command::Generate { mtp: Some(k), .. } if k.to_string() == depth)
            );
        }
        assert!(Cli::try_parse_from(args.into_iter().chain(["--mtp", "5"])).is_err());
        assert_eq!(resolve_mtp(None, true, false)?, Some(4));
        assert_eq!(resolve_mtp(None, false, false)?, None);
        for component in [false, true] {
            assert_eq!(resolve_mtp(Some(0), component, false)?, None);
            assert_eq!(resolve_mtp(Some(0), component, true)?, None);
            assert_eq!(resolve_mtp(None, component, true)?, None);
        }
        for depth in 1..=4 {
            assert_eq!(resolve_mtp(Some(depth), true, false)?, Some(depth));
            assert!(resolve_mtp(Some(depth), false, false).is_err());
            assert!(resolve_mtp(Some(depth), true, true).is_err());
        }
        Ok(())
    }

    #[test]
    fn generate_graph_json_records_the_resolved_plain_mode() {
        let output = flyingfish::model::GenerationOutput {
            ids: vec![7, 8, 9],
            prefill_ms: 2.0,
            capture_ms: 1.25,
            decode_ms: 4.0,
        };
        let report = generation_report(&[1, 2], &output, "answer".into(), 5.0, true, &None, None);
        assert_eq!(
            report,
            serde_json::json!({
                "input_ids": [1, 2], "generated_ids": [7, 8, 9], "text": "answer",
                "load_ms": 5.0, "prefill_ms": 2.0, "capture_ms": 1.25,
                "decode_ms": 4.0, "decode_steps": 2, "decode_tok_s": 500.0,
                "graph": true, "dump": null, "trace_layer": 0, "mode": "plain"
            })
        );
        let first_only = flyingfish::model::GenerationOutput {
            ids: vec![7],
            ..output
        };
        let first_report = generation_report(
            &[1, 2],
            &first_only,
            "answer".into(),
            5.0,
            true,
            &None,
            None,
        );
        assert_eq!(first_report["decode_steps"], 0);
        assert_eq!(first_report["decode_tok_s"], serde_json::Value::Null);
        assert_eq!(
            first_report.as_object().unwrap().keys().collect::<Vec<_>>(),
            report.as_object().unwrap().keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn generation_capacity_accounts_for_prefill_token_and_capture_slot() -> Result<()> {
        for (count, required) in [(1, 4), (2, 4), (5, 7)] {
            check_generation_capacity(3, count, required)?;
            let error = check_generation_capacity(3, count, required - 1)
                .expect_err("one-row-short generation capacity must fail")
                .to_string();
            assert!(error.contains(&format!("require KV capacity {required}")));
        }
        assert!(check_generation_capacity(3, 0, 4).is_err());
        assert!(check_generation_capacity(usize::MAX, 1, usize::MAX).is_err());
        Ok(())
    }
}
