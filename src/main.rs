use anyhow::{Context, Result, ensure};
use clap::{Args, Parser, Subcommand};
use flyingfish::backend;
use flyingfish::{
    bench as measure,
    config::Config,
    model::{Model, check_generation_capacity},
    tokenizer,
    trace::Trace,
};
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
