//! Resolve a task to a registered model adapter before parsing its options.

use super::adapters::{Adapter, BUILTINS, Metadata, Task};
use anyhow::{Context, Result, bail, ensure};
use clap::{Arg, Command, CommandFactory, FromArgMatches};
use std::{collections::BTreeMap, ffi::OsString, path::PathBuf};

pub(super) struct Registry<'a> {
    adapters: &'a [Adapter],
}

impl<'a> Registry<'a> {
    pub(super) fn new(adapters: &'a [Adapter]) -> Self {
        Self { adapters }
    }

    fn for_task(&self, task: Task) -> impl Iterator<Item = &'a Adapter> + '_ {
        self.adapters
            .iter()
            .filter(move |adapter| adapter.task == task)
    }

    fn select(&self, task: Task, args: &[OsString]) -> Result<Option<&'a Adapter>> {
        let explicit = option_value(args, "--adapter");
        let model = option_value(args, "--model").map(PathBuf::from);
        let help = args.iter().any(|arg| arg == "--help" || arg == "-h");
        let providers = self.for_task(task).collect::<Vec<_>>();
        let selected = explicit
            .as_ref()
            .map(|id| {
                providers
                    .iter()
                    .copied()
                    .find(|adapter| id == adapter.id)
                    .with_context(|| {
                        format!(
                            "unknown adapter {:?} for {}; available: {}",
                            id,
                            task.name(),
                            providers
                                .iter()
                                .map(|a| a.id)
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    })
            })
            .transpose()?;
        if help && selected.is_some() {
            return Ok(selected);
        }
        if let Some(model) = model {
            // General help remains usable before downloading a checkpoint.
            if help && !model.is_dir() {
                return Ok(None);
            }
            let metadata = Metadata::read(&model)?;
            if let Some(adapter) = selected {
                ensure!(
                    (adapter.recognizes)(&metadata),
                    "adapter {} does not support the metadata in {}",
                    adapter.id,
                    model.display()
                );
                return Ok(Some(adapter));
            }
            let recognized = self
                .adapters
                .iter()
                .filter(|adapter| (adapter.recognizes)(&metadata))
                .collect::<Vec<_>>();
            let candidates = recognized
                .iter()
                .copied()
                .filter(|adapter| adapter.task == task)
                .collect::<Vec<_>>();
            return match candidates.as_slice() {
                [adapter] => Ok(Some(*adapter)),
                [] if recognized.is_empty() => {
                    bail!(
                        "no registered adapter recognizes the model in {}",
                        model.display()
                    )
                }
                [] => bail!(
                    "model in {} does not support the {} task; available tasks: {}",
                    model.display(),
                    task.name(),
                    recognized
                        .iter()
                        .map(|a| a.task.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                _ => bail!(
                    "multiple adapters support this model; select --adapter from: {}",
                    candidates
                        .iter()
                        .map(|a| a.id)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            };
        }
        if selected.is_some() {
            return Ok(selected);
        }
        // Operations on saved artifacts may have no model argument. A unique
        // provider can still own such an operation (for example video history).
        let operation = args
            .iter()
            .filter_map(|value| value.to_str())
            .find(|value| {
                providers.iter().any(|adapter| {
                    (adapter.command)()
                        .get_subcommands()
                        .any(|command| command.get_name() == *value)
                })
            });
        let candidates = providers
            .into_iter()
            .filter(|adapter| {
                operation.is_none_or(|name| {
                    (adapter.command)()
                        .get_subcommands()
                        .any(|command| command.get_name() == name)
                })
            })
            .collect::<Vec<_>>();
        Ok(match candidates.as_slice() {
            [adapter] => Some(*adapter),
            _ => None,
        })
    }

    fn task_command(&self, task: Task, selected: Option<&Adapter>) -> Command {
        let providers = self.for_task(task).collect::<Vec<_>>();
        let mut command = if let Some(adapter) = selected {
            (adapter.command)()
        } else {
            let mut operations = BTreeMap::<String, Vec<Command>>::new();
            for adapter in &providers {
                for operation in (adapter.command)().get_subcommands() {
                    operations
                        .entry(operation.get_name().to_owned())
                        .or_default()
                        .push(operation.clone());
                }
            }
            let mut command = Command::new(task.name());
            for (name, providers) in operations {
                let operation = if providers.len() == 1 {
                    providers.into_iter().next().unwrap()
                } else {
                    common_command(&name, &providers)
                };
                command = command.subcommand(operation);
            }
            command
        };
        let ids = providers
            .iter()
            .map(|adapter| adapter.id)
            .collect::<Vec<_>>();
        command = command
            .name(task.name())
            .about(task.about())
            .subcommand_required(true)
            .arg_required_else_help(true)
            .arg(
                Arg::new("adapter")
                    .long("adapter")
                    .value_name("ADAPTER")
                    .global(true)
                    .value_parser(clap::builder::PossibleValuesParser::new(ids.clone()))
                    .help("Select a model adapter explicitly; otherwise detect from --model"),
            );
        command.after_help(format!(
            "Adapters: {}. Options and defaults follow the selected model.\n\
             To see a model's full options without a checkpoint, use --adapter <name> with --help.",
            ids.join(", ")
        ))
    }

    fn root_command(&self, selected: Option<&Adapter>) -> Command {
        let mut root = super::Args::command();
        let utilities = root
            .get_subcommands()
            .map(|command| command.get_name().to_owned())
            .collect::<Vec<_>>();
        for (index, name) in utilities.iter().enumerate() {
            root = root.mut_subcommand(name, |command| command.display_order(100 + index));
        }
        // Existing automation can still invoke the old model entry points.
        // Newly registered adapters are reached entirely through task commands.
        for alias in ["h3", "glm", "minicpm", "trellis", "clip"] {
            root = root.mut_subcommand(alias, |command| command.hide(true));
        }
        for (index, task) in Task::ALL.into_iter().enumerate() {
            let chosen = selected.filter(|adapter| adapter.task == task);
            let command = self.task_command(task, chosen).display_order(index);
            root = if root.find_subcommand(task.name()).is_some() {
                root.mut_subcommand(task.name(), |_| command)
            } else {
                root.subcommand(command)
            };
        }
        root
    }
}

/// Generic help contains only options shared by all providers of an operation.
/// Actual execution always parses the selected adapter's complete schema, so
/// unsupported flags cannot be silently ignored or acquire another model's defaults.
fn common_command(name: &str, providers: &[Command]) -> Command {
    let mut command =
        Command::new(name.to_owned()).about("Run this operation with a compatible model");
    for source in providers[0].get_arguments() {
        let Some(long) = source.get_long() else {
            continue;
        };
        if !providers.iter().all(|provider| {
            provider
                .get_arguments()
                .any(|arg| arg.get_id() == source.get_id() && arg.get_long() == Some(long))
        }) {
            continue;
        }
        // Rebuild presentation without model-specific dependency groups or defaults.
        let mut arg = Arg::new(source.get_id().clone())
            .long(long.to_owned())
            .action(source.get_action().clone())
            .value_parser(source.get_value_parser().clone())
            .required(long == "model");
        if long == "model" {
            arg = arg.help("Checkpoint directory; detect the adapter from its configuration");
        } else if let Some(help) = source.get_help() {
            arg = arg.help(help.clone());
        }
        if let Some(count) = source.get_num_args() {
            arg = arg.num_args(count);
        }
        if let Some(names) = source.get_value_names() {
            arg = arg.value_names(names.iter().cloned());
        }
        command = command.arg(arg);
    }
    command.after_help("Use --adapter <name> with --help for a model's full options and defaults.")
}

fn option_value(args: &[OsString], name: &str) -> Option<OsString> {
    let prefix = format!("{name}=");
    let mut values = args.iter();
    let mut result = None;
    while let Some(value) = values.next() {
        if value == "--" {
            break;
        }
        if value == name {
            result = values.next().cloned();
        } else if let Some(value) = value.to_str().and_then(|value| value.strip_prefix(&prefix)) {
            result = Some(value.into());
        }
    }
    result
}

pub(super) fn run() -> Result<()> {
    let args = std::env::args_os().collect::<Vec<_>>();
    let registry = Registry::new(BUILTINS);
    let task = args
        .get(1)
        .and_then(|name| Task::ALL.into_iter().find(|task| name == task.name()));
    let selected = task
        .map(|task| registry.select(task, &args[2..]))
        .transpose()?
        .flatten();
    let matches = registry.root_command(selected).get_matches_from(args);
    if let Some(task) = task {
        let task_matches = matches
            .subcommand_matches(task.name())
            .context("missing task command")?;
        let adapter =
            selected.context("specify --model to identify the model, or select --adapter")?;
        return (adapter.run)(task_matches);
    }
    super::dispatch(super::Args::from_arg_matches(&matches)?.command)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint(value: serde_json::Value) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();
        directory
    }

    fn args(directory: &std::path::Path) -> Vec<OsString> {
        vec!["generate".into(), "--model".into(), directory.into()]
    }

    #[test]
    fn adding_a_registered_model_requires_no_task_router_changes() {
        let adapter = Adapter {
            id: "new-text-model",
            task: Task::Text,
            recognizes: |metadata| metadata.architecture("NewTextArchitecture"),
            command: || {
                Command::new("text").subcommand(
                    Command::new("generate")
                        .arg(Arg::new("model").long("model").required(true))
                        .arg(Arg::new("new-option").long("new-option").required(true)),
                )
            },
            run: |matches| {
                let generate = matches.subcommand_matches("generate").unwrap();
                ensure!(generate.get_one::<String>("new-option").unwrap() == "supported");
                Ok(())
            },
        };
        let mut adapters = BUILTINS.to_vec();
        adapters.push(adapter);
        let registry = Registry::new(&adapters);
        let directory = checkpoint(serde_json::json!({"architectures":["NewTextArchitecture"]}));
        let mut arguments = args(directory.path());
        arguments.extend(["--new-option".into(), "supported".into()]);
        let selected = registry.select(Task::Text, &arguments).unwrap().unwrap();
        assert_eq!(selected.id, "new-text-model");
        let mut argv = vec!["ff".into(), "text".into()];
        argv.extend(arguments);
        let matches = registry
            .root_command(Some(selected))
            .try_get_matches_from(argv)
            .unwrap();
        (selected.run)(matches.subcommand_matches("text").unwrap()).unwrap();
    }

    #[test]
    fn overlapping_adapters_require_an_explicit_choice() {
        let directory = checkpoint(
            serde_json::json!({"architectures":["LlamaForCausalLM"],"model_type":"llama"}),
        );
        let mut adapters = BUILTINS.to_vec();
        let mut alternate = *BUILTINS
            .iter()
            .find(|adapter| adapter.id == "minicpm")
            .unwrap();
        alternate.id = "alternate-text";
        adapters.push(alternate);
        let registry = Registry::new(&adapters);
        let mut arguments = args(directory.path());
        let error = registry.select(Task::Text, &arguments).err().unwrap();
        assert!(error.to_string().contains("multiple adapters"));
        arguments.extend(["--adapter".into(), "alternate-text".into()]);
        assert_eq!(
            registry.select(Task::Text, &arguments).unwrap().unwrap().id,
            "alternate-text"
        );
    }

    #[test]
    fn mismatched_tasks_and_explicit_adapters_are_rejected_from_metadata() {
        let directory = checkpoint(serde_json::json!({"_class_name":"MiniMaxH3ModularPipeline"}));
        let registry = Registry::new(BUILTINS);
        let mut arguments = args(directory.path());
        let error = registry
            .select(Task::Text, &arguments)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("available tasks: video"), "{error}");
        arguments.extend(["--adapter".into(), "glm".into()]);
        let error = registry
            .select(Task::Text, &arguments)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("does not support the metadata"), "{error}");
    }

    #[test]
    fn edge0_and_qwen35_checkpoints_select_their_adapters() {
        let registry = Registry::new(BUILTINS);
        let cases = [
            (
                serde_json::json!({"architectures":["Qwen3_5MoeForConditionalGeneration"]}),
                "edge0",
            ),
            (serde_json::json!({"model_type":"qwen3_5_moe"}), "edge0"),
            (
                serde_json::json!({"architectures":["Qwen3_5ForConditionalGeneration"],"quantization":{"format":"groupwise-int4-u32"}}),
                "qwen35",
            ),
        ];
        for (metadata, expected) in cases {
            let directory = checkpoint(metadata);
            let selected = registry
                .select(Task::Text, &args(directory.path()))
                .unwrap()
                .unwrap();
            assert_eq!(selected.id, expected);
        }
        // The upstream BF16 Qwen3.8 checkpoint shares the architecture but
        // lacks the requantizer's quantization stamp: no adapter may claim it.
        let upstream =
            checkpoint(serde_json::json!({"architectures":["Qwen3_5ForConditionalGeneration"]}));
        let error = registry
            .select(Task::Text, &args(upstream.path()))
            .err()
            .expect("upstream BF16 checkpoint must not be claimed");
        assert!(
            error.to_string().contains("no registered adapter"),
            "{error}"
        );
    }

    #[test]
    fn both_trellis_generations_use_the_3d_task() {
        let registry = Registry::new(BUILTINS);
        for architecture in [
            "TrellisTextTo3DPipeline",
            "TrellisImageTo3DPipeline",
            "Trellis2ImageTo3DPipeline",
        ] {
            let directory = checkpoint(serde_json::json!({"name":architecture}));
            let selected = registry
                .select(Task::ThreeD, &args(directory.path()))
                .unwrap()
                .unwrap();
            assert_eq!(selected.id, "trellis");
        }
    }

    #[test]
    fn help_and_each_adapter_schema_are_valid_without_a_checkpoint() {
        let registry = Registry::new(BUILTINS);
        registry.root_command(None).debug_assert();
        for adapter in BUILTINS {
            registry.root_command(Some(adapter)).debug_assert();
            let arguments = vec!["--adapter".into(), adapter.id.into(), "--help".into()];
            assert_eq!(
                registry
                    .select(adapter.task, &arguments)
                    .unwrap()
                    .unwrap()
                    .id,
                adapter.id
            );
            let error = registry
                .root_command(Some(adapter))
                .try_get_matches_from([
                    "ff",
                    adapter.task.name(),
                    "--adapter",
                    adapter.id,
                    "--help",
                ])
                .unwrap_err();
            assert_eq!(error.kind(), clap::error::ErrorKind::DisplayHelp);
        }
    }

    #[test]
    fn model_options_are_not_silently_accepted_by_another_adapter() {
        let registry = Registry::new(BUILTINS);
        let selected = BUILTINS.iter().find(|adapter| adapter.id == "glm").unwrap();
        let error = registry
            .root_command(Some(selected))
            .try_get_matches_from([
                "ff",
                "text",
                "generate",
                "--model",
                "checkpoint",
                "--prompt",
                "hello",
                "--draft-model",
                "draft",
            ])
            .unwrap_err();
        assert_eq!(error.kind(), clap::error::ErrorKind::UnknownArgument);
    }
}
