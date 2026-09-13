use super::{Adapter, Task};
use crate::cli::minicpm::{self, MiniCpmCommand};
use clap::{FromArgMatches, Subcommand};

pub(super) const ADAPTER: Adapter = Adapter {
    id: "minicpm",
    task: Task::Text,
    recognizes: |metadata| {
        metadata.architecture("LlamaForCausalLM") && metadata.model_type("llama")
    },
    command: || MiniCpmCommand::augment_subcommands(clap::Command::new("text")),
    run: |matches| minicpm::run(MiniCpmCommand::from_arg_matches(matches)?),
};
