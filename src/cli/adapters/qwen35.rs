use super::{Adapter, Task};
use crate::cli::qwen35::{self, Qwen35Command};
use clap::{FromArgMatches, Subcommand};
use flyingfish::qwen35::config::QWEN35_ARCHITECTURE;

pub(super) const ADAPTER: Adapter = Adapter {
    id: "qwen35",
    task: Task::Text,
    recognizes: |metadata| metadata.architecture(QWEN35_ARCHITECTURE),
    command: || Qwen35Command::augment_subcommands(clap::Command::new(Task::Text.name())),
    run: |matches| qwen35::run(Qwen35Command::from_arg_matches(matches)?),
};
