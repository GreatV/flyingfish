use super::{Adapter, Task};
use crate::cli::music::{self, MusicCommand};
use clap::{FromArgMatches, Subcommand};

pub(super) const ADAPTER: Adapter = Adapter {
    id: "music3",
    task: Task::Music,
    recognizes: |metadata| metadata.architecture("MiniMaxMusic3ModularPipeline"),
    command: || MusicCommand::augment_subcommands(clap::Command::new("music")),
    run: |matches| music::run(MusicCommand::from_arg_matches(matches)?),
};
