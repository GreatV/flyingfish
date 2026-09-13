use super::{Adapter, Task};
use crate::cli::clip::{self, ClipCommand};
use clap::{FromArgMatches, Subcommand};

pub(super) const ADAPTER: Adapter = Adapter {
    id: "clip",
    task: Task::Similarity,
    recognizes: |metadata| metadata.architecture("CLIPModel"),
    command: || ClipCommand::augment_subcommands(clap::Command::new("similarity")),
    run: |matches| clip::run(ClipCommand::from_arg_matches(matches)?),
};
