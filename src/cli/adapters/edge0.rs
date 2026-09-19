use super::{Adapter, Task};
use crate::cli::edge0::{self, Edge0Command};
use clap::{FromArgMatches, Subcommand};
use flyingfish::edge0::config::{EDGE0_ARCHITECTURE, EDGE0_MODEL_TYPE};

pub(super) const ADAPTER: Adapter = Adapter {
    id: "edge0",
    task: Task::Text,
    recognizes: |metadata| {
        metadata.architecture(EDGE0_ARCHITECTURE) || metadata.model_type(EDGE0_MODEL_TYPE)
    },
    command: || Edge0Command::augment_subcommands(clap::Command::new(Task::Text.name())),
    run: |matches| edge0::run(Edge0Command::from_arg_matches(matches)?),
};
