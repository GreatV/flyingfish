use super::{Adapter, Task};
use crate::cli::dsv41::{self, Dsv41Command};
use clap::{FromArgMatches, Subcommand};
use flyingfish::dsv41::config::{DSV41_ARCHITECTURE, DSV41_MODEL_TYPE};

pub(super) const ADAPTER: Adapter = Adapter {
    id: "dsv41",
    task: Task::Text,
    recognizes: |metadata| {
        metadata.architecture(DSV41_ARCHITECTURE) && metadata.model_type(DSV41_MODEL_TYPE)
    },
    command: || Dsv41Command::augment_subcommands(clap::Command::new(Task::Text.name())),
    run: |matches| dsv41::run(Dsv41Command::from_arg_matches(matches)?),
};
