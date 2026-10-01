use super::{Adapter, Task};
use crate::cli::qwen35::{self, Qwen35Command};
use clap::{FromArgMatches, Subcommand};
use flyingfish::qwen35::config::{QUANTIZATION_FORMAT, QWEN35_ARCHITECTURE};

pub(super) const ADAPTER: Adapter = Adapter {
    id: "qwen35",
    task: Task::Text,
    recognizes: |metadata| {
        // The upstream 16-bit checkpoint carries no quantization marker;
        // the requant stamp selects int4. The weight format itself is
        // detected from tensor dtypes when the checkpoint opens.
        metadata.architecture(QWEN35_ARCHITECTURE)
            && (metadata.quantization_format(QUANTIZATION_FORMAT) || !metadata.has_quantization())
    },
    command: || Qwen35Command::augment_subcommands(clap::Command::new(Task::Text.name())),
    run: |matches| qwen35::run(Qwen35Command::from_arg_matches(matches)?),
};
