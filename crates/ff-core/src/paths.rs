//! Local model-checkpoint locations.
//!
//! The `FF_MODELS_DIR` environment variable names the machine's models root
//! and is the only source for it: when it is unset there is no path to
//! resolve, and callers skip or fail rather than guessing one.

use std::path::PathBuf;

/// The directory `relative` names under the `FF_MODELS_DIR` root, or `None`
/// when the variable is unset.
pub fn checkpoint_dir(relative: &str) -> Option<PathBuf> {
    std::env::var_os("FF_MODELS_DIR").map(|root| PathBuf::from(root).join(relative))
}
