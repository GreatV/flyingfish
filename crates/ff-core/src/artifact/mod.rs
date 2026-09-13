use anyhow::{Context, Result, anyhow, bail};

use serde::{Deserialize, Serialize};

use std::{
    ffi::OsString,
    fmt,
    fs::{self, File, Metadata, OpenOptions},
    io::{ErrorKind, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

mod compare;
mod digest;
mod identity;
mod staging;

pub use compare::compare_artifact_files;
pub use digest::*;
pub use identity::*;
pub use staging::*;

static STAGING_NONCE: AtomicU64 = AtomicU64::new(0);
