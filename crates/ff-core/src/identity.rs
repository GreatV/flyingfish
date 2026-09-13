//! Model and build identities.
//!
//! A component is identified by what the filesystem already knows about it —
//! each shard's size and modification time — rather than by rehashing hundreds
//! of gigabytes of weights on every run.

pub use crate::calibration_identity::{
    BinaryIdentity, CALIBRATION_IDENTITY_SCHEMA_VERSION, FileStamp, InputIdentity,
    ModelIdentityStrength, NamedFileStamp, WeakModelIdentity, WeakShardIdentity, stamp_file,
};
