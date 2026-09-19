//! Qwen3.8-27B (dense qwen3_5) adapter.

pub mod config;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod model;
#[cfg(feature = "cuda")]
pub mod spec;
pub mod vision;
pub mod weights;
#[cfg(feature = "cuda")]
pub mod wide;
