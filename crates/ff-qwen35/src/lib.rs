//! Qwen3.8-27B (dense qwen3_5) adapter — see docs/qwen35-design.md.

pub mod config;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod model;
#[cfg(feature = "cuda")]
pub mod spec;
pub mod weights;
#[cfg(feature = "cuda")]
pub mod wide;
