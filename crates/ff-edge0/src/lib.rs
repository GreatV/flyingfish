//! Edge0-35B-A3B (Qwen3.5-MoE multimodal) adapter.
//!
//! P0 scope: text-only greedy generation over the hybrid GDN/full-attention
//! MoE stack with groupwise-int4 streaming.

pub mod config;
pub mod int4;
pub mod mode;
pub mod model;
pub mod weights;

#[cfg(feature = "cuda")]
pub mod gpu;
