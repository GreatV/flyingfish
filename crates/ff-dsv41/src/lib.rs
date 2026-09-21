//! DeepSeek-V4.1-Flash (deepseek_v41) text and image inference.
//!
//! First-cut execution profile: batch size one, single-turn text or one image,
//! with the static skeleton loaded from the checkpoint's FP8/FP4 shards and
//! the routed experts streamed per use.

pub mod attention;
pub mod block;
pub mod config;
pub mod encoding;
pub mod engram;
pub mod math;
pub mod model;
pub mod moe;
pub mod numpy_random;
pub mod quant;
pub mod transformer;
pub mod vision;
pub mod weights;
