//! Pinned Qwen3-VL operators for the exact CUDA profile.
//!
//! Each module reproduces one upstream operator's arithmetic and rounding
//! boundaries exactly, and admits only the profiles its recorded oracle
//! covers. Anything else fails before it can produce unverified numbers.

pub(crate) mod attention;
pub(crate) mod gelu;
pub(crate) mod layer_norm;
pub(crate) mod patch;
pub(crate) mod position;
