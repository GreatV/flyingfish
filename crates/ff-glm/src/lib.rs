//! Disk-streamed GLM-5.3-Flash text inference.
//!
//! The first supported execution profile is deliberately narrow: batch size
//! one, text-only prompts, greedy or nucleus sampling, and contexts no longer
//! than the checkpoint's DSA `index_topk`. Within that profile every visible token is
//! selected by DSA, so the expensive long-context indexer is exactly
//! equivalent to causal attention over the complete retained cache.
//!
//! Router observation is optional and post-selection. Versioned routing traces
//! feed a bounded offline SRP/SCH and cache-policy replay. Expert-cache
//! topology, replacement, and token-boundary re-admission remain
//! performance-only and are bound by a separate versioned execution policy and
//! manifest.

pub(crate) use ff_core::required_option;

pub mod admission;
pub mod config;
pub mod execution_manifest;
pub mod execution_policy;
pub mod expert_cache;
pub mod expert_cache_manager;
pub mod fp8;
#[cfg(feature = "cuda")]
pub mod kernel_assets;
pub mod math;
mod model;
pub mod partition;
pub mod resources;
pub mod routing_trace;

pub use execution_manifest::GlmExecutionManifest;
pub use execution_policy::GlmExecutionPolicy;
pub use expert_cache::ExpertCacheReplacementPolicy;
pub use expert_cache_manager::ExpertCacheLayout;
pub use model::{
    GlmGeneration, GlmGenerationOptions, GlmParityCapture, PreparedGlm, StreamedGlm,
    StreamedGlmOptions,
};
#[cfg(feature = "cuda")]
pub use model::{
    GlmPartitionAdmission, GlmPartitionGeneration, GlmRankAdmission, GlmRankCacheStats,
    LayerPartitionOptions, LayerPartitionedGlm,
};
pub use routing_trace::{
    RoutingReplayOptions, RoutingReplayReport, RoutingTrace, RoutingTracePhase,
};

#[cfg(test)]
mod test_support;
