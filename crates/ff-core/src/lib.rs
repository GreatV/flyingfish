//! Model-independent storage, identity, resource bounds and observation.
//! This crate intentionally cannot depend on model adapters or the application.
//!
//! The core can compile and expose byte budgets without either model:
//! ```
//! let budget = ff_core::bounds::ResourceBudget::default();
//! assert!(budget.check_peaks(1, 1).within_budget);
//! ```
//! Adapter imports are deliberately unavailable from this dependency layer:
//! ```compile_fail
//! use ff_h3::model::StreamedTransformer;
//! ```
//! ```compile_fail
//! use ff_glm::StreamedGlm;
//! ```
//! ```compile_fail
//! use flyingfish::recovery::CheckpointIdentity;
//! ```

#[doc(hidden)]
pub fn required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    <Option<T> as serde::Deserialize>::deserialize(deserializer)
}

pub mod artifact;
pub mod bounds;
mod calibration_identity;
pub mod cold_cache;
pub mod configure;
#[cfg(feature = "cuda")]
pub mod cuda_kernel_assets;
pub mod frame_manifest;
pub mod identity;
pub mod interconnect;
pub mod io_calibration;
pub mod math;
pub mod parity;
pub mod probe;
pub mod residency;
pub mod resource_selection;
pub mod storage;
pub mod telemetry;
pub mod topology;
pub mod weights;
