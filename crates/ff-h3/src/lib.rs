//! H3 model mathematics and explicit numerical/resource contracts.
pub(crate) use ff_core::required_option;

pub mod audio_vae;
pub mod audio_vae_encoder;
pub mod conditioning_provenance;
pub mod config;
pub mod core;
pub mod cuda;
pub mod embeddings;
pub mod execution;
pub mod fl2va;
pub mod h3_conditioning;
pub mod layout;
pub mod model;
pub mod multimodal_text_encoder;
pub mod pipeline;
pub mod policy;
#[cfg(feature = "cuda")]
pub(crate) mod prefetch;
pub mod ref2va;
pub mod resources;
pub mod scheduler;
pub mod solver;
pub mod target_geometry;
pub mod tensor_parallel;
pub mod text_encoder;
pub mod timing;
mod vae_tiling;
pub mod video_vae;
pub mod video_vae_encoder;
