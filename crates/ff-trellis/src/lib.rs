//! TRELLIS structured-3D-latent generation.
//!
//! A TRELLIS checkpoint is not one model. It is a directory of independently
//! named components — flow models, encoders, decoders — plus a `pipeline.json`
//! that says which of them a given pipeline uses and how to sample them. The
//! components are shared across published checkpoints: the text pipelines carry
//! only their own two flow models and reach into the image checkpoint for every
//! decoder.
//!
//! This crate resolves that layout and executes TRELLIS-1 Gaussian generation
//! and TRELLIS.2 direct-512 shape/material generation. Sparse geometry and
//! attention stay in the adapter alongside the model-specific sampling rules.

pub mod checkpoint;
pub mod clip;
pub mod clip_text;
pub mod config;
pub mod conv3d;
pub mod dinov2;
pub mod dinov3;
pub mod gaussian;
pub mod generation;
pub mod inventory;
pub mod mesh;
pub mod pipeline;
pub mod sampler;
pub mod slat_flow;
mod slat_ops;
pub mod sparse;
pub mod sparse_structure_decoder;
pub mod sparse_structure_flow;
pub mod trellis2;
pub mod trellis2_flow;
pub mod trellis2_vae;
