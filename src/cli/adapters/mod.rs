//! Model-specific command schemas and execution behind task-based routing.

mod clip;
mod edge0;
mod glm;
mod h3;
mod minicpm;
mod music;
mod qwen35;
mod trellis;

pub(super) use glm::GlmCommand;
pub(super) use h3::{H3Command, H3DecodeCommand};
pub(super) use trellis::TrellisCommand;

use anyhow::{Context, Result, bail};
use clap::{ArgMatches, Command};
use serde_json::Value;
use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Task {
    Text,
    Video,
    Music,
    ThreeD,
    Similarity,
}

impl Task {
    pub(super) const ALL: [Self; 5] = [
        Self::Text,
        Self::Video,
        Self::Music,
        Self::ThreeD,
        Self::Similarity,
    ];

    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Video => "video",
            Self::Music => "music",
            Self::ThreeD => "3d",
            Self::Similarity => "similarity",
        }
    }

    pub(super) fn about(self) -> &'static str {
        match self {
            Self::Text => "Generate text with a compatible model",
            Self::Video => "Generate video and audio with a compatible model",
            Self::Music => "Generate music with a compatible model",
            Self::ThreeD => "Generate and decode 3D assets with a compatible model",
            Self::Similarity => "Score image/text similarity with a compatible model",
        }
    }
}

/// Only architecture metadata is read while choosing an adapter.
pub(super) struct Metadata(Value);

impl Metadata {
    pub(super) fn read(root: &Path) -> Result<Self> {
        for name in [
            "modular_model_index.json",
            "model_index.json",
            "pipeline.json",
            "config.json",
            "transformer/config.json",
        ] {
            let path = root.join(name);
            if path.is_file() {
                let bytes = std::fs::read(&path)
                    .with_context(|| format!("read model metadata {}", path.display()))?;
                return serde_json::from_slice(&bytes)
                    .map(Self)
                    .with_context(|| format!("invalid model metadata {}", path.display()));
            }
        }
        bail!("no supported model metadata in {}", root.display())
    }

    pub(super) fn architecture(&self, name: &str) -> bool {
        self.0["_class_name"] == name
            || self.0["name"] == name
            || self.0["architectures"]
                .as_array()
                .is_some_and(|names| names.iter().any(|value| value == name))
    }

    pub(super) fn model_type(&self, name: &str) -> bool {
        self.0["model_type"] == name
    }
}

/// Registering another model supplies its recognition, CLI and execution in
/// one place. Task routing does not match on model names or model families.
#[derive(Clone, Copy)]
pub(super) struct Adapter {
    pub(super) id: &'static str,
    pub(super) task: Task,
    pub(super) recognizes: fn(&Metadata) -> bool,
    pub(super) command: fn() -> Command,
    pub(super) run: fn(&ArgMatches) -> Result<()>,
}

pub(super) const BUILTINS: &[Adapter] = &[
    glm::ADAPTER,
    minicpm::ADAPTER,
    edge0::ADAPTER,
    qwen35::ADAPTER,
    h3::ADAPTER,
    music::ADAPTER,
    trellis::ADAPTER,
    clip::ADAPTER,
];
