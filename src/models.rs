//! Local model discovery and component loading, independent of inference support.
use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::Value;
use std::path::{Component, Path, PathBuf};

use crate::runtime::weights::{CachePolicy, ModelWeights, WeightSource};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFamily {
    Dsv41,
    H3,
    Music3,
    Glm,
    #[serde(rename = "minicpm")]
    MiniCpm,
    #[serde(rename = "dspark")]
    DSpark,
    Trellis,
    Trellis2,
    Clip,
    DinoV2,
    DinoV3,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelComponent {
    pub role: String,
    pub architecture: String,
    pub directory: PathBuf,
    /// TRELLIS stores separately named single-file components in one directory.
    pub file_name: Option<String>,
}

impl ModelComponent {
    pub fn open_weights(&self, source: WeightSource, cache: CachePolicy) -> Result<ModelWeights> {
        match &self.file_name {
            Some(name) => ModelWeights::open_component(&self.directory, name, source, cache),
            None => ModelWeights::open(&self.directory, source, cache),
        }
        .with_context(|| format!("failed to open component {}", self.role))
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct LocalModel {
    pub root: PathBuf,
    pub family: ModelFamily,
    pub architecture: String,
    pub components: Vec<ModelComponent>,
    /// Models required by the pipeline but shipped separately.
    pub dependencies: Vec<String>,
    /// Executable scope, deliberately separate from checkpoint loading.
    pub inference_scope: String,
}

fn json(path: &Path) -> Result<Value> {
    serde_json::from_slice(
        &std::fs::read(path).with_context(|| format!("read {}", path.display()))?,
    )
    .with_context(|| format!("invalid JSON in {}", path.display()))
}

fn local_subfolder(path: &str) -> Result<&Path> {
    let path = Path::new(path);
    anyhow::ensure!(
        !path.as_os_str().is_empty()
            && path
                .components()
                .all(|part| matches!(part, Component::Normal(_))),
        "model component subfolder must be a relative path without traversal"
    );
    Ok(path)
}

impl LocalModel {
    /// Recognize published architecture metadata, rather than directory names.
    pub fn open(root: impl AsRef<Path>, models_root: Option<&Path>) -> Result<Self> {
        let root = root.as_ref();
        anyhow::ensure!(
            root.is_dir(),
            "model directory is missing: {}",
            root.display()
        );
        if root.join("pipeline.json").is_file() {
            return Self::trellis(root, models_root);
        }
        for index in ["modular_model_index.json", "model_index.json"] {
            if root.join(index).is_file() {
                return Self::diffusion(root, &json(&root.join(index))?);
            }
        }
        let config = json(&root.join("config.json"))?;
        let architecture = config["architectures"][0]
            .as_str()
            .context("missing model architecture")?;
        let (family, dependencies, scope) = match architecture {
            "LlamaForCausalLM" if config["model_type"] == "llama" => (
                ModelFamily::MiniCpm,
                vec![],
                "ff text generate: batch-one greedy text generation",
            ),
            "Qwen3DSparkModel" => (
                ModelFamily::DSpark,
                vec!["openbmb/MiniCPM5-2B".into()],
                "ff text generate --draft-model PATH: paired greedy speculative decoding",
            ),
            "Dinov2WithRegistersModel" => (
                ModelFamily::DinoV2,
                vec![],
                "DINOv2 register-token conditioning for TRELLIS-1 image generation",
            ),
            "DINOv3ViTModel" => (
                ModelFamily::DinoV3,
                vec![],
                "DINOv3 ViT-L/16 conditioning for TRELLIS.2 image generation",
            ),
            "CLIPModel" => (
                ModelFamily::Clip,
                vec![],
                "ff similarity score: normalized image/text similarity; also TRELLIS text conditioning",
            ),
            "Glm5NextForConditionalGeneration" => (
                ModelFamily::Glm,
                vec![],
                "ff text generate: batch-one text, up to 2048 total tokens",
            ),
            "DeepseekV41ForCausalLM" if config["model_type"] == "deepseek_v41" => (
                ModelFamily::Dsv41,
                vec![],
                "ff text generate: CED MoE text/image with sliding-window attention and the engram memory",
            ),
            _ => bail!(
                "unsupported model architecture {architecture:?} in {}",
                root.display()
            ),
        };
        Ok(Self {
            root: root.into(),
            family,
            architecture: architecture.into(),
            dependencies,
            inference_scope: scope.into(),
            components: vec![ModelComponent {
                role: "model".into(),
                architecture: architecture.into(),
                directory: root.into(),
                file_name: None,
            }],
        })
    }

    fn diffusion(root: &Path, index: &Value) -> Result<Self> {
        let architecture = index["_class_name"]
            .as_str()
            .context("missing diffusion pipeline class")?;
        let (family, roles, scope): (_, &[&str], _) = match architecture {
            "MiniMaxH3ModularPipeline" => (
                ModelFamily::H3,
                &[
                    "audio_vae",
                    "text_encoder",
                    "transformer",
                    "transformer_ref",
                    "vae",
                ],
                "ff video: T2VA, FL2VA and Ref2VA generation and decoding",
            ),
            "MiniMaxMusic3ModularPipeline" => (
                ModelFamily::Music3,
                &[
                    "condition_encoder",
                    "language_model",
                    "rvq_depth_decoder",
                    "transformer",
                    "vocoder",
                ],
                "ff music generate: lyrics and caption to native 44.1 kHz stereo WAV",
            ),
            _ => bail!("unsupported diffusion pipeline {architecture:?}"),
        };
        let mut components = Vec::new();
        for &role in roles {
            let entry = &index[role];
            let class = entry[1]
                .as_str()
                .with_context(|| format!("missing {role} component class"))?;
            let subfolder = entry.get(2).and_then(|metadata| metadata.get("subfolder"));
            let subfolder = match subfolder {
                None | Some(Value::Null) => role,
                Some(value) => value
                    .as_str()
                    .context("component subfolder must be a string")?,
            };
            components.push(ModelComponent {
                role: role.into(),
                architecture: class.into(),
                directory: root.join(local_subfolder(subfolder)?),
                file_name: None,
            });
        }
        Ok(Self {
            root: root.into(),
            family,
            architecture: architecture.into(),
            components,
            dependencies: vec![],
            inference_scope: scope.into(),
        })
    }

    fn trellis(root: &Path, models_root: Option<&Path>) -> Result<Self> {
        let checkpoint = crate::trellis::checkpoint::TrellisCheckpoint::open(root, models_root)?;
        let config = json(&root.join("pipeline.json"))?;
        let architecture = config["name"]
            .as_str()
            .context("missing TRELLIS pipeline name")?;
        let (family, scope) = match architecture {
            "TrellisTextTo3DPipeline" => (
                ModelFamily::Trellis,
                "ff 3d generate --prompt TEXT: text to 3D Gaussian PLY",
            ),
            "TrellisImageTo3DPipeline" => (
                ModelFamily::Trellis,
                "ff 3d generate --image PNG: prepared image to 3D Gaussian PLY",
            ),
            "Trellis2ImageTo3DPipeline" => (
                ModelFamily::Trellis2,
                "ff 3d generate --image PNG --resolution 512: raw colored mesh PLY",
            ),
            _ => bail!("unsupported TRELLIS pipeline {architecture:?}"),
        };
        // `rembg_model` is deliberately absent: the image commands take an
        // already-prepared square RGB PNG, so background removal happens before
        // this tool is invoked and its checkpoint is never loaded here. Naming
        // it as a dependency would ask the operator to fetch weights nothing
        // reads.
        let mut dependencies = Vec::new();
        for role in ["text_cond_model", "image_cond_model"] {
            let entry = &config["args"][role];
            if let Some(name) = entry
                .as_str()
                .or_else(|| entry["args"]["model_name"].as_str())
            {
                dependencies.push(name.to_owned());
            }
        }
        let components = checkpoint
            .components
            .iter()
            .map(|component| ModelComponent {
                role: component.role.clone(),
                architecture: component.config.name().into(),
                directory: component.directory.clone(),
                file_name: Some(component.weights_file_name()),
            })
            .collect();
        Ok(Self {
            root: root.into(),
            family,
            architecture: architecture.into(),
            components,
            dependencies,
            inference_scope: scope.into(),
        })
    }

    pub fn component(&self, role: &str) -> Result<&ModelComponent> {
        self.components
            .iter()
            .find(|component| component.role == role)
            .with_context(|| {
                format!(
                    "model has no component {role:?}; available: {}",
                    self.components
                        .iter()
                        .map(|c| c.role.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
    }
}

#[derive(Debug, Serialize)]
pub struct CatalogEntry {
    pub path: PathBuf,
    pub model: Option<LocalModel>,
    pub error: Option<String>,
}

/// Enumerate `<root>/<owner>/<repository>`, retaining per-model errors so a bad
/// checkpoint cannot hide the remaining models. Payloads are not read.
pub fn discover(root: &Path) -> Result<Vec<CatalogEntry>> {
    let mut repositories = Vec::new();
    for owner in std::fs::read_dir(root).with_context(|| format!("read {}", root.display()))? {
        let owner = owner?.path();
        if !owner.is_dir()
            || owner
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with('.'))
        {
            continue;
        }
        for repository in std::fs::read_dir(&owner)? {
            let repository = repository?.path();
            if repository.is_dir()
                && !repository
                    .file_name()
                    .is_some_and(|name| name.to_string_lossy().starts_with('.'))
            {
                repositories.push(repository);
            }
        }
    }
    repositories.sort();
    Ok(repositories
        .into_iter()
        .map(|path| match LocalModel::open(&path, Some(root)) {
            Ok(model) => CatalogEntry {
                path,
                model: Some(model),
                error: None,
            },
            Err(error) => CatalogEntry {
                path,
                model: None,
                error: Some(format!("{error:#}")),
            },
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deepseek_v41_config_selects_the_dsv41_family() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("deepseek-ai/DeepSeek-V4.1-Flash");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(
            model.join("config.json"),
            r#"{"architectures":["DeepseekV41ForCausalLM"],"model_type":"deepseek_v41"}"#,
        )
        .unwrap();
        let opened = LocalModel::open(&model, None).unwrap();
        assert_eq!(opened.family, ModelFamily::Dsv41);
        // The architecture alone, without the model_type belt, is refused.
        let imposter = dir.path().join("imposter");
        std::fs::create_dir_all(&imposter).unwrap();
        std::fs::write(
            imposter.join("config.json"),
            r#"{"architectures":["DeepseekV41ForCausalLM"]}"#,
        )
        .unwrap();
        assert!(LocalModel::open(&imposter, None).is_err());
    }

    #[test]
    fn discovery_uses_metadata_and_retains_broken_models() {
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("arbitrary/renamed-model");
        let bad = dir.path().join("arbitrary/broken-model");
        std::fs::create_dir_all(&good).unwrap();
        std::fs::create_dir_all(&bad).unwrap();
        std::fs::write(
            good.join("config.json"),
            r#"{"architectures":["CLIPModel"]}"#,
        )
        .unwrap();
        let entries = discover(dir.path()).unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries[0].error.is_some());
        assert_eq!(entries[1].model.as_ref().unwrap().family, ModelFamily::Clip);
    }

    #[test]
    fn diffusion_rejects_component_path_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let index = serde_json::json!({
            "_class_name": "MiniMaxMusic3ModularPipeline",
            "condition_encoder": ["diffusers", "Encoder", {"subfolder": "../outside"}]
        });
        assert!(
            LocalModel::diffusion(dir.path(), &index)
                .unwrap_err()
                .to_string()
                .contains("traversal")
        );
    }

    #[test]
    fn named_component_loads_nonstandard_safetensors_filename() {
        let dir = tempfile::tempdir().unwrap();
        let tensor = candle_core::Tensor::new(&[3f32, 7.], &candle_core::Device::Cpu).unwrap();
        candle_core::safetensors::save(
            &std::collections::HashMap::from([("probe", tensor)]),
            dir.path().join("named.safetensors"),
        )
        .unwrap();
        let component = ModelComponent {
            role: "decoder".into(),
            architecture: "Test".into(),
            directory: dir.path().into(),
            file_name: Some("named.safetensors".into()),
        };
        let weights = component
            .open_weights(WeightSource::Mmap, CachePolicy::new(1))
            .unwrap();
        assert_eq!(weights.verify().unwrap().checked_tensors, 1);
        assert_eq!(
            weights
                .load("probe", &candle_core::Device::Cpu)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            [3., 7.]
        );
    }
}
