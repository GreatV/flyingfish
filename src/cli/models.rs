use super::WeightCacheArgs;
use super::device_parse::parse_device_single;
use anyhow::Result;
use clap::Subcommand;
use flyingfish::models::{LocalModel, discover};
use flyingfish::runtime::weights::{CachePolicy, WeightSource};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Debug, Subcommand)]
pub(super) enum ModelsCommand {
    #[command(about = "List every local model, its components and executable scope")]
    List {
        #[arg(long)]
        models_root: PathBuf,
        #[arg(long)]
        json: bool,
    },
    #[command(about = "Inspect all components of a model, including shared TRELLIS weights")]
    Inspect {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        models_root: Option<PathBuf>,
        #[arg(long)]
        verify: bool,
        #[arg(long)]
        json: bool,
    },
    #[command(about = "Load one tensor from a named model component")]
    Tensor {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        models_root: Option<PathBuf>,
        #[arg(long, default_value = "model")]
        component: String,
        #[arg(long)]
        name: String,
        #[arg(long, default_value = "cpu")]
        device: String,
        #[command(flatten)]
        weights: WeightCacheArgs,
    },
}

#[derive(Serialize)]
struct ComponentInventory {
    role: String,
    tensors: usize,
    shards: usize,
    indexed_bytes: Option<u64>,
    verified: bool,
}

#[derive(Serialize)]
struct Inspection {
    model: LocalModel,
    inventory: Vec<ComponentInventory>,
}

pub(super) fn run(command: ModelsCommand) -> Result<()> {
    match command {
        ModelsCommand::List { models_root, json } => {
            let entries = discover(&models_root)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&entries)?);
            } else {
                for entry in entries {
                    match entry.model {
                        Some(model) => {
                            println!(
                                "{}: {:?} ({} components)",
                                entry.path.display(),
                                model.family,
                                model.components.len()
                            );
                            println!("  {}", model.inference_scope);
                            if !model.dependencies.is_empty() {
                                println!("  dependencies: {}", model.dependencies.join(", "));
                            }
                        }
                        None => println!(
                            "{}: {}",
                            entry.path.display(),
                            entry.error.unwrap_or_default()
                        ),
                    }
                }
            }
        }
        ModelsCommand::Inspect {
            model,
            models_root,
            verify,
            json,
        } => {
            let model = LocalModel::open(model, models_root.as_deref())?;
            let mut inventory = Vec::new();
            for component in &model.components {
                let weights = component.open_weights(WeightSource::Mmap, CachePolicy::new(1))?;
                let item = weights.inventory();
                if verify {
                    weights.verify()?;
                }
                inventory.push(ComponentInventory {
                    role: component.role.clone(),
                    tensors: item.tensors,
                    shards: item.shards,
                    indexed_bytes: item.indexed_bytes,
                    verified: verify,
                });
            }
            let report = Inspection { model, inventory };
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!(
                    "{}: {}",
                    report.model.root.display(),
                    report.model.architecture
                );
                println!("inference: {}", report.model.inference_scope);
                for item in report.inventory {
                    println!(
                        "{}: {} tensors, {} shards{}",
                        item.role,
                        item.tensors,
                        item.shards,
                        if item.verified { ", verified" } else { "" }
                    );
                }
            }
        }
        ModelsCommand::Tensor {
            model,
            models_root,
            component,
            name,
            device,
            weights,
        } => {
            let model = LocalModel::open(model, models_root.as_deref())?;
            let component = model.component(&component)?;
            let weights = component.open_weights(weights.weight_source, weights.cache_policy()?)?;
            let tensor = weights.load(&name, &parse_device_single(&device)?)?;
            println!(
                "{name}: dtype={:?}, shape={:?}, device={:?}",
                tensor.dtype(),
                tensor.shape(),
                tensor.device()
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use flyingfish::models::ModelFamily;
    use flyingfish::qwen35::config::{QUANTIZATION_FORMAT, QWEN35_ARCHITECTURE};
    use flyingfish::runtime::weights::CacheGranularity;
    use serde_json::json;

    #[test]
    fn tensor_command_uses_qwen_model_components() -> Result<()> {
        let root = tempfile::tempdir()?;
        Tensor::new(&[1f32, 2.], &Device::Cpu)?
            .save_safetensors("weight", root.path().join("part.safetensors"))?;
        std::fs::write(
            root.path().join("model.safetensors.index.json"),
            r#"{"metadata":{"total_size":8.0},"weight_map":{"weight":"part.safetensors"}}"#,
        )?;
        let args = WeightCacheArgs {
            weight_source: WeightSource::Mmap,
            host_cache_mib: None,
            host_cache_granularity: CacheGranularity::Shard,
        };
        for quantization in [
            serde_json::Value::Null,
            json!({"format": QUANTIZATION_FORMAT}),
        ] {
            let mut config = json!({"architectures": [QWEN35_ARCHITECTURE]});
            if !quantization.is_null() {
                config["quantization"] = quantization;
            }
            std::fs::write(
                root.path().join("config.json"),
                serde_json::to_vec(&config)?,
            )?;
            let model = LocalModel::open(root.path(), None)?;
            assert_eq!(model.family, ModelFamily::Qwen35);
            assert!(model.dependencies.is_empty());
            assert_eq!(model.components.len(), 1);
            assert_eq!(model.component("model")?.directory, root.path());
            assert!(model.component("missing").is_err());
            run(ModelsCommand::Tensor {
                model: root.path().to_owned(),
                models_root: None,
                component: "model".into(),
                name: "weight".into(),
                device: "cpu".into(),
                weights: args,
            })?;
        }
        Ok(())
    }

    #[test]
    fn local_qwen_tensor_metadata_without_payload_access() -> Result<()> {
        for checkpoint in ["Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-27B-int4-rtn"] {
            let Some(model) = ff_core::paths::checkpoint_dir(checkpoint) else {
                continue;
            };
            if !model.is_dir() {
                continue;
            }
            let model = model.canonicalize()?;
            println!("model: {}", model.display());
            run(ModelsCommand::Inspect {
                model: model.clone(),
                models_root: None,
                verify: false,
                json: true,
            })?;
            let model = LocalModel::open(&model, None)?;
            assert_eq!(model.family, ModelFamily::Qwen35);
            let weights = model
                .component("model")?
                .open_weights(WeightSource::Mmap, CachePolicy::new(1))?;
            let name = weights.tensor_names().next().unwrap();
            let metadata = weights.raw_tensor_metadata(name)?;
            let stats = weights.cache_stats();
            assert_eq!(stats.misses, 0);
            assert_eq!(stats.memory_source_reads, 0);
            assert_eq!(stats.resident_bytes, 0);
            assert_eq!(weights.access_stats().device_tensor_materializations, 0);
            println!(
                "{checkpoint}: indexed_bytes={:?}, {name} dtype={:?} shape={:?} bytes={}, payload_misses={}",
                weights.inventory().indexed_bytes,
                metadata.dtype,
                metadata.shape,
                metadata.bytes,
                stats.misses,
            );
        }
        Ok(())
    }
}
