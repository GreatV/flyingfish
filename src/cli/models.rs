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
