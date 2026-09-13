//! Streamed MiniMax Music3, following the published diffusers component layout.
mod acoustic;
mod autoregressive;
mod math;
pub mod pipeline;
pub mod prompt;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use ff_core::residency::WeightPhase;
use ff_core::weights::{CachePolicy, DeviceCache, ModelWeights, WeightSource};
use serde_json::Value;
use std::path::Path;

struct Component {
    weights: ModelWeights,
    config: Value,
    device: Device,
    dtype: DType,
}

impl Component {
    fn open(
        root: &Path,
        class: &str,
        device: &Device,
        source: WeightSource,
        cache: CachePolicy,
        residency: DeviceCache,
    ) -> Result<Self> {
        let config: Value = serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
        anyhow::ensure!(
            config["_class_name"] == class || config["architectures"][0] == class,
            "{} is not a {class} checkpoint",
            root.display()
        );
        let mut weights = ModelWeights::open(root, source, cache)?;
        weights.configure_device_cache(residency)?;
        let dtype = if device.is_cpu()
            || class != "Qwen3ForCausalLM" && class != "MiniMaxMusic3RVQDepthDecoder"
        {
            DType::F32
        } else {
            DType::BF16
        };
        Ok(Self {
            weights,
            config,
            device: device.clone(),
            dtype,
        })
    }
    fn n(&self, key: &str) -> Result<usize> {
        let value = self.config[key]
            .as_u64()
            .with_context(|| format!("missing positive config field {key}"))?;
        let value = usize::try_from(value).context("config dimension exceeds usize")?;
        anyhow::ensure!(value > 0, "{key} must be positive");
        Ok(value)
    }
    fn expect(&self, key: &str, expected: Value) -> Result<()> {
        anyhow::ensure!(
            self.config[key] == expected,
            "unsupported Music3 config {key}: {}",
            self.config[key]
        );
        Ok(())
    }

    /// This component's residency phase: every tensor it holds, read
    /// `reuse_count` times over one request.
    fn weight_phase(&self, name: &str, reuse_count: u64) -> WeightPhase {
        WeightPhase::new(
            name,
            self.weights.tensor_names().map(str::to_owned),
            reuse_count,
        )
    }

    /// Full tensors eligible for retention. Row gathers never populate the
    /// device tensor cache. Recurrent blocks are independent placement units
    /// so a partial model can stay resident without cycling the entire cache.
    fn device_weight_phases(&self, name: &str, reuse_count: u64) -> Vec<WeightPhase> {
        let mut groups = std::collections::BTreeMap::<String, Vec<String>>::new();
        for tensor in self.weights.tensor_names() {
            if matches!(
                tensor,
                "model.embed_tokens.weight" | "audio_embeddings.weight" | "pos_embedding.weight"
            ) {
                continue;
            }
            let layer = ["model.layers.", "layers.", "transformer_blocks."]
                .into_iter()
                .find_map(|prefix| tensor.strip_prefix(prefix))
                .and_then(|suffix| suffix.split_once('.'))
                .and_then(|(index, _)| index.parse::<usize>().ok());
            let group = match layer {
                Some(index) => format!("{name}.layer-{index:04}"),
                None => format!("{name}.shared"),
            };
            groups.entry(group).or_default().push(tensor.to_owned());
        }
        groups
            .into_iter()
            .map(|(name, tensors)| WeightPhase::new(name, tensors, reuse_count))
            .collect()
    }
}
