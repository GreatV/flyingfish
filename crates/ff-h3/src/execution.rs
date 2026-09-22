use crate::config::TransformerConfig;
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use ff_core::{residency, weights::ModelWeights};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use std::{collections::BTreeMap, fmt};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "index",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum StageKind {
    ContextInput,
    TimeInput,
    LatentInput,
    RefinerAttention(usize),
    RefinerFeedForward(usize),
    RefinerOutputNorm,
    BlockAdaLn(usize),
    BlockAttention(usize),
    BlockFeedForward(usize),
    Output,
}

impl fmt::Display for StageKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextInput => write!(f, "context-input"),
            Self::TimeInput => write!(f, "time-input"),
            Self::LatentInput => write!(f, "latent-input"),
            Self::RefinerAttention(i) => write!(f, "refiner-{i}-attention"),
            Self::RefinerFeedForward(i) => write!(f, "refiner-{i}-feed-forward"),
            Self::RefinerOutputNorm => write!(f, "refiner-output-norm"),
            Self::BlockAdaLn(i) => write!(f, "block-{i}-adaln"),
            Self::BlockAttention(i) => write!(f, "block-{i}-attention"),
            Self::BlockFeedForward(i) => write!(f, "block-{i}-feed-forward"),
            Self::Output => write!(f, "output"),
        }
    }
}

impl StageKind {
    /// The timing bucket a stage's load and compute durations accumulate into.
    pub(crate) fn timing_bucket(&self) -> &'static str {
        match self {
            Self::BlockAdaLn(_) => "adaln",
            Self::BlockAttention(_) | Self::RefinerAttention(_) => "attention",
            Self::BlockFeedForward(_) | Self::RefinerFeedForward(_) => "feed_forward",
            Self::ContextInput
            | Self::TimeInput
            | Self::LatentInput
            | Self::RefinerOutputNorm
            | Self::Output => "other",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionStage {
    pub kind: StageKind,
    pub tensor_names: Vec<String>,
    pub weight_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3ExecutionPlan {
    stages: Vec<ExecutionStage>,
}

impl H3ExecutionPlan {
    pub fn from_config(weights: &ModelWeights, config: &TransformerConfig) -> Result<Self> {
        Self::build(weights, config.num_layers, config.num_refiner_layers)
    }

    pub fn build(weights: &ModelWeights, layers: usize, refiner_layers: usize) -> Result<Self> {
        let expected = expected_stage_kinds(layers, refiner_layers);
        let mut groups = expected
            .iter()
            .cloned()
            .map(|kind| (kind, Vec::new()))
            .collect::<BTreeMap<_, _>>();

        for name in weights.tensor_names() {
            let kind = classify(name).with_context(|| {
                format!(
                    "unrecognized MiniMax-H3 diffusers tensor {name:?}; this planner targets the public diffusers checkpoint"
                )
            })?;
            anyhow::ensure!(
                groups.contains_key(&kind),
                "tensor {name:?} belongs to unexpected stage {kind}"
            );
            groups.get_mut(&kind).unwrap().push(name.to_owned());
        }

        let mut stages = Vec::with_capacity(expected.len());
        for kind in expected {
            let names = groups.remove(&kind).unwrap();
            anyhow::ensure!(
                !names.is_empty(),
                "checkpoint has no tensors for required stage {kind}"
            );
            let mut weight_bytes = 0u64;
            for name in &names {
                weight_bytes = weight_bytes
                    .checked_add(weights.metadata(name)?.bytes as u64)
                    .context("stage weight size overflow")?;
            }
            stages.push(ExecutionStage {
                kind,
                tensor_names: names,
                weight_bytes,
            });
        }
        Ok(Self { stages })
    }

    pub fn stages(&self) -> &[ExecutionStage] {
        &self.stages
    }

    pub fn peak_stage_weight_bytes(&self) -> u64 {
        self.stages
            .iter()
            .map(|stage| stage.weight_bytes)
            .max()
            .unwrap_or(0)
    }

    /// Bytes one evaluation reads. Every stage runs once per evaluation and no
    /// stage's tensors are read by another, so this is also the distance
    /// between two reads of the same weight -- the quantity that decides
    /// whether the host could still be holding it.
    pub fn evaluation_weight_bytes(&self) -> u64 {
        self.stages
            .iter()
            .map(|stage| stage.weight_bytes)
            .fold(0u64, u64::saturating_add)
    }

    pub fn stage_index(&self, kind: &StageKind) -> Option<usize> {
        self.stages.iter().position(|stage| &stage.kind == kind)
    }

    pub fn with_stage<T>(
        &self,
        weights: &ModelWeights,
        stage_index: usize,
        device: &Device,
        f: impl FnOnce(&ExecutionStage, &BTreeMap<String, Tensor>) -> Result<T>,
    ) -> Result<T> {
        let stage = self
            .stages
            .get(stage_index)
            .with_context(|| format!("execution stage {stage_index} is out of range"))?;
        let names = stage
            .tensor_names
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        residency::with_tensors(weights, &names, device, |loaded| f(stage, loaded))
    }

    /// `with_stage` with the weight load and the stage computation timed
    /// separately; each phase ends at a device synchronization before its
    /// duration is read.
    pub fn with_stage_timed<T>(
        &self,
        weights: &ModelWeights,
        stage_index: usize,
        device: &Device,
        record: &mut dyn FnMut(&'static str, Duration, Duration),
        f: impl FnOnce(&ExecutionStage, &BTreeMap<String, Tensor>) -> Result<T>,
    ) -> Result<T> {
        let stage = self
            .stages
            .get(stage_index)
            .with_context(|| format!("execution stage {stage_index} is out of range"))?;
        let names = stage
            .tensor_names
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let load_started = Instant::now();
        let loaded = residency::materialize(weights, &names, device)?;
        device.synchronize()?;
        let load = load_started.elapsed();
        let compute_started = Instant::now();
        let result = f(stage, &loaded);
        device.synchronize()?;
        record(
            stage.kind.timing_bucket(),
            load,
            compute_started.elapsed(),
        );
        result
    }
}

fn expected_stage_kinds(layers: usize, refiner_layers: usize) -> Vec<StageKind> {
    let mut result = vec![StageKind::ContextInput];
    for i in 0..refiner_layers {
        result.push(StageKind::RefinerAttention(i));
        result.push(StageKind::RefinerFeedForward(i));
    }
    result.push(StageKind::RefinerOutputNorm);
    result.push(StageKind::TimeInput);
    result.push(StageKind::LatentInput);
    for i in 0..layers {
        result.push(StageKind::BlockAdaLn(i));
        result.push(StageKind::BlockAttention(i));
        result.push(StageKind::BlockFeedForward(i));
    }
    result.push(StageKind::Output);
    result
}

fn classify(name: &str) -> Option<StageKind> {
    if name.starts_with("context_embedder.") {
        return Some(StageKind::ContextInput);
    }
    if name.starts_with("time_embedder.") {
        return Some(StageKind::TimeInput);
    }
    if ["proj_in.", "audio_proj_in."]
        .iter()
        .any(|prefix| name.starts_with(prefix))
    {
        return Some(StageKind::LatentInput);
    }
    if name.starts_with("token_refiner.final_norm.") {
        return Some(StageKind::RefinerOutputNorm);
    }
    if let Some((index, suffix)) = indexed_suffix(name, "token_refiner.refiner_blocks.") {
        if suffix.starts_with("norm1.") || suffix.starts_with("attn.") {
            return Some(StageKind::RefinerAttention(index));
        }
        if suffix.starts_with("norm2.") || suffix.starts_with("ff.") {
            return Some(StageKind::RefinerFeedForward(index));
        }
        return None;
    }
    if let Some((index, suffix)) = indexed_suffix(name, "transformer_blocks.") {
        if suffix.starts_with("adaln_proj.") {
            return Some(StageKind::BlockAdaLn(index));
        }
        if suffix.starts_with("norm1.") || suffix.starts_with("attn.") {
            return Some(StageKind::BlockAttention(index));
        }
        if suffix.starts_with("norm2.") || suffix.starts_with("ff.") {
            return Some(StageKind::BlockFeedForward(index));
        }
        return None;
    }
    if ["norm_out.", "proj_out.", "audio_proj_out."]
        .iter()
        .any(|prefix| name.starts_with(prefix))
    {
        return Some(StageKind::Output);
    }
    None
}

fn indexed_suffix<'a>(name: &'a str, prefix: &str) -> Option<(usize, &'a str)> {
    let rest = name.strip_prefix(prefix)?;
    let (index, suffix) = rest.split_once('.')?;
    Some((index.parse().ok()?, suffix))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, safetensors};
    use ff_core::weights::{CachePolicy, WeightSource};
    use serde_json::json;
    use std::{collections::HashMap, fs};

    const TOY_NAMES: [&str; 15] = [
        "context_embedder.weight",
        "time_embedder.linear_1.weight",
        "proj_in.weight",
        "token_refiner.refiner_blocks.0.norm1.weight",
        "token_refiner.refiner_blocks.0.attn.to_q.weight",
        "token_refiner.refiner_blocks.0.norm2.weight",
        "token_refiner.refiner_blocks.0.ff.net.2.weight",
        "token_refiner.final_norm.weight",
        "transformer_blocks.0.adaln_proj.linear.weight",
        "transformer_blocks.0.norm1.weight",
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.0.norm2.weight",
        "transformer_blocks.0.ff.net.2.weight",
        "norm_out.norm.weight",
        "proj_out.weight",
    ];

    fn toy_plan() -> (tempfile::TempDir, ModelWeights, H3ExecutionPlan) {
        let dir = tempfile::tempdir().unwrap();
        let tensors = TOY_NAMES
            .iter()
            .map(|name| {
                (
                    (*name).to_owned(),
                    Tensor::new(&[1f32], &Device::Cpu).unwrap(),
                )
            })
            .collect::<HashMap<_, _>>();
        safetensors::save(&tensors, dir.path().join("weights.safetensors")).unwrap();
        let weight_map = TOY_NAMES
            .iter()
            .map(|name| (*name, "weights.safetensors"))
            .collect::<BTreeMap<_, _>>();
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({
                "metadata": {"total_size": 60},
                "weight_map": weight_map
            }))
            .unwrap(),
        )
        .unwrap();
        let weights =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        let plan = H3ExecutionPlan::build(&weights, 1, 1).unwrap();
        (dir, weights, plan)
    }

    #[test]
    fn produces_ordered_low_memory_stages() {
        let (_dir, weights, plan) = toy_plan();
        assert_eq!(plan.stages().len(), 10);
        assert_eq!(plan.stages()[0].kind, StageKind::ContextInput);
        assert_eq!(plan.stages()[4].kind, StageKind::TimeInput);
        assert_eq!(plan.stages()[5].kind, StageKind::LatentInput);
        assert_eq!(plan.stages()[6].kind, StageKind::BlockAdaLn(0));
        assert_eq!(plan.stages()[9].kind, StageKind::Output);
        assert_eq!(plan.peak_stage_weight_bytes(), 8);
        plan.with_stage(&weights, 6, &Device::Cpu, |stage, tensors| {
            assert_eq!(stage.kind, StageKind::BlockAdaLn(0));
            assert_eq!(tensors.len(), 1);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn with_stage_timed_records_the_bucket_once_per_stage() {
        let (_dir, weights, plan) = toy_plan();
        let mut recorded = Vec::new();
        let value = plan
            .with_stage_timed(
                &weights,
                6,
                &Device::Cpu,
                &mut |bucket, load, compute| recorded.push((bucket, load, compute)),
                |_stage, _tensors| Ok(11u32),
            )
            .unwrap();
        assert_eq!(value, 11);
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].0, "adaln");
    }

    #[test]
    fn recognizes_wider_attention_than_residual_stream() {
        assert_eq!(
            classify("transformer_blocks.49.attn.to_k.weight"),
            Some(StageKind::BlockAttention(49))
        );
    }
}
