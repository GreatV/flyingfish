//! Bounded top-k expert/mixture-weight traces and deterministic cache replay.
//! Tracing preserves routing order and retains no activations.
//! SRP/SCH follow the ICLR 2026 paper and `ljcleo/moe-lrc`
//! commit `30425e6418a9ff5f6afcb2d14b72975888060d28`.

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const ROUTING_TRACE_SCHEMA_VERSION: u32 = 3;
pub const ROUTING_REPLAY_SCHEMA_VERSION: u32 = 2;
pub const MAX_ROUTING_TRACE_JSON_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_ROUTING_REPLAY_JSON_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_ROUTING_TRACE_TOKENS: usize = 2_048;
pub const MAX_ROUTING_TRACE_LAYERS: usize = 256;
pub const MAX_ROUTING_TRACE_EXPERTS: usize = 4_096;
pub const MAX_ROUTING_TRACE_TOP_K: usize = 256;
pub const MAX_ROUTING_TRACE_DOMAIN_BYTES: usize = 128;
pub const MAX_ROUTING_TRACE_EXPERT_ACCESSES: u64 = 4 * 1024 * 1024;
pub const MAX_ROUTING_REPLAY_SEGMENT_LENGTHS: usize = 64;
pub const MAX_ROUTING_REPLAY_BUDGETS: usize = 64;
pub const MAX_ROUTING_REPLAY_BUDGET_BYTES: u64 = 1 << 50;
pub const MAX_ROUTING_REPLAY_MEASUREMENTS: usize = 16 * 1024;
pub const MAX_ROUTING_REPLAY_SIMULATED_ACCESSES: u128 = 256 * 1024 * 1024;
const MAX_EXPERT_ENTRY_BYTES: u64 = 1 << 40;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingModelFamily {
    Glm5Next,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingTracePhase {
    Prefill,
    Decode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingPrefillSchedule {
    TokenSerial,
    LayerBatchedExpertGrouped,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertCacheEntryDtype {
    Bfloat16,
    Float32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertCacheEntryUnit {
    RoutedExpertAllProjections,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutingTraceLayer {
    pub layer_index: u32,
    /// Logical device bytes for each expert's gate, up, and down projections.
    pub expert_bytes: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutingDecision {
    pub token_index: u32,
    pub phase: RoutingTracePhase,
    pub layer_index: u32,
    /// Expert identities in router-returned order (CUDA is not score-sorted).
    pub experts: Vec<u32>,
    /// Final mixture weights aligned with `experts`, after normalization and scaling.
    /// Empty for schema-2 traces, which skip weight validation.
    #[serde(default)]
    pub gate_weights: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutingTrace {
    pub schema_version: u32,
    /// Router scaling used to validate gate weights; absent before schema 3.
    pub routed_scaling_factor: Option<f64>,
    pub norm_topk_prob: Option<bool>,
    pub prefill_schedule: RoutingPrefillSchedule,
    pub model_family: RoutingModelFamily,
    pub domain: String,
    pub cache_entry_dtype: ExpertCacheEntryDtype,
    pub cache_entry_unit: ExpertCacheEntryUnit,
    pub num_hidden_layers: u32,
    pub num_experts: u32,
    pub experts_per_token: u32,
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    /// Tokens that actually traversed the decoder. The last sampled token is
    /// not forwarded, so this is `prompt_tokens + generated_tokens - 1`.
    pub routed_tokens: u32,
    pub layers: Vec<RoutingTraceLayer>,
    pub decisions: Vec<RoutingDecision>,
}

impl RoutingTrace {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (2..=ROUTING_TRACE_SCHEMA_VERSION).contains(&self.schema_version),
            "unsupported GLM routing-trace schema {}; this build supports schemas 2..={}",
            self.schema_version,
            ROUTING_TRACE_SCHEMA_VERSION
        );
        validate_routing_trace_domain(&self.domain)?;
        let hidden_layers = usize::try_from(self.num_hidden_layers)
            .context("routing-trace hidden-layer count exceeds usize")?;
        let num_experts = usize::try_from(self.num_experts)
            .context("routing-trace expert count exceeds usize")?;
        let top_k =
            usize::try_from(self.experts_per_token).context("routing-trace top-k exceeds usize")?;
        let prompt_tokens = usize::try_from(self.prompt_tokens)
            .context("routing-trace prompt-token count exceeds usize")?;
        let generated_tokens = usize::try_from(self.generated_tokens)
            .context("routing-trace generated-token count exceeds usize")?;
        let routed_tokens = usize::try_from(self.routed_tokens)
            .context("routing-trace routed-token count exceeds usize")?;
        ensure!(
            hidden_layers > 0 && hidden_layers <= MAX_ROUTING_TRACE_LAYERS,
            "routing-trace hidden-layer count must be in 1..={MAX_ROUTING_TRACE_LAYERS}"
        );
        ensure!(
            num_experts > 0 && num_experts <= MAX_ROUTING_TRACE_EXPERTS,
            "routing-trace expert count must be in 1..={MAX_ROUTING_TRACE_EXPERTS}"
        );
        ensure!(
            top_k > 0 && top_k <= num_experts && top_k <= MAX_ROUTING_TRACE_TOP_K,
            "routing-trace top-k must be in 1..=min(num_experts, {MAX_ROUTING_TRACE_TOP_K})"
        );
        ensure!(
            prompt_tokens > 0,
            "routing trace must contain prompt tokens"
        );
        ensure!(
            generated_tokens > 0,
            "routing trace must contain at least one sampled token"
        );
        let expected_routed_tokens = prompt_tokens
            .checked_add(generated_tokens)
            .and_then(|tokens| tokens.checked_sub(1))
            .context("routing-trace token count overflow")?;
        ensure!(
            routed_tokens == expected_routed_tokens,
            "routing trace has {routed_tokens} routed tokens; expected {expected_routed_tokens} from prompt/generated counts"
        );
        ensure!(
            routed_tokens <= MAX_ROUTING_TRACE_TOKENS,
            "routing trace has {routed_tokens} tokens, exceeding the {MAX_ROUTING_TRACE_TOKENS}-token limit"
        );
        ensure!(
            !self.layers.is_empty() && self.layers.len() <= hidden_layers,
            "routing trace must describe at least one sparse layer and no more than num_hidden_layers"
        );
        ensure!(
            self.layers.len() <= MAX_ROUTING_TRACE_LAYERS,
            "routing trace has too many sparse layers"
        );

        let mut previous_layer = None;
        for layer in &self.layers {
            let layer_index = usize::try_from(layer.layer_index)
                .context("routing-trace layer index exceeds usize")?;
            ensure!(
                layer_index < hidden_layers,
                "routing-trace layer {} is outside num_hidden_layers {}",
                layer.layer_index,
                self.num_hidden_layers
            );
            if let Some(previous) = previous_layer {
                ensure!(
                    previous < layer.layer_index,
                    "routing-trace sparse layers must be strictly increasing"
                );
            }
            previous_layer = Some(layer.layer_index);
            ensure!(
                layer.expert_bytes.len() == num_experts,
                "routing-trace layer {} has {} expert sizes; expected {num_experts}",
                layer.layer_index,
                layer.expert_bytes.len()
            );
            for &bytes in &layer.expert_bytes {
                ensure!(
                    bytes > 0 && bytes <= MAX_EXPERT_ENTRY_BYTES,
                    "routing-trace expert entry bytes must be in 1..={MAX_EXPERT_ENTRY_BYTES}"
                );
            }
        }

        let expected_decisions = routed_tokens
            .checked_mul(self.layers.len())
            .context("routing-trace decision count overflow")?;
        ensure!(
            self.decisions.len() == expected_decisions,
            "routing trace has {} decisions; expected {expected_decisions}",
            self.decisions.len()
        );
        let maximum_decisions = MAX_ROUTING_TRACE_TOKENS
            .checked_mul(MAX_ROUTING_TRACE_LAYERS)
            .expect("routing-trace static decision bound fits usize");
        ensure!(
            self.decisions.len() <= maximum_decisions,
            "routing trace exceeds the decision-count limit"
        );
        ensure!(
            self.selection_count()? <= MAX_ROUTING_TRACE_EXPERT_ACCESSES,
            "routing trace exceeds the {MAX_ROUTING_TRACE_EXPERT_ACCESSES}-access analysis limit"
        );
        for (sequence, decision) in self.decisions.iter().enumerate() {
            let token_index = sequence / self.layers.len();
            let layer_ordinal = sequence % self.layers.len();
            ensure!(
                usize::try_from(decision.token_index).ok() == Some(token_index),
                "routing decision {sequence} has token index {}; expected {token_index}",
                decision.token_index
            );
            ensure!(
                decision.layer_index == self.layers[layer_ordinal].layer_index,
                "routing decision {sequence} has layer {}; expected {}",
                decision.layer_index,
                self.layers[layer_ordinal].layer_index
            );
            let expected_phase = if token_index < prompt_tokens {
                RoutingTracePhase::Prefill
            } else {
                RoutingTracePhase::Decode
            };
            ensure!(
                decision.phase == expected_phase,
                "routing decision {sequence} has the wrong phase for token {token_index}"
            );
            ensure!(
                decision.experts.len() == top_k,
                "routing decision {sequence} has {} experts; expected {top_k}",
                decision.experts.len()
            );
            if !decision.gate_weights.is_empty() {
                ensure!(
                    decision.gate_weights.len() == decision.experts.len(),
                    "routing decision {sequence} has {} gate weights for {} experts",
                    decision.gate_weights.len(),
                    decision.experts.len()
                );
                // Sanity ceiling for malformed weights, independent of model scaling.
                ensure!(
                    decision
                        .gate_weights
                        .iter()
                        .all(|weight| weight.is_finite() && *weight >= 0.0 && *weight <= 1e3),
                    "routing decision {sequence} gate weights must be finite and in 0..=1e3"
                );
                // Normalized weights must sum to the routed scaling factor.
                if self.norm_topk_prob == Some(true)
                    && let Some(routed_scaling) = self.routed_scaling_factor
                {
                    let sum: f64 = decision
                        .gate_weights
                        .iter()
                        .map(|weight| f64::from(*weight))
                        .sum();
                    let tolerance = 1e-4 * routed_scaling.abs().max(1.0);
                    ensure!(
                        (sum - routed_scaling).abs() <= tolerance,
                        "routing decision {sequence} gate weights sum to {sum}; expected the routed scaling factor {routed_scaling}"
                    );
                }
            }
            for &expert in &decision.experts {
                ensure!(
                    usize::try_from(expert)
                        .ok()
                        .is_some_and(|expert| expert < num_experts),
                    "routing decision {sequence} contains out-of-range expert {expert}"
                );
            }
            ensure!(
                decision.experts.iter().collect::<BTreeSet<_>>().len() == top_k,
                "routing decision {sequence} experts must be unique"
            );
        }
        Ok(())
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() <= MAX_ROUTING_TRACE_JSON_BYTES,
            "GLM routing-trace JSON is {} bytes, exceeding the {MAX_ROUTING_TRACE_JSON_BYTES}-byte limit",
            bytes.len()
        );
        let trace: Self =
            serde_json::from_slice(bytes).context("invalid GLM routing-trace JSON")?;
        trace.validate()?;
        Ok(trace)
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut json = serde_json::to_vec_pretty(self)
            .context("failed to serialize GLM routing-trace JSON")?;
        json.push(b'\n');
        ensure!(
            json.len() <= MAX_ROUTING_TRACE_JSON_BYTES,
            "GLM routing-trace JSON is {} bytes, exceeding the {MAX_ROUTING_TRACE_JSON_BYTES}-byte limit",
            json.len()
        );
        Ok(json)
    }

    fn layer_ordinal(&self, layer_index: u32) -> Result<usize> {
        self.layers
            .binary_search_by_key(&layer_index, |layer| layer.layer_index)
            .map_err(|_| {
                anyhow::anyhow!("routing trace does not describe sparse layer {layer_index}")
            })
    }

    fn selection_count(&self) -> Result<u64> {
        u64::try_from(self.decisions.len())
            .context("routing-trace decision count exceeds u64")?
            .checked_mul(u64::from(self.experts_per_token))
            .context("routing-trace expert access count overflow")
    }

    fn runtime_access_groups(&self) -> Vec<(u32, Vec<u32>)> {
        let mut result = Vec::new();
        if self.prefill_schedule == RoutingPrefillSchedule::LayerBatchedExpertGrouped {
            let mut prefill = BTreeMap::<u32, BTreeSet<u32>>::new();
            for decision in &self.decisions {
                if decision.phase == RoutingTracePhase::Prefill {
                    prefill
                        .entry(decision.layer_index)
                        .or_default()
                        .extend(&decision.experts);
                }
            }
            for (layer, experts) in prefill {
                for expert in experts {
                    result.push((layer, vec![expert]));
                }
            }
        }
        for decision in &self.decisions {
            if self.prefill_schedule == RoutingPrefillSchedule::LayerBatchedExpertGrouped
                && decision.phase == RoutingTracePhase::Prefill
            {
                continue;
            }
            let mut experts = decision.experts.clone();
            experts.sort_unstable();
            result.push((decision.layer_index, experts));
        }
        result
    }

    fn access_count(&self) -> Result<u64> {
        self.runtime_access_groups()
            .iter()
            .try_fold(0u64, |count, (_, experts)| {
                count
                    .checked_add(u64::try_from(experts.len())?)
                    .context("GLM runtime access count overflow")
            })
    }
}

pub fn validate_routing_trace_domain(domain: &str) -> Result<()> {
    ensure!(!domain.is_empty(), "routing-trace domain must not be empty");
    ensure!(
        domain.len() <= MAX_ROUTING_TRACE_DOMAIN_BYTES,
        "routing-trace domain exceeds {MAX_ROUTING_TRACE_DOMAIN_BYTES} UTF-8 bytes"
    );
    ensure!(
        domain.trim() == domain,
        "routing-trace domain must not have leading or trailing whitespace"
    );
    ensure!(
        !domain.chars().any(char::is_control),
        "routing-trace domain must not contain control characters"
    );
    Ok(())
}

pub(crate) struct RoutingTraceBuilder {
    trace: RoutingTrace,
    max_routed_tokens: usize,
}

impl RoutingTraceBuilder {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        domain: &str,
        cache_entry_dtype: ExpertCacheEntryDtype,
        routed_scaling_factor: f64,
        norm_topk_prob: bool,
        num_hidden_layers: usize,
        num_experts: usize,
        experts_per_token: usize,
        prompt_tokens: usize,
        max_routed_tokens: usize,
        layers: Vec<RoutingTraceLayer>,
    ) -> Result<Self> {
        validate_routing_trace_domain(domain)?;
        ensure!(
            max_routed_tokens > 0 && max_routed_tokens <= MAX_ROUTING_TRACE_TOKENS,
            "routing-trace token capacity must be in 1..={MAX_ROUTING_TRACE_TOKENS}"
        );
        ensure!(
            prompt_tokens > 0 && prompt_tokens <= max_routed_tokens,
            "routing-trace prompt-token count exceeds its token capacity"
        );
        ensure!(
            !layers.is_empty() && layers.len() <= MAX_ROUTING_TRACE_LAYERS,
            "routing trace must describe 1..={MAX_ROUTING_TRACE_LAYERS} sparse layers"
        );
        let capacity = max_routed_tokens
            .checked_mul(layers.len())
            .context("routing-trace event capacity overflow")?;
        let maximum_accesses = capacity
            .checked_mul(experts_per_token)
            .context("routing-trace expert-access capacity overflow")?;
        let declared_expert_sizes = layers
            .len()
            .checked_mul(num_experts)
            .context("routing-trace expert-size count overflow")?;
        let conservative_json_bytes = 4_096usize
            .checked_add(domain.len())
            .and_then(|bytes| bytes.checked_add(layers.len().checked_mul(256)?))
            .and_then(|bytes| bytes.checked_add(declared_expert_sizes.checked_mul(32)?))
            .and_then(|bytes| bytes.checked_add(capacity.checked_mul(256)?))
            // Each access prints an expert id line and an f32 weight line.
            .and_then(|bytes| bytes.checked_add(maximum_accesses.checked_mul(64)?))
            .context("routing-trace JSON bound estimate overflow")?;
        ensure!(
            conservative_json_bytes <= MAX_ROUTING_TRACE_JSON_BYTES,
            "routing trace configuration has a conservative JSON bound of {conservative_json_bytes} bytes, exceeding the {MAX_ROUTING_TRACE_JSON_BYTES}-byte limit"
        );
        let trace = RoutingTrace {
            schema_version: ROUTING_TRACE_SCHEMA_VERSION,
            routed_scaling_factor: Some(routed_scaling_factor),
            norm_topk_prob: Some(norm_topk_prob),
            prefill_schedule: RoutingPrefillSchedule::LayerBatchedExpertGrouped,
            model_family: RoutingModelFamily::Glm5Next,
            domain: domain.to_owned(),
            cache_entry_dtype,
            cache_entry_unit: ExpertCacheEntryUnit::RoutedExpertAllProjections,
            num_hidden_layers: u32::try_from(num_hidden_layers)
                .context("routing-trace hidden-layer count exceeds u32")?,
            num_experts: u32::try_from(num_experts)
                .context("routing-trace expert count exceeds u32")?,
            experts_per_token: u32::try_from(experts_per_token)
                .context("routing-trace top-k exceeds u32")?,
            prompt_tokens: u32::try_from(prompt_tokens)
                .context("routing-trace prompt-token count exceeds u32")?,
            generated_tokens: 0,
            routed_tokens: 0,
            layers,
            decisions: Vec::with_capacity(capacity),
        };
        ensure!(
            num_hidden_layers > 0 && num_hidden_layers <= MAX_ROUTING_TRACE_LAYERS,
            "routing-trace hidden-layer count is out of bounds"
        );
        ensure!(
            num_experts > 0 && num_experts <= MAX_ROUTING_TRACE_EXPERTS,
            "routing-trace expert count is out of bounds"
        );
        ensure!(
            experts_per_token > 0
                && experts_per_token <= num_experts
                && experts_per_token <= MAX_ROUTING_TRACE_TOP_K,
            "routing-trace top-k is out of bounds"
        );
        ensure!(
            u64::try_from(maximum_accesses).context("routing-trace access capacity exceeds u64")?
                <= MAX_ROUTING_TRACE_EXPERT_ACCESSES,
            "routing trace configuration exceeds the {MAX_ROUTING_TRACE_EXPERT_ACCESSES}-access analysis limit"
        );
        let mut previous = None;
        for layer in &trace.layers {
            ensure!(
                usize::try_from(layer.layer_index)
                    .ok()
                    .is_some_and(|index| index < num_hidden_layers),
                "routing-trace sparse layer {} is out of range",
                layer.layer_index
            );
            ensure!(
                previous.is_none_or(|index| index < layer.layer_index),
                "routing-trace sparse layers must be strictly increasing"
            );
            previous = Some(layer.layer_index);
            ensure!(
                layer.expert_bytes.len() == num_experts,
                "routing-trace layer {} has the wrong expert-size count",
                layer.layer_index
            );
            ensure!(
                layer
                    .expert_bytes
                    .iter()
                    .all(|&bytes| bytes > 0 && bytes <= MAX_EXPERT_ENTRY_BYTES),
                "routing-trace expert entry size is out of bounds"
            );
        }
        Ok(Self {
            trace,
            max_routed_tokens,
        })
    }

    pub(crate) fn record(
        &mut self,
        token_index: usize,
        phase: RoutingTracePhase,
        layer_index: usize,
        experts: &[u32],
        gate_weights: &[f32],
    ) -> Result<()> {
        ensure!(
            token_index < self.max_routed_tokens,
            "routing trace exceeded its {}-token capacity",
            self.max_routed_tokens
        );
        let layer_count = self.trace.layers.len();
        let sequence = self.trace.decisions.len();
        let expected_token = sequence / layer_count;
        let expected_layer = self.trace.layers[sequence % layer_count].layer_index;
        ensure!(
            token_index == expected_token,
            "routing trace received token {token_index}; expected {expected_token}"
        );
        ensure!(
            u32::try_from(layer_index).ok() == Some(expected_layer),
            "routing trace received layer {layer_index}; expected {expected_layer}"
        );
        let prompt_tokens = usize::try_from(self.trace.prompt_tokens)
            .context("routing-trace prompt count exceeds usize")?;
        let expected_phase = if token_index < prompt_tokens {
            RoutingTracePhase::Prefill
        } else {
            RoutingTracePhase::Decode
        };
        ensure!(
            phase == expected_phase,
            "routing trace received the wrong phase for token {token_index}"
        );
        ensure!(
            experts.len()
                == usize::try_from(self.trace.experts_per_token)
                    .context("routing-trace top-k exceeds usize")?,
            "routing trace received the wrong number of experts"
        );
        let num_experts = self.trace.num_experts;
        ensure!(
            experts.iter().all(|&expert| expert < num_experts),
            "routing trace received an out-of-range expert"
        );
        ensure!(
            experts.iter().collect::<BTreeSet<_>>().len() == experts.len(),
            "routing trace requires unique expert identities"
        );
        ensure!(
            gate_weights.len() == experts.len(),
            "routing trace requires one gate weight per expert"
        );
        ensure!(
            gate_weights
                .iter()
                .all(|weight| weight.is_finite() && *weight >= 0.0),
            "routing trace gate weights must be finite and non-negative"
        );
        self.trace.decisions.push(RoutingDecision {
            token_index: u32::try_from(token_index)
                .context("routing-trace token index exceeds u32")?,
            phase,
            layer_index: expected_layer,
            experts: experts.to_vec(),
            gate_weights: gate_weights.to_vec(),
        });
        Ok(())
    }

    pub(crate) fn finish(mut self, generated_tokens: usize) -> Result<RoutingTrace> {
        let layer_count = self.trace.layers.len();
        ensure!(
            self.trace.decisions.len().is_multiple_of(layer_count),
            "routing trace ended partway through a token"
        );
        self.trace.generated_tokens = u32::try_from(generated_tokens)
            .context("routing-trace generated-token count exceeds u32")?;
        self.trace.routed_tokens = u32::try_from(self.trace.decisions.len() / layer_count)
            .context("routing-trace routed-token count exceeds u32")?;
        self.trace.validate()?;
        Ok(self.trace)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoutingReplayOptions {
    pub segment_lengths: Vec<usize>,
    pub cache_budgets_bytes: Vec<u64>,
}

impl RoutingReplayOptions {
    pub fn new(mut segment_lengths: Vec<usize>, mut cache_budgets_bytes: Vec<u64>) -> Result<Self> {
        ensure!(
            !segment_lengths.is_empty()
                && segment_lengths.len() <= MAX_ROUTING_REPLAY_SEGMENT_LENGTHS,
            "routing replay requires 1..={MAX_ROUTING_REPLAY_SEGMENT_LENGTHS} segment lengths"
        );
        ensure!(
            !cache_budgets_bytes.is_empty()
                && cache_budgets_bytes.len() <= MAX_ROUTING_REPLAY_BUDGETS,
            "routing replay requires 1..={MAX_ROUTING_REPLAY_BUDGETS} cache budgets"
        );
        ensure!(
            segment_lengths
                .iter()
                .all(|&length| length > 0 && length <= MAX_ROUTING_TRACE_TOKENS),
            "routing-replay segment lengths must be in 1..={MAX_ROUTING_TRACE_TOKENS}"
        );
        ensure!(
            cache_budgets_bytes
                .iter()
                .all(|&bytes| bytes > 0 && bytes <= MAX_ROUTING_REPLAY_BUDGET_BYTES),
            "routing-replay cache budgets must be in 1..={MAX_ROUTING_REPLAY_BUDGET_BYTES} bytes"
        );
        segment_lengths.sort_unstable();
        cache_budgets_bytes.sort_unstable();
        ensure!(
            segment_lengths.windows(2).all(|pair| pair[0] != pair[1]),
            "routing-replay segment lengths contain a duplicate"
        );
        ensure!(
            cache_budgets_bytes
                .windows(2)
                .all(|pair| pair[0] != pair[1]),
            "routing-replay cache budgets contain a duplicate"
        );
        Ok(Self {
            segment_lengths,
            cache_budgets_bytes,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RoutingAnalysisScope {
    Model,
    Layer { layer_index: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayCacheLayout {
    PerLayerSplit,
    SharedPool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayCachePolicy {
    Lru,
    Lfu,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SrpMeasurement {
    pub scope: RoutingAnalysisScope,
    pub segment_length: u32,
    pub windows: u32,
    pub activation_threshold: u32,
    pub matched_activations: u64,
    pub predicted_activations: u64,
    pub actual_activations: u64,
    pub best_f1: f64,
    pub segment_routing_size_ratio: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchMeasurement {
    pub scope: RoutingAnalysisScope,
    pub cache_layout: ReplayCacheLayout,
    pub segment_length: u32,
    pub requested_total_budget_bytes: u64,
    pub assigned_scope_budget_bytes: u64,
    pub expert_entry_bytes: u64,
    pub capacity_entries: u64,
    pub active_experts_per_token: u32,
    pub cache_ratio: f64,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheReplayMeasurement {
    pub cache_layout: ReplayCacheLayout,
    pub cache_policy: ReplayCachePolicy,
    pub requested_budget_bytes: u64,
    pub allocated_budget_bytes: u64,
    pub accesses: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub uncacheable_accesses: u64,
    pub final_resident_bytes: u64,
    pub peak_resident_bytes: u64,
    pub hit_rate: f64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutingReplayTraceSummary {
    pub domain: String,
    pub prefill_schedule: RoutingPrefillSchedule,
    pub routed_tokens: u32,
    pub prompt_tokens: u32,
    pub decode_tokens: u32,
    pub sparse_layers: Vec<u32>,
    pub experts_per_token: u32,
    pub expert_selections: u64,
    pub prefill_expert_accesses: u64,
    pub expert_accesses: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutingReplayReport {
    pub schema_version: u32,
    /// Size of the trace file this report was produced from.
    pub trace_bytes: u64,
    pub trace: RoutingReplayTraceSummary,
    pub segment_lengths: Vec<u32>,
    pub cache_budgets_bytes: Vec<u64>,
    pub srp: Vec<SrpMeasurement>,
    pub sch: Vec<SchMeasurement>,
    pub cache_replays: Vec<CacheReplayMeasurement>,
}

impl RoutingReplayReport {
    pub fn analyze(
        trace: &RoutingTrace,
        trace_bytes: u64,
        options: &RoutingReplayOptions,
    ) -> Result<Self> {
        trace.validate()?;
        ensure!(trace_bytes > 0, "routing trace file is empty");
        let routed_tokens = usize::try_from(trace.routed_tokens)
            .context("routing-trace token count exceeds usize")?;
        ensure!(
            options
                .segment_lengths
                .iter()
                .all(|&length| length <= routed_tokens),
            "routing-replay segment length exceeds the trace's {routed_tokens} routed tokens"
        );
        let checked_options = RoutingReplayOptions::new(
            options.segment_lengths.clone(),
            options.cache_budgets_bytes.clone(),
        )?;
        ensure!(
            &checked_options == options,
            "routing-replay options must be normalized by RoutingReplayOptions::new"
        );

        let scope_count = trace
            .layers
            .len()
            .checked_add(1)
            .context("routing-replay scope count overflow")?;
        let measurement_count = options
            .segment_lengths
            .len()
            .checked_mul(scope_count)
            .and_then(|srp| {
                srp.checked_add(
                    options
                        .cache_budgets_bytes
                        .len()
                        .checked_mul(options.segment_lengths.len())?
                        .checked_mul(scope_count)?,
                )
            })
            .and_then(|rows| rows.checked_add(options.cache_budgets_bytes.len().checked_mul(4)?))
            .context("routing-replay measurement count overflow")?;
        ensure!(
            measurement_count <= MAX_ROUTING_REPLAY_MEASUREMENTS,
            "routing replay would produce {measurement_count} measurements, exceeding the {MAX_ROUTING_REPLAY_MEASUREMENTS}-row limit"
        );
        let simulated_accesses = u128::from(trace.access_count()?)
            .checked_mul(
                (2usize
                    .checked_mul(options.segment_lengths.len())
                    .and_then(|work| {
                        work.checked_add(4usize.checked_mul(options.cache_budgets_bytes.len())?)
                    })
                    .context("routing-replay work multiplier overflow")?) as u128,
            )
            .context("routing-replay work bound overflow")?;
        ensure!(
            simulated_accesses <= MAX_ROUTING_REPLAY_SIMULATED_ACCESSES,
            "routing replay would simulate {simulated_accesses} expert accesses, exceeding the {MAX_ROUTING_REPLAY_SIMULATED_ACCESSES}-access limit"
        );

        let scopes = build_scope_sequences(trace)?;
        let mut srp = Vec::with_capacity(
            scopes
                .len()
                .checked_mul(options.segment_lengths.len())
                .context("routing-replay SRP result count overflow")?,
        );
        for &segment_length in &options.segment_lengths {
            for scope in &scopes {
                srp.push(calculate_srp(scope, segment_length)?);
            }
        }

        let mut sch_curves = Vec::with_capacity(
            options
                .segment_lengths
                .len()
                .checked_mul(scopes.len())
                .context("routing-replay SCH curve count overflow")?,
        );
        for &segment_length in &options.segment_lengths {
            for scope in &scopes {
                sch_curves.push(calculate_segment_frequency_curve(
                    &scope.token_keys,
                    scope.universe,
                    segment_length,
                )?);
            }
        }
        let mut sch = Vec::new();
        for &budget in &options.cache_budgets_bytes {
            let quotas = split_budget(budget, trace.layers.len())?;
            for (segment_ordinal, &segment_length) in options.segment_lengths.iter().enumerate() {
                for (scope_ordinal, scope) in scopes.iter().enumerate() {
                    let (layout, assigned) = match scope.scope {
                        RoutingAnalysisScope::Model => (ReplayCacheLayout::SharedPool, budget),
                        RoutingAnalysisScope::Layer { .. } => {
                            (ReplayCacheLayout::PerLayerSplit, quotas[scope_ordinal - 1])
                        }
                    };
                    sch.push(calculate_sch(
                        scope,
                        &sch_curves[segment_ordinal * scopes.len() + scope_ordinal],
                        layout,
                        segment_length,
                        budget,
                        assigned,
                    )?);
                }
            }
        }

        let mut cache_replays = Vec::with_capacity(
            options
                .cache_budgets_bytes
                .len()
                .checked_mul(4)
                .context("routing-replay cache result count overflow")?,
        );
        for &budget in &options.cache_budgets_bytes {
            for layout in [
                ReplayCacheLayout::PerLayerSplit,
                ReplayCacheLayout::SharedPool,
            ] {
                for policy in [ReplayCachePolicy::Lru, ReplayCachePolicy::Lfu] {
                    cache_replays.push(replay_cache(trace, layout, policy, budget)?);
                }
            }
        }
        let decode_tokens = trace
            .routed_tokens
            .checked_sub(trace.prompt_tokens)
            .context("routing-trace prompt count exceeds routed count")?;
        let report = Self {
            schema_version: ROUTING_REPLAY_SCHEMA_VERSION,
            trace_bytes,
            trace: RoutingReplayTraceSummary {
                domain: trace.domain.clone(),
                prefill_schedule: trace.prefill_schedule,
                routed_tokens: trace.routed_tokens,
                prompt_tokens: trace.prompt_tokens,
                decode_tokens,
                sparse_layers: trace.layers.iter().map(|layer| layer.layer_index).collect(),
                experts_per_token: trace.experts_per_token,
                expert_selections: trace.selection_count()?,
                prefill_expert_accesses: trace
                    .access_count()?
                    .checked_sub(
                        u64::from(decode_tokens)
                            * trace.layers.len() as u64
                            * u64::from(trace.experts_per_token),
                    )
                    .context("GLM prefill access count underflow")?,
                expert_accesses: trace.access_count()?,
            },
            segment_lengths: options
                .segment_lengths
                .iter()
                .copied()
                .map(|length| {
                    u32::try_from(length).context("routing-replay segment length exceeds u32")
                })
                .collect::<Result<_>>()?,
            cache_budgets_bytes: options.cache_budgets_bytes.clone(),
            srp,
            sch,
            cache_replays,
        };
        report.validate()?;
        Ok(report)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == ROUTING_REPLAY_SCHEMA_VERSION,
            "unsupported GLM routing-replay schema {}; this build supports schema {}",
            self.schema_version,
            ROUTING_REPLAY_SCHEMA_VERSION
        );
        ensure!(
            self.trace_bytes > 0 && self.trace_bytes <= MAX_ROUTING_TRACE_JSON_BYTES as u64,
            "routing-replay source trace byte count is out of bounds"
        );
        validate_routing_trace_domain(&self.trace.domain)?;
        ensure!(
            self.trace.routed_tokens > 0
                && self.trace.prompt_tokens > 0
                && self.trace.prompt_tokens <= self.trace.routed_tokens,
            "routing-replay trace token summary is invalid"
        );
        ensure!(
            self.trace.decode_tokens == self.trace.routed_tokens - self.trace.prompt_tokens,
            "routing-replay decode-token summary is inconsistent"
        );
        ensure!(
            !self.trace.sparse_layers.is_empty()
                && self.trace.sparse_layers.len() <= MAX_ROUTING_TRACE_LAYERS
                && self
                    .trace
                    .sparse_layers
                    .windows(2)
                    .all(|pair| pair[0] < pair[1]),
            "routing-replay sparse-layer summary is invalid"
        );
        ensure!(
            self.trace.experts_per_token > 0
                && usize::try_from(self.trace.experts_per_token)
                    .ok()
                    .is_some_and(|top_k| top_k <= MAX_ROUTING_TRACE_TOP_K),
            "routing-replay top-k summary is invalid"
        );
        let expected_accesses = u64::from(self.trace.routed_tokens)
            .checked_mul(
                u64::try_from(self.trace.sparse_layers.len())
                    .context("routing-replay layer count exceeds u64")?,
            )
            .and_then(|count| count.checked_mul(u64::from(self.trace.experts_per_token)))
            .context("routing-replay access summary overflow")?;
        ensure!(
            self.trace.expert_selections == expected_accesses,
            "routing-replay expert-selection summary is inconsistent"
        );
        let active =
            self.trace.sparse_layers.len() as u64 * u64::from(self.trace.experts_per_token);
        let prefill_selections = u64::from(self.trace.prompt_tokens) * active;
        let prefill_accesses = self.trace.prefill_expert_accesses;
        ensure!(
            prefill_accesses >= active && prefill_accesses <= prefill_selections,
            "routing-replay grouped prefill accesses are out of bounds"
        );
        if self.trace.prefill_schedule == RoutingPrefillSchedule::TokenSerial {
            ensure!(
                prefill_accesses == prefill_selections,
                "serial prefill access count differs from selections"
            );
        }
        ensure!(
            self.trace.expert_accesses
                == prefill_accesses + u64::from(self.trace.decode_tokens) * active,
            "routing-replay expert-access summary is inconsistent"
        );
        ensure!(
            self.trace.expert_accesses <= MAX_ROUTING_TRACE_EXPERT_ACCESSES,
            "routing-replay expert-access summary exceeds its bound"
        );
        let normalized = RoutingReplayOptions::new(
            self.segment_lengths
                .iter()
                .copied()
                .map(|length| usize::try_from(length).context("segment length exceeds usize"))
                .collect::<Result<_>>()?,
            self.cache_budgets_bytes.clone(),
        )?;
        ensure!(
            normalized.segment_lengths.iter().copied().eq(self
                .segment_lengths
                .iter()
                .copied()
                .map(|value| value as usize))
                && normalized.cache_budgets_bytes == self.cache_budgets_bytes,
            "routing-replay settings are not canonical"
        );
        let scope_count = self
            .trace
            .sparse_layers
            .len()
            .checked_add(1)
            .context("routing-replay scope count overflow")?;
        ensure!(
            self.srp.len()
                == scope_count
                    .checked_mul(self.segment_lengths.len())
                    .context("routing-replay expected SRP count overflow")?,
            "routing-replay SRP result count is inconsistent"
        );
        ensure!(
            self.sch.len()
                == scope_count
                    .checked_mul(self.segment_lengths.len())
                    .and_then(|count| count.checked_mul(self.cache_budgets_bytes.len()))
                    .context("routing-replay expected SCH count overflow")?,
            "routing-replay SCH result count is inconsistent"
        );
        ensure!(
            self.cache_replays.len()
                == self
                    .cache_budgets_bytes
                    .len()
                    .checked_mul(4)
                    .context("routing-replay expected cache result count overflow")?,
            "routing-replay cache result count is inconsistent"
        );
        let measurement_count = self
            .srp
            .len()
            .checked_add(self.sch.len())
            .and_then(|rows| rows.checked_add(self.cache_replays.len()))
            .context("routing-replay measurement count overflow")?;
        ensure!(
            measurement_count <= MAX_ROUTING_REPLAY_MEASUREMENTS,
            "routing-replay report exceeds its measurement-count bound"
        );
        for (index, row) in self.srp.iter().enumerate() {
            let segment_ordinal = index / scope_count;
            let scope_ordinal = index % scope_count;
            let segment = self.segment_lengths[segment_ordinal];
            let scope = replay_scope(&self.trace.sparse_layers, scope_ordinal)?;
            ensure!(
                row.segment_length == segment && row.scope == scope,
                "routing-replay SRP rows are out of canonical order"
            );
            ensure!(
                segment <= self.trace.routed_tokens,
                "routing-replay SRP segment exceeds routed tokens"
            );
            let windows = self.trace.routed_tokens - segment + 1;
            ensure!(row.windows == windows, "SRP window count is inconsistent");
            ensure!(
                row.activation_threshold <= segment,
                "SRP threshold exceeds segment length"
            );
            let active_per_token = replay_scope_active(&self.trace, scope_ordinal)?;
            let expected_actual = u64::from(windows)
                .checked_mul(u64::from(segment))
                .and_then(|count| count.checked_mul(active_per_token))
                .context("SRP expected activation count overflow")?;
            ensure!(
                row.actual_activations == expected_actual,
                "SRP actual activation count is inconsistent"
            );
            validate_rate(row.best_f1, "SRP F1")?;
            ensure!(
                row.segment_routing_size_ratio.is_finite() && row.segment_routing_size_ratio >= 0.0,
                "SRP routing-size ratio is invalid"
            );
            ensure!(
                row.matched_activations <= row.actual_activations
                    && row.matched_activations <= row.predicted_activations,
                "SRP activation counters are inconsistent"
            );
            let denominator = row
                .predicted_activations
                .checked_add(row.actual_activations)
                .context("SRP F1 denominator overflow")?;
            ensure!(denominator > 0, "SRP F1 denominator is zero");
            ensure!(
                row.best_f1 == row.matched_activations as f64 * 2.0 / denominator as f64
                    && row.segment_routing_size_ratio
                        == row.predicted_activations as f64 / row.actual_activations as f64,
                "SRP floating metrics disagree with their integer counters"
            );
        }
        let sch_rows_per_budget = self
            .segment_lengths
            .len()
            .checked_mul(scope_count)
            .context("SCH rows-per-budget overflow")?;
        for (index, row) in self.sch.iter().enumerate() {
            let budget_ordinal = index / sch_rows_per_budget;
            let within_budget = index % sch_rows_per_budget;
            let segment_ordinal = within_budget / scope_count;
            let scope_ordinal = within_budget % scope_count;
            let budget = self.cache_budgets_bytes[budget_ordinal];
            let segment = self.segment_lengths[segment_ordinal];
            let scope = replay_scope(&self.trace.sparse_layers, scope_ordinal)?;
            let quotas = split_budget(budget, self.trace.sparse_layers.len())?;
            let (layout, assigned) = if scope_ordinal == 0 {
                (ReplayCacheLayout::SharedPool, budget)
            } else {
                (ReplayCacheLayout::PerLayerSplit, quotas[scope_ordinal - 1])
            };
            ensure!(
                row.scope == scope
                    && row.cache_layout == layout
                    && row.segment_length == segment
                    && row.requested_total_budget_bytes == budget
                    && row.assigned_scope_budget_bytes == assigned,
                "routing-replay SCH rows are out of canonical order"
            );
            validate_rate(row.hit_rate, "SCH hit rate")?;
            ensure!(
                row.cache_ratio.is_finite() && row.cache_ratio >= 0.0,
                "SCH cache ratio is invalid"
            );
            let accesses = row
                .hits
                .checked_add(row.misses)
                .context("SCH access counters overflow")?;
            let expected_scope_accesses = u64::from(self.trace.routed_tokens)
                .checked_mul(replay_scope_active(&self.trace, scope_ordinal)?)
                .context("SCH expected access count overflow")?;
            ensure!(
                accesses == expected_scope_accesses,
                "SCH access counters are inconsistent"
            );
            ensure!(
                row.requested_total_budget_bytes > 0
                    && row.assigned_scope_budget_bytes <= row.requested_total_budget_bytes
                    && row.expert_entry_bytes > 0,
                "SCH budget fields are invalid"
            );
            ensure!(
                row.capacity_entries == assigned / row.expert_entry_bytes
                    && row.active_experts_per_token as u64
                        == replay_scope_active(&self.trace, scope_ordinal)?
                    && row.cache_ratio
                        == row.capacity_entries as f64 / row.active_experts_per_token as f64
                    && row.hit_rate == rate(row.hits, accesses),
                "SCH derived fields are inconsistent"
            );
        }
        for (index, row) in self.cache_replays.iter().enumerate() {
            let budget_ordinal = index / 4;
            let variant = index % 4;
            let expected_layout = if variant < 2 {
                ReplayCacheLayout::PerLayerSplit
            } else {
                ReplayCacheLayout::SharedPool
            };
            let expected_policy = if variant.is_multiple_of(2) {
                ReplayCachePolicy::Lru
            } else {
                ReplayCachePolicy::Lfu
            };
            ensure!(
                row.requested_budget_bytes == self.cache_budgets_bytes[budget_ordinal]
                    && row.cache_layout == expected_layout
                    && row.cache_policy == expected_policy,
                "routing cache-replay rows are out of canonical order"
            );
            validate_rate(row.hit_rate, "cache-replay hit rate")?;
            ensure!(
                row.requested_budget_bytes == row.allocated_budget_bytes,
                "cache replay did not allocate the same byte budget to both layouts"
            );
            ensure!(
                row.accesses == self.trace.expert_accesses
                    && row.hits.checked_add(row.misses) == Some(row.accesses),
                "cache-replay access counters are inconsistent"
            );
            ensure!(
                row.final_resident_bytes <= row.allocated_budget_bytes
                    && row.peak_resident_bytes <= row.allocated_budget_bytes
                    && row.final_resident_bytes <= row.peak_resident_bytes,
                "cache-replay residency counters exceed the budget"
            );
            ensure!(
                row.uncacheable_accesses <= row.misses
                    && row.hit_rate == rate(row.hits, row.accesses),
                "cache-replay derived counters are inconsistent"
            );
        }
        Ok(())
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() <= MAX_ROUTING_REPLAY_JSON_BYTES,
            "GLM routing-replay JSON is {} bytes, exceeding the {MAX_ROUTING_REPLAY_JSON_BYTES}-byte limit",
            bytes.len()
        );
        let report: Self =
            serde_json::from_slice(bytes).context("invalid GLM routing-replay JSON")?;
        report.validate()?;
        Ok(report)
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut json = serde_json::to_vec_pretty(self)
            .context("failed to serialize GLM routing-replay JSON")?;
        json.push(b'\n');
        ensure!(
            json.len() <= MAX_ROUTING_REPLAY_JSON_BYTES,
            "GLM routing-replay JSON is {} bytes, exceeding the {MAX_ROUTING_REPLAY_JSON_BYTES}-byte limit",
            json.len()
        );
        Ok(json)
    }
}

fn validate_rate(value: f64, label: &str) -> Result<()> {
    ensure!(
        value.is_finite() && (0.0..=1.0).contains(&value),
        "{label} must be finite and in [0, 1]"
    );
    Ok(())
}

fn replay_scope(layers: &[u32], ordinal: usize) -> Result<RoutingAnalysisScope> {
    if ordinal == 0 {
        return Ok(RoutingAnalysisScope::Model);
    }
    Ok(RoutingAnalysisScope::Layer {
        layer_index: *layers
            .get(ordinal - 1)
            .context("routing-replay scope ordinal exceeds sparse layers")?,
    })
}

fn replay_scope_active(summary: &RoutingReplayTraceSummary, ordinal: usize) -> Result<u64> {
    let top_k = u64::from(summary.experts_per_token);
    if ordinal == 0 {
        top_k
            .checked_mul(
                u64::try_from(summary.sparse_layers.len())
                    .context("routing-replay sparse-layer count exceeds u64")?,
            )
            .context("routing-replay model active-expert count overflow")
    } else {
        ensure!(
            ordinal <= summary.sparse_layers.len(),
            "routing-replay layer scope ordinal is out of range"
        );
        Ok(top_k)
    }
}

#[derive(Clone)]
struct ScopeSequence {
    scope: RoutingAnalysisScope,
    universe: usize,
    active_per_token: usize,
    token_keys: Vec<Vec<usize>>,
    expert_entry_bytes: u64,
}

fn build_scope_sequences(trace: &RoutingTrace) -> Result<Vec<ScopeSequence>> {
    let tokens = usize::try_from(trace.routed_tokens).context("routed tokens exceed usize")?;
    let experts = usize::try_from(trace.num_experts).context("experts exceed usize")?;
    let top_k = usize::try_from(trace.experts_per_token).context("top-k exceeds usize")?;
    let layer_count = trace.layers.len();
    let model_universe = layer_count
        .checked_mul(experts)
        .context("model expert universe overflow")?;
    let model_active = layer_count
        .checked_mul(top_k)
        .context("model active-expert count overflow")?;
    let model_entry_bytes = uniform_entry_bytes(
        trace
            .layers
            .iter()
            .flat_map(|layer| layer.expert_bytes.iter().copied()),
        "model",
    )?;
    let mut model_tokens = vec![Vec::with_capacity(model_active); tokens];
    let mut layer_tokens = (0..layer_count)
        .map(|_| vec![Vec::with_capacity(top_k); tokens])
        .collect::<Vec<_>>();
    for decision in &trace.decisions {
        let token = usize::try_from(decision.token_index).context("token index exceeds usize")?;
        let layer = trace.layer_ordinal(decision.layer_index)?;
        for &expert in &decision.experts {
            let expert = usize::try_from(expert).context("expert index exceeds usize")?;
            layer_tokens[layer][token].push(expert);
            model_tokens[token].push(
                layer
                    .checked_mul(experts)
                    .and_then(|base| base.checked_add(expert))
                    .context("model expert key overflow")?,
            );
        }
    }
    let mut scopes = Vec::with_capacity(layer_count + 1);
    scopes.push(ScopeSequence {
        scope: RoutingAnalysisScope::Model,
        universe: model_universe,
        active_per_token: model_active,
        token_keys: model_tokens,
        expert_entry_bytes: model_entry_bytes,
    });
    for (ordinal, tokens) in layer_tokens.into_iter().enumerate() {
        let layer = &trace.layers[ordinal];
        scopes.push(ScopeSequence {
            scope: RoutingAnalysisScope::Layer {
                layer_index: layer.layer_index,
            },
            universe: experts,
            active_per_token: top_k,
            token_keys: tokens,
            expert_entry_bytes: uniform_entry_bytes(
                layer.expert_bytes.iter().copied(),
                &format!("layer {}", layer.layer_index),
            )?,
        });
    }
    Ok(scopes)
}

fn uniform_entry_bytes(mut bytes: impl Iterator<Item = u64>, scope: &str) -> Result<u64> {
    let first = bytes
        .next()
        .with_context(|| format!("routing trace has no expert sizes for {scope}"))?;
    ensure!(
        bytes.all(|candidate| candidate == first),
        "SCH requires uniform expert-entry bytes within {scope}; byte-weighted SCH is not the paper's metric"
    );
    Ok(first)
}

fn calculate_srp(scope: &ScopeSequence, segment_length: usize) -> Result<SrpMeasurement> {
    ensure!(
        segment_length > 0 && segment_length <= scope.token_keys.len(),
        "SRP segment length is outside the trace"
    );
    let windows = scope.token_keys.len() - segment_length + 1;
    let mut frequencies = vec![0usize; scope.universe];
    let mut current_histogram = vec![0u64; segment_length + 1];
    current_histogram[0] = u64::try_from(scope.universe).context("SRP universe exceeds u64")?;
    for token in &scope.token_keys[..segment_length] {
        for &key in token {
            adjust_frequency(&mut frequencies, &mut current_histogram, key, true)?;
        }
    }
    let mut aggregate = vec![0u128; segment_length + 1];
    for start in 0..windows {
        for (frequency, &count) in current_histogram.iter().enumerate() {
            aggregate[frequency] = aggregate[frequency]
                .checked_add(u128::from(count))
                .context("SRP histogram overflow")?;
        }
        if start + 1 < windows {
            for &key in &scope.token_keys[start] {
                adjust_frequency(&mut frequencies, &mut current_histogram, key, false)?;
            }
            for &key in &scope.token_keys[start + segment_length] {
                adjust_frequency(&mut frequencies, &mut current_histogram, key, true)?;
            }
        }
    }
    let actual = aggregate
        .iter()
        .enumerate()
        .try_fold(0u128, |total, (frequency, &count)| {
            total
                .checked_add(
                    count
                        .checked_mul(frequency as u128)
                        .context("SRP actual activation product overflow")?,
                )
                .context("SRP actual activation sum overflow")
        })?;
    ensure!(actual > 0, "SRP scope contains no activations");
    let mut suffix_pairs = 0u128;
    let mut suffix_matches = 0u128;
    let mut candidates = Vec::with_capacity(segment_length + 1);
    for threshold in (0..=segment_length).rev() {
        let count = aggregate[threshold];
        suffix_pairs = suffix_pairs
            .checked_add(count)
            .context("SRP predicted pair count overflow")?;
        suffix_matches = suffix_matches
            .checked_add(
                count
                    .checked_mul(threshold as u128)
                    .context("SRP match product overflow")?,
            )
            .context("SRP match count overflow")?;
        let predicted = suffix_pairs
            .checked_mul(segment_length as u128)
            .context("SRP predicted activation count overflow")?;
        candidates.push((threshold, suffix_matches, predicted));
    }
    candidates.sort_unstable_by_key(|(threshold, _, _)| *threshold);
    let mut best = candidates[0];
    for candidate in candidates.into_iter().skip(1) {
        if ratio_is_greater(
            candidate
                .1
                .checked_mul(2)
                .context("SRP numerator overflow")?,
            candidate
                .2
                .checked_add(actual)
                .context("SRP denominator overflow")?,
            best.1.checked_mul(2).context("SRP numerator overflow")?,
            best.2
                .checked_add(actual)
                .context("SRP denominator overflow")?,
        )? {
            best = candidate;
        }
    }
    let numerator = best.1 * 2;
    let denominator = best.2 + actual;
    Ok(SrpMeasurement {
        scope: scope.scope.clone(),
        segment_length: u32::try_from(segment_length).context("SRP segment exceeds u32")?,
        windows: u32::try_from(windows).context("SRP window count exceeds u32")?,
        activation_threshold: u32::try_from(best.0).context("SRP threshold exceeds u32")?,
        matched_activations: u64::try_from(best.1).context("SRP matches exceed u64")?,
        predicted_activations: u64::try_from(best.2).context("SRP predictions exceed u64")?,
        actual_activations: u64::try_from(actual).context("SRP activations exceed u64")?,
        best_f1: numerator as f64 / denominator as f64,
        segment_routing_size_ratio: best.2 as f64 / actual as f64,
    })
}

fn adjust_frequency(
    frequencies: &mut [usize],
    histogram: &mut [u64],
    key: usize,
    increment: bool,
) -> Result<()> {
    let previous = *frequencies
        .get(key)
        .with_context(|| format!("routing expert key {key} exceeds its scope"))?;
    let next = if increment {
        previous.checked_add(1).context("SRP frequency overflow")?
    } else {
        previous
            .checked_sub(1)
            .context("SRP sliding window removed an inactive expert")?
    };
    ensure!(
        next < histogram.len(),
        "SRP frequency exceeds segment length"
    );
    histogram[previous] = histogram[previous]
        .checked_sub(1)
        .context("SRP histogram underflow")?;
    histogram[next] = histogram[next]
        .checked_add(1)
        .context("SRP histogram overflow")?;
    frequencies[key] = next;
    Ok(())
}

fn ratio_is_greater(
    left_numerator: u128,
    left_denominator: u128,
    right_numerator: u128,
    right_denominator: u128,
) -> Result<bool> {
    let left = left_numerator
        .checked_mul(right_denominator)
        .context("ratio comparison overflow")?;
    let right = right_numerator
        .checked_mul(left_denominator)
        .context("ratio comparison overflow")?;
    Ok(left > right)
}

fn calculate_sch(
    scope: &ScopeSequence,
    curve: &SchHitCurve,
    cache_layout: ReplayCacheLayout,
    segment_length: usize,
    requested_total_budget_bytes: u64,
    assigned_scope_budget_bytes: u64,
) -> Result<SchMeasurement> {
    let capacity_entries = assigned_scope_budget_bytes / scope.expert_entry_bytes;
    let capacity =
        usize::try_from(capacity_entries).context("SCH cache entry capacity exceeds usize")?;
    let outcome = curve.outcome(capacity)?;
    let accesses = outcome
        .hits
        .checked_add(outcome.misses)
        .context("SCH access count overflow")?;
    Ok(SchMeasurement {
        scope: scope.scope.clone(),
        cache_layout,
        segment_length: u32::try_from(segment_length).context("SCH segment exceeds u32")?,
        requested_total_budget_bytes,
        assigned_scope_budget_bytes,
        expert_entry_bytes: scope.expert_entry_bytes,
        capacity_entries,
        active_experts_per_token: u32::try_from(scope.active_per_token)
            .context("SCH active-expert count exceeds u32")?,
        cache_ratio: capacity_entries as f64 / scope.active_per_token as f64,
        hits: outcome.hits,
        misses: outcome.misses,
        hit_rate: rate(outcome.hits, accesses),
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct HitOutcome {
    hits: u64,
    misses: u64,
}

#[derive(Clone, Debug)]
struct SchHitCurve {
    hits_by_capacity: Vec<u64>,
    accesses: u64,
}

impl SchHitCurve {
    fn outcome(&self, capacity: usize) -> Result<HitOutcome> {
        let hits = if capacity == 0 {
            0
        } else {
            *self
                .hits_by_capacity
                .get(capacity.min(self.hits_by_capacity.len()) - 1)
                .context("SCH capacity curve is empty")?
        };
        ensure!(hits <= self.accesses, "SCH hits exceed traced accesses");
        Ok(HitOutcome {
            hits,
            misses: self.accesses - hits,
        })
    }
}

/// Author-compatible future-segment-frequency (FSF) oracle used for SCH.
///
/// The algorithm maintains the complete capacity curve in one pass. Its cold
/// initialization and within-token batching intentionally match
/// `src/41_sch_calc.py::calc_fsf` from the cited authors' repository; replacing
/// it with an ordinary look-ahead eviction simulation changes published SCH.
/// This is an algorithmic Rust translation of that MIT-licensed implementation.
#[cfg(test)]
fn replay_segment_frequency_oracle(
    tokens: &[Vec<usize>],
    universe: usize,
    capacity: usize,
    segment_length: usize,
) -> Result<HitOutcome> {
    calculate_segment_frequency_curve(tokens, universe, segment_length)?.outcome(capacity)
}

fn calculate_segment_frequency_curve(
    tokens: &[Vec<usize>],
    universe: usize,
    segment_length: usize,
) -> Result<SchHitCurve> {
    ensure!(
        segment_length > 0 && segment_length <= tokens.len(),
        "SCH segment length is outside the trace"
    );
    let total_accesses = tokens.iter().try_fold(0u64, |total, token| {
        total
            .checked_add(u64::try_from(token.len()).context("SCH token width exceeds u64")?)
            .context("SCH access count overflow")
    })?;
    ensure!(universe > 0, "SCH expert universe must not be empty");
    const FUTURE_MASK: u64 = 0xffff_ffff_0000_0000;
    const FUTURE_OFFSET: u64 = 0x1_0000_0000;
    let none = universe
        .checked_add(1)
        .context("SCH linked-list sentinel overflow")?;
    let mut timestamp = 0u64;
    let mut property = vec![0u64; universe];
    let mut statistic = vec![0u64; universe];
    let mut rank = vec![universe; universe];
    let mut previous = (0..=universe).collect::<Vec<_>>();
    let mut next = (0..=universe).collect::<Vec<_>>();
    let mut heap_previous = (0..=universe).collect::<Vec<_>>();
    let mut heap_next = (0..=universe).collect::<Vec<_>>();
    let mut unique = 0usize;

    for token in tokens.iter().take(segment_length) {
        for &expert in token {
            property[expert] = property[expert]
                .checked_add(FUTURE_OFFSET)
                .context("SCH future-frequency property overflow")?;
        }
    }

    for (token_index, token) in tokens.iter().enumerate() {
        let mut cursor = next[universe];
        let mut position = 0usize;
        while cursor != universe {
            ensure!(position < universe, "SCH rank list contains a cycle");
            rank[cursor] = position;
            cursor = next[cursor];
            position += 1;
        }

        for &expert in token {
            let position = rank[expert];
            if position == universe {
                ensure!(
                    unique < universe,
                    "SCH unique-expert count exceeds its universe"
                );
                statistic[unique] = statistic[unique]
                    .checked_add(1)
                    .context("SCH capacity statistic overflow")?;
                unique += 1;
            } else {
                statistic[position] = statistic[position]
                    .checked_add(1)
                    .context("SCH capacity statistic overflow")?;
            }
        }

        for &expert in token {
            let insertion_boundary = if next[expert] == expert {
                universe
            } else if heap_next[expert] != none {
                expert
            } else {
                let mut left = expert;
                let mut right = next[expert];
                while heap_next[left] == none && heap_next[right] == none {
                    left = previous[left];
                    right = next[right];
                }
                let parent = if heap_next[left] == none {
                    heap_previous[right]
                } else {
                    left
                };
                let boundary = heap_next[parent];
                heap_next[parent] = expert;
                heap_previous[expert] = parent;
                heap_next[expert] = boundary;
                heap_previous[boundary] = expert;
                expert
            };

            let mut heap_cursor = heap_next[universe];
            let mut rotations = 0usize;
            while heap_cursor != universe
                && heap_cursor != insertion_boundary
                && heap_next[heap_cursor] != universe
                && heap_next[heap_cursor] != insertion_boundary
            {
                ensure!(rotations < universe, "SCH heap traversal contains a cycle");
                rotations += 1;
                let candidate = previous[heap_next[heap_cursor]];
                let mut scan = heap_next[heap_cursor];
                while heap_next[scan] != universe
                    && heap_next[scan] != insertion_boundary
                    && property[previous[heap_next[scan]]] > property[candidate]
                {
                    scan = heap_next[scan];
                }
                ensure!(
                    property[scan] > property[candidate],
                    "SCH ordering invariant failed"
                );
                let mut forward = scan;
                let mut backward = heap_next[scan];
                while heap_next[next[forward]] == none
                    && property[next[forward]] > property[candidate]
                    && heap_next[previous[backward]] == none
                    && property[previous[backward]] < property[candidate]
                {
                    forward = next[forward];
                    backward = previous[backward];
                }
                let destination = if heap_next[next[forward]] == none
                    && property[next[forward]] > property[candidate]
                {
                    previous[backward]
                } else {
                    forward
                };

                let left = previous[candidate];
                let right = next[candidate];
                next[left] = right;
                previous[right] = left;
                if heap_next[candidate] != none {
                    let parent = heap_previous[candidate];
                    let child = heap_next[candidate];
                    heap_next[parent] = child;
                    heap_previous[child] = parent;
                    heap_previous[candidate] = none;
                    heap_next[candidate] = none;
                }
                if left != universe && right != universe && property[left] > property[right] {
                    ensure!(
                        heap_next[right] != none,
                        "SCH heap ordering invariant failed"
                    );
                    let parent = heap_previous[right];
                    let child = heap_next[right];
                    heap_next[parent] = child;
                    heap_previous[child] = parent;
                    heap_previous[right] = none;
                    heap_next[right] = none;
                    if scan == right {
                        scan = parent;
                    }
                }
                let after = next[destination];
                next[destination] = candidate;
                previous[candidate] = destination;
                next[candidate] = after;
                previous[after] = candidate;
                heap_cursor = scan;
            }

            if next[expert] != expert {
                let left = previous[expert];
                let right = next[expert];
                next[left] = right;
                previous[right] = left;
                ensure!(
                    heap_next[expert] != none,
                    "SCH expert is missing its heap edge"
                );
                let parent = heap_previous[expert];
                let child = heap_next[expert];
                if child == right {
                    if left != universe && right != universe && property[left] > property[right] {
                        let grandchild = heap_next[child];
                        heap_next[parent] = grandchild;
                        heap_previous[grandchild] = parent;
                        heap_previous[child] = none;
                        heap_next[child] = none;
                    } else {
                        heap_next[parent] = child;
                        heap_previous[child] = parent;
                    }
                } else {
                    ensure!(heap_next[right] == none, "SCH heap edge is inconsistent");
                    if left != universe && right != universe && property[left] > property[right] {
                        heap_next[parent] = child;
                        heap_previous[child] = parent;
                    } else {
                        heap_next[parent] = right;
                        heap_previous[right] = parent;
                        heap_next[right] = child;
                        heap_previous[child] = right;
                    }
                }
                heap_previous[expert] = none;
                heap_next[expert] = none;
            }

            let first = next[universe];
            next[universe] = expert;
            previous[expert] = universe;
            next[expert] = first;
            previous[first] = expert;
            ensure!(
                heap_next[universe] == first,
                "SCH list and heap heads disagree"
            );
            heap_next[universe] = expert;
            heap_previous[expert] = universe;
            timestamp = timestamp.checked_add(1).context("SCH timestamp overflow")?;
            ensure!(
                timestamp < FUTURE_OFFSET,
                "SCH trace has too many accesses for the author-compatible timestamp encoding"
            );
            let future_property = property[expert] & FUTURE_MASK;
            property[expert] = future_property
                .checked_sub(FUTURE_OFFSET)
                .context("SCH current expert is absent from its future window")?
                | timestamp;
            if first != universe && property[expert] > property[first] {
                let child = heap_next[first];
                heap_next[expert] = child;
                heap_previous[child] = expert;
                heap_previous[first] = none;
                heap_next[first] = none;
            } else {
                heap_next[expert] = first;
                heap_previous[first] = expert;
            }
        }

        let future_index = token_index
            .checked_add(segment_length)
            .context("SCH future-window index overflow")?;
        if future_index < tokens.len() {
            for &expert in &tokens[future_index] {
                property[expert] = property[expert]
                    .checked_add(FUTURE_OFFSET)
                    .context("SCH future-frequency property overflow")?;
                if previous[expert] != universe
                    && heap_next[expert] == none
                    && property[previous[expert]] < property[expert]
                {
                    let mut left = expert;
                    let mut right = next[expert];
                    while heap_next[left] == none && heap_next[right] == none {
                        left = previous[left];
                        right = next[right];
                    }
                    let parent = if heap_next[left] == none {
                        heap_previous[right]
                    } else {
                        left
                    };
                    let child = heap_next[parent];
                    heap_next[parent] = expert;
                    heap_previous[expert] = parent;
                    heap_next[expert] = child;
                    heap_previous[child] = expert;
                }
                if next[expert] != universe
                    && heap_next[next[expert]] != none
                    && property[expert] > property[next[expert]]
                {
                    let parent = heap_previous[next[expert]];
                    let child = next[expert];
                    let grandchild = heap_next[child];
                    heap_next[parent] = grandchild;
                    heap_previous[grandchild] = parent;
                    heap_previous[child] = none;
                    heap_next[child] = none;
                }
            }
        }
    }

    let mut hits_by_capacity = Vec::with_capacity(universe);
    let mut hits = 0u64;
    for count in statistic {
        hits = hits.checked_add(count).context("SCH hit sum overflow")?;
        ensure!(hits <= total_accesses, "SCH hits exceed traced accesses");
        hits_by_capacity.push(hits);
    }
    ensure!(
        hits_by_capacity.last().copied() == Some(total_accesses),
        "SCH full-capacity curve does not cover every traced access"
    );
    Ok(SchHitCurve {
        hits_by_capacity,
        accesses: total_accesses,
    })
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CacheKey {
    layer_index: u32,
    expert: u32,
}

#[derive(Clone, Copy, Debug)]
struct ResidentEntry {
    bytes: u64,
    last_access: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct AccessHistory {
    frequency: u64,
    last_access: u64,
}

struct SimulatedCache {
    policy: ReplayCachePolicy,
    budget_bytes: u64,
    resident_bytes: u64,
    resident: BTreeMap<CacheKey, ResidentEntry>,
    history: BTreeMap<CacheKey, AccessHistory>,
    clock: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
    uncacheable_accesses: u64,
}

impl SimulatedCache {
    fn new(policy: ReplayCachePolicy, budget_bytes: u64) -> Self {
        Self {
            policy,
            budget_bytes,
            resident_bytes: 0,
            resident: BTreeMap::new(),
            history: BTreeMap::new(),
            clock: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            uncacheable_accesses: 0,
        }
    }

    fn access_batch(&mut self, accesses: &[(CacheKey, u64)]) -> Result<()> {
        let mut misses = Vec::new();
        for &(key, bytes) in accesses {
            if self.resident.contains_key(&key) {
                self.hits = self
                    .hits
                    .checked_add(1)
                    .context("cache replay hit overflow")?;
                self.record_access(key)?;
                let entry = self
                    .resident
                    .get_mut(&key)
                    .context("cache replay lost a resident hit")?;
                ensure!(
                    entry.bytes == bytes,
                    "cache replay observed inconsistent bytes for one expert"
                );
                entry.last_access = self.clock;
            } else {
                self.misses = self
                    .misses
                    .checked_add(1)
                    .context("cache replay miss overflow")?;
                misses.push((key, bytes));
            }
        }
        for (key, bytes) in misses {
            self.record_access(key)?;
            if self.budget_bytes == 0 || bytes > self.budget_bytes {
                self.uncacheable_accesses = self
                    .uncacheable_accesses
                    .checked_add(1)
                    .context("cache replay uncacheable count overflow")?;
                continue;
            }
            self.resident_bytes = self
                .resident_bytes
                .checked_add(bytes)
                .context("cache replay resident bytes overflow")?;
            ensure!(
                self.resident
                    .insert(
                        key,
                        ResidentEntry {
                            bytes,
                            last_access: self.clock,
                        },
                    )
                    .is_none(),
                "cache replay inserted an already-resident miss"
            );
            while self.resident_bytes > self.budget_bytes {
                self.evict_one()?;
            }
        }
        Ok(())
    }

    fn record_access(&mut self, key: CacheKey) -> Result<()> {
        self.clock = self
            .clock
            .checked_add(1)
            .context("cache replay clock overflow")?;
        let history = self.history.entry(key).or_default();
        history.frequency = history
            .frequency
            .checked_add(1)
            .context("cache replay frequency overflow")?;
        history.last_access = self.clock;
        Ok(())
    }

    fn evict_one(&mut self) -> Result<()> {
        let victim = self
            .resident
            .keys()
            .copied()
            .min_by_key(|key| match self.policy {
                ReplayCachePolicy::Lru => {
                    let entry = self.resident[key];
                    (0, entry.last_access, *key)
                }
                ReplayCachePolicy::Lfu => {
                    let history = self.history[key];
                    (history.frequency, history.last_access, *key)
                }
            })
            .context("cache replay eviction requested from an empty cache")?;
        let entry = self
            .resident
            .remove(&victim)
            .context("cache replay eviction victim disappeared")?;
        self.resident_bytes = self
            .resident_bytes
            .checked_sub(entry.bytes)
            .context("cache replay resident-byte underflow")?;
        self.evictions = self
            .evictions
            .checked_add(1)
            .context("cache replay eviction count overflow")?;
        Ok(())
    }
}

fn replay_cache(
    trace: &RoutingTrace,
    layout: ReplayCacheLayout,
    policy: ReplayCachePolicy,
    budget: u64,
) -> Result<CacheReplayMeasurement> {
    let quotas = split_budget(budget, trace.layers.len())?;
    let mut caches = match layout {
        ReplayCacheLayout::PerLayerSplit => quotas
            .iter()
            .copied()
            .map(|quota| SimulatedCache::new(policy, quota))
            .collect::<Vec<_>>(),
        ReplayCacheLayout::SharedPool => vec![SimulatedCache::new(policy, budget)],
    };
    let mut peak_resident_bytes = 0u64;
    for (layer_index, runtime_experts) in trace.runtime_access_groups() {
        let layer = trace.layer_ordinal(layer_index)?;
        let accesses = runtime_experts
            .into_iter()
            .map(|expert| {
                let expert_index = usize::try_from(expert).context("expert index exceeds usize")?;
                Ok((
                    CacheKey {
                        layer_index,
                        expert,
                    },
                    trace.layers[layer].expert_bytes[expert_index],
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let cache = match layout {
            ReplayCacheLayout::PerLayerSplit => &mut caches[layer],
            ReplayCacheLayout::SharedPool => &mut caches[0],
        };
        cache.access_batch(&accesses)?;
        let resident = caches.iter().try_fold(0u64, |total, cache| {
            total
                .checked_add(cache.resident_bytes)
                .context("cache replay aggregate residency overflow")
        })?;
        peak_resident_bytes = peak_resident_bytes.max(resident);
    }
    let totals = caches.iter().try_fold(
        (0u64, 0u64, 0u64, 0u64, 0u64),
        |(hits, misses, evictions, uncacheable, resident), cache| {
            Ok::<_, anyhow::Error>((
                hits.checked_add(cache.hits)
                    .context("cache replay hit sum overflow")?,
                misses
                    .checked_add(cache.misses)
                    .context("cache replay miss sum overflow")?,
                evictions
                    .checked_add(cache.evictions)
                    .context("cache replay eviction sum overflow")?,
                uncacheable
                    .checked_add(cache.uncacheable_accesses)
                    .context("cache replay uncacheable sum overflow")?,
                resident
                    .checked_add(cache.resident_bytes)
                    .context("cache replay resident sum overflow")?,
            ))
        },
    )?;
    let accesses = totals
        .0
        .checked_add(totals.1)
        .context("cache replay access sum overflow")?;
    ensure!(
        accesses == trace.access_count()?,
        "cache replay did not consume every traced access"
    );
    let allocated_budget_bytes = caches.iter().try_fold(0u64, |total, cache| {
        total
            .checked_add(cache.budget_bytes)
            .context("cache replay allocated-budget overflow")
    })?;
    ensure!(
        allocated_budget_bytes == budget,
        "cache replay layout did not receive the requested byte budget"
    );
    Ok(CacheReplayMeasurement {
        cache_layout: layout,
        cache_policy: policy,
        requested_budget_bytes: budget,
        allocated_budget_bytes,
        accesses,
        hits: totals.0,
        misses: totals.1,
        evictions: totals.2,
        uncacheable_accesses: totals.3,
        final_resident_bytes: totals.4,
        peak_resident_bytes,
        hit_rate: rate(totals.0, accesses),
    })
}

fn split_budget(total: u64, parts: usize) -> Result<Vec<u64>> {
    ensure!(parts > 0, "cannot split a cache budget across zero layers");
    let parts_u64 = u64::try_from(parts).context("cache budget part count exceeds u64")?;
    let base = total / parts_u64;
    let remainder = total % parts_u64;
    let quotas = (0..parts)
        .map(|index| base + u64::from((index as u64) < remainder))
        .collect::<Vec<_>>();
    ensure!(
        quotas.iter().copied().sum::<u64>() == total,
        "split cache budget does not sum to its input"
    );
    Ok(quotas)
}

fn rate(hits: u64, accesses: u64) -> f64 {
    if accesses == 0 {
        0.0
    } else {
        hits as f64 / accesses as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The size of the trace file a report names.
    fn trace_bytes() -> u64 {
        123
    }

    fn hand_trace(sequence: &[u32]) -> RoutingTrace {
        let decisions = sequence
            .iter()
            .copied()
            .enumerate()
            .map(|(token_index, expert)| RoutingDecision {
                token_index: token_index as u32,
                phase: if token_index < 2 {
                    RoutingTracePhase::Prefill
                } else {
                    RoutingTracePhase::Decode
                },
                layer_index: 1,
                experts: vec![expert],
                gate_weights: vec![0.5],
            })
            .collect();
        RoutingTrace {
            schema_version: ROUTING_TRACE_SCHEMA_VERSION,
            routed_scaling_factor: Some(0.5),
            norm_topk_prob: Some(true),
            prefill_schedule: RoutingPrefillSchedule::TokenSerial,
            model_family: RoutingModelFamily::Glm5Next,
            domain: "hand-calculated".to_owned(),
            cache_entry_dtype: ExpertCacheEntryDtype::Bfloat16,
            cache_entry_unit: ExpertCacheEntryUnit::RoutedExpertAllProjections,
            num_hidden_layers: 2,
            num_experts: 4,
            experts_per_token: 1,
            prompt_tokens: 2,
            generated_tokens: sequence.len() as u32 - 1,
            routed_tokens: sequence.len() as u32,
            layers: vec![RoutingTraceLayer {
                layer_index: 1,
                expert_bytes: vec![1; 4],
            }],
            decisions,
        }
    }

    #[test]
    fn strict_trace_round_trip_rejects_unknown_and_oversize_json() {
        let trace = hand_trace(&[0, 0, 1, 0]);
        let json = trace.canonical_json().unwrap();
        assert_eq!(RoutingTrace::from_json(&json).unwrap(), trace);

        let mut value: serde_json::Value = serde_json::from_slice(&json).unwrap();
        value["unexpected"] = serde_json::json!(true);
        let error = RoutingTrace::from_json(&serde_json::to_vec(&value).unwrap()).unwrap_err();
        assert!(error.to_string().contains("invalid GLM routing-trace JSON"));

        let error =
            RoutingTrace::from_json(&vec![b' '; MAX_ROUTING_TRACE_JSON_BYTES + 1]).unwrap_err();
        assert!(error.to_string().contains("exceeding"));
    }

    #[test]
    fn schema_two_traces_validate_without_gate_weights() {
        let trace = hand_trace(&[0, 0, 1, 0]);
        let mut value: serde_json::Value =
            serde_json::from_slice(&trace.canonical_json().unwrap()).unwrap();
        value["schema_version"] = serde_json::json!(2);
        value
            .as_object_mut()
            .unwrap()
            .remove("routed_scaling_factor");
        value.as_object_mut().unwrap().remove("norm_topk_prob");
        for decision in value["decisions"].as_array_mut().unwrap() {
            decision.as_object_mut().unwrap().remove("gate_weights");
        }
        let v2 = RoutingTrace::from_json(&serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(v2.decisions.iter().all(|d| d.gate_weights.is_empty()));
        assert_eq!(v2.routed_scaling_factor, None);
        v2.validate().unwrap();

        value["schema_version"] = serde_json::json!(ROUTING_TRACE_SCHEMA_VERSION + 1);
        let error = RoutingTrace::from_json(&serde_json::to_vec(&value).unwrap()).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unsupported GLM routing-trace schema")
        );
    }

    #[test]
    fn gate_weight_contract_is_enforced_when_present() {
        let mut trace = hand_trace(&[0, 0, 1, 0]);
        trace.decisions[0].gate_weights = vec![0.4];
        let error = trace.validate().unwrap_err();
        assert!(error.to_string().contains("routed scaling factor"));

        let mut trace = hand_trace(&[0, 0, 1, 0]);
        trace.decisions[0].gate_weights = vec![f32::NAN];
        assert!(
            trace
                .validate()
                .unwrap_err()
                .to_string()
                .contains("0..=1e3")
        );

        let mut trace = hand_trace(&[0, 0, 1, 0]);
        trace.decisions[0].gate_weights = vec![1e3 + 1.0];
        assert!(
            trace
                .validate()
                .unwrap_err()
                .to_string()
                .contains("0..=1e3")
        );

        // Without router parameters, the sum cannot be validated.
        let mut trace = hand_trace(&[0, 0, 1, 0]);
        trace.norm_topk_prob = None;
        trace.decisions[0].gate_weights = vec![0.4];
        trace.validate().unwrap();
    }

    #[test]
    fn trace_validation_rejects_incomplete_or_ambiguous_routes() {
        let mut trace = hand_trace(&[0, 0, 1, 0]);
        trace.decisions.pop();
        assert!(
            trace
                .validate()
                .unwrap_err()
                .to_string()
                .contains("decisions")
        );

        let mut trace = hand_trace(&[0, 0, 1, 0]);
        trace.experts_per_token = 2;
        trace.decisions[0].experts = vec![0, 0];
        trace.decisions[0].gate_weights = vec![0.25, 0.25];
        assert!(trace.validate().unwrap_err().to_string().contains("unique"));

        let mut trace = hand_trace(&[0, 0, 1, 0]);
        trace.experts_per_token = 2;
        for decision in &mut trace.decisions {
            decision.experts = vec![1, 0];
            decision.gate_weights = vec![0.25, 0.25];
        }
        trace.validate().unwrap();

        let mut trace = hand_trace(&[0, 0, 1, 0]);
        trace.domain = "x".repeat(MAX_ROUTING_TRACE_DOMAIN_BYTES + 1);
        assert!(trace.validate().unwrap_err().to_string().contains("domain"));
    }

    #[test]
    fn srp_sch_lru_and_lfu_match_a_hand_calculated_trace() {
        let trace = hand_trace(&[0, 0, 1, 0]);
        let options = RoutingReplayOptions::new(vec![2], vec![1]).unwrap();
        let report = RoutingReplayReport::analyze(&trace, trace_bytes(), &options).unwrap();
        let model_srp = report
            .srp
            .iter()
            .find(|row| row.scope == RoutingAnalysisScope::Model)
            .unwrap();
        assert_eq!(model_srp.activation_threshold, 1);
        assert_eq!(model_srp.matched_activations, 6);
        assert_eq!(model_srp.predicted_activations, 10);
        assert_eq!(model_srp.actual_activations, 6);
        assert_eq!(model_srp.best_f1, 0.75);
        assert!((model_srp.segment_routing_size_ratio - 5.0 / 3.0).abs() < 1e-12);

        let model_sch = report
            .sch
            .iter()
            .find(|row| row.scope == RoutingAnalysisScope::Model)
            .unwrap();
        assert_eq!((model_sch.hits, model_sch.misses), (2, 2));
        assert_eq!(model_sch.hit_rate, 0.5);

        for row in &report.cache_replays {
            assert_eq!(row.requested_budget_bytes, 1);
            assert_eq!(row.allocated_budget_bytes, 1);
            assert_eq!(row.accesses, 4);
            let expected_hits = match row.cache_policy {
                ReplayCachePolicy::Lru => 1,
                ReplayCachePolicy::Lfu => 2,
            };
            assert_eq!(row.hits, expected_hits);
        }
    }

    #[test]
    fn grouped_prefill_replay_preserves_selection_statistics_without_fake_cache_hits() {
        let serial = hand_trace(&[0, 0, 1, 0]);
        let mut grouped = serial.clone();
        grouped.prefill_schedule = RoutingPrefillSchedule::LayerBatchedExpertGrouped;
        let options = RoutingReplayOptions::new(vec![2], vec![1]).unwrap();
        let before = RoutingReplayReport::analyze(&serial, trace_bytes(), &options).unwrap();
        let after = RoutingReplayReport::analyze(&grouped, trace_bytes(), &options).unwrap();
        assert_eq!(before.srp, after.srp);
        assert_eq!(before.sch, after.sch);
        assert_eq!(after.trace.expert_selections, 4);
        assert_eq!(after.trace.prefill_expert_accesses, 1);
        assert_eq!(after.trace.expert_accesses, 3);
        assert_eq!(after.cache_replays[0].hits, 0);
        assert_eq!(before.cache_replays[0].hits, 1);
        assert_eq!(
            grouped.runtime_access_groups(),
            [(1, vec![0]), (1, vec![1]), (1, vec![0])]
        );
        let encoded = after.canonical_json().unwrap();
        RoutingReplayReport::from_json(&encoded).unwrap();
        let mut bad = after;
        bad.trace.prefill_schedule = RoutingPrefillSchedule::TokenSerial;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn shared_pool_borrows_an_idle_layers_bytes_under_the_same_total_budget() {
        let routes = [(0, 0), (0, 1), (0, 0), (0, 1)];
        let mut decisions = Vec::new();
        for (token_index, (layer_zero, layer_one)) in routes.into_iter().enumerate() {
            let phase = if token_index < 2 {
                RoutingTracePhase::Prefill
            } else {
                RoutingTracePhase::Decode
            };
            decisions.push(RoutingDecision {
                token_index: token_index as u32,
                phase,
                layer_index: 0,
                experts: vec![layer_zero],
                gate_weights: vec![0.5],
            });
            decisions.push(RoutingDecision {
                token_index: token_index as u32,
                phase,
                layer_index: 1,
                experts: vec![layer_one],
                gate_weights: vec![0.5],
            });
        }
        let trace = RoutingTrace {
            schema_version: ROUTING_TRACE_SCHEMA_VERSION,
            routed_scaling_factor: Some(0.5),
            norm_topk_prob: Some(true),
            prefill_schedule: RoutingPrefillSchedule::TokenSerial,
            model_family: RoutingModelFamily::Glm5Next,
            domain: "pool-ablation".to_owned(),
            cache_entry_dtype: ExpertCacheEntryDtype::Bfloat16,
            cache_entry_unit: ExpertCacheEntryUnit::RoutedExpertAllProjections,
            num_hidden_layers: 2,
            num_experts: 4,
            experts_per_token: 1,
            prompt_tokens: 2,
            generated_tokens: 3,
            routed_tokens: 4,
            layers: vec![
                RoutingTraceLayer {
                    layer_index: 0,
                    expert_bytes: vec![1; 4],
                },
                RoutingTraceLayer {
                    layer_index: 1,
                    expert_bytes: vec![1; 4],
                },
            ],
            decisions,
        };
        let report = RoutingReplayReport::analyze(
            &trace,
            trace_bytes(),
            &RoutingReplayOptions::new(vec![2], vec![3]).unwrap(),
        )
        .unwrap();
        for policy in [ReplayCachePolicy::Lru, ReplayCachePolicy::Lfu] {
            let split = report
                .cache_replays
                .iter()
                .find(|row| {
                    row.cache_layout == ReplayCacheLayout::PerLayerSplit
                        && row.cache_policy == policy
                })
                .unwrap();
            let shared = report
                .cache_replays
                .iter()
                .find(|row| {
                    row.cache_layout == ReplayCacheLayout::SharedPool && row.cache_policy == policy
                })
                .unwrap();
            assert_eq!(split.allocated_budget_bytes, 3);
            assert_eq!(shared.allocated_budget_bytes, 3);
            assert_eq!(split.hits, 3);
            assert_eq!(shared.hits, 5);
        }
    }

    #[test]
    fn sch_matches_author_calc_fsf_capacity_curves() {
        let scalar = vec![vec![0], vec![1], vec![2], vec![0]];
        for (segment, expected_hits) in [(1, [1, 2, 4]), (2, [1, 3, 4])] {
            for (capacity, expected) in expected_hits.into_iter().enumerate() {
                let outcome =
                    replay_segment_frequency_oracle(&scalar, 3, capacity + 1, segment).unwrap();
                assert_eq!(outcome.hits, expected);
                assert_eq!(outcome.hits + outcome.misses, 4);
            }
        }

        let top_two = vec![vec![0, 1], vec![1, 2], vec![0, 2], vec![1, 2]];
        for (segment, expected_hits) in [(1, [4, 5, 8]), (2, [4, 6, 8])] {
            for (capacity, expected) in expected_hits.into_iter().enumerate() {
                let outcome =
                    replay_segment_frequency_oracle(&top_two, 3, capacity + 1, segment).unwrap();
                assert_eq!(outcome.hits, expected);
                assert_eq!(outcome.hits + outcome.misses, 8);
            }
        }
    }

    #[test]
    fn sch_small_exhaustive_sweep_matches_author_calc_fsf() {
        let mut total_hits = 0u128;
        let mut samples: Vec<(String, u64)> = Vec::new();
        let mut rows = 0usize;
        for universe in 2usize..=4 {
            for top_k in 1usize..=2.min(universe) {
                let batches = if top_k == 1 {
                    (0..universe).map(|expert| vec![expert]).collect::<Vec<_>>()
                } else {
                    (0..universe)
                        .flat_map(|left| (left + 1..universe).map(move |right| vec![left, right]))
                        .collect::<Vec<_>>()
                };
                for token_count in 1usize..=5 {
                    let sequence_count = batches.len().pow(token_count as u32);
                    for ordinal in 0..sequence_count {
                        let mut remainder = ordinal;
                        let mut sequence = Vec::with_capacity(token_count);
                        for position in 0..token_count {
                            let divisor = batches.len().pow((token_count - position - 1) as u32);
                            let batch = remainder / divisor;
                            remainder %= divisor;
                            sequence.push(batches[batch].clone());
                        }
                        let flat = sequence
                            .iter()
                            .map(|batch| {
                                batch
                                    .iter()
                                    .map(usize::to_string)
                                    .collect::<Vec<_>>()
                                    .join(",")
                            })
                            .collect::<Vec<_>>()
                            .join(";");
                        for segment in 1..=token_count {
                            for capacity in 1..=universe {
                                let outcome = replay_segment_frequency_oracle(
                                    &sequence, universe, capacity, segment,
                                )
                                .unwrap();
                                total_hits += outcome.hits as u128;
                                if (universe, top_k, token_count, segment, capacity)
                                    == (4, 2, 5, 3, 2)
                                {
                                    samples.push((flat.clone(), outcome.hits));
                                }
                                rows += 1;
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(rows, 215_040);
        assert_eq!(total_hits, 1_371_466);
        // Four sequences at one coordinate, so a failure shows which routing
        // outcome moved rather than only that the sweep as a whole did.
        assert_eq!(
            samples[..4],
            [
                ("0,1;0,1;0,1;0,1;0,1".to_owned(), 10),
                ("0,1;0,1;0,1;0,1;0,2".to_owned(), 9),
                ("0,1;0,1;0,1;0,1;0,3".to_owned(), 9),
                ("0,1;0,1;0,1;0,1;1,2".to_owned(), 9),
            ]
        );
    }

    #[test]
    fn srp_small_exhaustive_sweep_matches_author_threshold_aggregation() {
        let mut totals = (0u128, 0u128, 0u128, 0u128);
        let mut srp_samples: Vec<(String, u64, u64)> = Vec::new();
        let mut rows = 0usize;
        for universe in 2usize..=4 {
            for top_k in 1usize..=2.min(universe) {
                let batches = if top_k == 1 {
                    (0..universe).map(|expert| vec![expert]).collect::<Vec<_>>()
                } else {
                    (0..universe)
                        .flat_map(|left| (left + 1..universe).map(move |right| vec![left, right]))
                        .collect::<Vec<_>>()
                };
                for token_count in 1usize..=5 {
                    let sequence_count = batches.len().pow(token_count as u32);
                    for ordinal in 0..sequence_count {
                        let mut remainder = ordinal;
                        let mut sequence = Vec::with_capacity(token_count);
                        for position in 0..token_count {
                            let divisor = batches.len().pow((token_count - position - 1) as u32);
                            let batch = remainder / divisor;
                            remainder %= divisor;
                            sequence.push(batches[batch].clone());
                        }
                        let flat = sequence
                            .iter()
                            .map(|batch| {
                                batch
                                    .iter()
                                    .map(usize::to_string)
                                    .collect::<Vec<_>>()
                                    .join(",")
                            })
                            .collect::<Vec<_>>()
                            .join(";");
                        let scope = ScopeSequence {
                            scope: RoutingAnalysisScope::Model,
                            universe,
                            active_per_token: top_k,
                            token_keys: sequence,
                            expert_entry_bytes: 1,
                        };
                        for segment in 1..=token_count {
                            let row = calculate_srp(&scope, segment).unwrap();
                            totals = (
                                totals.0 + row.activation_threshold as u128,
                                totals.1 + row.matched_activations as u128,
                                totals.2 + row.predicted_activations as u128,
                                totals.3 + row.actual_activations as u128,
                            );
                            if (universe, top_k, token_count, segment) == (4, 2, 5, 3) {
                                srp_samples.push((
                                    flat.clone(),
                                    row.matched_activations,
                                    row.actual_activations,
                                ));
                            }
                            rows += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(rows, 54_717);
        // Thresholds, matched, predicted and actual activations over the sweep.
        assert_eq!(totals, (70_632, 623_300, 934_690, 675_428));
        assert_eq!(
            srp_samples[..3],
            [
                ("0,1;0,1;0,1;0,1;0,1".to_owned(), 18, 18),
                ("0,1;0,1;0,1;0,1;0,2".to_owned(), 17, 18),
                ("0,1;0,1;0,1;0,1;0,3".to_owned(), 17, 18),
            ]
        );
    }

    #[test]
    fn replay_fractional_metrics_survive_json_round_trip_exactly() {
        for seed in 0..32u32 {
            let mut state = seed;
            let sequence = (0..28)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (state >> 16) % 16
                })
                .collect::<Vec<_>>();
            let mut trace = hand_trace(&sequence);
            trace.num_experts = 16;
            trace.layers[0].expert_bytes = vec![1; 16];
            let options = RoutingReplayOptions::new(vec![4, 16], vec![1, 2]).unwrap();
            let report = RoutingReplayReport::analyze(&trace, trace_bytes(), &options).unwrap();
            let encoded = report.canonical_json().unwrap();
            let decoded = RoutingReplayReport::from_json(&encoded)
                .unwrap_or_else(|error| panic!("seed {seed}: {error}"));
            assert_eq!(decoded.canonical_json().unwrap(), encoded);
        }
    }

    #[test]
    fn replay_is_byte_deterministic_and_report_parser_is_bounded() {
        let trace = hand_trace(&[0, 0, 1, 0]);
        let options = RoutingReplayOptions::new(vec![2, 1], vec![2, 1]).unwrap();
        assert_eq!(options.segment_lengths, [1, 2]);
        assert_eq!(options.cache_budgets_bytes, [1, 2]);
        let first = RoutingReplayReport::analyze(&trace, trace_bytes(), &options)
            .unwrap()
            .canonical_json()
            .unwrap();
        let second = RoutingReplayReport::analyze(&trace, trace_bytes(), &options)
            .unwrap()
            .canonical_json()
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(
            RoutingReplayReport::from_json(&first)
                .unwrap()
                .canonical_json()
                .unwrap(),
            first
        );
        let mut tampered: serde_json::Value = serde_json::from_slice(&first).unwrap();
        tampered["cache_replays"][0]["hit_rate"] = serde_json::json!(0.875);
        assert!(
            RoutingReplayReport::from_json(&serde_json::to_vec(&tampered).unwrap())
                .unwrap_err()
                .to_string()
                .contains("derived counters")
        );
        let mut unknown: serde_json::Value = serde_json::from_slice(&first).unwrap();
        unknown["unknown"] = serde_json::json!(true);
        assert!(
            RoutingReplayReport::from_json(&serde_json::to_vec(&unknown).unwrap())
                .unwrap_err()
                .to_string()
                .contains("invalid GLM routing-replay JSON")
        );
        assert!(
            RoutingReplayReport::from_json(&vec![b' '; MAX_ROUTING_REPLAY_JSON_BYTES + 1])
                .unwrap_err()
                .to_string()
                .contains("exceeding")
        );
    }

    #[test]
    fn options_reject_duplicates_and_out_of_range_values() {
        assert!(RoutingReplayOptions::new(vec![1, 1], vec![1]).is_err());
        assert!(RoutingReplayOptions::new(vec![0], vec![1]).is_err());
        assert!(RoutingReplayOptions::new(vec![1], vec![0]).is_err());
        assert!(
            RoutingReplayOptions::new(vec![1], vec![MAX_ROUTING_REPLAY_BUDGET_BYTES + 1]).is_err()
        );
    }

    #[test]
    fn unequal_expert_sizes_fail_sch_instead_of_changing_its_definition() {
        let mut trace = hand_trace(&[0, 0, 1, 0]);
        trace.layers[0].expert_bytes[3] = 2;
        trace.validate().unwrap();
        let error = RoutingReplayReport::analyze(
            &trace,
            trace_bytes(),
            &RoutingReplayOptions::new(vec![2], vec![1]).unwrap(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("uniform expert-entry bytes"));
    }

    #[test]
    fn retained_a4_shape_full_replay_matrix_stays_within_artifact_bounds() {
        let sparse_layers = (3u32..45).collect::<Vec<_>>();
        let mut decisions = Vec::with_capacity(96 * sparse_layers.len());
        for token in 0u32..96 {
            for &layer in &sparse_layers {
                let start = (token * 7 + layer * 11) % 288;
                decisions.push(RoutingDecision {
                    token_index: token,
                    phase: if token < 33 {
                        RoutingTracePhase::Prefill
                    } else {
                        RoutingTracePhase::Decode
                    },
                    layer_index: layer,
                    experts: (0..8).map(|offset| (start + offset * 13) % 288).collect(),
                    gate_weights: vec![0.125; 8],
                });
            }
        }
        let trace = RoutingTrace {
            schema_version: ROUTING_TRACE_SCHEMA_VERSION,
            routed_scaling_factor: Some(1.0),
            norm_topk_prob: Some(true),
            prefill_schedule: RoutingPrefillSchedule::TokenSerial,
            model_family: RoutingModelFamily::Glm5Next,
            domain: "a4-shape-synthetic-routes".to_owned(),
            cache_entry_dtype: ExpertCacheEntryDtype::Bfloat16,
            cache_entry_unit: ExpertCacheEntryUnit::RoutedExpertAllProjections,
            num_hidden_layers: 45,
            num_experts: 288,
            experts_per_token: 8,
            prompt_tokens: 33,
            generated_tokens: 64,
            routed_tokens: 96,
            layers: sparse_layers
                .iter()
                .copied()
                .map(|layer_index| RoutingTraceLayer {
                    layer_index,
                    expert_bytes: vec![48 * 1024 * 1024; 288],
                })
                .collect(),
            decisions,
        };
        trace.validate().unwrap();
        assert!(trace.canonical_json().unwrap().len() < MAX_ROUTING_TRACE_JSON_BYTES);
        let report = RoutingReplayReport::analyze(
            &trace,
            trace_bytes(),
            &RoutingReplayOptions::new(
                vec![4, 16, 64],
                vec![4_096, 8_192, 16_128, 32_256]
                    .into_iter()
                    .map(|mib| mib * 1024 * 1024)
                    .collect(),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(report.srp.len(), 3 * 43);
        assert_eq!(report.sch.len(), 4 * 3 * 43);
        assert_eq!(report.cache_replays.len(), 4 * 4);
        assert!(report.canonical_json().unwrap().len() < MAX_ROUTING_REPLAY_JSON_BYTES);
    }
    use super::super::test_support::*;
    use super::super::*;

    #[test]
    fn routing_trace_is_post_selection_and_preserves_generation() {
        let checkpoint = tiny_checkpoint();
        let options = GlmGenerationOptions {
            max_new_tokens: 2,
            max_context_tokens: 8,
            reasoning_effort: "low".to_owned(),
            temperature: 1.0,
            top_p: 0.9,
            seed: 7,
            progress: false,
        };
        let baseline_model = StreamedGlm::open(checkpoint.path(), tiny_options()).unwrap();
        let baseline = baseline_model
            .generate("exercise every tiny GLM path", &options)
            .unwrap();

        let traced_model = StreamedGlm::open(checkpoint.path(), tiny_options()).unwrap();
        let (traced, trace) = traced_model
            .generate_with_routing_trace(
                "exercise every tiny GLM path",
                &options,
                "tiny-determinism",
            )
            .unwrap();
        assert_eq!(traced.generated_token_ids, baseline.generated_token_ids);
        assert_eq!(traced.text, baseline.text);
        assert_eq!(trace.domain, "tiny-determinism");
        assert_eq!(trace.cache_entry_dtype, ExpertCacheEntryDtype::Float32);
        assert_eq!(
            trace.cache_entry_unit,
            ExpertCacheEntryUnit::RoutedExpertAllProjections
        );
        assert_eq!(trace.prompt_tokens, 1);
        assert_eq!(trace.generated_tokens, 2);
        assert_eq!(trace.routed_tokens, 2);
        assert_eq!(trace.layers.len(), 1);
        assert_eq!(trace.layers[0].layer_index, 1);
        assert_eq!(trace.layers[0].expert_bytes, [192, 192]);
        assert_eq!(trace.decisions.len(), 2);
        assert_eq!(trace.decisions[0].phase, RoutingTracePhase::Prefill);
        assert_eq!(trace.decisions[1].phase, RoutingTracePhase::Decode);
        assert_eq!(trace.decisions[0].experts, [0, 1]);
        assert_eq!(trace.decisions[1].experts, [0, 1]);
        // The fixture normalizes top-k weights and scales their sum to 2.5.
        for decision in &trace.decisions {
            assert_eq!(decision.gate_weights.len(), 2);
            let sum: f32 = decision.gate_weights.iter().sum();
            assert!((sum - 2.5).abs() < 1e-5, "gate weights sum to {sum}");
        }
        assert_eq!(
            RoutingTrace::from_json(&trace.canonical_json().unwrap()).unwrap(),
            trace
        );
    }
}
