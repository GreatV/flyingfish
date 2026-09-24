use super::{
    execution_policy::GlmExecutionPolicy,
    expert_cache::ExpertCacheStats,
    routing_trace::{MAX_ROUTING_TRACE_TOKENS, RoutingTracePhase},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

pub const GLM_EXECUTION_MANIFEST_SCHEMA_VERSION: u32 = 2;
pub const MAX_GLM_EXECUTION_MANIFEST_JSON_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmExpertCacheSnapshot {
    pub bound_bytes: u64,
    pub realized_bytes: u64,
    pub entries: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl GlmExpertCacheSnapshot {
    pub fn from_stats(stats: ExpertCacheStats) -> Result<Self> {
        let snapshot = Self {
            bound_bytes: u64::try_from(stats.max_bytes)
                .context("GLM expert-cache bound exceeds u64")?,
            realized_bytes: u64::try_from(stats.bytes)
                .context("GLM expert-cache realized bytes exceed u64")?,
            entries: u64::try_from(stats.entries)
                .context("GLM expert-cache entry count exceeds u64")?,
            hits: stats.hits,
            misses: stats.misses,
            evictions: stats.evictions,
        };
        snapshot.validate()?;
        Ok(snapshot)
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.realized_bytes <= self.bound_bytes,
            "GLM expert-cache realized occupancy exceeds its bound"
        );
        ensure!(
            (self.realized_bytes == 0) == (self.entries == 0),
            "GLM expert-cache entry count disagrees with realized occupancy"
        );
        self.hits
            .checked_add(self.misses)
            .context("GLM expert-cache access counters overflow")?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlmCacheAdmissionDecision {
    Unchanged,
    Shrunk,
    Grown,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmCacheReadmissionEvent {
    pub sequence: u32,
    /// One-based count after the full prefill or a decode token synchronized.
    pub after_routed_token: u32,
    pub phase: RoutingTracePhase,
    pub available_memory_bytes: u64,
    /// Available memory plus current cache occupancy, without double-counting
    /// any already-resident non-cache tensors.
    pub reclaimable_capacity_bytes: u64,
    pub required_future_headroom_bytes: u64,
    pub admissible_cache_bound_bytes: u64,
    pub previous_bound_bytes: u64,
    pub new_bound_bytes: u64,
    pub realized_before_bytes: u64,
    pub realized_after_bytes: u64,
    pub evictions: u64,
    pub decision: GlmCacheAdmissionDecision,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmExecutionManifest {
    pub schema_version: u32,
    pub policy: GlmExecutionPolicy,
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    pub routed_tokens: u32,
    pub initial_expert_cache: GlmExpertCacheSnapshot,
    pub final_expert_cache: GlmExpertCacheSnapshot,
    pub maximum_observed_safe_point_realized_bytes: u64,
    pub readmissions: Vec<GlmCacheReadmissionEvent>,
}

impl GlmExecutionManifest {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == GLM_EXECUTION_MANIFEST_SCHEMA_VERSION,
            "unsupported GLM execution-manifest schema {}; this build supports schema {}",
            self.schema_version,
            GLM_EXECUTION_MANIFEST_SCHEMA_VERSION
        );
        self.policy.validate()?;
        ensure!(
            self.prompt_tokens > 0 && self.generated_tokens > 0,
            "GLM execution manifest requires prompt and generated tokens"
        );
        let expected_routed = self
            .prompt_tokens
            .checked_add(self.generated_tokens)
            .and_then(|tokens| tokens.checked_sub(1))
            .context("GLM execution manifest token count overflow")?;
        ensure!(
            self.routed_tokens == expected_routed
                && usize::try_from(self.routed_tokens)
                    .ok()
                    .is_some_and(|tokens| tokens <= MAX_ROUTING_TRACE_TOKENS),
            "GLM execution manifest routed-token count is inconsistent"
        );
        ensure!(
            self.routed_tokens <= self.policy.dsa_context_bound_tokens,
            "GLM execution manifest exceeds its policy DSA context bound"
        );
        self.initial_expert_cache.validate()?;
        self.final_expert_cache.validate()?;
        let cache_policy = &self.policy.expert_cache;
        ensure!(
            (cache_policy.minimum_bound_bytes..=cache_policy.maximum_bound_bytes)
                .contains(&self.initial_expert_cache.bound_bytes),
            "GLM execution manifest initial cache bound {} is outside the policy range {}..={}",
            self.initial_expert_cache.bound_bytes,
            cache_policy.minimum_bound_bytes,
            cache_policy.maximum_bound_bytes
        );
        ensure!(
            self.final_expert_cache.bound_bytes >= cache_policy.minimum_bound_bytes
                && self.final_expert_cache.bound_bytes <= cache_policy.maximum_bound_bytes,
            "GLM execution manifest final cache bound is outside policy"
        );
        let adaptive = cache_policy.readmission_interval_tokens.is_some();
        ensure!(
            self.readmissions.len()
                == if adaptive {
                    self.generated_tokens as usize
                } else {
                    0
                },
            "GLM execution manifest readmission count disagrees with policy"
        );
        let mut previous_bound = self.initial_expert_cache.bound_bytes;
        let mut maximum_realized = self.initial_expert_cache.realized_bytes;
        for (index, event) in self.readmissions.iter().enumerate() {
            ensure!(
                usize::try_from(event.sequence).ok() == Some(index),
                "GLM cache readmission sequence is not contiguous"
            );
            let after_token = self
                .prompt_tokens
                .checked_add(u32::try_from(index)?)
                .context("GLM cache readmission token index exceeds u32")?;
            ensure!(
                event.after_routed_token == after_token,
                "GLM cache readmission token boundary is not contiguous"
            );
            let expected_phase = if after_token <= self.prompt_tokens {
                RoutingTracePhase::Prefill
            } else {
                RoutingTracePhase::Decode
            };
            ensure!(
                event.phase == expected_phase,
                "GLM cache readmission phase disagrees with its token boundary"
            );
            ensure!(
                event.previous_bound_bytes == previous_bound,
                "GLM cache readmission does not continue the preceding cache bound"
            );
            ensure!(
                event.realized_before_bytes <= event.previous_bound_bytes
                    && event.realized_after_bytes <= event.new_bound_bytes,
                "GLM cache readmission realized occupancy exceeds a bound"
            );
            ensure!(
                event.reclaimable_capacity_bytes
                    == event
                        .available_memory_bytes
                        .checked_add(event.realized_before_bytes)
                        .context("GLM cache readmission reclaimable capacity overflow")?,
                "GLM cache readmission reclaimable capacity is inconsistent"
            );
            ensure!(
                event.admissible_cache_bound_bytes
                    == event
                        .reclaimable_capacity_bytes
                        .saturating_sub(event.required_future_headroom_bytes),
                "GLM cache readmission admissible bound is inconsistent"
            );
            ensure!(
                event.admissible_cache_bound_bytes >= cache_policy.minimum_bound_bytes,
                "successful GLM cache readmission fell below the policy minimum"
            );
            let expected_bound = cache_policy
                .maximum_bound_bytes
                .min(event.admissible_cache_bound_bytes);
            ensure!(
                event.new_bound_bytes == expected_bound,
                "GLM cache readmission did not choose the maximum admitted policy bound"
            );
            let expected_decision = match event.new_bound_bytes.cmp(&event.previous_bound_bytes) {
                Ordering::Less => GlmCacheAdmissionDecision::Shrunk,
                Ordering::Equal => GlmCacheAdmissionDecision::Unchanged,
                Ordering::Greater => GlmCacheAdmissionDecision::Grown,
            };
            ensure!(
                event.decision == expected_decision,
                "GLM cache readmission decision disagrees with its bounds"
            );
            match event.decision {
                GlmCacheAdmissionDecision::Shrunk => ensure!(
                    event.realized_after_bytes <= event.realized_before_bytes,
                    "GLM cache shrink increased realized occupancy"
                ),
                GlmCacheAdmissionDecision::Unchanged | GlmCacheAdmissionDecision::Grown => {
                    ensure!(
                        event.realized_after_bytes == event.realized_before_bytes
                            && event.evictions == 0,
                        "unchanged/grown GLM cache admission mutated realized occupancy"
                    )
                }
            }
            previous_bound = event.new_bound_bytes;
            maximum_realized = maximum_realized
                .max(event.realized_before_bytes)
                .max(event.realized_after_bytes);
        }
        ensure!(
            self.final_expert_cache.bound_bytes == previous_bound
                && self.readmissions.last().is_none_or(|event| {
                    self.final_expert_cache.realized_bytes == event.realized_after_bytes
                }),
            "GLM execution manifest final cache state differs from its readmission chain"
        );
        maximum_realized = maximum_realized.max(self.final_expert_cache.realized_bytes);
        ensure!(
            self.maximum_observed_safe_point_realized_bytes == maximum_realized,
            "GLM execution manifest maximum observed occupancy is inconsistent"
        );
        ensure!(
            self.final_expert_cache.hits >= self.initial_expert_cache.hits
                && self.final_expert_cache.misses >= self.initial_expert_cache.misses
                && self.final_expert_cache.evictions >= self.initial_expert_cache.evictions,
            "GLM execution manifest cache counters moved backwards"
        );
        Ok(())
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut json = serde_json::to_vec_pretty(self)
            .context("failed to serialize GLM execution manifest")?;
        json.push(b'\n');
        ensure!(
            json.len() <= MAX_GLM_EXECUTION_MANIFEST_JSON_BYTES,
            "GLM execution-manifest JSON exceeds {MAX_GLM_EXECUTION_MANIFEST_JSON_BYTES} bytes"
        );
        Ok(json)
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() <= MAX_GLM_EXECUTION_MANIFEST_JSON_BYTES,
            "GLM execution-manifest JSON exceeds {MAX_GLM_EXECUTION_MANIFEST_JSON_BYTES} bytes"
        );
        let manifest: Self =
            serde_json::from_slice(bytes).context("invalid GLM execution-manifest JSON")?;
        manifest.validate()?;
        Ok(manifest)
    }
}

pub(crate) struct GlmExecutionManifestRecorder {
    policy: GlmExecutionPolicy,
    initial: GlmExpertCacheSnapshot,
    maximum_realized: u64,
    readmissions: Vec<GlmCacheReadmissionEvent>,
}

impl GlmExecutionManifestRecorder {
    pub(crate) fn next_sequence(&self) -> Result<u32> {
        u32::try_from(self.readmissions.len()).context("GLM readmission sequence exceeds u32")
    }
    pub(crate) fn new(policy: GlmExecutionPolicy, initial: ExpertCacheStats) -> Result<Self> {
        policy.validate()?;
        let initial = GlmExpertCacheSnapshot::from_stats(initial)?;
        ensure!(
            (policy.expert_cache.minimum_bound_bytes..=policy.expert_cache.maximum_bound_bytes)
                .contains(&initial.bound_bytes),
            "initial GLM cache bound {} is outside the execution-policy range {}..={}",
            initial.bound_bytes,
            policy.expert_cache.minimum_bound_bytes,
            policy.expert_cache.maximum_bound_bytes
        );
        Ok(Self {
            policy,
            maximum_realized: initial.realized_bytes,
            initial,
            readmissions: Vec::new(),
        })
    }

    pub(crate) fn record(&mut self, event: GlmCacheReadmissionEvent) -> Result<()> {
        ensure!(
            self.readmissions.len() < MAX_ROUTING_TRACE_TOKENS,
            "GLM cache readmission event limit exceeded"
        );
        self.maximum_realized = self.maximum_realized.max(event.realized_after_bytes);
        self.readmissions.push(event);
        Ok(())
    }

    pub(crate) fn finish(
        self,
        prompt_tokens: usize,
        generated_tokens: usize,
        routed_tokens: usize,
        final_stats: ExpertCacheStats,
    ) -> Result<GlmExecutionManifest> {
        let final_expert_cache = GlmExpertCacheSnapshot::from_stats(final_stats)?;
        let maximum_observed_safe_point_realized_bytes =
            self.maximum_realized.max(final_expert_cache.realized_bytes);
        let manifest = GlmExecutionManifest {
            schema_version: GLM_EXECUTION_MANIFEST_SCHEMA_VERSION,
            policy: self.policy,
            prompt_tokens: u32::try_from(prompt_tokens)
                .context("GLM manifest prompt-token count exceeds u32")?,
            generated_tokens: u32::try_from(generated_tokens)
                .context("GLM manifest generated-token count exceeds u32")?,
            routed_tokens: u32::try_from(routed_tokens)
                .context("GLM manifest routed-token count exceeds u32")?,
            initial_expert_cache: self.initial,
            final_expert_cache,
            maximum_observed_safe_point_realized_bytes,
            readmissions: self.readmissions,
        };
        manifest.validate()?;
        Ok(manifest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_policy::ExpertCacheOptions;
    use crate::{
        expert_cache::ExpertCacheReplacementPolicy, expert_cache_manager::ExpertCacheLayout,
    };
    use candle_core::Device;
    use ff_core::weights::{CachePolicy, WeightSource};

    fn policy(adaptive: bool) -> GlmExecutionPolicy {
        GlmExecutionPolicy::from_runtime(
            &Device::Cpu,
            WeightSource::Mmap,
            CachePolicy::new(1),
            false,
            8,
            ExpertCacheOptions {
                layout: ExpertCacheLayout::SharedPool,
                replacement: ExpertCacheReplacementPolicy::Lru,
                maximum_bound_bytes: 100,
                minimum_bound_bytes: if adaptive { 20 } else { 100 },
                adaptive,
            },
        )
        .unwrap()
    }

    #[test]
    fn adaptive_recorder_accepts_a_shrunk_initial_bound_from_a_reused_engine() {
        let policy = policy(true);
        let minimum = usize::try_from(policy.expert_cache.minimum_bound_bytes).unwrap();
        let maximum = usize::try_from(policy.expert_cache.maximum_bound_bytes).unwrap();
        let shrunk = (minimum + maximum) / 2;
        assert!(minimum < shrunk && shrunk < maximum);

        let recorder =
            GlmExecutionManifestRecorder::new(policy.clone(), stats(shrunk, 0, 0)).unwrap();
        assert_eq!(recorder.initial.bound_bytes, u64::try_from(shrunk).unwrap());

        for outside in [minimum - 1, maximum + 1] {
            assert!(
                GlmExecutionManifestRecorder::new(policy.clone(), stats(outside, 0, 0)).is_err(),
                "bound {outside} must stay outside the admitted range"
            );
        }
    }

    #[test]
    fn adaptive_manifest_validates_with_a_shrunk_initial_bound() {
        let policy = policy(true);
        let minimum = usize::try_from(policy.expert_cache.minimum_bound_bytes).unwrap();
        let maximum = usize::try_from(policy.expert_cache.maximum_bound_bytes).unwrap();
        let shrunk = (minimum + maximum) / 2;
        let mut recorder = GlmExecutionManifestRecorder::new(policy, stats(shrunk, 0, 0)).unwrap();
        recorder
            .record(GlmCacheReadmissionEvent {
                sequence: 0,
                after_routed_token: 1,
                phase: RoutingTracePhase::Prefill,
                available_memory_bytes: u64::try_from(shrunk).unwrap(),
                reclaimable_capacity_bytes: u64::try_from(shrunk).unwrap(),
                required_future_headroom_bytes: 0,
                admissible_cache_bound_bytes: u64::try_from(shrunk).unwrap(),
                previous_bound_bytes: u64::try_from(shrunk).unwrap(),
                new_bound_bytes: u64::try_from(shrunk).unwrap(),
                realized_before_bytes: 0,
                realized_after_bytes: 0,
                evictions: 0,
                decision: GlmCacheAdmissionDecision::Unchanged,
            })
            .unwrap();
        let manifest = recorder.finish(1, 1, 1, stats(shrunk, 0, 0)).unwrap();
        assert_eq!(
            manifest.initial_expert_cache.bound_bytes,
            u64::try_from(shrunk).unwrap()
        );
        manifest.validate().unwrap();
    }

    #[test]
    fn non_adaptive_recorder_still_requires_the_single_admitted_bound() {
        let policy = policy(false);
        let maximum = usize::try_from(policy.expert_cache.maximum_bound_bytes).unwrap();
        GlmExecutionManifestRecorder::new(policy.clone(), stats(maximum, 0, 0)).unwrap();
        assert!(GlmExecutionManifestRecorder::new(policy, stats(maximum - 1, 0, 0)).is_err());
    }

    fn stats(bound: usize, realized: usize, evictions: u64) -> ExpertCacheStats {
        ExpertCacheStats {
            bytes: realized,
            entries: usize::from(realized > 0),
            max_bytes: bound,
            hits: 2,
            misses: 3,
            evictions,
        }
    }

    #[test]
    fn adaptive_manifest_binds_policy_and_bound_occupancy_transitions() {
        let policy = policy(true);
        let mut recorder = GlmExecutionManifestRecorder::new(policy, stats(100, 60, 0)).unwrap();
        recorder
            .record(GlmCacheReadmissionEvent {
                sequence: 0,
                after_routed_token: 3,
                phase: RoutingTracePhase::Prefill,
                available_memory_bytes: 30,
                reclaimable_capacity_bytes: 90,
                required_future_headroom_bytes: 40,
                admissible_cache_bound_bytes: 50,
                previous_bound_bytes: 100,
                new_bound_bytes: 50,
                realized_before_bytes: 60,
                realized_after_bytes: 50,
                evictions: 1,
                decision: GlmCacheAdmissionDecision::Shrunk,
            })
            .unwrap();
        recorder
            .record(GlmCacheReadmissionEvent {
                sequence: 1,
                after_routed_token: 4,
                phase: RoutingTracePhase::Decode,
                available_memory_bytes: 110,
                reclaimable_capacity_bytes: 160,
                required_future_headroom_bytes: 40,
                admissible_cache_bound_bytes: 120,
                previous_bound_bytes: 50,
                new_bound_bytes: 100,
                realized_before_bytes: 50,
                realized_after_bytes: 50,
                evictions: 0,
                decision: GlmCacheAdmissionDecision::Grown,
            })
            .unwrap();
        let manifest = recorder.finish(3, 2, 4, stats(100, 50, 1)).unwrap();
        assert_eq!(manifest.maximum_observed_safe_point_realized_bytes, 60);
        let json = manifest.canonical_json().unwrap();
        assert_eq!(GlmExecutionManifest::from_json(&json).unwrap(), manifest);
        let mut fake_token_boundary = manifest;
        fake_token_boundary.readmissions[0].after_routed_token = 1;
        assert!(fake_token_boundary.validate().is_err());
    }

    #[test]
    fn fixed_manifest_has_no_readmission_events() {
        let policy = policy(false);
        let manifest = GlmExecutionManifestRecorder::new(policy, stats(100, 0, 0))
            .unwrap()
            .finish(1, 1, 1, stats(100, 0, 0))
            .unwrap();
        assert!(manifest.readmissions.is_empty());
    }

    #[test]
    fn parser_refuses_a_legacy_policy_digest_and_rejects_oversize() {
        let policy = policy(false);
        let manifest = GlmExecutionManifestRecorder::new(policy, stats(100, 0, 0))
            .unwrap()
            .finish(1, 1, 1, stats(100, 0, 0))
            .unwrap();
        let mut value: serde_json::Value =
            serde_json::from_slice(&manifest.canonical_json().unwrap()).unwrap();
        // The digest key these manifests once carried is no longer tolerated:
        // the format migrated in one step rather than keeping a parallel path.
        value["policy_sha256"] = serde_json::json!("00".repeat(32));
        assert!(GlmExecutionManifest::from_json(&serde_json::to_vec(&value).unwrap()).is_err());
        assert!(
            GlmExecutionManifest::from_json(&vec![b' '; MAX_GLM_EXECUTION_MANIFEST_JSON_BYTES + 1])
                .is_err()
        );
    }
}
