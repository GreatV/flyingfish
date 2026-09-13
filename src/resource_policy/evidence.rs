//! Exact-scope measured evidence. No inference from memory occupancy or a
//! universal hit-rate threshold; actual paired command times decide eligibility.
use crate::{
    glm::GlmExecutionPolicy,
    h3::policy::ExecutionPolicy,
    runtime::{
        artifact::{FileStat, compare_artifact_files, read_artifact_snapshot},
        identity::WeakModelIdentity,
        probe::HardwareFingerprint,
    },
};
use anyhow::{Context, Result, ensure};
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const MAX_EVIDENCE_BYTES: u64 = 4 * 1024 * 1024;

/// A named auxiliary file beside a checkpoint, recorded by size.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuxiliaryFile {
    pub name: String,
    pub bytes: u64,
}

/// The checkpoint a measurement was taken against: the component's own weak
/// identity (config and index names, shard sizes and modification times) plus
/// the auxiliary files that change what a request means.
///
/// This keys a local measurement to a checkpoint. It is not a payload proof
/// and does not try to be.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelEvidence {
    pub identity: WeakModelIdentity,
    pub auxiliary_files: Vec<AuxiliaryFile>,
}

/// What the host was configured to do, as opposed to what it is.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentEvidence {
    pub model_root: String,
    pub model_root_file_key: String,
    #[serde(deserialize_with = "crate::required_option")]
    pub read_ahead: Option<ReadAheadEvidence>,
    pub threading: std::collections::BTreeMap<String, String>,
    pub cgroup_quotas: Vec<CgroupQuota>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadAheadEvidence {
    pub control_file: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CgroupQuota {
    pub path: String,
    #[serde(deserialize_with = "crate::required_option")]
    pub cpu_max: Option<String>,
    #[serde(deserialize_with = "crate::required_option")]
    pub memory_max: Option<String>,
}

/// Everything a measurement is only valid under, recorded as itself.
///
/// These were four digests. A digest can say a replayed context differs but
/// never which of the checkpoint, the request, the device or the environment
/// moved, and it costs a read of everything it covers. Recording the material
/// lets [`Self::first_difference`] name the field.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceContext {
    pub model: ModelEvidence,
    pub request: serde_json::Value,
    pub hardware: HardwareFingerprint,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executable_metadata: Option<ExecutableMetadata>,
    pub environment: EnvironmentEvidence,
    pub cache_state: CacheState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheState {
    Uncontrolled,
    Warm,
    Cold,
}

/// Local build binding without reading or hashing executable contents.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutableMetadata {
    pub file_key: String,
    pub bytes: u64,
    pub modified_unix_ns: u128,
}

impl ExecutableMetadata {
    fn collect(path: &Path) -> Result<Self> {
        let stat = crate::runtime::artifact::FileStat::of_target(path)?;
        ensure!(stat.is_file(), "executable path is not a file");
        Ok(Self {
            file_key: stat.recorded_key(),
            bytes: stat.len(),
            modified_unix_ns: stat
                .modified()?
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos(),
        })
    }
}

impl PartialEq for EvidenceContext {
    fn eq(&self, other: &Self) -> bool {
        self.model == other.model
            && self.request == other.request
            && self.hardware == other.hardware
            && self.environment == other.environment
            && self.cache_state == other.cache_state
            && self.executable_metadata == other.executable_metadata
    }
}
impl Eq for EvidenceContext {}

impl EvidenceContext {
    pub fn collect(model: &Path, device: &Device, request: serde_json::Value) -> Result<Self> {
        let identity = WeakModelIdentity::collect(model)?;
        let mut auxiliary_files = Vec::new();
        for name in [
            "tokenizer.json",
            "generation_config.json",
            "chat_template.jinja",
        ] {
            let path = model.join(name);
            if path.try_exists()? {
                let stat = FileStat::of_target(&path)?;
                ensure!(stat.is_file(), "auxiliary model file is not a regular file");
                auxiliary_files.push(AuxiliaryFile {
                    name: name.to_owned(),
                    bytes: stat.len(),
                });
            }
        }
        let hardware = HardwareFingerprint::collect(device);
        hardware.validate()?;
        let root = std::fs::canonicalize(model)?;
        let storage = FileStat::of_target(&root)?;
        let read_ahead =
            crate::runtime::storage::read_ahead_window(&root).map(|w| ReadAheadEvidence {
                control_file: w.control_file.display().to_string(),
                bytes: w.bytes,
            });
        let threading = [
            "RAYON_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "CUBLAS_WORKSPACE_CONFIG",
            "NVIDIA_TF32_OVERRIDE",
            "CUDA_VISIBLE_DEVICES",
            "CUDA_DEVICE_ORDER",
            "CUDA_LAUNCH_BLOCKING",
        ]
        .into_iter()
        .map(|key| (key.to_owned(), std::env::var(key).unwrap_or_default()))
        .collect::<std::collections::BTreeMap<_, _>>();
        let cgroup = std::fs::read_to_string("/proc/self/cgroup").ok();
        let mut quotas = Vec::new();
        if let Some(path) = cgroup
            .as_ref()
            .and_then(|c| c.lines().find_map(|line| line.strip_prefix("0::")))
        {
            let mount = std::path::PathBuf::from("/sys/fs/cgroup");
            let directory = mount.join(path.trim_start_matches('/'));
            for ancestor in directory.ancestors().take_while(|p| p.starts_with(&mount)) {
                quotas.push(CgroupQuota {
                    path: ancestor.display().to_string(),
                    cpu_max: std::fs::read_to_string(ancestor.join("cpu.max")).ok(),
                    memory_max: std::fs::read_to_string(ancestor.join("memory.max")).ok(),
                });
            }
        }
        Ok(Self {
            model: ModelEvidence {
                identity,
                auxiliary_files,
            },
            request,
            hardware,
            executable_metadata: Some(ExecutableMetadata::collect(&std::env::current_exe()?)?),
            cache_state: CacheState::Uncontrolled,
            environment: EnvironmentEvidence {
                model_root: root.display().to_string(),
                model_root_file_key: storage.recorded_key(),
                read_ahead,
                threading,
                cgroup_quotas: quotas,
            },
        })
    }

    pub fn validate(&self) -> Result<()> {
        self.model.identity.validate()?;
        self.hardware.validate()?;
        ensure!(
            !self.request.is_null(),
            "resource evidence records no request"
        );
        ensure!(
            !self.environment.model_root.is_empty()
                && !self.environment.model_root_file_key.is_empty(),
            "resource evidence records no model root"
        );
        if let Some(metadata) = &self.executable_metadata {
            ensure!(
                metadata.bytes > 0 && !metadata.file_key.is_empty(),
                "invalid executable metadata"
            );
        }
        Ok(())
    }

    /// The first recorded field that differs, so a refusal to reuse a
    /// measurement can say what about this machine or request moved.
    pub fn first_difference(&self, other: &Self) -> Option<&'static str> {
        for (name, differs) in [
            ("model", self.model != other.model),
            ("request", self.request != other.request),
            ("hardware", self.hardware != other.hardware),
            (
                "executable_metadata",
                self.executable_metadata != other.executable_metadata,
            ),
            ("environment", self.environment != other.environment),
            ("cache_state", self.cache_state != other.cache_state),
        ] {
            if differs {
                return Some(name);
            }
        }
        None
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "family",
    content = "policy",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum EvidencePolicy {
    H3(ExecutionPolicy),
    Glm(GlmExecutionPolicy),
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairedObservation {
    pub baseline_wall_us: u64,
    pub candidate_wall_us: u64,
    /// Artifacts retaining phase/counter/peak observations. Relative to the
    /// evidence file; read and size checked before automatic consumption.
    pub baseline_record: EvidenceArtifact,
    pub candidate_record: EvidenceArtifact,
    /// Whether the two retained outputs are the same bytes.
    ///
    /// Set by [`ResourceEvidence::load`], which compares the two output files
    /// directly. It is not serialized: a pair read from JSON has not had its
    /// outputs compared, and [`MeasuredCandidate::qualifies`] refuses it until
    /// it has.
    #[serde(skip)]
    pub(crate) outputs_match: bool,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceArtifact {
    pub file: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct VerifiedRoutingProfile {
    pub prompt_tokens: u32,
    pub num_hidden_layers: u32,
    pub num_experts: u32,
    pub experts_per_token: u32,
    pub layers: Vec<u32>,
    pub expert_bytes: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredCandidate {
    pub baseline_policy: EvidencePolicy,
    pub candidate: EvidencePolicy,
    pub minimum_improvement_basis_points: u32,
    pub pairs: Vec<PairedObservation>,
    #[serde(deserialize_with = "crate::required_option")]
    pub routing_trace: Option<EvidenceArtifact>,
    #[serde(deserialize_with = "crate::required_option")]
    pub routing_replay: Option<EvidenceArtifact>,
    #[serde(skip)]
    pub(crate) observed_peak_deltas: Option<(u64, Option<u64>)>,
    #[serde(skip)]
    pub(crate) routing_verified: bool,
    #[serde(skip)]
    pub(crate) routing_profile: Option<VerifiedRoutingProfile>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceEvidence {
    pub schema_version: u32,
    pub context: EvidenceContext,
    pub candidates: Vec<MeasuredCandidate>,
}

/// Retained online observation used to cross-check the pair summary. These
/// timings are measurements, not sums of possibly overlapping stage durations.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrialRecord {
    pub schema_version: u32,
    pub context: EvidenceContext,
    pub policy: EvidencePolicy,
    pub wall_us: u64,
    pub output_bytes: u64,
    pub output_file: String,
    pub phase_us: std::collections::BTreeMap<String, u64>,
    pub peak_process_rss_bytes: u64,
    pub baseline_process_rss_bytes: u64,
    pub peak_device_bytes: Option<u64>,
    pub baseline_device_bytes: Option<u64>,
    pub counters: std::collections::BTreeMap<String, u64>,
}

fn load_artifact(root: &Path, artifact: &EvidenceArtifact, limit: u64) -> Result<Vec<u8>> {
    let components = Path::new(&artifact.file).components().collect::<Vec<_>>();
    ensure!(
        !components.is_empty()
            && components
                .iter()
                .all(|c| matches!(c, std::path::Component::Normal(_))),
        "evidence artifact path must be relative without traversal"
    );
    let mut path = root.to_path_buf();
    for component in components {
        path.push(component);
        ensure!(
            !std::fs::symlink_metadata(&path)?.file_type().is_symlink(),
            "evidence artifact path contains a symlink"
        );
    }
    let snapshot = read_artifact_snapshot(path, limit)?;
    ensure!(
        snapshot.bytes.len() as u64 == artifact.bytes,
        "resource evidence artifact {} is {} bytes, and the record says {}",
        artifact.file,
        snapshot.bytes.len(),
        artifact.bytes
    );
    Ok(snapshot.bytes)
}

impl ResourceEvidence {
    pub fn load(path: &Path) -> Result<(Self, u64)> {
        let bytes = read_artifact_snapshot(path, MAX_EVIDENCE_BYTES)?;
        let mut record: Self =
            serde_json::from_slice(&bytes.bytes).context("invalid resource evidence JSON")?;
        record.validate()?;
        let root = path
            .parent()
            .context("resource evidence needs a parent directory")?;
        for candidate in &mut record.candidates {
            let baseline_policy = candidate.baseline_policy.clone();
            let candidate_policy = candidate.candidate.clone();
            let mut observed_peak_deltas = candidate.observed_peak_deltas;
            for pair in &mut candidate.pairs {
                // Each side's retained output, so the two can be compared
                // directly rather than through identities standing in for them.
                let mut outputs = Vec::new();
                for (artifact, policy, wall) in [
                    (
                        &pair.baseline_record,
                        &baseline_policy,
                        pair.baseline_wall_us,
                    ),
                    (
                        &pair.candidate_record,
                        &candidate_policy,
                        pair.candidate_wall_us,
                    ),
                ] {
                    let observation: TrialRecord = serde_json::from_slice(&load_artifact(
                        root,
                        artifact,
                        MAX_EVIDENCE_BYTES,
                    )?)?;
                    ensure!(
                        observation.schema_version == 1
                            && observation.context == record.context
                            && observation.policy == *policy
                            && observation.wall_us == wall,
                        "paired summary disagrees with the retained trial observation"
                    );
                    ensure!(
                        observation.peak_process_rss_bytes
                            >= observation.baseline_process_rss_bytes,
                        "trial host baseline exceeds its peak"
                    );
                    let device_delta = match (
                        observation.peak_device_bytes,
                        observation.baseline_device_bytes,
                    ) {
                        (Some(peak), Some(base)) => Some(
                            peak.checked_sub(base)
                                .context("trial device baseline exceeds its peak")?,
                        ),
                        (None, None) => None,
                        _ => anyhow::bail!("trial device peak and baseline must both be present"),
                    };
                    if *policy == candidate_policy {
                        let host_delta = observation.peak_process_rss_bytes
                            - observation.baseline_process_rss_bytes;
                        let previous = observed_peak_deltas.unwrap_or((0, None));
                        observed_peak_deltas = Some((
                            previous.0.max(host_delta),
                            match (previous.1, device_delta) {
                                (Some(a), Some(b)) => Some(a.max(b)),
                                (a, b) => a.or(b),
                            },
                        ));
                    }
                    let output_path = Path::new(&observation.output_file);
                    ensure!(
                        output_path
                            .components()
                            .all(|c| matches!(c, std::path::Component::Normal(_)))
                            && !observation.output_file.is_empty(),
                        "trial output path must be relative without traversal"
                    );
                    let mut checked_path = root.to_path_buf();
                    for component in output_path.components() {
                        checked_path.push(component);
                        ensure!(
                            !std::fs::symlink_metadata(&checked_path)?
                                .file_type()
                                .is_symlink(),
                            "trial output path contains a symlink"
                        );
                    }
                    let output = FileStat::of_path(&checked_path).with_context(|| {
                        format!(
                            "failed to inspect retained trial output {}",
                            observation.output_file
                        )
                    })?;
                    ensure!(
                        output.len() == observation.output_bytes,
                        "retained trial output is {} bytes, and the record says {}",
                        output.len(),
                        observation.output_bytes
                    );
                    outputs.push((checked_path, observation.output_bytes));
                    ensure!(
                        observation.peak_process_rss_bytes > 0
                            && !observation.phase_us.is_empty()
                            && !observation.counters.is_empty(),
                        "trial lacks phase, memory or counter observations"
                    );
                }
                // A policy that is faster because it computed something else is
                // not faster. The two retained outputs decide that directly.
                pair.outputs_match = compare_artifact_files(
                    &outputs[0].0,
                    outputs[0].1,
                    &outputs[1].0,
                    outputs[1].1,
                )?;
            }
            candidate.observed_peak_deltas = observed_peak_deltas.take();
            match (&candidate.routing_trace, &candidate.routing_replay) {
                (None, None) => {}
                (Some(trace_artifact), Some(replay_artifact)) => {
                    use crate::glm::routing_trace::{
                        MAX_ROUTING_REPLAY_JSON_BYTES, MAX_ROUTING_TRACE_JSON_BYTES,
                        RoutingReplayOptions, RoutingReplayReport, RoutingTrace,
                    };
                    let trace = RoutingTrace::from_json(&load_artifact(
                        root,
                        trace_artifact,
                        MAX_ROUTING_TRACE_JSON_BYTES as u64,
                    )?)?;
                    let report = RoutingReplayReport::from_json(&load_artifact(
                        root,
                        replay_artifact,
                        MAX_ROUTING_REPLAY_JSON_BYTES as u64,
                    )?)?;
                    ensure!(
                        report.trace_bytes == trace_artifact.bytes,
                        "routing replay names a trace of another size"
                    );
                    let options = RoutingReplayOptions::new(
                        report
                            .segment_lengths
                            .iter()
                            .map(|n| usize::try_from(*n))
                            .collect::<std::result::Result<Vec<_>, _>>()?,
                        report.cache_budgets_bytes.clone(),
                    )?;
                    let recomputed =
                        RoutingReplayReport::analyze(&trace, trace_artifact.bytes, &options)?;
                    ensure!(
                        report == recomputed,
                        "routing replay does not reproduce from the retained trace"
                    );
                    if let EvidencePolicy::Glm(policy) = &candidate.candidate {
                        use crate::glm::{
                            execution_policy::GlmExecutionBackend,
                            routing_trace::{ExpertCacheEntryDtype, RoutingPrefillSchedule},
                        };
                        let dtype = if policy.backend == GlmExecutionBackend::Cpu {
                            ExpertCacheEntryDtype::Float32
                        } else {
                            ExpertCacheEntryDtype::Bfloat16
                        };
                        ensure!(
                            trace.cache_entry_dtype == dtype
                                && trace.prefill_schedule
                                    == RoutingPrefillSchedule::LayerBatchedExpertGrouped,
                            "routing evidence dtype or access schedule differs from execution"
                        );
                        let layout = serde_json::to_value(policy.expert_cache.layout)?;
                        let replacement = serde_json::to_value(policy.expert_cache.replacement)?;
                        let row = report
                            .cache_replays
                            .iter()
                            .find(|row| {
                                row.requested_budget_bytes
                                    == policy.expert_cache.maximum_bound_bytes
                                    && serde_json::to_value(row.cache_layout).ok().as_ref()
                                        == Some(&layout)
                                    && serde_json::to_value(row.cache_policy).ok().as_ref()
                                        == Some(&replacement)
                            })
                            .context(
                                "routing replay has no matching candidate unit/layout/budget",
                            )?;
                        candidate.routing_verified = row.hits > 0;
                        candidate.routing_profile = Some(VerifiedRoutingProfile {
                            prompt_tokens: trace.prompt_tokens,
                            num_hidden_layers: trace.num_hidden_layers,
                            num_experts: trace.num_experts,
                            experts_per_token: trace.experts_per_token,
                            layers: trace.layers.iter().map(|l| l.layer_index).collect(),
                            expert_bytes: trace
                                .layers
                                .iter()
                                .flat_map(|l| l.expert_bytes.iter().copied())
                                .collect(),
                        });
                    } else {
                        anyhow::bail!("routing evidence belongs to a GLM candidate");
                    }
                }
                _ => anyhow::bail!("routing trace and replay must be supplied together"),
            }
        }
        let evidence_bytes = bytes.bytes.len() as u64;
        Ok((record, evidence_bytes))
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == 1,
            "unsupported resource evidence schema"
        );
        self.context.validate()?;
        ensure!(
            !self.candidates.is_empty() && self.candidates.len() <= 64,
            "invalid measured candidate count"
        );
        let mut ids = std::collections::BTreeSet::new();
        for candidate in &self.candidates {
            ensure!(
                ids.insert(serde_json::to_vec(&candidate.candidate)?),
                "duplicate measured candidate policy"
            );
            ensure!(
                candidate.baseline_policy != candidate.candidate,
                "a candidate measured against itself shows nothing"
            );
            ensure!(
                candidate.minimum_improvement_basis_points > 0
                    && candidate.minimum_improvement_basis_points < 10000,
                "invalid predeclared improvement margin"
            );
            ensure!(
                !candidate.pairs.is_empty() && candidate.pairs.len() <= 64,
                "invalid paired observation count"
            );
            let mut baseline_records = std::collections::BTreeSet::new();
            let mut candidate_records = std::collections::BTreeSet::new();
            for pair in &candidate.pairs {
                ensure!(
                    baseline_records.insert(&pair.baseline_record.file)
                        && candidate_records.insert(&pair.candidate_record.file),
                    "paired trials must be independent observations"
                );
                ensure!(
                    pair.baseline_wall_us > 0 && pair.candidate_wall_us > 0,
                    "zero measured command duration"
                );
                ensure!(
                    pair.baseline_record.bytes > 0 && pair.candidate_record.bytes > 0,
                    "a retained trial record is empty"
                );
            }
        }
        Ok(())
    }
}

impl MeasuredCandidate {
    /// All retained pairs must clear the predeclared margin. A known
    /// regression or unequal output vetoes the whole row, not just its median.
    pub fn qualifies(&self) -> bool {
        self.pairs.len() >= 3
            && self.pairs.iter().all(|p| {
                p.outputs_match
                    && u128::from(p.candidate_wall_us) * 10000
                        < u128::from(p.baseline_wall_us)
                            * (10000 - u128::from(self.minimum_improvement_basis_points))
            })
    }
    pub fn median_us(&self) -> u64 {
        let mut values = self
            .pairs
            .iter()
            .map(|p| p.candidate_wall_us)
            .collect::<Vec<_>>();
        values.sort_unstable();
        values[values.len() / 2]
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn executable_matching_uses_stat_metadata_rather_than_content() -> Result<()> {
        let (_directory, evidence) = fixture();
        let mut context = evidence.context.clone();
        context.executable_metadata = Some(ExecutableMetadata::collect(&std::env::current_exe()?)?);
        context.validate()?;
        assert_eq!(
            serde_json::from_value::<EvidenceContext>(serde_json::to_value(&context)?)?,
            context
        );
        let mut changed = context.clone();
        changed.executable_metadata.as_mut().unwrap().bytes += 1;
        assert_ne!(context, changed);
        changed.executable_metadata = None;
        assert_ne!(
            context, changed,
            "old code-hashed evidence must not match a fresh metadata-bound build"
        );
        Ok(())
    }
    use crate::{
        glm::{ExpertCacheLayout, ExpertCacheReplacementPolicy},
        runtime::weights::{CachePolicy, WeightSource},
    };
    use std::{collections::BTreeMap, fs};

    /// A context standing for one machine, one checkpoint and one request.
    /// Tests vary one field at a time to say which difference they mean.
    pub(crate) fn sample_context() -> EvidenceContext {
        let model_root = std::env::temp_dir()
            .join("flyingfish-evidence-test-model")
            .to_string_lossy()
            .into_owned();
        EvidenceContext {
            model: ModelEvidence {
                identity: WeakModelIdentity {
                    schema_version: crate::runtime::identity::CALIBRATION_IDENTITY_SCHEMA_VERSION,
                    strength:
                        crate::runtime::identity::ModelIdentityStrength::LocalMetadataManifest,
                    cacheable: false,
                    canonical_component_path: model_root.clone(),
                    config: crate::runtime::identity::NamedFileStamp {
                        relative_path: "config.json".to_owned(),
                        bytes: 2,
                        modified_ns: 1,
                    },
                    index: None,
                    indexed_checkpoint_bytes: 1024,
                    shards: vec![crate::runtime::identity::WeakShardIdentity {
                        relative_path: "model.safetensors".to_owned(),
                        file_bytes: 1024,
                        modified_ns: 7,
                    }],
                },
                auxiliary_files: vec![AuxiliaryFile {
                    name: "tokenizer.json".into(),
                    bytes: 128,
                }],
            },
            request: serde_json::json!({"prompt": "a", "max_new_tokens": 4}),
            hardware: HardwareFingerprint::collect(&Device::Cpu),
            executable_metadata: None,
            environment: EnvironmentEvidence {
                model_root,
                model_root_file_key: "1:2".into(),
                read_ahead: None,
                threading: std::collections::BTreeMap::new(),
                cgroup_quotas: vec![],
            },
            cache_state: CacheState::Uncontrolled,
        }
    }

    fn fixture() -> (tempfile::TempDir, ResourceEvidence) {
        let root = tempfile::tempdir().unwrap();
        let context = sample_context();
        let base = GlmExecutionPolicy::from_runtime(
            &Device::Cpu,
            WeightSource::Mmap,
            CachePolicy::new(1),
            false,
            32,
            ExpertCacheLayout::PerLayerSplit,
            ExpertCacheReplacementPolicy::Lru,
            0,
            0,
            false,
        )
        .unwrap();
        let mut policy = base.clone();
        policy.resident_static = true;
        fs::write(root.path().join("baseline-output.bin"), b"same tokens").unwrap();
        fs::write(root.path().join("candidate-output.bin"), b"same tokens").unwrap();
        let pairs = (0..3)
            .map(|n| {
                let write = |candidate: bool| {
                    let trial_policy = if candidate {
                        EvidencePolicy::Glm(policy.clone())
                    } else {
                        EvidencePolicy::Glm(base.clone())
                    };
                    let row = TrialRecord {
                        schema_version: 1,
                        context: context.clone(),
                        policy: trial_policy,
                        wall_us: if candidate { 500 } else { 1000 },
                        output_bytes: b"same tokens".len() as u64,
                        output_file: if candidate {
                            "candidate-output.bin".into()
                        } else {
                            "baseline-output.bin".into()
                        },
                        phase_us: BTreeMap::from([("preparation".into(), n + 1)]),
                        peak_process_rss_bytes: 100,
                        baseline_process_rss_bytes: 10,
                        peak_device_bytes: None,
                        baseline_device_bytes: None,
                        counters: BTreeMap::from([("materializations".into(), 10)]),
                    };
                    let bytes = serde_json::to_vec(&row).unwrap();
                    let file = format!("{candidate}-{n}.json");
                    fs::write(root.path().join(&file), &bytes).unwrap();
                    EvidenceArtifact {
                        file,
                        bytes: bytes.len() as u64,
                    }
                };
                PairedObservation {
                    baseline_wall_us: 1000,
                    candidate_wall_us: 500,
                    baseline_record: write(false),
                    candidate_record: write(true),
                    outputs_match: false,
                }
            })
            .collect();
        (
            root,
            ResourceEvidence {
                schema_version: 1,
                context,
                candidates: vec![MeasuredCandidate {
                    baseline_policy: EvidencePolicy::Glm(base.clone()),
                    candidate: EvidencePolicy::Glm(policy),
                    minimum_improvement_basis_points: 200,
                    pairs,
                    routing_trace: None,
                    routing_replay: None,
                    observed_peak_deltas: None,
                    routing_verified: false,
                    routing_profile: None,
                }],
            },
        )
    }
    fn publish(root: &Path, evidence: &ResourceEvidence) -> std::path::PathBuf {
        let path = root.join("evidence.json");
        fs::write(&path, serde_json::to_vec(evidence).unwrap()).unwrap();
        path
    }
    #[test]
    fn measured_evidence_verifies_trial_records_and_actual_output_bytes() {
        let (root, evidence) = fixture();
        let path = publish(root.path(), &evidence);
        let (loaded, _) = ResourceEvidence::load(&path).unwrap();
        assert!(loaded.candidates[0].qualifies());
        fs::write(root.path().join("candidate-output.bin"), b"changed tokens").unwrap();
        assert!(
            ResourceEvidence::load(&path)
                .unwrap_err()
                .to_string()
                .contains("retained trial output is 14 bytes, and the record says 11")
        );

        // Same length, different tokens. Size alone cannot separate these, so
        // the outputs are compared byte for byte and the row is vetoed rather
        // than credited with a speed-up for computing something else.
        let (root, evidence) = fixture();
        fs::write(root.path().join("candidate-output.bin"), b"othr tokens").unwrap();
        let (loaded, _) = ResourceEvidence::load(&publish(root.path(), &evidence)).unwrap();
        assert!(!loaded.candidates[0].pairs[0].outputs_match);
        assert!(!loaded.candidates[0].qualifies());
    }

    #[test]
    fn evidence_compares_large_outputs_but_keeps_json_bounded() {
        use std::io::{Seek, SeekFrom, Write};
        let (root, mut evidence) = fixture();
        let size = MAX_EVIDENCE_BYTES + 123;
        for name in ["baseline-output.bin", "candidate-output.bin"] {
            fs::File::create(root.path().join(name))
                .unwrap()
                .set_len(size)
                .unwrap();
        }
        for pair in &mut evidence.candidates[0].pairs {
            for artifact in [&mut pair.baseline_record, &mut pair.candidate_record] {
                let path = root.path().join(&artifact.file);
                let mut trial: TrialRecord =
                    serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
                trial.output_bytes = size;
                let bytes = serde_json::to_vec(&trial).unwrap();
                fs::write(path, &bytes).unwrap();
                artifact.bytes = bytes.len() as u64;
            }
        }
        let path = publish(root.path(), &evidence);
        assert!(ResourceEvidence::load(&path).unwrap().0.candidates[0].qualifies());
        let mut output = fs::OpenOptions::new()
            .write(true)
            .open(root.path().join("candidate-output.bin"))
            .unwrap();
        output.seek(SeekFrom::Start(size - 1)).unwrap();
        output.write_all(&[1]).unwrap();
        assert!(!ResourceEvidence::load(&path).unwrap().0.candidates[0].qualifies());

        fs::File::create(&path)
            .unwrap()
            .set_len(MAX_EVIDENCE_BYTES + 1)
            .unwrap();
        assert!(
            ResourceEvidence::load(&path)
                .unwrap_err()
                .to_string()
                .contains("exceeding")
        );
    }
    #[test]
    fn summaries_cannot_replace_measurements_or_repeat_one_pair() {
        let (root, mut evidence) = fixture();
        evidence.candidates[0].pairs[0].candidate_wall_us = 1;
        let path = publish(root.path(), &evidence);
        assert!(
            ResourceEvidence::load(&path)
                .unwrap_err()
                .to_string()
                .contains("summary disagrees")
        );
        let (_, mut evidence) = fixture();
        evidence.candidates[0].pairs[1] = evidence.candidates[0].pairs[0].clone();
        assert!(
            evidence
                .validate()
                .unwrap_err()
                .to_string()
                .contains("independent")
        );
    }
    #[test]
    fn traversal_and_tampered_observation_artifacts_are_rejected() {
        let (root, mut evidence) = fixture();
        evidence.candidates[0].pairs[0].baseline_record.file = "../elsewhere".into();
        assert!(ResourceEvidence::load(&publish(root.path(), &evidence)).is_err());
        let (root, evidence) = fixture();
        fs::write(
            root.path()
                .join(&evidence.candidates[0].pairs[0].baseline_record.file),
            b"{}",
        )
        .unwrap();
        assert!(
            ResourceEvidence::load(&publish(root.path(), &evidence))
                .unwrap_err()
                .to_string()
                .contains("is 2 bytes, and the record says")
        );
    }
}
