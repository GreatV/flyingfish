//! Audit records for application-level resource selection, never executable
//! policy fields. Historical observations do not authorize live admission.
use crate::{
    probe::{RESOURCE_SNAPSHOT_SCHEMA_VERSION, ResourceSnapshot},
    weights::accounting::CacheInventory,
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const RESOURCE_SELECTION_SCHEMA_VERSION: u32 = 1;
pub const MAX_RESOURCE_SELECTION_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ResourcePolicyMode {
    Conservative,
    #[default]
    Performance,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionOrigin {
    Baseline,
    OperatorExplicit,
    CostModel,
    MeasuredEvidence,
    Pinned,
    Recorded,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectedResourceAxis {
    pub axis: String,
    pub value: String,
    pub origin: SelectionOrigin,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateDisposition {
    Selected,
    CapacityRejected,
    NoBenefitEvidence,
    EvidenceMismatch,
    KnownRegression,
    OperatorExcluded,
    Slower,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedRequestCost {
    pub initialization_us: u64,
    pub preparation_us: u64,
    pub repeated_work_us: u64,
    pub output_us: u64,
    pub uncertainty_us: u64,
}

impl ExpectedRequestCost {
    pub fn total_us(&self) -> Result<u64> {
        [
            self.initialization_us,
            self.preparation_us,
            self.repeated_work_us,
            self.output_us,
        ]
        .into_iter()
        .try_fold(0_u64, |sum, value| {
            sum.checked_add(value).context("request cost overflow")
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceCandidateObservation {
    pub candidate_id: String,
    pub disposition: CandidateDisposition,
    pub reason: String,
    #[serde(deserialize_with = "crate::required_option")]
    pub expected_cost: Option<ExpectedRequestCost>,
    /// What was measured for this candidate, recorded as the observations
    /// themselves rather than as identities standing in for them.
    pub evidence: Vec<serde_json::Value>,
}

/// Future additional allocations at the capture boundary. Already-present
/// process/context/catalog storage is described separately, not charged twice.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourcePhaseEstimate {
    pub phase: String,
    pub required_host_bytes: u64,
    pub optional_host_bytes: u64,
    pub host_promotion_reserve_bytes: u64,
    #[serde(deserialize_with = "crate::required_option")]
    pub required_device_bytes: Option<u64>,
    #[serde(deserialize_with = "crate::required_option")]
    pub optional_device_bytes: Option<u64>,
    pub device_reserve_bytes: u64,
}

impl ResourcePhaseEstimate {
    pub fn host_peak_bytes(&self) -> Result<u64> {
        self.required_host_bytes
            .checked_add(self.optional_host_bytes)
            .context("host phase peak overflow")
    }

    /// Required device bytes exclude this once-only reserve. CPU phases use
    /// the combined host ledger and leave both device byte fields absent.
    pub fn device_peak_bytes(&self) -> Result<Option<u64>> {
        match (self.required_device_bytes, self.optional_device_bytes) {
            (Some(required), Some(optional)) => Ok(Some(
                required
                    .checked_add(optional)
                    .and_then(|v| v.checked_add(self.device_reserve_bytes))
                    .context("device phase peak overflow")?,
            )),
            (None, None) => {
                ensure!(
                    self.device_reserve_bytes == 0,
                    "host-only phase has a device reserve"
                );
                Ok(None)
            }
            _ => anyhow::bail!("phase must specify both device fields or neither"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// What a selection was made against, recorded as the material itself.
///
/// These fields were digests of that material. A digest can say two runs
/// differ but never which field moved, and reading everything it covers is
/// work the run does not otherwise need. Recording the material lets a
/// mismatch name the field, and [`Self::first_difference`] is what names it.
pub struct ResourceSelectionProvenance {
    pub schema_version: u32,
    pub policy: serde_json::Value,
    pub selector_revision: String,
    pub request: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<serde_json::Value>,
    pub model: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hardware: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executable: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment: Option<serde_json::Value>,
    #[serde(default)]
    pub mode: ResourcePolicyMode,
    pub selection_snapshot: ResourceSnapshot,
    #[serde(deserialize_with = "crate::required_option")]
    pub final_admission_snapshot: Option<ResourceSnapshot>,
    pub already_present_at_capture: Vec<String>,
    pub inventory: CacheInventory,
    pub phases: Vec<ResourcePhaseEstimate>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub workload: std::collections::BTreeMap<String, u64>,
    pub axes: Vec<SelectedResourceAxis>,
    pub candidates: Vec<ResourceCandidateObservation>,
}

impl ResourceSelectionProvenance {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == RESOURCE_SELECTION_SCHEMA_VERSION,
            "unsupported resource selection schema"
        );
        ensure!(
            !self.policy.is_null(),
            "resource selection records no execution policy"
        );
        ensure!(
            !self.request.is_null(),
            "resource selection records no request"
        );
        ensure!(!self.model.is_null(), "resource selection records no model");
        bounded_text(&self.selector_revision, 256)?;
        self.inventory.validate()?;
        ensure!(self.workload.len() <= 64, "too many workload dimensions");
        for key in self.workload.keys() {
            bounded_text(key, 128)?;
        }
        ensure!(
            self.selection_snapshot.schema_version == RESOURCE_SNAPSHOT_SCHEMA_VERSION,
            "unsupported selection snapshot schema"
        );
        if let Some(snapshot) = &self.final_admission_snapshot {
            ensure!(
                snapshot.schema_version == RESOURCE_SNAPSHOT_SCHEMA_VERSION,
                "unsupported final admission snapshot schema"
            );
        }
        ensure!(
            self.already_present_at_capture.len() <= 64,
            "too many capture-boundary allocations"
        );
        for allocation in &self.already_present_at_capture {
            bounded_text(allocation, 256)?;
        }
        ensure!(
            !self.phases.is_empty() && self.phases.len() <= 64,
            "invalid phase estimate count"
        );
        let mut phases = BTreeSet::new();
        for phase in &self.phases {
            bounded_text(&phase.phase, 128)?;
            ensure!(phases.insert(&phase.phase), "duplicate resource phase");
            phase.host_peak_bytes()?;
            phase.device_peak_bytes()?;
        }
        ensure!(
            !self.axes.is_empty() && self.axes.len() <= 64,
            "invalid resource axis count"
        );
        let mut axes = BTreeSet::new();
        for axis in &self.axes {
            bounded_text(&axis.axis, 128)?;
            bounded_text(&axis.value, 1024)?;
            ensure!(axes.insert(&axis.axis), "duplicate resource axis");
        }
        ensure!(
            !self.candidates.is_empty() && self.candidates.len() <= 256,
            "invalid resource candidate count"
        );
        let mut ids = BTreeSet::new();
        let mut selected = 0;
        for candidate in &self.candidates {
            bounded_text(&candidate.candidate_id, 256)?;
            bounded_text(&candidate.reason, 4096)?;
            ensure!(
                ids.insert(&candidate.candidate_id),
                "duplicate resource candidate"
            );
            selected += usize::from(candidate.disposition == CandidateDisposition::Selected);
            if let Some(cost) = &candidate.expected_cost {
                cost.total_us()?;
            }
            ensure!(
                candidate.evidence.len() <= 64,
                "too many candidate evidence records"
            );
            for record in &candidate.evidence {
                ensure!(
                    !record.is_null(),
                    "a candidate evidence record is empty: {}",
                    candidate.candidate_id
                );
            }
        }
        // A recorded refusal (workload["refused"] == 1) selects nothing; every
        // other record must select exactly one candidate. Old records carry no
        // `refused` key and keep the strict rule.
        let refused = self.workload.get("refused").copied().unwrap_or(0) == 1;
        ensure!(
            selected == 1 || (refused && selected == 0),
            "resource selection must select exactly one candidate (or none on a recorded refusal)"
        );
        Ok(())
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)?;
        ensure!(
            bytes.len() <= MAX_RESOURCE_SELECTION_BYTES,
            "resource selection exceeds size limit"
        );
        Ok(bytes)
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() <= MAX_RESOURCE_SELECTION_BYTES,
            "resource selection exceeds size limit"
        );
        let value: Self =
            serde_json::from_slice(bytes).context("invalid resource selection JSON")?;
        value.validate()?;
        Ok(value)
    }

    /// Whether this selection was made for the given execution policy.
    ///
    /// The comparison is over the recorded policy itself, so a refusal can say
    /// what differs instead of only that something did.
    pub fn validate_policy_binding(&self, policy: &serde_json::Value) -> Result<()> {
        self.validate()?;
        ensure!(
            &self.policy == policy,
            "resource selection belongs to another execution policy"
        );
        Ok(())
    }

    /// The first recorded field that differs, for a caller reporting why a
    /// recorded selection does not describe the run in front of it.
    pub fn first_difference(&self, other: &Self) -> Option<&'static str> {
        for (name, differs) in [
            ("policy", self.policy != other.policy),
            ("request", self.request != other.request),
            ("input", self.input != other.input),
            ("model", self.model != other.model),
            ("hardware", self.hardware != other.hardware),
            ("executable", self.executable != other.executable),
            ("environment", self.environment != other.environment),
            (
                "selector_revision",
                self.selector_revision != other.selector_revision,
            ),
            ("mode", self.mode != other.mode),
        ] {
            if differs {
                return Some(name);
            }
        }
        None
    }
}

fn bounded_text(text: &str, limit: usize) -> Result<()> {
    ensure!(
        !text.trim().is_empty() && text.len() <= limit && !text.chars().any(char::is_control),
        "invalid resource selection text"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{probe::ResourceMeasurementScopes, weights::accounting::CacheShardInventory};
    use serde_json::json;

    fn record() -> ResourceSelectionProvenance {
        ResourceSelectionProvenance {
            schema_version: 1,
            policy: json!({"weight_source": "mmap", "expert_cache_bytes": 0}),
            selector_revision: "resource-policy-v1".into(),
            request: json!({"prompt_tokens": 10, "max_new_tokens": 4}),
            input: None,
            model: json!({"component": "transformer", "shards": 1}),
            hardware: None,
            executable: None,
            environment: None,
            mode: ResourcePolicyMode::Performance,
            selection_snapshot: ResourceSnapshot {
                schema_version: 1,
                measured_at_unix_ms: 1,
                host_memory_available_bytes: None,
                cgroup_v2_memory_limit: None,
                cgroup_v2_memory_current_bytes: None,
                cgroup_v2_memory_available_bytes: None,
                device_free_memory_bytes: None,
                host_device_memory_is_unified: None,
                measurement_scope: ResourceMeasurementScopes {
                    host_memory: None,
                    cgroup_memory: None,
                    device_memory: None,
                },
            },
            final_admission_snapshot: None,
            already_present_at_capture: vec!["device context".into()],
            inventory: CacheInventory {
                shards: vec![CacheShardInventory {
                    name: "a.safetensors".into(),
                    file_bytes: 108,
                    header_bytes: 8,
                    selected_tensor_bytes: 100,
                    selected_tensor_count: 1,
                    largest_tensor_bytes: 100,
                }],
            },
            phases: vec![ResourcePhaseEstimate {
                phase: "decode".into(),
                required_host_bytes: 1,
                optional_host_bytes: 0,
                host_promotion_reserve_bytes: 100,
                required_device_bytes: None,
                optional_device_bytes: None,
                device_reserve_bytes: 0,
            }],
            workload: Default::default(),
            axes: vec![SelectedResourceAxis {
                axis: "weights.source".into(),
                value: "mmap".into(),
                origin: SelectionOrigin::Baseline,
            }],
            candidates: vec![ResourceCandidateObservation {
                candidate_id: "baseline".into(),
                disposition: CandidateDisposition::Selected,
                reason: "baseline_no_benefit_evidence".into(),
                expected_cost: None,
                evidence: vec![],
            }],
        }
    }

    #[test]
    fn snapshot_changes_are_auditable_without_changing_policy_binding() {
        let mut record = record();
        let before = record.canonical_json().unwrap();
        let mut observation = record.selection_snapshot.clone();
        observation.measured_at_unix_ms = 2;
        observation.host_memory_available_bytes = Some(1);
        record.final_admission_snapshot = Some(observation);
        let after = record.canonical_json().unwrap();
        assert_ne!(before, after);
        record
            .validate_policy_binding(&json!({"weight_source": "mmap", "expert_cache_bytes": 0}))
            .unwrap();
        assert!(
            record
                .validate_policy_binding(&json!({"weight_source": "memory"}))
                .is_err()
        );
        assert_eq!(
            ResourceSelectionProvenance::from_json(&after).unwrap(),
            record
        );
    }

    #[test]
    fn contradictory_selections_and_unknown_fields_are_rejected() {
        let record = record();
        let mut value = serde_json::to_value(&record).unwrap();
        value["unknown"] = true.into();
        assert!(
            ResourceSelectionProvenance::from_json(&serde_json::to_vec(&value).unwrap()).is_err()
        );
        let mut duplicate = record.clone();
        duplicate.axes.push(duplicate.axes[0].clone());
        assert!(duplicate.validate().is_err());
        let mut missing = record.clone();
        missing.candidates[0].disposition = CandidateDisposition::NoBenefitEvidence;
        assert!(missing.validate().is_err());
        let mut both = record.clone();
        both.candidates.push(ResourceCandidateObservation {
            candidate_id: "second".into(),
            ..record.candidates[0].clone()
        });
        assert!(both.validate().is_err());
    }

    #[test]
    fn phase_ledger_counts_reserve_once_and_checks_overflow() {
        let mut phase = record().phases.remove(0);
        assert_eq!(phase.host_peak_bytes().unwrap(), 1);
        phase.required_device_bytes = Some(10);
        phase.optional_device_bytes = Some(20);
        phase.device_reserve_bytes = 3;
        assert_eq!(phase.device_peak_bytes().unwrap(), Some(33));
        phase.optional_device_bytes = Some(u64::MAX);
        assert!(phase.device_peak_bytes().is_err());
        phase.required_device_bytes = None;
        assert!(phase.device_peak_bytes().is_err());
    }
}
