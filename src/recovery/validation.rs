use super::{CheckpointIdentity, MAX_RECOVERY_JSON_BYTES, PolicyHistory};
use anyhow::{Context, Result};
use serde::{Serialize, de::DeserializeOwned};

pub(super) fn validate_schema(actual: u32, expected: u32, kind: &str) -> Result<()> {
    anyhow::ensure!(
        actual == expected,
        "unsupported {kind} schema {actual}; this build supports schema {expected}"
    );
    Ok(())
}

pub(super) trait RecoveryRecord {
    fn validate_record(&self) -> Result<()>;
}

impl RecoveryRecord for PolicyHistory {
    fn validate_record(&self) -> Result<()> {
        self.validate()
    }
}

impl RecoveryRecord for CheckpointIdentity {
    fn validate_record(&self) -> Result<()> {
        self.validate()
    }
}

pub(super) fn from_json<T>(bytes: &[u8], kind: &str) -> Result<T>
where
    T: DeserializeOwned + Serialize + RecoveryRecord,
{
    anyhow::ensure!(
        bytes.len() <= MAX_RECOVERY_JSON_BYTES,
        "{kind} JSON is {} bytes, exceeding the {MAX_RECOVERY_JSON_BYTES}-byte limit",
        bytes.len()
    );
    let record: T =
        serde_json::from_slice(bytes).with_context(|| format!("invalid {kind} JSON"))?;
    record.validate_record()?;
    let supplied: serde_json::Value =
        serde_json::from_slice(bytes).with_context(|| format!("invalid {kind} JSON"))?;
    let normalized =
        serde_json::to_value(&record).with_context(|| format!("failed to normalize {kind}"))?;
    anyhow::ensure!(
        supplied == normalized,
        "{kind} JSON contains unknown, omitted, or noncanonical fields"
    );
    Ok(record)
}

pub(super) fn canonical_json<T>(record: &T, kind: &str) -> Result<Vec<u8>>
where
    T: Serialize + RecoveryRecord,
{
    record.validate_record()?;
    let bytes =
        serde_json::to_vec(record).with_context(|| format!("failed to serialize {kind}"))?;
    anyhow::ensure!(
        bytes.len() <= MAX_RECOVERY_JSON_BYTES,
        "canonical {kind} JSON is {} bytes, exceeding the {MAX_RECOVERY_JSON_BYTES}-byte limit",
        bytes.len()
    );
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::super::*;

    #[test]
    fn policy_history_covers_every_evaluation_and_merges_successes() {
        let first = policy(32);
        let second = policy(16);
        let mut history = policy_history(5, first.clone());
        history
            .append_successful_evaluations(7, first.clone())
            .unwrap();
        history.append_successful_evaluations(9, second).unwrap();

        assert_eq!(history.completed_evaluations, 9);
        assert_eq!(history.segments.len(), 2);
        assert_eq!(history.segments[0].start_evaluation, 0);
        assert_eq!(history.segments[0].end_evaluation_exclusive, 7);
        assert_eq!(history.segments[1].start_evaluation, 7);
        assert_eq!(history.segments[1].end_evaluation_exclusive, 9);
        history.validate().unwrap();

        let canonical = history.canonical_json().unwrap();
        assert_eq!(PolicyHistory::from_json(&canonical).unwrap(), history);
    }

    #[test]
    fn policy_history_extension_cannot_rewrite_a_merged_prefix() {
        let first = policy(32);
        let prior = policy_history(2, first.clone());
        let mut extended = prior.clone();
        extended.append_successful_evaluations(4, first).unwrap();
        extended.validate_extends(&prior).unwrap();

        extended.segments[0].policy = policy(16);
        assert!(extended.validate_extends(&prior).is_err());
    }

    #[test]
    fn checkpoint_identity_binds_its_completed_boundary_and_history() {
        let history = policy_history(2, policy(32));
        let identity = checkpoint(history);
        let bytes = identity.canonical_json().unwrap();
        assert_eq!(CheckpointIdentity::from_json(&bytes).unwrap(), identity);

        let mut wrong_count = identity.clone();
        wrong_count.completed_evaluations += 1;
        assert!(wrong_count.validate().is_err());
        let mut oversized = identity.clone();
        oversized.checkpoint_bytes = super::super::MAX_RECOVERY_CHECKPOINT_BYTES + 1;
        assert!(oversized.validate().is_err());
    }
}
