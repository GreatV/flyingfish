use crate::policy::{
    H3_QWEN_NUMERICAL_CONTRACT_JSON_TENSOR, H3_QWEN_NUMERICAL_CONTRACT_SCHEMA_TENSOR,
    H3QwenNumericalContract,
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use ff_core::artifact::{ArtifactSnapshot, FileStat, read_artifact_snapshot};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::Path};

pub const EXTERNAL_PROMPT_PROVENANCE_SCHEMA_VERSION: u32 = 1;
pub const EXTERNAL_PROMPT_PROVENANCE_JSON_TENSOR: &str = "ff_external_prompt_provenance_json";
pub const EXTERNAL_PROMPT_PROVENANCE_SCHEMA_TENSOR: &str = "ff_external_prompt_provenance_schema";
const MAX_EXTERNAL_PROMPT_MANIFEST_BYTES: u64 = 1024 * 1024;
const MAX_EXTERNAL_PROMPT_PROVENANCE_BYTES: usize = 16 * 1024;
const OFFICIAL_H3_FIXTURE_PRODUCER: &str = "official MiniMax-H3 Modular Diffusers pipeline";
const OFFICIAL_DIFFUSERS_COMMIT: &str = "f37ab93e621d5ce206c9662e8291ca8b67d9c555";
const OFFICIAL_TRANSFORMERS_COMMIT: &str = "838763bf4372a5d0e5643fbd76f88294fb66277f";
const OFFICIAL_PYTORCH_COMMIT: &str = "7269437d655783a26cba32aa88195b741ff496aa";
const OFFICIAL_MODEL_REVISION: &str = "57559a6720ddaa2eaee52b09ec2ef011127997b2";
const OFFICIAL_A1_MANIFEST_BYTES: u64 = 29_478;
const OFFICIAL_A1_INPUT_BYTES: u64 = 3_726_732;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalPromptProvenance {
    pub schema_version: u32,
    pub binding: String,
    pub manifest_bytes: u64,
    pub input_bytes: u64,
    pub manifest_producer: String,
    pub manifest_schema_version: u32,
}

impl ExternalPromptProvenance {
    fn pinned_manifest_snapshot(manifest: &Path) -> Result<ArtifactSnapshot> {
        let snapshot = read_artifact_snapshot(manifest, MAX_EXTERNAL_PROMPT_MANIFEST_BYTES)
            .with_context(|| {
                format!(
                    "failed to read external prompt manifest {}",
                    manifest.display()
                )
            })?;
        anyhow::ensure!(
            snapshot_bytes(&snapshot) == OFFICIAL_A1_MANIFEST_BYTES,
            "external prompt manifest is not the pinned official A1 fixture: got {} bytes, expected {OFFICIAL_A1_MANIFEST_BYTES}",
            snapshot_bytes(&snapshot)
        );
        Ok(snapshot)
    }

    pub fn verify_official_fixture_manifest(manifest: &Path) -> Result<()> {
        Self::pinned_manifest_snapshot(manifest).map(|_| ())
    }

    pub fn from_official_fixture_manifest(manifest: &Path, input: &Path) -> Result<Self> {
        let manifest_snapshot = Self::pinned_manifest_snapshot(manifest)?;
        let input_bytes = file_bytes(input).with_context(|| {
            format!(
                "failed to identify external prompt input {}",
                input.display()
            )
        })?;
        anyhow::ensure!(
            input_bytes == OFFICIAL_A1_INPUT_BYTES,
            "external prompt input is not the pinned official A1 input: got {} bytes, expected {OFFICIAL_A1_INPUT_BYTES}",
            input_bytes
        );
        let document: serde_json::Value = serde_json::from_slice(&manifest_snapshot.bytes)
            .context("external prompt manifest is not valid JSON")?;
        let schema = document
            .get("schema_version")
            .and_then(serde_json::Value::as_u64)
            .context("external prompt manifest has no integer schema_version")?;
        anyhow::ensure!(
            schema == 2,
            "external prompt manifest must use official fixture schema 2"
        );
        let producer = document
            .get("producer")
            .and_then(serde_json::Value::as_str)
            .context("external prompt manifest has no producer")?;
        anyhow::ensure!(
            producer == OFFICIAL_H3_FIXTURE_PRODUCER,
            "external prompt manifest producer is not the pinned official H3 exporter"
        );
        anyhow::ensure!(
            document
                .pointer("/tensors/input.safetensors")
                .is_some_and(serde_json::Value::is_object),
            "external prompt manifest does not describe input.safetensors"
        );
        for (pointer, expected) in [
            ("/request/qwen_attention_backend", "eager"),
            ("/runtime/qwen_attention_backend", "eager"),
            ("/runtime/attention_backend", "_native_math"),
            ("/diffusers/required_commit", OFFICIAL_DIFFUSERS_COMMIT),
            ("/diffusers/observed_git_commit", OFFICIAL_DIFFUSERS_COMMIT),
            (
                "/transformers/required_commit",
                OFFICIAL_TRANSFORMERS_COMMIT,
            ),
            (
                "/transformers/observed_git_commit",
                OFFICIAL_TRANSFORMERS_COMMIT,
            ),
            ("/pytorch/required_commit", OFFICIAL_PYTORCH_COMMIT),
            ("/pytorch/observed_git_commit", OFFICIAL_PYTORCH_COMMIT),
            ("/model/revision", OFFICIAL_MODEL_REVISION),
        ] {
            anyhow::ensure!(
                document
                    .pointer(pointer)
                    .and_then(serde_json::Value::as_str)
                    == Some(expected),
                "external prompt manifest {pointer} is not {expected:?}"
            );
        }
        anyhow::ensure!(
            document
                .pointer("/runtime/qwen_use_kernels")
                .and_then(serde_json::Value::as_bool)
                == Some(false),
            "external prompt manifest must disable Qwen remote kernels"
        );
        anyhow::ensure!(
            document
                .pointer("/model/filesystem_read_only")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
                && document
                    .pointer("/producer_files/filesystem_read_only")
                    .and_then(serde_json::Value::as_bool)
                    == Some(true),
            "external prompt manifest was not produced from read-only model/source mounts"
        );
        anyhow::ensure!(
            document.pointer("/model/strong_shards").is_some()
                && document.pointer("/pytorch/source_files").is_some()
                && document.pointer("/transformers/source_files").is_some()
                && document.pointer("/diffusers/source_files").is_some(),
            "external prompt manifest lacks strong model/source identity"
        );
        let provenance = Self {
            schema_version: EXTERNAL_PROMPT_PROVENANCE_SCHEMA_VERSION,
            binding: "externally_validated_unbound_to_flyingfish_qwen".to_owned(),
            manifest_bytes: snapshot_bytes(&manifest_snapshot),
            input_bytes,
            manifest_producer: producer.to_owned(),
            manifest_schema_version: u32::try_from(schema)
                .context("external prompt manifest schema exceeds u32")?,
        };
        provenance.validate()?;
        Ok(provenance)
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == EXTERNAL_PROMPT_PROVENANCE_SCHEMA_VERSION,
            "unsupported external prompt provenance schema {}",
            self.schema_version
        );
        anyhow::ensure!(
            self.binding == "externally_validated_unbound_to_flyingfish_qwen"
                && self.manifest_producer == OFFICIAL_H3_FIXTURE_PRODUCER
                && self.manifest_schema_version == 2,
            "external prompt provenance is not the pinned official/unbound contract"
        );
        anyhow::ensure!(
            self.manifest_bytes == OFFICIAL_A1_MANIFEST_BYTES
                && self.input_bytes == OFFICIAL_A1_INPUT_BYTES,
            "external prompt provenance is not bound to the pinned official A1 manifest/input"
        );
        for (name, bytes) in [
            ("manifest", self.manifest_bytes),
            ("input", self.input_bytes),
        ] {
            anyhow::ensure!(
                bytes > 0,
                "external prompt {name} byte size must be non-zero"
            );
        }
        Ok(())
    }

    pub fn verify_manifest(&self, manifest: &Path) -> Result<()> {
        self.validate()?;
        let snapshot = Self::pinned_manifest_snapshot(manifest)?;
        anyhow::ensure!(
            snapshot_bytes(&snapshot) == self.manifest_bytes,
            "external prompt manifest identity differs from the recorded provenance"
        );
        Ok(())
    }

    pub fn verify_initial_input(&self, input: &Path) -> Result<()> {
        self.validate()?;
        let bytes = file_bytes(input).with_context(|| {
            format!("failed to verify external prompt input {}", input.display())
        })?;
        anyhow::ensure!(
            bytes == self.input_bytes,
            "external prompt input identity changed while loading"
        );
        Ok(())
    }

    fn canonical_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).context("serialize external prompt provenance")?;
        anyhow::ensure!(
            (1..=MAX_EXTERNAL_PROMPT_PROVENANCE_BYTES).contains(&bytes.len()),
            "external prompt provenance JSON is outside its bound"
        );
        Ok(bytes)
    }

    fn insert_artifact_tensors(
        &self,
        tensors: &mut HashMap<&'static str, Tensor>,
        device: &Device,
    ) -> Result<()> {
        for name in [
            EXTERNAL_PROMPT_PROVENANCE_JSON_TENSOR,
            EXTERNAL_PROMPT_PROVENANCE_SCHEMA_TENSOR,
        ] {
            anyhow::ensure!(
                !tensors.contains_key(name),
                "artifact already contains external prompt provenance tensor {name}"
            );
        }
        let json = self.canonical_json()?;
        tensors.insert(
            EXTERNAL_PROMPT_PROVENANCE_JSON_TENSOR,
            Tensor::from_vec(json.clone(), json.len(), device)?,
        );
        tensors.insert(
            EXTERNAL_PROMPT_PROVENANCE_SCHEMA_TENSOR,
            Tensor::new(EXTERNAL_PROMPT_PROVENANCE_SCHEMA_VERSION, device)?,
        );
        Ok(())
    }

    fn take_artifact_tensors(tensors: &mut HashMap<String, Tensor>) -> Result<Option<Self>> {
        let names = [
            EXTERNAL_PROMPT_PROVENANCE_JSON_TENSOR,
            EXTERNAL_PROMPT_PROVENANCE_SCHEMA_TENSOR,
        ];
        let present = names.map(|name| tensors.contains_key(name));
        if present.iter().all(|present| !present) {
            return Ok(None);
        }
        anyhow::ensure!(
            present.iter().all(|present| *present),
            "artifact has incomplete external prompt provenance"
        );
        let json = tensors
            .remove(EXTERNAL_PROMPT_PROVENANCE_JSON_TENSOR)
            .context("missing external prompt provenance JSON")?;
        anyhow::ensure!(
            json.dtype() == DType::U8
                && json.rank() == 1
                && (1..=MAX_EXTERNAL_PROMPT_PROVENANCE_BYTES).contains(&json.elem_count()),
            "external prompt provenance JSON tensor is invalid"
        );
        let json = json.to_vec1::<u8>()?;
        let schema = tensors
            .remove(EXTERNAL_PROMPT_PROVENANCE_SCHEMA_TENSOR)
            .context("missing external prompt provenance schema")?;
        anyhow::ensure!(schema.dtype() == DType::U32 && schema.rank() == 0);
        anyhow::ensure!(
            schema.to_scalar::<u32>()? == EXTERNAL_PROMPT_PROVENANCE_SCHEMA_VERSION,
            "external prompt provenance schema tensor is unsupported"
        );
        let provenance: Self =
            serde_json::from_slice(&json).context("invalid external prompt provenance JSON")?;
        provenance.validate()?;
        anyhow::ensure!(
            provenance.canonical_json()? == json,
            "external prompt provenance JSON is not canonical"
        );
        Ok(Some(provenance))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum H3ConditioningProvenance {
    FlyingfishQwen(Box<H3QwenNumericalContract>),
    ExternallyValidated(ExternalPromptProvenance),
}

impl H3ConditioningProvenance {
    pub fn insert_artifact_tensors(
        &self,
        tensors: &mut HashMap<&'static str, Tensor>,
        device: &Device,
    ) -> Result<()> {
        match self {
            Self::FlyingfishQwen(contract) => contract.insert_artifact_tensors(tensors, device),
            Self::ExternallyValidated(provenance) => {
                provenance.insert_artifact_tensors(tensors, device)
            }
        }
    }

    pub fn take_artifact_tensors(tensors: &mut HashMap<String, Tensor>) -> Result<Option<Self>> {
        let has_qwen = [
            H3_QWEN_NUMERICAL_CONTRACT_JSON_TENSOR,
            H3_QWEN_NUMERICAL_CONTRACT_SCHEMA_TENSOR,
        ]
        .iter()
        .any(|name| tensors.contains_key(*name));
        let has_external = [
            EXTERNAL_PROMPT_PROVENANCE_JSON_TENSOR,
            EXTERNAL_PROMPT_PROVENANCE_SCHEMA_TENSOR,
        ]
        .iter()
        .any(|name| tensors.contains_key(*name));
        anyhow::ensure!(
            !(has_qwen && has_external),
            "artifact cannot contain both Flyingfish Qwen and external prompt provenance"
        );
        if has_qwen {
            return Ok(H3QwenNumericalContract::take_artifact_tensors(tensors)?
                .map(|contract| Self::FlyingfishQwen(Box::new(contract))));
        }
        Ok(
            ExternalPromptProvenance::take_artifact_tensors(tensors)?
                .map(Self::ExternallyValidated),
        )
    }
}

/// The length of a file, which is what this module records about its inputs.
fn file_bytes(path: &Path) -> Result<u64> {
    let stat = FileStat::of_target(path)?;
    anyhow::ensure!(
        stat.is_file(),
        "external prompt input is not a regular file: {}",
        path.display()
    );
    Ok(stat.len())
}

fn snapshot_bytes(snapshot: &ArtifactSnapshot) -> u64 {
    snapshot.bytes.len() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_external_prompt_provenance_is_complete_and_unbound() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("input.safetensors");
        std::fs::write(&input, b"official input bytes").unwrap();
        let manifest = directory.path().join("manifest.json");
        std::fs::write(
            &manifest,
            serde_json::to_vec(&serde_json::json!({
                "schema_version": 2,
                "producer": OFFICIAL_H3_FIXTURE_PRODUCER,
                "request": {"qwen_attention_backend": "eager"},
                "runtime": {
                    "qwen_attention_backend": "eager",
                    "qwen_use_kernels": false,
                    "attention_backend": "_native_math"
                },
                "model": {
                    "revision": OFFICIAL_MODEL_REVISION,
                    "filesystem_read_only": true
                },
                "producer_files": {"filesystem_read_only": true},
                "pytorch": {
                    "required_commit": OFFICIAL_PYTORCH_COMMIT,
                    "observed_git_commit": OFFICIAL_PYTORCH_COMMIT,
                    "source_files": {}
                },
                "transformers": {
                    "required_commit": OFFICIAL_TRANSFORMERS_COMMIT,
                    "observed_git_commit": OFFICIAL_TRANSFORMERS_COMMIT,
                    "source_files": {}
                },
                "diffusers": {
                    "required_commit": OFFICIAL_DIFFUSERS_COMMIT,
                    "observed_git_commit": OFFICIAL_DIFFUSERS_COMMIT,
                    "source_files": {}
                },
                "tensors": {"input.safetensors": {"bytes": 20}}
            }))
            .unwrap(),
        )
        .unwrap();
        let error = ExternalPromptProvenance::from_official_fixture_manifest(&manifest, &input)
            .unwrap_err()
            .to_string();
        assert!(error.contains("pinned official A1 fixture"), "{error}");
        let external = ExternalPromptProvenance {
            schema_version: EXTERNAL_PROMPT_PROVENANCE_SCHEMA_VERSION,
            binding: "externally_validated_unbound_to_flyingfish_qwen".to_owned(),
            manifest_bytes: OFFICIAL_A1_MANIFEST_BYTES,
            input_bytes: OFFICIAL_A1_INPUT_BYTES,
            manifest_producer: OFFICIAL_H3_FIXTURE_PRODUCER.to_owned(),
            manifest_schema_version: 2,
        };
        external.validate().unwrap();
        assert_eq!(
            external.binding,
            "externally_validated_unbound_to_flyingfish_qwen"
        );
        let provenance = H3ConditioningProvenance::ExternallyValidated(external.clone());
        let mut encoded = HashMap::new();
        provenance
            .insert_artifact_tensors(&mut encoded, &Device::Cpu)
            .unwrap();
        // The JSON and its schema; the digest tensor that used to sit beside
        // them described data already there.
        assert_eq!(encoded.len(), 2);
        let mut loaded = encoded
            .iter()
            .map(|(name, tensor)| ((*name).to_owned(), tensor.clone()))
            .collect();
        assert_eq!(
            H3ConditioningProvenance::take_artifact_tensors(&mut loaded).unwrap(),
            Some(provenance)
        );
        assert!(loaded.is_empty());
        assert!(external.verify_manifest(&manifest).is_err());

        let mut partial = HashMap::from([(
            EXTERNAL_PROMPT_PROVENANCE_JSON_TENSOR.to_owned(),
            encoded[EXTERNAL_PROMPT_PROVENANCE_JSON_TENSOR].clone(),
        )]);
        assert!(
            H3ConditioningProvenance::take_artifact_tensors(&mut partial)
                .unwrap_err()
                .to_string()
                .contains("incomplete")
        );

        let mut forged = external;
        forged.manifest_bytes += 1;
        assert!(
            forged
                .validate()
                .unwrap_err()
                .to_string()
                .contains("pinned official A1 manifest/input")
        );
    }
}
