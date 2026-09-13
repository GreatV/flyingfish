//! Locating the components a TRELLIS pipeline names.
//!
//! A component reference in `pipeline.json` is a path without an extension, in
//! one of two forms. `ckpts/<stem>` is relative to the checkpoint that names
//! it. `<owner>/<repo>/ckpts/<stem>` names a component published in a different
//! checkpoint — the text pipelines carry only their own two flow models and
//! reach into the image checkpoint for every decoder — so resolving it needs a
//! directory holding both.

use anyhow::{Context, Result};
use ff_core::residency::WeightPhase;
use ff_core::weights::{CachePolicy, DeviceCache, ModelWeights, WeightSource};
use std::path::{Path, PathBuf};

use crate::config::{ComponentConfig, PipelineConfig};

/// Where a component reference resolved to, and what it turned out to be.
#[derive(Clone, Debug)]
pub struct ResolvedComponent {
    /// The role the pipeline gave it, such as `slat_decoder_mesh`.
    pub role: String,
    /// The reference exactly as published.
    pub reference: String,
    /// Directory holding the component's weights and sidecar.
    pub directory: PathBuf,
    /// Filename stem shared by `<stem>.json` and `<stem>.safetensors`.
    pub stem: String,
    pub config: ComponentConfig,
    /// Whether the reference left this checkpoint to reach another.
    pub is_cross_checkpoint: bool,
}

impl ResolvedComponent {
    pub fn weights_file_name(&self) -> String {
        format!("{}.safetensors", self.stem)
    }

    pub fn weights_path(&self) -> PathBuf {
        self.directory.join(self.weights_file_name())
    }

    /// Open the component's weights through the shared checkpoint store,
    /// retaining up to `residency` bytes of materialized tensors on the
    /// device.
    pub fn open_weights(
        &self,
        source: WeightSource,
        cache_policy: CachePolicy,
        residency: DeviceCache,
    ) -> Result<ModelWeights> {
        let mut weights = ModelWeights::open_component(
            &self.directory,
            &self.weights_file_name(),
            source,
            cache_policy,
        )
        .with_context(|| format!("failed to open TRELLIS component {}", self.reference))?;
        let prioritized = if residency.phase_selected(&format!("trellis.{}", self.role)) {
            weights
                .tensor_names()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        weights.configure_device_cache_with_priority(residency, prioritized)?;
        Ok(weights)
    }

    /// The component's residency phase: every
    /// tensor it holds, read `reuse_count` times over one request — the step
    /// count for a flow model, one for a decoder run once.
    pub fn weight_phase(&self, weights: &ModelWeights, reuse_count: u64) -> WeightPhase {
        WeightPhase::new(
            format!("trellis.{}", self.role),
            weights.tensor_names().map(str::to_owned),
            reuse_count,
        )
    }
}

/// One published TRELLIS checkpoint directory and the components it names.
#[derive(Clone, Debug)]
pub struct TrellisCheckpoint {
    pub root: PathBuf,
    pub pipeline: PipelineConfig,
    pub components: Vec<ResolvedComponent>,
}

impl TrellisCheckpoint {
    /// Read a checkpoint directory, resolving cross-checkpoint references
    /// against `models_root`.
    ///
    /// `models_root` defaults to the directory holding the published
    /// checkpoints. Under a Hugging Face-aligned layout that is the
    /// grandparent, because a checkpoint sits at `<models>/<owner>/<repo>`;
    /// under a flat one it is the parent. Both are tried.
    pub fn open(root: impl AsRef<Path>, models_root: Option<&Path>) -> Result<Self> {
        let root = root.as_ref();
        anyhow::ensure!(
            root.is_dir(),
            "TRELLIS checkpoint directory does not exist: {}",
            root.display()
        );
        let pipeline_path = root.join("pipeline.json");
        let pipeline = PipelineConfig::from_json(
            &std::fs::read(&pipeline_path)
                .with_context(|| format!("failed to read {}", pipeline_path.display()))?,
        )
        .with_context(|| format!("invalid {}", pipeline_path.display()))?;

        let mut roots = Vec::new();
        if let Some(path) = models_root {
            roots.push(path.to_path_buf());
        } else {
            let parent = root
                .parent()
                .context("TRELLIS checkpoint has no parent directory to resolve siblings in")?;
            if let Some(grandparent) = parent.parent() {
                roots.push(grandparent.to_path_buf());
            }
            roots.push(parent.to_path_buf());
        }

        let mut components = Vec::with_capacity(pipeline.args.models.len());
        for (role, reference) in &pipeline.args.models {
            components.push(resolve(
                root,
                &roots,
                role,
                reference,
                models_root.is_some(),
            )?);
        }
        Ok(Self {
            root: root.to_path_buf(),
            pipeline,
            components,
        })
    }

    /// Components this checkpoint reaches into another checkpoint for.
    pub fn cross_checkpoint_components(&self) -> impl Iterator<Item = &ResolvedComponent> {
        self.components
            .iter()
            .filter(|component| component.is_cross_checkpoint)
    }
}

fn resolve(
    root: &Path,
    models_roots: &[PathBuf],
    role: &str,
    reference: &str,
    allow_renamed: bool,
) -> Result<ResolvedComponent> {
    anyhow::ensure!(
        !reference.is_empty() && !reference.starts_with('/') && !reference.contains(".."),
        "TRELLIS component reference {reference:?} is not a relative path"
    );
    let segments: Vec<&str> = reference.split('/').collect();
    let stem = (*segments
        .last()
        .context("TRELLIS component reference has no final segment")?)
    .to_owned();
    anyhow::ensure!(
        !stem.is_empty(),
        "TRELLIS component reference {reference:?} has an empty name"
    );

    if segments.len() <= 2 {
        let directory = root.join(segments[..segments.len() - 1].join("/"));
        return finish(role, reference, directory, stem, false);
    }

    let owner = (segments.len() >= 4).then(|| segments[segments.len() - 4]);
    let repository = segments[segments.len() - 3];
    let subdirectory = segments[segments.len() - 2];
    let mut searched = Vec::new();
    for models_root in models_roots {
        for candidate in owner_candidates(models_root, owner, repository) {
            let directory = candidate.join(subdirectory);
            if directory.join(format!("{stem}.json")).is_file() {
                return finish(role, reference, directory, stem, true);
            }
            searched.push(directory);
        }
    }
    if allow_renamed {
        let mut matches = std::collections::BTreeSet::new();
        for root in models_roots {
            let mut candidates = vec![root.clone()];
            let mut frontier = vec![root.clone()];
            for _ in 0..2 {
                let mut next = Vec::new();
                for parent in frontier {
                    for entry in std::fs::read_dir(parent)? {
                        let entry = entry?;
                        if entry.file_type()?.is_dir() {
                            next.push(entry.path());
                        }
                    }
                }
                candidates.extend(next.iter().cloned());
                frontier = next;
            }
            for candidate in candidates {
                let directory = candidate.join(subdirectory);
                if directory.join(format!("{stem}.json")).is_file()
                    && directory.join(format!("{stem}.safetensors")).is_file()
                {
                    matches.insert(std::fs::canonicalize(directory)?);
                }
            }
        }
        anyhow::ensure!(
            matches.len() <= 1,
            "ambiguous component {reference:?}; set --models-root to the intended shared checkpoint directory"
        );
        if let Some(directory) = matches.into_iter().next() {
            return finish(role, reference, directory, stem, true);
        }
    }
    anyhow::bail!(
        "TRELLIS component {reference:?} (role {role}) was not found; looked in {}",
        searched
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Where a `<owner>/<repository>` reference may live.
///
/// The published pipelines disagree about the owner of one repository: the
/// TRELLIS-1 text pipelines name `JeffreyXiang/TRELLIS-image-large` and the
/// TRELLIS.2 pipeline names `microsoft/TRELLIS-image-large`, for the same
/// weights. Storing that twice to satisfy both spellings would be the wrong
/// answer, so the owner is a preference and the repository name is the key.
fn owner_candidates(models_root: &Path, owner: Option<&str>, repository: &str) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(owner) = owner {
        candidates.push(models_root.join(owner).join(repository));
    }
    candidates.push(models_root.join(repository));
    if let Ok(entries) = std::fs::read_dir(models_root) {
        let mut others: Vec<PathBuf> = entries
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .map(|path| path.join(repository))
            .filter(|path| path.is_dir() && !candidates.contains(path))
            .collect();
        others.sort();
        candidates.extend(others);
    }
    candidates
}

fn finish(
    role: &str,
    reference: &str,
    directory: PathBuf,
    stem: String,
    is_cross_checkpoint: bool,
) -> Result<ResolvedComponent> {
    let sidecar = directory.join(format!("{stem}.json"));
    anyhow::ensure!(
        sidecar.is_file(),
        "TRELLIS component {reference:?} (role {role}) has no configuration at {}",
        sidecar.display()
    );
    let config = ComponentConfig::from_json(
        &std::fs::read(&sidecar)
            .with_context(|| format!("failed to read {}", sidecar.display()))?,
    )
    .with_context(|| format!("invalid {}", sidecar.display()))?;
    let weights = directory.join(format!("{stem}.safetensors"));
    anyhow::ensure!(
        weights.is_file(),
        "TRELLIS component {reference:?} (role {role}) has no weights at {}",
        weights.display()
    );
    Ok(ResolvedComponent {
        role: role.to_owned(),
        reference: reference.to_owned(),
        directory,
        stem,
        config,
        is_cross_checkpoint,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLOW: &str = r#"{"name":"SparseStructureFlowModel","args":{"resolution":16,
        "in_channels":8,"out_channels":8,"model_channels":768,"cond_channels":768,
        "num_blocks":12,"num_heads":12,"mlp_ratio":4,"patch_size":1,"pe_mode":"ape",
        "qk_rms_norm":true,"use_fp16":true}}"#;

    fn write_component(directory: &Path, stem: &str, config: &str) {
        std::fs::create_dir_all(directory).unwrap();
        std::fs::write(directory.join(format!("{stem}.json")), config).unwrap();
        std::fs::write(directory.join(format!("{stem}.safetensors")), b"").unwrap();
    }

    #[test]
    fn explicit_models_root_supports_renamed_checkpoints_and_rejects_ambiguity() {
        let models = tempfile::tempdir().unwrap();
        let here = models.path().join("my-text-model");
        let shared = models.path().join("my-shared-weights");
        std::fs::create_dir_all(&here).unwrap();
        write_component(&shared.join("ckpts"), "shared_flow", FLOW);
        std::fs::write(here.join("pipeline.json"),r#"{"name":"TrellisTextTo3DPipeline","args":{"models":{"flow":"owner/original-name/ckpts/shared_flow"}}}"#).unwrap();
        let result = TrellisCheckpoint::open(&here, Some(models.path())).unwrap();
        assert_eq!(
            result.components[0].directory,
            std::fs::canonicalize(shared.join("ckpts")).unwrap()
        );
        write_component(&models.path().join("duplicate/ckpts"), "shared_flow", FLOW);
        assert!(
            TrellisCheckpoint::open(&here, Some(models.path()))
                .unwrap_err()
                .to_string()
                .contains("ambiguous")
        );
        assert!(TrellisCheckpoint::open(&here, Some(&shared)).is_ok());
    }

    #[test]
    fn ownerless_cross_checkpoint_references_do_not_panic() {
        let models = tempfile::tempdir().unwrap();
        write_component(&models.path().join("repo/ckpts"), "flow", FLOW);
        let result = resolve(
            models.path(),
            &[models.path().to_path_buf()],
            "flow",
            "repo/ckpts/flow",
            false,
        )
        .unwrap();
        assert_eq!(result.stem, "flow");
    }

    #[test]
    fn resolves_both_reference_forms_against_a_shared_models_root() {
        let models = tempfile::tempdir().unwrap();
        let here = models.path().join("TRELLIS-text-base");
        let other = models.path().join("TRELLIS-image-large");
        write_component(&here.join("ckpts"), "own_flow", FLOW);
        write_component(&other.join("ckpts"), "shared_flow", FLOW);
        std::fs::write(
            here.join("pipeline.json"),
            r#"{"name":"TrellisTextTo3DPipeline","args":{"models":{
                 "a_own":"ckpts/own_flow",
                 "b_shared":"JeffreyXiang/TRELLIS-image-large/ckpts/shared_flow"},
                 "text_cond_model":"openai/clip-vit-large-patch14"}}"#,
        )
        .unwrap();

        let checkpoint = TrellisCheckpoint::open(&here, None).unwrap();
        assert_eq!(checkpoint.components.len(), 2);
        let own = &checkpoint.components[0];
        assert!(!own.is_cross_checkpoint);
        assert_eq!(own.directory, here.join("ckpts"));
        let shared = &checkpoint.components[1];
        assert!(shared.is_cross_checkpoint);
        assert_eq!(shared.directory, other.join("ckpts"));
        assert_eq!(checkpoint.cross_checkpoint_components().count(), 1);
    }

    #[test]
    fn a_missing_cross_checkpoint_component_names_the_path_it_looked_in() {
        let models = tempfile::tempdir().unwrap();
        let here = models.path().join("TRELLIS-text-base");
        std::fs::create_dir_all(&here).unwrap();
        std::fs::write(
            here.join("pipeline.json"),
            r#"{"name":"TrellisTextTo3DPipeline","args":{"models":{
                 "decoder":"JeffreyXiang/TRELLIS-image-large/ckpts/absent"}}}"#,
        )
        .unwrap();
        let error = TrellisCheckpoint::open(&here, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("was not found"), "{error}");
        assert!(
            error.contains("JeffreyXiang/TRELLIS-image-large/ckpts"),
            "{error}"
        );
    }

    #[test]
    fn an_escaping_reference_is_refused() {
        let models = tempfile::tempdir().unwrap();
        let here = models.path().join("checkpoint");
        std::fs::create_dir_all(&here).unwrap();
        std::fs::write(
            here.join("pipeline.json"),
            r#"{"name":"P","args":{"models":{"a":"../../etc/passwd"}}}"#,
        )
        .unwrap();
        let error = TrellisCheckpoint::open(&here, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("not a relative path"), "{error}");
    }
}
