//! Application resource selection. Family estimators and executable policies
//! remain in their adapters; dynamic observations are separate audit records.
pub mod evidence;
pub mod glm;
pub mod h3;
use crate::runtime::{artifact::ArtifactStaging, resource_selection::ResourceSelectionProvenance};
use anyhow::Result;
use std::path::Path;

pub fn publish_selection(path: &Path, record: &ResourceSelectionProvenance) -> Result<()> {
    let bytes = record.canonical_json()?;
    let staging = ArtifactStaging::new(path)?;
    staging.write_bytes(&bytes)?;
    staging.publish()?;
    Ok(())
}

#[derive(Clone, Copy)]
struct MeasuredScore {
    median_us: u64,
    uncertainty_us: u64,
    retained_peak_bytes: u128,
}

impl MeasuredScore {
    fn prefers(self, old: Self) -> bool {
        if self.median_us.saturating_add(self.uncertainty_us)
            < old.median_us.saturating_sub(old.uncertainty_us)
        {
            return true;
        }
        if old.median_us.saturating_add(old.uncertainty_us)
            < self.median_us.saturating_sub(self.uncertainty_us)
        {
            return false;
        }
        (self.retained_peak_bytes, self.median_us) < (old.retained_peak_bytes, old.median_us)
    }
}

pub fn prepare_cold_cache(
    component: &std::path::Path,
) -> anyhow::Result<crate::runtime::cold_cache::ColdCacheObservation> {
    prepare_cold_cache_components(&[component.to_owned()])
}

/// Establish one cold observation across every checkpoint used by a pipeline.
pub fn prepare_cold_cache_components(
    components: &[std::path::PathBuf],
) -> anyhow::Result<crate::runtime::cold_cache::ColdCacheObservation> {
    use crate::runtime::weights::{CachePolicy, ModelWeights, WeightSource};
    let mut paths = std::collections::BTreeSet::new();
    for component in components {
        let metadata = ModelWeights::open(component, WeightSource::Mmap, CachePolicy::new(1))?;
        for name in metadata.indexed_shard_names() {
            paths.insert(std::fs::canonicalize(component.join(name))?);
        }
    }
    crate::runtime::cold_cache::evict_and_verify(&paths.into_iter().collect::<Vec<_>>())
}

#[cfg(all(test, target_os = "linux"))]
mod cold_pipeline_tests {
    use super::*;
    #[test]
    fn cold_preparation_covers_all_unique_component_checkpoints() {
        let root = tempfile::tempdir().unwrap();
        let mut components = Vec::new();
        let mut expected = 0;
        for name in ["transformer", "encoder"] {
            let component = root.path().join(name);
            std::fs::create_dir(&component).unwrap();
            let path = component.join("model.safetensors");
            let tensor = candle_core::Tensor::zeros(
                1024,
                candle_core::DType::F32,
                &candle_core::Device::Cpu,
            )
            .unwrap();
            candle_core::safetensors::save(
                &std::collections::HashMap::from([("weight".to_owned(), tensor)]),
                &path,
            )
            .unwrap();
            std::fs::File::open(&path).unwrap().sync_all().unwrap();
            expected += path.metadata().unwrap().len();
            components.push(component);
        }
        components.push(components[0].clone());
        match prepare_cold_cache_components(&components) {
            Ok(report) => {
                assert_eq!(report.files, 2);
                assert_eq!(report.advised_bytes, expected);
                assert_eq!(report.resident_pages, 0);
            }
            Err(error) => assert!(
                error
                    .to_string()
                    .contains("cold file-cache state was not established")
            ),
        }
        assert!(prepare_cold_cache_components(&[]).is_err());
    }
}
