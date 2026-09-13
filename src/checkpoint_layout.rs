//! Read-only H3 stage-layout inspection and explicit offline exact-stage repacking.
//!
//! Source tensors remain raw safetensors byte ranges. This module never
//! materializes them through Candle.

use crate::{
    durable_fs::{create_private_directory, sync_directory},
    h3::config::TransformerConfig,
    h3::execution::{H3ExecutionPlan, StageKind},
    runtime::identity::{FileStamp, NamedFileStamp, WeakModelIdentity, stamp_file},
    runtime::weights::{CachePolicy, ModelWeights, RawTensorMetadata, WeightSource},
};
use anyhow::{Context, Result, bail};
use memmap2::{Mmap, MmapOptions};
use safetensors::{
    SafeTensors,
    tensor::{Metadata, TensorInfo, TensorView, serialize_to_file},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

pub const H3_EXACT_STAGE_LAYOUT_REPORT_SCHEMA_VERSION: u32 = 1;
pub const H3_EXACT_STAGE_MANIFEST_SCHEMA_VERSION: u32 = 1;
pub const H3_EXACT_STAGE_LAYOUT_VERSION: &str = "h3-base-transformer-exact-stage-v1";
pub const H3_EXACT_STAGE_MANIFEST_FILE: &str = "flyingfish-exact-stage-layout.json";

const RELEASED_BASE_TENSORS: usize = 638;
const RELEASED_BASE_SHARDS: usize = 14;
const RELEASED_BASE_PAYLOAD_BYTES: u64 = 66_280_430_080;
const RELEASED_BASE_STAGES: usize = 159;
const MAX_EXACT_STAGE_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const SAFETENSORS_HEADER_PREFIX_BYTES: u64 = size_of::<u64>() as u64;
static STAGING_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct H3StageSourceShard {
    pub name: String,
    pub file_bytes: u64,
    pub header_bytes: u64,
    pub payload_bytes: u64,
    pub tensor_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct H3StageTensorLayout {
    pub name: String,
    pub source_shard: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub source_file_offset: u64,
    pub payload_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct H3StageContiguousExtent {
    pub source_shard: String,
    pub source_file_offset: u64,
    pub payload_bytes: u64,
    pub tensor_names: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct H3ExactStageLayout {
    pub ordinal: usize,
    pub kind: StageKind,
    pub destination_shard: String,
    pub payload_bytes: u64,
    pub destination_file_bytes: u64,
    pub source_shards: Vec<String>,
    pub contiguous_extents: Vec<H3StageContiguousExtent>,
    pub tensors: Vec<H3StageTensorLayout>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct H3ExactStageLayoutReport {
    pub schema_version: u32,
    pub layout_version: String,
    pub source_component: String,
    pub source_index: NamedFileStamp,
    pub source_shards: Vec<H3StageSourceShard>,
    pub stages: Vec<H3ExactStageLayout>,
    pub tensor_count: usize,
    pub total_payload_bytes: u64,
    pub peak_stage_payload_bytes: u64,
    pub contiguous_extent_count: usize,
    pub destination_weight_file_bytes: u64,
    pub required_destination_capacity_bytes: u64,
    pub source_and_destination_capacity_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct H3VerifiedStageTensor {
    pub name: String,
    pub payload_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct H3ExactStageManifest {
    pub schema_version: u32,
    pub layout_version: String,
    pub source_fingerprint: WeakModelIdentity,
    pub source_fingerprint_verified_unchanged: bool,
    pub layout: H3ExactStageLayoutReport,
    pub destination_index: NamedFileStamp,
    pub verified_tensors: Vec<H3VerifiedStageTensor>,
}

#[derive(Clone, Copy)]
struct ExpectedLayoutProfile {
    layers: usize,
    refiner_layers: usize,
    tensors: usize,
    shards: usize,
    payload_bytes: u64,
    stages: usize,
    released_config: bool,
}

const RELEASED_BASE_PROFILE: ExpectedLayoutProfile = ExpectedLayoutProfile {
    layers: 50,
    refiner_layers: 2,
    tensors: RELEASED_BASE_TENSORS,
    shards: RELEASED_BASE_SHARDS,
    payload_bytes: RELEASED_BASE_PAYLOAD_BYTES,
    stages: RELEASED_BASE_STAGES,
    released_config: true,
};

impl H3ExactStageLayoutReport {
    pub fn canonical_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        serde_json::to_vec(self).context("failed to serialize H3 exact-stage layout report")
    }

    pub fn pretty_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        serde_json::to_vec_pretty(self).context("failed to serialize H3 exact-stage layout report")
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == H3_EXACT_STAGE_LAYOUT_REPORT_SCHEMA_VERSION,
            "unsupported H3 exact-stage layout report schema {}",
            self.schema_version
        );
        anyhow::ensure!(
            self.layout_version == H3_EXACT_STAGE_LAYOUT_VERSION,
            "unsupported H3 exact-stage layout version {}",
            self.layout_version
        );
        anyhow::ensure!(
            !self.stages.is_empty(),
            "H3 exact-stage report has no stages"
        );
        anyhow::ensure!(
            self.source_shards
                .windows(2)
                .all(|pair| pair[0].name < pair[1].name),
            "H3 exact-stage source shards are not strictly sorted"
        );
        let mut names = BTreeSet::new();
        let mut total_payload = 0u64;
        let mut destination_weight_bytes = 0u64;
        let mut peak = 0u64;
        let mut extent_count = 0usize;
        for (ordinal, stage) in self.stages.iter().enumerate() {
            anyhow::ensure!(
                stage.ordinal == ordinal,
                "H3 exact-stage ordinals must be contiguous from zero"
            );
            anyhow::ensure!(
                !stage.tensors.is_empty() && !stage.contiguous_extents.is_empty(),
                "H3 exact stage {ordinal} is empty"
            );
            let stage_payload = stage.tensors.iter().try_fold(0u64, |sum, tensor| {
                anyhow::ensure!(
                    names.insert(tensor.name.as_str()),
                    "H3 exact-stage tensor occurs more than once: {}",
                    tensor.name
                );
                sum.checked_add(tensor.payload_bytes)
                    .context("H3 exact-stage payload total overflow")
            })?;
            anyhow::ensure!(
                stage_payload == stage.payload_bytes,
                "H3 exact-stage payload mismatch at stage {ordinal}"
            );
            let extent_payload =
                stage
                    .contiguous_extents
                    .iter()
                    .try_fold(0u64, |sum, extent| {
                        sum.checked_add(extent.payload_bytes)
                            .context("H3 exact-stage extent total overflow")
                    })?;
            anyhow::ensure!(
                extent_payload == stage.payload_bytes,
                "H3 exact-stage extent payload mismatch at stage {ordinal}"
            );
            total_payload = total_payload
                .checked_add(stage.payload_bytes)
                .context("H3 exact-stage total payload overflow")?;
            destination_weight_bytes = destination_weight_bytes
                .checked_add(stage.destination_file_bytes)
                .context("H3 exact-stage destination size overflow")?;
            peak = peak.max(stage.payload_bytes);
            extent_count = extent_count
                .checked_add(stage.contiguous_extents.len())
                .context("H3 exact-stage extent count overflow")?;
        }
        anyhow::ensure!(
            names.len() == self.tensor_count
                && total_payload == self.total_payload_bytes
                && peak == self.peak_stage_payload_bytes
                && extent_count == self.contiguous_extent_count
                && destination_weight_bytes == self.destination_weight_file_bytes,
            "H3 exact-stage report aggregate fields are inconsistent"
        );
        anyhow::ensure!(
            self.required_destination_capacity_bytes >= self.destination_weight_file_bytes,
            "H3 exact-stage required capacity is smaller than its weight files"
        );
        let source_file_bytes = self.source_shards.iter().try_fold(0u64, |sum, shard| {
            sum.checked_add(shard.file_bytes)
                .context("H3 exact-stage source size overflow")
        })?;
        anyhow::ensure!(
            self.source_and_destination_capacity_bytes
                == source_file_bytes
                    .checked_add(self.required_destination_capacity_bytes)
                    .context("H3 source-and-destination size overflow")?,
            "H3 exact-stage combined capacity is inconsistent"
        );
        Ok(())
    }
}

impl H3ExactStageManifest {
    pub fn canonical_json(&self) -> Result<Vec<u8>> {
        anyhow::ensure!(
            self.schema_version == H3_EXACT_STAGE_MANIFEST_SCHEMA_VERSION,
            "unsupported H3 exact-stage manifest schema {}",
            self.schema_version
        );
        anyhow::ensure!(
            self.layout_version == H3_EXACT_STAGE_LAYOUT_VERSION
                && self.layout.layout_version == self.layout_version,
            "H3 exact-stage manifest layout version is inconsistent"
        );
        anyhow::ensure!(
            self.source_fingerprint_verified_unchanged,
            "H3 exact-stage manifest does not prove an unchanged source fingerprint"
        );
        self.source_fingerprint.validate()?;
        self.layout.validate()?;
        anyhow::ensure!(
            self.verified_tensors.len() == self.layout.tensor_count,
            "H3 exact-stage verified tensor count is inconsistent"
        );
        let mut previous: Option<&str> = None;
        for tensor in &self.verified_tensors {
            if let Some(previous) = previous {
                anyhow::ensure!(
                    previous < tensor.name.as_str(),
                    "H3 exact-stage verified tensors are not strictly sorted"
                );
            }
            previous = Some(&tensor.name);
        }
        let json =
            serde_json::to_vec(self).context("failed to serialize H3 exact-stage manifest")?;
        anyhow::ensure!(
            json.len() as u64 <= MAX_EXACT_STAGE_MANIFEST_BYTES,
            "H3 exact-stage manifest exceeds {MAX_EXACT_STAGE_MANIFEST_BYTES} bytes"
        );
        Ok(json)
    }
}

pub fn inspect_h3_base_exact_stage_layout(source: &Path) -> Result<H3ExactStageLayoutReport> {
    build_layout(source, RELEASED_BASE_PROFILE)
}

pub fn repack_h3_base_exact_stages(
    source: &Path,
    destination: &Path,
) -> Result<H3ExactStageManifest> {
    repack_with_profile(source, destination, RELEASED_BASE_PROFILE)
}

fn build_layout(source: &Path, profile: ExpectedLayoutProfile) -> Result<H3ExactStageLayoutReport> {
    let source = validate_source_component(source, profile.released_config)?;
    let config = TransformerConfig::from_file(source.join("config.json"))?;
    validate_profile_config(&config, profile)?;
    let weights = ModelWeights::open(&source, WeightSource::Mmap, CachePolicy::new(1))?;
    let index_name = weights
        .index_path()
        .strip_prefix(&source)
        .context("checkpoint index is outside the source component")?
        .to_str()
        .context("checkpoint index filename is not UTF-8")?
        .to_owned();
    anyhow::ensure!(
        matches!(
            index_name.as_str(),
            "diffusion_pytorch_model.safetensors.index.json" | "model.safetensors.index.json"
        ),
        "exact-stage conversion requires an indexed checkpoint"
    );
    let inventory = weights.inventory();
    anyhow::ensure!(
        inventory.tensors == profile.tensors
            && inventory.shards == profile.shards
            && inventory.indexed_bytes == Some(profile.payload_bytes),
        "checkpoint does not match the H3 Base tensor/shard/payload profile"
    );
    let verification = weights.verify()?;
    anyhow::ensure!(
        verification.checked_tensors == profile.tensors
            && verification.checked_shards == profile.shards,
        "checkpoint index/header verification did not cover the Base profile"
    );

    let source_index_stamp = stamp_file(weights.index_path()).with_context(|| {
        format!(
            "failed to hash source index {}",
            weights.index_path().display()
        )
    })?;
    let source_index = NamedFileStamp {
        relative_path: index_name,
        bytes: source_index_stamp.bytes,
        modified_ns: source_index_stamp.modified_ns,
    };

    let mut indexed_by_shard: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for name in weights.tensor_names() {
        let raw = weights.raw_tensor_metadata(name)?;
        indexed_by_shard
            .entry(raw.shard)
            .or_default()
            .insert(name.to_owned());
    }
    let shard_names = weights.indexed_shard_names();
    let mut source_shards = Vec::with_capacity(shard_names.len());
    let mut source_file_bytes = 0u64;
    let mut header_payload_bytes = 0u64;
    for shard_name in shard_names {
        let shard = weights.raw_shard_metadata(&shard_name)?;
        let indexed = indexed_by_shard
            .get(&shard_name)
            .with_context(|| format!("source shard has no indexed tensors: {shard_name}"))?;
        let header_names = shard
            .tensors
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect::<BTreeSet<_>>();
        anyhow::ensure!(
            &header_names == indexed,
            "source shard header and index tensor names disagree: {shard_name}"
        );
        source_file_bytes = source_file_bytes
            .checked_add(shard.file_bytes)
            .context("source shard file-byte total overflow")?;
        header_payload_bytes = header_payload_bytes
            .checked_add(shard.payload_bytes)
            .context("source shard payload-byte total overflow")?;
        source_shards.push(H3StageSourceShard {
            name: shard.name,
            file_bytes: shard.file_bytes,
            header_bytes: shard.header_bytes,
            payload_bytes: shard.payload_bytes,
            tensor_count: shard.tensors.len(),
        });
    }
    anyhow::ensure!(
        header_payload_bytes == weights.indexed_payload_bytes()
            && header_payload_bytes == profile.payload_bytes,
        "source headers, index, and Base payload totals disagree"
    );

    let plan = H3ExecutionPlan::build(&weights, profile.layers, profile.refiner_layers)?;
    anyhow::ensure!(
        plan.stages().len() == profile.stages,
        "checkpoint produced {} execution stages, expected {}",
        plan.stages().len(),
        profile.stages
    );
    let stage_count = plan.stages().len();
    let mut stages = Vec::with_capacity(stage_count);
    let mut weight_map = BTreeMap::new();
    for (ordinal, stage) in plan.stages().iter().enumerate() {
        let destination_shard = stage_file_name(ordinal, stage_count);
        let mut raw = Vec::with_capacity(stage.tensor_names.len());
        let mut tensors = Vec::with_capacity(stage.tensor_names.len());
        for name in &stage.tensor_names {
            let info = weights.raw_tensor_metadata(name)?;
            weight_map.insert(name.clone(), destination_shard.clone());
            tensors.push(stage_tensor_layout(&info)?);
            raw.push(info);
        }
        let payload_bytes = raw.iter().try_fold(0u64, |sum, tensor| {
            sum.checked_add(tensor.bytes as u64)
                .context("stage payload-byte total overflow")
        })?;
        anyhow::ensure!(
            payload_bytes == stage.weight_bytes,
            "execution-plan and header bytes disagree for stage {ordinal}"
        );
        let destination_file_bytes = serialized_stage_file_bytes(&raw)?;
        let contiguous_extents = contiguous_extents(&raw)?;
        let source_shards = raw
            .iter()
            .map(|tensor| tensor.shard.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        stages.push(H3ExactStageLayout {
            ordinal,
            kind: stage.kind.clone(),
            destination_shard,
            payload_bytes,
            destination_file_bytes,
            source_shards,
            contiguous_extents,
            tensors,
        });
    }

    let total_payload_bytes = stages.iter().try_fold(0u64, |sum, stage| {
        sum.checked_add(stage.payload_bytes)
            .context("stage payload-byte total overflow")
    })?;
    let destination_weight_file_bytes = stages.iter().try_fold(0u64, |sum, stage| {
        sum.checked_add(stage.destination_file_bytes)
            .context("destination weight-file total overflow")
    })?;
    let destination_index = DestinationIndex {
        metadata: DestinationIndexMetadata {
            total_size: total_payload_bytes,
        },
        weight_map,
    };
    let destination_index_bytes = u64::try_from(
        serde_json::to_vec(&destination_index)
            .context("failed to size destination checkpoint index")?
            .len(),
    )
    .context("destination index length exceeds u64")?;
    let config_bytes = fs::metadata(source.join("config.json"))
        .context("failed to stat source transformer config")?
        .len();
    let required_destination_capacity_bytes = destination_weight_file_bytes
        .checked_add(config_bytes)
        .and_then(|bytes| bytes.checked_add(destination_index_bytes))
        .and_then(|bytes| bytes.checked_add(MAX_EXACT_STAGE_MANIFEST_BYTES))
        .context("required destination capacity overflow")?;
    let report = H3ExactStageLayoutReport {
        schema_version: H3_EXACT_STAGE_LAYOUT_REPORT_SCHEMA_VERSION,
        layout_version: H3_EXACT_STAGE_LAYOUT_VERSION.to_owned(),
        source_component: source
            .to_str()
            .context("source component path is not UTF-8")?
            .to_owned(),
        source_index,
        source_shards,
        tensor_count: inventory.tensors,
        total_payload_bytes,
        peak_stage_payload_bytes: stages
            .iter()
            .map(|stage| stage.payload_bytes)
            .max()
            .unwrap_or(0),
        contiguous_extent_count: stages
            .iter()
            .map(|stage| stage.contiguous_extents.len())
            .sum(),
        destination_weight_file_bytes,
        required_destination_capacity_bytes,
        source_and_destination_capacity_bytes: source_file_bytes
            .checked_add(required_destination_capacity_bytes)
            .context("combined source-and-destination capacity overflow")?,
        stages,
    };
    report.validate()?;
    Ok(report)
}

fn stage_tensor_layout(info: &RawTensorMetadata) -> Result<H3StageTensorLayout> {
    Ok(H3StageTensorLayout {
        name: info.name.clone(),
        source_shard: info.shard.clone(),
        dtype: format!("{:?}", info.dtype),
        shape: info.shape.clone(),
        source_file_offset: u64::try_from(info.file_offset)
            .context("source tensor offset exceeds u64")?,
        payload_bytes: u64::try_from(info.bytes).context("source tensor bytes exceed u64")?,
    })
}

fn contiguous_extents(raw: &[RawTensorMetadata]) -> Result<Vec<H3StageContiguousExtent>> {
    let mut ordered = raw.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        (&left.shard, left.file_offset, &left.name).cmp(&(
            &right.shard,
            right.file_offset,
            &right.name,
        ))
    });
    let mut extents: Vec<H3StageContiguousExtent> = Vec::new();
    for tensor in ordered {
        let offset =
            u64::try_from(tensor.file_offset).context("source tensor offset exceeds u64")?;
        let bytes = u64::try_from(tensor.bytes).context("source tensor bytes exceed u64")?;
        if let Some(last) = extents.last_mut() {
            let end = last
                .source_file_offset
                .checked_add(last.payload_bytes)
                .context("source extent end overflow")?;
            if last.source_shard == tensor.shard && end == offset {
                last.payload_bytes = last
                    .payload_bytes
                    .checked_add(bytes)
                    .context("source extent length overflow")?;
                last.tensor_names.push(tensor.name.clone());
                continue;
            }
        }
        extents.push(H3StageContiguousExtent {
            source_shard: tensor.shard.clone(),
            source_file_offset: offset,
            payload_bytes: bytes,
            tensor_names: vec![tensor.name.clone()],
        });
    }
    Ok(extents)
}

fn serialized_stage_file_bytes(raw: &[RawTensorMetadata]) -> Result<u64> {
    let mut ordered = raw.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        right
            .dtype
            .cmp(&left.dtype)
            .then_with(|| left.name.cmp(&right.name))
    });
    let mut offset = 0usize;
    let mut infos = Vec::with_capacity(ordered.len());
    for tensor in ordered {
        let end = offset
            .checked_add(tensor.bytes)
            .context("destination tensor offset overflow")?;
        infos.push((
            tensor.name.clone(),
            TensorInfo {
                dtype: tensor.dtype,
                shape: tensor.shape.clone(),
                data_offsets: (offset, end),
            },
        ));
        offset = end;
    }
    let metadata = Metadata::new(None, infos).context("invalid destination stage metadata")?;
    let header_len = serde_json::to_vec(&metadata)
        .context("failed to serialize destination stage metadata")?
        .len()
        .next_multiple_of(size_of::<u64>());
    SAFETENSORS_HEADER_PREFIX_BYTES
        .checked_add(u64::try_from(header_len).context("stage header length exceeds u64")?)
        .and_then(|bytes| bytes.checked_add(offset as u64))
        .context("destination stage file size overflow")
}

fn stage_file_name(ordinal: usize, count: usize) -> String {
    format!("model-{:05}-of-{:05}.safetensors", ordinal + 1, count)
}

#[derive(Serialize)]
struct DestinationIndex {
    metadata: DestinationIndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Serialize)]
struct DestinationIndexMetadata {
    total_size: u64,
}

fn destination_index(layout: &H3ExactStageLayoutReport) -> DestinationIndex {
    let mut weight_map = BTreeMap::new();
    for stage in &layout.stages {
        for tensor in &stage.tensors {
            weight_map.insert(tensor.name.clone(), stage.destination_shard.clone());
        }
    }
    DestinationIndex {
        metadata: DestinationIndexMetadata {
            total_size: layout.total_payload_bytes,
        },
        weight_map,
    }
}

fn repack_with_profile(
    source: &Path,
    destination: &Path,
    profile: ExpectedLayoutProfile,
) -> Result<H3ExactStageManifest> {
    let source = validate_source_component(source, profile.released_config)?;
    let destination = checked_new_destination(&source, destination)?;
    ensure_same_filesystem(
        &source,
        destination.parent().context("destination has no parent")?,
    )?;
    let source_before = WeakModelIdentity::collect(&source)
        .context("failed to fingerprint source before exact-stage conversion")?;
    validate_indexed_fingerprint(&source_before, profile.shards)?;
    let layout = build_layout(&source, profile)?;
    let weights = ModelWeights::open(&source, WeightSource::Mmap, CachePolicy::new(1))?;
    let mut staging =
        StagingDirectory::create(destination.parent().context("destination has no parent")?)?;

    let conversion = (|| {
        copy_regular_file_new(
            &source.join("config.json"),
            &staging.path.join("config.json"),
        )?;
        let mut verified = BTreeMap::new();
        for stage in &layout.stages {
            write_and_verify_stage(&source, &staging.path, stage, &weights, &mut verified)?;
        }
        anyhow::ensure!(
            verified.len() == layout.tensor_count,
            "exact-stage conversion verified {} tensors, expected {}",
            verified.len(),
            layout.tensor_count
        );

        let destination_index_name = layout.source_index.relative_path.clone();
        let destination_index_path = staging.path.join(&destination_index_name);
        write_json_new(&destination_index_path, &destination_index(&layout))?;
        let staged_weights =
            ModelWeights::open(&staging.path, WeightSource::Mmap, CachePolicy::new(1))?;
        let report = staged_weights.verify()?;
        anyhow::ensure!(
            report.checked_tensors == layout.tensor_count
                && report.checked_shards == layout.stages.len(),
            "destination index does not cover every exact-stage tensor"
        );
        anyhow::ensure!(
            staged_weights.indexed_payload_bytes() == layout.total_payload_bytes,
            "destination index payload total changed"
        );
        let FileStamp { bytes, modified_ns } = stamp_file(&destination_index_path)
            .context("failed to hash destination checkpoint index")?;
        let destination_index = NamedFileStamp {
            relative_path: destination_index_name,
            bytes,
            modified_ns,
        };

        let source_after = WeakModelIdentity::collect(&source)
            .context("failed to fingerprint source after exact-stage conversion")?;
        anyhow::ensure!(
            source_before == source_after,
            "source fingerprint changed during exact-stage conversion"
        );
        let manifest = H3ExactStageManifest {
            schema_version: H3_EXACT_STAGE_MANIFEST_SCHEMA_VERSION,
            layout_version: H3_EXACT_STAGE_LAYOUT_VERSION.to_owned(),
            source_fingerprint: source_before,
            source_fingerprint_verified_unchanged: true,
            layout,
            destination_index,
            verified_tensors: verified.into_values().collect(),
        };
        write_bytes_new(
            &staging.path.join(H3_EXACT_STAGE_MANIFEST_FILE),
            &manifest.canonical_json()?,
        )?;
        sync_directory(&staging.path)?;
        Ok(manifest)
    })();

    let manifest = match conversion {
        Ok(manifest) => manifest,
        Err(error) => {
            if let Err(cleanup) = staging.cleanup() {
                return Err(error.context(format!(
                    "conversion also failed to clean private staging: {cleanup:#}"
                )));
            }
            return Err(error);
        }
    };
    rename_directory_noreplace(&staging.path, &destination)?;
    staging.disarm();
    sync_directory(
        destination
            .parent()
            .context("published destination has no parent")?,
    )?;
    Ok(manifest)
}

fn write_and_verify_stage(
    source: &Path,
    staging: &Path,
    stage: &H3ExactStageLayout,
    weights: &ModelWeights,
    verified: &mut BTreeMap<String, H3VerifiedStageTensor>,
) -> Result<()> {
    let mut raw = Vec::with_capacity(stage.tensors.len());
    for tensor in &stage.tensors {
        let info = weights.raw_tensor_metadata(&tensor.name)?;
        anyhow::ensure!(
            stage_tensor_layout(&info)? == *tensor,
            "source tensor metadata changed before stage {}: {}",
            stage.ordinal,
            tensor.name
        );
        raw.push(info);
    }
    let source_mmaps = open_source_mmaps(source, &stage.source_shards)?;
    let mut views = Vec::with_capacity(raw.len());
    for tensor in &raw {
        let mmap = source_mmaps
            .get(&tensor.shard)
            .with_context(|| format!("source mmap is missing {}", tensor.shard))?;
        let end = tensor
            .file_offset
            .checked_add(tensor.bytes)
            .context("source tensor byte range overflow")?;
        let data = mmap
            .get(tensor.file_offset..end)
            .with_context(|| format!("source tensor exceeds its shard: {}", tensor.name))?;
        let view = TensorView::new(tensor.dtype, tensor.shape.clone(), data)
            .with_context(|| format!("invalid source raw tensor view: {}", tensor.name))?;
        views.push((tensor.name.as_str(), view));
    }
    let destination = staging.join(&stage.destination_shard);
    anyhow::ensure!(
        !destination.exists(),
        "exact-stage destination file already exists in private staging"
    );
    serialize_to_file(
        views.iter().map(|(name, view)| (*name, view)),
        None,
        &destination,
    )
    .with_context(|| format!("failed to write exact stage {}", stage.ordinal))?;
    File::open(&destination)
        .with_context(|| format!("failed to reopen exact stage {}", stage.ordinal))?
        .sync_all()
        .with_context(|| format!("failed to synchronize exact stage {}", stage.ordinal))?;
    anyhow::ensure!(
        fs::metadata(&destination)?.len() == stage.destination_file_bytes,
        "serialized stage {} has an unexpected file size",
        stage.ordinal
    );
    verify_stage_payloads(&destination, &raw, &source_mmaps, verified)
        .with_context(|| format!("failed to verify exact stage {}", stage.ordinal))
}

fn open_source_mmaps(source: &Path, shard_names: &[String]) -> Result<BTreeMap<String, Mmap>> {
    let mut result = BTreeMap::new();
    for shard in shard_names {
        let path = source.join(shard);
        ensure_regular_nonsymlink(&path, "source shard")?;
        let file = File::open(&path)
            .with_context(|| format!("failed to open source shard {}", path.display()))?;
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("failed to mmap source shard {}", path.display()))?;
        result.insert(shard.clone(), mmap);
    }
    Ok(result)
}

fn verify_stage_payloads(
    destination: &Path,
    raw: &[RawTensorMetadata],
    source_mmaps: &BTreeMap<String, Mmap>,
    verified: &mut BTreeMap<String, H3VerifiedStageTensor>,
) -> Result<()> {
    let file = File::open(destination)
        .with_context(|| format!("failed to open destination stage {}", destination.display()))?;
    let mmap = unsafe { MmapOptions::new().map(&file) }
        .with_context(|| format!("failed to mmap destination stage {}", destination.display()))?;
    let tensors = SafeTensors::deserialize(&mmap)
        .with_context(|| format!("invalid destination stage {}", destination.display()))?;
    let expected = raw
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<BTreeSet<_>>();
    let actual = tensors.names().into_iter().collect::<BTreeSet<_>>();
    anyhow::ensure!(
        expected == actual,
        "destination stage tensor names differ from the execution stage"
    );
    for source in raw {
        let destination = tensors.tensor(&source.name)?;
        anyhow::ensure!(
            destination.dtype() == source.dtype
                && destination.shape() == source.shape
                && destination.data().len() == source.bytes,
            "destination tensor dtype/shape/length differs: {}",
            source.name
        );
        let source_mmap = source_mmaps
            .get(&source.shard)
            .with_context(|| format!("source mmap is missing {}", source.shard))?;
        let end = source
            .file_offset
            .checked_add(source.bytes)
            .context("source tensor range overflow")?;
        let source_bytes = source_mmap
            .get(source.file_offset..end)
            .with_context(|| format!("source tensor exceeds its shard: {}", source.name))?;
        anyhow::ensure!(
            source_bytes == destination.data(),
            "destination tensor payload differs byte-for-byte: {}",
            source.name
        );
        anyhow::ensure!(
            verified
                .insert(
                    source.name.clone(),
                    H3VerifiedStageTensor {
                        name: source.name.clone(),
                        payload_bytes: source.bytes as u64,
                    },
                )
                .is_none(),
            "tensor was written to more than one exact stage: {}",
            source.name
        );
    }
    Ok(())
}

fn validate_source_component(source: &Path, released: bool) -> Result<PathBuf> {
    let metadata = fs::symlink_metadata(source)
        .with_context(|| format!("failed to inspect source component {}", source.display()))?;
    anyhow::ensure!(
        metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
        "source component must be a non-symlink directory"
    );
    let source = fs::canonicalize(source).with_context(|| {
        format!(
            "failed to canonicalize source component {}",
            source.display()
        )
    })?;
    if source
        .file_name()
        .is_some_and(|name| name == "transformer_ref")
    {
        bail!("exact-stage conversion rejects transformer_ref; only transformer Base is supported");
    }
    if released {
        anyhow::ensure!(
            source.file_name().is_some_and(|name| name == "transformer"),
            "exact-stage conversion only accepts the released Base transformer component"
        );
    }
    ensure_regular_nonsymlink(&source.join("config.json"), "source config")?;
    Ok(source)
}

fn validate_profile_config(
    config: &TransformerConfig,
    profile: ExpectedLayoutProfile,
) -> Result<()> {
    anyhow::ensure!(
        config.num_layers == profile.layers && config.num_refiner_layers == profile.refiner_layers,
        "transformer config does not match the requested exact-stage profile"
    );
    if profile.released_config {
        anyhow::ensure!(
            config.class_name == "MiniMaxH3Transformer3DModel"
                && config.num_attention_heads == 56
                && config.attention_head_dim == 128
                && config.hidden_size == 5_376
                && config.ffn_dim == 14_336
                && config.in_channels == 24
                && config.audio_in_channels == 32
                && config.patch_size == [1, 2, 2]
                && config.text_dim == 5_120
                && config.freq_dim == 256
                && config.time_embed_hidden_dim == 5_376
                && config.time_embed_dim == 2_688
                && config.rope_freq_dim == 16
                && config.rope_theta.to_bits() == 10_000f64.to_bits()
                && config.norm_eps.to_bits() == 0.00001f64.to_bits()
                && config.qk_norm_eps.to_bits() == 0.00001f64.to_bits()
                && config.final_norm_eps.to_bits() == 0.00001f64.to_bits(),
            "transformer config is not the released H3 Base profile"
        );
    }
    Ok(())
}

fn validate_indexed_fingerprint(
    identity: &WeakModelIdentity,
    expected_shards: usize,
) -> Result<()> {
    anyhow::ensure!(
        identity.shards.len() == expected_shards,
        "source fingerprint has {} shards, expected {expected_shards}",
        identity.shards.len()
    );
    Ok(())
}

fn checked_new_destination(source: &Path, destination: &Path) -> Result<PathBuf> {
    anyhow::ensure!(
        !destination.as_os_str().is_empty() && !destination.exists() && !destination.is_symlink(),
        "exact-stage destination must be a new path"
    );
    let name = destination
        .file_name()
        .context("exact-stage destination must have a final path component")?;
    let parent = destination
        .parent()
        .context("exact-stage destination has no parent")?;
    let parent = fs::canonicalize(parent).with_context(|| {
        format!(
            "failed to canonicalize exact-stage destination parent {}",
            parent.display()
        )
    })?;
    let destination = parent.join(name);
    let model_root = if source.file_name().is_some_and(|name| name == "transformer") {
        source
            .parent()
            .context("Base transformer has no model root")?
    } else {
        source
    };
    anyhow::ensure!(
        !destination.starts_with(model_root),
        "exact-stage destination must be outside the source model"
    );
    Ok(destination)
}

#[cfg(unix)]
fn ensure_same_filesystem(source: &Path, destination_parent: &Path) -> Result<()> {
    use std::os::unix::fs::MetadataExt as _;
    anyhow::ensure!(
        fs::metadata(source)?.dev() == fs::metadata(destination_parent)?.dev(),
        "exact-stage source and destination must be on the same filesystem"
    );
    Ok(())
}

#[cfg(not(unix))]
fn ensure_same_filesystem(_source: &Path, _destination_parent: &Path) -> Result<()> {
    bail!("exact-stage same-filesystem verification is unsupported on this platform")
}

fn ensure_regular_nonsymlink(path: &Path, label: &str) -> Result<()> {
    let metadata =
        fs::symlink_metadata(path).with_context(|| format!("failed to inspect {label}"))?;
    anyhow::ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "{label} must be a regular non-symlink file: {}",
        path.display()
    );
    Ok(())
}

fn copy_regular_file_new(source: &Path, destination: &Path) -> Result<()> {
    ensure_regular_nonsymlink(source, "source config")?;
    let mut source_file = File::open(source)
        .with_context(|| format!("failed to open source config {}", source.display()))?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    let mut destination_file = options.open(destination).with_context(|| {
        format!(
            "failed to create destination config {}",
            destination.display()
        )
    })?;
    std::io::copy(&mut source_file, &mut destination_file)
        .context("failed to copy source config bytes")?;
    destination_file
        .sync_all()
        .context("failed to synchronize destination config")?;
    let mut source_bytes = Vec::new();
    let mut destination_bytes = Vec::new();
    File::open(source)?.read_to_end(&mut source_bytes)?;
    File::open(destination)?.read_to_end(&mut destination_bytes)?;
    anyhow::ensure!(
        source_bytes == destination_bytes,
        "destination config differs byte-for-byte from source"
    );
    Ok(())
}

fn write_json_new(path: &Path, value: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec(value).context("failed to serialize checkpoint JSON")?;
    write_bytes_new(path, &bytes)
}

fn write_bytes_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    let mut file = options
        .open(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("failed to write {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("failed to synchronize {}", path.display()))
}

struct StagingDirectory {
    path: PathBuf,
    active: bool,
}

impl StagingDirectory {
    fn create(parent: &Path) -> Result<Self> {
        let sequence = STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(
            ".ff-exact-stage-staging-{}-{sequence}",
            std::process::id()
        ));
        create_private_directory(&path, "exact-stage private staging")?;
        Ok(Self { path, active: true })
    }

    fn cleanup(&mut self) -> Result<()> {
        if self.active {
            fs::remove_dir_all(&self.path).with_context(|| {
                format!(
                    "failed to clean exact-stage private staging {}",
                    self.path.display()
                )
            })?;
            self.active = false;
        }
        Ok(())
    }

    fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for StagingDirectory {
    fn drop(&mut self) {
        if self.active {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

#[cfg(target_os = "linux")]
fn rename_directory_noreplace(source: &Path, destination: &Path) -> Result<()> {
    use std::{ffi::CString, os::unix::ffi::OsStrExt as _};

    const AT_FDCWD: i32 = -100;
    const RENAME_NOREPLACE: u32 = 1;
    unsafe extern "C" {
        fn renameat2(
            olddirfd: i32,
            oldpath: *const std::os::raw::c_char,
            newdirfd: i32,
            newpath: *const std::os::raw::c_char,
            flags: u32,
        ) -> i32;
    }

    let destination_display = destination.display().to_string();
    let source = CString::new(source.as_os_str().as_bytes())
        .context("exact-stage staging path contains a NUL byte")?;
    let destination = CString::new(destination.as_os_str().as_bytes())
        .context("exact-stage destination path contains a NUL byte")?;
    let result = unsafe {
        renameat2(
            AT_FDCWD,
            source.as_ptr(),
            AT_FDCWD,
            destination.as_ptr(),
            RENAME_NOREPLACE,
        )
    };
    if result == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error()).with_context(|| {
            format!(
                "failed to atomically publish exact-stage directory without replacing {destination_display}"
            )
        })
    }
}

#[cfg(not(target_os = "linux"))]
fn rename_directory_noreplace(_source: &Path, _destination: &Path) -> Result<()> {
    bail!("atomic no-replace exact-stage publication requires Linux renameat2")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_source(root: &Path) -> (PathBuf, ExpectedLayoutProfile) {
        let source = root.join("model/transformer");
        fs::create_dir_all(&source).unwrap();
        let names = [
            "context_embedder.weight",
            "token_refiner.refiner_blocks.0.norm1.weight",
            "token_refiner.refiner_blocks.0.norm2.weight",
            "token_refiner.final_norm.weight",
            "time_embedder.linear_1.weight",
            "proj_in.weight",
            "transformer_blocks.0.adaln_proj.linear.weight",
            "transformer_blocks.0.norm1.weight",
            "transformer_blocks.0.norm2.weight",
            "norm_out.norm.weight",
        ];
        let payload = [0u8, 0x80, 0xc1, 0x7f];
        let views = names
            .iter()
            .map(|name| {
                (
                    *name,
                    TensorView::new(safetensors::Dtype::BF16, vec![2], &payload).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &source.join("weights.safetensors")).unwrap();
        let weight_map = names
            .iter()
            .map(|name| (*name, "weights.safetensors"))
            .collect::<BTreeMap<_, _>>();
        write_json_new(
            &source.join("model.safetensors.index.json"),
            &serde_json::json!({
                "metadata": {"total_size": 40}, "weight_map": weight_map
            }),
        )
        .unwrap();
        write_json_new(
            &source.join("config.json"),
            &serde_json::json!({
                "_class_name": "MiniMaxH3Transformer3DModel", "num_attention_heads": 1,
                "attention_head_dim": 6, "hidden_size": 4, "num_layers": 1,
                "num_refiner_layers": 1, "ffn_dim": 5, "in_channels": 1,
                "audio_in_channels": 1, "patch_size": [1,1,1], "text_dim": 2,
                "freq_dim": 2, "time_embed_hidden_dim": 2, "time_embed_dim": 2,
                "rope_freq_dim": 1, "rope_theta": 10000., "norm_eps": 1e-5,
                "qk_norm_eps": 1e-5, "final_norm_eps": 1e-5
            }),
        )
        .unwrap();
        (
            source,
            ExpectedLayoutProfile {
                layers: 1,
                refiner_layers: 1,
                tensors: 10,
                shards: 1,
                payload_bytes: 40,
                stages: 10,
                released_config: false,
            },
        )
    }

    #[cfg(unix)]
    #[test]
    fn raw_repack_preserves_bf16_payloads_and_source_identity() {
        let root = tempfile::tempdir().unwrap();
        let (source, profile) = toy_source(root.path());
        let before = WeakModelIdentity::collect(&source).unwrap();
        let report = build_layout(&source, profile).unwrap();
        assert_eq!(report.total_payload_bytes, 40);
        let destination = root.path().join("repacked");
        let manifest = repack_with_profile(&source, &destination, profile).unwrap();
        assert_eq!(manifest.source_fingerprint, before);
        assert_eq!(WeakModelIdentity::collect(&source).unwrap(), before);
        let original = fs::read(source.join("weights.safetensors")).unwrap();
        let original = SafeTensors::deserialize(&original).unwrap();
        for stage in &report.stages {
            let bytes = fs::read(destination.join(&stage.destination_shard)).unwrap();
            assert_eq!(bytes.len() as u64, stage.destination_file_bytes);
            let actual = SafeTensors::deserialize(&bytes).unwrap();
            for tensor in &stage.tensors {
                let expected = original.tensor(&tensor.name).unwrap();
                let value = actual.tensor(&tensor.name).unwrap();
                assert_eq!(value.dtype(), expected.dtype());
                assert_eq!(value.shape(), expected.shape());
                assert_eq!(value.data(), expected.data());
            }
        }
        assert!(destination.join(H3_EXACT_STAGE_MANIFEST_FILE).is_file());
        assert!(repack_with_profile(&source, &destination, profile).is_err());
        assert_eq!(
            manifest.canonical_json().unwrap(),
            fs::read(destination.join(H3_EXACT_STAGE_MANIFEST_FILE)).unwrap()
        );
    }

    #[cfg(unix)]
    #[test]
    fn destination_must_be_outside_the_whole_original_model() {
        let root = tempfile::tempdir().unwrap();
        let (source, _) = toy_source(root.path());
        for path in [
            source.join("repacked"),
            source.parent().unwrap().join("repacked"),
        ] {
            assert!(
                checked_new_destination(&source, &path)
                    .unwrap_err()
                    .to_string()
                    .contains("outside the source model")
            );
            assert!(!path.exists());
        }
    }

    #[cfg(unix)]
    #[test]
    fn destination_rejects_dangling_symlinks_and_parent_aliases() {
        use std::os::unix::fs::symlink;
        let root = tempfile::tempdir().unwrap();
        let (source, _) = toy_source(root.path());
        let dangling = root.path().join("dangling");
        symlink(root.path().join("absent"), &dangling).unwrap();
        assert!(checked_new_destination(&source, &dangling).is_err());
        let alias = root.path().join("alias");
        symlink(source.parent().unwrap(), &alias).unwrap();
        assert!(checked_new_destination(&source, &alias.join("repacked")).is_err());
    }

    #[test]
    fn invalid_inventory_fails_without_publishing_or_leaving_staging() {
        let root = tempfile::tempdir().unwrap();
        let (source, mut profile) = toy_source(root.path());
        profile.tensors += 1;
        let output = root.path().join("repacked");
        assert!(repack_with_profile(&source, &output, profile).is_err());
        assert!(!output.exists());
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 1);
    }

    #[test]
    fn payload_eviction_does_not_reparse_cached_headers() {
        let root = tempfile::tempdir().unwrap();
        let (source, _) = toy_source(root.path());
        let weights = ModelWeights::open(&source, WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        weights.verify().unwrap();
        let before = weights.cache_stats();
        assert_eq!(before.header_parses, 1);
        assert_eq!(before.resident_shards, 0);
        weights.verify().unwrap();
        assert_eq!(weights.cache_stats().header_parses, 1);
        assert_eq!(weights.access_stats().device_tensor_materializations, 0);
    }
}
