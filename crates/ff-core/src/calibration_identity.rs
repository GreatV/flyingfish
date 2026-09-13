use crate::artifact::{FileStat, identifies_same_file};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, Metadata},
    io::{BufReader, ErrorKind, Read},
    path::{Component, Path, PathBuf},
    time::UNIX_EPOCH,
};

pub const CALIBRATION_IDENTITY_SCHEMA_VERSION: u32 = 1;

const INDEX_NAMES: [&str; 2] = [
    "diffusion_pytorch_model.safetensors.index.json",
    "model.safetensors.index.json",
];
const CONFIG_NAME: &str = "config.json";
const SINGLE_FILE_NAMES: [&str; 2] = ["diffusion_pytorch_model.safetensors", "model.safetensors"];
const HASH_BUFFER_BYTES: usize = 1024 * 1024;
const MAX_MODEL_INDEX_BYTES: u64 = 64 * 1024 * 1024;
const MAX_IDENTITY_JSON_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileStamp {
    pub bytes: u64,
    pub modified_ns: u64,
}

impl FileStamp {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.modified_ns > 0, "file stamp has no modification time");
        Ok(())
    }
}

/// Stamp a file by what the filesystem already knows: its size and last
/// modification time.
///
/// Reading the bytes to hash them would cost the whole file in disk bandwidth
/// and CPU for a property no decision here depends on. The stamp changes
/// whenever a write does, which is what a local cache key needs.
pub fn stamp_file(path: &Path) -> Result<FileStamp> {
    let path_before = FileStat::of_path(path)
        .with_context(|| format!("failed to inspect identity input {}", path.display()))?;
    anyhow::ensure!(
        path_before.file_type().is_file(),
        "identity input is not a regular non-symlink file: {}",
        path.display()
    );
    let file = File::open(path)
        .with_context(|| format!("failed to open identity input {}", path.display()))?;
    let stat = FileStat::of_file(&file)
        .with_context(|| format!("failed to stat identity input {}", path.display()))?;
    anyhow::ensure!(
        stat.is_file() && identifies_same_file(&path_before, &stat),
        "identity input is not a regular file: {}",
        path.display()
    );
    Ok(FileStamp {
        bytes: stat.len(),
        modified_ns: modified_ns(stat.metadata())?,
    })
}

#[derive(Clone, Debug, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BinaryIdentity {
    pub schema_version: u32,
    pub package_name: String,
    pub package_version: String,
    /// Legacy field, ignored when matching builds. New records omit it.
    #[serde(
        default = "absent_executable_digest",
        skip_serializing_if = "executable_digest_absent"
    )]
    pub executable: FileStamp,
    pub compiled_features: Vec<String>,
}

fn absent_executable_digest() -> FileStamp {
    FileStamp {
        bytes: 0,
        modified_ns: 0,
    }
}

fn executable_digest_absent(value: &FileStamp) -> bool {
    value.bytes == 0 && value.modified_ns == 0
}

impl PartialEq for BinaryIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.schema_version == other.schema_version
            && self.package_name == other.package_name
            && self.package_version == other.package_version
            && self.compiled_features == other.compiled_features
    }
}

impl BinaryIdentity {
    pub fn collect(
        _current_exe: &Path,
        package_name: &str,
        package_version: &str,
        features: &[&str],
    ) -> Result<Self> {
        let mut compiled_features = features
            .iter()
            .map(|feature| (*feature).to_owned())
            .collect::<Vec<_>>();
        compiled_features.sort_unstable();

        let identity = Self {
            schema_version: CALIBRATION_IDENTITY_SCHEMA_VERSION,
            package_name: package_name.to_owned(),
            package_version: package_version.to_owned(),
            executable: absent_executable_digest(),
            compiled_features,
        };
        identity.validate()?;
        Ok(identity)
    }

    pub fn validate(&self) -> Result<()> {
        validate_schema(self.schema_version, "binary identity")?;
        anyhow::ensure!(
            !self.package_name.is_empty(),
            "binary package name is empty"
        );
        anyhow::ensure!(
            !self.package_version.is_empty(),
            "binary package version is empty"
        );
        validate_sorted_unique_strings(&self.compiled_features, "compiled feature")?;
        Ok(())
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        from_json(bytes, "binary identity")
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>> {
        canonical_json(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelIdentityStrength {
    LocalMetadataManifest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NamedFileStamp {
    pub relative_path: String,
    pub bytes: u64,
    pub modified_ns: u64,
}

impl NamedFileStamp {
    fn from_stamp(relative_path: &str, stamp: FileStamp) -> Self {
        Self {
            relative_path: relative_path.to_owned(),
            bytes: stamp.bytes,
            modified_ns: stamp.modified_ns,
        }
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            is_safe_relative_path(&self.relative_path),
            "unsafe identity file path: {}",
            self.relative_path
        );
        anyhow::ensure!(
            self.bytes > 0,
            "identity file is empty: {}",
            self.relative_path
        );
        anyhow::ensure!(
            self.modified_ns > 0,
            "named file stamp has no modification time"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeakShardIdentity {
    pub relative_path: String,
    pub file_bytes: u64,
    pub modified_ns: u64,
}

impl WeakShardIdentity {
    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            is_safe_relative_path(&self.relative_path),
            "unsafe shard path in calibration identity: {}",
            self.relative_path
        );
        anyhow::ensure!(
            self.file_bytes > 0,
            "model shard is empty: {}",
            self.relative_path
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeakModelIdentity {
    pub schema_version: u32,
    pub strength: ModelIdentityStrength,
    pub cacheable: bool,
    pub canonical_component_path: String,
    pub config: NamedFileStamp,
    /// Absent for a single-file checkpoint, which has no index to name.
    #[serde(deserialize_with = "crate::required_option")]
    pub index: Option<NamedFileStamp>,
    pub indexed_checkpoint_bytes: u64,
    pub shards: Vec<WeakShardIdentity>,
}

impl WeakModelIdentity {
    pub fn collect(component: &Path) -> Result<Self> {
        let canonical_root = fs::canonicalize(component).with_context(|| {
            format!(
                "failed to canonicalize model component {}",
                component.display()
            )
        })?;
        anyhow::ensure!(
            canonical_root.is_dir(),
            "model component is not a directory: {}",
            canonical_root.display()
        );
        let canonical_component_path = canonical_root
            .to_str()
            .with_context(|| {
                format!(
                    "model component path is not valid UTF-8: {}",
                    canonical_root.display()
                )
            })?
            .to_owned();

        let config_path = checked_model_file(&canonical_root, Path::new(CONFIG_NAME))?;
        let config = NamedFileStamp::from_stamp(CONFIG_NAME, stamp_file(&config_path)?);

        let (index, indexed_checkpoint_bytes, shard_names) =
            match find_model_index(&canonical_root)? {
                Some((index_name, index_path)) => {
                    let index_bytes = read_file_bounded(
                        &index_path,
                        MAX_MODEL_INDEX_BYTES,
                        "model safetensors index",
                    )?;
                    let index_stamp = stamp_file(&index_path)?;
                    let parsed: SafetensorsIndex = serde_json::from_slice(&index_bytes)
                        .with_context(|| format!("invalid model index {}", index_path.display()))?;
                    anyhow::ensure!(
                        !parsed.weight_map.is_empty(),
                        "model index contains no tensors: {}",
                        index_path.display()
                    );
                    anyhow::ensure!(
                        parsed.metadata.total_size > 0,
                        "model index declares zero checkpoint bytes: {}",
                        index_path.display()
                    );
                    (
                        Some(NamedFileStamp::from_stamp(index_name, index_stamp)),
                        parsed.metadata.total_size,
                        parsed.weight_map.into_values().collect::<BTreeSet<_>>(),
                    )
                }
                None => {
                    let names = single_model_file(&canonical_root)?;
                    let bytes = names
                        .iter()
                        .map(|name| {
                            checked_model_file_metadata(&canonical_root, Path::new(name))
                                .map(|metadata| metadata.len())
                        })
                        .sum::<Result<u64>>()?;
                    (None, bytes, names)
                }
            };

        let mut shards = Vec::with_capacity(shard_names.len());
        for relative_path in shard_names {
            anyhow::ensure!(
                is_safe_relative_path(&relative_path),
                "unsafe shard path in model index: {relative_path}"
            );
            let metadata = checked_model_file_metadata(&canonical_root, Path::new(&relative_path))?;
            shards.push(WeakShardIdentity {
                relative_path,
                file_bytes: metadata.len(),
                modified_ns: modified_ns(&metadata)?,
            });
        }

        let identity = Self {
            schema_version: CALIBRATION_IDENTITY_SCHEMA_VERSION,
            strength: ModelIdentityStrength::LocalMetadataManifest,
            cacheable: false,
            canonical_component_path,
            config,
            index,
            indexed_checkpoint_bytes,
            shards,
        };
        identity.validate()?;
        Ok(identity)
    }

    pub fn validate(&self) -> Result<()> {
        validate_schema(self.schema_version, "model identity")?;
        anyhow::ensure!(
            self.strength == ModelIdentityStrength::LocalMetadataManifest,
            "unsupported model identity strength"
        );
        anyhow::ensure!(
            !self.cacheable,
            "local-metadata model identity cannot authorize cache reuse"
        );
        anyhow::ensure!(
            !self.canonical_component_path.is_empty(),
            "canonical model component path is empty"
        );
        anyhow::ensure!(
            Path::new(&self.canonical_component_path).is_absolute(),
            "canonical model component path is not absolute: {}",
            self.canonical_component_path
        );
        self.config.validate()?;
        anyhow::ensure!(
            self.config.relative_path == CONFIG_NAME,
            "model identity config path must be {CONFIG_NAME}"
        );
        if let Some(index) = &self.index {
            index.validate()?;
            anyhow::ensure!(
                INDEX_NAMES.contains(&index.relative_path.as_str()),
                "unsupported model index path: {}",
                index.relative_path
            );
        }
        anyhow::ensure!(
            self.indexed_checkpoint_bytes > 0,
            "indexed checkpoint byte count must be non-zero"
        );
        anyhow::ensure!(!self.shards.is_empty(), "model shard manifest is empty");
        let mut previous: Option<&str> = None;
        for shard in &self.shards {
            shard.validate()?;
            if let Some(previous) = previous {
                anyhow::ensure!(
                    previous < shard.relative_path.as_str(),
                    "model shard manifest is not strictly sorted and unique"
                );
            }
            previous = Some(&shard.relative_path);
        }
        Ok(())
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        from_json(bytes, "model identity")
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>> {
        canonical_json(self)
    }

    /// Total size of the shard files this component's index names.
    pub fn weight_file_bytes(&self) -> u64 {
        self.shards.iter().map(|shard| shard.file_bytes).sum()
    }

    /// How many shard files this component's index names.
    pub fn weight_file_count(&self) -> usize {
        self.shards.len()
    }

    pub fn stable_key_material(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let key = WeakModelStableKey {
            schema_version: self.schema_version,
            strength: self.strength,
            cacheable: self.cacheable,
            config: &self.config,
            index: &self.index,
            indexed_checkpoint_bytes: self.indexed_checkpoint_bytes,
            shards: self
                .shards
                .iter()
                .map(|shard| WeakShardStableKey {
                    relative_path: &shard.relative_path,
                    file_bytes: shard.file_bytes,
                })
                .collect(),
        };
        serde_json::to_vec(&key).context("failed to serialize stable model identity key material")
    }
}

#[derive(Serialize)]
struct WeakModelStableKey<'a> {
    schema_version: u32,
    strength: ModelIdentityStrength,
    cacheable: bool,
    config: &'a NamedFileStamp,
    index: &'a Option<NamedFileStamp>,
    indexed_checkpoint_bytes: u64,
    shards: Vec<WeakShardStableKey<'a>>,
}

#[derive(Serialize)]
struct WeakShardStableKey<'a> {
    relative_path: &'a str,
    file_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputIdentity {
    pub schema_version: u32,
    pub bytes: u64,
    pub modified_ns: u64,
}

impl InputIdentity {
    pub fn collect(path: &Path) -> Result<Self> {
        let stamp = stamp_file(path)?;
        let identity = Self {
            schema_version: CALIBRATION_IDENTITY_SCHEMA_VERSION,
            bytes: stamp.bytes,
            modified_ns: stamp.modified_ns,
        };
        identity.validate()?;
        Ok(identity)
    }

    pub fn validate(&self) -> Result<()> {
        validate_schema(self.schema_version, "input identity")?;
        anyhow::ensure!(self.bytes > 0, "calibration input is empty");
        anyhow::ensure!(
            self.modified_ns > 0,
            "calibration input has no modification time"
        );
        Ok(())
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        from_json(bytes, "input identity")
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>> {
        canonical_json(self)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SafetensorsIndex {
    metadata: IndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IndexMetadata {
    total_size: u64,
}

fn checked_model_file(root: &Path, relative: &Path) -> Result<PathBuf> {
    checked_model_file_metadata(root, relative)?;
    Ok(root.join(relative))
}

fn read_file_bounded(path: &Path, max_bytes: u64, label: &str) -> Result<Vec<u8>> {
    let file =
        File::open(path).with_context(|| format!("failed to open {label} {}", path.display()))?;
    let before = file
        .metadata()
        .with_context(|| format!("failed to stat {label} {}", path.display()))?;
    anyhow::ensure!(
        before.is_file(),
        "{label} is not a regular file: {}",
        path.display()
    );
    anyhow::ensure!(
        before.len() <= max_bytes,
        "{label} {} is {} bytes, exceeding the {max_bytes}-byte limit",
        path.display(),
        before.len()
    );
    let before_modified = before.modified().with_context(|| {
        format!(
            "failed to read modified time for {label} {}",
            path.display()
        )
    })?;
    let expected_bytes = usize::try_from(before.len())
        .with_context(|| format!("{label} size exceeds usize: {}", path.display()))?;
    let read_limit = max_bytes
        .checked_add(1)
        .context("bounded file-read limit overflow")?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(expected_bytes)
        .with_context(|| format!("failed to reserve {label} buffer for {}", path.display()))?;
    let mut reader = BufReader::with_capacity(HASH_BUFFER_BYTES, file).take(read_limit);
    reader
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read {label} {}", path.display()))?;
    anyhow::ensure!(
        u64::try_from(bytes.len()).context("bounded file-read length exceeds u64")? <= max_bytes,
        "{label} {} grew beyond the {max_bytes}-byte limit while reading",
        path.display()
    );
    let after = reader
        .into_inner()
        .into_inner()
        .metadata()
        .with_context(|| format!("failed to restat {label} {}", path.display()))?;
    let after_modified = after.modified().with_context(|| {
        format!(
            "failed to reread modified time for {label} {}",
            path.display()
        )
    })?;
    anyhow::ensure!(
        bytes.len() == expected_bytes && after.len() == before.len(),
        "{label} {} changed length while reading (before {}, read {}, after {})",
        path.display(),
        before.len(),
        bytes.len(),
        after.len()
    );
    anyhow::ensure!(
        after_modified == before_modified,
        "{label} modified while reading {}",
        path.display()
    );
    Ok(bytes)
}

/// The one checkpoint file an unindexed component holds.
///
/// A component without an index must contain exactly one supported single-file
/// checkpoint; two would leave which one carries the weights undecided.
fn single_model_file(root: &Path) -> Result<BTreeSet<String>> {
    let mut found = BTreeSet::new();
    for name in SINGLE_FILE_NAMES {
        match fs::symlink_metadata(root.join(name)) {
            Ok(_) => {
                found.insert(name.to_owned());
            }
            Err(error) if error.kind() == ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to stat model file {}", root.join(name).display())
                });
            }
        }
    }
    anyhow::ensure!(
        found.len() == 1,
        "model component {} has no safetensors index ({}) and does not hold exactly one of {}",
        root.display(),
        INDEX_NAMES.join(" or "),
        SINGLE_FILE_NAMES.join(" or ")
    );
    Ok(found)
}

/// Locate the component's index, distinguishing "there is none" from "there is
/// one and it is unusable".
///
/// Only the first answer may fall back to a single-file layout. Collapsing the
/// two would let an unsafe index path silently downgrade to a different file.
fn find_model_index(root: &Path) -> Result<Option<(&'static str, PathBuf)>> {
    for name in INDEX_NAMES {
        let candidate = root.join(name);
        match fs::symlink_metadata(&candidate) {
            Ok(_) => {
                return Ok(Some((name, checked_model_file(root, Path::new(name))?)));
            }
            Err(error) if error.kind() == ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to stat model index {}", candidate.display())
                });
            }
        }
    }
    Ok(None)
}

fn checked_model_file_metadata(root: &Path, relative: &Path) -> Result<Metadata> {
    anyhow::ensure!(
        is_safe_relative_path_os(relative),
        "unsafe model identity path: {}",
        relative.display()
    );
    let mut path = root.to_path_buf();
    let components = relative.components().collect::<Vec<_>>();
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(name) = component else {
            bail!("unsafe model identity path: {}", relative.display());
        };
        path.push(name);
        let metadata = fs::symlink_metadata(&path)
            .with_context(|| format!("failed to stat model identity path {}", path.display()))?;
        anyhow::ensure!(
            !metadata.file_type().is_symlink(),
            "model identity path contains a symlink: {}",
            path.display()
        );
        if index + 1 == components.len() {
            anyhow::ensure!(
                metadata.is_file(),
                "model identity path is not a regular file: {}",
                path.display()
            );
            return Ok(metadata);
        }
        anyhow::ensure!(
            metadata.is_dir(),
            "model identity path component is not a directory: {}",
            path.display()
        );
    }
    bail!("model identity path is empty")
}

fn is_safe_relative_path(value: &str) -> bool {
    !value.is_empty() && is_safe_relative_path_os(Path::new(value))
}

fn is_safe_relative_path_os(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn modified_ns(metadata: &Metadata) -> Result<u64> {
    let modified = metadata
        .modified()
        .context("failed to read model shard modified time")?;
    let elapsed = modified
        .duration_since(UNIX_EPOCH)
        .context("model shard modified time predates Unix epoch")?;
    u64::try_from(elapsed.as_nanos()).context("model shard modified time exceeds u64 nanoseconds")
}

fn validate_schema(schema_version: u32, kind: &str) -> Result<()> {
    anyhow::ensure!(
        schema_version == CALIBRATION_IDENTITY_SCHEMA_VERSION,
        "unsupported {kind} schema {schema_version}; this build supports schema {}",
        CALIBRATION_IDENTITY_SCHEMA_VERSION
    );
    Ok(())
}

fn validate_sorted_unique_strings(values: &[String], kind: &str) -> Result<()> {
    for value in values {
        anyhow::ensure!(!value.is_empty(), "{kind} name is empty");
    }
    anyhow::ensure!(
        values.windows(2).all(|pair| pair[0] < pair[1]),
        "{kind} names are not strictly sorted and unique"
    );
    Ok(())
}

fn from_json<T>(bytes: &[u8], kind: &str) -> Result<T>
where
    T: DeserializeOwned + IdentityRecord,
{
    from_json_with_limit(bytes, kind, MAX_IDENTITY_JSON_BYTES)
}

fn from_json_with_limit<T>(bytes: &[u8], kind: &str, max_bytes: usize) -> Result<T>
where
    T: DeserializeOwned + IdentityRecord,
{
    anyhow::ensure!(
        bytes.len() <= max_bytes,
        "{kind} JSON is {} bytes, exceeding the {max_bytes}-byte limit",
        bytes.len()
    );
    let record: T =
        serde_json::from_slice(bytes).with_context(|| format!("invalid {kind} JSON"))?;
    record.validate_identity()?;
    Ok(record)
}

fn canonical_json<T>(value: &T) -> Result<Vec<u8>>
where
    T: Serialize + IdentityRecord,
{
    value.validate_identity()?;
    serde_json::to_vec(value).context("failed to serialize calibration identity")
}

trait IdentityRecord {
    fn validate_identity(&self) -> Result<()>;
}

impl IdentityRecord for BinaryIdentity {
    fn validate_identity(&self) -> Result<()> {
        self.validate()
    }
}

impl IdentityRecord for WeakModelIdentity {
    fn validate_identity(&self) -> Result<()> {
        self.validate()
    }
}

impl IdentityRecord for InputIdentity {
    fn validate_identity(&self) -> Result<()> {
        self.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn model_fixture(reverse_map_order: bool) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join(CONFIG_NAME), br#"{"hidden_size":8}"#).unwrap();
        fs::write(directory.path().join("a.safetensors"), b"not-read-a").unwrap();
        fs::write(directory.path().join("b.safetensors"), b"not-read-bb").unwrap();
        let weight_map = if reverse_map_order {
            json!({"z": "b.safetensors", "a": "a.safetensors"})
        } else {
            json!({"a": "a.safetensors", "z": "b.safetensors"})
        };
        fs::write(
            directory.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({
                "metadata": {"total_size": 1234},
                "weight_map": weight_map
            }))
            .unwrap(),
        )
        .unwrap();
        directory
    }

    #[test]
    fn stamping_reports_the_checked_size_without_reading_the_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("input.bin");
        fs::write(&path, b"abc").unwrap();
        let stamp = stamp_file(&path).unwrap();
        assert_eq!(stamp.bytes, 3);
        assert!(stamp.modified_ns > 0);
        stamp.validate().unwrap();

        assert!(stamp_file(directory.path()).is_err());
    }

    #[test]
    fn bounded_file_read_rejects_oversize_before_allocating_the_limit() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("small-index.json");
        fs::write(&path, b"123456789").unwrap();
        let error = read_file_bounded(&path, 8, "test index")
            .unwrap_err()
            .to_string();
        assert!(error.contains("exceeding the 8-byte limit"), "{error}");
    }

    #[test]
    fn binary_identity_round_trips_strictly() {
        let executable = std::env::current_exe().unwrap();
        let identity =
            BinaryIdentity::collect(&executable, "example-app", "1.2.3", &["cuda"]).unwrap();
        assert_eq!(identity.package_name, "example-app");
        assert_eq!(identity.package_version, "1.2.3");
        assert_eq!(identity.compiled_features, ["cuda"]);
        assert_eq!(
            BinaryIdentity::from_json(&identity.canonical_json().unwrap()).unwrap(),
            identity
        );
    }

    #[test]
    fn model_manifest_is_sorted_and_does_not_read_shard_payloads() {
        let directory = model_fixture(true);
        let identity = WeakModelIdentity::collect(directory.path()).unwrap();
        assert_eq!(
            identity.strength,
            ModelIdentityStrength::LocalMetadataManifest
        );
        assert!(!identity.cacheable);
        assert_eq!(identity.indexed_checkpoint_bytes, 1234);
        assert_eq!(
            identity
                .shards
                .iter()
                .map(|shard| shard.relative_path.as_str())
                .collect::<Vec<_>>(),
            ["a.safetensors", "b.safetensors"]
        );
        assert_eq!(
            WeakModelIdentity::from_json(&identity.canonical_json().unwrap()).unwrap(),
            identity
        );
    }

    #[test]
    fn stable_model_key_excludes_diagnostic_path_and_mtime() {
        let directory = model_fixture(false);
        let identity = WeakModelIdentity::collect(directory.path()).unwrap();
        let mut relocated = identity.clone();
        relocated.canonical_component_path = std::env::temp_dir()
            .join("flyingfish-relocated-test-component")
            .to_string_lossy()
            .into_owned();
        for shard in &mut relocated.shards {
            shard.modified_ns = shard.modified_ns.saturating_add(17);
        }
        relocated.validate().unwrap();
        assert_ne!(
            identity.canonical_json().unwrap(),
            relocated.canonical_json().unwrap()
        );
        assert_eq!(
            identity.stable_key_material().unwrap(),
            relocated.stable_key_material().unwrap()
        );
    }

    #[test]
    /// A stamp separates writes the filesystem clock can separate.
    ///
    /// Two same-length writes inside one timestamp tick produce the same
    /// stamp; that is the price of not reading the bytes, and it is the same
    /// bound every other identity in this crate now accepts. A real checkpoint
    /// swap is many ticks apart, so this bounds a race, not a workflow.
    fn input_identity_separates_writes_the_filesystem_clock_separates() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("input.safetensors");
        fs::write(&path, b"first").unwrap();
        let before = InputIdentity::collect(&path).unwrap();

        fs::write(&path, b"much later and longer").unwrap();
        let resized = InputIdentity::collect(&path).unwrap();
        assert_ne!(before.bytes, resized.bytes);
        assert_ne!(before, resized);

        std::thread::sleep(std::time::Duration::from_millis(20));
        fs::write(&path, b"later").unwrap();
        let rewritten = InputIdentity::collect(&path).unwrap();
        assert_eq!(before.bytes, rewritten.bytes);
        assert_ne!(before.modified_ns, rewritten.modified_ns);
        assert_ne!(before, rewritten);
    }

    #[test]
    fn empty_input_identity_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("empty.bin");
        fs::write(&path, []).unwrap();
        let error = InputIdentity::collect(&path).unwrap_err().to_string();
        assert!(error.contains("empty"), "{error}");
    }

    #[test]
    fn unknown_fields_and_tampered_invariants_are_rejected() {
        let directory = model_fixture(false);
        let identity = WeakModelIdentity::collect(directory.path()).unwrap();
        let mut unknown: serde_json::Value =
            serde_json::from_slice(&identity.canonical_json().unwrap()).unwrap();
        unknown["future"] = json!(true);
        assert!(WeakModelIdentity::from_json(&serde_json::to_vec(&unknown).unwrap()).is_err());

        let mut cacheable = identity.clone();
        cacheable.cacheable = true;
        assert!(cacheable.validate().is_err());

        let mut stampless = InputIdentity {
            schema_version: CALIBRATION_IDENTITY_SCHEMA_VERSION,
            bytes: 1,
            modified_ns: 0,
        };
        assert!(stampless.validate().is_err());
        stampless.modified_ns = 1;
        stampless.schema_version += 1;
        assert!(stampless.validate().is_err());
    }

    #[test]
    fn identity_json_rejects_oversize_and_duplicate_fields_before_validation() {
        let identity = InputIdentity {
            schema_version: CALIBRATION_IDENTITY_SCHEMA_VERSION,
            bytes: 8,
            modified_ns: 1,
        };
        let json = identity.canonical_json().unwrap();
        let error = from_json_with_limit::<InputIdentity>(
            &json,
            "input identity",
            json.len().saturating_sub(1),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("exceeding"), "{error}");

        let duplicate = format!(
            r#"{{"schema_version":1,"bytes":8,"bytes":9,"modified_ns":{}}}"#,
            identity.modified_ns
        );
        let error = format!(
            "{:#}",
            InputIdentity::from_json(duplicate.as_bytes()).unwrap_err()
        );
        assert!(error.contains("duplicate field"), "{error}");
    }

    #[test]
    fn model_manifest_validation_rejects_reordering() {
        let directory = model_fixture(false);
        let mut identity = WeakModelIdentity::collect(directory.path()).unwrap();
        identity.shards.reverse();
        assert!(identity.validate().is_err());
    }

    #[test]
    fn unsafe_shard_path_is_rejected() {
        let directory = model_fixture(false);
        fs::write(
            directory.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({
                "metadata": {"total_size": 1},
                "weight_map": {"a": "../outside.safetensors"}
            }))
            .unwrap(),
        )
        .unwrap();
        let error = WeakModelIdentity::collect(directory.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("unsafe shard path"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn shard_symlink_is_rejected() {
        use std::os::unix::fs::symlink;

        let directory = model_fixture(false);
        let target = directory.path().join("real.safetensors");
        fs::write(&target, b"target").unwrap();
        let link = directory.path().join("linked.safetensors");
        symlink(&target, &link).unwrap();
        fs::write(
            directory.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({
                "metadata": {"total_size": 1},
                "weight_map": {"a": "linked.safetensors"}
            }))
            .unwrap(),
        )
        .unwrap();
        let error = WeakModelIdentity::collect(directory.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("symlink"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn unsafe_preferred_index_does_not_fall_back() {
        use std::os::unix::fs::symlink;

        let directory = model_fixture(false);
        let outside = tempfile::NamedTempFile::new().unwrap();
        symlink(
            outside.path(),
            directory
                .path()
                .join("diffusion_pytorch_model.safetensors.index.json"),
        )
        .unwrap();
        let error = WeakModelIdentity::collect(directory.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("symlink"), "{error}");
    }
}
