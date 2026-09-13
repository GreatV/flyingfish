//! Staged, atomically published artifacts.
//!
//! A staging file is written, synchronized and only then hard-linked
//! into place, so a destination path never exists in a partial state and an
//! existing path is never replaced. Failures are typed by how far publication
//! got, because "not published" and "published but unverified" need different
//! recovery.

use super::identity::*;
use super::*;

pub(super) const STAGING_NAME_ATTEMPTS: usize = 1024;

/// A builder for the staging directory, private to this user where the
/// platform can express that.
///
/// Unix sets the mode at creation so the directory is never briefly readable.
/// Windows has no mode to set here and inherits the parent's ACL.
#[cfg(unix)]
fn private_directory_builder() -> fs::DirBuilder {
    use std::os::unix::fs::DirBuilderExt as _;
    let mut builder = fs::DirBuilder::new();
    builder.mode(0o700);
    builder
}

#[cfg(not(unix))]
fn private_directory_builder() -> fs::DirBuilder {
    fs::DirBuilder::new()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactDurability {
    AtomicVisibility,
    CrashDurable,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublishedArtifact {
    pub destination: PathBuf,
    pub durability: ArtifactDurability,
}

#[derive(Debug)]
pub enum ArtifactPublishError {
    NotPublished {
        destination: PathBuf,
        staging_path: PathBuf,
        source: anyhow::Error,
    },
    LinkedButIdentityUnverified {
        destination: PathBuf,
        staging_path: PathBuf,
        source: anyhow::Error,
    },
    PublishedButFinalizationFailed {
        published: PublishedArtifact,
        staging_path: PathBuf,
        source: anyhow::Error,
    },
}

impl ArtifactPublishError {
    pub fn destination(&self) -> &Path {
        match self {
            Self::NotPublished { destination, .. } => destination,
            Self::LinkedButIdentityUnverified { destination, .. } => destination,
            Self::PublishedButFinalizationFailed { published, .. } => &published.destination,
        }
    }

    pub fn published(&self) -> Option<&PublishedArtifact> {
        match self {
            Self::NotPublished { .. } | Self::LinkedButIdentityUnverified { .. } => None,
            Self::PublishedButFinalizationFailed { published, .. } => Some(published),
        }
    }

    pub fn destination_was_linked(&self) -> bool {
        !matches!(self, Self::NotPublished { .. })
    }

    pub fn staging_path(&self) -> &Path {
        match self {
            Self::NotPublished { staging_path, .. }
            | Self::LinkedButIdentityUnverified { staging_path, .. }
            | Self::PublishedButFinalizationFailed { staging_path, .. } => staging_path,
        }
    }

    fn source_error(&self) -> &anyhow::Error {
        match self {
            Self::NotPublished { source, .. }
            | Self::LinkedButIdentityUnverified { source, .. }
            | Self::PublishedButFinalizationFailed { source, .. } => source,
        }
    }
}

impl fmt::Display for ArtifactPublishError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotPublished {
                destination,
                source,
                ..
            } => write!(
                formatter,
                "artifact was not published to {}: {source:#}",
                destination.display()
            ),
            Self::LinkedButIdentityUnverified {
                destination,
                source,
                ..
            } => write!(
                formatter,
                "artifact was linked at {}, but its verified identity became uncertain: {source:#}",
                destination.display()
            ),
            Self::PublishedButFinalizationFailed {
                published, source, ..
            } => write!(
                formatter,
                "artifact was linked at {} with {:?} durability, but finalization failed: {source:#}",
                published.destination.display(),
                published.durability
            ),
        }
    }
}

impl std::error::Error for ArtifactPublishError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.source_error().as_ref())
    }
}

#[derive(Debug)]
pub struct ArtifactStaging {
    destination: PathBuf,
    parent: PathBuf,
    staging_parent: PathBuf,
    staging_path: PathBuf,
    staging_directory: PathBuf,
    producer_mode: ProducerMode,
    staging_file: Option<File>,
    identity: Option<FileIdentity>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ProducerMode {
    ManagedWriter,
    ExternalPath,
}

impl ArtifactStaging {
    pub fn new(destination: impl AsRef<Path>) -> Result<Self> {
        let staging = Self::new_for_path_producer(destination)?;
        Self::managed(staging)
    }

    pub fn new_with_staging_parent(
        destination: impl AsRef<Path>,
        staging_parent: impl AsRef<Path>,
    ) -> Result<Self> {
        let staging =
            Self::new_for_path_producer_inner(destination, Some(staging_parent.as_ref()))?;
        Self::managed(staging)
    }

    fn managed(mut staging: Self) -> Result<Self> {
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        let staging_file = options.open(&staging.staging_path).with_context(|| {
            format!(
                "failed to create artifact staging file {}",
                staging.staging_path.display()
            )
        })?;
        let metadata = FileStat::of_file(&staging_file).with_context(|| {
            format!(
                "failed to inspect new artifact staging file {}",
                staging.staging_path.display()
            )
        })?;
        staging.identity = Some(FileIdentity::from_stat(&metadata)?);
        staging.staging_file = Some(staging_file);
        staging.producer_mode = ProducerMode::ManagedWriter;
        Ok(staging)
    }

    pub fn new_for_path_producer(destination: impl AsRef<Path>) -> Result<Self> {
        Self::new_for_path_producer_inner(destination, None)
    }

    pub fn new_for_path_producer_with_staging_parent(
        destination: impl AsRef<Path>,
        staging_parent: impl AsRef<Path>,
    ) -> Result<Self> {
        Self::new_for_path_producer_inner(destination, Some(staging_parent.as_ref()))
    }

    fn new_for_path_producer_inner(
        destination: impl AsRef<Path>,
        staging_parent: Option<&Path>,
    ) -> Result<Self> {
        let destination = destination.as_ref();
        let file_name = destination
            .file_name()
            .context("artifact destination must name a file")?;
        let parent = destination
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let parent = fs::canonicalize(parent).with_context(|| {
            format!(
                "failed to resolve artifact destination parent {}",
                parent.display()
            )
        })?;
        anyhow::ensure!(
            parent.is_dir(),
            "artifact destination parent is not a directory: {}",
            parent.display()
        );
        let destination = parent.join(file_name);
        let staging_parent = match staging_parent {
            Some(staging_parent) => fs::canonicalize(staging_parent).with_context(|| {
                format!(
                    "failed to resolve artifact staging parent {}",
                    staging_parent.display()
                )
            })?,
            None => parent.clone(),
        };
        anyhow::ensure!(
            staging_parent.is_dir(),
            "artifact staging parent is not a directory: {}",
            staging_parent.display()
        );

        for _ in 0..STAGING_NAME_ATTEMPTS {
            let nonce = STAGING_NONCE.fetch_add(1, Ordering::Relaxed);
            let mut name = OsString::from(".ff-stage-dir-");
            name.push(std::process::id().to_string());
            name.push("-");
            name.push(format!("{nonce:016x}"));
            let staging_directory = staging_parent.join(name);
            match private_directory_builder().create(&staging_directory) {
                Ok(()) => {
                    let staging_path = staging_directory.join("artifact");
                    let staging = Self {
                        destination,
                        parent,
                        staging_parent,
                        staging_path,
                        staging_directory,
                        producer_mode: ProducerMode::ExternalPath,
                        staging_file: None,
                        identity: None,
                    };
                    staging.ensure_staging_filesystem()?;
                    staging.preflight_local_hard_link()?;
                    return Ok(staging);
                }
                Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!(
                            "failed to create private artifact staging directory in {}",
                            staging_parent.display()
                        )
                    });
                }
            }
        }
        bail!(
            "failed to allocate a unique artifact staging directory in {} after {} attempts",
            staging_parent.display(),
            STAGING_NAME_ATTEMPTS
        )
    }

    fn ensure_staging_filesystem(&self) -> Result<()> {
        let destination_parent = FileStat::of_target(&self.parent).with_context(|| {
            format!(
                "failed to inspect artifact destination parent {}",
                self.parent.display()
            )
        })?;
        let staging_parent = FileStat::of_target(&self.staging_parent).with_context(|| {
            format!(
                "failed to inspect artifact staging parent {}",
                self.staging_parent.display()
            )
        })?;
        anyhow::ensure!(
            on_same_filesystem(&destination_parent, &staging_parent),
            "artifact destination and staging parents are on different filesystems"
        );
        Ok(())
    }

    pub fn producer_path(&self) -> &Path {
        &self.staging_path
    }

    fn preflight_local_hard_link(&self) -> Result<()> {
        let source = self.staging_directory.join("hard-link-probe-source");
        let linked = self.staging_directory.join("hard-link-probe-target");
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        let probe_result = (|| -> Result<()> {
            let file = options.open(&source).with_context(|| {
                format!(
                    "failed to create artifact hard-link probe {}",
                    source.display()
                )
            })?;
            let identity =
                FileIdentity::from_stat(&FileStat::of_file(&file).with_context(|| {
                    format!(
                        "failed to inspect artifact hard-link probe {}",
                        source.display()
                    )
                })?)?;
            drop(file);
            fs::hard_link(&source, &linked).with_context(|| {
                format!(
                    "artifact destination filesystem does not support required local hard-link publication in {}",
                    self.parent.display()
                )
            })?;
            ensure_paths_are_same_file(&source, &linked, identity)
        })();
        let linked_cleanup = match fs::remove_file(&linked) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error).with_context(|| {
                format!(
                    "failed to remove artifact hard-link probe {}",
                    linked.display()
                )
            }),
        };
        let source_cleanup = match fs::remove_file(&source) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error).with_context(|| {
                format!(
                    "failed to remove artifact hard-link probe {}",
                    source.display()
                )
            }),
        };
        probe_result.and(linked_cleanup).and(source_cleanup)
    }

    pub fn write_with<T>(&self, producer: impl FnOnce(&mut dyn Write) -> Result<T>) -> Result<T> {
        let mut file = if self.producer_mode == ProducerMode::ExternalPath {
            anyhow::ensure!(
                !self.staging_path.exists(),
                "path-producer staging output already exists: {}",
                self.staging_path.display()
            );
            let mut options = OpenOptions::new();
            options.read(true).write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt as _;
                options.mode(0o600);
            }
            options.open(&self.staging_path).with_context(|| {
                format!(
                    "failed to create path-producer staging output {}",
                    self.staging_path.display()
                )
            })?
        } else {
            self.ensure_owned_path()?;
            let mut file = self
                .staging_file
                .as_ref()
                .context("artifact staging file is no longer available")?
                .try_clone()
                .with_context(|| {
                    format!(
                        "failed to clone artifact staging file {}",
                        self.staging_path.display()
                    )
                })?;
            file.set_len(0).with_context(|| {
                format!(
                    "failed to truncate artifact staging file {}",
                    self.staging_path.display()
                )
            })?;
            file.seek(SeekFrom::Start(0)).with_context(|| {
                format!(
                    "failed to rewind artifact staging file {}",
                    self.staging_path.display()
                )
            })?;
            file
        };
        let output = producer(&mut file).with_context(|| {
            format!(
                "artifact producer failed while writing {}",
                self.staging_path.display()
            )
        })?;
        file.flush().with_context(|| {
            format!(
                "failed to flush artifact staging file {}",
                self.staging_path.display()
            )
        })?;
        Ok(output)
    }

    pub fn write_bytes(&self, bytes: &[u8]) -> Result<()> {
        self.write_with(|writer| {
            writer
                .write_all(bytes)
                .context("failed to write artifact bytes")
        })
    }

    /// The staging file is still the one this handle created, in the directory
    /// it created it in.
    ///
    /// This is what publication can establish: a path nothing else has
    /// substituted. It says nothing about the bytes, and the caller is owed no
    /// impression that it does.
    fn ensure_staging_is_intact(&self) -> Result<()> {
        if self.producer_mode == ProducerMode::ExternalPath && self.identity.is_none() {
            anyhow::ensure!(
                self.staging_directory.is_dir(),
                "private artifact staging directory disappeared: {}",
                self.staging_directory.display()
            );
            return Ok(());
        }
        self.ensure_owned_path()?;
        self.staging_file
            .as_ref()
            .context("artifact staging file is no longer available")?;
        Ok(())
    }

    pub fn publish(self) -> std::result::Result<PublishedArtifact, ArtifactPublishError> {
        self.publish_with_parent_sync(sync_parent_directory)
    }

    fn publish_with_parent_sync<F>(
        mut self,
        mut sync_parent: F,
    ) -> std::result::Result<PublishedArtifact, ArtifactPublishError>
    where
        F: FnMut(&Path) -> std::io::Result<ArtifactDurability>,
    {
        if let Err(error) = self.adopt_path_producer_file() {
            return Err(self.not_published(error));
        }
        if let Err(error) = self.ensure_staging_is_intact() {
            return Err(self.not_published(error));
        }

        if let Err(error) = self
            .staging_file
            .as_ref()
            .context("artifact staging file is no longer available")
            .and_then(|file| {
                file.sync_all().with_context(|| {
                    format!(
                        "failed to synchronize artifact staging file {}",
                        self.staging_path.display()
                    )
                })
            })
        {
            return Err(self.not_published(error));
        }
        if let Err(error) = self.ensure_owned_path() {
            return Err(self.not_published(error));
        }

        let publication = match publish_staged_file(&self.staging_path, &self.destination)
            .with_context(|| {
                format!(
                    "failed to atomically publish artifact {} without replacing an existing path",
                    self.destination.display()
                )
            }) {
            Ok(publication) => publication,
            Err(error) => return Err(self.not_published(error)),
        };

        let mut published = PublishedArtifact {
            destination: self.destination.clone(),
            durability: publication.durability,
        };

        let expected_identity = match self.identity {
            Some(identity) => identity,
            None => {
                return Err(self.published_failure(
                    published,
                    anyhow!("published staging file has no recorded identity"),
                ));
            }
        };
        let verified = if publication.staging_retained {
            ensure_paths_are_same_file(&self.staging_path, &self.destination, expected_identity)
        } else {
            ensure_destination_is_the_verified_file(&self.destination, expected_identity)
        };
        if let Err(error) =
            verified.context("published artifact does not identify the verified staging file")
        {
            return Err(self.linked_unverified_failure(error));
        }

        if published.durability != ArtifactDurability::CrashDurable {
            match sync_parent(&self.parent) {
                Ok(durability) => published.durability = durability,
                Err(error) => {
                    let error = anyhow!(error).context(format!(
                        "artifact is visible at {}, but its parent directory was not synchronized",
                        self.destination.display()
                    ));
                    return Err(self.published_failure(published, error));
                }
            }
        }

        if publication.staging_retained {
            if let Err(error) = self.remove_owned_staging() {
                return Err(self.published_failure(
                    published,
                    error.context("artifact was published but its staging name was not removed"),
                ));
            }

            #[cfg(unix)]
            if let Err(error) = sync_parent(&self.staging_parent) {
                return Err(self.published_failure(
                    published,
                    anyhow!(error).context(
                        "artifact destination is crash-durable, but staging cleanup was not synchronized",
                    ),
                ));
            }
        } else {
            self.staging_file = None;
        }

        Ok(published)
    }

    fn ensure_owned_path(&self) -> Result<()> {
        let file = self
            .staging_file
            .as_ref()
            .context("artifact staging file is no longer available")?;
        let path_metadata = FileStat::of_path(&self.staging_path).with_context(|| {
            format!(
                "artifact staging path disappeared: {}",
                self.staging_path.display()
            )
        })?;
        anyhow::ensure!(
            path_metadata.file_type().is_file(),
            "artifact staging path is no longer a regular file: {}",
            self.staging_path.display()
        );
        let file_metadata = FileStat::of_file(file).with_context(|| {
            format!(
                "failed to inspect artifact staging file {}",
                self.staging_path.display()
            )
        })?;
        let identity = self
            .identity
            .context("artifact staging file has not been adopted")?;
        anyhow::ensure!(
            identity.matches(&path_metadata)
                && identity.matches(&file_metadata)
                && identifies_same_file(&path_metadata, &file_metadata),
            "artifact staging path was replaced after creation: {}",
            self.staging_path.display()
        );
        Ok(())
    }

    fn adopt_path_producer_file(&mut self) -> Result<()> {
        if self.producer_mode == ProducerMode::ManagedWriter || self.staging_file.is_some() {
            return self.ensure_owned_path();
        }
        anyhow::ensure!(
            self.staging_directory.is_dir(),
            "private artifact staging directory disappeared: {}",
            self.staging_directory.display()
        );
        let path_metadata = FileStat::of_path(&self.staging_path).with_context(|| {
            format!(
                "path producer did not create artifact staging output {}",
                self.staging_path.display()
            )
        })?;
        anyhow::ensure!(
            path_metadata.file_type().is_file(),
            "path-producer staging output is not a regular non-symlink file: {}",
            self.staging_path.display()
        );
        let mut source = OpenOptions::new()
            .read(true)
            .open(&self.staging_path)
            .with_context(|| {
                format!(
                    "failed to open path-producer staging output {}",
                    self.staging_path.display()
                )
            })?;
        let source_before = FileStat::of_file(&source).with_context(|| {
            format!(
                "failed to inspect path-producer staging output {}",
                self.staging_path.display()
            )
        })?;
        anyhow::ensure!(
            identifies_same_file(&path_metadata, &source_before)
                && path_metadata.len() == source_before.len()
                && path_metadata.modified().ok() == source_before.modified().ok()
                && change_marker(&path_metadata) == change_marker(&source_before),
            "path-producer staging output changed while opening {}",
            self.staging_path.display()
        );
        let sealed_path = self.staging_directory.join("sealed-artifact");
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        let mut sealed = options.open(&sealed_path).with_context(|| {
            format!(
                "failed to create sealed path-producer output {}",
                sealed_path.display()
            )
        })?;
        let seal_result = (|| -> Result<FileIdentity> {
            let copied = std::io::copy(&mut source, &mut sealed).with_context(|| {
                format!(
                    "failed to seal path-producer output {}",
                    self.staging_path.display()
                )
            })?;
            sealed.flush().with_context(|| {
                format!(
                    "failed to flush sealed path-producer output {}",
                    sealed_path.display()
                )
            })?;
            let source_after = FileStat::of_file(&source).with_context(|| {
                format!(
                    "failed to re-inspect path-producer output {}",
                    self.staging_path.display()
                )
            })?;
            let path_after = FileStat::of_path(&self.staging_path).with_context(|| {
                format!(
                    "failed to re-inspect path-producer staging path {}",
                    self.staging_path.display()
                )
            })?;
            anyhow::ensure!(
                path_after.file_type().is_file()
                    && identifies_same_file(&source_before, &source_after)
                    && identifies_same_file(&source_after, &path_after)
                    && copied == source_before.len()
                    && source_after.len() == source_before.len()
                    && path_after.len() == source_before.len()
                    && source_after.modified().ok() == source_before.modified().ok()
                    && path_after.modified().ok() == source_before.modified().ok()
                    && change_marker(&source_after) == change_marker(&source_before)
                    && change_marker(&path_after) == change_marker(&source_before),
                "path-producer output changed while being sealed: {}",
                self.staging_path.display()
            );
            let sealed_metadata = FileStat::of_file(&sealed).with_context(|| {
                format!(
                    "failed to inspect sealed path-producer output {}",
                    sealed_path.display()
                )
            })?;
            anyhow::ensure!(
                sealed_metadata.is_file() && sealed_metadata.len() == copied,
                "sealed path-producer output has the wrong size"
            );
            FileIdentity::from_stat(&sealed_metadata)
        })();
        let identity = match seal_result {
            Ok(identity) => identity,
            Err(error) => {
                let _ = fs::remove_file(&sealed_path);
                return Err(error);
            }
        };
        if let Err(error) = fs::remove_file(&self.staging_path) {
            let _ = fs::remove_file(&sealed_path);
            return Err(error).with_context(|| {
                format!(
                    "failed to retire path-producer output {}",
                    self.staging_path.display()
                )
            });
        }
        self.staging_path = sealed_path;
        self.identity = Some(identity);
        self.staging_file = Some(sealed);
        self.producer_mode = ProducerMode::ManagedWriter;
        self.ensure_owned_path()
    }

    fn remove_owned_staging(&mut self) -> Result<()> {
        if self.staging_file.is_some() {
            match self.ensure_owned_path() {
                Ok(()) => {}
                Err(_error) if !self.staging_path.exists() => {}
                Err(error) => return Err(error),
            }
        } else if self.staging_path.exists() {
            let metadata = fs::symlink_metadata(&self.staging_path).with_context(|| {
                format!(
                    "failed to inspect unadopted path-producer output {}",
                    self.staging_path.display()
                )
            })?;
            anyhow::ensure!(
                metadata.file_type().is_file(),
                "refusing to remove non-regular path-producer output {}",
                self.staging_path.display()
            );
        }
        self.staging_file.take();
        match fs::remove_file(&self.staging_path) {
            Ok(()) => {}
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "failed to remove path-producer staging output {}",
                        self.staging_path.display()
                    )
                });
            }
        }
        match fs::remove_dir(&self.staging_directory) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error).with_context(|| {
                format!(
                    "failed to remove private artifact staging directory {}",
                    self.staging_directory.display()
                )
            }),
        }
    }

    fn not_published(mut self, primary: anyhow::Error) -> ArtifactPublishError {
        let source = match self.remove_owned_staging() {
            Ok(()) => primary,
            Err(cleanup) => primary.context(format!(
                "staging cleanup also failed for {}: {cleanup:#}",
                self.staging_path.display()
            )),
        };
        ArtifactPublishError::NotPublished {
            destination: self.destination.clone(),
            staging_path: self.staging_path.clone(),
            source,
        }
    }

    fn published_failure(
        mut self,
        published: PublishedArtifact,
        primary: anyhow::Error,
    ) -> ArtifactPublishError {
        let source = match self.remove_owned_staging() {
            Ok(()) => primary,
            Err(cleanup) => primary.context(format!(
                "staging cleanup also failed for {}: {cleanup:#}",
                self.staging_path.display()
            )),
        };
        ArtifactPublishError::PublishedButFinalizationFailed {
            published,
            staging_path: self.staging_path.clone(),
            source,
        }
    }

    fn linked_unverified_failure(mut self, primary: anyhow::Error) -> ArtifactPublishError {
        let source = match self.remove_owned_staging() {
            Ok(()) => primary,
            Err(cleanup) => primary.context(format!(
                "staging cleanup also failed for {}: {cleanup:#}",
                self.staging_path.display()
            )),
        };
        ArtifactPublishError::LinkedButIdentityUnverified {
            destination: self.destination.clone(),
            staging_path: self.staging_path.clone(),
            source,
        }
    }
}

impl Drop for ArtifactStaging {
    fn drop(&mut self) {
        let _ = self.remove_owned_staging();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::Arc;
    use std::sync::Barrier;
    use std::thread;

    #[test]
    fn publishes_verified_sibling_without_leaving_staging() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("result.bin");
        let staging = ArtifactStaging::new(&destination).unwrap();
        let staging_path = staging.producer_path().to_owned();
        assert_eq!(
            staging_path.parent().unwrap().parent().unwrap(),
            destination.parent().unwrap().canonicalize().unwrap()
        );
        staging.write_bytes(b"flying fish").unwrap();

        let published = staging.publish().unwrap();
        assert_eq!(published.destination, destination.canonicalize().unwrap());
        assert_eq!(published.durability, ArtifactDurability::CrashDurable);
        assert_eq!(fs::read(&destination).unwrap(), b"flying fish");
        assert!(!staging_path.exists());
    }

    #[test]
    fn path_producer_can_publish_a_rename_created_file() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("result.safetensors");
        let staging = ArtifactStaging::new_for_path_producer(&destination).unwrap();
        let producer_path = staging.producer_path().to_owned();
        let private_directory = producer_path.parent().unwrap().to_owned();
        assert_eq!(
            private_directory.parent().unwrap(),
            destination.parent().unwrap().canonicalize().unwrap()
        );
        assert!(!producer_path.exists());
        assert_eq!(fs::read_dir(&private_directory).unwrap().count(), 0);

        let producer_temporary = private_directory.join("producer.tmp");
        fs::write(&producer_temporary, b"rename-produced").unwrap();
        fs::rename(&producer_temporary, &producer_path).unwrap();
        let published = staging.publish().unwrap();

        assert_eq!(fs::read(&destination).unwrap(), b"rename-produced");
        assert_eq!(published.destination, destination.canonicalize().unwrap());
        assert!(!private_directory.exists());
    }

    #[test]
    fn missing_path_producer_output_fails_before_publication_and_cleans_directory() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("result.bin");
        let staging = ArtifactStaging::new_for_path_producer(&destination).unwrap();
        let private_directory = staging.producer_path().parent().unwrap().to_owned();

        let error = staging.publish().unwrap_err();
        assert!(matches!(error, ArtifactPublishError::NotPublished { .. }));
        assert!(!error.destination_was_linked());
        assert!(!destination.exists());
        assert!(!private_directory.exists());
    }

    #[test]
    fn existing_destination_is_never_replaced() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("result.bin");
        fs::write(&destination, b"winner").unwrap();
        let staging = ArtifactStaging::new(&destination).unwrap();
        let staging_path = staging.producer_path().to_owned();
        staging.write_bytes(b"loser").unwrap();

        let error = staging.publish().unwrap_err();
        assert!(matches!(error, ArtifactPublishError::NotPublished { .. }));
        assert_eq!(fs::read(&destination).unwrap(), b"winner");
        assert!(!staging_path.exists());
    }

    #[test]
    fn concurrent_publishers_have_exactly_one_winner() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("result.bin");
        let barrier = Arc::new(Barrier::new(3));
        let mut workers = Vec::new();
        for payload in [b"first".as_slice(), b"second".as_slice()] {
            let staging = ArtifactStaging::new(&destination).unwrap();
            staging.write_bytes(payload).unwrap();
            let barrier = Arc::clone(&barrier);
            workers.push(thread::spawn(move || {
                barrier.wait();
                staging.publish()
            }));
        }
        barrier.wait();
        let results = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(results.iter().filter(|result| result.is_err()).count(), 1);
        let contents = fs::read(&destination).unwrap();
        assert!(contents == b"first" || contents == b"second");
        let loser = results
            .iter()
            .find_map(|result| result.as_ref().err())
            .unwrap();
        assert!(matches!(loser, ArtifactPublishError::NotPublished { .. }));
    }

    #[cfg(unix)]
    #[test]
    fn post_link_sync_failure_is_not_reported_as_retryable() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("result.bin");
        let staging = ArtifactStaging::new(&destination).unwrap();
        let staging_path = staging.producer_path().to_owned();
        staging.write_bytes(b"published").unwrap();

        let error = staging
            .publish_with_parent_sync(|_| {
                Err(std::io::Error::other("injected directory sync failure"))
            })
            .unwrap_err();
        let published = error.published().expect("hard link must be represented");
        assert_eq!(published.destination, destination.canonicalize().unwrap());
        assert_eq!(published.durability, ArtifactDurability::AtomicVisibility);
        assert!(error.destination_was_linked());
        assert_eq!(fs::read(&destination).unwrap(), b"published");
        assert!(!staging_path.exists());
        assert!(
            error
                .to_string()
                .contains("parent directory was not synchronized")
        );
    }

    #[cfg(unix)]
    #[test]
    fn cleanup_sync_failure_preserves_crash_durable_publication_state() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("result.bin");
        let staging = ArtifactStaging::new(&destination).unwrap();
        let staging_path = staging.producer_path().to_owned();
        staging.write_bytes(b"published").unwrap();
        let mut sync_count = 0usize;

        let error = staging
            .publish_with_parent_sync(|_| {
                sync_count += 1;
                if sync_count == 1 {
                    Ok(ArtifactDurability::CrashDurable)
                } else {
                    Err(std::io::Error::other(
                        "injected cleanup directory sync failure",
                    ))
                }
            })
            .unwrap_err();
        let published = error.published().expect("hard link must be represented");
        assert_eq!(published.durability, ArtifactDurability::CrashDurable);
        assert_eq!(fs::read(&destination).unwrap(), b"published");
        assert!(!staging_path.exists());
        assert!(error.to_string().contains("cleanup was not synchronized"));
    }

    #[cfg(unix)]
    #[test]
    fn post_link_staging_cleanup_failure_reports_published_destination() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("result.bin");
        let staging = ArtifactStaging::new_for_path_producer(&destination).unwrap();
        let private_directory = staging.producer_path().parent().unwrap().to_owned();
        staging.write_bytes(b"published").unwrap();
        let residue = private_directory.join("producer-residue");
        fs::write(&residue, b"residue").unwrap();

        let error = staging.publish().unwrap_err();
        let published = error.published().expect("hard link must be represented");
        assert_eq!(published.durability, ArtifactDurability::CrashDurable);
        assert_eq!(fs::read(&destination).unwrap(), b"published");
        assert!(private_directory.exists());
        assert!(error.to_string().contains("staging name was not removed"));

        fs::remove_file(residue).unwrap();
        fs::remove_dir(private_directory).unwrap();
    }

    #[test]
    fn missing_or_replaced_staging_is_not_published_or_deleted() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("result.bin");
        let staging = ArtifactStaging::new(&destination).unwrap();
        let staging_path = staging.producer_path().to_owned();
        fs::remove_file(&staging_path).unwrap();
        fs::write(&staging_path, b"replacement").unwrap();

        let error = staging.publish().unwrap_err();
        assert!(matches!(error, ArtifactPublishError::NotPublished { .. }));
        assert!(!destination.exists());
        assert_eq!(fs::read(&staging_path).unwrap(), b"replacement");
    }

    #[test]
    fn post_link_check_binds_destination_to_the_recorded_inode() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("result.bin");
        let staging = ArtifactStaging::new(&destination).unwrap();
        let staging_path = staging.producer_path().to_owned();
        staging.write_bytes(b"verified").unwrap();
        let verified_identity = staging.identity.unwrap();

        fs::remove_file(&staging_path).unwrap();
        fs::write(&staging_path, b"replacement").unwrap();
        fs::hard_link(&staging_path, &destination).unwrap();
        assert!(
            ensure_paths_are_same_file(&staging_path, &destination, verified_identity)
                .unwrap_err()
                .to_string()
                .contains("verified regular file")
        );

        fs::remove_file(&destination).unwrap();
        fs::remove_file(&staging_path).unwrap();
    }

    #[test]
    fn an_unverified_link_never_exposes_the_prelink_digest_as_published() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("result.bin");
        let staging = ArtifactStaging::new(&destination).unwrap();
        let staging_path = staging.producer_path().to_owned();
        staging.write_bytes(b"verified-before-link").unwrap();
        fs::hard_link(&staging_path, &destination).unwrap();

        let error = staging
            .linked_unverified_failure(anyhow::anyhow!("injected post-link identity uncertainty"));
        assert!(matches!(
            error,
            ArtifactPublishError::LinkedButIdentityUnverified { .. }
        ));
        assert!(error.destination_was_linked());
        assert!(error.published().is_none());
        assert_eq!(fs::read(&destination).unwrap(), b"verified-before-link");
    }

    #[test]
    fn drop_and_explicit_write_failure_paths_clean_owned_staging() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("result.bin");
        let path = {
            let staging = ArtifactStaging::new(&destination).unwrap();
            let path = staging.producer_path().to_owned();
            let error = staging
                .write_with(|producer| -> anyhow::Result<()> {
                    producer.write_all(b"partial")?;
                    anyhow::bail!("injected producer failure")
                })
                .unwrap_err();
            assert!(format!("{error:#}").contains("injected producer failure"));
            path
        };
        assert!(!path.exists());
        assert!(!destination.exists());
    }

    #[test]
    fn explicit_staging_parent_keeps_private_entries_outside_the_destination_directory() {
        let directory = tempfile::tempdir().unwrap();
        let destination_parent = directory.path().join("destination");
        let staging_parent = directory.path().join("staging");
        fs::create_dir(&destination_parent).unwrap();
        fs::create_dir(&staging_parent).unwrap();
        let destination = destination_parent.join("artifact.bin");
        let staging =
            ArtifactStaging::new_with_staging_parent(&destination, &staging_parent).unwrap();
        assert!(fs::read_dir(&destination_parent).unwrap().next().is_none());
        staging.write_bytes(b"sealed").unwrap();
        staging.publish().unwrap();
        assert_eq!(fs::read(&destination).unwrap(), b"sealed");
        assert!(fs::read_dir(&staging_parent).unwrap().next().is_none());
    }

    #[test]
    fn path_producer_supports_an_explicit_staging_parent() {
        let directory = tempfile::tempdir().unwrap();
        let destination_parent = directory.path().join("destination");
        let staging_parent = directory.path().join("staging");
        fs::create_dir(&destination_parent).unwrap();
        fs::create_dir(&staging_parent).unwrap();
        let destination = destination_parent.join("artifact.bin");
        let staging = ArtifactStaging::new_for_path_producer_with_staging_parent(
            &destination,
            &staging_parent,
        )
        .unwrap();
        fs::write(staging.producer_path(), b"produced").unwrap();
        staging.publish().unwrap();

        assert_eq!(fs::read(destination).unwrap(), b"produced");
        assert!(fs::read_dir(destination_parent).unwrap().next().is_some());
        assert!(fs::read_dir(staging_parent).unwrap().next().is_none());
    }
}
