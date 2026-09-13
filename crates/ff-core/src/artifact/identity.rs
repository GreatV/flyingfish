//! Platform file identity and directory durability.
//!
//! Publication has to prove that the path it linked is the file it verified,
//! which needs an inode-level identity rather than a path comparison. Unix,
//! Windows and other targets each supply what they can; the fallback is
//! explicitly unable to prove sameness rather than silently claiming it.

use super::staging::*;
use super::*;

/// What the operating system uses to say *which* file this is: a device and
/// inode on Unix, a volume serial and file index on Windows.
///
/// Both platforms answer the same question with a pair of integers, so one
/// opaque, comparable value covers them and every identity check in the
/// workspace is written once instead of once per platform.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct FileKey {
    volume: u64,
    index: u64,
}

impl std::fmt::Display for FileKey {
    /// The stable textual form recorded identities are written in.
    ///
    /// Evidence and manifests outlive the build that wrote them, so this
    /// spelling is stated here rather than borrowed from the derived `Debug`,
    /// whose shape is not part of the type's contract: renaming a field would
    /// otherwise invalidate every recorded key without changing a line that
    /// mentions one.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:016x}:{:016x}", self.volume, self.index)
    }
}

/// A file's metadata paired with the platform identity that says *which* file
/// it is.
///
/// Unix already carries device and inode inside `Metadata`, so this is a thin
/// wrapper there. Windows carries the volume serial and file index only behind
/// the unstable `windows_by_handle` accessors, so the identity has to be read
/// from a handle at the moment the metadata is taken. Pairing the two at
/// acquisition is what lets every later comparison stay an identity check on
/// both platforms instead of degrading to a path or timestamp comparison.
///
/// It derefs to the `Metadata` it wraps, so `is_file`, `len` and `modified`
/// read as before.
#[derive(Debug)]
pub struct FileStat {
    metadata: Metadata,
    #[cfg(windows)]
    identity: Option<(u32, u64)>,
}

impl FileStat {
    /// Describe `path` itself, never a symbolic link's target.
    pub fn of_path(path: &Path) -> std::io::Result<Self> {
        let metadata = fs::symlink_metadata(path)?;
        Ok(Self {
            #[cfg(windows)]
            identity: crate::storage::windows_open_for_identity(path)
                .ok()
                .and_then(|file| crate::storage::windows_file_identity(&file).ok()),
            metadata,
        })
    }

    /// Describe what `path` resolves to, following a final symbolic link.
    ///
    /// This is what a filesystem question wants: a link's target decides which
    /// volume the bytes live on, not the link.
    pub fn of_target(path: &Path) -> std::io::Result<Self> {
        let metadata = fs::metadata(path)?;
        Ok(Self {
            #[cfg(windows)]
            identity: crate::storage::windows_open_for_target_identity(path)
                .ok()
                .and_then(|file| crate::storage::windows_file_identity(&file).ok()),
            metadata,
        })
    }

    /// Describe an already-open file, which cannot have been swapped under the
    /// handle between the two observations.
    pub fn of_file(file: &File) -> std::io::Result<Self> {
        let metadata = file.metadata()?;
        Ok(Self {
            #[cfg(windows)]
            identity: crate::storage::windows_file_identity(file).ok(),
            metadata,
        })
    }

    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// This file's identity, or `None` where the platform could not supply one.
    #[cfg(unix)]
    pub fn key(&self) -> Option<FileKey> {
        use std::os::unix::fs::MetadataExt as _;
        Some(FileKey {
            volume: self.dev(),
            index: self.ino(),
        })
    }

    #[cfg(windows)]
    pub fn key(&self) -> Option<FileKey> {
        self.identity.map(|(volume, index)| FileKey {
            volume: u64::from(volume),
            index,
        })
    }

    #[cfg(not(any(unix, windows)))]
    pub fn key(&self) -> Option<FileKey> {
        None
    }

    /// This file's identity in the form a recorded binding stores it in.
    ///
    /// A platform that cannot answer records `unavailable`, which is what the
    /// key already degraded to: two such records compare equal, so on those
    /// platforms a binding rests on the remaining size and modification time
    /// alone. Unix and Windows both answer.
    pub fn recorded_key(&self) -> String {
        self.key()
            .map_or_else(|| "unavailable".to_owned(), |key| key.to_string())
    }

    /// Whether both describe the same filesystem object, whatever its kind.
    ///
    /// `identifies_same_file` answers this for regular files only, because
    /// artifact publication must never accept anything else. Directory
    /// stability checks need the same comparison without that restriction.
    pub fn identifies_same_file_as(&self, other: &Self) -> bool {
        self.key().is_some_and(|key| Some(key) == other.key())
    }
}

impl std::ops::Deref for FileStat {
    type Target = Metadata;

    fn deref(&self) -> &Metadata {
        &self.metadata
    }
}

pub(super) fn ensure_paths_are_same_file(
    left: &Path,
    right: &Path,
    expected_identity: FileIdentity,
) -> Result<()> {
    let left_metadata = FileStat::of_path(left).with_context(|| {
        format!(
            "failed to inspect linked artifact source {}",
            left.display()
        )
    })?;
    let right_metadata = FileStat::of_path(right).with_context(|| {
        format!(
            "failed to inspect linked artifact destination {}",
            right.display()
        )
    })?;
    anyhow::ensure!(
        left_metadata.file_type().is_file()
            && right_metadata.file_type().is_file()
            && expected_identity.matches(&left_metadata)
            && expected_identity.matches(&right_metadata)
            && identifies_same_file(&left_metadata, &right_metadata),
        "artifact source {} and destination {} do not identify the verified regular file",
        left.display(),
        right.display()
    );
    Ok(())
}

/// The same as `sync_parent_directory`, for a directory that is already open.
///
/// Callers that hold a pinned directory handle use this so the flush cannot be
/// redirected by a path change between opening and flushing.
#[cfg(unix)]
pub fn sync_open_directory(directory: &File) -> std::io::Result<ArtifactDurability> {
    if !directory.metadata()?.is_dir() {
        return Err(std::io::Error::new(
            ErrorKind::InvalidInput,
            "sync target is no longer a directory",
        ));
    }
    directory.sync_all()?;
    Ok(ArtifactDurability::CrashDurable)
}

#[cfg(not(unix))]
pub fn sync_open_directory(_directory: &File) -> std::io::Result<ArtifactDurability> {
    Ok(ArtifactDurability::AtomicVisibility)
}

/// Make a directory entry durable, and report which durability was reached.
///
/// Only Unix can flush a directory's own metadata, so elsewhere a rename or
/// link is atomically visible but its directory entry is not proven on disk.
/// Callers record the difference rather than assuming the stronger one.
/// What publishing a staged file achieved on its own, before any parent flush.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct StagedPublication {
    pub durability: ArtifactDurability,
    /// Whether the staging name survived the publication and still needs
    /// removing. Only a link leaves one behind; a move consumes it.
    pub staging_retained: bool,
}

/// Publish `staging` at `destination` atomically, never replacing an existing
/// path.
///
/// The two platforms reach the same guarantee by different routes. Unix links
/// the file into place — atomic and never clobbering, but the new directory
/// entry is only durable once the caller flushes the parent, so this reports
/// `AtomicVisibility` and leaves the staging name behind. Windows cannot flush
/// a directory at all, and instead moves the file write-through, which the
/// operating system commits to disk before returning; that reaches
/// `CrashDurable` outright and consumes the staging name.
#[cfg(unix)]
pub(super) fn publish_staged_file(
    staging: &Path,
    destination: &Path,
) -> std::io::Result<StagedPublication> {
    fs::hard_link(staging, destination)?;
    Ok(StagedPublication {
        durability: ArtifactDurability::AtomicVisibility,
        staging_retained: true,
    })
}

#[cfg(windows)]
pub(super) fn publish_staged_file(
    staging: &Path,
    destination: &Path,
) -> std::io::Result<StagedPublication> {
    crate::storage::windows_move_file(staging, destination, false)?;
    Ok(StagedPublication {
        durability: ArtifactDurability::CrashDurable,
        staging_retained: false,
    })
}

#[cfg(not(any(unix, windows)))]
pub(super) fn publish_staged_file(
    _staging: &Path,
    _destination: &Path,
) -> std::io::Result<StagedPublication> {
    Err(std::io::Error::new(
        ErrorKind::Unsupported,
        "artifact publication requires Unix or Windows",
    ))
}

/// Replace `destination` with `source` atomically and durably.
///
/// Unlike `publish_staged_file` this is for a path that is meant to be
/// replaced, such as a mutable head record. Unix renames and leaves the
/// directory flush to the caller; Windows moves write-through and is done.
#[cfg(unix)]
pub fn replace_file_durably(
    source: &Path,
    destination: &Path,
) -> std::io::Result<ArtifactDurability> {
    fs::rename(source, destination)?;
    Ok(ArtifactDurability::AtomicVisibility)
}

#[cfg(windows)]
pub fn replace_file_durably(
    source: &Path,
    destination: &Path,
) -> std::io::Result<ArtifactDurability> {
    crate::storage::windows_replace_file(source, destination)?;
    Ok(ArtifactDurability::AtomicVisibility)
}

#[cfg(not(any(unix, windows)))]
pub fn replace_file_durably(
    _source: &Path,
    _destination: &Path,
) -> std::io::Result<ArtifactDurability> {
    Err(std::io::Error::new(
        ErrorKind::Unsupported,
        "durable replacement requires Unix or Windows",
    ))
}

/// Prove that `destination` is the file whose identity was recorded before it
/// was published, for platforms where publication consumed the staging name.
pub(super) fn ensure_destination_is_the_verified_file(
    destination: &Path,
    expected_identity: FileIdentity,
) -> Result<()> {
    let stat = FileStat::of_path(destination).with_context(|| {
        format!(
            "failed to inspect published artifact {}",
            destination.display()
        )
    })?;
    anyhow::ensure!(
        expected_identity.matches(&stat),
        "published artifact {} is not the verified staging file",
        destination.display()
    );
    Ok(())
}

#[cfg(unix)]
pub fn sync_parent_directory(parent: &Path) -> std::io::Result<ArtifactDurability> {
    let directory = File::open(parent)?;
    if !directory.metadata()?.is_dir() {
        return Err(std::io::Error::new(
            ErrorKind::InvalidInput,
            "artifact parent is no longer a directory",
        ));
    }
    directory.sync_all()?;
    Ok(ArtifactDurability::CrashDurable)
}

#[cfg(not(unix))]
pub fn sync_parent_directory(_parent: &Path) -> std::io::Result<ArtifactDurability> {
    Ok(ArtifactDurability::AtomicVisibility)
}

/// The identity of a regular file that publication verified.
///
/// Publication must never accept anything but a regular file, so the kind
/// check lives here rather than at each call site.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct FileIdentity(FileKey);

impl FileIdentity {
    pub(super) fn from_stat(stat: &FileStat) -> Result<Self> {
        anyhow::ensure!(
            stat.is_file(),
            "artifact staging inode is not a regular file"
        );
        let key = stat
            .key()
            .context("artifact staging file identity is unavailable on this platform")?;
        Ok(Self(key))
    }

    pub(super) fn matches(self, stat: &FileStat) -> bool {
        stat.is_file() && stat.key() == Some(self.0)
    }
}

pub fn identifies_same_file(left: &FileStat, right: &FileStat) -> bool {
    FileIdentity::from_stat(left)
        .ok()
        .is_some_and(|identity| identity.matches(right))
}

#[cfg(unix)]
pub fn change_marker(stat: &FileStat) -> (i64, i64) {
    use std::os::unix::fs::MetadataExt as _;
    (stat.ctime(), stat.ctime_nsec())
}

/// Whether both live on the same volume, which is what a hard link requires.
pub(super) fn on_same_filesystem(left: &FileStat, right: &FileStat) -> bool {
    left.key()
        .zip(right.key())
        .is_some_and(|(left, right)| left.volume == right.volume)
}

#[cfg(windows)]
pub fn change_marker(stat: &FileStat) -> u64 {
    use std::os::windows::fs::MetadataExt as _;
    stat.last_write_time()
}

#[cfg(not(any(unix, windows)))]
pub fn change_marker(_stat: &FileStat) {}

#[cfg(test)]
mod tests {
    use super::*;

    /// The recorded spelling of a file key is a persisted format: evidence
    /// written by one build is matched against a key collected by the next, so
    /// it is pinned here rather than left to whatever `Debug` happens to emit.
    #[test]
    fn recorded_file_keys_are_stable_fixed_width_and_distinguish_files() {
        assert_eq!(
            FileKey {
                volume: 0x102fe,
                index: 0x3039,
            }
            .to_string(),
            "00000000000102fe:0000000000003039"
        );
        let directory = tempfile::tempdir().unwrap();
        let (one, two) = (directory.path().join("one"), directory.path().join("two"));
        std::fs::write(&one, b"one").unwrap();
        std::fs::write(&two, b"two").unwrap();
        let stat = FileStat::of_target(&one).unwrap();
        assert_ne!(
            stat.recorded_key(),
            FileStat::of_target(&two).unwrap().recorded_key()
        );
        assert_ne!(stat.recorded_key(), "unavailable");
        assert_eq!(
            stat.recorded_key(),
            FileStat::of_target(&one).unwrap().recorded_key()
        );
    }
}
