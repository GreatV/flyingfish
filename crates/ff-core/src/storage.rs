//! The storage properties that bound weight streaming.

use std::{
    fs::File,
    path::{Path, PathBuf},
};

/// Tell the kernel a file's pages are not worth retaining, dropping the clean
/// ones already cached, and report whether there was a primitive to do it with.
///
/// This is the opposite of a prefetch hint and exists for one measured reason.
/// A checkpoint larger than host memory is read cyclically -- every evaluation
/// walks every block -- which is the worst case for an LRU page cache: by the
/// time the scan returns to the first shard it has been evicted by the last,
/// so nothing is ever reused and the cache still pays to hold it. Measured on
/// the pinned host against MiniMax-H3's 61.7 GiB transformer on a 62 GiB
/// machine, a full scan ran at 0.30 GB/s and a second scan at 0.13 GB/s --
/// slower, because by then the cache was full of pages that would not be hit.
/// Dropping each shard behind the read held 1.82 GB/s on the same scan, and
/// left 35 GiB of host memory free instead of filled.
///
/// A checkpoint that does fit is a different case entirely and must not be
/// advised this way: there the retention is the whole point.
///
/// Unix only. Windows exposes no equivalent for a file's cached pages, so this
/// reports `false` there rather than pretending the advice landed.
pub(crate) fn advise_dropped(file: &File) -> bool {
    platform_advise_dropped(file)
}

#[cfg(unix)]
fn platform_advise_dropped(file: &File) -> bool {
    use std::os::fd::AsRawFd;
    // Length zero means "to end of file".
    unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) == 0 }
}

#[cfg(not(unix))]
fn platform_advise_dropped(_file: &File) -> bool {
    false
}

/// The kernel readahead window that bounds streaming from `path`'s device.
///
/// The mmap fault-in that materializes a stage never issues larger I/O than
/// this window allows, so it sets the ceiling on streaming throughput.
///
/// Measured on the pinned RTX 4090 host against a 294 MiB attention stage:
/// 594 MB/s at 128 KiB (the Linux default), 2,046 MB/s at 2 MiB, 2,428 MB/s at
/// 8 MiB. The window is a runtime setting, not a device property: the same
/// NVMe reads at either rate depending on it.
///
/// It is Linux's mechanism specifically — Windows' cache manager sizes
/// readahead itself — so `read_ahead_window` reports `None` off Unix rather
/// than inventing advice that would not apply.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReadAheadWindow {
    pub bytes: u64,
    /// The file the window is written to, so a caller can name it exactly
    /// rather than making the reader derive the device numbers themselves.
    pub control_file: PathBuf,
}

/// The window below which weight streaming is meaningfully throttled.
///
/// The Linux default of 128 KiB sits far under this; 2 MiB reached within 20%
/// of the best measured rate on the pinned host, and larger windows returned
/// little more.
pub const RECOMMENDED_READ_AHEAD_BYTES: u64 = 2 * 1024 * 1024;

impl ReadAheadWindow {
    /// True when the window is small enough to bound streaming below what the
    /// device can deliver.
    pub const fn throttles_weight_streaming(&self) -> bool {
        self.bytes < RECOMMENDED_READ_AHEAD_BYTES
    }
}

/// Read the window for the device backing `path`, or `None` where the platform
/// does not expose one.
///
/// This is a best-effort observation, like the rest of the probe: an absent or
/// unreadable value is reported as unknown rather than as an error, because no
/// decision depends on it.
pub fn read_ahead_window(path: &Path) -> Option<ReadAheadWindow> {
    read_ahead_window_under(Path::new("/sys/dev/block"), path)
}

fn read_ahead_window_under(sysfs_block: &Path, path: &Path) -> Option<ReadAheadWindow> {
    let (major, minor) = device_numbers(path)?;
    let device = sysfs_block.join(format!("{major}:{minor}"));
    for relative in ["bdi/read_ahead_kb", "queue/read_ahead_kb"] {
        if let Some(window) = read_kib(device.join(relative)) {
            return Some(window);
        }
    }
    None
}

fn read_kib(path: PathBuf) -> Option<ReadAheadWindow> {
    let text = std::fs::read_to_string(&path).ok()?;
    let kib = text.trim().parse::<u64>().ok()?;
    Some(ReadAheadWindow {
        bytes: kib.checked_mul(1024)?,
        control_file: path,
    })
}

#[cfg(unix)]
fn device_numbers(path: &Path) -> Option<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    let device = std::fs::metadata(path).ok()?.dev();
    let major = ((device >> 8) & 0xfff) | ((device >> 32) & !0xfff);
    let minor = (device & 0xff) | ((device >> 12) & !0xff);
    Some((major, minor))
}

#[cfg(not(unix))]
fn device_numbers(_path: &Path) -> Option<(u64, u64)> {
    None
}

#[cfg(windows)]
fn windows_last_error() -> u32 {
    unsafe { windows_sys::Win32::Foundation::GetLastError() }
}

/// The volume and file index that identify a Windows file the way a device and
/// inode identify a Unix one.
///
/// `std::fs::Metadata` carries these, but only behind the unstable
/// `windows_by_handle` feature, so they are read from the handle instead. That
/// is the same call the standard library makes; the difference is that it must
/// come from a handle rather than from an already-taken `Metadata`.
#[cfg(windows)]
pub(crate) fn windows_file_identity(file: &std::fs::File) -> anyhow::Result<(u32, u64)> {
    use std::os::windows::io::AsRawHandle as _;
    use windows_sys::Win32::Storage::FileSystem::{
        BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
    };

    let mut information = unsafe { std::mem::zeroed::<BY_HANDLE_FILE_INFORMATION>() };
    let read =
        unsafe { GetFileInformationByHandle(file.as_raw_handle() as _, &raw mut information) };
    anyhow::ensure!(
        read != 0,
        "GetFileInformationByHandle failed with error {}",
        windows_last_error()
    );
    let index =
        (u64::from(information.nFileIndexHigh) << 32) | u64::from(information.nFileIndexLow);
    Ok((information.dwVolumeSerialNumber, index))
}

/// Open what `path` resolves to, following a final symbolic link, only to
/// identify it.
#[cfg(windows)]
pub(crate) fn windows_open_for_target_identity(path: &Path) -> std::io::Result<std::fs::File> {
    windows_open_with_identity_flags(path, 0)
}

/// Open `path` only to identify it, without following a final symbolic link
/// and without excluding other openers.
///
/// `FILE_FLAG_BACKUP_SEMANTICS` is what allows a directory to be opened at all,
/// and `FILE_FLAG_OPEN_REPARSE_POINT` makes this identify the same thing
/// `symlink_metadata` describes rather than the link's target.
#[cfg(windows)]
pub(crate) fn windows_open_for_identity(path: &Path) -> std::io::Result<std::fs::File> {
    windows_open_with_identity_flags(
        path,
        windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT,
    )
}

#[cfg(windows)]
fn windows_open_with_identity_flags(path: &Path, extra: u32) -> std::io::Result<std::fs::File> {
    use std::os::windows::fs::OpenOptionsExt as _;
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_FLAG_BACKUP_SEMANTICS, FILE_READ_ATTRIBUTES, FILE_SHARE_DELETE, FILE_SHARE_READ,
        FILE_SHARE_WRITE,
    };

    std::fs::OpenOptions::new()
        .access_mode(FILE_READ_ATTRIBUTES)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS | extra)
        .open(path)
}

/// Replace `destination` with `source` atomically, tolerating readers.
///
/// Windows offers three ways to replace a file and only this one keeps both
/// properties the protocol needs. Measured on Windows 10.0.26200:
///
/// - `MoveFileEx` with `MOVEFILE_REPLACE_EXISTING` is atomic for a reader, but
///   refuses with access denied whenever *anyone* holds the destination open,
///   including a reader that is only mid-`read`.
/// - `ReplaceFileW` tolerates an open destination but unlinks before it
///   renames, so a concurrent reader can observe the path missing.
/// - `FileRenameInfoEx` with POSIX semantics does both, which is what it exists
///   for. Over 200 replacements against a reader looping on the destination it
///   failed zero times and the reader never once saw the path absent.
///
/// It carries no write-through flag, so this reports only atomic visibility --
/// the same level a Unix `rename` reaches before its parent is flushed.
/// Available since Windows 10 1607; older releases fall back to `MoveFileEx`,
/// which is atomic but cannot proceed against an open destination.
#[cfg(windows)]
pub(crate) fn windows_replace_file(source: &Path, destination: &Path) -> std::io::Result<()> {
    use std::os::windows::{ffi::OsStrExt as _, fs::OpenOptionsExt as _, io::AsRawHandle as _};
    use windows_sys::Win32::Storage::FileSystem::{
        DELETE, FILE_RENAME_INFO, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE,
        FileRenameInfoEx, SetFileInformationByHandle,
    };

    const REPLACE_IF_EXISTS: u32 = 0x1;
    const POSIX_SEMANTICS: u32 = 0x2;

    let renamed = std::fs::OpenOptions::new()
        .access_mode(DELETE)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
        .open(source)
        .and_then(|handle| {
            let name: Vec<u16> = destination.as_os_str().encode_wide().collect();
            let name_bytes = name.len() * 2;
            let mut buffer = vec![0u8; size_of::<FILE_RENAME_INFO>() + name_bytes + 2];
            unsafe {
                let info = buffer.as_mut_ptr().cast::<FILE_RENAME_INFO>();
                (*info).Anonymous.Flags = REPLACE_IF_EXISTS | POSIX_SEMANTICS;
                (*info).RootDirectory = std::ptr::null_mut();
                (*info).FileNameLength = u32::try_from(name_bytes).unwrap_or(u32::MAX);
                std::ptr::copy_nonoverlapping(
                    name.as_ptr().cast::<u8>(),
                    buffer
                        .as_mut_ptr()
                        .add(std::mem::offset_of!(FILE_RENAME_INFO, FileName)),
                    name_bytes,
                );
                let set = SetFileInformationByHandle(
                    handle.as_raw_handle() as _,
                    FileRenameInfoEx,
                    buffer.as_ptr().cast(),
                    u32::try_from(buffer.len()).unwrap_or(u32::MAX),
                );
                if set == 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            Ok(())
        });
    match renamed {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::InvalidInput => {
            windows_move_file(source, destination, true)
        }
        Err(error) => Err(error),
    }
}

/// Move `source` onto `destination`, flushed to disk before returning.
///
/// This is the whole of Windows' answer to "link or rename, then make the
/// directory entry durable". Without `MOVEFILE_REPLACE_EXISTING` the call fails
/// when the destination exists, which is the never-clobber half;
/// `MOVEFILE_WRITE_THROUGH` does not return until the move is on disk, which is
/// the durability half. Windows offers no way to flush a directory's own
/// metadata, so this pairing is what stands in for it.
///
/// Measured on Windows 10.0.26200: refusing an existing destination leaves that
/// destination unchanged and the source intact, a same-volume move preserves
/// the file index so the moved file is still provably the one that was
/// verified, and write-through costs about 2.2x a plain move.
#[cfg(windows)]
pub(crate) fn windows_move_file(
    source: &Path,
    destination: &Path,
    replace_existing: bool,
) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt as _;
    use windows_sys::Win32::Storage::FileSystem::{
        MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
    };

    let wide =
        |path: &Path| -> Vec<u16> { path.as_os_str().encode_wide().chain(Some(0)).collect() };
    let (wide_source, wide_destination) = (wide(source), wide(destination));

    let mut flags = MOVEFILE_WRITE_THROUGH;
    if replace_existing {
        flags |= MOVEFILE_REPLACE_EXISTING;
    }
    let moved = unsafe { MoveFileExW(wide_source.as_ptr(), wide_destination.as_ptr(), flags) };
    if moved == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// Open a directory, for platforms that need to be told that is the intent.
///
/// Unix opens a directory like any other path. Windows refuses unless
/// `FILE_FLAG_BACKUP_SEMANTICS` is set, which is why a plain `File::open` on a
/// directory fails there with access denied.
pub fn open_directory(path: &Path) -> std::io::Result<std::fs::File> {
    #[cfg(windows)]
    {
        windows_open_with_identity_flags(path, 0)
    }
    #[cfg(not(windows))]
    {
        std::fs::File::open(path)
    }
}

/// The resolved spelling of `path`, in the form a caller would write.
///
/// `fs::canonicalize` answers with the operating system's own canonical form,
/// which on Windows carries the `\\?\` extended-length prefix that no caller
/// types and no configuration file contains. A check that compares a supplied
/// path against the resolved one has to compare like with like, or it can never
/// hold on Windows.
///
/// The prefix is presentational, not semantic: it only lifts the legacy path
/// length limit. Removing it for comparison keeps the check exactly as strong
/// as it is on Unix -- the path must already be the resolved one, with no
/// relative components and no symbolic links left to follow.
pub fn canonical_comparable_path(path: &Path) -> std::io::Result<PathBuf> {
    let canonical = std::fs::canonicalize(path)?;
    #[cfg(windows)]
    {
        Ok(strip_verbatim_prefix(&canonical))
    }
    #[cfg(not(windows))]
    {
        Ok(canonical)
    }
}

#[cfg(windows)]
fn strip_verbatim_prefix(path: &Path) -> PathBuf {
    use std::path::{Component, Prefix};

    let mut components = path.components();
    let Some(Component::Prefix(prefix)) = components.next() else {
        return path.to_path_buf();
    };
    match prefix.kind() {
        Prefix::VerbatimDisk(letter) => {
            let mut rebuilt = PathBuf::from(format!("{}:\\", letter as char));
            rebuilt.extend(components.filter(|part| !matches!(part, Component::RootDir)));
            rebuilt
        }
        Prefix::VerbatimUNC(server, share) => {
            let mut rebuilt = PathBuf::from(r"\\");
            rebuilt.push(server);
            rebuilt.push(share);
            rebuilt.extend(components.filter(|part| !matches!(part, Component::RootDir)));
            rebuilt
        }
        _ => path.to_path_buf(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_or_unreadable_window_is_unknown_rather_than_an_error() {
        let empty = tempfile::tempdir().unwrap();
        assert_eq!(read_ahead_window_under(empty.path(), empty.path()), None);
        assert_eq!(read_ahead_window(Path::new("/definitely/not/a/path")), None);
    }

    #[cfg(unix)]
    #[test]
    fn the_window_is_read_from_the_device_backing_the_path() {
        use std::os::unix::fs::MetadataExt;
        let root = tempfile::tempdir().unwrap();
        let sysfs = root.path().join("block");
        let subject = root.path().join("model");
        std::fs::create_dir(&subject).unwrap();
        let device = std::fs::metadata(&subject).unwrap().dev();
        let major = ((device >> 8) & 0xfff) | ((device >> 32) & !0xfff);
        let minor = (device & 0xff) | ((device >> 12) & !0xff);

        let entry = sysfs.join(format!("{major}:{minor}"));
        std::fs::create_dir_all(entry.join("queue")).unwrap();
        std::fs::write(entry.join("queue/read_ahead_kb"), "128\n").unwrap();
        assert_eq!(
            read_ahead_window_under(&sysfs, &subject),
            Some(ReadAheadWindow {
                bytes: 131_072,
                control_file: entry.join("queue/read_ahead_kb")
            })
        );
        std::fs::create_dir_all(entry.join("bdi")).unwrap();
        std::fs::write(entry.join("bdi/read_ahead_kb"), "2048\n").unwrap();
        assert_eq!(
            read_ahead_window_under(&sysfs, &subject),
            Some(ReadAheadWindow {
                bytes: 2_097_152,
                control_file: entry.join("bdi/read_ahead_kb")
            })
        );
    }

    #[test]
    fn only_windows_under_the_recommendation_are_reported_as_throttling() {
        let window = |bytes| ReadAheadWindow {
            bytes,
            control_file: PathBuf::new(),
        };
        assert!(window(131_072).throttles_weight_streaming());
        assert!(!window(RECOMMENDED_READ_AHEAD_BYTES).throttles_weight_streaming());
        assert!(!window(8_388_608).throttles_weight_streaming());
    }
}
