//! The storage properties that bound weight streaming.

#[cfg(unix)]
mod parallel_read;
#[cfg(unix)]
pub use parallel_read::{ParallelReadSource, read_parallel_into};

use std::{
    fs::File,
    path::{Path, PathBuf},
};

/// The system page alignment used for weight-reader partitions.
pub fn read_alignment() -> std::io::Result<Option<usize>> {
    #[cfg(unix)]
    {
        let value = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let page = usize::try_from(value).map_err(|_| {
            std::io::Error::other(format!("sysconf page alignment failed: {value}"))
        })?;
        if !page.is_power_of_two() {
            return Err(std::io::Error::other(format!(
                "invalid system page alignment: {page}"
            )));
        }
        Ok(Some(page))
    }
    #[cfg(not(unix))]
    {
        Ok(None)
    }
}

/// Checks sustained reads on the backing block device during calibration.
#[cfg(target_os = "linux")]
pub struct ReadHealth {
    device: PathBuf,
    counters: Vec<u64>,
    sampled: std::time::Instant,
}

#[cfg(target_os = "linux")]
impl ReadHealth {
    pub fn new(path: &Path) -> anyhow::Result<Self> {
        use anyhow::Context;
        let (major, minor) = device_numbers(path)
            .with_context(|| format!("cannot identify storage device for {}", path.display()))?;
        let device = std::fs::canonicalize(format!("/sys/dev/block/{major}:{minor}"))
            .with_context(|| format!("cannot resolve storage device {major}:{minor}"))?;
        let counters = Self::read(&device)?;
        Ok(Self {
            device,
            counters,
            sampled: std::time::Instant::now(),
        })
    }

    fn read(device: &Path) -> anyhow::Result<Vec<u64>> {
        use anyhow::Context;
        let path = device.join("stat");
        let counters = std::fs::read_to_string(&path)
            .with_context(|| format!("cannot read storage counters {}", path.display()))?
            .split_whitespace()
            .map(str::parse::<u64>)
            .collect::<Result<Vec<_>, _>>()
            .with_context(|| format!("invalid storage counters {}", path.display()))?;
        anyhow::ensure!(
            counters.len() >= 11,
            "incomplete storage counters {}",
            path.display()
        );
        Ok(counters)
    }

    pub fn check(&mut self) -> anyhow::Result<()> {
        let elapsed = self.sampled.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return Ok(());
        }
        let now = Self::read(&self.device)?;
        Self::validate(&self.device, &self.counters, &now, elapsed.as_secs_f64())?;
        self.counters = now;
        self.sampled = std::time::Instant::now();
        Ok(())
    }

    fn validate(device: &Path, before: &[u64], after: &[u64], seconds: f64) -> anyhow::Result<()> {
        use anyhow::Context;
        let delta = |index: usize| {
            after[index]
                .checked_sub(before[index])
                .with_context(|| format!("storage counter reset on {}", device.display()))
        };
        let reads = delta(0)? as f64;
        let bytes_per_second = delta(2)? as f64 * 512.0 / seconds;
        let await_ms = delta(3)? as f64 / reads.max(1.0);
        let queue = delta(10)? as f64 / (seconds * 1000.0);
        let busy = delta(9)? as f64 / (seconds * 1000.0);
        if reads > 0.0 && busy >= 0.8 {
            anyhow::ensure!(
                queue > 0.0,
                "missing busy queue time on {}",
                device.display()
            );
            anyhow::ensure!(
                bytes_per_second >= 300_000_000.0 && await_ms / queue <= 1.5,
                "storage degraded on {}: {bytes_per_second:.0} B/s, r_await {await_ms:.3} ms, aqu-sz {queue:.3}, service {:.3} ms",
                device.display(),
                await_ms / queue
            );
        }
        Ok(())
    }
}

/// Advise away clean file-cache pages in a byte range; zero length means through EOF.
pub(crate) fn advise_dropped(file: &File, offset: u64, len: u64) -> bool {
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        let (Ok(offset), Ok(len)) = (libc::off_t::try_from(offset), libc::off_t::try_from(len))
        else {
            return false;
        };
        if offset.checked_add(len).is_none() {
            return false;
        }
        unsafe {
            libc::posix_fadvise(file.as_raw_fd(), offset, len, libc::POSIX_FADV_DONTNEED) == 0
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (file, offset, len);
        false
    }
}

/// Kernel readahead window for the device backing a path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReadAheadWindow {
    pub bytes: u64,
    /// The file the window is written to, so a caller can name it exactly
    /// rather than making the reader derive the device numbers themselves.
    pub control_file: PathBuf,
}

/// Recommended minimum readahead window for weight streaming.
pub const RECOMMENDED_READ_AHEAD_BYTES: u64 = 2 * 1024 * 1024;

impl ReadAheadWindow {
    /// True when the window is small enough to bound streaming below what the
    /// device can deliver.
    pub const fn throttles_weight_streaming(&self) -> bool {
        self.bytes < RECOMMENDED_READ_AHEAD_BYTES
    }
}

/// Read the backing device's readahead window, or None when unavailable.
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

/// Atomically replace the destination while allowing shared readers; InvalidInput retries MoveFileEx.
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

/// Move source to destination with write-through durability; replace_existing controls destination replacement.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn calibration_rejects_sustained_degradation_without_rejecting_queue_depth() {
        let before = [0; 11];
        let mut after = [0; 11];
        after[0] = 4_000;
        after[2] = 1_000_000;
        after[3] = 40_000;
        after[9] = 1_000;
        after[10] = 10_000;
        let device = Path::new("test-device");
        ReadHealth::validate(device, &before, &after, 1.0).unwrap();
        after[0] = 475;
        after[2] = 121_600;
        after[3] = 7_016;
        after[9] = 888;
        after[10] = 7_020;
        let error = ReadHealth::validate(device, &before, &after, 1.0).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("test-device") && message.contains("62259200 B/s"));
        after[2] = 1_000_000;
        assert!(ReadHealth::validate(device, &before, &after, 1.0).is_err());
        after[3] = 475;
        after[2] = 121_600;
        assert!(ReadHealth::validate(device, &before, &after, 1.0).is_err());
        ReadHealth::validate(device, &before, &before, 1.0).unwrap();
        assert!(ReadHealth::validate(device, &after, &before, 1.0).is_err());
    }

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
