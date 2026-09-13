//! Directory snapshots and the completion rule a frame set is sealed under.
//!
//! A frame directory is read twice and the two snapshots must agree, so a
//! manifest can never describe a directory that changed while it was scanned.

use super::*;

pub(super) const MAX_COMPLETION_FILE_NAME_BYTES: usize = 128;

#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum CompletionRule {
    MustBeAbsent,
    Optional,
    MustBePresent,
}

pub(super) struct DirectorySnapshot {
    pub(super) frames: Vec<FileStat>,
    pub(super) completion: Option<FileStat>,
}

pub(super) fn validate_completion_file_name(file_name: &str) -> Result<()> {
    anyhow::ensure!(
        !file_name.is_empty() && file_name.len() <= MAX_COMPLETION_FILE_NAME_BYTES,
        "completion file name must contain 1..={MAX_COMPLETION_FILE_NAME_BYTES} bytes"
    );
    anyhow::ensure!(
        file_name
            .bytes()
            .all(|byte| { byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-') }),
        "completion file name must contain only ASCII letters, digits, dot, underscore, or hyphen"
    );
    anyhow::ensure!(
        file_name != "." && file_name != "..",
        "completion file name must name one file"
    );
    anyhow::ensure!(
        parse_frame_file_name(file_name).is_none(),
        "completion file name must not collide with a PNG frame name"
    );
    Ok(())
}

pub(super) fn inspect_directory(directory: &Path) -> Result<FileStat> {
    let metadata = FileStat::of_path(directory).with_context(|| {
        format!(
            "failed to inspect PNG frame directory {}",
            directory.display()
        )
    })?;
    anyhow::ensure!(
        metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
        "PNG frame directory is not a non-symlink directory: {}",
        directory.display()
    );
    Ok(metadata)
}

pub(super) fn scan_directory(
    directory: &Path,
    completion_file_name: &str,
    expected_frame_count: u64,
    completion_rule: CompletionRule,
) -> Result<DirectorySnapshot> {
    let count = usize::try_from(expected_frame_count).context("PNG frame count exceeds usize")?;
    let mut frames = Vec::with_capacity(count);
    frames.resize_with(count, || None);
    let mut completion = None;
    let entries = fs::read_dir(directory)
        .with_context(|| format!("failed to read PNG frame directory {}", directory.display()))?;
    let maximum_entries = count
        .checked_add(1)
        .context("PNG frame directory entry bound overflow")?;
    let mut entry_count = 0usize;
    for entry in entries {
        entry_count = entry_count
            .checked_add(1)
            .context("PNG frame directory entry count overflow")?;
        anyhow::ensure!(
            entry_count <= maximum_entries,
            "PNG frame directory contains more than {maximum_entries} allowed entries"
        );
        let entry = entry.with_context(|| {
            format!(
                "failed to inspect entry in PNG frame directory {}",
                directory.display()
            )
        })?;
        let file_name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("PNG frame directory contains a non-UTF-8 file name"))?;
        let metadata = FileStat::of_path(&entry.path())
            .with_context(|| format!("failed to inspect PNG frame directory entry {file_name}"))?;
        anyhow::ensure!(
            metadata.file_type().is_file(),
            "PNG frame directory entry is not a regular non-symlink file: {file_name}"
        );
        if file_name == completion_file_name {
            anyhow::ensure!(completion.is_none(), "duplicate completion file entry");
            completion = Some(metadata);
            continue;
        }
        let index = parse_frame_file_name(&file_name)
            .with_context(|| format!("unexpected PNG frame directory entry: {file_name}"))?;
        anyhow::ensure!(
            index < count,
            "unexpected PNG frame outside declared range: {file_name}"
        );
        anyhow::ensure!(
            frames[index].is_none(),
            "duplicate PNG frame directory entry: {file_name}"
        );
        frames[index] = Some(metadata);
    }
    if completion_rule == CompletionRule::MustBeAbsent {
        anyhow::ensure!(
            completion.is_none(),
            "completion file already exists: {completion_file_name}"
        );
    } else if completion_rule == CompletionRule::MustBePresent {
        anyhow::ensure!(
            completion.is_some(),
            "completion file is missing: {completion_file_name}"
        );
    }
    let frames = frames
        .into_iter()
        .enumerate()
        .map(|(index, metadata)| {
            metadata.with_context(|| format!("missing PNG frame {}", frame_file_name(index)))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(DirectorySnapshot { frames, completion })
}

pub(super) fn ensure_snapshots_match(
    before: &DirectorySnapshot,
    after: &DirectorySnapshot,
) -> Result<()> {
    anyhow::ensure!(
        before.frames.len() == after.frames.len(),
        "PNG frame directory changed while being inspected"
    );
    for (index, (before, after)) in before.frames.iter().zip(&after.frames).enumerate() {
        anyhow::ensure!(
            unchanged_regular_file(before, after),
            "PNG frame changed while being inspected: {}",
            frame_file_name(index)
        );
    }
    match (&before.completion, &after.completion) {
        (None, None) => {}
        (Some(before), Some(after)) => {
            anyhow::ensure!(
                unchanged_regular_file(before, after),
                "completion file changed while PNG frames were being inspected"
            );
        }
        _ => {
            anyhow::bail!("completion file presence changed while PNG frames were being inspected")
        }
    }
    Ok(())
}

pub(super) fn unchanged_regular_file(before: &FileStat, after: &FileStat) -> bool {
    before.file_type().is_file()
        && after.file_type().is_file()
        && identifies_same_file(before, after)
        && before.len() == after.len()
        && before.modified().ok() == after.modified().ok()
        && change_marker(before) == change_marker(after)
}

pub(super) fn ensure_directory_unchanged(directory: &Path, before: &FileStat) -> Result<()> {
    let after = inspect_directory(directory)?;
    anyhow::ensure!(
        same_directory(before, &after),
        "PNG frame directory changed while being inspected: {}",
        directory.display()
    );
    Ok(())
}

/// Directories cannot use `identifies_same_file`, which requires a regular
/// file; the identity comparison is the same one.
pub(super) fn same_directory(before: &FileStat, after: &FileStat) -> bool {
    before.is_dir() && after.is_dir() && before.identifies_same_file_as(after)
}

pub(super) fn frame_file_name(index: usize) -> String {
    format!("frame_{index:05}.png")
}

pub(super) fn parse_frame_file_name(file_name: &str) -> Option<usize> {
    let bytes = file_name.as_bytes();
    if bytes.len() != 15 || &bytes[..6] != b"frame_" || &bytes[11..] != b".png" {
        return None;
    }
    bytes[6..11].iter().try_fold(0usize, |value, byte| {
        if !byte.is_ascii_digit() {
            return None;
        }
        value.checked_mul(10)?.checked_add(usize::from(byte - b'0'))
    })
}
