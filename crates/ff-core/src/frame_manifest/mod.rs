use crate::artifact::{FileStat, change_marker, identifies_same_file, read_artifact_snapshot};

use anyhow::{Context, Result};

use serde::{Deserialize, Serialize};

use std::{
    fs::{self},
    io::Cursor,
    path::Path,
};

mod decode;
mod scan;

use decode::*;
use scan::*;

pub const PNG_FRAME_SET_MANIFEST_SCHEMA_VERSION: u32 = 1;

pub const MAX_PNG_FRAME_SET_MANIFEST_JSON_BYTES: usize = 32 * 1024 * 1024;

pub const MAX_PNG_FRAME_COUNT: u64 = 100_000;

pub const MAX_PNG_FRAME_DIMENSION: u32 = 8192;

pub const MAX_PNG_FRAME_BYTES: u64 = 64 * 1024 * 1024;

pub const MAX_PNG_DECODED_FRAME_BYTES: u64 = 256 * 1024 * 1024;

pub const MAX_PNG_FRAME_SET_BYTES: u64 = 4 * 1024 * 1024 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PngFrameMember {
    pub file_name: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PngFrameSetManifest {
    pub schema_version: u32,
    pub frame_count: u64,
    pub width: u32,
    pub height: u32,
    pub total_bytes: u64,
    pub frames: Vec<PngFrameMember>,
}

impl PngFrameSetManifest {
    pub fn collect(
        directory: impl AsRef<Path>,
        completion_file_name: &str,
        expected_frame_count: usize,
    ) -> Result<Self> {
        let directory = directory.as_ref();
        let expected_frame_count = validate_frame_count(expected_frame_count)?;
        validate_completion_file_name(completion_file_name)?;
        let directory_before = inspect_directory(directory)?;
        let before = scan_directory(
            directory,
            completion_file_name,
            expected_frame_count,
            CompletionRule::MustBeAbsent,
        )?;
        let frame_capacity =
            usize::try_from(expected_frame_count).context("PNG frame count exceeds usize")?;
        let mut frames = Vec::with_capacity(frame_capacity);
        let mut total_bytes = 0u64;
        let mut dimensions = None;
        for index in 0..expected_frame_count {
            let index = usize::try_from(index).context("PNG frame index exceeds usize")?;
            let file_name = frame_file_name(index);
            let path = directory.join(&file_name);
            let snapshot = read_artifact_snapshot(&path, MAX_PNG_FRAME_BYTES)
                .with_context(|| format!("failed to read PNG frame {file_name}"))?;
            let frame_dimensions = decode_png_rgb8(&file_name, &snapshot.bytes)?;
            match dimensions {
                None => dimensions = Some(frame_dimensions),
                Some(expected) => anyhow::ensure!(
                    frame_dimensions == expected,
                    "PNG frame {file_name} dimensions {}x{} do not match {}x{}",
                    frame_dimensions.0,
                    frame_dimensions.1,
                    expected.0,
                    expected.1
                ),
            }
            let bytes =
                u64::try_from(snapshot.bytes.len()).context("PNG frame size exceeds u64")?;
            validate_frame_bytes(&file_name, bytes)?;
            total_bytes = total_bytes
                .checked_add(bytes)
                .context("PNG frame-set byte total overflow")?;
            anyhow::ensure!(
                total_bytes <= MAX_PNG_FRAME_SET_BYTES,
                "PNG frame-set size exceeds {MAX_PNG_FRAME_SET_BYTES} bytes"
            );
            let after_digest = FileStat::of_path(&path)
                .with_context(|| format!("failed to re-inspect PNG frame {file_name}"))?;
            anyhow::ensure!(
                unchanged_regular_file(&before.frames[index], &after_digest),
                "PNG frame changed while collecting manifest: {file_name}"
            );
            frames.push(PngFrameMember { file_name, bytes });
        }
        let after = scan_directory(
            directory,
            completion_file_name,
            expected_frame_count,
            CompletionRule::MustBeAbsent,
        )?;
        ensure_snapshots_match(&before, &after)?;
        ensure_directory_unchanged(directory, &directory_before)?;
        let (width, height) = dimensions.context("PNG frame set has no dimensions")?;
        let manifest = Self {
            schema_version: PNG_FRAME_SET_MANIFEST_SCHEMA_VERSION,
            frame_count: expected_frame_count,
            width,
            height,
            total_bytes,
            frames,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == PNG_FRAME_SET_MANIFEST_SCHEMA_VERSION,
            "unsupported PNG frame-set manifest schema {}; this build supports schema {}",
            self.schema_version,
            PNG_FRAME_SET_MANIFEST_SCHEMA_VERSION
        );
        anyhow::ensure!(
            (1..=MAX_PNG_FRAME_COUNT).contains(&self.frame_count),
            "PNG frame count must be in 1..={MAX_PNG_FRAME_COUNT}"
        );
        validate_dimensions(self.width, self.height)?;
        let frame_count =
            usize::try_from(self.frame_count).context("PNG frame count exceeds usize")?;
        anyhow::ensure!(
            self.frames.len() == frame_count,
            "PNG frame count {} does not match {} manifest members",
            self.frame_count,
            self.frames.len()
        );
        let mut total_bytes = 0u64;
        for (index, frame) in self.frames.iter().enumerate() {
            let expected_name = frame_file_name(index);
            anyhow::ensure!(
                frame.file_name == expected_name,
                "PNG frame member {} must be named {expected_name}, found {}",
                index,
                frame.file_name
            );
            validate_frame_bytes(&frame.file_name, frame.bytes)?;
            total_bytes = total_bytes
                .checked_add(frame.bytes)
                .context("PNG frame-set byte total overflow")?;
            anyhow::ensure!(
                total_bytes <= MAX_PNG_FRAME_SET_BYTES,
                "PNG frame-set size exceeds {MAX_PNG_FRAME_SET_BYTES} bytes"
            );
        }
        anyhow::ensure!(
            self.total_bytes == total_bytes,
            "PNG frame-set total_bytes {} does not match derived total {total_bytes}",
            self.total_bytes
        );
        let json =
            serde_json::to_vec(self).context("failed to serialize PNG frame-set manifest")?;
        anyhow::ensure!(
            json.len() <= MAX_PNG_FRAME_SET_MANIFEST_JSON_BYTES,
            "PNG frame-set manifest JSON is {} bytes, exceeding the {}-byte limit",
            json.len(),
            MAX_PNG_FRAME_SET_MANIFEST_JSON_BYTES
        );
        Ok(())
    }

    pub fn parse_spec_json(bytes: &[u8]) -> Result<Self> {
        anyhow::ensure!(!bytes.is_empty(), "PNG frame-set manifest JSON is empty");
        anyhow::ensure!(
            bytes.len() <= MAX_PNG_FRAME_SET_MANIFEST_JSON_BYTES,
            "PNG frame-set manifest JSON is {} bytes, exceeding the {}-byte limit",
            bytes.len(),
            MAX_PNG_FRAME_SET_MANIFEST_JSON_BYTES
        );
        let manifest: Self =
            serde_json::from_slice(bytes).context("invalid PNG frame-set manifest JSON")?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn from_canonical_json(bytes: &[u8]) -> Result<Self> {
        let manifest = Self::parse_spec_json(bytes)?;
        anyhow::ensure!(
            bytes == manifest.canonical_json()?,
            "PNG frame-set manifest JSON is not in canonical schema-1 encoding"
        );
        Ok(manifest)
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let bytes =
            serde_json::to_vec(self).context("failed to serialize PNG frame-set manifest")?;
        anyhow::ensure!(
            bytes.len() <= MAX_PNG_FRAME_SET_MANIFEST_JSON_BYTES,
            "PNG frame-set manifest JSON is {} bytes, exceeding the {}-byte limit",
            bytes.len(),
            MAX_PNG_FRAME_SET_MANIFEST_JSON_BYTES
        );
        Ok(bytes)
    }

    pub fn verify_payload(
        &self,
        directory: impl AsRef<Path>,
        completion_file_name: &str,
    ) -> Result<()> {
        self.verify_directory_inner(
            directory.as_ref(),
            completion_file_name,
            CompletionRule::Optional,
            false,
        )
    }

    /// Verify the directory and require the completion manifest to be present
    /// and to equal this manifest's own canonical bytes.
    pub fn verify_completed_directory(
        &self,
        directory: impl AsRef<Path>,
        completion_file_name: &str,
    ) -> Result<()> {
        self.verify_directory_inner(
            directory.as_ref(),
            completion_file_name,
            CompletionRule::MustBePresent,
            true,
        )
    }

    fn verify_directory_inner(
        &self,
        directory: &Path,
        completion_file_name: &str,
        completion_rule: CompletionRule,
        verify_completion_bytes: bool,
    ) -> Result<()> {
        self.validate()?;
        validate_completion_file_name(completion_file_name)?;
        let directory_before = inspect_directory(directory)?;
        let before = scan_directory(
            directory,
            completion_file_name,
            self.frame_count,
            completion_rule,
        )?;
        if verify_completion_bytes {
            let path = directory.join(completion_file_name);
            let snapshot = read_artifact_snapshot(
                &path,
                u64::try_from(MAX_PNG_FRAME_SET_MANIFEST_JSON_BYTES)
                    .context("PNG frame-set manifest JSON limit exceeds u64")?,
            )
            .with_context(|| {
                format!("failed to read completion manifest {completion_file_name}")
            })?;
            anyhow::ensure!(
                snapshot.bytes == self.canonical_json()?,
                "completion manifest bytes do not equal the canonical PNG frame-set manifest"
            );
            let after_read = FileStat::of_path(&path).with_context(|| {
                format!("failed to re-inspect completion manifest {completion_file_name}")
            })?;
            anyhow::ensure!(
                before
                    .completion
                    .as_ref()
                    .is_some_and(|metadata| unchanged_regular_file(metadata, &after_read)),
                "completion manifest changed while verifying directory"
            );
        }
        let mut total_bytes = 0u64;
        for (index, frame) in self.frames.iter().enumerate() {
            let path = directory.join(&frame.file_name);
            let snapshot = read_artifact_snapshot(&path, MAX_PNG_FRAME_BYTES)
                .with_context(|| format!("failed to read PNG frame {}", frame.file_name))?;
            let bytes =
                u64::try_from(snapshot.bytes.len()).context("PNG frame size exceeds u64")?;
            anyhow::ensure!(
                bytes == frame.bytes,
                "PNG frame {} is {bytes} bytes, and the manifest records {}",
                frame.file_name,
                frame.bytes
            );
            let dimensions = decode_png_rgb8(&frame.file_name, &snapshot.bytes)?;
            anyhow::ensure!(
                dimensions == (self.width, self.height),
                "PNG frame {} dimensions {}x{} do not match manifest dimensions {}x{}",
                frame.file_name,
                dimensions.0,
                dimensions.1,
                self.width,
                self.height
            );
            total_bytes = total_bytes
                .checked_add(bytes)
                .context("PNG frame-set byte total overflow")?;
            let after_digest = FileStat::of_path(&path)
                .with_context(|| format!("failed to re-inspect PNG frame {}", frame.file_name))?;
            anyhow::ensure!(
                unchanged_regular_file(&before.frames[index], &after_digest),
                "PNG frame changed while verifying manifest: {}",
                frame.file_name
            );
        }
        anyhow::ensure!(
            total_bytes == self.total_bytes,
            "verified PNG frame-set total {total_bytes} does not match manifest total {}",
            self.total_bytes
        );
        let after = scan_directory(
            directory,
            completion_file_name,
            self.frame_count,
            completion_rule,
        )?;
        ensure_snapshots_match(&before, &after)?;
        ensure_directory_unchanged(directory, &directory_before)?;
        Ok(())
    }
}

fn validate_frame_count(frame_count: usize) -> Result<u64> {
    let frame_count = u64::try_from(frame_count).context("PNG frame count exceeds u64")?;
    anyhow::ensure!(
        (1..=MAX_PNG_FRAME_COUNT).contains(&frame_count),
        "PNG frame count must be in 1..={MAX_PNG_FRAME_COUNT}"
    );
    Ok(frame_count)
}

fn validate_frame_bytes(file_name: &str, bytes: u64) -> Result<()> {
    anyhow::ensure!(
        (1..=MAX_PNG_FRAME_BYTES).contains(&bytes),
        "PNG frame {file_name} size must be in 1..={MAX_PNG_FRAME_BYTES} bytes"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::fs::File;

    fn png_bytes(
        width: u32,
        height: u32,
        color: png::ColorType,
        depth: png::BitDepth,
        animated: bool,
    ) -> Vec<u8> {
        let sample_bytes = match depth {
            png::BitDepth::Eight => 1,
            png::BitDepth::Sixteen => 2,
            _ => panic!("test encoder only supports byte-aligned depths"),
        };
        let length = usize::try_from(width).unwrap()
            * usize::try_from(height).unwrap()
            * color.samples()
            * sample_bytes;
        let pixels = (0..length)
            .map(|index| u8::try_from(index % 251).unwrap())
            .collect::<Vec<_>>();
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, width, height);
            encoder.set_color(color);
            encoder.set_depth(depth);
            if animated {
                encoder.set_animated(2, 0).unwrap();
            }
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&pixels).unwrap();
            if animated {
                writer.write_image_data(&pixels).unwrap();
            }
        }
        bytes
    }

    fn rgb_png(width: u32, height: u32) -> Vec<u8> {
        png_bytes(
            width,
            height,
            png::ColorType::Rgb,
            png::BitDepth::Eight,
            false,
        )
    }

    fn write_frames(directory: &Path, dimensions: &[(u32, u32)]) {
        for (index, &(width, height)) in dimensions.iter().enumerate() {
            fs::write(
                directory.join(frame_file_name(index)),
                rgb_png(width, height),
            )
            .unwrap();
        }
    }

    fn sample_manifest() -> PngFrameSetManifest {
        PngFrameSetManifest {
            schema_version: PNG_FRAME_SET_MANIFEST_SCHEMA_VERSION,
            frame_count: 1,
            width: 2,
            height: 1,
            total_bytes: 3,
            frames: vec![PngFrameMember {
                file_name: "frame_00000.png".to_owned(),
                bytes: 3,
            }],
        }
    }

    fn write_completion(directory: &Path, manifest: &PngFrameSetManifest, bytes: &[u8]) {
        let path = directory.join("frames.manifest.json");
        fs::write(&path, bytes).unwrap();
        assert!(!bytes.is_empty());
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn collects_decoded_rgb8_members_and_derived_fields() {
        let directory = tempfile::tempdir().unwrap();
        write_frames(directory.path(), &[(2, 1), (2, 1)]);
        let manifest =
            PngFrameSetManifest::collect(directory.path(), "frames.manifest.json", 2).unwrap();
        assert_eq!(manifest.schema_version, 1);
        assert_eq!(manifest.frame_count, 2);
        assert_eq!((manifest.width, manifest.height), (2, 1));
        assert_eq!(
            manifest.total_bytes,
            manifest.frames.iter().map(|frame| frame.bytes).sum::<u64>()
        );
        assert_eq!(manifest.frames[0].file_name, "frame_00000.png");
        assert_eq!(manifest.frames[1].file_name, "frame_00001.png");
        manifest
            .verify_payload(directory.path(), "frames.manifest.json")
            .unwrap();
    }

    #[test]
    fn canonical_json_is_locked_and_round_trips() {
        let manifest = sample_manifest();
        let json = manifest.canonical_json().unwrap();
        assert_eq!(
            String::from_utf8(json.clone()).unwrap(),
            "{\"schema_version\":1,\"frame_count\":1,\"width\":2,\"height\":1,\
             \"total_bytes\":3,\"frames\":[{\"file_name\":\"frame_00000.png\",\"bytes\":3}]}"
        );
        assert_eq!(
            PngFrameSetManifest::from_canonical_json(&json).unwrap(),
            manifest
        );
    }

    #[test]
    fn canonical_parser_rejects_noncanonical_unknown_and_missing_fields() {
        let manifest = sample_manifest();
        let pretty = serde_json::to_vec_pretty(&manifest).unwrap();
        assert!(
            PngFrameSetManifest::from_canonical_json(&pretty)
                .unwrap_err()
                .to_string()
                .contains("not in canonical")
        );
        let mut value = serde_json::to_value(&manifest).unwrap();
        value["extra"] = serde_json::json!(true);
        assert!(
            PngFrameSetManifest::parse_spec_json(&serde_json::to_vec(&value).unwrap()).is_err()
        );
        value.as_object_mut().unwrap().remove("width");
        value.as_object_mut().unwrap().remove("extra");
        assert!(
            PngFrameSetManifest::parse_spec_json(&serde_json::to_vec(&value).unwrap()).is_err()
        );
    }

    #[test]
    fn validation_rejects_order_count_dimensions_and_total_mismatches() {
        let mut manifest = sample_manifest();
        manifest.frames[0].file_name = "frame_00001.png".to_owned();
        assert!(manifest.validate().is_err());

        let mut manifest = sample_manifest();
        manifest.frame_count = 2;
        assert!(manifest.validate().is_err());

        let mut manifest = sample_manifest();
        manifest.width = 0;
        assert!(manifest.validate().is_err());

        let mut manifest = sample_manifest();
        manifest.height = MAX_PNG_FRAME_DIMENSION + 1;
        assert!(manifest.validate().is_err());

        let mut manifest = sample_manifest();
        manifest.total_bytes = 4;
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn collect_rejects_missing_and_extra_frames() {
        let missing = tempfile::tempdir().unwrap();
        write_frames(missing.path(), &[(2, 1)]);
        assert!(
            PngFrameSetManifest::collect(missing.path(), "frames.manifest.json", 2)
                .unwrap_err()
                .to_string()
                .contains("missing PNG frame frame_00001.png")
        );

        let extra = tempfile::tempdir().unwrap();
        write_frames(extra.path(), &[(2, 1)]);
        fs::write(extra.path().join("notes.txt"), b"extra").unwrap();
        assert!(
            PngFrameSetManifest::collect(extra.path(), "frames.manifest.json", 1)
                .unwrap_err()
                .to_string()
                .contains("unexpected PNG frame directory entry")
        );

        let out_of_range = tempfile::tempdir().unwrap();
        write_frames(out_of_range.path(), &[(2, 1), (2, 1)]);
        assert!(
            PngFrameSetManifest::collect(out_of_range.path(), "frames.manifest.json", 1).is_err()
        );
    }

    #[test]
    fn collect_rejects_arbitrary_corrupt_animated_and_non_rgb8_payloads() {
        let arbitrary = tempfile::tempdir().unwrap();
        fs::write(arbitrary.path().join("frame_00000.png"), b"not a PNG").unwrap();
        assert!(PngFrameSetManifest::collect(arbitrary.path(), "frames.manifest.json", 1).is_err());

        let corrupt = tempfile::tempdir().unwrap();
        let mut bytes = rgb_png(2, 1);
        let index = bytes.len() - 13;
        bytes[index] ^= 0x80;
        fs::write(corrupt.path().join("frame_00000.png"), bytes).unwrap();
        assert!(PngFrameSetManifest::collect(corrupt.path(), "frames.manifest.json", 1).is_err());

        let animated = tempfile::tempdir().unwrap();
        fs::write(
            animated.path().join("frame_00000.png"),
            png_bytes(2, 1, png::ColorType::Rgb, png::BitDepth::Eight, true),
        )
        .unwrap();
        assert!(PngFrameSetManifest::collect(animated.path(), "frames.manifest.json", 1).is_err());

        for (color, depth) in [
            (png::ColorType::Grayscale, png::BitDepth::Eight),
            (png::ColorType::Rgb, png::BitDepth::Sixteen),
            (png::ColorType::Rgba, png::BitDepth::Eight),
        ] {
            let directory = tempfile::tempdir().unwrap();
            fs::write(
                directory.path().join("frame_00000.png"),
                png_bytes(2, 1, color, depth, false),
            )
            .unwrap();
            assert!(
                PngFrameSetManifest::collect(directory.path(), "frames.manifest.json", 1).is_err()
            );
        }
    }

    #[test]
    fn collect_rejects_inconsistent_or_excessive_dimensions_and_encoded_size() {
        let inconsistent = tempfile::tempdir().unwrap();
        write_frames(inconsistent.path(), &[(2, 1), (1, 2)]);
        assert!(
            PngFrameSetManifest::collect(inconsistent.path(), "frames.manifest.json", 2).is_err()
        );

        let wide = tempfile::tempdir().unwrap();
        write_frames(wide.path(), &[(MAX_PNG_FRAME_DIMENSION + 1, 1)]);
        assert!(PngFrameSetManifest::collect(wide.path(), "frames.manifest.json", 1).is_err());

        let oversized = tempfile::tempdir().unwrap();
        let path = oversized.path().join("frame_00000.png");
        File::create(&path)
            .unwrap()
            .set_len(MAX_PNG_FRAME_BYTES + 1)
            .unwrap();
        assert!(PngFrameSetManifest::collect(oversized.path(), "frames.manifest.json", 1).is_err());
    }

    #[test]
    fn payload_verification_allows_missing_completion_but_checks_dimensions_and_bytes() {
        let directory = tempfile::tempdir().unwrap();
        write_frames(directory.path(), &[(2, 1)]);
        let manifest =
            PngFrameSetManifest::collect(directory.path(), "frames.manifest.json", 1).unwrap();
        manifest
            .verify_payload(directory.path(), "frames.manifest.json")
            .unwrap();
        assert!(
            manifest
                .verify_completed_directory(directory.path(), "frames.manifest.json")
                .unwrap_err()
                .to_string()
                .contains("completion file is missing")
        );

        let mut wrong_dimensions = manifest.clone();
        wrong_dimensions.width = 1;
        assert!(
            wrong_dimensions
                .verify_payload(directory.path(), "frames.manifest.json")
                .is_err()
        );

        let mut changed = rgb_png(2, 1);
        let index = changed.len() - 20;
        changed[index] ^= 1;
        fs::write(directory.path().join("frame_00000.png"), changed).unwrap();
        assert!(
            manifest
                .verify_payload(directory.path(), "frames.manifest.json")
                .is_err()
        );
    }

    #[test]
    fn completed_directory_requires_the_exact_canonical_marker() {
        let directory = tempfile::tempdir().unwrap();
        write_frames(directory.path(), &[(2, 1)]);
        let manifest =
            PngFrameSetManifest::collect(directory.path(), "frames.manifest.json", 1).unwrap();
        let canonical = manifest.canonical_json().unwrap();
        write_completion(directory.path(), &manifest, &canonical);
        manifest
            .verify_completed_directory(directory.path(), "frames.manifest.json")
            .unwrap();

        // A marker holding the same manifest in another spelling is still not
        // the canonical bytes this manifest publishes.
        fs::remove_file(directory.path().join("frames.manifest.json")).unwrap();
        let pretty = serde_json::to_vec_pretty(&manifest).unwrap();
        write_completion(directory.path(), &manifest, &pretty);
        assert!(
            manifest
                .verify_completed_directory(directory.path(), "frames.manifest.json")
                .unwrap_err()
                .to_string()
                .contains("canonical")
        );

        fs::remove_file(directory.path().join("frames.manifest.json")).unwrap();
        write_completion(directory.path(), &manifest, b"wrong marker");
        assert!(
            manifest
                .verify_completed_directory(directory.path(), "frames.manifest.json")
                .unwrap_err()
                .to_string()
                .contains("canonical")
        );
    }

    #[test]
    fn collect_rejects_existing_completion() {
        let directory = tempfile::tempdir().unwrap();
        write_frames(directory.path(), &[(2, 1)]);
        fs::write(directory.path().join("frames.manifest.json"), b"completion").unwrap();
        assert!(
            PngFrameSetManifest::collect(directory.path(), "frames.manifest.json", 1)
                .unwrap_err()
                .to_string()
                .contains("completion file already exists")
        );
    }

    #[test]
    fn invalid_completion_names_are_rejected() {
        let directory = tempfile::tempdir().unwrap();
        write_frames(directory.path(), &[(2, 1)]);
        for file_name in [
            "",
            "../completion.json",
            "nested/completion.json",
            "frame_00000.png",
        ] {
            assert!(PngFrameSetManifest::collect(directory.path(), file_name, 1).is_err());
        }
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_frame_directory_entry_is_rejected() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("target.png"), rgb_png(2, 1)).unwrap();
        symlink(
            directory.path().join("target.png"),
            directory.path().join("frame_00000.png"),
        )
        .unwrap();
        assert!(PngFrameSetManifest::collect(directory.path(), "frames.manifest.json", 1).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn completed_directory_rejects_symlinked_completion_and_directory() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        write_frames(directory.path(), &[(2, 1)]);
        let manifest =
            PngFrameSetManifest::collect(directory.path(), "frames.manifest.json", 1).unwrap();
        let canonical = manifest.canonical_json().unwrap();
        let target = directory.path().join("target.json");
        fs::write(&target, &canonical).unwrap();
        symlink(&target, directory.path().join("frames.manifest.json")).unwrap();
        assert!(
            manifest
                .verify_completed_directory(directory.path(), "frames.manifest.json")
                .is_err()
        );

        let parent = tempfile::tempdir().unwrap();
        symlink(directory.path(), parent.path().join("frames")).unwrap();
        assert!(
            manifest
                .verify_payload(parent.path().join("frames"), "completion.json")
                .is_err()
        );
    }

    #[test]
    fn validation_rejects_zero_and_unbounded_values() {
        let mut manifest = sample_manifest();
        manifest.frames[0].bytes = 0;
        manifest.total_bytes = 0;
        assert!(manifest.validate().is_err());

        let mut manifest = sample_manifest();
        manifest.frames[0].bytes = MAX_PNG_FRAME_BYTES + 1;
        manifest.total_bytes = MAX_PNG_FRAME_BYTES + 1;
        assert!(manifest.validate().is_err());

        let mut manifest = sample_manifest();
        manifest.frame_count = MAX_PNG_FRAME_COUNT + 1;
        assert!(manifest.validate().is_err());

        assert_eq!(
            validate_dimensions(MAX_PNG_FRAME_DIMENSION, MAX_PNG_FRAME_DIMENSION).unwrap(),
            201_326_592
        );
    }
}
