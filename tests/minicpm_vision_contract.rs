use anyhow::{Context, Result, ensure};
use flyingfish::{
    model_spec::{Architecture, ModelSpec},
    serve::detok::Shared,
    serve::image::{
        PreparedInput, Processor, RgbImage, VideoMetadata, VideoOptions, sample_frame_indices,
    },
    vision_input::{ImageGrid, merge_groups, nearest_position_ids, window_gather},
};
use safetensors::{Dtype, SafeTensors};
use std::path::PathBuf;

const CASES: [&str; 10] = [
    "no-slice-224",
    "wide-1xn",
    "tall-nx1",
    "grid-3x3",
    "aspect-20x1",
    "extreme-80x1",
    "odd-979x653",
    "odd-1000x700",
    "multi-two",
    "many-11",
];

/// Reference-only preprocess noise floor: torchvision vs PIL processor, max 0.0156863257 normalized
/// (2 u8 levels), `evidence/T0-ref/preprocess_noise_floor.json` sha256
/// `3431df00de14adfbd0b50c5ef650319523412b8278cce25fa2fb9ef115ea02c1`; the 4·EPS slack covers
/// f32 accumulation, not candidate measurements.
const PIXEL_TOLERANCE: f32 = 4.0 / 255.0 + 4.0 * f32::EPSILON;

fn tensor_values<T, const N: usize>(
    tensors: &SafeTensors<'_>,
    name: &str,
    dtype: Dtype,
    convert: impl Fn([u8; N]) -> T,
) -> Result<Vec<T>> {
    let tensor = tensors.tensor(name)?;
    ensure!(tensor.dtype() == dtype, "{name} must be {dtype:?}");
    Ok(tensor
        .data()
        .as_chunks::<N>()
        .0
        .iter()
        .map(|v| convert(*v))
        .collect())
}

#[test]
#[ignore = "CPU-only external fixtures; set FF_MODEL and FF_MINICPM_FIXTURE_DIR"]
fn processor_layout_pixels_and_canvas_match_reference() -> Result<()> {
    let model = PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
    let fixtures = PathBuf::from(
        std::env::var_os("FF_MINICPM_FIXTURE_DIR").context("FF_MINICPM_FIXTURE_DIR missing")?,
    );
    let spec = ModelSpec::read(&model)?;
    ensure!(
        spec.architecture == Architecture::MiniCpmV47,
        "fixtures require the MiniCPM spec"
    );
    let tokenizer = Shared::load(&model)?;
    let processor =
        Processor::load(&model, Some(&tokenizer))?.context("MiniCPM image processor")?;
    let mut worst_pixels = 0.0_f32;
    for case in CASES {
        let path = fixtures.join(format!("{case}.safetensors"));
        let bytes = std::fs::read(&path).with_context(|| format!("open {}", path.display()))?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        let count = usize::try_from(
            tensor_values(&tensors, "image_count", Dtype::I64, i64::from_le_bytes)?[0],
        )?;
        let widths = tensor_values(&tensors, "image_widths", Dtype::I64, i64::from_le_bytes)?;
        let heights = tensor_values(&tensors, "image_heights", Dtype::I64, i64::from_le_bytes)?;
        let mut images = Vec::with_capacity(count);
        for index in 0..count {
            let tensor = tensors.tensor(&format!("rgb_{index}"))?;
            ensure!(tensor.dtype() == Dtype::U8, "{case} rgb_{index} must be U8");
            let width = usize::try_from(widths[index])?;
            let height = usize::try_from(heights[index])?;
            ensure!(
                tensor.shape() == [height, width, 3],
                "{case} rgb_{index} shape"
            );
            images.push(processor.prepare_rgb(&RgbImage {
                width,
                height,
                pixels: tensor.data().to_vec(),
            })?);
        }
        let prompt: Vec<u32> =
            tensor_values(&tensors, "prompt_ids", Dtype::I64, i64::from_le_bytes)?
                .into_iter()
                .map(u32::try_from)
                .collect::<std::result::Result<_, _>>()?;
        let expected_ids: Vec<u32> =
            tensor_values(&tensors, "input_ids", Dtype::I64, i64::from_le_bytes)?
                .into_iter()
                .map(u32::try_from)
                .collect::<std::result::Result<_, _>>()?;
        let capacity = expected_ids.len() + 64;
        let (expanded, prepared) = processor.expand(&prompt, images, Vec::new(), capacity)?;
        assert_eq!(expanded, expected_ids, "{case} ids");

        let expected_modality: Vec<u8> = tensor_values(
            &tensors,
            "mm_token_type_ids",
            Dtype::I64,
            i64::from_le_bytes,
        )?
        .into_iter()
        .map(u8::try_from)
        .collect::<std::result::Result<_, _>>()?;
        assert_eq!(prepared.modality_ids, expected_modality, "{case} modality");

        let target_sizes = tensor_values(&tensors, "target_sizes", Dtype::I64, i64::from_le_bytes)?;
        let expected_grids: Vec<ImageGrid> = target_sizes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|size| {
                Ok(ImageGrid {
                    t: 1,
                    h: u32::try_from(size[0])?,
                    w: u32::try_from(size[1])?,
                })
            })
            .collect::<Result<_>>()?;
        assert_eq!(prepared.grids, expected_grids, "{case} grids");

        let reference = tensor_values(
            &tensors,
            "pixel_values_fp32",
            Dtype::F32,
            f32::from_le_bytes,
        )?;
        assert_eq!(
            prepared.pixel_values.len(),
            reference.len(),
            "{case} pixel count"
        );
        let mut worst = 0.0_f32;
        let mut differing = 0usize;
        for (actual, expected) in prepared.pixel_values.iter().zip(&reference) {
            worst = worst.max((actual - expected).abs());
            differing += usize::from(actual.to_bits() != expected.to_bits());
        }
        worst_pixels = worst_pixels.max(worst);
        println!(
            "{}",
            serde_json::json!({
                "case": case,
                "ids": expanded.len(),
                "crops": prepared.grids.len(),
                "pixels": reference.len(),
                "pixel_max_abs": worst,
                "pixel_different": differing,
            })
        );
        assert!(
            worst <= PIXEL_TOLERANCE,
            "{case} pixel error {worst} exceeds {PIXEL_TOLERANCE}"
        );

        let plan = prepared.input(&expanded).validate(&spec, capacity)?;
        let positions = tensor_values(&tensors, "positions", Dtype::I64, i64::from_le_bytes)?;
        let len = expanded.len();
        ensure!(
            positions.len() == 3 * len,
            "{case} reference position count"
        );
        for axis in 0..3 {
            for row in 0..len {
                assert_eq!(
                    i64::from(plan.axes()[axis][row]),
                    positions[axis * len + row],
                    "{case} axis {axis} row {row}"
                );
            }
        }
        assert_eq!(
            plan.rope_delta(),
            tensor_values(&tensors, "rope_delta", Dtype::I64, i64::from_le_bytes)?[0],
            "{case} rope delta"
        );
        if case == CASES[0] {
            let run = prepared
                .modality_ids
                .iter()
                .position(|&m| m == 1)
                .context("no image tokens")?;
            let mut bad_ids = expanded.clone();
            let mut bad_modality = prepared.modality_ids.clone();
            bad_ids.remove(run);
            bad_modality.remove(run);
            let bad = PreparedInput {
                modality_ids: bad_modality,
                pixel_values: prepared.pixel_values.clone(),
                grids: prepared.grids.clone(),
            };
            assert!(
                bad.input(&bad_ids).validate(&spec, capacity).is_err(),
                "{case} short image run must be rejected"
            );
            let mut bad_ids = expanded.clone();
            bad_ids[run - 1] = 0;
            let bad = PreparedInput {
                modality_ids: prepared.modality_ids.clone(),
                pixel_values: prepared.pixel_values.clone(),
                grids: prepared.grids.clone(),
            };
            assert!(
                bad.input(&bad_ids).validate(&spec, capacity).is_err(),
                "{case} missing opening marker must be rejected"
            );
        }
    }
    println!("{{\"worst_pixel_max_abs\":{worst_pixels}}}");
    Ok(())
}

/// T10 video processor cases: (fixture prefix, decoded clip, max_slice_nums).
const VIDEO_CASES: [(&str, &str, usize); 4] = [
    ("football_4s_16x_s1", "football_4s", 1),
    ("football_4s_16x_s9", "football_4s", 9),
    ("boat_pan_4s_16x_s1", "boat_pan_4s", 1),
    ("sample_demo_1_16x_s1", "sample_demo_1", 1),
];

fn video_ref_dir() -> Result<PathBuf> {
    Ok(PathBuf::from(
        std::env::var_os("FF_VIDEO_REF_DIR").context("FF_VIDEO_REF_DIR missing")?,
    ))
}

fn i64_values(tensors: &SafeTensors<'_>, name: &str) -> Result<Vec<i64>> {
    tensor_values(tensors, name, Dtype::I64, i64::from_le_bytes)
}

fn i32_values(tensors: &SafeTensors<'_>, name: &str) -> Result<Vec<i32>> {
    tensor_values(tensors, name, Dtype::I32, i32::from_le_bytes)
}

fn u32_values(tensors: &SafeTensors<'_>, name: &str) -> Result<Vec<u32>> {
    i64_values(tensors, name)?
        .into_iter()
        .map(u32::try_from)
        .collect::<std::result::Result<_, _>>()
        .map_err(Into::into)
}

/// The pre-expansion prompt: the fixture ids with the video block replaced by one placeholder.
fn video_prompt(ids: &[u32], vision: &flyingfish::model_spec::VisionConfig) -> Result<Vec<u32>> {
    let markers = vision.minicpm.as_ref().context("MiniCPM markers missing")?;
    let first = ids
        .iter()
        .position(|&id| id == vision.video_token)
        .context("no video placeholder in the fixture ids")?;
    let start = ids[..first]
        .iter()
        .rposition(|&id| id == vision.start_token)
        .context("no video opening marker")?;
    let mut cursor = start;
    loop {
        ensure!(
            ids.get(cursor) == Some(&vision.start_token),
            "video frame has no opening marker"
        );
        cursor += 1;
        let mut thumb = 0usize;
        while cursor < ids.len() && ids[cursor] == vision.video_token {
            cursor += 1;
            thumb += 1;
        }
        ensure!(thumb > 0, "empty video frame run");
        ensure!(
            ids.get(cursor) == Some(&vision.end_token),
            "video frame has no end marker"
        );
        cursor += 1;
        loop {
            if ids.get(cursor) == Some(&markers.slice_start_token) {
                cursor += 1;
                let mut run = 0usize;
                while cursor < ids.len() && ids[cursor] == vision.video_token {
                    cursor += 1;
                    run += 1;
                }
                ensure!(run > 0, "empty video slice run");
                ensure!(
                    ids.get(cursor) == Some(&markers.slice_end_token),
                    "video slice has no end marker"
                );
                cursor += 1;
            } else if ids.get(cursor) == Some(&markers.newline_token)
                && ids.get(cursor + 1) == Some(&markers.slice_start_token)
            {
                cursor += 1;
            } else {
                break;
            }
        }
        if ids.get(cursor) != Some(&vision.start_token) {
            break;
        }
    }
    let mut prompt = ids[..start].to_vec();
    prompt.push(vision.video_token);
    prompt.extend_from_slice(&ids[cursor..]);
    Ok(prompt)
}

#[test]
#[ignore = "CPU-only external fixtures; set FF_MODEL and FF_VIDEO_REF_DIR"]
fn video_processor_and_canvas_match_reference() -> Result<()> {
    let model = PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
    let ref_dir = video_ref_dir()?;
    let spec = ModelSpec::read(&model)?;
    ensure!(
        spec.architecture == Architecture::MiniCpmV47,
        "fixtures require the MiniCPM spec"
    );
    let vision = spec
        .vision
        .as_ref()
        .context("vision configuration missing")?;
    let tokenizer = Shared::load(&model)?;
    let processor =
        Processor::load(&model, Some(&tokenizer))?.context("MiniCPM image processor")?;
    let processor_bytes =
        std::fs::read(ref_dir.join("fixtures/video_processor_stack1.safetensors"))?;
    let processor_tensors = SafeTensors::deserialize(&processor_bytes)?;
    let frames_bytes = std::fs::read(ref_dir.join("fixtures/video_decoded_frames.safetensors"))?;
    let frames_tensors = SafeTensors::deserialize(&frames_bytes)?;
    let reference: serde_json::Value = serde_json::from_slice(&std::fs::read(
        ref_dir.join("fixtures/video_decode_reference.json"),
    )?)?;
    for (case, clip, slice_nums) in VIDEO_CASES {
        let tensor = frames_tensors.tensor(&format!("{clip}_sampled_rgb"))?;
        ensure!(tensor.dtype() == Dtype::U8, "{case} sampled rgb dtype");
        let shape = tensor.shape();
        ensure!(
            shape.len() == 4 && shape[3] == 3,
            "{case} sampled rgb shape"
        );
        let (count, height, width) = (shape[0], shape[1], shape[2]);
        let plane = height * width * 3;
        let frames: Vec<RgbImage> = (0..count)
            .map(|index| RgbImage {
                width,
                height,
                pixels: tensor.data()[index * plane..(index + 1) * plane].to_vec(),
            })
            .collect();
        let video_meta = VideoMetadata {
            total_num_frames: count,
            fps: 1.0,
            duration: count as f64,
        };
        let options = VideoOptions {
            max_num_frames: count,
            stack_frames: 1,
            max_slice_nums: Some(slice_nums),
        };
        let video = processor.prepare_video(&frames, &video_meta, &options)?;
        assert_eq!(video.frames.len(), count, "{case} frames");
        let expected_grids = i32_values(&processor_tensors, &format!("{case}_grids_videos"))?;
        let expected_counts =
            i32_values(&processor_tensors, &format!("{case}_num_patches_per_frame"))?;
        let expected_sizes =
            i32_values(&processor_tensors, &format!("{case}_target_sizes_videos"))?;
        for (frame, grid) in video.frames.iter().zip(expected_grids.as_chunks::<2>().0) {
            assert_eq!(
                [frame.rows as i32, frame.cols as i32],
                *grid,
                "{case} frame grid"
            );
        }
        for (frame, crops) in video.frames.iter().zip(&expected_counts) {
            assert_eq!(frame.grids.len() as i32, *crops, "{case} crops per frame");
        }
        let ours: Vec<i32> = video
            .frames
            .iter()
            .flat_map(|frame| {
                frame
                    .grids
                    .iter()
                    .flat_map(|grid| [grid.h as i32, grid.w as i32])
            })
            .collect();
        assert_eq!(ours, expected_sizes, "{case} target sizes");
        let ids = u32_values(&processor_tensors, &format!("{case}_input_ids"))?;
        let prompt = video_prompt(&ids, vision)?;
        let capacity = ids.len() + 64;
        let (expanded, prepared) = processor.expand(&prompt, Vec::new(), vec![video], capacity)?;
        assert_eq!(expanded, ids, "{case} ids");
        let modality: Vec<u8> =
            i64_values(&processor_tensors, &format!("{case}_mm_token_type_ids"))?
                .into_iter()
                .map(u8::try_from)
                .collect::<std::result::Result<_, _>>()?;
        assert_eq!(prepared.modality_ids, modality, "{case} modality");
        let reference_pixels = tensor_values(
            &processor_tensors,
            &format!("{case}_pixel_values_videos"),
            Dtype::F32,
            f32::from_le_bytes,
        )?;
        assert_eq!(
            prepared.pixel_values.len(),
            reference_pixels.len(),
            "{case} pixel count"
        );
        let mut worst = 0.0_f32;
        for (actual, expected) in prepared.pixel_values.iter().zip(&reference_pixels) {
            worst = worst.max((actual - expected).abs());
        }
        let pil_pixels = tensor_values(
            &processor_tensors,
            &format!("{case}_pixel_values_videos_pil"),
            Dtype::F32,
            f32::from_le_bytes,
        )?;
        let mut pil_floor = 0.0_f32;
        for (primary, pil) in reference_pixels.iter().zip(&pil_pixels) {
            pil_floor = pil_floor.max((primary - pil).abs());
        }
        println!(
            "{}",
            serde_json::json!({
                "case": case,
                "ids": expanded.len(),
                "crops": prepared.grids.len(),
                "pixel_max_abs": worst,
                "pil_arm_max_abs": pil_floor,
            })
        );
        assert!(
            worst <= PIXEL_TOLERANCE,
            "{case} pixel error {worst} exceeds {PIXEL_TOLERANCE}"
        );
        let clip_meta = &reference["clips"][clip]["ffprobe"];
        let real = VideoMetadata {
            total_num_frames: clip_meta["total_num_frames"]
                .as_u64()
                .context("clip total frames")? as usize,
            fps: clip_meta["fps"].as_f64().context("clip fps")?,
            duration: clip_meta["duration"].as_f64().context("clip duration")?,
        };
        let sampled = sample_frame_indices(&real, count, 1)?;
        let expected_indices: Vec<usize> =
            i64_values(&frames_tensors, &format!("{clip}_sampled_indices"))?
                .into_iter()
                .map(|value| value as usize)
                .collect();
        assert_eq!(sampled, expected_indices, "{case} sample indices");
    }
    Ok(())
}

#[test]
#[ignore = "CPU-only external fixtures; set FF_MODEL and FF_VIDEO_REF_DIR"]
fn video_canvas_positions_match_reference() -> Result<()> {
    let model = PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
    let ref_dir = video_ref_dir()?;
    let spec = ModelSpec::read(&model)?;
    ensure!(
        spec.architecture == Architecture::MiniCpmV47,
        "fixtures require the MiniCPM spec"
    );
    let patch_elements = spec
        .vision
        .as_ref()
        .context("vision configuration missing")?
        .patch_elements(spec.architecture)?;
    let canvas_bytes = std::fs::read(ref_dir.join("fixtures/video_canvas_positions.safetensors"))?;
    let canvas = SafeTensors::deserialize(&canvas_bytes)?;
    let processor_bytes =
        std::fs::read(ref_dir.join("fixtures/video_processor_stack1.safetensors"))?;
    let processor_tensors = SafeTensors::deserialize(&processor_bytes)?;
    for case in [
        "synth_2f_64x48_s1",
        "synth_2f_640x480_s9",
        "synth_2f_1024x1024_s9",
        "synth_mixed_img_plus_2f_64x48_s1",
        "football_4s_16x_s1",
        "football_4s_16x_s9",
        "sample_demo_1_16x_s1",
        "boat_pan_4s_16x_s1",
    ] {
        let ids = u32_values(&canvas, &format!("{case}_input_ids"))?;
        let modality: Vec<u8> = i64_values(&canvas, &format!("{case}_mm_token_type_ids"))?
            .into_iter()
            .map(u8::try_from)
            .collect::<std::result::Result<_, _>>()?;
        let mut grids = Vec::new();
        if canvas.tensor(&format!("{case}_target_sizes")).is_ok() {
            let sizes = i32_values(&canvas, &format!("{case}_target_sizes"))?;
            for row in sizes.as_chunks::<2>().0 {
                grids.push(ImageGrid {
                    t: 1,
                    h: u32::try_from(row[0])?,
                    w: u32::try_from(row[1])?,
                });
            }
        }
        let video_sizes = if canvas
            .tensor(&format!("{case}_target_sizes_videos"))
            .is_ok()
        {
            i32_values(&canvas, &format!("{case}_target_sizes_videos"))?
        } else {
            i32_values(&processor_tensors, &format!("{case}_target_sizes_videos"))?
        };
        for row in video_sizes.as_chunks::<2>().0 {
            grids.push(ImageGrid {
                t: 1,
                h: u32::try_from(row[0])?,
                w: u32::try_from(row[1])?,
            });
        }
        let patch_rows: usize = grids
            .iter()
            .map(|grid| grid.h as usize * grid.w as usize)
            .sum();
        let grid_count = grids.len();
        let prepared = PreparedInput {
            modality_ids: modality,
            pixel_values: vec![0.0; patch_rows * patch_elements],
            grids,
        };
        let capacity = ids.len() + 64;
        let plan = prepared.input(&ids).validate(&spec, capacity)?;
        let positions = i64_values(&canvas, &format!("{case}_canvas_position_ids"))?;
        let len = ids.len();
        ensure!(positions.len() == 3 * len, "{case} position extent");
        for axis in 0..3 {
            for row in 0..len {
                assert_eq!(
                    i64::from(plan.axes()[axis][row]),
                    positions[axis * len + row],
                    "{case} axis {axis} row {row}"
                );
            }
        }
        let deltas = i64_values(&canvas, &format!("{case}_rope_deltas"))?;
        assert_eq!(plan.rope_delta(), deltas[0], "{case} rope delta");
        println!(
            "{}",
            serde_json::json!({
                "case": case,
                "seq": len,
                "grids": grid_count,
                "rope_delta": plan.rope_delta(),
            })
        );
    }
    Ok(())
}

/// Oracle: transformers 5.19 official modeling code (`get_vision_nearest_position_ids`,
/// `get_vision_window_index`, `MiniCPMV4_7Merger.forward` ordering).
#[test]
fn vision_bookkeeping_matches_the_model_definition() -> Result<()> {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("data/minicpm-vision-bookkeeping.json"))?;
    let side = fixture["side"].as_u64().context("side")? as u32;
    let window = fixture["window"].as_u64().context("window")? as u32;
    let cases = fixture["cases"].as_array().context("cases")?;
    assert!(!cases.is_empty(), "empty bookkeeping fixture");
    for case in cases {
        let grids: Vec<ImageGrid> = case["grids"]
            .as_array()
            .context("grids")?
            .iter()
            .map(|grid| {
                let grid = grid.as_array().context("grid")?;
                Ok(ImageGrid {
                    t: 1,
                    h: grid[0].as_u64().context("h")? as u32,
                    w: grid[1].as_u64().context("w")? as u32,
                })
            })
            .collect::<Result<_>>()?;
        let patch_rows: usize = grids.iter().map(|g| (g.h * g.w) as usize).sum();
        assert_eq!(
            patch_rows,
            case["patch_rows"].as_u64().context("patch_rows")? as usize,
            "patch rows {grids:?}"
        );
        assert_eq!(
            patch_rows / 4,
            case["tower_rows"].as_u64().context("tower_rows")? as usize
        );
        assert_eq!(
            patch_rows / 16,
            case["output_rows"].as_u64().context("output_rows")? as usize
        );

        let expected: Vec<i32> = case["pos_ids"]
            .as_array()
            .context("pos_ids")?
            .iter()
            .map(|id| Ok(i32::try_from(id.as_i64().context("pos id")?)?))
            .collect::<Result<_>>()?;
        assert_eq!(
            nearest_position_ids(&grids, side)?,
            expected,
            "nearest position ids {grids:?}"
        );

        let gather = window_gather(&grids, window)?;
        let expected_index: Vec<u32> = case["window_index"]
            .as_array()
            .context("window_index")?
            .iter()
            .map(|row| Ok(u32::try_from(row.as_u64().context("window row")?)?))
            .collect::<Result<_>>()?;
        assert_eq!(gather.indices, expected_index, "window order {grids:?}");
        let expected_cu: Vec<i32> = case["cu_window_seqlens"]
            .as_array()
            .context("cu_window_seqlens")?
            .iter()
            .map(|bound| Ok(i32::try_from(bound.as_i64().context("window bound")?)?))
            .collect::<Result<_>>()?;
        assert_eq!(gather.cu_seqlens, expected_cu, "window bounds {grids:?}");
        assert_eq!(
            gather.cu_seqlens.last().copied(),
            i32::try_from(patch_rows).ok(),
            "window bounds must span every patch row"
        );
        let mut seen = vec![false; patch_rows];
        for &row in &gather.indices {
            let row = row as usize;
            assert!(
                row < patch_rows && !seen[row],
                "window gather is not a permutation"
            );
            seen[row] = true;
        }
        let expected_mean: Vec<f64> = case["window_tag_mean"]
            .as_array()
            .context("window_tag_mean")?
            .iter()
            .map(|mean| mean.as_f64().context("window mean"))
            .collect::<Result<_>>()?;
        let span = (window * window) as usize;
        assert_eq!(expected_mean.len(), patch_rows / span);
        for (index, expected) in expected_mean.iter().enumerate() {
            let rows = &gather.indices[index * span..][..span];
            let mean = rows.iter().map(|&row| f64::from(row)).sum::<f64>() / span as f64;
            assert_eq!(mean, *expected, "window mean {index} of {grids:?}");
        }

        let row_grids: Vec<(u32, u32)> = grids.iter().map(|g| (g.h / 2, g.w / 2)).collect();
        let groups = merge_groups(&row_grids, 2)?;
        let expected_groups: Vec<u32> = case["merge_group_rows"]
            .as_array()
            .context("merge_group_rows")?
            .iter()
            .map(|row| Ok(u32::try_from(row.as_u64().context("merge row")?)?))
            .collect::<Result<_>>()?;
        assert_eq!(groups, expected_groups, "merge groups {grids:?}");
        assert_eq!(groups.len(), 4 * patch_rows / 16, "merge group count");
    }
    assert!(nearest_position_ids(&[], 0).is_err());
    assert!(
        window_gather(&[ImageGrid { t: 1, h: 5, w: 12 }], 2).is_err(),
        "odd window grid must be rejected"
    );
    assert!(
        merge_groups(&[(6, 12)], 4).is_err(),
        "unaligned merge grid must be rejected"
    );
    Ok(())
}

/// NaViT patch pack `[C, side, patches*side]` in (c, ky, patch, kx) order -> `[patches, C*side*side]`
/// patch rows in (c, ky, kx) order, rounded to BF16. `transposed` swaps ky/kx for the negative
/// control.
fn patch_rows(
    pixels: &[f32],
    patches: usize,
    channels: usize,
    side: usize,
    transposed: bool,
) -> Vec<f32> {
    let width = channels * side * side;
    let mut rows = vec![0f32; patches * width];
    for patch in 0..patches {
        for c in 0..channels {
            for ky in 0..side {
                for kx in 0..side {
                    let (row_axis, column_axis) = if transposed { (kx, ky) } else { (ky, kx) };
                    let source =
                        (c * side + row_axis) * patches * side + patch * side + column_axis;
                    let target = patch * width + (c * side + ky) * side + kx;
                    rows[target] = half::bf16::from_f32(pixels[source]).to_f32();
                }
            }
        }
    }
    rows
}

fn patch_embedding(
    rows: &[f32],
    weight: &[f32],
    bias: &[f32],
    patches: usize,
    outputs: usize,
    width: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; patches * outputs];
    for patch in 0..patches {
        let x = &rows[patch * width..][..width];
        for o in 0..outputs {
            let w = &weight[o * width..][..width];
            let mut sum = 0f32;
            for j in 0..width {
                sum += x[j] * w[j];
            }
            out[patch * outputs + o] = half::bf16::from_f32(sum + bias[o]).to_f32();
        }
    }
    out
}

fn checkpoint_bf16(model_dir: &std::path::Path, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        model_dir.join("model.safetensors.index.json"),
    )?)?;
    let shard = index["weight_map"][name]
        .as_str()
        .with_context(|| format!("shard for {name}"))?;
    let file = std::fs::File::open(model_dir.join(shard))?;
    let map = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let tensors = SafeTensors::deserialize(&map)?;
    let shape = tensors.tensor(name)?.shape().to_vec();
    let values = tensor_values(&tensors, name, Dtype::BF16, |v| {
        half::bf16::from_bits(u16::from_le_bytes(v)).to_f32()
    })?;
    Ok((shape, values))
}

/// Pins the NaViT pack order at the patch embedding: the correct (c, ky, patch, kx) gather plus the
/// position add reproduces the T0 hockey tap, while a ky/kx-transposed gather must fail.
/// Measured on CPU (torch): correct 0.0625, transposed 6.4375 max_abs; the tolerance sits between.
#[test]
#[ignore = "CPU-only external fixtures; set FF_MODEL and FF_MINICPM_REFERENCE_DIR; run with --release"]
fn patch_rows_reorder_matches_the_reference() -> Result<()> {
    use flyingfish::vision_input::{ImageGrid, nearest_position_ids};
    let model_dir = PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
    let reference_dir = PathBuf::from(
        std::env::var_os("FF_MINICPM_REFERENCE_DIR").context("FF_MINICPM_REFERENCE_DIR missing")?,
    );
    let spec = ModelSpec::read(&model_dir)?;
    let vision = spec
        .vision
        .as_ref()
        .context("vision configuration missing")?;
    let channels = vision.channels as usize;
    let side = vision.patch as usize;
    let width = channels * side * side;
    let hidden = vision.hidden as usize;
    let path = reference_dir.join("image_hockey_16x.safetensors");
    let file = std::fs::File::open(&path).with_context(|| format!("open {}", path.display()))?;
    let map = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let source = SafeTensors::deserialize(&map)?;
    let sizes = tensor_values(&source, "target_sizes", Dtype::I32, i32::from_le_bytes)?;
    ensure!(sizes.len() == 2, "single-image fixture expected");
    let grids = [ImageGrid {
        t: 1,
        h: u32::try_from(sizes[0])?,
        w: u32::try_from(sizes[1])?,
    }];
    let patches = (grids[0].h * grids[0].w) as usize;
    let pixels = tensor_values(&source, "pixel_values", Dtype::F32, f32::from_le_bytes)?;
    let reference = tensor_values(&source, "vision_patch_embeddings", Dtype::BF16, |v| {
        half::bf16::from_bits(u16::from_le_bytes(v)).to_f32()
    })?;
    let (weight_shape, weight) = checkpoint_bf16(
        &model_dir,
        "model.vision_tower.embeddings.patch_embedding.weight",
    )?;
    ensure!(
        weight_shape == [hidden, channels, side, side] && weight.len() == hidden * width,
        "patch weight shape {:?}",
        weight_shape
    );
    let (bias_shape, bias) = checkpoint_bf16(
        &model_dir,
        "model.vision_tower.embeddings.patch_embedding.bias",
    )?;
    ensure!(
        bias_shape == [hidden] && bias.len() == hidden,
        "patch bias shape"
    );
    let (table_shape, table) = checkpoint_bf16(
        &model_dir,
        "model.vision_tower.embeddings.position_embedding.weight",
    )?;
    ensure!(
        table_shape == [(vision.position_side as usize).pow(2), hidden],
        "position table shape"
    );
    let ids = nearest_position_ids(&grids, vision.position_side)?;
    ensure!(ids.len() == patches, "one position id per patch");
    let mut worst = [0f32; 2];
    for (arm, transposed) in [false, true].into_iter().enumerate() {
        let rows = patch_rows(&pixels, patches, channels, side, transposed);
        let embedded = patch_embedding(&rows, &weight, &bias, patches, hidden, width);
        let mut added = vec![0f32; patches * hidden];
        for patch in 0..patches {
            let row = ids[patch] as usize * hidden;
            for o in 0..hidden {
                let sum = embedded[patch * hidden + o] + table[row + o];
                added[patch * hidden + o] = half::bf16::from_f32(sum).to_f32();
            }
        }
        let mut diff = 0f32;
        let mut mean = 0f64;
        for (a, r) in added.iter().zip(&reference) {
            diff = diff.max((a - r).abs());
            mean += f64::from((a - r).abs());
        }
        worst[arm] = diff;
        println!(
            "{}",
            serde_json::json!({"patch_rows": {"transposed": transposed, "max_abs": diff,
                "mean_abs": mean / added.len() as f64}})
        );
    }
    let tolerance = 0.25;
    ensure!(
        worst[0] <= tolerance,
        "interleaved patch rows exceed the reference tolerance: {} > {tolerance}",
        worst[0]
    );
    ensure!(
        worst[1] > tolerance,
        "the ky/kx-transposed mutant must fail, measured {}",
        worst[1]
    );
    Ok(())
}

/// Abramowitz-Stegun 7.1.26; ~1.5e-7 absolute error, far below one BF16 ulp here.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0f32 / (1.0f32 + 0.327_591_1f32 * x);
    let y = 1.0f32
        - (((((1.061_405_4f32 * t - 1.453_152_1f32) * t) + 1.421_413_8f32) * t - 0.284_496_73f32)
            * t
            + 0.254_829_6f32)
            * t
            * (-x * x).exp();
    sign * y
}

fn layer_norm_f32(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(weight)
        .zip(bias)
        .map(|((v, w), b)| (v - mean) * inv * w + b)
        .collect()
}

fn matmul_f32(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    rows: usize,
    width: usize,
    outputs: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; rows * outputs];
    for r in 0..rows {
        for o in 0..outputs {
            let mut sum = 0f32;
            for j in 0..width {
                sum += x[r * width + j] * w[o * width + j];
            }
            out[r * outputs + o] = half::bf16::from_f32(sum + bias[o]).to_f32();
        }
    }
    out
}

/// Pins the final-merger 2x2 grouping over the `(h/2, w/2)` tower grid: the production
/// `merge_groups` map must reproduce the T0 merger output from the fixture tower output, while
/// contiguous four-row chunks must fail.
#[test]
#[ignore = "CPU-only external fixtures; set FF_MODEL and FF_MINICPM_REFERENCE_DIR; run with --release"]
fn final_merger_groups_match_the_reference() -> Result<()> {
    use flyingfish::vision_input::merge_groups;
    let model_dir = PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
    let reference_dir = PathBuf::from(
        std::env::var_os("FF_MINICPM_REFERENCE_DIR").context("FF_MINICPM_REFERENCE_DIR missing")?,
    );
    let spec = ModelSpec::read(&model_dir)?;
    let vision = spec
        .vision
        .as_ref()
        .context("vision configuration missing")?;
    let path = reference_dir.join("image_hockey_16x.safetensors");
    let file = std::fs::File::open(&path).with_context(|| format!("open {}", path.display()))?;
    let map = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let source = SafeTensors::deserialize(&map)?;
    let sizes = tensor_values(&source, "target_sizes", Dtype::I32, i32::from_le_bytes)?;
    let (h, w) = (sizes[0] as u32, sizes[1] as u32);
    let tower = tensor_values(&source, "vision_tower_out", Dtype::BF16, |v| {
        half::bf16::from_bits(u16::from_le_bytes(v)).to_f32()
    })?;
    let tower_shape = source.tensor("vision_tower_out")?.shape().to_vec();
    ensure!(
        tower_shape == [1, ((h / 2) * (w / 2)) as usize, vision.hidden as usize],
        "tower shape {:?}",
        tower_shape
    );
    let merger_shape = source.tensor("merger_out")?.shape().to_vec();
    let reference = tensor_values(&source, "merger_out", Dtype::BF16, |v| {
        half::bf16::from_bits(u16::from_le_bytes(v)).to_f32()
    })?;
    ensure!(
        merger_shape == [((h / 4) * (w / 4)) as usize, vision.output as usize],
        "merger shape {:?}",
        merger_shape
    );
    let hidden = vision.hidden as usize;
    let fc1 = (hidden * 4) as usize;
    let output = vision.output as usize;
    let tokens = merger_shape[0];
    let (norm_w_shape, norm_w) = checkpoint_bf16(&model_dir, "model.merger.mlp.0.pre_norm.weight")?;
    let (norm_b_shape, norm_b) = checkpoint_bf16(&model_dir, "model.merger.mlp.0.pre_norm.bias")?;
    let (l1_shape, l1_w) = checkpoint_bf16(&model_dir, "model.merger.mlp.0.linear_1.weight")?;
    let (l1_b_shape, l1_b) = checkpoint_bf16(&model_dir, "model.merger.mlp.0.linear_1.bias")?;
    let (l2_shape, l2_w) = checkpoint_bf16(&model_dir, "model.merger.mlp.0.linear_2.weight")?;
    let (l2_b_shape, l2_b) = checkpoint_bf16(&model_dir, "model.merger.mlp.0.linear_2.bias")?;
    ensure!(
        norm_w_shape == [fc1] && norm_b_shape == [fc1],
        "norm shapes"
    );
    ensure!(
        l1_shape == [fc1, fc1] && l1_b_shape == [fc1],
        "linear_1 shapes"
    );
    ensure!(
        l2_shape == [output, fc1] && l2_b_shape == [output],
        "linear_2 shapes"
    );
    let row_grids = [(h / 2, w / 2)];
    let groups = merge_groups(&row_grids, 2)?;
    ensure!(groups.len() == tokens * 4, "group map extent");
    let mut worst = [0f32; 2];
    for (arm, contiguous) in [false, true].into_iter().enumerate() {
        let mut out = vec![0f32; tokens * output];
        for token in 0..tokens {
            let mut grouped = vec![0f32; fc1];
            for j in 0..4 {
                let row = if contiguous {
                    token * 4 + j
                } else {
                    groups[token * 4 + j] as usize
                };
                grouped[j * hidden..][..hidden].copy_from_slice(&tower[row * hidden..][..hidden]);
            }
            let normed: Vec<f32> = layer_norm_f32(&grouped, &norm_w, &norm_b, 1e-6)
                .into_iter()
                .map(|v| half::bf16::from_f32(v).to_f32())
                .collect();
            let hidden_1 = matmul_f32(&normed, &l1_w, &l1_b, 1, fc1, fc1);
            let activated: Vec<f32> = hidden_1
                .iter()
                .map(|&v| {
                    half::bf16::from_f32(0.5 * v * (1.0 + erf(v / std::f32::consts::SQRT_2)))
                        .to_f32()
                })
                .collect();
            let row = matmul_f32(&activated, &l2_w, &l2_b, 1, fc1, output);
            out[token * output..][..output].copy_from_slice(&row);
        }
        let mut diff = 0f32;
        for (a, r) in out.iter().zip(&reference) {
            diff = diff.max((a - r).abs());
        }
        worst[arm] = diff;
        println!(
            "{}",
            serde_json::json!({"merger_groups": {"contiguous": contiguous, "max_abs": diff}})
        );
    }
    let tolerance = 0.25;
    ensure!(
        worst[0] <= tolerance,
        "grouped merger exceeds the reference tolerance: {} > {tolerance}",
        worst[0]
    );
    ensure!(
        worst[1] > tolerance,
        "the contiguous-grouping mutant must fail, measured {}",
        worst[1]
    );
    Ok(())
}
