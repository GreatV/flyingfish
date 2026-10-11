use anyhow::{Context, Result};
use flyingfish::{
    checkpoint::{LoadPolicy, MappedCheckpoint, TensorScope},
    model_spec::ModelSpec,
    vision_input::{ImageGrid, MultimodalInput, PositionPlan, SpliceRange, VisionPositions},
};
use serde_json::{Value, json};

fn spec() -> ModelSpec {
    ModelSpec::qwen(
        &serde_json::from_str(include_str!("data/q38-config.json")).unwrap(),
        &Value::Null,
    )
    .unwrap()
}

#[test]
fn vision_config_and_scope_are_independent_of_load_selection() {
    let s = spec();
    assert_eq!(s.vision.as_ref().unwrap().patch_elements().unwrap(), 1536);
    for policy in [
        LoadPolicy::TextOnly,
        LoadPolicy::TextWithVision,
        LoadPolicy::VisionOnly,
    ] {
        assert!(!policy.selects(TensorScope::Mtp));
    }
    assert!(LoadPolicy::TextOnly.selects(TensorScope::Text));
    assert!(!LoadPolicy::TextOnly.selects(TensorScope::Vision));
    assert!(LoadPolicy::TextWithVision.selects(TensorScope::Vision));
    assert!(!LoadPolicy::VisionOnly.selects(TensorScope::Text));
    let root: Value = serde_json::from_str(include_str!("data/q38-config.json")).unwrap();
    for (pointer, value) in [
        ("/vision_config/hidden_size", json!(1024)),
        ("/vision_config/patch_size", json!(14)),
        ("/vision_config/out_hidden_size", json!(3584)),
        ("/vision_config/hidden_act", json!("gelu")),
        ("/vision_config/deepstack_visual_indexes", json!([0])),
        ("/image_token_id", json!(248057)),
    ] {
        let mut bad = root.clone();
        *bad.pointer_mut(pointer).unwrap() = value;
        assert!(ModelSpec::qwen(&bad, &Value::Null).is_err(), "{pointer}");
    }
}

#[test]
fn rectangular_image_positions_and_chunk_splices() -> Result<()> {
    let s = spec();
    let mut ids = vec![100, 248053];
    ids.extend([248056; 6]);
    ids.extend([248054, 200]);
    let mut modality = vec![0; 10];
    modality[2..8].fill(1);
    let pixels = vec![0.0; 24 * 1536];
    let grid = [ImageGrid { t: 1, h: 4, w: 6 }];
    let plan = MultimodalInput {
        input_ids: &ids,
        modality_ids: &modality,
        pixel_values: &pixels,
        grids: &grid,
    }
    .validate(&s, 32)?;
    assert_eq!(
        plan.axes(),
        [
            &[0, 1, 2, 2, 2, 2, 2, 2, 5, 6][..],
            &[0, 1, 2, 2, 2, 3, 3, 3, 5, 6][..],
            &[0, 1, 2, 3, 4, 2, 3, 4, 5, 6][..],
        ]
    );
    assert_eq!(plan.rope_delta(), -3);
    assert_eq!(plan.decode_position(10)?, 7);
    assert_eq!(plan.decode_position(31)?, 28);
    let table = plan.packed_axes()?;
    assert_eq!(table.len(), 32 * 3);
    for (row, values) in table.as_chunks::<3>().0.iter().enumerate() {
        let expected = if row < 10 {
            plan.axes().map(|a| a[row])
        } else {
            [row as i32 - 3; 3]
        };
        assert_eq!(*values, expected);
    }
    let text = PositionPlan::text(10, 32)?.packed_axes()?;
    assert_eq!(text, (0..32).flat_map(|p| [p; 3]).collect::<Vec<_>>());
    assert_ne!(
        table, text,
        "reset must replace image coordinates with text coordinates"
    );
    assert!(plan.decode_position(9).is_err());
    assert!(plan.decode_position(32).is_err());
    assert_eq!(plan.slots()[0].tokens, 2..8);
    assert_eq!(plan.slots()[0].patches, 0..24);
    assert_eq!(
        plan.splices(0..5)?,
        [SpliceRange {
            chunk_rows: 2..5,
            feature_rows: 0..3
        }]
    );
    assert_eq!(
        plan.splices(5..10)?,
        [SpliceRange {
            chunk_rows: 0..3,
            feature_rows: 3..6
        }]
    );
    assert!(plan.splices(0..11).is_err());
    Ok(())
}

#[test]
fn multiple_images_keep_distinct_feature_and_token_ranges() -> Result<()> {
    let ids = [248053, 248056, 248054, 100, 248053, 248056, 248056, 248054];
    let modality = [0, 1, 0, 0, 0, 1, 1, 0];
    let grids = [
        ImageGrid { t: 1, h: 2, w: 2 },
        ImageGrid { t: 1, h: 2, w: 4 },
    ];
    let pixels = vec![0.0; 12 * 1536];
    let plan = MultimodalInput {
        input_ids: &ids,
        modality_ids: &modality,
        pixel_values: &pixels,
        grids: &grids,
    }
    .validate(&spec(), 16)?;
    assert_eq!(plan.slots()[0].features, 0..1);
    assert_eq!(plan.slots()[1].features, 1..3);
    assert_eq!(plan.slots()[1].patches, 4..12);
    assert_eq!(
        plan.splices(1..6)?,
        [
            SpliceRange {
                chunk_rows: 0..1,
                feature_rows: 0..1
            },
            SpliceRange {
                chunk_rows: 4..5,
                feature_rows: 1..2
            },
        ]
    );
    assert_eq!(plan.axes()[2], &[0, 1, 2, 3, 4, 5, 6, 7]);
    Ok(())
}

#[test]
fn packed_input_rejects_ambiguous_slots_and_invalid_pixels() {
    let s = spec();
    let pixels = vec![0.0; 4 * 1536];
    let grid = [ImageGrid { t: 1, h: 2, w: 2 }];
    let valid = |ids: &[u32], m: &[u8], p: &[f32], g: &[ImageGrid], capacity| {
        MultimodalInput {
            input_ids: ids,
            modality_ids: m,
            pixel_values: p,
            grids: g,
        }
        .validate(&s, capacity)
    };
    assert!(valid(&[248053, 248056, 248054], &[0, 1, 0], &pixels, &grid, 3).is_ok());
    for (ids, m) in [
        (vec![100, 248056, 248054], vec![0, 1, 0]),
        (vec![248053, 248056, 100], vec![0, 1, 0]),
        (vec![248053, 248056, 248054], vec![0, 0, 0]),
        (vec![248053, 100, 248054], vec![0, 1, 0]),
        (vec![248053, 248056, 248056, 248054], vec![0, 1, 1, 0]),
        (vec![248053, 248056, 248054], vec![0, 2, 0]),
        (vec![248057, 248056, 248054], vec![0, 1, 0]),
    ] {
        assert!(valid(&ids, &m, &pixels, &grid, 16).is_err());
    }
    assert!(valid(&[248053, 248056, 248054], &[0, 1, 0], &pixels, &grid, 2).is_err());
    assert!(valid(&[248053, 248056, 248054], &[0, 1], &pixels, &grid, 16).is_err());
    assert!(
        valid(
            &[248053, 248056, 248054],
            &[0, 1, 0],
            &pixels[..pixels.len() - 1],
            &grid,
            16
        )
        .is_err()
    );
    for bad in [f32::NAN, f32::INFINITY, f32::MAX] {
        let mut p = pixels.clone();
        p[0] = bad;
        assert!(valid(&[248053, 248056, 248054], &[0, 1, 0], &p, &grid, 16).is_err());
    }
    for g in [
        ImageGrid { t: 2, h: 2, w: 2 },
        ImageGrid { t: 1, h: 3, w: 2 },
        ImageGrid { t: 1, h: 0, w: 2 },
        ImageGrid {
            t: 1,
            h: u32::MAX - 1,
            w: u32::MAX - 1,
        },
    ] {
        assert!(g.patch_rows().is_err());
    }
    assert!(valid(&[100], &[0], &pixels, &grid, 16).is_err());
}

#[test]
fn text_axes_and_all_rotary_frequencies_are_exactly_linear() -> Result<()> {
    let plan = PositionPlan::text(65, 192)?;
    let mut axes = [0; 3];
    for j in 0..32 {
        axes[PositionPlan::rotary_axis(j)?] += 1;
        for physical in 0..192 {
            assert_eq!(
                plan.rotary_position(j, physical)?.to_le_bytes(),
                (physical as i32).to_le_bytes()
            );
        }
    }
    assert_eq!(axes, [11, 11, 10]);
    assert_eq!(plan.rope_delta(), 0);
    assert!(PositionPlan::rotary_axis(32).is_err());
    assert!(plan.rotary_position(0, 192).is_err());
    assert!(PositionPlan::text(66, 65).is_err());
    assert!(PositionPlan::text(0, 65).is_err());
    let ids = [1, 2, 3];
    let plan = MultimodalInput {
        input_ids: &ids,
        modality_ids: &[0; 3],
        pixel_values: &[],
        grids: &[],
    }
    .validate(&spec(), 4)?;
    assert_eq!(plan.axes(), [&[0, 1, 2][..]; 3]);
    assert!(plan.slots().is_empty());
    Ok(())
}

#[test]
fn patch_coordinates_are_merge_grouped_and_identity_grid_is_exact() -> Result<()> {
    let s = spec();
    let v = s.vision.as_ref().unwrap();
    let p = VisionPositions::new(v, &[ImageGrid { t: 1, h: 4, w: 6 }], 24)?;
    assert_eq!(
        &p.coordinates[..8],
        &[
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3]
        ]
    );
    assert!(VisionPositions::new(v, &[ImageGrid { t: 1, h: 4, w: 6 }], 23).is_err());
    let p = VisionPositions::new(v, &[ImageGrid { t: 1, h: 48, w: 48 }], 48 * 48)?;
    for (i, &[y, x]) in p.coordinates.iter().enumerate() {
        assert_eq!(p.bilinear_indices[0][i], y * 48 + x);
        assert_eq!(
            p.bilinear_weights.each_ref().map(|a| a[i].to_bits()),
            [1f32.to_bits(), 0, 0, 0]
        );
    }
    Ok(())
}

#[test]
#[ignore = "CPU-only external fixtures; set FF_Q38_FIXTURE_DIR"]
fn image_fixture_positions_and_bilinear_tables_match_bitwise() -> Result<()> {
    use safetensors::{Dtype, SafeTensors};
    let dir = std::path::PathBuf::from(
        std::env::var_os("FF_Q38_FIXTURE_DIR").context("FF_Q38_FIXTURE_DIR missing")?,
    );
    let s = spec();
    for case in ["img-report", "img-nonsquare", "img-marker", "img-tiny"] {
        let file = std::fs::File::open(dir.join(format!("{case}.safetensors")))?;
        let mapped = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let tensors = SafeTensors::deserialize(&mapped)?;
        let integers = |key: &str| -> Result<Vec<i64>> {
            let t = tensors.tensor(key)?;
            anyhow::ensure!(t.dtype() == Dtype::I64, "wrong integer dtype");
            Ok(t.data()
                .as_chunks::<8>()
                .0
                .iter()
                .map(|v| i64::from_le_bytes(*v))
                .collect())
        };
        let ids = integers("input_ids")?
            .into_iter()
            .map(u32::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        let modalities = integers("mm_token_type_ids")?
            .into_iter()
            .map(u8::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        let g = integers("image_grid_thw")?;
        let grid = [ImageGrid {
            t: g[0].try_into()?,
            h: g[1].try_into()?,
            w: g[2].try_into()?,
        }];
        let t = tensors.tensor("pixel_values_fp32")?;
        anyhow::ensure!(t.dtype() == Dtype::F32, "wrong pixel dtype");
        let pixels: Vec<_> = t
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect();
        let plan = MultimodalInput {
            input_ids: &ids,
            modality_ids: &modalities,
            pixel_values: &pixels,
            grids: &grid,
        }
        .validate(&s, ids.len() + 64)?;
        let reference = integers("mrope_position_ids")?;
        assert_eq!(
            plan.axes()
                .into_iter()
                .flatten()
                .map(|&x| i64::from(x))
                .collect::<Vec<_>>(),
            reference,
            "{case}: text mRoPE"
        );
        let tables = VisionPositions::new(s.vision.as_ref().unwrap(), &grid, plan.patch_rows())?;
        assert_eq!(
            tables
                .coordinates
                .iter()
                .flatten()
                .map(|&x| i64::from(x))
                .collect::<Vec<_>>(),
            integers("vision_position_ids")?,
            "{case}: vision coordinates"
        );
        let idx = tensors.tensor("bilinear_indices")?;
        anyhow::ensure!(idx.dtype() == Dtype::I32, "wrong bilinear index dtype");
        assert_eq!(
            tables
                .bilinear_indices
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
            idx.data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| i32::from_le_bytes(*v))
                .collect::<Vec<_>>(),
            "{case}: bilinear indices"
        );
        let w = tensors.tensor("bilinear_weights")?;
        anyhow::ensure!(w.dtype() == Dtype::F32, "wrong bilinear weight dtype");
        assert_eq!(
            tables
                .bilinear_weights
                .iter()
                .flatten()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            w.data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| u32::from_le_bytes(*v))
                .collect::<Vec<_>>(),
            "{case}: bilinear weights"
        );
        eprintln!(
            "VISION_CPU_FIXTURE {case} positions/indices/weights bit-exact; delta={}",
            plan.rope_delta()
        );
    }
    Ok(())
}

#[test]
#[ignore = "CPU-only checkpoint header audit; set FF_Q38_MODEL"]
fn vision_load_policy_checks_all_headers_without_loading_payloads() -> Result<()> {
    let dir =
        std::path::PathBuf::from(std::env::var_os("FF_Q38_MODEL").context("FF_Q38_MODEL missing")?);
    let s = ModelSpec::read(&dir)?;
    let mapped = MappedCheckpoint::open(&dir)?;
    for (policy, bytes, vision_count, text_count, mtp_count) in [
        (LoadPolicy::TextOnly, 29_468_003_328, 0, 1251, 0),
        (LoadPolicy::TextWithVision, 30_389_463_520, 333, 1251, 0),
        (LoadPolicy::VisionOnly, 921_460_192, 333, 0, 0),
        (LoadPolicy::TextWithMtp, 29_945_203_072, 0, 1251, 22),
        (LoadPolicy::MtpOnly, 477_199_744, 0, 0, 22),
    ] {
        let m = mapped.q38_manifest_with_policy(&s, policy)?;
        assert_eq!(m.resident_bytes, bytes);
        assert_eq!(
            m.entries
                .values()
                .filter(|e| e.load && e.scope == TensorScope::Vision)
                .count(),
            vision_count
        );
        assert_eq!(
            m.entries
                .values()
                .filter(|e| e.load && e.scope == TensorScope::Text)
                .count(),
            text_count
        );
        assert_eq!(
            m.entries
                .values()
                .filter(|e| e.scope == TensorScope::Mtp && e.load)
                .count(),
            mtp_count
        );
        if mtp_count > 0 {
            let plan = flyingfish::linear_plan::LayerDequantPlan::mtp(&m)?;
            assert_eq!(plan.bytes as usize, flyingfish::mtp::ARENA_BYTES);
            assert_eq!(plan.slots.len(), 7);
        }
        assert_eq!(
            m.entries
                .values()
                .filter(|e| e.scope == TensorScope::Vision)
                .count(),
            333
        );
    }
    Ok(())
}
