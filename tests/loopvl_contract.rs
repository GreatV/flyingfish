use anyhow::{Context, Result};
use flyingfish::{
    checkpoint::{LoadPolicy, MappedCheckpoint},
    model_spec::{Architecture, ModelSpec},
};

#[test]
fn recurrent_geometry_rejects_incompatible_mask_and_cycle_counts() -> Result<()> {
    let root: serde_json::Value = serde_json::from_str(include_str!("data/loopvl-config.json"))?;
    let spec = ModelSpec::loopvl(&root, &serde_json::Value::Null)?;
    assert_eq!(spec.architecture, Architecture::LoopVl);
    assert_eq!(spec.layers.len(), 128);
    assert!(
        spec.layers
            .iter()
            .enumerate()
            .all(|(i, l)| l.state_index as usize == i)
    );
    assert_eq!(
        spec.layers.len() * 2 * spec.kv_heads as usize * spec.head_dim as usize * 2,
        786432
    );
    for (key, value) in [
        ("prefix_lm", serde_json::json!(false)),
        ("num_layers_per_stack", serde_json::json!(32)),
        ("H_cycles", serde_json::json!(3)),
        ("embedding_scale", serde_json::json!(1)),
        ("num_hidden_layers", serde_json::json!(32)),
    ] {
        let mut bad = root.clone();
        bad["text_config"][key] = value;
        assert!(
            ModelSpec::loopvl(&bad, &serde_json::Value::Null).is_err(),
            "accepted {key}"
        );
    }
    let ids = vec![1; 580];
    let mut modalities = vec![0; ids.len()];
    modalities[6..557].fill(1);
    let packed = flyingfish::serve::image::PreparedInput {
        modality_ids: modalities,
        pixel_values: (0..551 * 588).map(|i| (i % 17) as f32 / 17.0).collect(),
        grids: vec![flyingfish::vision_input::ImageGrid { t: 1, h: 19, w: 29 }],
    };
    let path =
        std::env::temp_dir().join(format!("loopvl-image-{}.safetensors", std::process::id()));
    packed.save(&spec, &ids, &path)?;
    let result = flyingfish::serve::image::PreparedInput::read(&spec, &path, &ids);
    std::fs::remove_file(path)?;
    let loaded = result?;
    assert_eq!(loaded.pixel_values, packed.pixel_values);
    assert_eq!(loaded.modality_ids, packed.modality_ids);
    let positions = loaded.input(&ids).validate(&spec, 8192)?;
    assert_eq!(positions.decode_position(580)?, 58);
    Ok(())
}

#[test]
#[ignore = "published checkpoint; set FF_MODEL; CPU only"]
fn published_loopvl_manifest_accounts_for_all_components() -> Result<()> {
    let path = std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
    let spec = ModelSpec::read(&path)?;
    let manifest = MappedCheckpoint::open(&path)?.manifest(&spec, LoadPolicy::TextOnly)?;
    assert_eq!(manifest.entries.len(), 580);
    assert_eq!(manifest.entries.values().filter(|e| e.load).count(), 259);
    assert_eq!(
        manifest
            .entries
            .values()
            .filter(|e| e.dtype == safetensors::Dtype::F32)
            .count(),
        10
    );
    assert_eq!(manifest.resident_bytes, 2_365_590_528);
    assert_eq!(
        manifest.entries.values().map(|e| e.bytes).sum::<usize>(),
        3_266_619_408
    );
    println!("PASS loopvl_manifest tensors=580 text=259 skipped=321 missing=0 unexpected=0");
    let mapped = MappedCheckpoint::open(&path)?;
    let full = mapped.manifest(&spec, LoadPolicy::TextWithVision)?;
    assert_eq!(full.entries.values().filter(|e| e.load).count(), 580);
    assert_eq!(full.resident_bytes, 3_266_619_408);
    let vision = mapped.manifest(&spec, LoadPolicy::VisionOnly)?;
    assert_eq!(vision.entries.values().filter(|e| e.load).count(), 321);
    assert_eq!(vision.resident_bytes, 901_028_880);
    println!("PASS loopvl_vision_manifest full=580 vision=321 f32=10 missing=0 unexpected=0");
    Ok(())
}

fn compact_positions(header: usize, grid_h: usize, grid_w: usize, tail: usize) -> Vec<[i64; 2]> {
    let visual = grid_h * grid_w;
    (0..header + visual + tail)
        .map(|position| {
            if position < header {
                [position as i64, position as i64]
            } else if position < header + visual {
                let patch = position - header;
                [
                    (header + patch / grid_w) as i64,
                    (header + patch % grid_w) as i64,
                ]
            } else {
                let compact = header + grid_h.max(grid_w) + position - header - visual;
                [compact as i64, compact as i64]
            }
        })
        .collect()
}

#[test]
#[ignore = "published oracle fixture; set FF_LOOPVL_FIXTURES; CPU only"]
fn loopvl_compact_positions_match_oracle_and_controls() -> Result<()> {
    let fixtures = std::path::PathBuf::from(
        std::env::var_os("FF_LOOPVL_FIXTURES").context("FF_LOOPVL_FIXTURES missing")?,
    );
    let bytes = std::fs::read(fixtures.join("image_hockey.safetensors"))?;
    let source = safetensors::SafeTensors::deserialize(&bytes)?;
    let tensor = source.tensor("rope_positions")?;
    assert_eq!(tensor.shape(), &[2, 580]);
    let expected: Vec<i64> = tensor
        .data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|v| i64::from_le_bytes(*v))
        .collect();
    let actual = compact_positions(6, 19, 29, 23);
    assert_eq!(actual.len(), 580);
    for (position, axes) in actual.iter().enumerate() {
        assert_eq!(axes[0], expected[position]);
        assert_eq!(axes[1], expected[580 + position]);
    }
    let raw_index: Vec<[i64; 2]> = (0..580).map(|p| [p as i64, p as i64]).collect();
    assert_ne!(actual, raw_index, "control4 raw index accepted");
    let swapped: Vec<[i64; 2]> = actual.iter().map(|&[r, c]| [c, r]).collect();
    assert_ne!(actual, swapped, "control4 swapped axes accepted");
    println!("PASS loopvl_compact_positions positions=580 control4=raw_index,swapped_axes");
    Ok(())
}
