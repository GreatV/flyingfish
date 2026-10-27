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
    Ok(())
}
