use anyhow::{Context, Result, ensure};
use flyingfish::{
    model_spec::ModelSpec,
    music3::{ArCheckpoint, DepthConfig, MemoryRequest},
};
use serde_json::{Value, json};

#[test]
fn music3_geometry_rejects_wrong_attention_and_codebook_layouts() -> Result<()> {
    let root: Value = serde_json::from_str(include_str!("data/music3-qwen3-config.json"))?;
    let spec = ModelSpec::qwen3(&root)?;
    ensure!(
        (spec.hidden, spec.q_width, spec.kv_heads, spec.head_dim) == (4096, 4096, 8, 128),
        "Qwen3 geometry"
    );
    ensure!(
        spec.layers.len() == 36 && spec.rotary_dim == 128 && spec.eos.is_empty(),
        "Qwen3 position/EOS contract"
    );
    for (pointer, value) in [
        ("/attention_bias", json!(true)),
        ("/layer_types/0", json!("sliding_attention")),
        ("/num_key_value_heads", json!(3)),
        ("/head_dim", json!(127)),
        ("/rope_parameters/rope_type", json!("linear")),
        ("/tie_word_embeddings", json!(true)),
    ] {
        let mut bad = root.clone();
        *bad.pointer_mut(pointer).context("control field missing")? = value;
        ensure!(
            ModelSpec::qwen3(&bad).is_err(),
            "accepted Qwen3 mutant: {pointer}"
        );
    }
    let root: Value = serde_json::from_str(include_str!("data/music3-depth-config.json"))?;
    serde_json::from_value::<DepthConfig>(root.clone())?.validate()?;
    let mut bad = root;
    bad["num_codebooks"] = json!(7);
    ensure!(
        serde_json::from_value::<DepthConfig>(bad)?
            .validate()
            .is_err(),
        "accepted depth offset mutant"
    );
    Ok(())
}

#[test]
#[ignore = "CPU-only published checkpoint audit; set FF_MODEL to Music3"]
fn music3_published_reader_and_capacity() -> Result<()> {
    let dir = std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
    let ar = ArCheckpoint::open(&dir)?;
    ensure!(
        ar.language_manifest.entries.len() == 399
            && ar.language_manifest.resident_bytes == 17_168_951_296,
        "global tensor census"
    );
    ensure!(
        ar.depth_manifest.entries.len() == 47 && ar.depth_manifest.resident_bytes == 1_292_050_432,
        "depth tensor census"
    );
    for (prompt, frames, device_gb) in [
        (130, 25, 22),
        (5000, 1500, 22),
        (5000, 9000, 22),
        (130, 25, 14),
    ] {
        let request = MemoryRequest {
            prompt,
            frames,
            device_free: device_gb * 1_000_000_000,
            device_total: 24 << 30,
            host_available: 20 << 30,
            workspace_bytes: 512 << 20,
        };
        let plan = ar.memory_plan(&request)?;
        ensure!(
            plan.capacity == prompt + frames
                && plan.global_kv_bytes == 294_912 * (prompt + frames)
                && plan.depth_kv_bytes == 2_097_152,
            "paired KV capacity"
        );
        ensure!(
            plan.device_bytes + request.device_total / 16 <= request.device_free,
            "device budget exceeded"
        );
        ensure!(
            plan.host_fixed_bytes + plan.placement.host_bytes <= request.host_available,
            "host budget exceeded"
        );
        ensure!(
            (prompt == 130 && device_gb == 22) == (plan.placement.slot_bytes == 0),
            "residency derivation"
        );
        let mut invalid = request;
        invalid.host_available = plan.host_fixed_bytes - 1;
        ensure!(
            ar.memory_plan(&invalid).is_err(),
            "accepted host capacity mutant"
        );
        invalid.device_free = plan.depth_kv_bytes;
        ensure!(
            ar.memory_plan(&invalid).is_err(),
            "accepted capacity mutant"
        );
        println!(
            "PASS music3_capacity prompt={prompt} frames={frames} plan={}",
            serde_json::to_string(&plan)?
        );
    }
    println!(
        "PASS music3_published_reader language_tensors=399 depth_tensors=47 bytes=18461001728 capacity_arms=4"
    );
    Ok(())
}
