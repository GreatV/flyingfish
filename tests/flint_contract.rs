//! Flint model-contract gates.
use anyhow::{Context, Result, ensure};
use flyingfish::{
    checkpoint::{LoadPolicy, MappedCheckpoint, TensorEntry},
    experts::{
        catalogue::{ExpertCatalogue, ExpertGeometry, ExpertRange},
        disk::DiskReader,
    },
    model_spec::{Architecture, ExpertFormat, FfnKind, ModelSpec},
};
use std::{env, path::PathBuf};

#[test]
fn flint_config_normalizes_the_published_layer_plan() -> Result<()> {
    let root: serde_json::Value = serde_json::from_str(include_str!("data/flint-config.json"))?;
    let spec = ModelSpec::flint(
        &root,
        &serde_json::json!({"eos_token_id": [150001, 150003, 150005, 150018]}),
    )?;
    ensure!(
        spec.architecture == Architecture::Flint
            && spec.hidden == 2048
            && spec.intermediate == 6144
            && spec.vocab == 150144
            && spec.q_heads == 32
            && spec.kv_heads == 4
            && spec.head_dim == 128
            && spec.q_width == 4096
            && spec.max_position == 262144
            && spec.rope_theta == 10000.0
            && spec.rotary_dim == 128,
        "unsupported normalized Flint geometry: {spec:?}"
    );
    ensure!(spec.layers.len() == 49, "Flint layer count");
    ensure!(
        spec.layers
            .iter()
            .filter(|layer| layer.ffn == FfnKind::Dense)
            .count()
            == 2
            && spec
                .layers
                .iter()
                .filter(|layer| layer.ffn == FfnKind::Moe)
                .count()
                == 47,
        "Flint FFN layer count"
    );
    ensure!(
        spec.layers
            .iter()
            .filter(|layer| layer.rope && layer.sliding_window == Some(513))
            .count()
            == 35,
        "Flint sliding/RoPE layer count"
    );
    ensure!(
        spec.layers
            .iter()
            .filter(|layer| !layer.rope && layer.sliding_window.is_none())
            .count()
            == 14,
        "Flint full/NoPE layer count"
    );
    let moe = spec.moe.context("Flint MoE geometry missing")?;
    ensure!(
        moe.experts == 256
            && moe.experts_per_tok == 6
            && moe.expert_intermediate == 512
            && moe.shared_intermediate == 1024
            && !moe.shared_gate,
        "Flint MoE geometry: {moe:?}"
    );
    ensure!(
        spec.expert_format == Some(ExpertFormat::Bf16Separate),
        "Flint expert format"
    );

    let mut mutant = root.clone();
    mutant["unknown_flint_field"] = serde_json::json!(1);
    let error = ModelSpec::flint(&mutant, &serde_json::Value::Null)
        .unwrap_err()
        .to_string();
    ensure!(
        error.contains("unknown flint config key unknown_flint_field"),
        "mutant config rejection: {error}"
    );
    Ok(())
}

#[test]
#[ignore = "CPU-only real checkpoint; set FF_MODEL"]
fn flint_manifest_and_separate_experts_match_the_checkpoint() -> Result<()> {
    let dir = PathBuf::from(
        env::var_os("FF_MODEL")
            .context("FF_MODEL missing; point it at the published Flint directory")?,
    );
    let spec = ModelSpec::read(&dir)?;
    ensure!(
        spec.architecture == Architecture::Flint,
        "FF_MODEL is not Flint: {spec:?}"
    );
    let checkpoint = MappedCheckpoint::open(&dir)?;
    let mut manifest = checkpoint.manifest(&spec, LoadPolicy::TextOnly)?;
    ensure!(
        manifest.entries.len() == 36_734 && manifest.resident_bytes == 79_570_574_912,
        "Flint manifest count={} resident_bytes={}",
        manifest.entries.len(),
        manifest.resident_bytes
    );
    ensure!(
        manifest
            .entries
            .values()
            .filter(|entry| entry.role.starts_with("mlp.experts."))
            .count()
            == 36_096,
        "Flint routed-expert tensor count"
    );
    ensure!(
        manifest
            .entries
            .values()
            .filter(|entry| entry.role == "self_attn.softmax_offset")
            .count()
            == 49,
        "Flint attention-sink count"
    );
    ensure!(
        !manifest
            .entries
            .values()
            .any(|entry| entry.role == "mlp.shared_expert_gate.weight"),
        "Flint unexpectedly has a shared-expert gate"
    );

    let geometry = ExpertGeometry::from_model(&spec)?;
    ensure!(
        geometry.layers == 47
            && geometry.first_layer == 2
            && geometry.experts == 256
            && geometry.top_k == 6
            && geometry.hidden == 2048
            && geometry.intermediate == 512,
        "Flint expert geometry: {geometry:?}"
    );
    let catalogue =
        ExpertCatalogue::from_manifest(&manifest, geometry, ExpertFormat::Bf16Separate)?;
    ensure!(
        catalogue.slot_bytes() == 6_291_456,
        "Flint expert slot bytes={}",
        catalogue.slot_bytes()
    );
    for (logical, checkpoint_layer, expert) in [(0, 2, 0), (23, 25, 127), (46, 48, 255)] {
        let ExpertRange::Bf16Separate { gate, up, down } = catalogue.range(logical, expert)? else {
            anyhow::bail!("Flint separate expert range missing: {logical}/{expert}");
        };
        for (name, native) in [
            (
                format!("model.layers.{checkpoint_layer}.mlp.experts.{expert}.gate_proj.weight"),
                gate,
            ),
            (
                format!("model.layers.{checkpoint_layer}.mlp.experts.{expert}.up_proj.weight"),
                up,
            ),
            (
                format!("model.layers.{checkpoint_layer}.mlp.experts.{expert}.down_proj.weight"),
                down,
            ),
        ] {
            let entry = manifest.entries.get(&name).with_context(|| name.clone())?;
            ensure!(
                native.offset == entry.data_start + entry.data_offsets[0]
                    && native.len == entry.data_offsets[1] - entry.data_offsets[0]
                    && native.shard.as_ref() == entry.shard,
                "native range mismatch for {name}"
            );
        }
    }

    let reader = DiskReader::open(&catalogue, &dir, 1)?;
    let ring = reader.read_expert(0, 0, &catalogue)?;
    let values = reader.values(ring)?;
    let matrix_words = 512 * 2048;
    for (index, role) in [(0, "gate_proj"), (1, "up_proj"), (2, "down_proj")] {
        let name = format!("model.layers.2.mlp.experts.0.{role}.weight");
        let entry = manifest.entries.get(&name).context(name.clone())?;
        let expected = read_native_bf16(&dir, entry)?;
        ensure!(
            values[index * matrix_words..(index + 1) * matrix_words]
                .iter()
                .map(|value| half::bf16::to_bits(*value))
                .zip(expected.iter().map(|value| half::bf16::to_bits(*value)))
                .all(|(a, b)| a == b),
            "disk reader changed native BF16 bits for {name}"
        );
    }

    let mutant = &mut manifest;
    let name = "model.layers.2.mlp.experts.0.gate_proj.weight";
    mutant.entries.get_mut(name).context(name)?.shape[0] += 1;
    let error = ExpertCatalogue::from_manifest(mutant, geometry, ExpertFormat::Bf16Separate)
        .unwrap_err()
        .to_string();
    ensure!(
        error.contains("separate BF16 expert tensor") && error.contains(name),
        "mutant expert rejection: {error}"
    );
    Ok(())
}

fn read_native_bf16(dir: &std::path::Path, entry: &TensorEntry) -> Result<Vec<half::bf16>> {
    use std::os::unix::fs::FileExt;
    let mut bytes = vec![0u8; entry.bytes];
    std::fs::File::open(dir.join(entry.shard.as_str()))?
        .read_exact_at(&mut bytes, entry.data_start + entry.data_offsets[0])?;
    ensure!(bytes.len().is_multiple_of(2), "odd BF16 byte count");
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|word| half::bf16::from_le_bytes(*word))
        .collect())
}
