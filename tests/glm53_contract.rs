use anyhow::{Context, Result};
use flyingfish::{
    checkpoint::{LoadPolicy, MappedCheckpoint, TensorScope},
    experts::catalogue::{ExpertCatalogue, ExpertGeometry, ExpertLayout, ExpertRange},
    model_spec::{
        Architecture, ExpertFormat, LinearFormat, Mixer, ModelSpec, RoutedWeightBoundary,
        RouterScore, SwigluRounding,
    },
};
use safetensors::SafeTensors;
use serde_json::{Value, json};
use std::{collections::BTreeSet, io::Write, path::Path};

fn spec() -> ModelSpec {
    let root: Value = serde_json::from_str(include_str!("data/glm53-flash-config.json")).unwrap();
    let generation = json!({"eos_token_id":[154820,154827,154829]});
    ModelSpec::glm5_next(&root, &generation).unwrap()
}

#[test]
fn glm53_normalizes_every_layer_and_fp8_format() {
    let spec = spec();
    assert_eq!(spec.architecture, Architecture::Glm5Next);
    assert_eq!(
        spec.linear_format,
        LinearFormat::fp8_block([128, 128], safetensors::Dtype::F32).unwrap()
    );
    assert_eq!(
        spec.expert_format,
        Some(ExpertFormat::Fp8W8a16Block {
            block: [128, 128],
            scale_dtype: safetensors::Dtype::F32,
        })
    );
    assert_eq!(spec.layers.len(), 45);
    assert_eq!(
        spec.layers
            .iter()
            .filter(|layer| layer.mixer == Mixer::FullAttention)
            .count(),
        11
    );
    assert_eq!(
        spec.layers
            .iter()
            .filter(|layer| layer.mixer == Mixer::Gdn)
            .count(),
        34
    );
    assert_eq!(spec.layers[43].state_index, 10);
    assert_eq!(spec.layers[44].state_index, 33);
    let glm = spec.glm5_next.as_ref().unwrap();
    assert_eq!(glm.dense_layers, 3);
    assert_eq!(glm.nextn_layers, 1);
    assert_eq!((glm.hc_mult, glm.hc_sinkhorn_iters), (4, 20));
    assert_eq!((glm.q_lora_rank, glm.kv_lora_rank), (1536, 512));
    assert_eq!((glm.index_heads, glm.index_head_dim), (32, 128));
    assert_eq!((glm.index_topk, glm.index_kpool), (2048, 4));
    let moe = spec.moe.as_ref().unwrap();
    assert_eq!(moe.score, RouterScore::Sigmoid);
    assert_eq!(spec.swiglu_rounding(), SwigluRounding::ClampedTwice);
    assert_eq!(
        spec.routed_weight_boundary(),
        RoutedWeightBoundary::BeforeContributionBf16
    );
    assert_eq!((moe.experts, moe.experts_per_tok), (288, 8));
    assert_eq!(
        (moe.expert_intermediate, moe.shared_intermediate),
        (2048, 2048)
    );
    let geometry = ExpertGeometry::from_model(&spec).unwrap();
    assert_eq!(geometry.layers, 42);
    let layout = ExpertLayout::new(
        ExpertFormat::Fp8W8a16Block {
            block: [128, 128],
            scale_dtype: safetensors::Dtype::F32,
        },
        &geometry,
    )
    .unwrap();
    assert_eq!(layout.slot_bytes(), 25_171_968);
    let slot = layout.fp8_slot().unwrap();
    assert_eq!(slot.gate_scale.start, 25_165_824);
    assert_eq!(slot.up_scale.start, 25_167_872);
    assert_eq!(slot.down_scale.start, 25_169_920);
    assert_eq!(slot.down_scale.end, 25_171_968);
}

#[test]
fn glm53_rejects_unsupported_semantics() {
    let root: Value = serde_json::from_str(include_str!("data/glm53-flash-config.json")).unwrap();
    let generation = json!({"eos_token_id":[154820,154827,154829]});
    for (pointer, value) in [
        ("/text_config/index_kpool_compress", json!(false)),
        ("/text_config/indexer_types/0", json!("shared")),
        ("/text_config/mlp_layer_types/0", json!("sparse")),
        (
            "/text_config/linear_attn_config/gate_lower_bound",
            json!(1.0),
        ),
        ("/text_config/moe_router_dtype", json!("bfloat16")),
        ("/text_config/routed_scaling_factor", json!(1e100)),
        ("/text_config/rms_norm_eps", json!(1e-100)),
        ("/text_config/qk_rope_head_dim", json!(64)),
        ("/quantization_config/weight_block_size", json!([64, 128])),
    ] {
        let mut bad = root.clone();
        *bad.pointer_mut(pointer).unwrap() = value;
        assert!(
            ModelSpec::glm5_next(&bad, &generation).is_err(),
            "{pointer}"
        );
    }
    let mut bad = root;
    bad["quantization_config"]["unknown"] = Value::Null;
    assert!(ModelSpec::glm5_next(&bad, &generation).is_err());
}

fn write_checkpoint(
    real: &Path,
    root: &Path,
    rename: Option<(&str, &str)>,
    reshape: Option<(&str, &[usize])>,
) -> Result<()> {
    if root.exists() {
        std::fs::remove_dir_all(root)?;
    }
    std::fs::create_dir_all(root)?;
    for name in ["config.json", "generation_config.json"] {
        std::fs::copy(real.join(name), root.join(name))?;
    }
    let mut index: Value =
        serde_json::from_slice(&std::fs::read(real.join("model.safetensors.index.json"))?)?;
    let shards = index["weight_map"]
        .as_object()
        .context("weight_map missing")?
        .values()
        .map(|value| value.as_str().unwrap().to_owned())
        .collect::<BTreeSet<_>>();
    for shard in shards {
        let file = std::fs::File::open(real.join(&shard))?;
        let map = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let tensors = SafeTensors::deserialize(&map)?;
        let mut header = serde_json::Map::new();
        let mut extent = 0u64;
        for name in tensors.names() {
            let tensor = tensors.tensor(name)?;
            let output_name = rename
                .and_then(|(from, to)| (name == from).then_some(to))
                .unwrap_or(name);
            let shape = reshape
                .and_then(|(from, shape)| (name == from).then_some(shape))
                .unwrap_or(tensor.shape());
            let bytes = (tensor.dtype().bitsize() * shape.iter().product::<usize>() / 8) as u64;
            header.insert(
                output_name.into(),
                json!({
                    "dtype": tensor.dtype().to_string(),
                    "shape": shape,
                    "data_offsets": [extent, extent + bytes],
                }),
            );
            if output_name != name {
                let shard_name = index["weight_map"][name].as_str().unwrap().to_owned();
                index["weight_map"].as_object_mut().unwrap().remove(name);
                index["weight_map"][output_name] = Value::String(shard_name);
            }
            extent += bytes;
        }
        let serialized = serde_json::to_vec(&header)?;
        let mut out = std::fs::File::create(root.join(&shard))?;
        out.write_all(&(serialized.len() as u64).to_le_bytes())?;
        out.write_all(&serialized)?;
        out.set_len(8 + serialized.len() as u64 + extent)?;
    }
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::to_vec(&index)?,
    )?;
    Ok(())
}

#[test]
#[ignore = "CPU-only official GLM-5.3 checkpoint header audit; set FF_MODEL"]
fn glm53_checkpoint_manifest_and_mutants() -> Result<()> {
    let real = std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
    let spec = ModelSpec::read(&real)?;
    let mapped = MappedCheckpoint::open(&real)?;
    for (policy, bytes, text, mtp, vision) in [
        (LoadPolicy::TextOnly, 319_706_118_392usize, 74_001, 0, 0),
        (LoadPolicy::TextWithVision, 320_833_372_408, 74_001, 0, 347),
        (LoadPolicy::TextWithMtp, 327_199_517_560, 74_001, 1_760, 0),
        (LoadPolicy::VisionOnly, 1_127_254_016, 0, 0, 347),
        (LoadPolicy::MtpOnly, 7_493_399_168, 0, 1_760, 0),
    ] {
        let manifest = mapped.manifest(&spec, policy)?;
        assert_eq!(manifest.resident_bytes, bytes, "{policy:?}");
        assert_eq!(manifest.entries.len(), 76_108, "{policy:?}");
        for (scope, expected) in [
            (TensorScope::Text, text),
            (TensorScope::Mtp, mtp),
            (TensorScope::Vision, vision),
        ] {
            assert_eq!(
                manifest
                    .entries
                    .values()
                    .filter(|entry| entry.load && entry.scope == scope)
                    .count(),
                expected,
                "{policy:?} {scope:?}"
            );
        }
    }
    let text_manifest = mapped.manifest(&spec, LoadPolicy::TextOnly)?;
    let scale = &text_manifest.entries["model.language_model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv"];
    assert_eq!(scale.dtype, safetensors::Dtype::F32);
    assert_eq!(scale.shape, [16, 32]);
    let catalogue = ExpertCatalogue::from_manifest(
        &text_manifest,
        ExpertGeometry::from_model(&spec)?,
        ExpertFormat::Fp8W8a16Block {
            block: [128, 128],
            scale_dtype: safetensors::Dtype::F32,
        },
    )?;
    let ExpertRange::Fp8 {
        gate, down_scale, ..
    } = catalogue.range(0, 0)?
    else {
        anyhow::bail!("FP8 range expected");
    };
    let first =
        &text_manifest.entries["model.language_model.layers.3.mlp.experts.0.gate_proj.weight"];
    assert_eq!(gate.offset, first.data_start + first.data_offsets[0]);
    assert_eq!(down_scale.len, 2048);
    let ExpertRange::Fp8 { gate, .. } = catalogue.range(41, 287)? else {
        anyhow::bail!("FP8 range expected");
    };
    let last =
        &text_manifest.entries["model.language_model.layers.44.mlp.experts.287.gate_proj.weight"];
    assert_eq!(gate.offset, last.data_start + last.data_offsets[0]);
    drop(mapped);

    let root = std::env::temp_dir().join(format!("ff-glm53-manifest-{}", std::process::id()));
    let run = |rename: Option<(&str, &str)>,
               reshape: Option<(&str, &[usize])>|
     -> std::result::Result<String, String> {
        write_checkpoint(&real, &root, rename, reshape).map_err(|error| error.to_string())?;
        let spec = ModelSpec::read(&root).map_err(|error| error.to_string())?;
        let mapped = MappedCheckpoint::open(&root).map_err(|error| error.to_string())?;
        let result = mapped
            .manifest(&spec, LoadPolicy::TextOnly)
            .map(|manifest| manifest.resident_bytes.to_string())
            .map_err(|error| error.to_string());
        std::fs::remove_dir_all(&root).map_err(|error| error.to_string())?;
        result
    };
    assert_eq!(run(None, None).map_err(anyhow::Error::msg)?, "319706118392");
    let error = run(Some(("lm_head.weight", "lm_head_renamed.weight")), None)
        .map_err(anyhow::Error::msg)
        .unwrap_err();
    assert!(
        error.to_string().contains("lm_head_renamed.weight"),
        "{error}"
    );
    let error = run(
        None,
        Some((
            "model.language_model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv",
            &[15, 32],
        )),
    )
    .map_err(anyhow::Error::msg)
    .unwrap_err();
    assert!(
        error.to_string().contains("dtype/shape mismatch"),
        "{error}"
    );
    Ok(())
}
