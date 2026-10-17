use anyhow::{Context, Result};
use flyingfish::{
    checkpoint::{LoadPolicy, MappedCheckpoint, TensorScope},
    model_spec::{Architecture, LinearFormat, Mixer, ModelSpec},
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, io::Write, path::Path};

fn config() -> Value {
    serde_json::from_str(include_str!("data/minicpm-config.json")).unwrap()
}
fn generation() -> Value {
    json!({"eos_token_id": [248044, 248046]})
}
fn spec() -> ModelSpec {
    ModelSpec::minicpm(&config(), &generation()).unwrap()
}

#[test]
fn spec_records_text_moe_and_vision_geometry() {
    let s = spec();
    assert_eq!(s.architecture, Architecture::MiniCpmV47);
    assert_eq!(s.linear_format, LinearFormat::Bf16);
    assert_eq!(
        (s.hidden, s.vocab, s.q_heads, s.kv_heads, s.head_dim),
        (2048, 248144, 16, 2, 256)
    );
    assert_eq!((s.q_width, s.rotary_dim), (4096, 64));
    assert_eq!(s.max_position, 262144);
    assert_eq!(s.eos, vec![248044, 248046]);
    assert_eq!(s.layers.len(), 40);
    assert_eq!(
        s.layers
            .iter()
            .filter(|p| p.mixer == Mixer::FullAttention)
            .count(),
        10
    );
    assert_eq!(
        s.layers.iter().filter(|p| p.mixer == Mixer::Gdn).count(),
        30
    );
    assert_eq!(s.layers[39].state_index, 9);
    assert_eq!(s.layers[38].state_index, 29);
    let moe = s.moe.unwrap();
    assert_eq!(
        (
            moe.experts,
            moe.experts_per_tok,
            moe.expert_intermediate,
            moe.shared_intermediate
        ),
        (256, 8, 512, 512)
    );
    assert_eq!(s.intermediate, 512);
    let gdn = s.gdn.unwrap();
    assert_eq!(
        (
            gdn.key_heads,
            gdn.value_heads,
            gdn.key_head_dim,
            gdn.value_head_dim,
            gdn.conv_kernel
        ),
        (16, 32, 128, 128, 4)
    );
    let v = s.vision.as_ref().unwrap();
    assert_eq!(
        (v.depth, v.hidden, v.heads, v.intermediate),
        (27, 1152, 16, 4304)
    );
    assert_eq!(
        (v.patch, v.temporal_patch, v.merge, v.output),
        (14, 1, 4, 2048)
    );
    assert_eq!(v.position_side, 70);
    let m = v.minicpm.as_ref().unwrap();
    assert_eq!((m.image_size, m.insert_layer), (980, 6));
    assert_eq!(
        (v.image_token, v.video_token, v.start_token, v.end_token),
        (248056, 248057, 248078, 248079)
    );
    assert_eq!(
        (m.slice_start_token, m.slice_end_token, m.newline_token),
        (248088, 248089, 198)
    );
    assert_eq!(v.patch_elements(s.architecture).unwrap(), 588);
    let c = s.config();
    assert_eq!(c.model_type, "qwen3_5_moe_text");
    assert_eq!(
        (c.hidden_size, c.num_hidden_layers, c.vocab_size),
        (2048, 40, 248144)
    );
    assert_eq!(c.eos_token_id, s.eos);
    assert!(ModelSpec::minicpm(&config(), &Value::Null).is_err());
}

fn at<'a>(node: &'a mut Value, part: &str) -> &'a mut Value {
    match node {
        Value::Array(values) => &mut values[part.parse::<usize>().unwrap()],
        _ => &mut node[part],
    }
}

fn set(root: &mut Value, pointer: &str, value: Value) {
    let parts: Vec<&str> = pointer.trim_start_matches('/').split('/').collect();
    let (last, parents) = parts.split_last().unwrap();
    let mut node = root;
    for part in parents {
        node = at(node, part);
    }
    *at(node, last) = value;
}

#[test]
fn spec_rejects_every_unsupported_field() {
    for (pointer, value) in [
        ("/text_config/hidden_size", json!(0)),
        ("/text_config/num_experts_per_tok", json!(257)),
        ("/text_config/num_key_value_heads", json!(3)),
        ("/text_config/linear_num_value_heads", json!(33)),
        ("/text_config/num_hidden_layers", json!(41)),
        ("/text_config/layer_types/3", json!("linear_attention")),
        ("/text_config/intermediate_size", json!(512)),
        ("/text_config/rope_parameters/rope_type", json!("linear")),
        (
            "/text_config/rope_parameters/mrope_section",
            json!([10, 11, 11]),
        ),
        ("/text_config/partial_rotary_factor", json!(0.3)),
        ("/text_config/rms_norm_eps", json!(0.0)),
        ("/text_config/attn_output_gate", json!(false)),
        ("/text_config/attention_bias", json!(true)),
        ("/text_config/attention_dropout", json!(0.1)),
        ("/text_config/output_router_logits", json!(true)),
        ("/text_config/mtp_use_dedicated_embeddings", json!(true)),
        ("/vision_config/hidden_act", json!("gelu")),
        ("/vision_config/image_size", json!(981)),
        ("/vision_config/window_kernel_size", json!([4, 4])),
        ("/insert_layer_id", json!(5)),
        ("/downsample_mode", json!("4x")),
        ("/downsample_mode", json!("8x")),
        ("/merge_kernel_size", json!([4, 4])),
        ("/merger_times", json!(2)),
        ("/image_start_id", json!(248053)),
        ("/slice_end_id", json!(248088)),
        ("/newline_id", json!(13)),
        ("/architectures", json!(["MiniCPMForConditionalGeneration"])),
        ("/dtype", json!("float16")),
        ("/mrope_mode", json!("legacy")),
        ("/model_type", json!("minicpmv")),
        (
            "/quantization_config",
            json!({"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic", "weight_block_size": [64, 128]}),
        ),
    ] {
        let mut bad = config();
        set(&mut bad, pointer, value);
        assert!(
            ModelSpec::minicpm(&bad, &generation()).is_err(),
            "{pointer}"
        );
    }
    assert!(ModelSpec::minicpm(&config(), &json!({"eos_token_id": 248144})).is_err());
}

fn write_checkpoint(
    real: &Path,
    root: &Path,
    rename: Option<(&str, &str)>,
    reshape: Option<(&str, &[usize])>,
) -> Result<()> {
    std::fs::create_dir_all(root)?;
    for name in ["config.json", "generation_config.json"] {
        std::fs::copy(real.join(name), root.join(name))?;
    }
    let index: Value =
        serde_json::from_slice(&std::fs::read(real.join("model.safetensors.index.json"))?)?;
    let mut weight_map = index["weight_map"]
        .as_object()
        .context("weight_map")?
        .clone();
    let mut shards: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (name, shard) in &weight_map {
        shards
            .entry(shard.as_str().context("shard name")?.into())
            .or_default()
            .push(name.clone());
    }
    if let Some((old, new)) = rename {
        let shard = weight_map.remove(old).context("renamed tensor missing")?;
        weight_map.insert(new.into(), shard);
    }
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"weight_map": weight_map}))?,
    )?;
    for (shard, mut names) in shards {
        names.sort();
        let file = std::fs::File::open(real.join(&shard))?;
        let map = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let tensors = safetensors::SafeTensors::deserialize(&map)?;
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;
        for name in &names {
            let t = tensors.tensor(name)?;
            let mut shape = t.shape().to_vec();
            if let Some((target, replacement)) = reshape
                && name == target
            {
                shape = replacement.to_vec();
            }
            let elements: usize = shape.iter().try_fold(1usize, |n, &d| {
                n.checked_mul(d).context("fixture shape overflow")
            })?;
            let bytes = elements
                .checked_mul(t.dtype().bitsize())
                .context("fixture size overflow")?
                / 8;
            let written = rename
                .and_then(|(old, new)| (old == name.as_str()).then_some(new))
                .unwrap_or(name);
            header.insert(
                written.into(),
                json!({
                    "dtype": t.dtype().to_string(),
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        }
        let serialized = serde_json::to_vec(&header)?;
        let mut out = std::fs::File::create(root.join(&shard))?;
        out.write_all(&(serialized.len() as u64).to_le_bytes())?;
        out.write_all(&serialized)?;
        out.set_len(8 + serialized.len() as u64 + offset as u64)?;
    }
    Ok(())
}

fn temp_root(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("ff-minicpm-manifest-{tag}-{}", std::process::id()))
}

#[test]
#[ignore = "CPU-only checkpoint header audit; set FF_MODEL"]
fn checkpoint_manifest_classifies_every_tensor() -> Result<()> {
    let dir = std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
    let s = ModelSpec::read(&dir)?;
    assert_eq!(s.architecture, Architecture::MiniCpmV47);
    let mapped = MappedCheckpoint::open(&dir)?;
    for (policy, bytes, text, vision) in [
        (LoadPolicy::TextOnly, 69_319_779_584usize, 693, 0),
        (LoadPolicy::TextWithVision, 70_425_751_648, 693, 459),
        (LoadPolicy::VisionOnly, 1_105_972_064, 0, 459),
    ] {
        let m = mapped.manifest(&s, policy)?;
        assert_eq!(m.resident_bytes, bytes, "{policy:?}");
        assert_eq!(m.entries.len(), 1152, "{policy:?}");
        assert_eq!(
            m.entries
                .values()
                .filter(|e| e.load && e.scope == TensorScope::Text)
                .count(),
            text,
            "{policy:?}"
        );
        assert_eq!(
            m.entries
                .values()
                .filter(|e| e.load && e.scope == TensorScope::Vision)
                .count(),
            vision,
            "{policy:?}"
        );
        assert_eq!(
            m.entries
                .values()
                .filter(|e| e.scope == TensorScope::Mtp)
                .count(),
            0
        );
    }
    assert!(mapped.manifest(&s, LoadPolicy::TextWithMtp).is_err());
    assert!(mapped.manifest(&s, LoadPolicy::MtpOnly).is_err());
    Ok(())
}

#[test]
#[ignore = "CPU-only checkpoint header audit; set FF_MODEL"]
fn manifest_rejects_renamed_and_misshaped_tensors() -> Result<()> {
    let real = std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
    let root = temp_root("negative");
    let run = |rename: Option<(&str, &str)>,
               reshape: Option<(&str, &[usize])>|
     -> std::result::Result<String, String> {
        write_checkpoint(&real, &root, rename, reshape).map_err(|e| e.to_string())?;
        let s = ModelSpec::read(&root).map_err(|e| e.to_string())?;
        let mapped = MappedCheckpoint::open(&root).map_err(|e| e.to_string())?;
        let outcome = mapped
            .manifest(&s, LoadPolicy::TextWithVision)
            .map(|m| m.resident_bytes.to_string())
            .map_err(|e| e.to_string());
        std::fs::remove_dir_all(&root).map_err(|e| e.to_string())?;
        outcome
    };
    assert_eq!(run(None, None).map_err(anyhow::Error::msg)?, "70425751648");
    let err = run(Some(("lm_head.weight", "lm_head_renamed.weight")), None)
        .map_err(anyhow::Error::msg)
        .unwrap_err();
    assert!(err.to_string().contains("lm_head_renamed.weight"), "{err}");
    let err = run(None, Some(("lm_head.weight", &[248144, 2047])))
        .map_err(anyhow::Error::msg)
        .unwrap_err();
    assert!(err.to_string().contains("dtype/shape mismatch"), "{err}");
    Ok(())
}
