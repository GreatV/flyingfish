use flyingfish::{
    model_spec::{GdnConfig, LinearFormat, Mixer, ModelSpec},
    op_contract::{AttentionWorkspace, SequenceMeta},
};
use serde_json::{Value, json};
#[test]
fn g0_normalized_geometry_and_rejections() {
    let root: Value = serde_json::from_str(include_str!("data/q38-config.json")).unwrap();
    let spec = ModelSpec::qwen(&root, &json!({"eos_token_id":[248046,248044,248046]})).unwrap();
    assert_eq!((spec.hidden, spec.q_width), (5120, 6144));
    assert_eq!(
        spec.gdn,
        Some(GdnConfig {
            key_heads: 16,
            value_heads: 48,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel: 4,
        })
    );
    assert_eq!(spec.moe, None);
    assert_eq!(
        spec.linear_format,
        LinearFormat::fp8_block([128, 128], safetensors::Dtype::BF16).unwrap()
    );
    let canonical = serde_json::to_string(&spec).unwrap();
    assert_eq!(
        canonical,
        include_str!("data/q38-spec.json").trim_end_matches('\n'),
        "Q38 spec serialization changed"
    );
    let restored: ModelSpec = serde_json::from_str(&canonical).unwrap();
    assert_eq!(serde_json::to_string(&restored).unwrap(), canonical);
    let public = spec.config();
    assert_eq!(
        (
            public.hidden_size,
            public.num_attention_heads * public.head_dim
        ),
        (5120, 6144)
    );
    assert_eq!(public.model_type, "qwen3_5_text");
    assert_eq!(public.eos_token_id, spec.eos);
    assert_eq!(
        flyingfish::op_contract::GdnSpec::q38().conv_dim().unwrap(),
        10240
    );
    assert_eq!(spec.eos, vec![248044, 248046]);
    assert_eq!(
        spec.layers
            .iter()
            .filter(|p| p.mixer == Mixer::FullAttention)
            .count(),
        16
    );
    assert_eq!(spec.layers[63].state_index, 15);
    assert_eq!(spec.layers[62].state_index, 47);
    assert_eq!(
        ModelSpec::qwen(&root, &Value::Null).unwrap().eos,
        vec![248044]
    );
    for (pointer, value) in [
        ("/text_config/layer_types/3", json!("linear_attention")),
        ("/text_config/rope_parameters/rope_type", json!("linear")),
        (
            "/text_config/rope_parameters/mrope_section",
            json!([10, 11, 11]),
        ),
        ("/text_config/attention_bias", json!(true)),
        ("/text_config/attn_output_gate", json!(false)),
        ("/text_config/partial_rotary_factor", json!(0.3)),
        ("/quantization_config/quant_method", json!("awq")),
        ("/quantization_config/weight_block_size", json!([64, 128])),
    ] {
        let mut bad = root.clone();
        *bad.pointer_mut(pointer).unwrap() = value;
        assert!(ModelSpec::qwen(&bad, &Value::Null).is_err(), "{pointer}");
    }
    assert!(ModelSpec::qwen(&root, &json!({"eos_token_id":248320})).is_err());
    let mut unquantized = root.clone();
    unquantized
        .as_object_mut()
        .unwrap()
        .remove("quantization_config");
    let unquantized =
        ModelSpec::qwen(&unquantized, &Value::Null).expect("absent quantization_config is BF16");
    assert_eq!(unquantized.linear_format, LinearFormat::Bf16);
}
#[test]
fn g4_workspace_segmentation_and_metadata() {
    let w = AttentionWorkspace::new(
        &flyingfish::op_contract::FullAttentionSpec::q38(),
        65,
        16640,
        128,
    )
    .unwrap();
    assert_eq!((w.segment_rows, w.max_parts), (64, 130));
    assert_eq!(w.k_norm, 64 * 24 * 130 * 258 * 4);
    assert_eq!(w.bytes, w.k_norm + 65 * 1024 * 2);
    assert!(
        [w.p_max, w.p_sum, w.p_out, w.k_norm, w.bytes]
            .iter()
            .all(|x| x % 256 == 0)
    );
    assert!(
        AttentionWorkspace::new(
            &flyingfish::op_contract::FullAttentionSpec::q38(),
            1,
            16640,
            0
        )
        .is_err()
    );
    assert!(
        AttentionWorkspace::new(
            &flyingfish::op_contract::FullAttentionSpec::q38(),
            65,
            64,
            128
        )
        .is_err()
    );
    let good = SequenceMeta {
        prefix: 63,
        valid_rows: 65,
        key_count: 128,
        capacity: 16640,
        causal: 1,
        reserved: [0; 3],
    };
    good.validate(16640, 65).unwrap();
    let bad = SequenceMeta {
        key_count: 127,
        ..good
    };
    assert!(bad.validate(16640, 65).is_err());
    let bad = SequenceMeta {
        prefix: i32::MAX,
        valid_rows: 1,
        ..good
    };
    assert!(bad.validate(16640, 65).is_err());
}

#[test]
fn g1_all_fp8_codes_and_effective_rounding() {
    use flyingfish::fp8::{decode_e4m3, effective_weight};
    use half::bf16;
    for code in 0u8..=255 {
        let sign = if code < 128 { 1.0f32 } else { -1.0f32 };
        let exp = i32::from((code >> 3) & 15);
        let mant = i32::from(code & 7);
        let got = decode_e4m3(code);
        if exp == 15 && mant == 7 {
            assert!(got.is_nan());
            continue;
        }
        let want = sign
            * if exp == 0 {
                mant as f32 * 2.0f32.powi(-9)
            } else {
                (8 + mant) as f32 * 2.0f32.powi(exp - 10)
            };
        assert_eq!(got.to_bits(), want.to_bits(), "code {code:02x}");
        for s in [0.0, 1.0, 0.00390625, 1.0078125, 3.140625] {
            let scale = bf16::from_f32(s);
            let product = want * scale.to_f32();
            let bits = product.to_bits();
            let rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16;
            assert_eq!(effective_weight(code, scale).to_bits(), rounded);
        }
    }
    assert_eq!(decode_e4m3(0x7e), 448.0);
    assert_eq!(decode_e4m3(0xfe), -448.0);
}

#[test]
fn g0_checkpoint_index_and_header_rejections() {
    use flyingfish::checkpoint::MappedCheckpoint;
    let dir = std::env::temp_dir().join(format!("ff-q38-checkpoint-{}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("model.safetensors");
    let write = |header: &str, data: &[u8]| {
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        std::fs::write(&path, bytes).unwrap();
    };
    write(
        r#"{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#,
        &[0, 0],
    );
    assert!(MappedCheckpoint::open(&dir).is_ok());
    let component = dir.join("depth");
    std::fs::create_dir(&component).unwrap();
    std::fs::copy(&path, component.join("diffusion_pytorch_model.safetensors")).unwrap();
    assert!(MappedCheckpoint::open_component(&component, "diffusion_pytorch_model").is_ok());
    std::fs::write(
        dir.join("outside.safetensors.index.json"),
        r#"{"weight_map":{"a":"diffusion_pytorch_model.safetensors"}}"#,
    )
    .unwrap();
    assert!(MappedCheckpoint::open_component(&component, "../outside").is_err());
    let index = dir.join("model.safetensors.index.json");
    std::fs::write(
        &index,
        r#"{"weight_map":{"a":"model.safetensors","a":"model.safetensors"}}"#,
    )
    .unwrap();
    assert!(MappedCheckpoint::open(&dir).is_err());
    std::fs::write(&index, r#"{"weight_map":{"b":"model.safetensors"}}"#).unwrap();
    assert!(MappedCheckpoint::open(&dir).is_err());
    std::fs::write(&index, r#"{"weight_map":{"a":"../model.safetensors"}}"#).unwrap();
    assert!(MappedCheckpoint::open(&dir).is_err());
    std::fs::remove_file(index).unwrap();
    write(
        r#"{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#,
        &[0, 0],
    );
    assert!(MappedCheckpoint::open(&dir).is_err());
    write(
        r#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#,
        &[0, 0],
    );
    assert!(MappedCheckpoint::open(&dir).is_err());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn g6_workspace_budget_counts_resident_once_and_checks_tail() {
    use flyingfish::{
        op_contract::GdnSpec,
        prefill::{WorkspaceBudget, WorkspaceRequest},
    };
    let gdn = GdnSpec::q38();
    gdn.validate().unwrap();
    assert_eq!(gdn.chunk, 32);
    assert_eq!(gdn.state_bytes().unwrap(), 3_145_728);
    assert_eq!(gdn.history_bytes().unwrap(), 122_880);
    assert!(GdnSpec { chunk: 64, ..gdn }.validate().is_err());
    let request = WorkspaceRequest {
        capacity: 65,
        requested: Some(65),
        free_after_resident: 9000,
        total: 16000,
        alignment: 32,
        row_bytes: 100,
        fixed_transient_bytes: 128,
    };
    let p = WorkspaceBudget::new(request, |rows| Ok(rows.div_ceil(32) * 256), Ok).unwrap();
    assert_eq!(p.planned_transient_bytes, 65 * 100 + 128 + 3 * 256);
    assert_eq!(p.plan.reserve_bytes, 1000);
    assert!(p.planned_transient_bytes + p.plan.reserve_bytes <= p.plan.free_bytes);
    assert!(
        WorkspaceBudget::new(
            WorkspaceRequest {
                free_after_resident: 8395,
                ..request
            },
            |rows| Ok(rows.div_ceil(32) * 256),
            Ok,
        )
        .is_err()
    );
    assert!(
        WorkspaceBudget::new(
            WorkspaceRequest {
                requested: None,
                ..request
            },
            |_| anyhow::bail!("GDN workspace pending"),
            Ok,
        )
        .is_err()
    );
}

#[test]
fn g9_shared_layer_program_preserves_llama_order_and_types_qwen() {
    use flyingfish::{
        model::{LAYER, LLAMA_MIXER, LayerProgram, LayerStage, layer_program},
        model_spec::ModelSpec,
    };
    let root: serde_json::Value =
        serde_json::from_str(include_str!("data/q38-config.json")).unwrap();
    let spec = ModelSpec::qwen(&root, &serde_json::Value::Null).unwrap();
    let LayerProgram::Standard(full) = layer_program(&spec, 3).unwrap() else {
        panic!("Qwen3.5 layers use the standard program");
    };
    let mut flattened = Vec::new();
    for stage in full {
        match stage {
            LayerStage::Common(op) => flattened.push(std::mem::discriminant(&op)),
            LayerStage::Mixer(Mixer::FullAttention) => {
                flattened.extend(LLAMA_MIXER.iter().map(std::mem::discriminant))
            }
            LayerStage::Ffn => flattened.extend(
                [
                    flyingfish::backend::Operator::GateUp,
                    flyingfish::backend::Operator::Activate,
                    flyingfish::backend::Operator::Down,
                ]
                .iter()
                .map(std::mem::discriminant),
            ),
            _ => panic!("unexpected GDN mixer on full layer"),
        }
    }
    assert_eq!(
        flattened,
        LAYER.iter().map(std::mem::discriminant).collect::<Vec<_>>()
    );
    let LayerProgram::Standard(first) = layer_program(&spec, 0).unwrap() else {
        panic!("Qwen3.5 layers use the standard program");
    };
    assert!(matches!(first[1], LayerStage::Mixer(Mixer::Gdn)));
    assert!(layer_program(&spec, 64).is_err());
}

#[test]
fn g6_row_work_uses_mixer_width_and_separate_projection_storage() {
    let root: serde_json::Value =
        serde_json::from_str(include_str!("data/q38-config.json")).unwrap();
    let spec = flyingfish::model_spec::ModelSpec::qwen(&root, &serde_json::Value::Null).unwrap();
    let w = flyingfish::work_layout::RowWorkSpec::new(&spec).unwrap();
    assert_eq!(
        (w.hidden, w.mixer, w.projection, w.ab),
        (5120, 6144, 14336, 96)
    );
    assert!(w.extra_mixer);
    assert_eq!(w.row_bytes().unwrap(), 200900);
    assert!(flyingfish::work_layout::RowWorkSpec::elements(usize::MAX, 2).is_err());
}

fn modelopt_nvfp4_quant_config() -> Value {
    json!({
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "dynamic": false,
                    "group_size": 16,
                    "num_bits": 4,
                    "type": "float"
                },
                "weights": {
                    "dynamic": false,
                    "group_size": 16,
                    "num_bits": 4,
                    "type": "float"
                },
                "targets": [
                    "lm_head",
                    "model.language_model.layers.0.linear_attn.in_proj_qkv"
                ]
            }
        },
        "group_size": 16,
        "ignore": [
            "model.language_model.embed_tokens",
            "model.visual*"
        ],
        "producer": {
            "name": "modelopt",
            "version": "0.48.0rc1.dev102+g28a38b6aa"
        },
        "quant_algo": "NVFP4",
        "quant_method": "modelopt"
    })
}

fn write_safetensors_file(path: &std::path::Path, entries: &[(&str, &str, &[usize])]) {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for &(name, dtype, shape) in entries {
        let elem_bytes = match dtype {
            "U8" | "F8_E4M3" => 1usize,
            "BF16" => 2,
            "F32" => 4,
            other => panic!("unexpected synthetic dtype {other}"),
        };
        let bytes = shape.iter().product::<usize>() * elem_bytes;
        header.insert(
            name.to_owned(),
            json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + bytes]
            }),
        );
        offset += bytes;
    }
    let header_bytes = serde_json::to_vec(&Value::Object(header)).unwrap();
    let mut file_bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
    file_bytes.extend_from_slice(&header_bytes);
    file_bytes.resize(file_bytes.len() + offset, 0u8);
    std::fs::write(path, file_bytes).unwrap();
}

#[test]
fn nvfp4_modelopt_config_manifest_and_controls() {
    use flyingfish::{
        checkpoint::{LoadPolicy, MappedCheckpoint},
        linear_plan::LayerDequantPlan,
    };

    let mut root: Value = serde_json::from_str(include_str!("data/q38-config.json")).unwrap();
    root["quantization_config"] = modelopt_nvfp4_quant_config();
    root["text_config"]["mtp_num_hidden_layers"] = json!(0);
    root["text_config"]
        .as_object_mut()
        .unwrap()
        .remove("mtp_use_dedicated_embeddings");
    root["text_config"]["eos_token_id"] = json!(248046);
    root["text_config"]["pad_token_id"] = json!(248044);
    root["vision_config"]["model_type"] = json!("qwen3_5_vision");
    let spec = ModelSpec::qwen(&root, &json!({"eos_token_id": 248046})).unwrap();
    assert_eq!(spec.linear_format, LinearFormat::NvFp4);
    assert_eq!(spec.mtp_layers, 0);
    assert_eq!(spec.eos, vec![248046]);
    assert!(spec.vision.is_some());
    let encoded = serde_json::to_string(&spec).unwrap();
    let decoded: ModelSpec = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded.linear_format, LinearFormat::NvFp4);

    for (pointer, bad_val) in [
        ("/quantization_config/quant_algo", json!("FP8")),
        ("/quantization_config/group_size", json!(32)),
        (
            "/quantization_config/producer/name",
            json!("compressed-tensors"),
        ),
        ("/quantization_config/producer/version", json!("")),
        (
            "/quantization_config/config_groups/group_0/weights/dynamic",
            json!(true),
        ),
        (
            "/quantization_config/config_groups/group_0/weights/num_bits",
            json!(8),
        ),
        (
            "/quantization_config/config_groups/group_0/input_activations/group_size",
            json!(32),
        ),
        (
            "/quantization_config/ignore",
            json!(["model.language_model.embed_tokens"]),
        ),
        ("/quantization_config/ignore", json!(["model.visual*"])),
        (
            "/quantization_config/ignore",
            json!([
                "model.language_model.embed_tokens",
                "model.visual*",
                "lm_head"
            ]),
        ),
        (
            "/quantization_config/config_groups/group_0/targets",
            json!(["model.language_model.layers.0.mlp.down_proj"]),
        ),
    ] {
        let mut bad = root.clone();
        *bad.pointer_mut(pointer).unwrap() = bad_val;
        assert!(ModelSpec::qwen(&bad, &Value::Null).is_err(), "{pointer}");
    }
    let mut bad_kv_quant = root.clone();
    bad_kv_quant["quantization_config"]["kv_cache_quant_algo"] = json!("FP8");
    assert!(ModelSpec::qwen(&bad_kv_quant, &Value::Null).is_err());
    let mut extra_key = root.clone();
    extra_key["quantization_config"]["unexpected_field"] = json!(1);
    assert!(ModelSpec::qwen(&extra_key, &Value::Null).is_err());

    let mut tiny = root.clone();
    tiny.as_object_mut().unwrap().remove("vision_config");
    let tc = tiny["text_config"].as_object_mut().unwrap();
    tc.insert("hidden_size".into(), json!(64));
    tc.insert("intermediate_size".into(), json!(128));
    tc.insert("vocab_size".into(), json!(256));
    tc.insert("bos_token_id".into(), json!(0));
    tc.insert("eos_token_id".into(), json!(1));
    tc.insert("pad_token_id".into(), json!(0));
    tc.insert("num_hidden_layers".into(), json!(2));
    tc.insert("full_attention_interval".into(), json!(2));
    tc.insert(
        "layer_types".into(),
        json!(["linear_attention", "full_attention"]),
    );
    tc.insert("num_attention_heads".into(), json!(2));
    tc.insert("num_key_value_heads".into(), json!(1));
    tc.insert("head_dim".into(), json!(256));
    tc.insert("linear_num_key_heads".into(), json!(2));
    tc.insert("linear_num_value_heads".into(), json!(4));
    tc.insert("linear_key_head_dim".into(), json!(16));
    tc.insert("linear_value_head_dim".into(), json!(16));
    tc.insert("linear_conv_kernel_dim".into(), json!(4));
    let tiny_spec = ModelSpec::qwen(&tiny, &json!({"eos_token_id": 1})).unwrap();

    let mut entries: Vec<(String, &'static str, Vec<usize>)> = Vec::new();
    let push_nvfp4 =
        |out: &mut Vec<(String, &'static str, Vec<usize>)>, stem: &str, n: usize, k: usize| {
            out.push((format!("{stem}.weight"), "U8", vec![n, k / 2]));
            out.push((format!("{stem}.weight_scale"), "F8_E4M3", vec![n, k / 16]));
            out.push((format!("{stem}.weight_scale_2"), "F32", vec![]));
            out.push((format!("{stem}.input_scale"), "F32", vec![]));
        };
    entries.push((
        "model.language_model.embed_tokens.weight".into(),
        "BF16",
        vec![256, 64],
    ));
    entries.push(("model.language_model.norm.weight".into(), "BF16", vec![64]));
    push_nvfp4(&mut entries, "lm_head", 256, 64);
    for (layer, is_full) in [(0usize, false), (1usize, true)] {
        let p = format!("model.language_model.layers.{layer}");
        entries.push((format!("{p}.input_layernorm.weight"), "BF16", vec![64]));
        entries.push((
            format!("{p}.post_attention_layernorm.weight"),
            "BF16",
            vec![64],
        ));
        push_nvfp4(&mut entries, &format!("{p}.mlp.gate_proj"), 128, 64);
        push_nvfp4(&mut entries, &format!("{p}.mlp.up_proj"), 128, 64);
        push_nvfp4(&mut entries, &format!("{p}.mlp.down_proj"), 64, 128);
        if is_full {
            push_nvfp4(&mut entries, &format!("{p}.self_attn.q_proj"), 1024, 64);
            push_nvfp4(&mut entries, &format!("{p}.self_attn.k_proj"), 256, 64);
            push_nvfp4(&mut entries, &format!("{p}.self_attn.v_proj"), 256, 64);
            push_nvfp4(&mut entries, &format!("{p}.self_attn.o_proj"), 64, 512);
            entries.push((format!("{p}.self_attn.q_norm.weight"), "BF16", vec![256]));
            entries.push((format!("{p}.self_attn.k_norm.weight"), "BF16", vec![256]));
        } else {
            push_nvfp4(
                &mut entries,
                &format!("{p}.linear_attn.in_proj_qkv"),
                128,
                64,
            );
            push_nvfp4(&mut entries, &format!("{p}.linear_attn.in_proj_z"), 64, 64);
            push_nvfp4(&mut entries, &format!("{p}.linear_attn.out_proj"), 64, 64);
            entries.push((
                format!("{p}.linear_attn.in_proj_a.weight"),
                "BF16",
                vec![4, 64],
            ));
            entries.push((
                format!("{p}.linear_attn.in_proj_b.weight"),
                "BF16",
                vec![4, 64],
            ));
            entries.push((
                format!("{p}.linear_attn.conv1d.weight"),
                "BF16",
                vec![128, 1, 4],
            ));
            entries.push((format!("{p}.linear_attn.norm.weight"), "BF16", vec![16]));
            entries.push((format!("{p}.linear_attn.A_log"), "BF16", vec![4]));
            entries.push((format!("{p}.linear_attn.dt_bias"), "BF16", vec![4]));
        }
    }

    let dir = std::env::temp_dir().join(format!("ff-nvfp4-contract-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let shard = dir.join("model.safetensors");
    let refs = entries
        .iter()
        .map(|(n, d, s)| (n.as_str(), *d, s.as_slice()))
        .collect::<Vec<_>>();
    write_safetensors_file(&shard, &refs);

    let mapped = MappedCheckpoint::open(&dir).unwrap();
    let manifest = mapped.manifest(&tiny_spec, LoadPolicy::TextOnly).unwrap();
    let plan0 = LayerDequantPlan::from_model(&manifest, 0, &tiny_spec).unwrap();
    let plan1 = LayerDequantPlan::from_model(&manifest, 1, &tiny_spec).unwrap();
    assert_eq!(plan0.slots.len(), 6);
    assert_eq!(plan1.slots.len(), 7);
    assert!(plan0.slots.values().all(|s| s.offset_bytes % 256 == 0));
    assert!(plan1.slots.values().all(|s| s.offset_bytes % 256 == 0));

    assert!(LayerDequantPlan::from_manifest(&manifest, 0, LinearFormat::Bf16).is_err());

    let mut missing_ws2 = manifest.clone();
    missing_ws2
        .entries
        .remove("model.language_model.layers.0.mlp.down_proj.weight_scale_2")
        .unwrap();
    assert!(LayerDequantPlan::from_model(&missing_ws2, 0, &tiny_spec).is_err());

    let mut missing_in_scale = manifest.clone();
    missing_in_scale
        .entries
        .remove("model.language_model.layers.0.mlp.down_proj.input_scale")
        .unwrap();
    assert!(LayerDequantPlan::from_model(&missing_in_scale, 0, &tiny_spec).is_err());

    let mut bad_group = manifest.clone();
    bad_group
        .entries
        .get_mut("model.language_model.layers.0.mlp.down_proj.weight_scale")
        .unwrap()
        .shape = vec![64, 4];
    assert!(LayerDequantPlan::from_model(&bad_group, 0, &tiny_spec).is_err());

    let dropped_ws2 = entries
        .iter()
        .filter(|(n, _, _)| n != "lm_head.weight_scale_2")
        .map(|(n, d, s)| (n.as_str(), *d, s.as_slice()))
        .collect::<Vec<_>>();
    write_safetensors_file(&shard, &dropped_ws2);
    let bad_mapped = MappedCheckpoint::open(&dir).unwrap();
    assert!(
        bad_mapped
            .manifest(&tiny_spec, LoadPolicy::TextOnly)
            .is_err()
    );

    std::fs::write(dir.join("config.json"), serde_json::to_vec(&root).unwrap()).unwrap();
    std::fs::write(
        dir.join("generation_config.json"),
        serde_json::to_vec(&json!({"eos_token_id": [248044, 248046]})).unwrap(),
    )
    .unwrap();
    assert!(flyingfish::serve::image::Processor::load(&dir, None).is_err());
    std::fs::write(
        dir.join("processor_config.json"),
        serde_json::to_vec(&json!({"processor_class": "Qwen3VLProcessor"})).unwrap(),
    )
    .unwrap();
    assert!(flyingfish::serve::image::Processor::load(&dir, None).is_err());
    std::fs::write(
        dir.join("processor_config.json"),
        serde_json::to_vec(&json!({
            "processor_class": "Qwen3VLProcessor",
            "image_processor": {
                "size": {"shortest_edge": 65536, "longest_edge": 16777216},
                "patch_size": 16,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5]
            }
        }))
        .unwrap(),
    )
    .unwrap();
    let proc = flyingfish::serve::image::Processor::load(&dir, None)
        .unwrap()
        .unwrap();
    assert_eq!(proc.merge(), 2);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[ignore = "CPU-only checkpoint header audit; set FF_H3_IMAGE_PE_MODEL"]
fn nvfp4_published_checkpoint_census() -> anyhow::Result<()> {
    use anyhow::Context;
    use flyingfish::{
        checkpoint::{LoadPolicy, MappedCheckpoint, TensorScope, has_mtp_component},
        linear_plan::LayerDequantPlan,
    };

    let dir = std::path::PathBuf::from(
        std::env::var_os("FF_H3_IMAGE_PE_MODEL").context("FF_H3_IMAGE_PE_MODEL missing")?,
    );
    assert!(!has_mtp_component(&dir)?);
    let spec = ModelSpec::read(&dir)?;
    assert_eq!(spec.linear_format, LinearFormat::NvFp4);
    assert_eq!(spec.mtp_layers, 0);
    assert_eq!(spec.eos, vec![248046]);
    assert!(spec.vision.is_some());
    let processor = flyingfish::serve::image::Processor::load(&dir, None)?
        .context("expected Qwen3.5 image processor from processor_config.json")?;
    assert_eq!(processor.merge(), 2);

    let mapped = MappedCheckpoint::open(&dir)?;
    for (policy, resident_bytes, text_loaded, vision_loaded) in [
        (LoadPolicy::TextOnly, 16_994_355_336usize, 2054usize, 0usize),
        (LoadPolicy::VisionOnly, 921_460_192, 0, 333),
        (LoadPolicy::TextWithVision, 17_915_815_528, 2054, 333),
    ] {
        let m = mapped.manifest(&spec, policy)?;
        assert_eq!(m.entries.len(), 2387);
        assert_eq!(m.resident_bytes, resident_bytes, "{policy:?}");
        assert_eq!(
            m.entries
                .values()
                .filter(|e| e.load && e.scope == TensorScope::Text)
                .count(),
            text_loaded,
            "{policy:?} text"
        );
        assert_eq!(
            m.entries
                .values()
                .filter(|e| e.load && e.scope == TensorScope::Vision)
                .count(),
            vision_loaded,
            "{policy:?} vision"
        );
    }
    assert!(mapped.manifest(&spec, LoadPolicy::TextWithMtp).is_err());

    let text_manifest = mapped.manifest(&spec, LoadPolicy::TextOnly)?;
    let mut max_arena = 0u64;
    for layer in 0..64u32 {
        let plan = LayerDequantPlan::from_model(&text_manifest, layer, &spec)?;
        if (layer + 1).is_multiple_of(4) {
            assert_eq!(plan.slots.len(), 7, "layer {layer}");
            assert_eq!(plan.bytes, 744_488_960, "layer {layer}");
        } else {
            assert_eq!(plan.slots.len(), 6, "layer {layer}");
            assert_eq!(plan.bytes, 765_460_480, "layer {layer}");
        }
        max_arena = max_arena.max(plan.bytes);
    }
    assert_eq!(max_arena, 765_460_480);
    println!(
        "PASS nvfp4_published_checkpoint_census entries=2387 text=2054/16994355336B vision=333/921460192B max_layer_arena={max_arena}"
    );
    Ok(())
}
