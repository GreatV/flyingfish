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
        model::{LAYER, LLAMA_MIXER, LayerStage, layer_program},
        model_spec::ModelSpec,
    };
    let root: serde_json::Value =
        serde_json::from_str(include_str!("data/q38-config.json")).unwrap();
    let spec = ModelSpec::qwen(&root, &serde_json::Value::Null).unwrap();
    let full = layer_program(&spec, 3).unwrap();
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
    assert!(matches!(
        layer_program(&spec, 0).unwrap()[1],
        LayerStage::Mixer(Mixer::Gdn)
    ));
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
