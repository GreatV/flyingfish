use flyingfish::{
    backend::Operator as O,
    checkpoint::{ExpectedTensor, MappedCheckpoint, TensorScope},
    model::{
        HcPrePolicy, HcPreSource, HcStage, HcStagePlan, HcSublayer, LayerProgram, LayerStage,
        hc_pre_policy, hc_stage_plan, layer_program,
    },
    model_spec::{
        Architecture, ExpertFormat, LinearFormat, Mixer, ModelSpec, RoutedWeightBoundary,
        RouterScore, SwigluRounding,
    },
    work_layout::RowWorkSpec,
};
use safetensors::Dtype;
use serde_json::{Value, json};
use std::collections::BTreeMap;

fn root() -> Value {
    serde_json::from_str(include_str!("data/deepseek41-flash-config.json")).unwrap()
}

fn spec() -> ModelSpec {
    ModelSpec::deepseek_v41(&root(), &json!({})).unwrap()
}

#[test]
fn deepseek41_normalizes_geometry_and_quant_formats() {
    let spec = spec();
    assert_eq!(spec.architecture, Architecture::DeepseekV41);
    assert_eq!(
        spec.linear_format,
        LinearFormat::fp8_block([32, 32], Dtype::F8_E8M0).unwrap()
    );
    assert_eq!(
        spec.expert_format,
        Some(ExpertFormat::Fp4E2m1Block {
            block: [1, 32],
            scale_dtype: Dtype::F8_E8M0,
        })
    );
    assert_eq!(spec.layers.len(), 40);
    assert!(
        spec.layers
            .iter()
            .all(|layer| layer.mixer == Mixer::FullAttention && layer.rope)
    );
    assert!(
        spec.layers
            .iter()
            .all(|layer| layer.sliding_window == Some(128))
    );
    assert_eq!(spec.hidden, 5120);
    assert_eq!(spec.vocab, 129280);
    assert_eq!((spec.q_heads, spec.kv_heads), (64, 1));
    assert_eq!(spec.head_dim, 512);
    assert_eq!(spec.rotary_dim, 64);
    assert_eq!(spec.q_width, 32768);
    let ds = spec.deepseek_v41.as_ref().unwrap();
    assert_eq!(ds.num_hidden_layers, 40);
    assert_eq!((ds.hc_mult, ds.hc_sinkhorn_iters), (4, 20));
    assert_eq!(ds.hc_eps, 1e-6);
    assert_eq!(ds.window_size, 128);
    assert_eq!(ds.qk_rope_head_dim, 64);
    assert_eq!(
        (ds.q_lora_rank, ds.o_lora_rank, ds.o_groups),
        (1280, 1024, 8)
    );
    assert_eq!(ds.swiglu_limit, 10.0);
    assert_eq!(ds.routed_scaling_factor, 1.5);
    assert_eq!((ds.index_heads, ds.index_head_dim), (32, 128));
    assert_eq!(ds.kv_source_layer_ids, vec![2u32, 8, 14, 20]);
    assert_eq!(
        ds.index_source_layer_ids,
        vec![2u32, 8, 14, 20, 24, 28, 32, 36]
    );
    assert_eq!(ds.candidate_source_layer_id, 20);
    assert_eq!(ds.engram_layer_ids, vec![1u32, 14]);
    assert_eq!(ds.engram_num_embeddings, vec![384_006_168u32, 384_016_682]);
    assert_eq!(ds.nextn_layers, 3);
    assert_eq!(ds.dspark_experts, 128);
    assert_eq!(ds.dspark_markov_rank, 256);
    assert_eq!((ds.vision_layers, ds.vision_hidden), (32, 1024));
    assert_eq!((ds.vision_intermediate, ds.vision_patch), (2816, 14));
    assert_eq!(ds.vision_downsample_ratio, 3);
    let moe = spec.moe.as_ref().unwrap();
    assert_eq!((moe.experts, moe.experts_per_tok), (384, 6));
    assert_eq!(
        (moe.expert_intermediate, moe.shared_intermediate),
        (2304, 2304)
    );
    assert_eq!(moe.score, RouterScore::SqrtSoftplus);
    assert!(!moe.shared_gate);
    assert_eq!(spec.swiglu_rounding(), SwigluRounding::ClampedTwice);
    assert_eq!(spec.swiglu_limit(), 10.0);
    assert_eq!(
        spec.routed_weight_boundary(),
        RoutedWeightBoundary::BeforeDownQuantization
    );
    assert_eq!(spec.eos, vec![1u32]);
}

#[test]
fn deepseek41_rejects_unsupported_semantics() {
    for (pointer, value) in [
        ("/quantization_config/weight_block_size", json!([64, 32])),
        ("/quantization_config/scale_fmt", json!("e8m0")),
        ("/quantization_config/expert_dtype", json!("fp8")),
        ("/text_config/hc_mult", json!(0)),
        ("/text_config/hc_sinkhorn_iters", json!(0)),
        ("/text_config/hc_eps", json!(0.0)),
        ("/text_config/scoring_func", json!("softmax")),
        ("/text_config/topk_method", json!("greedy")),
        ("/text_config/sliding_window", json!(0)),
        ("/text_config/rope_scaling/factor", json!(8)),
        ("/text_config/n_routed_experts", json!(0)),
    ] {
        let mut bad = root();
        *bad.pointer_mut(pointer).unwrap() = value;
        assert!(
            ModelSpec::deepseek_v41(&bad, &json!({})).is_err(),
            "{pointer}"
        );
    }
    let mut bad = root();
    bad["text_config"]["unknown"] = Value::Null;
    assert!(ModelSpec::deepseek_v41(&bad, &json!({})).is_err());
    let mut bad = root();
    bad["quantization_config"]["unknown"] = Value::Null;
    assert!(ModelSpec::deepseek_v41(&bad, &json!({})).is_err());
}

#[test]
fn deepseek41_layer_program_is_the_frozen_hc_form() {
    let spec = spec();
    let LayerProgram::Hc(stages) = layer_program(&spec, 0).unwrap() else {
        panic!("DeepSeek-V4.1 layers use the hyper-connection program");
    };
    assert!(matches!(stages[0], LayerStage::Hc(HcStage::Pre)));
    assert!(matches!(stages[1], LayerStage::Common(O::Norm)));
    assert!(matches!(stages[2], LayerStage::Mixer(Mixer::FullAttention)));
    assert!(matches!(stages[3], LayerStage::Hc(HcStage::Post)));
    assert!(matches!(stages[4], LayerStage::Hc(HcStage::Pre)));
    assert!(matches!(stages[5], LayerStage::Common(O::PostNorm)));
    assert!(matches!(stages[6], LayerStage::Ffn));
    assert!(matches!(stages[7], LayerStage::Hc(HcStage::Post)));
    assert!(matches!(stages[8], LayerStage::Common(O::NextNorm)));
    assert!(layer_program(&spec, 40).is_err());
}

fn expected_entries() -> BTreeMap<String, ExpectedTensor> {
    let spec = spec();
    let mut expected = BTreeMap::new();
    MappedCheckpoint::deepseek_v41_entries(&spec, &mut expected).unwrap();
    expected
}

#[test]
fn deepseek41_manifest_covers_the_published_tensor_set() {
    let expected = expected_entries();
    assert_eq!(expected.len(), 96_085);
    for (scope, count) in [
        (TensorScope::Text, 93_421),
        (TensorScope::Mtp, 2_401),
        (TensorScope::Vision, 263),
    ] {
        assert_eq!(
            expected
                .values()
                .filter(|(_, _, tensor_scope)| *tensor_scope == scope)
                .count(),
            count,
            "{scope:?}"
        );
    }
    assert_eq!(
        expected
            .keys()
            .filter(|name| name.starts_with("layers.0."))
            .count(),
        2_334
    );
    for (name, dtype, shape, scope) in [
        (
            "embed.weight",
            Dtype::BF16,
            vec![129_280, 5120],
            TensorScope::Text,
        ),
        (
            "head.weight",
            Dtype::BF16,
            vec![129_280, 5120],
            TensorScope::Text,
        ),
        ("norm.weight", Dtype::BF16, vec![5120], TensorScope::Text),
        (
            "layers.0.hc_attn_fn",
            Dtype::F32,
            vec![24, 20_480],
            TensorScope::Text,
        ),
        (
            "layers.0.hc_attn_base",
            Dtype::F32,
            vec![24],
            TensorScope::Text,
        ),
        (
            "layers.0.attn.wkv.weight",
            Dtype::F8_E4M3,
            vec![512, 5120],
            TensorScope::Text,
        ),
        (
            "layers.0.attn.wkv.scale",
            Dtype::F8_E8M0,
            vec![16, 160],
            TensorScope::Text,
        ),
        (
            "layers.0.attn.wo_a.weight",
            Dtype::F8_E4M3,
            vec![8192, 4096],
            TensorScope::Text,
        ),
        (
            "layers.0.attn.wo_a.scale",
            Dtype::F8_E8M0,
            vec![256, 128],
            TensorScope::Text,
        ),
        (
            "layers.0.attn.wq_b.weight",
            Dtype::F8_E4M3,
            vec![32768, 1280],
            TensorScope::Text,
        ),
        (
            "layers.0.ffn.gate.weight",
            Dtype::BF16,
            vec![384, 5120],
            TensorScope::Text,
        ),
        (
            "layers.0.ffn.gate.bias",
            Dtype::F32,
            vec![384],
            TensorScope::Text,
        ),
        (
            "layers.0.ffn.shared_experts.w1.weight",
            Dtype::F8_E4M3,
            vec![2304, 5120],
            TensorScope::Text,
        ),
        (
            "layers.0.ffn.experts.0.w1.weight",
            Dtype::I8,
            vec![2304, 2560],
            TensorScope::Text,
        ),
        (
            "layers.0.ffn.experts.0.w1.scale",
            Dtype::F8_E8M0,
            vec![2304, 160],
            TensorScope::Text,
        ),
        (
            "layers.2.attn.compressor.wgate.weight",
            Dtype::BF16,
            vec![512, 5120],
            TensorScope::Text,
        ),
        (
            "layers.2.attn.indexer.wq_b.weight",
            Dtype::F8_E4M3,
            vec![4096, 1280],
            TensorScope::Text,
        ),
        (
            "layers.24.attn.indexer.weights_proj.weight",
            Dtype::BF16,
            vec![32, 5120],
            TensorScope::Text,
        ),
        (
            "layers.1.engram.embed.weight",
            Dtype::F8_E4M3,
            vec![384_006_168, 256],
            TensorScope::Text,
        ),
        (
            "layers.1.engram.embed.scale",
            Dtype::F8_E8M0,
            vec![384_006_168, 8],
            TensorScope::Text,
        ),
        (
            "mtp.0.ffn.experts.0.w1.weight",
            Dtype::I8,
            vec![2304, 2560],
            TensorScope::Mtp,
        ),
        (
            "mtp.0.main_proj.weight",
            Dtype::F8_E4M3,
            vec![5120, 15_360],
            TensorScope::Mtp,
        ),
        (
            "mtp.2.confidence_head.proj.weight",
            Dtype::BF16,
            vec![1, 5376],
            TensorScope::Mtp,
        ),
        (
            "vision.blocks.0.attn.wqkv.weight",
            Dtype::BF16,
            vec![3072, 1024],
            TensorScope::Vision,
        ),
        (
            "vision.patch_embed.proj.weight",
            Dtype::BF16,
            vec![1024, 588],
            TensorScope::Vision,
        ),
        (
            "aligner.w1.weight",
            Dtype::BF16,
            vec![5120, 9216],
            TensorScope::Vision,
        ),
    ] {
        assert_eq!(expected.get(name), Some(&(dtype, shape, scope)), "{name}");
    }
    assert!(!expected.contains_key("layers.20.attn.compressor.wgate.weight"));
    assert!(!expected.contains_key("layers.1.attn.compressor.norm.weight"));
    assert!(expected.contains_key("layers.1.engram.wkv.weight"));
}

#[test]
fn deepseek41_hc_stage_plan_and_carry_chain_control() {
    let policy = hc_pre_policy(Architecture::DeepseekV41).unwrap();
    assert_eq!(
        policy,
        HcPrePolicy {
            attention: HcPreSource::Carried,
            ffn: HcPreSource::AttentionOwn,
        }
    );
    assert_eq!(
        hc_stage_plan(0, HcStage::Pre, policy).unwrap(),
        HcStagePlan::Pre {
            sublayer: HcSublayer::Attention,
            source: HcPreSource::Carried,
        }
    );
    assert_eq!(
        hc_stage_plan(3, HcStage::Post, policy).unwrap(),
        HcStagePlan::Post {
            sublayer: HcSublayer::Attention,
        }
    );
    assert_eq!(
        hc_stage_plan(4, HcStage::Pre, policy).unwrap(),
        HcStagePlan::Pre {
            sublayer: HcSublayer::Ffn,
            source: HcPreSource::AttentionOwn,
        }
    );
    assert_eq!(
        hc_stage_plan(7, HcStage::Post, policy).unwrap(),
        HcStagePlan::Post {
            sublayer: HcSublayer::Ffn,
        }
    );
    for stage in [1usize, 2, 5, 6, 8, 9] {
        for kind in [HcStage::Pre, HcStage::Post] {
            assert!(
                hc_stage_plan(stage, kind, policy)
                    .err()
                    .is_some_and(|error| error.to_string().contains(&stage.to_string())),
                "stage {stage} ({kind:?})"
            );
        }
    }
    for (stage, kind) in [
        (0, HcStage::Post),
        (3, HcStage::Pre),
        (4, HcStage::Post),
        (7, HcStage::Pre),
    ] {
        assert!(hc_stage_plan(stage, kind, policy).is_err());
    }
    // The carry chain as data: the pre-mix value each HcPre stage consumes.
    // The planted control swaps the FFN pre source to the carried pre-mix
    // (model.py:978-980), which must change the selected value.
    let carried = vec![1.0f32, 0.0, 0.0, 0.0];
    let attn_pre = vec![0.6f32, 0.4, 0.3, 0.2];
    let own_pre = vec![0.9f32, 0.8, 0.7, 0.6];
    let selected = |plan: HcStagePlan| match plan {
        HcStagePlan::Pre { source, .. } => match source {
            HcPreSource::Carried => carried.clone(),
            HcPreSource::AttentionOwn => attn_pre.clone(),
            HcPreSource::Own => own_pre.clone(),
        },
        HcStagePlan::Post { .. } => panic!("HcPost consumes no pre-mix"),
    };
    let real = hc_stage_plan(4, HcStage::Pre, policy).unwrap();
    let swapped = HcStagePlan::Pre {
        sublayer: HcSublayer::Ffn,
        source: HcPreSource::Carried,
    };
    assert_ne!(real, swapped);
    assert_ne!(
        selected(real),
        selected(swapped),
        "the carry-swap control did not change the ffn collapse pre-mix"
    );
    // The GLM {Own, Own} policy is the discriminating cross-model control.
    let glm = hc_pre_policy(Architecture::Glm5Next).unwrap();
    let glm_ffn = hc_stage_plan(4, HcStage::Pre, glm).unwrap();
    assert_ne!(
        selected(real),
        selected(glm_ffn),
        "the GLM own-source policy did not change the ffn collapse pre-mix"
    );
    assert_eq!(
        selected(hc_stage_plan(0, HcStage::Pre, policy).unwrap()),
        carried,
        "attention HcPre consumes the carried pre-mix"
    );
    assert_eq!(
        selected(hc_stage_plan(0, HcStage::Pre, glm).unwrap()),
        own_pre,
        "GLM attention HcPre consumes the sublayer's own pre-mix"
    );
}

#[test]
fn deepseek41_row_work_widths_match_the_config_geometry() {
    let spec = spec();
    let work = RowWorkSpec::new(&spec).unwrap();
    let moe = spec.moe.as_ref().unwrap();
    assert_eq!(work.hidden, 5120);
    // projection = wq_b output (q_width 32768) + wkv output (head_dim 512);
    // the buffer carries [rows, projection].
    assert_eq!(work.projection, 32_768 + 512);
    // mixer = hidden: the post-wo_b attention output; [rows, mixer].
    assert_eq!(work.mixer, 5120);
    // intermediate = max(routed, shared) expert width; gu carries [rows, 2*intermediate].
    assert_eq!(work.intermediate, 2304);
    assert_eq!(
        work.intermediate,
        moe.expert_intermediate.max(moe.shared_intermediate) as usize
    );
    assert!(!work.extra_mixer);
    assert_eq!(work.ab, 0);
    let expected = expected_entries();
    let out_width = |name: &str| expected.get(name).unwrap().1[0];
    assert_eq!(
        work.projection,
        out_width("layers.0.attn.wq_b.weight") + out_width("layers.0.attn.wkv.weight")
    );
    assert_eq!(work.mixer, out_width("layers.0.attn.wo_b.weight"));
    assert_eq!(
        work.intermediate,
        out_width("layers.0.ffn.experts.0.w1.weight")
            .max(out_width("layers.0.ffn.shared_experts.w1.weight"))
    );
}

#[test]
fn deepseek41_hc_weight_manifest_covers_all_backbone_layers() {
    let expected = expected_entries();
    // The runtime loads exactly the backbone layers.{0..40} hc set; the mtp.{0..2}
    // instances stay in the closed world but out of the load plan.
    for layer in 0..40 {
        for sublayer in ["attn", "ffn"] {
            for (suffix, shape) in [
                ("fn", vec![24, 20_480]),
                ("base", vec![24]),
                ("scale", vec![3]),
            ] {
                let name = format!("layers.{layer}.hc_{sublayer}_{suffix}");
                assert_eq!(
                    expected.get(name.as_str()),
                    Some(&(Dtype::F32, shape, TensorScope::Text)),
                    "{name}"
                );
            }
        }
    }
    let backbone = expected
        .keys()
        .filter(|name| name.starts_with("layers.") && name.contains("hc_"))
        .count();
    let mtp = expected
        .keys()
        .filter(|name| name.starts_with("mtp.") && name.contains("hc_"))
        .count();
    assert_eq!(backbone, 40 * 6);
    assert_eq!(mtp, 3 * 6);
}

#[test]
fn deepseek41_e2m1_expert_slot_layout_matches_the_published_grid() {
    let spec = spec();
    let geometry = flyingfish::experts::catalogue::ExpertGeometry::from_model(&spec).unwrap();
    assert_eq!(
        (
            geometry.layers,
            geometry.first_layer,
            geometry.experts,
            geometry.hidden,
            geometry.intermediate,
            geometry.top_k
        ),
        (40, 0, 384, 5120, 2304, 6)
    );
    let format = spec.expert_format.unwrap();
    let layout = flyingfish::experts::catalogue::ExpertLayout::new(format, &geometry).unwrap();
    // Codes [2304, 2560] / [5120, 1152] with [rows, cols/32] E8M0 scales:
    // 3 x 5,898,240 code bytes + 3 x 368,640 scale bytes.
    assert_eq!(layout.slot_bytes(), 18_800_640);
    let slot = match &layout {
        flyingfish::experts::catalogue::ExpertLayout::Fp4E2m1Block(slot, _) => slot,
        other => panic!("E2M1 format produced {other:?}"),
    };
    assert_eq!((slot.gate.start, slot.gate.end), (0, 5_898_240));
    assert_eq!((slot.up.start, slot.up.end), (5_898_240, 11_796_480));
    assert_eq!((slot.down.start, slot.down.end), (11_796_480, 17_694_720));
    assert_eq!(
        (slot.gate_scale.start, slot.gate_scale.end),
        (17_694_720, 18_063_360)
    );
    assert_eq!(
        (slot.up_scale.start, slot.up_scale.end),
        (18_063_360, 18_432_000)
    );
    assert_eq!(
        (slot.down_scale.start, slot.down_scale.end),
        (18_432_000, 18_800_640)
    );
    assert!(layout.fp8_slot().is_none());
    for bad in [
        ExpertFormat::Fp4E2m1Block {
            block: [32, 32],
            scale_dtype: Dtype::F8_E8M0,
        },
        ExpertFormat::Fp4E2m1Block {
            block: [1, 32],
            scale_dtype: Dtype::BF16,
        },
    ] {
        let error = flyingfish::experts::catalogue::ExpertLayout::new(bad, &geometry)
            .unwrap_err()
            .to_string();
        assert!(error.contains("E8M0"), "{error}");
    }
    assert_eq!(layout.projection_bytes(), (12_533_760, 6_266_880));
    assert_eq!(layout.down_offset(), 11_796_480);
}
