//! End-to-end dsv41 runs against a synthetic checkpoint laid out exactly like
//! the released DeepSeek-V4.1-Flash export (FP8 body + FP4 experts + BF16
//! norm/gate), small enough to load on any host.

#![recursion_limit = "512"]

use candle_core::Device;
use safetensors::Dtype as SdDtype;
use safetensors::tensor::TensorView;
use std::{fs, path::Path, process::Command};

/// Leaked fixture buffers: the safetensors views borrow them for the
/// lifetime of the process, which outlives the single shard write.
type Leaked = &'static [u8];

fn leak(bytes: Vec<u8>) -> Leaked {
    Box::leak(bytes.into_boxed_slice())
}

fn ff() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ff"))
}

const HIDDEN: usize = 64;
const HEADS: usize = 2;
const HEAD_DIM: usize = 32;
const QLORA: usize = 32;
const O_LORA: usize = 32;
const O_GROUPS: usize = 2;
const INTER: usize = 32;
const VOCAB: usize = 48;
const ROPE: usize = 8;
const SLIDING_WINDOW: usize = 4;

fn e4m3_bytes(count: usize) -> Vec<u8> {
    // Small positive normal-range patterns keep the dequantized weights
    // non-degenerate without needing an f32->e4m3 encoder here.
    (0..count).map(|index| 0x30 | ((index % 6) as u8)).collect()
}

fn bf16_bytes(values: Vec<f32>) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn f32_bytes(values: Vec<f32>) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}

fn fp4_triple(
    views: &mut Vec<(String, TensorView<'static>)>,
    projection: &str,
    rows: usize,
    columns: usize,
) {
    let packed: Vec<u8> = (0..rows * columns / 2)
        .map(|index| ((index % 31) as u8) | 0x10)
        .collect();
    let scale = vec![127u8; rows * columns.div_ceil(32)];
    views.push((
        format!("{projection}.weight"),
        TensorView::new(SdDtype::I8, vec![rows, columns / 2], leak(packed)).unwrap(),
    ));
    views.push((
        format!("{projection}.scale"),
        TensorView::new(
            SdDtype::F8_E8M0,
            vec![rows, columns.div_ceil(32)],
            leak(scale),
        )
        .unwrap(),
    ));
}

fn bf16_values(rows: usize, columns: usize) -> Vec<f32> {
    (0..rows * columns)
        .map(|index| ((index % 17) as f32 - 8.0) / 64.0)
        .collect()
}

fn f32_values(rows: usize, columns: usize) -> Vec<f32> {
    (0..rows * columns)
        .map(|index| ((index % 13) as f32 - 6.0) / 32.0)
        .collect()
}

fn write_fixture(root: &Path) {
    let model = root.join("DeepSeek-V4.1-mini");
    fs::create_dir_all(&model).unwrap();
    let mut views: Vec<(String, TensorView<'static>)> = Vec::new();
    let push = |views: &mut Vec<(String, TensorView<'static>)>,
                name: String,
                dtype: SdDtype,
                shape: Vec<usize>,
                bytes: Vec<u8>| {
        let view = TensorView::new(dtype, shape, leak(bytes)).unwrap();
        views.push((name, view));
    };

    push(
        &mut views,
        "embed.weight".to_owned(),
        SdDtype::BF16,
        vec![VOCAB, HIDDEN],
        bf16_bytes(bf16_values(VOCAB, HIDDEN)),
    );
    push(
        &mut views,
        "norm.weight".to_owned(),
        SdDtype::BF16,
        vec![HIDDEN],
        bf16_bytes(vec![1.0; HIDDEN]),
    );
    push(
        &mut views,
        "head.weight".to_owned(),
        SdDtype::BF16,
        vec![VOCAB, HIDDEN],
        bf16_bytes(bf16_values(VOCAB, HIDDEN)),
    );

    let layer_plans = [(0usize, 0u8, false, false), (1, 2, true, true)];
    for (layer, ratio, kv_source, index_source) in layer_plans {
        let attn = format!("layers.{layer}.attn.");
        push(
            &mut views,
            format!("{attn}attn_sink"),
            SdDtype::F32,
            vec![HEADS],
            f32_bytes(f32_values(HEADS, 1)),
        );
        let fp8_push = |views: &mut Vec<(String, TensorView<'static>)>,
                        projection: String,
                        rows: usize,
                        columns: usize| {
            views.push((
                format!("{projection}.weight"),
                TensorView::new(
                    SdDtype::F8_E4M3,
                    vec![rows, columns],
                    leak(e4m3_bytes(rows * columns)),
                )
                .unwrap(),
            ));
            views.push((
                format!("{projection}.scale"),
                TensorView::new(
                    SdDtype::F8_E8M0,
                    vec![rows.div_ceil(32), columns.div_ceil(32)],
                    leak(vec![130u8; rows.div_ceil(32) * columns.div_ceil(32)]),
                )
                .unwrap(),
            ));
        };
        fp8_push(&mut views, format!("{attn}wq_a"), QLORA, HIDDEN);
        push(
            &mut views,
            format!("{attn}q_norm.weight"),
            SdDtype::BF16,
            vec![QLORA],
            bf16_bytes(vec![1.0; QLORA]),
        );
        fp8_push(&mut views, format!("{attn}wq_b"), HEADS * HEAD_DIM, QLORA);
        fp8_push(&mut views, format!("{attn}wkv"), HEAD_DIM, HIDDEN);
        push(
            &mut views,
            format!("{attn}kv_norm.weight"),
            SdDtype::BF16,
            vec![HEAD_DIM],
            bf16_bytes(vec![1.0; HEAD_DIM]),
        );
        fp8_push(
            &mut views,
            format!("{attn}wo_a"),
            O_GROUPS * O_LORA,
            HEADS * HEAD_DIM / O_GROUPS,
        );
        fp8_push(&mut views, format!("{attn}wo_b"), HIDDEN, O_GROUPS * O_LORA);
        if kv_source {
            let prefix = format!("{attn}compressor.");
            push(
                &mut views,
                format!("{prefix}norm.weight"),
                SdDtype::BF16,
                vec![HEAD_DIM],
                bf16_bytes(vec![1.0; HEAD_DIM]),
            );
            push(
                &mut views,
                format!("{prefix}wkv.weight"),
                SdDtype::BF16,
                vec![HEAD_DIM, HIDDEN],
                bf16_bytes(bf16_values(HEAD_DIM, HIDDEN)),
            );
            if ratio > 1 {
                push(
                    &mut views,
                    format!("{prefix}wgate.weight"),
                    SdDtype::BF16,
                    vec![HEAD_DIM, HIDDEN],
                    bf16_bytes(bf16_values(HEAD_DIM, HIDDEN)),
                );
            }
        }
        if index_source {
            let prefix = format!("{attn}indexer.");
            fp8_push(&mut views, format!("{prefix}wq_b"), 2 * 32, QLORA);
            if kv_source {
                push(
                    &mut views,
                    format!("{prefix}wk.weight"),
                    SdDtype::BF16,
                    vec![32, HEAD_DIM],
                    bf16_bytes(bf16_values(32, HEAD_DIM)),
                );
                push(
                    &mut views,
                    format!("{prefix}k_norm.weight"),
                    SdDtype::BF16,
                    vec![32],
                    bf16_bytes(vec![1.0; 32]),
                );
            }
            push(
                &mut views,
                format!("{prefix}weights_proj.weight"),
                SdDtype::BF16,
                vec![2, HIDDEN],
                bf16_bytes(bf16_values(2, HIDDEN)),
            );
        }
        push(
            &mut views,
            format!("layers.{layer}.attn_norm.weight"),
            SdDtype::BF16,
            vec![HIDDEN],
            bf16_bytes(vec![1.0; HIDDEN]),
        );
        let ffn = format!("layers.{layer}.ffn.");
        push(
            &mut views,
            format!("{ffn}gate.weight"),
            SdDtype::BF16,
            vec![4, HIDDEN],
            bf16_bytes(bf16_values(4, HIDDEN)),
        );
        push(
            &mut views,
            format!("{ffn}gate.bias"),
            SdDtype::F32,
            vec![4],
            f32_bytes(f32_values(4, 1)),
        );
        for expert in 0..4 {
            for projection in ["w1", "w2", "w3"] {
                let (rows, columns) = match projection {
                    "w2" => (HIDDEN, INTER),
                    _ => (INTER, HIDDEN),
                };
                fp4_triple(
                    &mut views,
                    &format!("{ffn}experts.{expert}.{projection}"),
                    rows,
                    columns,
                );
            }
        }
        for projection in ["w1", "w2", "w3"] {
            let (rows, columns) = match projection {
                "w2" => (HIDDEN, INTER),
                _ => (INTER, HIDDEN),
            };
            fp4_triple(
                &mut views,
                &format!("{ffn}shared_experts.{projection}"),
                rows,
                columns,
            );
        }
        push(
            &mut views,
            format!("layers.{layer}.ffn_norm.weight"),
            SdDtype::BF16,
            vec![HIDDEN],
            bf16_bytes(vec![1.0; HIDDEN]),
        );
        let mix = (2 + 4) * 4;
        for name in ["hc_attn", "hc_ffn"] {
            push(
                &mut views,
                format!("layers.{layer}.{name}_fn"),
                SdDtype::F32,
                vec![mix, 4 * HIDDEN],
                f32_bytes(f32_values(mix, 4 * HIDDEN)),
            );
            push(
                &mut views,
                format!("layers.{layer}.{name}_scale"),
                SdDtype::F32,
                vec![3],
                f32_bytes(vec![0.3, 0.4, 0.2]),
            );
            push(
                &mut views,
                format!("layers.{layer}.{name}_base"),
                SdDtype::F32,
                vec![mix],
                f32_bytes(f32_values(mix, 1)),
            );
        }
    }

    let shard = "model-00001-of-00001.safetensors";
    safetensors::serialize_to_file(
        views.iter().map(|(name, view)| (name.as_str(), view)),
        None,
        &model.join(shard),
    )
    .unwrap();
    let weight_map: std::collections::BTreeMap<String, String> = views
        .iter()
        .map(|(name, _)| (name.clone(), shard.to_owned()))
        .collect();
    fs::write(
        model.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "metadata": {"total_size": 0},
            "weight_map": weight_map
        }))
        .unwrap(),
    )
    .unwrap();
    fs::write(
        model.join("config.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "architectures": ["DeepseekV41ForCausalLM"],
            "model_type": "deepseek_v41",
            "image_token_id": 47,
            "quantization_config": {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [32, 32],
                "scale_fmt": "ue8m0",
                "expert_dtype": "fp4"
            },
            "text_config": {
                "vocab_size": VOCAB,
                "hidden_size": HIDDEN,
                "moe_intermediate_size": INTER,
                "num_hidden_layers": 2,
                "num_attention_heads": HEADS,
                "num_key_value_heads": 1,
                "head_dim": HEAD_DIM,
                "qk_rope_head_dim": ROPE,
                "q_lora_rank": QLORA,
                "o_lora_rank": O_LORA,
                "o_groups": O_GROUPS,
                "attention_bias": false,
                "attention_dropout": 0.0,
                "hidden_act": "silu",
                "initializer_range": 0.02,
                "use_cache": true,
                "tie_word_embeddings": false,
                "swiglu_limit": 10.0,
                "rms_norm_eps": 1e-20,
                "max_position_embeddings": 128,
                "rope_theta": 10000.0,
                "rope_scaling": {
                    "rope_type": "yarn",
                    "factor": 2.0,
                    "beta_fast": 32,
                    "beta_slow": 1,
                    "original_max_position_embeddings": 64
                },
                "n_routed_experts": 4,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "scoring_func": "sqrtsoftplus",
                "topk_method": "noaux_tc",
                "norm_topk_prob": true,
                "routed_scaling_factor": 1.5,
                "sliding_window": SLIDING_WINDOW,
                "compress_ratios": [0, 2],
                "compress_rope_theta": 160000.0,
                "kv_source_layer_ids": [1],
                "index_source_layer_ids": [1],
                "index_n_heads": 2,
                "index_head_dim": 32,
                "index_topk": 2,
                "candidate_source_layer_id": 1,
                "candidate_topk_blocks": 4,
                "candidate_block_size": 4,
                "hc_mult": 4,
                "hc_sinkhorn_iters": 4,
                "hc_eps": 1e-6,
                "engram_layer_ids": [],
                "engram_num_embeddings": [],
                "engram_max_ngram_size": 4,
                "engram_vocab_size": 1000,
                "engram_n_heads": 8,
                "engram_head_dim": 256,
                "engram_pad_token_id": 2,
                "engram_compressed_vocab_size": 40,
                "num_nextn_predict_layers": 0
            }
        }))
        .unwrap(),
    )
    .unwrap();
    use tokenizers::models::wordlevel::WordLevel;
    let vocabulary: Vec<(String, u32)> = (0..VOCAB as u32)
        .map(|id| (format!("tok{id}"), id))
        .collect();
    let mut built = tokenizers::Tokenizer::new(
        WordLevel::builder()
            .vocab(vocabulary.into_iter().collect())
            .unk_token("tok0".to_owned())
            .build()
            .unwrap(),
    );
    built.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
    built
        .save(model.join("tokenizer.json"), true)
        .map_err(|error| anyhow::anyhow!("{error}"))
        .unwrap();
}

#[test]
fn dsv41_generate_decodes_from_a_synthetic_checkpoint() {
    let root = tempfile::tempdir().unwrap();
    write_fixture(root.path());
    let model = root.path().join("DeepSeek-V4.1-mini");
    let output = ff()
        .args(["text", "generate", "--adapter", "dsv41"])
        .arg("--model")
        .arg(&model)
        .args([
            "--prompt",
            "tok1 tok2 tok3",
            "--device",
            "cpu",
            "--max-new-tokens",
            "4",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "generate failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(!stdout.trim().is_empty());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("generated 4 tokens"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn dsv41_capture_parity_publishes_layer_snapshots_outside_the_model() {
    let root = tempfile::tempdir().unwrap();
    write_fixture(root.path());
    let model = root.path().join("DeepSeek-V4.1-mini");
    let capture = root.path().join("parity.safetensors");
    let output = ff()
        .args(["text", "capture-parity", "--adapter", "dsv41"])
        .arg("--model")
        .arg(&model)
        .args(["--prompt", "tok1 tok2", "--output"])
        .arg(&capture)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "capture failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(capture.is_file());
    let loaded = candle_core::safetensors::load(&capture, &Device::Cpu).unwrap();
    assert_eq!(loaded["layer_00"].dims()[0], 1);
    assert_eq!(loaded["layer_00"].dims().len(), 4);
    assert!(loaded.contains_key("layer_01"));
    assert!(loaded.contains_key("prompt_tokens"));
    assert!(!model.join("parity.safetensors").exists());
}
