//! Fixtures shared by the GLM adapter's module tests.
//!
//! These build tiny in-memory checkpoints so KDA, MLA, MoE routing and the
//! expert cache can be exercised without the 320B original.

use super::*;
use candle_core::DType;
use candle_core::Device;
use candle_core::Shape;
use candle_core::Tensor;
use candle_core::safetensors;
use ff_core::weights::CachePolicy;
use ff_core::weights::WeightSource;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;

pub(super) const VOCAB: usize = 8;

pub(super) const HIDDEN: usize = 4;

pub(super) const QKV: usize = 2;

pub(super) const TARGET_TOKEN: u32 = 3;

/// Make the generated fixture exercise actual FP8 expert and static loads.
/// The ordinary tiny checkpoint intentionally stores only BF16 linears.
pub(super) fn quantize_tiny_linears(directory: &Path) {
    let path = directory.join("model.safetensors");
    let mut tensors = safetensors::load(&path, &Device::Cpu).unwrap();
    let names: Vec<_> = tensors
        .keys()
        .filter(|name| {
            (name.contains(".mlp.")
                && ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
                    .iter()
                    .any(|suffix| name.ends_with(suffix)))
                || name.ends_with(".self_attn.q_a_proj.weight")
        })
        .cloned()
        .collect();
    assert!(names.iter().any(|name| name.contains(".experts.")));
    assert!(names.iter().any(|name| !name.contains(".experts.")));
    for name in names {
        let weight = tensors[&name].to_dtype(DType::F8E4M3).unwrap();
        let (rows, cols) = weight.dims2().unwrap();
        tensors.insert(
            format!("{name}_scale_inv"),
            Tensor::full(
                1f32 + 1. / 256.,
                (rows.div_ceil(128), cols.div_ceil(128)),
                &Device::Cpu,
            )
            .unwrap(),
        );
        tensors.insert(name, weight);
    }
    safetensors::save(&tensors, path).unwrap();
}

pub(super) fn f32_zeros(shape: impl Into<Shape>) -> Tensor {
    Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap()
}

pub(super) fn f32_ones(shape: impl Into<Shape>) -> Tensor {
    Tensor::ones(shape, DType::F32, &Device::Cpu).unwrap()
}

pub(super) fn patterned_bf16(shape: impl Into<Shape>, seed: usize) -> Tensor {
    let shape = shape.into();
    let values = (0..shape.elem_count())
        .map(|index| (((index * 5 + seed) % 11) as f32 - 5.0) / 32.0)
        .collect::<Vec<_>>();
    Tensor::from_vec(values, shape, &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
}

pub(super) fn insert_patterned(
    tensors: &mut HashMap<String, Tensor>,
    name: impl Into<String>,
    shape: impl Into<Shape>,
    seed: &mut usize,
) {
    tensors.insert(name.into(), patterned_bf16(shape, *seed));
    *seed += 1;
}

pub(super) fn insert_hyper_map(tensors: &mut HashMap<String, Tensor>, prefix: &str, site: &str) {
    let mix = (2 + 2) * 2;
    tensors.insert(
        format!("{prefix}.hc_{site}_fn"),
        f32_zeros((mix, 2 * HIDDEN)).to_dtype(DType::BF16).unwrap(),
    );
    tensors.insert(format!("{prefix}.hc_{site}_base"), f32_zeros(mix));
    tensors.insert(format!("{prefix}.hc_{site}_scale"), f32_ones(3));
}

pub(super) fn insert_mlp(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    width: usize,
    seed: &mut usize,
) {
    insert_patterned(
        tensors,
        format!("{prefix}.gate_proj.weight"),
        (width, HIDDEN),
        seed,
    );
    insert_patterned(
        tensors,
        format!("{prefix}.up_proj.weight"),
        (width, HIDDEN),
        seed,
    );
    insert_patterned(
        tensors,
        format!("{prefix}.down_proj.weight"),
        (HIDDEN, width),
        seed,
    );
}

pub(super) fn insert_kda(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    seed: &mut usize,
    qkv: usize,
) {
    let attention = format!("{prefix}.self_attn");
    for projection in ["q", "k", "v"] {
        insert_patterned(
            tensors,
            format!("{attention}.{projection}_proj.weight"),
            (qkv, HIDDEN),
            seed,
        );
        tensors.insert(
            format!("{attention}.{projection}_conv1d.weight"),
            patterned_bf16((qkv, 1, 2), *seed),
        );
        *seed += 1;
    }
    insert_patterned(
        tensors,
        format!("{attention}.f_a_proj.weight"),
        (2, HIDDEN),
        seed,
    );
    insert_patterned(
        tensors,
        format!("{attention}.f_b_proj.weight"),
        (qkv, 2),
        seed,
    );
    tensors.insert(format!("{attention}.dt_bias"), f32_zeros(qkv));
    tensors.insert(format!("{attention}.A_log"), f32_zeros(1));
    insert_patterned(
        tensors,
        format!("{attention}.b_proj.weight"),
        (1, HIDDEN),
        seed,
    );
    insert_patterned(
        tensors,
        format!("{attention}.g_a_proj.weight"),
        (2, HIDDEN),
        seed,
    );
    insert_patterned(
        tensors,
        format!("{attention}.g_b_proj.weight"),
        (qkv, 2),
        seed,
    );
    tensors.insert(format!("{attention}.o_norm.weight"), f32_ones(qkv));
    insert_patterned(
        tensors,
        format!("{attention}.o_proj.weight"),
        (HIDDEN, qkv),
        seed,
    );
}

pub(super) fn insert_mla(tensors: &mut HashMap<String, Tensor>, prefix: &str, seed: &mut usize) {
    let attention = format!("{prefix}.self_attn");
    insert_patterned(
        tensors,
        format!("{attention}.q_a_proj.weight"),
        (2, HIDDEN),
        seed,
    );
    tensors.insert(format!("{attention}.q_a_layernorm.weight"), f32_ones(2));
    insert_patterned(
        tensors,
        format!("{attention}.q_b_proj.weight"),
        (2, 2),
        seed,
    );
    insert_patterned(
        tensors,
        format!("{attention}.kv_a_proj_with_mqa.weight"),
        (2, HIDDEN),
        seed,
    );
    tensors.insert(format!("{attention}.kv_a_layernorm.weight"), f32_ones(2));
    insert_patterned(
        tensors,
        format!("{attention}.kv_b_proj.weight"),
        (4, 2),
        seed,
    );
    insert_patterned(
        tensors,
        format!("{attention}.o_proj.weight"),
        (HIDDEN, 2),
        seed,
    );
}

pub(super) fn write_config(directory: &Path, qkv: usize) {
    let config = r#"{
        "architectures":["Glm5NextForConditionalGeneration"],
        "model_type":"glm5_next",
        "text_config":{
            "model_type":"glm5_next_text","dtype":"bfloat16",
            "vocab_size":8,"hidden_size":4,
            "intermediate_size":8,"moe_intermediate_size":4,
            "num_hidden_layers":2,"num_attention_heads":1,"num_key_value_heads":1,
            "first_k_dense_replace":1,"n_shared_experts":1,"n_routed_experts":2,
            "num_experts_per_tok":2,"routed_scaling_factor":2.5,
            "n_group":1,"topk_group":1,"norm_topk_prob":true,
            "scoring_func":"sigmoid","topk_method":"noaux_tc",
            "moe_router_dtype":"float32","q_lora_rank":2,"kv_lora_rank":2,
            "qk_nope_head_dim":2,"qk_rope_head_dim":0,"qk_head_dim":2,
            "head_dim":0,"v_head_dim":2,"mhc":true,"mla_use_nope":true,
            "layer_types":["linear_attention","deepseek_sparse_attention"],
            "mlp_layer_types":["dense","sparse"],"indexer_types":["full","full"],
            "index_topk":8,"index_kpool":2,"index_kpool_always_select_tail":true,
            "index_n_heads":1,"index_head_dim":2,"linear_num_heads":1,
            "linear_head_dim":2,"linear_conv_kernel_dim":2,"linear_lower_bound":-5.0,
            "hc_mult":2,"hc_eps":0.000001,"hc_sinkhorn_iters":3,
            "hidden_act":"silu","swiglu_limit":10.0,"rms_norm_eps":0.00001,
            "attention_bias":false,"attention_dropout":0.0,"max_position_embeddings":64,
            "num_nextn_predict_layers":0,"use_cache":true,"tie_word_embeddings":false,
            "pad_token_id":0,"eos_token_id":[2]
        },
        "quantization_config":{
            "quant_method":"fp8","activation_scheme":"dynamic","fmt":"e4m3",
            "weight_block_size":[128,128]
        },
        "tie_word_embeddings":false
    }"#;
    let mut config: serde_json::Value = serde_json::from_str(config).unwrap();
    config["text_config"]["linear_head_dim"] = json!(qkv);
    fs::write(
        directory.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    fs::write(
        directory.join("generation_config.json"),
        serde_json::to_vec(&json!({
            "eos_token_id": [2],
            "pad_token_id": 0,
            "temperature": 1.0,
            "top_p": 1.0
        }))
        .unwrap(),
    )
    .unwrap();
}

pub(super) fn write_tokenizer(directory: &Path) {
    let vocab = [
        ("[UNK]", 0),
        ("<pad>", 1),
        ("<eos>", 2),
        ("winner", TARGET_TOKEN),
        ("four", 4),
        ("five", 5),
        ("six", 6),
        ("seven", 7),
    ]
    .into_iter()
    .map(|(token, id)| (token.to_owned(), id))
    .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".to_owned())
        .build()
        .unwrap();
    Tokenizer::new(model)
        .save(directory.join("tokenizer.json"), false)
        .unwrap();
}

pub(super) fn write_weights(directory: &Path, qkv: usize) {
    let mut tensors = HashMap::new();
    let mut embeddings = vec![0f32; VOCAB * HIDDEN];
    for row in 0..VOCAB {
        embeddings[row * HIDDEN] = 1.0;
    }
    tensors.insert(
        "model.language_model.embed_tokens.weight".to_owned(),
        Tensor::from_vec(embeddings, (VOCAB, HIDDEN), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap(),
    );
    tensors.insert(
        "model.language_model.norm.weight".to_owned(),
        f32_ones(HIDDEN),
    );
    let mut head = vec![0f32; VOCAB * HIDDEN];
    head[TARGET_TOKEN as usize * HIDDEN] = 16.0;
    tensors.insert(
        "lm_head.weight".to_owned(),
        Tensor::from_vec(head, (VOCAB, HIDDEN), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap(),
    );

    let mut seed = 1;
    for layer in 0..2 {
        let prefix = format!("model.language_model.layers.{layer}");
        insert_hyper_map(&mut tensors, &prefix, "attn");
        insert_hyper_map(&mut tensors, &prefix, "ffn");
        tensors.insert(format!("{prefix}.input_layernorm.weight"), f32_ones(HIDDEN));
        tensors.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            f32_ones(HIDDEN),
        );
        if layer == 0 {
            insert_kda(&mut tensors, &prefix, &mut seed, qkv);
            insert_mlp(&mut tensors, &format!("{prefix}.mlp"), 8, &mut seed);
        } else {
            insert_mla(&mut tensors, &prefix, &mut seed);
            let mlp = format!("{prefix}.mlp");
            tensors.insert(format!("{mlp}.gate.weight"), f32_zeros((2, HIDDEN)));
            tensors.insert(
                format!("{mlp}.gate.e_score_correction_bias"),
                Tensor::new(&[0.25f32, -0.25], &Device::Cpu).unwrap(),
            );
            insert_mlp(&mut tensors, &format!("{mlp}.shared_experts"), 4, &mut seed);
            for expert in 0..2 {
                insert_mlp(
                    &mut tensors,
                    &format!("{mlp}.experts.{expert}"),
                    4,
                    &mut seed,
                );
            }
        }
    }
    safetensors::save(&tensors, directory.join("model.safetensors")).unwrap();
}

pub(super) fn tiny_checkpoint() -> tempfile::TempDir {
    tiny_checkpoint_with_kda_width(QKV)
}

pub(super) fn tiny_checkpoint_with_kda_width(qkv: usize) -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    write_config(directory.path(), qkv);
    write_tokenizer(directory.path());
    write_weights(directory.path(), qkv);
    directory
}

pub(super) fn tiny_options() -> StreamedGlmOptions {
    StreamedGlmOptions::new(WeightSource::Mmap, CachePolicy::new(1), Device::Cpu)
        .with_resident_static(true)
        .with_expert_cache_bytes(1_024)
}
