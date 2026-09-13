use candle_core::{Device, Tensor};
use std::{collections::HashMap, path::Path, process::Command};
use tokenizers::{Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

fn fixture(model: &Path) {
    std::fs::create_dir_all(model).unwrap();
    let config = serde_json::json!({"architectures":["LlamaForCausalLM"],"model_type":"llama",
        "hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":2,
        "num_key_value_heads":1,"head_dim":2,"vocab_size":8,"max_position_embeddings":32,
        "rms_norm_eps":1e-6,"rope_theta":10000.0,"eos_token_id":[7],"bos_token_id":0,"hidden_act":"silu"});
    std::fs::write(
        model.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let mut tensors = HashMap::new();
    let mut index = 0;
    let mut add = |name: &str, shape: &[usize]| {
        let count = shape.iter().product();
        let values = (0..count)
            .map(|i| {
                if shape.len() == 1 {
                    1.
                } else {
                    ((i + index) as f32 * 0.7).sin() * 0.15
                }
            })
            .collect::<Vec<_>>();
        index += count;
        tensors.insert(
            name.to_owned(),
            Tensor::from_vec(values, shape, &Device::Cpu).unwrap(),
        );
    };
    add("model.embed_tokens.weight", &[8, 4]);
    add("lm_head.weight", &[8, 4]);
    add("model.norm.weight", &[4]);
    for name in ["input_layernorm", "post_attention_layernorm"] {
        add(&format!("model.layers.0.{name}.weight"), &[4]);
    }
    for (name, shape) in [
        ("self_attn.q_proj", [4, 4]),
        ("self_attn.k_proj", [2, 4]),
        ("self_attn.v_proj", [2, 4]),
        ("self_attn.o_proj", [4, 4]),
        ("mlp.gate_proj", [8, 4]),
        ("mlp.up_proj", [8, 4]),
        ("mlp.down_proj", [4, 8]),
    ] {
        add(&format!("model.layers.0.{name}.weight"), &shape);
    }
    candle_core::safetensors::save(&tensors, model.join("model.safetensors")).unwrap();
    let vocab = ["<s>", "<unk>", "a", "b", "c", "d", "e", "</s>", "invalid"]
        .into_iter()
        .enumerate()
        .map(|(i, s)| (s.to_owned(), i as u32))
        .collect();
    let mut tokenizer = Tokenizer::new(
        WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".into())
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer.save(model.join("tokenizer.json"), false).unwrap();
}

fn batch(model: &Path, requests: &Path, output: &Path, device: &str) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_ff"))
        .args(["text", "batch", "--model"])
        .arg(model)
        .arg("--requests")
        .arg(requests)
        .arg("--output-dir")
        .arg(output)
        .args(["--devices", device])
        .output()
        .unwrap()
}

fn json(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn batch_reuses_one_worker_without_cross_request_state_and_reserves_for_largest_prompt() {
    let directory = tempfile::tempdir().unwrap();
    let model = directory.path().join("model");
    fixture(&model);
    let requests = directory.path().join("queue.json");
    let queue = serde_json::json!([
        {"id":"first","prompt":"a b","raw":true,"max_new_tokens":4},
        {"id":"long","prompt":"a b c d a b c d a b c d a b c d","raw":true,"max_new_tokens":4},
        {"id":"repeat","prompt":"a b","raw":true,"max_new_tokens":4}]);
    std::fs::write(&requests, serde_json::to_vec(&queue).unwrap()).unwrap();
    let output = directory.path().join("batch");
    let result = batch(&model, &requests, &output, "cpu");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let first = json(&output.join("requests/first.json"));
    let repeat = json(&output.join("requests/repeat.json"));
    assert_eq!(first["token_ids"], repeat["token_ids"]);
    assert_eq!(first["text"], repeat["text"]);
    let fresh = Command::new(env!("CARGO_BIN_EXE_ff"))
        .args(["text", "generate", "--model"])
        .arg(&model)
        .args([
            "--device",
            "cpu",
            "--raw",
            "--prompt",
            "a b",
            "--max-new-tokens",
            "4",
        ])
        .output()
        .unwrap();
    assert!(fresh.status.success());
    assert_eq!(
        String::from_utf8(fresh.stdout).unwrap(),
        format!("{}\n", first["text"].as_str().unwrap())
    );
    let report = json(&output.join("report.json"));
    assert_eq!(report["requests"].as_array().unwrap().len(), 3);
    assert_eq!(report["skipped"], serde_json::json!([]));
    let mut last_finished = 0.;
    for request in report["requests"].as_array().unwrap() {
        let begin = request["started_offset_seconds"].as_f64().unwrap();
        let end = request["finished_offset_seconds"].as_f64().unwrap();
        assert!(begin >= last_finished && end >= begin);
        assert!((end - begin - request["elapsed_seconds"].as_f64().unwrap()).abs() < 1e-6);
        last_finished = end;
    }
    let reserve = report["requests"][0]["required_tensor_bytes"]
        .as_u64()
        .unwrap();
    assert!(
        report["requests"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["required_tensor_bytes"] == reserve)
    );
    let short = directory.path().join("short.json");
    std::fs::write(&short, serde_json::to_vec(&vec![queue[0].clone()]).unwrap()).unwrap();
    let short_output = directory.path().join("short");
    assert!(batch(&model, &short, &short_output, "cpu").status.success());
    assert!(
        reserve
            > json(&short_output.join("report.json"))["requests"][0]["required_tensor_bytes"]
                .as_u64()
                .unwrap()
    );
    let before = std::fs::read(output.join("report.json")).unwrap();
    assert!(!batch(&model, &requests, &output, "cpu").status.success());
    assert_eq!(std::fs::read(output.join("report.json")).unwrap(), before);
}

#[test]
fn failed_request_cancels_pending_work_and_startup_failure_is_reported() {
    let directory = tempfile::tempdir().unwrap();
    let model = directory.path().join("model");
    fixture(&model);
    let requests = directory.path().join("queue.json");
    std::fs::write(&requests,r#"[{"id":"ok","prompt":"a b","raw":true,"max_new_tokens":2},{"id":"bad","prompt":"invalid","raw":true,"max_new_tokens":2},{"id":"skip","prompt":"a","raw":true,"max_new_tokens":2}]"#).unwrap();
    let output = directory.path().join("failed");
    assert!(!batch(&model, &requests, &output, "cpu").status.success());
    let report = json(&output.join("report.json"));
    assert_eq!(report["requests"].as_array().unwrap().len(), 2);
    assert!(report["requests"][0]["error"].is_null());
    assert!(report["requests"][1]["error"].is_string());
    assert_eq!(report["skipped"], serde_json::json!(["skip"]));
    assert!(!output.join("requests/bad.json").exists());
    assert!(!output.join("requests/skip.json").exists());
    let output = directory.path().join("startup");
    assert!(
        !batch(&model, &requests, &output, "cuda:999999")
            .status
            .success()
    );
    let report = json(&output.join("report.json"));
    assert_eq!(report["worker_errors"].as_array().unwrap().len(), 1);
    assert_eq!(report["requests"], serde_json::json!([]));
    assert_eq!(report["skipped"].as_array().unwrap().len(), 3);
}
