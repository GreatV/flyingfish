use candle_core::{Device, Tensor};
use serde_json::{Value, json};
use std::{collections::HashMap, fs, process::Command};

fn ff() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ff"))
}

#[test]
fn model_paths_are_required_instead_of_assuming_a_local_directory_layout() {
    for (args, missing) in [
        (vec!["minicpm", "generate", "--prompt", "hello"], "--model"),
        (
            vec![
                "music",
                "generate",
                "--prompt",
                "piano",
                "--lyrics",
                "[Instrumental]",
                "--output",
                "unused.wav",
            ],
            "--model",
        ),
        (
            vec![
                "clip",
                "score",
                "--image",
                "input.png",
                "--text",
                "a rabbit",
            ],
            "--model",
        ),
        (
            vec![
                "trellis",
                "generate",
                "--conditioner",
                "encoder",
                "--prompt",
                "a rabbit",
                "--output",
                "unused.ply",
            ],
            "--model",
        ),
        (
            vec![
                "trellis",
                "generate",
                "--model",
                "checkpoint",
                "--prompt",
                "a rabbit",
                "--output",
                "unused.ply",
            ],
            "--conditioner",
        ),
        (
            vec![
                "trellis",
                "decode-gaussians",
                "--inputs",
                "input.safetensors",
                "--output",
                "unused.ply",
            ],
            "--model",
        ),
        (vec!["models", "list"], "--models-root"),
    ] {
        let output = ff().args(&args).output().unwrap();
        assert_eq!(
            output.status.code(),
            Some(2),
            "{args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(missing),
            "{args:?}"
        );
    }
}

#[test]
fn music_pipeline_inspects_every_component_and_materializes_a_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("owner/music");
    fs::create_dir_all(&model).unwrap();
    let mut index = json!({"_class_name": "MiniMaxMusic3ModularPipeline"});
    for role in [
        "condition_encoder",
        "language_model",
        "rvq_depth_decoder",
        "transformer",
        "vocoder",
    ] {
        let component = model.join(role);
        fs::create_dir_all(&component).unwrap();
        let tensor = Tensor::new(&[1f32, 2.], &Device::Cpu).unwrap();
        candle_core::safetensors::save(
            &HashMap::from([("weight", tensor)]),
            component.join("part.safetensors"),
        )
        .unwrap();
        fs::write(
            component.join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({
                "metadata": {"total_size": 8, "total_parameters": 2},
                "weight_map": {"weight": "part.safetensors"}
            }))
            .unwrap(),
        )
        .unwrap();
        index[role] = json!(["diffusers", "TestComponent", {"subfolder": role}]);
    }
    fs::write(
        model.join("modular_model_index.json"),
        serde_json::to_vec(&index).unwrap(),
    )
    .unwrap();
    let report = ff()
        .args(["models", "inspect", "--model"])
        .arg(&model)
        .args(["--verify", "--json"])
        .output()
        .unwrap();
    assert!(
        report.status.success(),
        "{}",
        String::from_utf8_lossy(&report.stderr)
    );
    let report: Value = serde_json::from_slice(&report.stdout).unwrap();
    assert_eq!(report["model"]["family"], "music3");
    let inventory = report["inventory"].as_array().unwrap();
    assert_eq!(inventory.len(), 5);
    assert!(
        inventory
            .iter()
            .all(|item| item["verified"] == true && item["tensors"] == 1)
    );
    let tensor = ff()
        .args(["models", "tensor", "--model"])
        .arg(&model)
        .args([
            "--component",
            "vocoder",
            "--name",
            "weight",
            "--device",
            "cpu",
        ])
        .output()
        .unwrap();
    assert!(
        tensor.status.success(),
        "{}",
        String::from_utf8_lossy(&tensor.stderr)
    );
    assert!(String::from_utf8_lossy(&tensor.stdout).contains("shape=[2]"));
    let unknown = ff()
        .args(["models", "tensor", "--model"])
        .arg(&model)
        .args(["--component", "missing", "--name", "weight"])
        .output()
        .unwrap();
    assert!(!unknown.status.success());
    assert!(String::from_utf8_lossy(&unknown.stderr).contains("available:"));
}

#[test]
fn model_list_reports_unrecognized_repositories_without_hiding_known_models() {
    let dir = tempfile::tempdir().unwrap();
    let known = dir.path().join("owner/known");
    fs::create_dir_all(&known).unwrap();
    fs::create_dir_all(dir.path().join("owner/unknown")).unwrap();
    fs::write(
        known.join("config.json"),
        r#"{"architectures":["Qwen3DSparkModel"]}"#,
    )
    .unwrap();
    let output = ff()
        .args(["models", "list", "--models-root"])
        .arg(dir.path())
        .arg("--json")
        .output()
        .unwrap();
    assert!(output.status.success());
    let entries: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(entries.as_array().unwrap().len(), 2);
    assert_eq!(
        entries[0]["model"]["dependencies"][0],
        "openbmb/MiniCPM5-2B"
    );
    assert!(entries[1]["error"].is_string());
}
