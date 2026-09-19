use serde_json::json;
use std::{fs, process::Command};

fn ff() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ff"))
}

#[test]
fn task_help_exposes_3d_and_model_specific_options_without_loading_models() {
    for (task, adapter, operation, expected) in [
        ("text", "glm", "generate", "--expert-cache-mib"),
        ("text", "minicpm", "generate", "--draft-model"),
        ("text", "edge0", "generate", "--resident-experts"),
        ("text", "qwen35", "generate", "--image"),
        ("video", "h3", "generate", "--duration-seconds"),
        ("music", "music3", "generate", "--lyrics"),
        ("3d", "trellis", "generate", "--conditioner"),
        ("similarity", "clip", "score", "--text"),
    ] {
        let output = ff()
            .args([task, operation, "--adapter", adapter, "--help"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let help = String::from_utf8(output.stdout).unwrap();
        assert!(help.contains(expected), "{help}");
        assert!(!help.contains("--dry-run"), "{help}");
    }
    let output = ff().args(["text", "generate", "--help"]).output().unwrap();
    assert!(output.status.success());
    let help = String::from_utf8(output.stdout).unwrap();
    assert!(help.contains("--max-new-tokens"), "{help}");
    assert!(!help.contains("--expert-cache-mib"), "{help}");
}

#[test]
fn automatic_routing_rejects_wrong_tasks_and_unsupported_flags_before_outputs() {
    let directory = tempfile::tempdir().unwrap();
    let model = directory.path().join("renamed-checkpoint");
    fs::create_dir(&model).unwrap();
    fs::write(
        model.join("config.json"),
        serde_json::to_vec(&json!({
            "architectures":["LlamaForCausalLM"], "model_type":"llama"
        }))
        .unwrap(),
    )
    .unwrap();
    let destination = directory.path().join("unwritten");
    let output = ff()
        .args(["video", "generate", "--model"])
        .arg(&model)
        .args(["--prompt", "hello", "--output-dir"])
        .arg(&destination)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("available tasks: text"), "{error}");
    assert!(!destination.exists());

    for unsupported in ["--expert-cache-mib", "--dry-run"] {
        let output = ff()
            .args(["text", "generate", "--model"])
            .arg(&model)
            .args(["--prompt", "hello", unsupported])
            .output()
            .unwrap();
        assert!(!output.status.success());
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(error.contains("unexpected argument"), "{error}");
        assert!(error.contains(unsupported), "{error}");
    }
}

#[test]
fn both_3d_architectures_select_the_registered_generation_schema() {
    for architecture in ["TrellisTextTo3DPipeline", "Trellis2ImageTo3DPipeline"] {
        let directory = tempfile::tempdir().unwrap();
        fs::write(
            directory.path().join("pipeline.json"),
            serde_json::to_vec(&json!({"name":architecture})).unwrap(),
        )
        .unwrap();
        let output = ff()
            .args(["3d", "generate", "--model"])
            .arg(directory.path())
            .arg("--help")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let help = String::from_utf8(output.stdout).unwrap();
        assert!(
            help.contains("--conditioner") && help.contains("--resolution"),
            "{help}"
        );
    }
}
