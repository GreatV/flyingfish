#[cfg(not(feature = "cuda"))]
#[test]
fn batch_creates_its_output_directory_and_reports_worker_startup_failure() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model");
    std::fs::create_dir(&model).unwrap();
    std::fs::write(
        model.join("model_index.json"),
        r#"{"_class_name":"MiniMaxMusic3ModularPipeline"}"#,
    )
    .unwrap();
    let requests = dir.path().join("requests.json");
    std::fs::write(
        &requests,
        r#"[{"id":"song","prompt":"guitar","lyrics":"[Instrumental]"}]"#,
    )
    .unwrap();
    let output_dir = dir.path().join("new-output");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_ff"))
        .args(["music", "batch", "--devices", "cuda:0", "--model"])
        .arg(model)
        .arg("--requests")
        .arg(requests)
        .arg("--output-dir")
        .arg(&output_dir)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output_dir.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["skipped"], serde_json::json!(["song"]));
    assert_eq!(report["worker_errors"].as_array().unwrap().len(), 1);
    assert!(report["requests"].as_array().unwrap().is_empty());
    assert!(!output_dir.join("song.wav").exists());
}
