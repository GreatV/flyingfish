mod support;

use candle_core::{DType, Device, Shape, Tensor, safetensors};
use serde_json::Value;
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
};
use support::{PROBE_TENSOR, TinyTransformerFixture, insert_test_qwen_contract};
use tokenizers::{Tokenizer, models::wordlevel::WordLevel};

fn ff() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ff"))
}

fn successful_output(command: &mut Command) -> std::process::Output {
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "ff failed with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

struct RunnableCalibrationFixture {
    _temporary: tempfile::TempDir,
    model: PathBuf,
    inputs: PathBuf,
    policy: PathBuf,
}

impl RunnableCalibrationFixture {
    fn new() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let model = temporary.path().join("model");
        let transformer = model.join("transformer");
        fs::create_dir_all(&transformer).unwrap();

        let hidden = 4;
        let heads = 1;
        let head_dim = 6;
        let ffn = 5;
        let time = 2;
        let mut tensors = HashMap::new();
        for (name, input_width) in [
            ("proj_in", 1),
            ("audio_proj_in", 1),
            ("context_embedder", 2),
        ] {
            tensors.insert(format!("{name}.weight"), ones((hidden, input_width)));
            tensors.insert(format!("{name}.bias"), ones(hidden));
        }
        tensors.insert("time_embedder.linear_1.weight".to_owned(), ones((time, 2)));
        tensors.insert("time_embedder.linear_1.bias".to_owned(), ones(time));
        tensors.insert(
            "time_embedder.linear_2.weight".to_owned(),
            ones((time, time)),
        );
        tensors.insert("time_embedder.linear_2.bias".to_owned(), ones(time));

        let refiner = "token_refiner.refiner_blocks.0";
        tensors.insert(format!("{refiner}.norm1.weight"), ones(hidden));
        tensors.insert(format!("{refiner}.norm2.weight"), ones(hidden));
        for name in ["to_q", "to_k", "to_v"] {
            tensors.insert(
                format!("{refiner}.attn.{name}.weight"),
                ones((heads * head_dim, hidden)),
            );
        }
        tensors.insert(format!("{refiner}.attn.norm_q.weight"), ones(head_dim));
        tensors.insert(format!("{refiner}.attn.norm_k.weight"), ones(head_dim));
        tensors.insert(
            format!("{refiner}.attn.to_out.0.weight"),
            ones((hidden, heads * head_dim)),
        );
        tensors.insert(
            format!("{refiner}.ff.net.0.proj.weight"),
            ones((2 * ffn, hidden)),
        );
        tensors.insert(format!("{refiner}.ff.net.2.weight"), ones((hidden, ffn)));
        tensors.insert("token_refiner.final_norm.weight".to_owned(), ones(hidden));

        let block = "transformer_blocks.0";
        tensors.insert(
            format!("{block}.adaln_proj.linear.weight"),
            ones((6 * hidden * 3, time)),
        );
        tensors.insert(format!("{block}.norm1.weight"), ones(hidden));
        for name in ["to_q", "to_k", "to_v"] {
            tensors.insert(
                format!("{block}.attn.{name}.weight"),
                ones((heads * head_dim, hidden)),
            );
        }
        tensors.insert(format!("{block}.attn.norm_q.weight"), ones(head_dim));
        tensors.insert(format!("{block}.attn.norm_k.weight"), ones(head_dim));
        tensors.insert(
            format!("{block}.attn.to_out.0.weight"),
            ones((hidden, heads * head_dim)),
        );
        tensors.insert(format!("{block}.norm2.weight"), ones(hidden));
        tensors.insert(
            format!("{block}.ff.net.0.proj.weight"),
            ones((2 * ffn, hidden)),
        );
        tensors.insert(format!("{block}.ff.net.2.weight"), ones((hidden, ffn)));
        tensors.insert("norm_out.norm.weight".to_owned(), ones(hidden));
        tensors.insert(
            "norm_out.linear.weight".to_owned(),
            ones((2 * hidden, time)),
        );
        tensors.insert("norm_out.linear.bias".to_owned(), ones(2 * hidden));
        for name in ["proj_out", "audio_proj_out"] {
            tensors.insert(format!("{name}.weight"), ones((1, hidden)));
            tensors.insert(format!("{name}.bias"), ones(1));
        }

        let indexed_checkpoint_bytes = tensors
            .values()
            .map(|tensor| tensor.elem_count() * tensor.dtype().size_in_bytes())
            .sum::<usize>();
        safetensors::save(&tensors, transformer.join("weights.safetensors")).unwrap();
        let weight_map = tensors
            .keys()
            .map(|name| (name.clone(), "weights.safetensors"))
            .collect::<BTreeMap<_, _>>();
        fs::write(
            transformer.join("model.safetensors.index.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "metadata": {"total_size": indexed_checkpoint_bytes},
                "weight_map": weight_map
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            transformer.join("config.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "_class_name": "MiniMaxH3Transformer3DModel",
                "num_attention_heads": heads,
                "attention_head_dim": head_dim,
                "hidden_size": hidden,
                "num_layers": 1,
                "num_refiner_layers": 1,
                "ffn_dim": ffn,
                "in_channels": 1,
                "audio_in_channels": 1,
                "patch_size": [1, 1, 1],
                "text_dim": 2,
                "freq_dim": 2,
                "time_embed_hidden_dim": 2,
                "time_embed_dim": time,
                "rope_freq_dim": 1,
                "rope_theta": 10_000.0,
                "norm_eps": 1e-5,
                "qk_norm_eps": 1e-5,
                "final_norm_eps": 1e-5
            }))
            .unwrap(),
        )
        .unwrap();

        let inputs = temporary.path().join("inputs.safetensors");
        let mut input_tensors = HashMap::from([
            (
                "prompt_embeddings",
                Tensor::zeros((1, 1, 2), DType::F32, &Device::Cpu).unwrap(),
            ),
            (
                "text_token_tags",
                Tensor::new(&[1u32], &Device::Cpu).unwrap(),
            ),
            (
                "video_latents",
                Tensor::zeros((1, 1, 1, 1, 1), DType::F32, &Device::Cpu).unwrap(),
            ),
            (
                "audio_latents",
                Tensor::zeros((1, 1, 1), DType::F32, &Device::Cpu).unwrap(),
            ),
        ]);
        insert_test_qwen_contract(&mut input_tensors, 1);
        safetensors::save(&input_tensors, &inputs).unwrap();

        let policy = temporary.path().join("policy.json");
        let numerics = flyingfish::h3::policy::H3NumericalContract::for_verified_target(
            flyingfish::h3::policy::ExecutionBackendPolicy::Cpu,
            flyingfish::h3::policy::AttentionBackendPolicy::FullSoftmax,
        )
        .unwrap();
        fs::write(
            &policy,
            serde_json::to_vec_pretty(&serde_json::json!({
                "schema_version": flyingfish::h3::policy::EXECUTION_POLICY_SCHEMA_VERSION,
                "execution_backend": "cpu",
                "numerics": numerics,
                "attention": {
                    "backend": "full",
                    "configured_projection_rows": 1,
                    "configured_query_rows": 1,
                    "configured_key_rows": null
                },
                "configured_ffn_rows": 1,
                "configured_output_rows": 1,
                "weights": {
                    "source": "mmap",
                    "cache_shards": 1,
                    "cache_bytes": null
                },
                "precompute_adaln": true
            }))
            .unwrap(),
        )
        .unwrap();

        Self {
            _temporary: temporary,
            model,
            inputs,
            policy,
        }
    }

    fn report(&self) -> PathBuf {
        self._temporary.path().join("calibration.json")
    }
}

fn ones<S: Into<Shape>>(shape: S) -> Tensor {
    Tensor::ones(shape, DType::F32, &Device::Cpu).unwrap()
}

#[test]
fn help_exposes_the_composable_and_unified_workflows() {
    let output = ff().arg("--help").output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    for command in [
        "probe",
        "inspect",
        "tensor",
        "identify",
        "diff",
        "bench",
        "text",
        "video",
        "music",
        "3d",
        "similarity",
    ] {
        assert!(stdout.contains(command), "missing command {command}");
    }
    for phrase in [
        "Measure explicit sequential and local-interconnect I/O profiles",
        "Generate video and audio with a compatible model",
    ] {
        assert!(
            stdout.contains(phrase),
            "missing command about {phrase:?} in:\n{stdout}"
        );
    }
    let h3 = successful_output(ff().arg("h3").arg("--help"));
    let h3_help = String::from_utf8(h3.stdout).unwrap();
    assert!(
        h3_help.contains("Decode H3 audio or video latents"),
        "missing h3 decode about in:\n{h3_help}"
    );
    for forbidden in [
        "plan-t2va",
        "denoise-t2va",
        "identify-model",
        "calibrate-io",
        "compare-tensors",
        "__calibrate-t2va-trial",
    ] {
        assert!(
            !stdout.contains(forbidden),
            "stale command surface still lists {forbidden}"
        );
    }

    let inspect_help =
        String::from_utf8(successful_output(ff().args(["inspect", "--help"])).stdout).unwrap();
    assert!(
        inspect_help.contains("--host-cache-mib"),
        "inspect help is missing --host-cache-mib:\n{inspect_help}"
    );
    assert!(
        !inspect_help.contains("--host-cache-shards"),
        "inspect help still lists --host-cache-shards:\n{inspect_help}"
    );
    assert!(
        !inspect_help.contains("--component"),
        "inspect help still lists --component:\n{inspect_help}"
    );
    assert!(
        !inspect_help.contains("--execution-plan"),
        "inspect help still lists --execution-plan:\n{inspect_help}"
    );

    let denoise_help =
        String::from_utf8(successful_output(ff().args(["h3", "denoise", "--help"])).stdout)
            .unwrap();
    assert!(
        denoise_help.contains("--host-cache-mib"),
        "h3 denoise help is missing --host-cache-mib:\n{denoise_help}"
    );
    assert!(
        !denoise_help.contains("--host-cache-shards"),
        "h3 denoise help still lists --host-cache-shards:\n{denoise_help}"
    );
    assert!(
        !denoise_help.contains("--component"),
        "hidden --component leaked into h3 denoise help:\n{denoise_help}"
    );

    let plan_help =
        String::from_utf8(successful_output(ff().args(["h3", "plan", "--help"])).stdout).unwrap();
    assert!(
        plan_help.contains("--execution-plan"),
        "h3 plan help is missing --execution-plan:\n{plan_help}"
    );
}

#[test]
fn strong_model_identity_crosses_the_cli_boundary_without_clobbering() {
    let fixture = TinyTransformerFixture::new();
    let identity_path = fixture.scratch_path("transformer.identity.json");
    let output = successful_output(
        ff().arg("identify")
            .arg("--checkpoint")
            .arg(fixture.checkpoint())
            .arg("--output")
            .arg(&identity_path),
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("bytes of weight files (file count 1)"));

    let bytes = fs::read(&identity_path).unwrap();
    let identity = flyingfish::runtime::identity::WeakModelIdentity::from_json(&bytes).unwrap();
    assert_eq!(
        identity.weight_file_bytes(),
        fs::metadata(fixture.model().join("transformer/weights.safetensors"))
            .unwrap()
            .len()
    );
    assert_eq!(identity.weight_file_count(), 1);
    let json = String::from_utf8(bytes.clone()).unwrap();
    assert!(json.contains("local_metadata_manifest"));

    let failed = ff()
        .arg("identify")
        .arg("--checkpoint")
        .arg(fixture.checkpoint())
        .arg("--output")
        .arg(&identity_path)
        .output()
        .unwrap();
    assert!(!failed.status.success());
    assert_eq!(fs::read(&identity_path).unwrap(), bytes);

    let audio_vae = fixture.model().join("audio_vae");
    fs::create_dir(&audio_vae).unwrap();
    fs::write(audio_vae.join("config.json"), b"{}").unwrap();
    fs::write(
        audio_vae.join("diffusion_pytorch_model.safetensors"),
        b"single-file-weights",
    )
    .unwrap();
    let audio_identity_path = fixture.scratch_path("audio-vae.identity.json");
    successful_output(
        ff().arg("identify")
            .arg("--checkpoint")
            .arg(&audio_vae)
            .arg("--output")
            .arg(&audio_identity_path),
    );
    let audio_identity = flyingfish::runtime::identity::WeakModelIdentity::from_json(
        &fs::read(audio_identity_path).unwrap(),
    )
    .unwrap();
    assert_eq!(audio_identity.weight_file_bytes(), 19);
    assert_eq!(audio_identity.weight_file_count(), 1);
}

#[test]
fn unified_generation_refuses_preflight_before_loading_weight_payloads_or_writing_output() {
    let fixture = RunnableCalibrationFixture::new();
    let tokenizer_dir = fixture.model.join("tokenizer");
    fs::create_dir(&tokenizer_dir).unwrap();
    let vocabulary = [("[UNK]".to_owned(), 0), ("kite".to_owned(), 1)]
        .into_iter()
        .collect();
    let word_level = WordLevel::builder()
        .vocab(vocabulary)
        .unk_token("[UNK]".to_owned())
        .build()
        .unwrap();
    Tokenizer::new(word_level)
        .save(tokenizer_dir.join("tokenizer.json"), false)
        .unwrap();

    fs::write(
        fixture.model.join("transformer/weights.safetensors"),
        b"invalid tensor payload",
    )
    .unwrap();
    let output_dir = fixture._temporary.path().join("must-not-exist");
    let output = ff()
        .args(["video", "generate"])
        .arg("--model")
        .arg(&fixture.model)
        .args([
            "--prompt",
            "kite",
            "--device",
            "cpu",
            "--latent-frames",
            "1",
            "--latent-height",
            "1",
            "--latent-width",
            "1",
            "--audio-frames",
            "1",
            "--audio-channels",
            "1",
            "--sigma-points",
            "2",
            "--max-host-mib",
            "0",
            "--backend-workspace-mib",
            "0",
        ])
        .arg("--output-dir")
        .arg(&output_dir)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("binding budget: host"), "{stderr}");
    assert!(!stderr.contains("invalid safetensors"), "{stderr}");
    assert!(!output_dir.exists());
}

#[test]
fn unified_generation_resume_matches_uninterrupted_latents_and_refuses_policy_change() {
    let fixture = RunnableCalibrationFixture::new();
    let tokenizer_dir = fixture.model.join("tokenizer");
    fs::create_dir(&tokenizer_dir).unwrap();
    let vocabulary = [("[UNK]".to_owned(), 0), ("kite".to_owned(), 1)]
        .into_iter()
        .collect();
    let word_level = WordLevel::builder()
        .vocab(vocabulary)
        .unk_token("[UNK]".to_owned())
        .build()
        .unwrap();
    Tokenizer::new(word_level)
        .save(tokenizer_dir.join("tokenizer.json"), false)
        .unwrap();

    let output_dir = fixture._temporary.path().join("generate-resume");
    let generate = |policy: &std::path::Path| {
        let mut command = ff();
        command
            .args(["video", "generate"])
            .arg("--model")
            .arg(&fixture.model)
            .arg("--prompt")
            .arg("kite")
            .arg("--output-dir")
            .arg(&output_dir)
            .arg("--policy")
            .arg(policy)
            .args([
                "--device",
                "cpu",
                "--latent-frames",
                "1",
                "--latent-height",
                "1",
                "--latent-width",
                "1",
                "--audio-frames",
                "1",
                "--audio-channels",
                "1",
                "--target-hidden-state",
                "1",
                "--sigma-points",
                "3",
                "--no-progress",
            ]);
        command
    };

    let initialized = generate(&fixture.policy).output().unwrap();
    assert!(!initialized.status.success());
    assert!(output_dir.join("generation-ready").is_file());
    assert!(output_dir.join("checkpoints").is_dir());

    let partial = fixture._temporary.path().join("partial.safetensors");
    successful_output(
        ff().args(["h3", "denoise"])
            .arg("--model")
            .arg(&fixture.model)
            .arg("--inputs")
            .arg(&fixture.inputs)
            .arg("--output")
            .arg(&partial)
            .arg("--policy")
            .arg(&fixture.policy)
            .args([
                "--device",
                "cpu",
                "--sigma-points",
                "3",
                "--max-steps",
                "1",
                "--no-progress",
            ]),
    );
    fs::copy(
        &partial,
        output_dir.join("checkpoints/checkpoint-step000001.safetensors"),
    )
    .unwrap();

    let uninterrupted = fixture._temporary.path().join("uninterrupted.safetensors");
    successful_output(
        ff().args(["h3", "denoise"])
            .arg("--model")
            .arg(&fixture.model)
            .arg("--inputs")
            .arg(&fixture.inputs)
            .arg("--output")
            .arg(&uninterrupted)
            .arg("--policy")
            .arg(&fixture.policy)
            .args(["--device", "cpu", "--sigma-points", "3", "--no-progress"]),
    );

    let resumed = generate(&fixture.policy).output().unwrap();
    assert!(!resumed.status.success());
    let final_path = output_dir.join("denoised-latents.safetensors");
    let resumed_tensors = safetensors::load(&final_path, &Device::Cpu).unwrap();
    let uninterrupted_tensors = safetensors::load(&uninterrupted, &Device::Cpu).unwrap();
    for name in ["video_latents", "audio_latents"] {
        assert_eq!(
            resumed_tensors[name]
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            uninterrupted_tensors[name]
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            "resumed {name} differs from uninterrupted execution"
        );
    }
    assert_eq!(
        resumed_tensors["completed_steps"]
            .to_scalar::<u32>()
            .unwrap(),
        2
    );
    assert!(
        output_dir
            .join("checkpoints/checkpoint-step000002.safetensors")
            .is_file()
    );

    let final_before_mismatch = fs::read(&final_path).unwrap();
    let complete_resume = generate(&fixture.policy).output().unwrap();
    assert!(!complete_resume.status.success());
    assert!(
        String::from_utf8(complete_resume.stdout)
            .unwrap()
            .contains("denoise already completed 2/2")
    );
    assert!(
        String::from_utf8(complete_resume.stderr)
            .unwrap()
            .contains("audio_vae")
    );
    assert_eq!(fs::read(&final_path).unwrap(), final_before_mismatch);

    let mismatched_policy = fixture._temporary.path().join("mismatched-policy.json");
    let mut policy_json: Value =
        serde_json::from_slice(&fs::read(&fixture.policy).unwrap()).unwrap();
    policy_json["configured_output_rows"] = serde_json::json!(2);
    fs::write(
        &mismatched_policy,
        serde_json::to_vec_pretty(&policy_json).unwrap(),
    )
    .unwrap();
    let mismatch = generate(&mismatched_policy).output().unwrap();
    assert!(!mismatch.status.success());
    assert!(
        String::from_utf8(mismatch.stderr)
            .unwrap()
            .contains("disagrees with the requested one")
    );
    assert_eq!(fs::read(final_path).unwrap(), final_before_mismatch);
}

#[test]
fn non_terminal_denoise_progress_is_sparse_and_excludes_the_cold_first_step_from_eta() {
    let fixture = RunnableCalibrationFixture::new();
    let output_path = fixture
        ._temporary
        .path()
        .join("progress-output.safetensors");
    let output = ff()
        .args(["h3", "denoise"])
        .arg("--model")
        .arg(&fixture.model)
        .arg("--inputs")
        .arg(&fixture.inputs)
        .arg("--output")
        .arg(&output_path)
        .arg("--policy")
        .arg(&fixture.policy)
        .args(["--device", "cpu", "--sigma-points", "7"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "denoise failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert_eq!(
        stderr
            .lines()
            .filter(|line| line.starts_with("H3 resource policy "))
            .count(),
        1
    );
    let lines = stderr
        .lines()
        .filter(|line| line.starts_with("prepare:") || line.starts_with("denoise "))
        .collect::<Vec<_>>();
    assert_eq!(lines.len(), 4, "unexpected progress output:\n{stderr}");
    assert!(lines[0].starts_with("prepare:"));
    assert!(lines[1].starts_with("denoise 1/6:"));
    assert!(lines[1].contains("forecast stabilizing (0/5)"));
    assert!(!lines[1].contains("ETA"));
    assert!(lines[2].starts_with("denoise 5/6:"));
    assert!(lines[2].contains("forecast stabilizing (4/5)"));
    assert!(!lines[2].contains("ETA"));
    assert!(lines[3].starts_with("denoise 6/6:"));
    assert!(lines[3].contains("trailing 5-eval mean"));
    assert!(lines[3].contains("ETA"));
}

#[test]
fn denoise_checkpoint_is_atomically_published_without_clobbering() {
    let fixture = RunnableCalibrationFixture::new();
    let checkpoint = fixture._temporary.path().join("checkpoint.safetensors");
    let run = |inputs: &std::path::Path, output: &std::path::Path| {
        ff().args(["h3", "denoise"])
            .arg("--model")
            .arg(&fixture.model)
            .arg("--inputs")
            .arg(inputs)
            .arg("--output")
            .arg(output)
            .arg("--policy")
            .arg(&fixture.policy)
            .args([
                "--device",
                "cpu",
                "--sigma-points",
                "3",
                "--max-steps",
                "1",
                "--no-progress",
            ])
            .output()
            .unwrap()
    };

    let first = run(&fixture.inputs, &checkpoint);
    assert!(
        first.status.success(),
        "denoise failed: {}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert!(
        String::from_utf8(first.stdout)
            .unwrap()
            .contains("checkpoint artifact:")
    );
    let original = fs::read(&checkpoint).unwrap();
    let tensors = safetensors::load(&checkpoint, &Device::Cpu).unwrap();
    assert_eq!(tensors["completed_steps"].to_scalar::<u32>().unwrap(), 1);
    let first_identity = flyingfish::recovery::CheckpointIdentity::collect(&checkpoint).unwrap();
    assert_eq!(first_identity.policy_history.completed_evaluations, 1);
    assert_eq!(first_identity.policy_history.segments.len(), 1);

    let second = run(&fixture.inputs, &checkpoint);
    assert!(!second.status.success());
    assert!(
        String::from_utf8(second.stderr)
            .unwrap()
            .contains("denoise output already exists")
    );
    assert_eq!(fs::read(&checkpoint).unwrap(), original);

    let resumed = fixture._temporary.path().join("resumed.safetensors");
    let resumed_output = run(&checkpoint, &resumed);
    assert!(
        resumed_output.status.success(),
        "resume failed: {}",
        String::from_utf8_lossy(&resumed_output.stderr)
    );
    let resumed_identity = flyingfish::recovery::CheckpointIdentity::collect(&resumed).unwrap();
    assert_eq!(resumed_identity.policy_history.completed_evaluations, 2);
    assert_eq!(resumed_identity.policy_history.segments.len(), 1);
    assert_eq!(
        resumed_identity.policy_history.segments[0].end_evaluation_exclusive,
        2
    );
    assert!(
        fs::read_dir(fixture._temporary.path())
            .unwrap()
            .all(|entry| !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".ff-stage-dir-"))
    );
}

#[test]
fn an_incompatible_checkpoint_is_auditable_but_resume_refuses_before_model_payload() {
    let fixture = RunnableCalibrationFixture::new();
    let current = fixture
        ._temporary
        .path()
        .join("current-checkpoint.safetensors");
    let first = ff()
        .args(["h3", "denoise"])
        .arg("--model")
        .arg(&fixture.model)
        .arg("--inputs")
        .arg(&fixture.inputs)
        .arg("--output")
        .arg(&current)
        .arg("--policy")
        .arg(&fixture.policy)
        .args([
            "--device",
            "cpu",
            "--sigma-points",
            "3",
            "--max-steps",
            "1",
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert!(
        first.status.success(),
        "denoise failed: {}",
        String::from_utf8_lossy(&first.stderr)
    );

    let mut loaded = safetensors::load(&current, &Device::Cpu).unwrap();
    let metadata = flyingfish::recovery::take_t2va_checkpoint_metadata(&mut loaded)
        .unwrap()
        .unwrap();
    // A checkpoint recorded on another backend. It stays auditable, and a
    // resume must refuse it before it reads a single weight.
    let mut legacy_history = metadata.policy_history;
    for segment in &mut legacy_history.segments {
        segment.policy.execution_backend = flyingfish::h3::policy::ExecutionBackendPolicy::Cuda;
        *segment.policy.numerics =
            flyingfish::h3::policy::H3NumericalContract::for_verified_target(
                flyingfish::h3::policy::ExecutionBackendPolicy::Cuda,
                segment.policy.attention.backend,
            )
            .unwrap();
    }
    legacy_history.validate().unwrap();
    let mut legacy_tensors = HashMap::from([
        ("video_latents", loaded.remove("video_latents").unwrap()),
        ("audio_latents", loaded.remove("audio_latents").unwrap()),
        (
            "prompt_embeddings",
            loaded.remove("prompt_embeddings").unwrap(),
        ),
        ("text_token_tags", loaded.remove("text_token_tags").unwrap()),
        (
            "completed_steps",
            Tensor::new(metadata.completed_evaluations, &Device::Cpu).unwrap(),
        ),
        (
            "sigma_points",
            Tensor::new(metadata.sigma_points, &Device::Cpu).unwrap(),
        ),
        (
            "video_shift",
            Tensor::new(metadata.video_shift, &Device::Cpu).unwrap(),
        ),
        (
            "audio_shift",
            Tensor::new(metadata.audio_shift, &Device::Cpu).unwrap(),
        ),
    ]);
    assert!(loaded.is_empty());
    legacy_history
        .insert_checkpoint_tensors(&mut legacy_tensors, &Device::Cpu)
        .unwrap();
    let legacy = fixture
        ._temporary
        .path()
        .join("legacy-checkpoint.safetensors");
    safetensors::save(&legacy_tensors, &legacy).unwrap();

    let audit = fixture._temporary.path().join("legacy-history.json");
    let shown = ff()
        .args(["h3", "history"])
        .arg("--checkpoint")
        .arg(&legacy)
        .arg("--output")
        .arg(&audit)
        .output()
        .unwrap();
    assert!(
        shown.status.success(),
        "history audit failed: {}",
        String::from_utf8_lossy(&shown.stderr)
    );
    assert_eq!(
        fs::read(audit).unwrap(),
        legacy_history.canonical_json().unwrap()
    );

    fs::write(
        fixture.model.join("transformer/weights.safetensors"),
        b"corrupt model payload that must not be opened",
    )
    .unwrap();
    let refused_output = fixture._temporary.path().join("must-not-exist.safetensors");
    let refused = ff()
        .args(["h3", "denoise"])
        .arg("--model")
        .arg(&fixture.model)
        .arg("--inputs")
        .arg(&legacy)
        .arg("--output")
        .arg(&refused_output)
        .args(["--device", "cpu", "--sigma-points", "3", "--no-progress"])
        .output()
        .unwrap();
    assert!(!refused.status.success());
    let stderr = String::from_utf8(refused.stderr).unwrap();
    // Refused on the recorded conditioning, before a weight is read: the
    // corrupted model payload above never produces a parse error.
    assert!(stderr.contains("no conditioning provenance"), "{stderr}");
    assert!(!stderr.contains("invalid safetensors"), "{stderr}");
    assert!(!refused_output.exists());
}

#[test]
fn denoise_publishes_each_absolute_evaluation_checkpoint_and_extends_resume_history() {
    let fixture = RunnableCalibrationFixture::new();
    let first_output = fixture._temporary.path().join("first-output.safetensors");
    let first_checkpoints = fixture._temporary.path().join("first-checkpoints");
    let first = ff()
        .args(["h3", "denoise"])
        .arg("--model")
        .arg(&fixture.model)
        .arg("--inputs")
        .arg(&fixture.inputs)
        .arg("--output")
        .arg(&first_output)
        .arg("--checkpoint-dir")
        .arg(&first_checkpoints)
        .arg("--policy")
        .arg(&fixture.policy)
        .args([
            "--device",
            "cpu",
            "--sigma-points",
            "4",
            "--max-steps",
            "2",
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert!(
        first.status.success(),
        "denoise failed: {}",
        String::from_utf8_lossy(&first.stderr)
    );
    let first_stdout = String::from_utf8(first.stdout).unwrap();
    assert!(first_stdout.contains("evaluation checkpoint 1/3:"));
    assert!(first_stdout.contains("evaluation checkpoint 2/3:"));
    let step_one = first_checkpoints.join("checkpoint-step000001.safetensors");
    let step_two = first_checkpoints.join("checkpoint-step000002.safetensors");
    let entries = fs::read_dir(&first_checkpoints)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect::<Vec<_>>();
    assert_eq!(entries.len(), 2);
    assert!(step_one.is_file());
    assert!(step_two.is_file());
    let step_one_identity = flyingfish::recovery::CheckpointIdentity::collect(&step_one).unwrap();
    let step_two_identity = flyingfish::recovery::CheckpointIdentity::collect(&step_two).unwrap();
    assert_eq!(step_one_identity.completed_evaluations, 1);
    assert_eq!(step_one_identity.policy_history.completed_evaluations, 1);
    assert_eq!(step_two_identity.completed_evaluations, 2);
    assert_eq!(step_two_identity.policy_history.completed_evaluations, 2);
    assert_eq!(step_two_identity.policy_history.segments.len(), 1);
    assert_eq!(
        step_two_identity.policy_history.segments[0].end_evaluation_exclusive,
        2
    );
    let final_identity = flyingfish::recovery::CheckpointIdentity::collect(&first_output).unwrap();
    assert_eq!(
        final_identity.policy_history,
        step_two_identity.policy_history
    );

    let unchanged_step_one = fs::read(&step_one).unwrap();
    let unchanged_step_two = fs::read(&step_two).unwrap();
    let rejected_output = fixture
        ._temporary
        .path()
        .join("rejected-output.safetensors");
    let rejected = ff()
        .args(["h3", "denoise"])
        .arg("--model")
        .arg(&fixture.model)
        .arg("--inputs")
        .arg(&fixture.inputs)
        .arg("--output")
        .arg(&rejected_output)
        .arg("--checkpoint-dir")
        .arg(&first_checkpoints)
        .arg("--policy")
        .arg(&fixture.policy)
        .args([
            "--device",
            "cpu",
            "--sigma-points",
            "4",
            "--max-steps",
            "1",
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    assert!(
        String::from_utf8(rejected.stderr)
            .unwrap()
            .contains("checkpoint directory already exists")
    );
    assert!(!rejected_output.exists());
    assert_eq!(fs::read(&step_one).unwrap(), unchanged_step_one);
    assert_eq!(fs::read(&step_two).unwrap(), unchanged_step_two);

    let resumed_output = fixture._temporary.path().join("resumed-output.safetensors");
    let resumed_checkpoints = fixture._temporary.path().join("resumed-checkpoints");
    let resumed = ff()
        .args(["h3", "denoise"])
        .arg("--model")
        .arg(&fixture.model)
        .arg("--inputs")
        .arg(&step_two)
        .arg("--output")
        .arg(&resumed_output)
        .arg("--checkpoint-dir")
        .arg(&resumed_checkpoints)
        .arg("--policy")
        .arg(&fixture.policy)
        .args([
            "--device",
            "cpu",
            "--sigma-points",
            "4",
            "--max-steps",
            "1",
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert!(
        resumed.status.success(),
        "resume failed: {}",
        String::from_utf8_lossy(&resumed.stderr)
    );
    let step_three = resumed_checkpoints.join("checkpoint-step000003.safetensors");
    let step_three_identity =
        flyingfish::recovery::CheckpointIdentity::collect(&step_three).unwrap();
    assert_eq!(step_three_identity.completed_evaluations, 3);
    assert_eq!(step_three_identity.policy_history.completed_evaluations, 3);
    assert_eq!(step_three_identity.policy_history.segments.len(), 1);
    assert_eq!(
        step_three_identity.policy_history.segments[0].end_evaluation_exclusive,
        3
    );
}

#[cfg(unix)]
#[test]
fn denoise_stops_when_its_pre_execution_record_cannot_be_published() {
    use std::os::unix::{
        fs::{MetadataExt as _, PermissionsExt as _},
        process::CommandExt as _,
    };

    let fixture = RunnableCalibrationFixture::new();
    let output = fixture
        ._temporary
        .path()
        .join("unpublished-output.safetensors");
    let checkpoints = fixture._temporary.path().join("unwritable-checkpoints");
    let mut command = Command::new("sh");
    if fs::metadata(fixture._temporary.path()).unwrap().uid() == 0 {
        fs::set_permissions(fixture._temporary.path(), fs::Permissions::from_mode(0o777)).unwrap();
        command.uid(65534).gid(65534);
    }
    let failed = command
        .arg("-c")
        .arg("umask 0777; exec \"$0\" \"$@\"")
        .arg(env!("CARGO_BIN_EXE_ff"))
        .args(["h3", "denoise"])
        .arg("--model")
        .arg(&fixture.model)
        .arg("--inputs")
        .arg(&fixture.inputs)
        .arg("--output")
        .arg(&output)
        .arg("--checkpoint-dir")
        .arg(&checkpoints)
        .arg("--policy")
        .arg(&fixture.policy)
        .args([
            "--device",
            "cpu",
            "--sigma-points",
            "4",
            "--max-steps",
            "2",
            "--no-progress",
        ])
        .output()
        .unwrap();
    assert!(!failed.status.success());
    let stderr = String::from_utf8(failed.stderr).unwrap();
    assert!(
        stderr.contains("artifact hard-link probe") && stderr.contains("Permission denied"),
        "{stderr}"
    );
    assert!(!stderr.contains("prepare:"));
    assert!(!output.exists());
    assert!(!checkpoints.exists());
}

#[test]
fn calibrates_a_tiny_cpu_policy_in_two_isolated_trajectory_warmed_trials() {
    let fixture = RunnableCalibrationFixture::new();
    let report_path = fixture.report();
    let output = successful_output(
        ff().args(["h3", "calibrate"])
            .arg("--model")
            .arg(&fixture.model)
            .arg("--inputs")
            .arg(&fixture.inputs)
            .arg("--policy")
            .arg(&fixture.policy)
            .arg("--output")
            .arg(&report_path)
            .args([
                "--device",
                "cpu",
                "--warmup-prefix-evaluations",
                "1",
                "--measured-evaluations",
                "1",
                "--trials",
                "2",
                "--sigma-points",
                "3",
                "--video-shift",
                "12",
                "--audio-shift",
                "3",
            ]),
    );
    assert!(output.stderr.is_empty());
    assert!(
        String::from_utf8(output.stdout)
            .unwrap()
            .contains("saved calibration report")
    );

    let report: Value = serde_json::from_slice(&fs::read(&report_path).unwrap()).unwrap();
    assert_eq!(report["schema_version"], 1);
    assert_eq!(report["device"], "cpu");
    assert_eq!(report["trials_per_policy"], 2);
    let calibration = &report["calibration"];
    assert_eq!(calibration["schema_version"], 2);
    assert_eq!(calibration["device_selector"], "cpu");
    assert_eq!(calibration["cache_condition"], "trajectory_warmed");
    assert_eq!(calibration["cacheable"], false);
    assert_eq!(calibration["winner_selected"], false);
    assert_eq!(calibration["candidate_output_statistics_all_equal"], true);
    assert!(calibration["selection"].is_null());

    let candidates = calibration["candidates"].as_array().unwrap();
    assert_eq!(candidates.len(), 1);
    let candidate = &candidates[0];
    assert_eq!(candidate["measured_summary"]["samples"], 2);
    let trials = candidate["trials"].as_array().unwrap();
    assert_eq!(trials.len(), 2);
    // Calibration records statistics only; these do not prove tensor equality.
    let expected_output = &trials[0]["output"];
    assert!(expected_output["video"]["elements"].as_u64().unwrap() > 0);
    assert!(expected_output["audio"]["elements"].as_u64().unwrap() > 0);
    for (trial_index, trial) in trials.iter().enumerate() {
        assert_eq!(trial["request"]["trial_index"], trial_index as u64);
        assert_eq!(
            trial["observed_input_identity"],
            trial["request"]["input_identity"]
        );
        assert_eq!(
            trial["observed_model_identity"],
            trial["request"]["model_identity"]
        );
        assert_eq!(trial["timings"]["warmup"].as_array().unwrap().len(), 1);
        assert_eq!(trial["timings"]["measured"].as_array().unwrap().len(), 1);
        assert_eq!(trial["timings"]["warmup"][0]["step_index"], 0);
        assert_eq!(trial["timings"]["measured"][0]["step_index"], 1);
        assert_eq!(trial["timings"]["measured_summary"]["samples"], 1);
        assert_eq!(&trial["output"], expected_output);
    }
    assert_eq!(trials[0]["request"]["invocation_order"], 0);
    assert_eq!(trials[1]["request"]["invocation_order"], 1);
}

#[test]
fn calibration_rejects_resume_inputs_and_malformed_hidden_requests() {
    let fixture = RunnableCalibrationFixture::new();
    let mut values = safetensors::load(&fixture.inputs, &Device::Cpu).unwrap();
    values.insert(
        "completed_steps".to_owned(),
        Tensor::new(1_u32, &Device::Cpu).unwrap(),
    );
    let resume_inputs = fixture._temporary.path().join("resume-inputs.safetensors");
    safetensors::save(&values, &resume_inputs).unwrap();
    let report = fixture._temporary.path().join("rejected-report.json");
    let output = ff()
        .args(["h3", "calibrate"])
        .arg("--model")
        .arg(&fixture.model)
        .arg("--inputs")
        .arg(&resume_inputs)
        .arg("--policy")
        .arg(&fixture.policy)
        .arg("--output")
        .arg(&report)
        .args(["--device", "cpu", "--trials", "1", "--sigma-points", "3"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!report.exists());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("requires initial inputs"), "{stderr}");
    assert!(
        fs::read_dir(fixture._temporary.path())
            .unwrap()
            .all(|entry| !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".ff-stage-dir-"))
    );

    let mut child = ff()
        .arg("__calibrate-t2va-trial")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"{}").unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .contains("invalid calibration trial-worker request JSON")
    );
}

#[test]
fn cpu_probe_json_separates_stable_identity_from_dynamic_measurements() {
    let output = successful_output(ff().args(["probe", "--device", "cpu", "--json"]));
    assert!(output.stderr.is_empty());
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    let fingerprint = report["fingerprint"].as_object().unwrap();
    let snapshot = report["snapshot"].as_object().unwrap();

    assert_eq!(fingerprint["backend"], "cpu");
    assert_eq!(fingerprint["schema_version"], 1);
    assert!(fingerprint.get("device_identifier").is_none());
    assert!(fingerprint.get("device_identifier_kind").is_none());
    assert_eq!(report["calibration_cache_reuse_supported"], false);
    for cuda_field in [
        "cuda_compute_capability",
        "cuda_compute_capability_source",
        "cuda_compute_capability_unavailable_reason",
        "cuda_device_uuid",
        "cuda_device_uuid_source",
        "cuda_device_uuid_unavailable_reason",
        "cuda_pci_bus_id",
        "cuda_pci_bus_id_source",
        "cuda_pci_bus_id_unavailable_reason",
        "cuda_driver_api_version",
        "cuda_driver_api_version_source",
        "cuda_driver_api_version_unavailable_reason",
        "cuda_binding_api_version",
        "cuda_binding_api_version_source",
        "cuda_binding_api_version_unavailable_reason",
        "driver_version_source",
        "driver_version_unavailable_reason",
        "runtime_version_source",
        "runtime_version_unavailable_reason",
    ] {
        assert!(
            fingerprint[cuda_field].is_null(),
            "CPU fingerprint field {cuda_field} must be null"
        );
    }
    for dynamic_field in [
        "measured_at_unix_ms",
        "host_memory_available_bytes",
        "cgroup_v2_memory_current_bytes",
        "cgroup_v2_memory_available_bytes",
        "device_free_memory_bytes",
        "measurement_scope",
    ] {
        assert!(
            !fingerprint.contains_key(dynamic_field),
            "dynamic field {dynamic_field} leaked into the stable fingerprint"
        );
    }

    assert_eq!(snapshot["schema_version"], 1);
    assert!(snapshot["measured_at_unix_ms"].as_u64().unwrap() > 0);
    assert!(snapshot["host_memory_available_bytes"].is_u64());
    assert!(snapshot["device_free_memory_bytes"].is_null());
    assert!(
        snapshot["cgroup_v2_memory_current_bytes"].is_null()
            || snapshot["cgroup_v2_memory_current_bytes"].is_u64()
    );
    assert!(
        snapshot["cgroup_v2_memory_available_bytes"].is_null()
            || snapshot["cgroup_v2_memory_available_bytes"].is_u64()
    );
    let cgroup_limit = &snapshot["cgroup_v2_memory_limit"];
    assert!(cgroup_limit.is_null() || cgroup_limit.is_object());
    if let Some(limit) = cgroup_limit.as_object() {
        assert!(limit["kind"].is_string());
        assert!(!limit.contains_key("bytes") || limit["bytes"].is_u64());
    }

    let scopes = snapshot["measurement_scope"].as_object().unwrap();
    for name in ["host_memory", "cgroup_memory", "device_memory"] {
        assert!(
            scopes[name].is_null() || scopes[name].is_string(),
            "scope {name} must be a string or null"
        );
    }
    assert_eq!(scopes["host_memory"], "host_wide");
    assert!(scopes["device_memory"].is_null());

    for bench_field in [
        "host_sequential_read",
        "host_to_device",
        "host_to_device_over_host_read_ratio",
        "payload_format",
        "payload",
        "bytes_per_second",
        "threshold",
        "hybrid",
    ] {
        assert!(
            !report.as_object().unwrap().contains_key(bench_field),
            "probe JSON must not contain I/O calibration field {bench_field}"
        );
        assert!(
            !fingerprint.contains_key(bench_field),
            "probe fingerprint must not contain I/O calibration field {bench_field}"
        );
        assert!(
            !snapshot.contains_key(bench_field),
            "probe snapshot must not contain I/O calibration field {bench_field}"
        );
    }
}

#[test]
fn calibrate_io_json_report_is_versioned_keyed_and_cpu_h2d_unavailable() {
    let temporary = tempfile::tempdir().unwrap();
    let payload = temporary.path().join("payload.bin");
    let output = temporary.path().join("io-calibration.json");
    fs::write(&payload, vec![0x3c; 64 * 1024]).unwrap();

    let result = successful_output(
        ff().args(["bench", "io"])
            .arg("--payload")
            .arg(&payload)
            .arg("--output")
            .arg(&output)
            .arg("--device")
            .arg("cpu"),
    );
    assert!(String::from_utf8_lossy(&result.stdout).contains("host-to-device unavailable"));

    let report: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    assert_eq!(report["schema_version"], 1);
    assert_eq!(report["key"]["backend"], "cpu");
    assert_eq!(report["key"]["payload_format"], "raw_sequential");
    assert!(report["key"]["cuda_device_uuid"].is_null());
    assert_eq!(report["payload_bytes"], 64 * 1024);
    assert!(report["host_sequential_read"]["bytes"].as_u64().unwrap() >= 64 * 1024);
    assert!(
        report["host_sequential_read"]["elapsed_ns"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(
        report["host_sequential_read"]["bytes_per_second"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    assert!(report["host_to_device"]["sample"].is_null());
    assert_eq!(
        report["host_to_device"]["unavailable_reason"],
        "cpu_device_has_no_host_to_device_path"
    );
    assert!(report["host_to_device_over_host_read_ratio"].is_null());
    assert!(report.get("threshold").is_none());
    assert!(report.get("hybrid").is_none());
    assert!(report.get("moe_backend").is_none());

    let mut mismatched = report.clone();
    mismatched["key"]["payload_format"] = Value::String("nvfp4".to_owned());
    mismatched["key"]["cuda_device_uuid"] =
        Value::String("00010203-0405-0607-0809-0a0b0c0d0e0f".to_owned());
    let live: flyingfish::runtime::io_calibration::IoCalibrationKey =
        serde_json::from_value(mismatched["key"].clone()).unwrap();
    let accepted: flyingfish::runtime::io_calibration::IoCalibrationReport =
        serde_json::from_value(report).unwrap();
    accepted.validate().unwrap();
    let error = accepted
        .require_matching_key(&live)
        .unwrap_err()
        .to_string();
    assert!(error.contains("cannot be applied"));
}

#[test]
fn handler_errors_cross_the_real_binary_boundary_without_panicking() {
    let missing =
        std::env::temp_dir().join(format!("flyingfish-missing-model-{}", std::process::id()));
    let output = ff()
        .args(["inspect", "--checkpoint", missing.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("checkpoint directory does not exist"));
    assert!(!stderr.contains("panicked"));
}

#[test]
fn transformer_inspection_verification_and_tensor_loading_cross_the_binary_boundary() {
    let fixture = TinyTransformerFixture::new();
    let rejected = ff()
        .args([
            "inspect",
            "--checkpoint",
            fixture.checkpoint().to_str().unwrap(),
            "--execution-plan",
        ])
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    let rejected_stderr = String::from_utf8(rejected.stderr).unwrap();
    assert!(
        rejected_stderr.contains("unexpected argument")
            || rejected_stderr.contains("execution-plan")
    );

    let output = successful_output(
        ff().arg("inspect")
            .arg("--checkpoint")
            .arg(fixture.checkpoint())
            .arg("--verify"),
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(!stdout.contains("MiniMax-H3"));
    assert!(stdout.contains("tensors: 10"));
    assert!(stdout.contains("shards: 1"));
    assert!(stdout.contains("weight bytes: 40"));
    assert!(stdout.contains("verified: 10 tensors across 1 shards"));

    let output = successful_output(
        ff().arg("tensor")
            .arg("--checkpoint")
            .arg(fixture.checkpoint())
            .arg("--name")
            .arg(PROBE_TENSOR)
            .args(["--device", "cpu"]),
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("context_embedder.weight: dtype=F32, shape=[1], bytes=4"));
    assert!(stdout.contains("weights.safetensors"));
    assert!(stdout.contains("materialized on Cpu: [1]"));
}

#[test]
fn tiny_t2va_resource_plan_is_machine_readable_at_the_binary_boundary() {
    let fixture = TinyTransformerFixture::new();
    let output = successful_output(
        ff().args(["h3", "plan"])
            .arg("--model")
            .arg(fixture.model())
            .args([
                "--text-rows",
                "1",
                "--latent-frames",
                "1",
                "--latent-height",
                "1",
                "--latent-width",
                "1",
                "--audio-frames",
                "1",
                "--audio-channels",
                "1",
                "--attention-projection-chunk-size",
                "1",
                "--attention-query-chunk-size",
                "1",
                "--ffn-token-chunk-size",
                "1",
                "--output-token-chunk-size",
                "1",
                "--sigma-points",
                "2",
                "--cpu",
                "--json",
            ]),
    );
    assert!(output.stderr.is_empty());
    let plan: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(plan["schema_version"].as_u64().unwrap() > 0);
    assert_eq!(plan["geometry"]["text_rows"], 1);
    assert_eq!(plan["geometry"]["latent_frames"], 1);
    assert_eq!(plan["geometry"]["attention_key_chunk_policy"], "full");
    assert_eq!(plan["model"]["num_layers"], 1);
    assert_eq!(plan["sequence_rows"]["text"], 1);
    assert_eq!(plan["sequence_rows"]["audio"], 1);
    assert_eq!(plan["sequence_rows"]["video"], 1);
    assert_eq!(plan["sequence_rows"]["total"], 3);
    assert_eq!(plan["assumptions"]["device_memory_is_host"], true);
    assert_eq!(plan["assumptions"]["evaluation_count"], 1);
    assert_eq!(plan["assumptions"]["checkpoint_weight_bytes"], 40);
    assert_eq!(plan["assumptions"]["backend_workspace_bytes"], 0);
    assert_eq!(plan["assumptions"]["host_weight_cache_bytes"], 0);
    assert!(
        plan["assumptions"]["mapped_weight_residency_bytes"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(plan["peak_host_bytes"].as_u64().unwrap() > 0);
    assert!(plan["peak_device_bytes"].as_u64().unwrap() > 0);
    assert!(
        plan["compute_and_traffic"]["total_schedule_flops"]
            .as_u64()
            .unwrap()
            > 0
    );
}

#[test]
fn h3_plan_lists_execution_stages() {
    let fixture = TinyTransformerFixture::new();
    let plan = || {
        let mut command = ff();
        command
            .args(["h3", "plan"])
            .arg("--model")
            .arg(fixture.model())
            .args([
                "--text-rows",
                "1",
                "--latent-frames",
                "1",
                "--latent-height",
                "1",
                "--latent-width",
                "1",
                "--audio-frames",
                "1",
                "--audio-channels",
                "1",
                "--attention-projection-chunk-size",
                "1",
                "--attention-query-chunk-size",
                "1",
                "--ffn-token-chunk-size",
                "1",
                "--output-token-chunk-size",
                "1",
                "--sigma-points",
                "2",
                "--cpu",
            ]);
        command
    };
    let default_stdout = String::from_utf8(successful_output(&mut plan()).stdout).unwrap();
    assert!(
        !default_stdout.contains("peak stage weights:"),
        "stage listing is no longer opt-in:\n{default_stdout}"
    );
    assert!(
        !default_stdout.contains("stage context-input"),
        "stage listing is no longer opt-in:\n{default_stdout}"
    );

    let listed = successful_output(plan().arg("--execution-plan"));
    let stdout = String::from_utf8(listed.stdout).unwrap();
    for stage in [
        "context-input",
        "refiner-0-attention",
        "refiner-0-feed-forward",
        "refiner-output-norm",
        "time-input",
        "latent-input",
        "block-0-adaln",
        "block-0-attention",
        "block-0-feed-forward",
        "output",
    ] {
        assert!(
            stdout.contains(&format!("stage {stage}")),
            "missing stage {stage} in:\n{stdout}"
        );
    }
    assert!(
        stdout.contains("peak stage weights:"),
        "missing peak stage line in:\n{stdout}"
    );
}

#[test]
fn conservative_t2va_solver_is_deterministic_and_reports_truncation() {
    let fixture = TinyTransformerFixture::new();
    let run = || {
        successful_output(
            ff().args(["h3", "solve"])
                .arg("--model")
                .arg(fixture.model())
                .args([
                    "--text-rows",
                    "1",
                    "--latent-frames",
                    "1",
                    "--latent-height",
                    "1",
                    "--latent-width",
                    "1",
                    "--audio-frames",
                    "1",
                    "--audio-channels",
                    "1",
                    "--sigma-points",
                    "2",
                    "--cpu",
                    "--limit",
                    "3",
                    "--json",
                ]),
        )
    };

    let first = run();
    let second = run();
    assert_eq!(first.stderr, second.stderr);
    assert_eq!(first.stdout, second.stdout);

    let report: Value = serde_json::from_slice(&first.stdout).unwrap();
    assert_eq!(report["schema_version"], 1);
    assert_eq!(
        report["resource_estimate_schema_version"],
        report["candidates"][0]["estimate"]["schema_version"]
    );
    assert_eq!(report["model"]["hidden_size"], 4);
    assert_eq!(report["model"]["num_layers"], 1);
    assert_eq!(report["geometry"]["text_rows"], 1);
    assert_eq!(report["geometry"]["latent_frames"], 1);
    assert_eq!(report["geometry"]["audio_frames"], 1);
    assert_eq!(report["assumptions"]["device_memory_is_host"], true);
    assert_eq!(report["assumptions"]["evaluation_count"], 1);
    assert_eq!(report["assumptions"]["checkpoint_weight_bytes"], 40);
    assert_eq!(report["assumptions"]["backend_workspace_bytes"], 0);
    let largest_shard_bytes = report["host_admission"]["largest_shard_file_bytes"]
        .as_u64()
        .unwrap();
    let mmap_transition_bytes = report["host_admission"]["mmap_transition_bytes"]
        .as_u64()
        .unwrap();
    assert!(largest_shard_bytes > 40);
    assert_eq!(report["host_admission"]["mmap_transition_mapping_count"], 1);
    assert_eq!(mmap_transition_bytes, largest_shard_bytes);
    // Nothing asks for host bytes beyond the mapping now that the deprecated
    // --host-weight-cache-mib is gone.
    assert_eq!(report["host_admission"]["additional_allowance_bytes"], 0);
    assert_eq!(
        report["host_admission"]["charged_host_weight_residency_bytes"],
        mmap_transition_bytes
    );
    assert_eq!(report["assumptions"]["host_weight_cache_bytes"], 0);
    assert_eq!(
        report["assumptions"]["mapped_weight_residency_bytes"],
        mmap_transition_bytes
    );
    assert_eq!(report["search"]["attention_backends"][0], "full");
    assert_eq!(report["search"]["weight_source"], "mmap");
    assert_eq!(report["search"]["winner_selected"], false);
    assert_eq!(report["search"]["attention_projection_rows"][0], 16);
    assert_eq!(report["search"]["attention_query_rows"][0], 1);
    assert_eq!(
        report["search"]["attention_key_rows"]
            .as_array()
            .unwrap()
            .len(),
        0
    );
    assert_eq!(report["search"]["feed_forward_rows"][0], 32);
    assert_eq!(report["search"]["output_rows"][0], 32);
    assert_eq!(
        report["search"]["precompute_adaln"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(report["total_feasible_candidates"], 6);
    assert_eq!(report["returned_candidates"], 3);
    assert_eq!(report["truncated"], true);
    assert_eq!(report["hard_budget"]["max_host_bytes"], Value::Null);
    assert_eq!(report["hard_budget"]["max_device_bytes"], Value::Null);

    let candidates = report["candidates"].as_array().unwrap();
    assert_eq!(candidates.len(), 3);
    for candidate in candidates {
        let candidate_id = candidate["candidate_id"].as_str().unwrap();
        assert!(candidate_id.starts_with("candidate-"));
        assert!(candidate["peak_host_bytes"].as_u64().unwrap() > 0);
        assert!(candidate["peak_device_bytes"].as_u64().unwrap() > 0);
        assert_eq!(candidate["policy"]["execution_backend"], "cpu");
        assert_eq!(candidate["policy"]["attention"]["backend"], "full");
        assert_eq!(candidate["policy"]["weights"]["source"], "mmap");
        assert_eq!(
            candidate["estimate"]["peak_host_bytes"],
            candidate["peak_host_bytes"]
        );
        assert_eq!(
            candidate["estimate"]["peak_device_bytes"],
            candidate["peak_device_bytes"]
        );
        assert_eq!(candidate["estimate"]["model"], report["model"]);
        assert_eq!(candidate["estimate"]["geometry"]["text_rows"], 1);
        assert_eq!(
            candidate["estimate"]["geometry"]["attention_projection_chunk_size"],
            candidate["policy"]["attention"]["configured_projection_rows"]
        );
        assert_eq!(
            candidate["estimate"]["geometry"]["attention_query_chunk_size"],
            candidate["policy"]["attention"]["configured_query_rows"]
        );
        assert_eq!(
            candidate["estimate"]["assumptions"]["device_memory_is_host"],
            true
        );
        assert!(candidate["estimate"]["weights"].is_object());
        assert!(candidate["estimate"]["activations"].is_object());
        assert!(candidate["estimate"]["compute_and_traffic"].is_object());
    }
}

#[cfg(not(feature = "cuda"))]
#[test]
fn non_cuda_binary_emits_a_bound_symbolic_cuda_solver_policy() {
    let fixture = TinyTransformerFixture::new();
    let output = successful_output(
        ff().args(["h3", "solve"])
            .arg("--model")
            .arg(fixture.model())
            .args([
                "--text-rows",
                "1",
                "--latent-frames",
                "1",
                "--latent-height",
                "1",
                "--latent-width",
                "1",
                "--audio-frames",
                "1",
                "--audio-channels",
                "1",
                "--sigma-points",
                "2",
                "--limit",
                "1",
                "--json",
            ]),
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    let policy = &report["candidates"][0]["policy"];
    assert_eq!(
        policy["schema_version"],
        flyingfish::h3::policy::EXECUTION_POLICY_SCHEMA_VERSION
    );
    assert_eq!(policy["execution_backend"], "cuda");
    assert_eq!(policy["numerics"]["tensor_backend"], "candle_cuda011_v1");
    assert_eq!(
        policy["numerics"]["cuda_capabilities"]["tuned_kernels"],
        true
    );
    assert_eq!(
        policy["numerics"]["cuda_capabilities"]["reference_libraries"],
        true
    );
    assert_eq!(
        policy["numerics"]["attention"],
        "pytorch7269437_native_math_persistent_and_regular_softmax_cuda_v1"
    );
    assert_eq!(
        policy["numerics"]["cuda_artifacts"]["tuned_kernels"]["ptx_architecture"],
        "compute80"
    );
}

#[test]
fn conservative_t2va_solver_rejects_an_unsatisfiable_hard_budget() {
    let fixture = TinyTransformerFixture::new();
    let output = ff()
        .args(["h3", "solve"])
        .arg("--model")
        .arg(fixture.model())
        .args([
            "--text-rows",
            "1",
            "--latent-frames",
            "1",
            "--latent-height",
            "1",
            "--latent-width",
            "1",
            "--audio-frames",
            "1",
            "--audio-channels",
            "1",
            "--sigma-points",
            "2",
            "--cpu",
            "--max-host-mib",
            "0",
            "--max-device-mib",
            "0",
            "--json",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["total_feasible_candidates"], 0);
    assert_eq!(report["returned_candidates"], 0);
    assert_eq!(report["truncated"], false);
    assert_eq!(report["hard_budget"]["max_host_bytes"], 0);
    assert_eq!(report["hard_budget"]["max_device_bytes"], 0);
    assert_eq!(report["candidates"].as_array().unwrap().len(), 0);
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("no candidate in the built-in conservative full-softmax mmap search grid fits the hard budget"));
    assert!(stderr.contains("host 0 MiB, device 0 MiB"));
    assert!(!stderr.contains("panicked"));
}

#[test]
fn prepare_t2va_inputs_writes_expected_tiny_tensor_shapes() {
    let fixture = TinyTransformerFixture::new();
    let prompt = fixture.write_prompt_encoding();
    let prepared = fixture.scratch_path("prepared.safetensors");
    let run = || {
        let mut command = ff();
        command
            .args(["h3", "prepare"])
            .arg("--model")
            .arg(fixture.model())
            .arg("--prompt-encoding")
            .arg(&prompt)
            .arg("--output")
            .arg(&prepared)
            .args([
                "--latent-frames",
                "1",
                "--latent-height",
                "1",
                "--latent-width",
                "1",
                "--audio-frames",
                "1",
                "--audio-channels",
                "1",
                "--seed",
                "7",
                "--device",
                "cpu",
            ]);
        command
    };
    let output = successful_output(&mut run());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("saved prepared T2VA inputs"));

    let mut tensors = safetensors::load(&prepared, &Device::Cpu).unwrap();
    assert!(
        flyingfish::h3::policy::H3QwenNumericalContract::take_artifact_tensors(&mut tensors)
            .unwrap()
            .is_some()
    );
    assert_eq!(tensors.len(), 4);
    assert_eq!(tensors["prompt_embeddings"].dims(), &[1, 1, 2]);
    assert_eq!(tensors["text_token_tags"].dims(), &[1]);
    assert_eq!(tensors["video_latents"].dims(), &[1, 1, 1, 1, 1]);
    assert_eq!(tensors["audio_latents"].dims(), &[1, 1, 1]);
    let bytes = fs::read(&prepared).unwrap();
    let failed = run().output().unwrap();
    assert!(!failed.status.success());
    assert_eq!(fs::read(&prepared).unwrap(), bytes);
    assert!(
        fs::read_dir(fixture.scratch_path("."))
            .unwrap()
            .all(|entry| !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".ff-stage-dir-"))
    );

    assert!(flyingfish::recovery::CheckpointIdentity::collect(&prepared).is_err());
    let checkpoint = fixture.scratch_path("step-zero.safetensors");
    let initialize = || {
        let mut command = ff();
        command
            .args(["h3", "init"])
            .arg("--inputs")
            .arg(&prepared)
            .arg("--output")
            .arg(&checkpoint)
            .args([
                "--sigma-points",
                "2",
                "--video-shift",
                "12",
                "--audio-shift",
                "3",
            ]);
        command
    };
    successful_output(&mut initialize());
    let identity = flyingfish::recovery::CheckpointIdentity::collect(&checkpoint).unwrap();
    assert_eq!(identity.completed_evaluations, 0);
    assert_eq!(identity.sigma_points, 2);
    assert_eq!(identity.video_shift_f32_bits, 12.0f32.to_bits());
    assert_eq!(identity.audio_shift_f32_bits, 3.0f32.to_bits());
    let checkpoint_bytes = fs::read(&checkpoint).unwrap();
    assert!(!initialize().status().unwrap().success());
    assert_eq!(fs::read(checkpoint).unwrap(), checkpoint_bytes);
}

#[test]
fn root_layout_checkpoint_inspects_and_loads_f8e4m3_without_a_component_flag() {
    let temporary = tempfile::tempdir().unwrap();
    let checkpoint = temporary.path().join("ckpt");
    fs::create_dir_all(&checkpoint).unwrap();
    let tensor = Tensor::zeros((2, 4), DType::F8E4M3, &Device::Cpu).unwrap();
    safetensors::save(
        &HashMap::from([("experts.0.gate_proj.weight".to_owned(), tensor)]),
        checkpoint.join("model.safetensors"),
    )
    .unwrap();
    let inspect = successful_output(ff().args([
        "inspect",
        "--checkpoint",
        checkpoint.to_str().unwrap(),
        "--verify",
    ]));
    let inspect_out = String::from_utf8(inspect.stdout).unwrap();
    assert!(inspect_out.contains("tensors: 1"));
    assert!(inspect_out.contains("shards: 1"));
    assert!(!inspect_out.contains("MiniMax-H3"));

    let tensor = successful_output(ff().args([
        "tensor",
        "--checkpoint",
        checkpoint.to_str().unwrap(),
        "--name",
        "experts.0.gate_proj.weight",
        "--device",
        "cpu",
    ]));
    let tensor_out = String::from_utf8(tensor.stdout).unwrap();
    assert!(
        tensor_out.contains("F8E4M3")
            || tensor_out.contains("F8_E4M3")
            || tensor_out.contains("f8e4m3"),
        "missing FP8 dtype in:\n{tensor_out}"
    );
    assert!(tensor_out.contains("[2, 4]") || tensor_out.contains("shape=[2, 4]"));
}

#[cfg(target_os = "linux")]
#[test]
fn resource_benchmark_producer_retains_real_tiny_pairs_and_loadable_evidence() {
    let fixture = RunnableCalibrationFixture::new();
    let plan = fixture._temporary.path().join("resource-plan.json");
    let output = fixture._temporary.path().join("resource-pairs");
    fs::write(&plan,serde_json::to_vec(&serde_json::json!({
        "schema_version":1,"family":"h3","cache_state":"uncontrolled","pairs":3,"minimum_improvement_basis_points":200,
        "common_args":["h3","denoise","--model",fixture.model,"--inputs",fixture.inputs,"--device","cpu","--sigma-points","3","--max-steps","1","--no-progress"],
        "baseline_args":[],"candidate_args":["--host-cache-mib","1"]
    })).unwrap()).unwrap();
    let completed = Command::new("python3")
        .arg(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("scripts/resource-policy-bench.py"),
        )
        .arg("--binary")
        .arg(env!("CARGO_BIN_EXE_ff"))
        .arg("--plan")
        .arg(&plan)
        .arg("--output")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        completed.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&completed.stdout),
        String::from_utf8_lossy(&completed.stderr)
    );
    let (record, _) = flyingfish::resource_policy::evidence::ResourceEvidence::load(
        &output.join("evidence.json"),
    )
    .unwrap();
    assert_eq!(record.candidates[0].pairs.len(), 3);
    assert!(
        record.candidates[0]
            .pairs
            .iter()
            .all(|p| p.baseline_record.bytes > 0 && p.candidate_record.bytes > 0)
    );
    let status: Value =
        serde_json::from_slice(&fs::read(output.join("status.json")).unwrap()).unwrap();
    assert_eq!(status["state"], "finished");
    assert_eq!(status["output_equal"], true);
}

#[test]
fn denoise_selection_records_effective_explicit_budgets_at_both_boundaries() {
    let fixture = RunnableCalibrationFixture::new();
    let output = fixture._temporary.path().join("budget-check.safetensors");
    successful_output(
        ff().args(["h3", "denoise"])
            .arg("--model")
            .arg(&fixture.model)
            .arg("--inputs")
            .arg(&fixture.inputs)
            .arg("--output")
            .arg(&output)
            .arg("--policy")
            .arg(&fixture.policy)
            .args([
                "--device",
                "cpu",
                "--sigma-points",
                "3",
                "--max-steps",
                "1",
                "--max-host-mib",
                "64",
                "--max-device-mib",
                "32",
                "--no-progress",
            ]),
    );
    let record = flyingfish::runtime::resource_selection::ResourceSelectionProvenance::from_json(
        &fs::read(output.with_extension("resource-selection.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(record.workload["budget_record_version"], 1);
    assert_eq!(record.workload["host_budget_operator_override"], 1);
    assert_eq!(record.workload["compute_budget_operator_cap"], 1);
    for boundary in ["selection", "final_admission"] {
        for (axis, limit) in [("host", 64 << 20), ("compute", 32 << 20)] {
            let prefix = format!("{boundary}_{axis}");
            assert_eq!(record.workload[&format!("{prefix}_budget_bytes")], limit);
            assert_eq!(record.workload[&format!("{prefix}_budget_is_bounded")], 1);
            assert_eq!(
                record.workload[&format!("{prefix}_remaining_bytes")],
                limit - record.workload[&format!("{prefix}_peak_bytes")]
            );
        }
    }
    assert!(record.final_admission_snapshot.is_some());
}

/// `trellis inspect` reports a pipeline's components across the binary
/// boundary, including the ones it reaches into another checkpoint for.
///
/// Built from a fixture rather than the published checkpoints so the command
/// surface stays covered on a machine that has no model downloaded.
#[test]
fn trellis_inspect_reports_components_and_cross_checkpoint_references() {
    let models = tempfile::tempdir().unwrap();
    let flow = r#"{"name":"SparseStructureFlowModel","args":{"resolution":16,
        "in_channels":8,"out_channels":8,"model_channels":768,"cond_channels":768,
        "num_blocks":12,"num_heads":12,"mlp_ratio":4,"patch_size":1,"pe_mode":"ape",
        "qk_rms_norm":true,"use_fp16":true}}"#;
    let decoder = r#"{"name":"SparseStructureDecoder","args":{"out_channels":1,
        "latent_channels":8,"num_res_blocks":2,"num_res_blocks_middle":2,
        "channels":[512,128,32],"use_fp16":true}}"#;
    let write = |directory: &std::path::Path, stem: &str, config: &str| {
        std::fs::create_dir_all(directory).unwrap();
        std::fs::write(directory.join(format!("{stem}.json")), config).unwrap();
        std::fs::write(directory.join(format!("{stem}.safetensors")), b"").unwrap();
    };
    let text = models.path().join("owner-a/TRELLIS-text-fixture");
    let image = models.path().join("owner-b/TRELLIS-image-fixture");
    write(&text.join("ckpts"), "own_flow", flow);
    write(&image.join("ckpts"), "shared_decoder", decoder);
    std::fs::write(
        text.join("pipeline.json"),
        r#"{"name":"TrellisTextTo3DPipeline","args":{"models":{
             "sparse_structure_flow_model":"ckpts/own_flow",
             "sparse_structure_decoder":"Someone/TRELLIS-image-fixture/ckpts/shared_decoder"},
             "text_cond_model":"openai/clip-vit-large-patch14",
             "sparse_structure_sampler":{"name":"FlowEulerGuidanceIntervalSampler",
               "args":{"sigma_min":1e-5},
               "params":{"steps":25,"cfg_strength":7.5,"cfg_interval":[0.5,0.95],"rescale_t":3.0}}}}"#,
    )
    .unwrap();

    let output = successful_output(
        ff().args(["trellis", "inspect"])
            .arg("--model")
            .arg(&text)
            .arg("--models-root")
            .arg(models.path())
            .arg("--json"),
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["pipeline"], "TrellisTextTo3DPipeline");
    assert_eq!(report["conditioner"]["kind"], "text");
    assert_eq!(
        report["conditioner"]["model"],
        "openai/clip-vit-large-patch14"
    );
    let components = report["components"].as_array().unwrap();
    assert_eq!(components.len(), 2);
    let decoder = components
        .iter()
        .find(|component| component["role"] == "sparse_structure_decoder")
        .unwrap();
    assert_eq!(decoder["component"], "SparseStructureDecoder");
    assert_eq!(decoder["cross_checkpoint"], true);
    let weights = decoder["weights"].as_str().unwrap().replace('\\', "/");
    assert!(
        weights.contains("owner-b/TRELLIS-image-fixture"),
        "{weights}"
    );
}

/// A text run refuses a conditioner the pipeline does not name, rather than
/// conditioning on the wrong model and reporting a structure anyway.
#[test]
fn trellis_structure_refuses_a_conditioner_the_pipeline_does_not_name() {
    let models = tempfile::tempdir().unwrap();
    let text = models.path().join("owner/TRELLIS-text-fixture");
    let ckpts = text.join("ckpts");
    std::fs::create_dir_all(&ckpts).unwrap();
    std::fs::write(
        ckpts.join("own_flow.json"),
        r#"{"name":"SparseStructureFlowModel","args":{"resolution":16,
            "in_channels":8,"out_channels":8,"model_channels":768,"cond_channels":768,
            "num_blocks":12,"num_heads":12,"mlp_ratio":4,"patch_size":1,"pe_mode":"ape",
            "qk_rms_norm":true,"use_fp16":true}}"#,
    )
    .unwrap();
    std::fs::write(ckpts.join("own_flow.safetensors"), b"").unwrap();
    std::fs::write(
        text.join("pipeline.json"),
        r#"{"name":"TrellisTextTo3DPipeline","args":{
             "models":{"sparse_structure_flow_model":"ckpts/own_flow"},
             "text_cond_model":"openai/clip-vit-large-patch14",
             "sparse_structure_sampler":{"name":"FlowEulerGuidanceIntervalSampler",
               "args":{"sigma_min":1e-5},
               "params":{"steps":25,"cfg_strength":7.5,"cfg_interval":[0.5,0.95],"rescale_t":3.0}}}}"#,
    )
    .unwrap();
    let elsewhere = models.path().join("owner/some-other-encoder");
    std::fs::create_dir_all(&elsewhere).unwrap();

    let output = ff()
        .args(["trellis", "structure"])
        .arg("--model")
        .arg(&text)
        .arg("--conditioner")
        .arg(&elsewhere)
        .args(["--prompt", "a lovely rabbit", "--device", "cpu"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(
        error.contains("conditions on openai/clip-vit-large-patch14"),
        "{error}"
    );
    assert!(error.contains("some-other-encoder"), "{error}");
}
