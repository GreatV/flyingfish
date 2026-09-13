use std::{path::PathBuf, process::Command};

/// The reference toolkit. A different one is recorded, not refused.
///
/// This used to be an assertion, which made a toolkit mismatch a build failure
/// rather than a difference worth recording — and so made the crate
/// unbuildable on any machine whose CUDA install was not this exact one. The
/// toolkit version below records the compiler used.
const REFERENCE_NVCC_VERSION: &str = "13.2.86";

fn main() {
    println!("cargo:rerun-if-changed=cuda/rsqrt_f32.cu");
    println!("cargo:rerun-if-changed=cuda/fp8_dequant.cu");
    println!("cargo:rerun-if-env-changed=NVCC");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let nvcc = std::env::var_os("NVCC").unwrap_or_else(|| "nvcc".into());
    let version = Command::new(&nvcc)
        .arg("--version")
        .output()
        .expect("GLM CUDA requires nvcc");
    assert!(
        version.status.success(),
        "{nvcc:?} --version failed:\n{}",
        String::from_utf8_lossy(&version.stderr)
    );
    let version = String::from_utf8_lossy(&version.stdout);
    let compiled_nvcc_version = version
        .split_whitespace()
        .find_map(|token| token.strip_prefix('V'))
        .map(|version| version.trim_end_matches(',').to_owned())
        .unwrap_or_else(|| {
            panic!("could not read a version from {nvcc:?} --version output:\n{version}")
        });
    if compiled_nvcc_version != REFERENCE_NVCC_VERSION {
        println!(
            "cargo:warning=building the GLM CUDA kernel with nvcc {compiled_nvcc_version}, not \
             the reference {REFERENCE_NVCC_VERSION}"
        );
    }
    let output = PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"))
        .join("glm_rsqrt_f32.ptx");
    let compiled = Command::new(&nvcc)
        .args([
            "--ptx",
            "--std=c++17",
            "-O2",
            "--gpu-architecture=compute_80",
            "--prec-div=true",
            "--prec-sqrt=true",
            "--ftz=false",
            "-o",
        ])
        .arg(&output)
        .arg("cuda/rsqrt_f32.cu")
        .output()
        .expect("failed to invoke GLM nvcc");
    assert!(
        compiled.status.success(),
        "GLM rsqrt compilation failed: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    println!("cargo:rustc-env=GLM_CUDA_NVCC_VERSION={compiled_nvcc_version}");
    let output = PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"))
        .join("glm_fp8_dequant.ptx");
    let compiled = Command::new(&nvcc)
        .args([
            "--ptx",
            "--std=c++17",
            "-O2",
            "--gpu-architecture=compute_80",
            "--prec-div=true",
            "--prec-sqrt=true",
            "--ftz=false",
            "--fmad=false",
            "-o",
        ])
        .arg(&output)
        .arg("cuda/fp8_dequant.cu")
        .output()
        .expect("failed to invoke GLM FP8 nvcc");
    assert!(
        compiled.status.success(),
        "GLM FP8 compilation failed: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
}
