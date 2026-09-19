use std::{path::PathBuf, process::Command};

// Kernels are emitted as compute_80 PTX; newer devices run the driver's translation of it.
fn main() {
    let kernels = [
        ("cuda/batch2.cu", "qwen_batch2.ptx"),
        ("cuda/wide_gemv.cu", "wide_gemv.ptx"),
    ];
    for (src, _) in &kernels {
        println!("cargo:rerun-if-changed={src}");
    }
    println!("cargo:rerun-if-env-changed=NVCC");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let nvcc = std::env::var_os("NVCC").unwrap_or_else(|| "nvcc".into());
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
    for (src, out_name) in &kernels {
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
            .arg(out_dir.join(out_name))
            .arg(src)
            .output()
            .unwrap_or_else(|e| panic!("failed to invoke nvcc for {src}: {e}"));
        assert!(
            compiled.status.success(),
            "nvcc failed for {src}:\n{}",
            String::from_utf8_lossy(&compiled.stderr)
        );
    }
}
