use std::{path::PathBuf, process::Command};

fn main() {
    let kernels = [
        ("cuda/edge0_gemv.cu", "edge0_gemv.ptx"),
        ("cuda/silu_mul.cu", "edge0_silu_mul.ptx"),
        ("cuda/batched_gemv.cu", "edge0_batched_gemv.ptx"),
        ("cuda/lora_add.cu", "lora_add.ptx"),
        ("cuda/gdn.cu", "edge0_gdn.ptx"),
        ("cuda/glue.cu", "edge0_glue.ptx"),
        ("cuda/mega.cu", "edge0_mega.ptx"),
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
        let output = out_dir.join(out_name);
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
