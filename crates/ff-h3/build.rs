// Compile compute_80 PTX and per-architecture cubins from repository-root source paths.
fn main() {
    let spec = |stem, source| ff_cuda::build::KernelSpec {
        stem,
        source: Some(source),
        staged_ptx: None,
        extra_flags: &[],
    };
    let specs = [
        spec("rms_norm_bf16", "crates/ff-cuda/cuda/rms_norm_bf16.cu"),
        spec("softmax_f32", "crates/ff-cuda/cuda/softmax_f32.cu"),
        spec("scale_bf16", "crates/ff-cuda/cuda/scale_bf16.cu"),
        spec("mean_f32", "crates/ff-cuda/cuda/mean_f32.cu"),
        spec("layer_norm_bf16", "crates/ff-cuda/cuda/layer_norm_bf16.cu"),
        ff_cuda::build::KernelSpec {
            stem: "gelu_bf16",
            source: None,
            staged_ptx: Some("crates/ff-cuda/cuda/gelu_bf16_nvrtc130.ptx"),
            extra_flags: &[],
        },
    ];
    let manifest_dir =
        std::env::var_os("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR");
    let workspace_root = std::path::PathBuf::from(manifest_dir).join("../..");
    let Some(report) = ff_cuda::build::run(&specs, &workspace_root) else {
        return;
    };

    const PINNED_NVCC_VERSION: &str = "13.2.86";
    if report.nvcc_version != PINNED_NVCC_VERSION {
        println!(
            "cargo:warning=building H3 CUDA kernels with nvcc {}, not the \
             pinned {PINNED_NVCC_VERSION}; this build selects the portable CUDA backend",
            report.nvcc_version
        );
    }
    println!(
        "cargo:rustc-env=FLYINGFISH_CUDA_NVCC_VERSION={}",
        report.nvcc_version
    );
    println!(
        "cargo:rustc-env=FLYINGFISH_H3_RMS_NORM_PTX_ARCH={}",
        ff_cuda::build::PTX_ARCHITECTURE
    );
    println!("cargo:rustc-env=FLYINGFISH_GELU_NVRTC_VERSION=13.0.88");
}
