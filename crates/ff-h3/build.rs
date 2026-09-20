// Kernels are emitted as compute_80 PTX plus per-architecture cubins; a device
// with no matching cubin runs the driver's translation of the PTX.
//
// The nvcc invocation is identity-sealed: sources stay at workspace-root
// `src/cuda/*.cu` with the workspace root as the working directory, so the PTX
// this build emits is byte-identical across machines.
fn main() {
    let spec = |stem, source| ff_cuda_build::KernelSpec {
        stem,
        source: Some(source),
        staged_ptx: None,
        extra_flags: &[],
    };
    let specs = [
        spec("h3_rms_norm_bf16", "src/cuda/h3_rms_norm_bf16.cu"),
        spec("h3_sdpa_softmax_f32", "src/cuda/h3_sdpa_softmax_f32.cu"),
        spec(
            "qwen_attention_scale_bf16",
            "src/cuda/qwen_attention_scale_bf16.cu",
        ),
        spec("qwen_head_mean_f32", "src/cuda/qwen_head_mean_f32.cu"),
        spec("qwen_layer_norm_bf16", "src/cuda/qwen_layer_norm_bf16.cu"),
        ff_cuda_build::KernelSpec {
            stem: "qwen_gelu_bf16",
            source: None,
            staged_ptx: Some("src/cuda/qwen_gelu_bf16_nvrtc130.ptx"),
            extra_flags: &[],
        },
    ];
    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let Some(report) = ff_cuda_build::run(&specs, &workspace_root) else {
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
        ff_cuda_build::PTX_ARCHITECTURE
    );
    println!("cargo:rustc-env=FLYINGFISH_QWEN_GELU_NVRTC_VERSION=13.0.88");
    println!(
        "cargo:rustc-env=FLYINGFISH_QWEN_GELU_NVRTC_SHA256=a49e67e8e74590f1e98de55c39c6287efd3f59e3c3797464d7bbe0fe01349b11"
    );
    println!(
        "cargo:rustc-env=FLYINGFISH_QWEN_GELU_NVRTC_BUILTINS_SHA256=91dcd944d01da9c0f08fff5d779db136a47f6d62bdb63bae900d2f481e92c3a2"
    );
}
