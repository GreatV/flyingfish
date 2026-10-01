/// The tuned backend requires the pinned toolkit; other toolkit builds use the portable backend.
const REFERENCE_NVCC_VERSION: &str = "13.2.86";

// Kernels are emitted as compute_80 PTX plus per-architecture cubins; a device
// with no matching cubin runs the driver's translation of the PTX.
fn main() {
    let spec = |stem, source, extra_flags| ff_cuda::build::KernelSpec {
        stem,
        source: Some(source),
        staged_ptx: None,
        extra_flags,
    };
    let specs = [
        spec("glm_rsqrt_f32", "cuda/rsqrt_f32.cu", &[]),
        spec("glm_fp8_dequant", "cuda/fp8_dequant.cu", &["--fmad=false"]),
        spec(
            "glm_mhc_sinkhorn_loop_f32",
            "cuda/mhc_sinkhorn_loop_f32.cu",
            &[],
        ),
        // --fmad=false preserves the F32 rounding of the square before accumulation.
        spec(
            "glm_normalized_f32",
            "cuda/normalized_f32.cu",
            &["--fmad=false"],
        ),
    ];
    let manifest_dir =
        std::env::var_os("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR");
    let Some(report) = ff_cuda::build::run(&specs, &std::path::PathBuf::from(manifest_dir)) else {
        return;
    };
    if report.nvcc_version != REFERENCE_NVCC_VERSION {
        println!(
            "cargo:warning=building the GLM CUDA kernel with nvcc {}, not \
             the reference {REFERENCE_NVCC_VERSION}",
            report.nvcc_version
        );
    }
    println!(
        "cargo:rustc-env=GLM_CUDA_NVCC_VERSION={}",
        report.nvcc_version
    );
}
