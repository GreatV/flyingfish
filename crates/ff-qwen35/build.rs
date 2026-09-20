// Kernels are emitted as compute_80 PTX plus per-architecture cubins; a device
// with no matching cubin runs the driver's translation of the PTX.
fn main() {
    let spec = |stem, source| ff_cuda_build::KernelSpec {
        stem,
        source: Some(source),
        staged_ptx: None,
        extra_flags: &[],
    };
    let specs = [
        spec("qwen_batch2", "cuda/batch2.cu"),
        spec("wide_gemv", "cuda/wide_gemv.cu"),
    ];
    ff_cuda_build::run(
        &specs,
        &std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")),
    );
}
