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
        spec("edge0_gemv", "cuda/edge0_gemv.cu"),
        spec("edge0_silu_mul", "cuda/silu_mul.cu"),
        spec("edge0_batched_gemv", "cuda/batched_gemv.cu"),
        spec("lora_add", "cuda/lora_add.cu"),
        spec("edge0_gdn", "cuda/gdn.cu"),
        spec("edge0_glue", "cuda/glue.cu"),
        spec("edge0_mega", "cuda/mega.cu"),
    ];
    ff_cuda_build::run(
        &specs,
        &std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")),
    );
}
