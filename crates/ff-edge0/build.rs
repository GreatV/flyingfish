// Kernels are emitted as compute_80 PTX plus per-architecture cubins; a device
// with no matching cubin runs the driver's translation of the PTX.
fn main() {
    let spec = |stem, source| ff_cuda::build::KernelSpec {
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
        spec("int4_gemv_wide", "../ff-cuda/cuda/int4_gemv_wide.cu"),
    ];
    let manifest_path = std::path::PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR"),
    );
    ff_cuda::build::run(&specs, &manifest_path);
    ff_cuda::build::write_defines_consts(
        &std::path::PathBuf::from(std::env::var("OUT_DIR").expect("Cargo sets OUT_DIR")),
        &manifest_path.join("../ff-cuda/cuda/int4_gemv_wide.cu"),
        "int4_gemv_wide",
        &["RPB_V4"],
    );
}
