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
        spec("qwen_batch2", "cuda/batch2.cu"),
        spec("int4_gemv_wide", "../ff-cuda/cuda/int4_gemv_wide.cu"),
        spec("gemv16", "cuda/gemv16.cu"),
        spec("prefill_rows", "cuda/prefill_rows.cu"),
        spec("mma", "../ff-cuda/cuda/mma.cu"),
        spec("gdn_prefill", "../ff-cuda/cuda/gdn_prefill.cu"),
        spec("attn_prefill", "../ff-cuda/cuda/attn_prefill.cu"),
        spec("qk_norm_rope_rows", "../ff-cuda/cuda/qk_norm_rope_rows.cu"),
    ];
    let manifest_dir =
        std::env::var_os("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR");
    let source_root = std::path::PathBuf::from(&manifest_dir);
    let out_dir =
        std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
    ff_cuda::build::write_defines_consts(
        &out_dir,
        &source_root.join("../ff-cuda/cuda/int4_gemv_wide.cu"),
        "int4_gemv_wide",
        &["RPB_V4", "RPB_V4D8"],
    );
    ff_cuda::build::write_defines_consts(
        &out_dir,
        &source_root.join("../ff-cuda/cuda/mma.cu"),
        "mma",
        &["BM", "BT", "BK"],
    );
    ff_cuda::build::run(&specs, &std::path::PathBuf::from(manifest_dir));
}
