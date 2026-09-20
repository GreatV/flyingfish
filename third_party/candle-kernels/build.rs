use cudaforge::{KernelBuilder, Result};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    // Build for PTX at the fleet baseline: PTX cannot be lowered below its
    // virtual architecture, so a newer-than-sm_80 emission would leave older
    // fleet devices with neither cubin nor loadable fallback.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let is_target_msvc = matches!(env::var("TARGET"), Ok(t) if t.contains("msvc"));
    let mut ptx_builder = KernelBuilder::new()
        .source_dir("src") // Scan src/ for .cu files
        .exclude(&["moe_*.cu", "mmvq_gguf.cu", "mmq_*.cu"]) // Exclude statically compiled kernels from ptx build
        .compute_cap_arch("80")
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");
    if is_target_msvc {
        // CUDA 13.3's CCCL hard-#errors (C1189) under MSVC's traditional
        // preprocessor; the conforming one is required. gcc/clang unaffected.
        ptx_builder = ptx_builder.arg("-Xcompiler=/Zc:preprocessor");
    }
    let bindings = ptx_builder.build_ptx()?;

    bindings.write(&ptx_path)?;

    // Translate each PTX to per-architecture cubins. A cubin loads where the
    // driver would reject the PTX's ISA (toolkit newer than the driver); the
    // runtime picks one only on an exact architecture match.
    {
        use std::fmt::Write as _;
        println!("cargo:rerun-if-env-changed=NVCC");
        println!("cargo:rerun-if-env-changed=PTXAS");
        println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
        println!(
            "cargo:rerun-if-env-changed={}",
            ff_cuda_build::ARCHITECTURE_ENVIRONMENT_VARIABLE
        );
        let nvcc = std::env::var_os("NVCC").unwrap_or_else(|| "nvcc".into());
        let ptxas = ff_cuda_build::resolve_ptxas(&nvcc);
        let architectures = ff_cuda_build::resolve_target_architectures();
        let mut cubins_rs = String::new();
        let mut ptx_files: Vec<PathBuf> = std::fs::read_dir(&out_dir)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "ptx"))
            .collect();
        ptx_files.sort();
        for ptx in ptx_files {
            let stem = ptx.file_stem().unwrap().to_string_lossy().into_owned();
            let constant = stem.to_uppercase().replace(['.', '-'], "_");
            let images: Vec<u32> = architectures
                .iter()
                .copied()
                .filter(|&arch| ff_cuda_build::assemble_cubin(&ptxas, &out_dir, &stem, arch))
                .collect();
            writeln!(
                cubins_rs,
                "pub const {constant}: &[crate::Cubin] = &["
            )
            .unwrap();
            for arch in images {
                writeln!(
                    cubins_rs,
                    "    crate::Cubin {{ architecture: {arch}, image: include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{stem}.sm_{arch}.cubin\")) }},"
                )
                .unwrap();
            }
            cubins_rs.push_str("];\n");
        }
        std::fs::write(out_dir.join("cubins.rs"), cubins_rs)?;
    }

    let mut moe_builder = KernelBuilder::default()
        .source_files(vec![
            "src/moe/moe_gguf.cu",
            "src/moe/moe_wmma.cu",
            "src/moe/moe_wmma_gguf.cu",
            "src/mmvq_gguf.cu",
            "src/mmq_gguf/mmq_quantize.cu",
            "src/mmq_gguf/mmq_instance_q4_0.cu",
            "src/mmq_gguf/mmq_instance_q4_1.cu",
            "src/mmq_gguf/mmq_instance_q5_0.cu",
            "src/mmq_gguf/mmq_instance_q5_1.cu",
            "src/mmq_gguf/mmq_instance_q8_0.cu",
            "src/mmq_gguf/mmq_instance_q2_k.cu",
            "src/mmq_gguf/mmq_instance_q3_k.cu",
            "src/mmq_gguf/mmq_instance_q4_k.cu",
            "src/mmq_gguf/mmq_instance_q5_k.cu",
            "src/mmq_gguf/mmq_instance_q6_k.cu",
        ])
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // Disable bf16 WMMA kernels on GPUs older than sm_80 (Ampere).
    // bf16 WMMA fragments require compute capability >= 8.0.
    let compute_cap = cudaforge::detect_compute_cap()
        .map(|arch| arch.base())
        .unwrap_or(80);
    if compute_cap < 80 {
        moe_builder = moe_builder.arg("-DNO_BF16_KERNEL");
    }

    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            moe_builder = moe_builder.arg("-D_USE_MATH_DEFINES");
            moe_builder = moe_builder.arg("-Xcompiler=/Zc:preprocessor");
        }
    }

    if !is_target_msvc {
        moe_builder = moe_builder.arg("-Xcompiler").arg("-fPIC");
    }

    moe_builder.build_lib(out_dir.join("libmoe.a"))?;
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
    Ok(())
}
