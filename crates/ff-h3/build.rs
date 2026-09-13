use std::{fmt::Write as _, path::PathBuf, process::Command};

const H3_RMS_NORM_SOURCE: &str = "src/cuda/h3_rms_norm_bf16.cu";
const H3_SDPA_SOFTMAX_SOURCE: &str = "src/cuda/h3_sdpa_softmax_f32.cu";
const QWEN_ATTENTION_SCALE_SOURCE: &str = "src/cuda/qwen_attention_scale_bf16.cu";
const QWEN_GELU_SOURCE: &str = "src/cuda/qwen_gelu_bf16.cu";
const QWEN_GELU_PTX: &str = "src/cuda/qwen_gelu_bf16_nvrtc130.ptx";
const QWEN_HEAD_MEAN_SOURCE: &str = "src/cuda/qwen_head_mean_f32.cu";
const QWEN_LAYER_NORM_SOURCE: &str = "src/cuda/qwen_layer_norm_bf16.cu";

/// The one virtual architecture every kernel's PTX is generated for.
///
/// This is the numerical contract and it does not vary by device. Compiling
/// the same source at a higher virtual architecture produces *different* PTX —
/// measurably so from `compute_90` upward, where NVCC unrolls differently and
/// selects different conversion instructions — which would fragment the
/// recorded identity across machines for no stated benefit. Instead the single
/// `compute_80` PTX is retargeted to real architectures by `ptxas` below, which
/// changes the SASS scheduling and not the arithmetic the PTX spells out.
const PTX_ARCHITECTURE: &str = "compute_80";

/// Architectures to translate the PTX to ahead of time.
///
/// Naming one is a build-time performance choice, never a correctness one: a
/// device with no matching cubin loads the `compute_80` PTX and the driver
/// compiles it, which is the same instructions by a different route.
const ARCHITECTURE_ENVIRONMENT_VARIABLE: &str = "FF_CUDA_ARCHS";

/// The floor, and the architecture the PTX itself is generated for.
const BASELINE_ARCHITECTURE: u32 = 80;

/// One kernel's emitted artifact set, as the generated manifest needs it.
struct ManifestEntry {
    kernel: &'static str,
    /// Architectures this build translated the kernel's PTX to.
    architectures: Vec<u32>,
}

/// Every kernel that gets an emitted artifact set, by artifact stem.
///
/// `qwen_gelu_bf16` is in the list even though it is staged from a pinned
/// NVRTC artifact rather than compiled here: it is retargeted by `ptxas` like
/// the rest, because that step reads PTX and does not care where it came from.
const KERNELS: [&str; 6] = [
    "h3_rms_norm_bf16",
    "h3_sdpa_softmax_f32",
    "qwen_attention_scale_bf16",
    "qwen_head_mean_f32",
    "qwen_layer_norm_bf16",
    "qwen_gelu_bf16",
];

fn compile_cuda(
    nvcc: &std::ffi::OsStr,
    out_dir: &std::path::Path,
    source: &str,
    output_name: &str,
) {
    let output_path = out_dir.join(output_name);
    let output = Command::new(nvcc)
        .args(["--ptx", "--std=c++17", "-O2"])
        .arg(format!("--gpu-architecture={PTX_ARCHITECTURE}"))
        .arg("--prec-div=true")
        .arg("--prec-sqrt=true")
        .arg("--ftz=false")
        .arg("-o")
        .arg(&output_path)
        .arg(source)
        .output()
        .unwrap_or_else(|error| {
            panic!(
                "failed to invoke {nvcc:?} for {source}; install the CUDA toolkit or set NVCC: {error}"
            )
        });
    if !output.status.success() {
        panic!(
            "{nvcc:?} failed to compile {source} for {PTX_ARCHITECTURE}:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn stage_pinned_qwen_gelu(out_dir: &std::path::Path) {
    std::fs::copy(QWEN_GELU_PTX, out_dir.join("qwen_gelu_bf16.ptx"))
        .unwrap_or_else(|error| panic!("failed to stage Qwen GELU PTX: {error}"));
}

/// The workspace root, in a form `nvcc` can be run from.
///
/// `canonicalize` returns a verbatim `\\?\` path on Windows, and parts of the
/// `nvcc` pipeline run through `cmd.exe`, which refuses to adopt one as its
/// working directory:
///
/// ```text
/// CMD.EXE was started with the above path as the current directory.
/// UNC paths are not supported.  Defaulting to Windows directory.
/// c1xx: fatal error C1083: Cannot open source file: '\\?\...\h3_rms_norm_bf16.cu'
/// ```
///
/// so every kernel fails to compile from a source file that plainly exists.
/// Only a verbatim *disk* path is unwrapped; a genuine UNC share keeps its
/// prefix, because there the prefix is the path rather than an encoding of it.
///
/// This cannot move a sealed PTX identity: `compile_cuda` passes `nvcc` the
/// workspace-relative source path, and it is that relative path — not the
/// working directory — that nvcc binds its anonymous-namespace symbols to.
fn stable_source_root(path: &std::path::Path) -> PathBuf {
    let canonical = path
        .canonicalize()
        .expect("missing workspace root for H3 native assets");
    #[cfg(windows)]
    {
        use std::path::{Component, Prefix};
        let verbatim_disk = matches!(
            canonical.components().next(),
            Some(Component::Prefix(prefix)) if matches!(prefix.kind(), Prefix::VerbatimDisk(_))
        );
        if verbatim_disk
            && let Some(plain) = canonical
                .to_str()
                .and_then(|text| text.strip_prefix(r"\\?\"))
        {
            return PathBuf::from(plain);
        }
    }
    canonical
}

/// `ptxas` beside the `nvcc` that produced the PTX, so both come from one
/// toolkit rather than whichever happens to be first on `PATH`.
fn resolve_ptxas(nvcc: &std::ffi::OsStr) -> std::ffi::OsString {
    if let Some(explicit) = std::env::var_os("PTXAS") {
        return explicit;
    }
    let nvcc_path = PathBuf::from(nvcc);
    if let Some(directory) = nvcc_path.parent() {
        let sibling = directory.join(if cfg!(windows) { "ptxas.exe" } else { "ptxas" });
        if sibling.is_file() {
            return sibling.into_os_string();
        }
    }
    "ptxas".into()
}

/// Which real architectures to translate the PTX to ahead of time.
///
/// `FF_CUDA_ARCHS=80,89,120` names them explicitly. Absent that, the build
/// asks the machine what it has, the way a `CMAKE_CUDA_ARCHITECTURES`-less
/// build does. Absent that too, only the baseline is emitted and every device
/// takes the driver's own translation of the same PTX.
fn resolve_target_architectures() -> Vec<u32> {
    let requested = std::env::var_os(ARCHITECTURE_ENVIRONMENT_VARIABLE);
    let mut architectures = if let Some(requested) = requested {
        let requested = requested.to_string_lossy().into_owned();
        requested
            .split(',')
            .map(str::trim)
            .filter(|entry| !entry.is_empty())
            .map(|entry| {
                entry.parse::<u32>().unwrap_or_else(|_| {
                    panic!(
                        "{ARCHITECTURE_ENVIRONMENT_VARIABLE} entry {entry:?} is not a compute \
                         capability written without its dot, such as 89 or 120"
                    )
                })
            })
            .collect()
    } else {
        detect_local_architectures()
    };
    architectures.push(BASELINE_ARCHITECTURE);
    architectures.sort_unstable();
    architectures.dedup();
    architectures
}

fn detect_local_architectures() -> Vec<u32> {
    let Ok(output) = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let (major, minor) = line.trim().split_once('.')?;
            let major: u32 = major.trim().parse().ok()?;
            let minor: u32 = minor.trim().parse().ok()?;
            Some(major * 10 + minor)
        })
        .collect()
}

/// Translate one already-compiled PTX to one real architecture.
///
/// Returns `None` when this toolkit cannot target the architecture, which is a
/// reason to emit fewer cubins and not a reason to fail the build: the PTX
/// remains loadable on that device through the driver.
fn assemble_cubin(
    ptxas: &std::ffi::OsStr,
    out_dir: &std::path::Path,
    kernel: &str,
    architecture: u32,
) -> bool {
    let input = out_dir.join(format!("{kernel}.ptx"));
    let output_path = out_dir.join(format!("{kernel}.sm_{architecture}.cubin"));
    let Ok(output) = Command::new(ptxas)
        .arg(format!("-arch=sm_{architecture}"))
        .arg("-O3")
        .arg(&input)
        .arg("-o")
        .arg(&output_path)
        .output()
    else {
        return false;
    };
    if !output.status.success() {
        println!(
            "cargo:warning=ptxas cannot target sm_{architecture} for {kernel}; that device will \
             load the {PTX_ARCHITECTURE} PTX instead: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
        let _ = std::fs::remove_file(&output_path);
        return false;
    }
    output_path.is_file()
}

/// Emit the table the runtime indexes by device architecture.
///
/// The architecture set is a build input, so it cannot be spelled as a fixed
/// list of `env!` constants the way a single architecture could.
fn write_kernel_manifest(out_dir: &std::path::Path, entries: &[ManifestEntry]) {
    let mut generated = String::new();
    generated.push_str(
        "// Generated by build.rs. The PTX is the numerical contract; each cubin is that\n\
         // same PTX translated to one architecture by ptxas.\n",
    );
    for ManifestEntry {
        kernel,
        architectures,
    } in entries
    {
        let identifier = kernel.to_uppercase();
        writeln!(
            generated,
            "pub(crate) static {identifier}: KernelAssets = KernelAssets {{\n    \
                 name: \"{kernel}\",\n    \
                 ptx: include_str!(concat!(env!(\"OUT_DIR\"), \"/{kernel}.ptx\")),\n    \
                 cubins: &["
        )
        .expect("string write");
        for architecture in architectures {
            writeln!(
                generated,
                "        Cubin {{ architecture: {architecture}, image: include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{kernel}.sm_{architecture}.cubin\")) }},"
            )
            .expect("string write");
        }
        generated.push_str("    ],\n};\n");
    }
    std::fs::write(out_dir.join("cuda_kernel_manifest.rs"), generated)
        .expect("failed to write the CUDA kernel manifest");
}

fn main() {
    let workspace_root =
        stable_source_root(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."));
    for source in [
        H3_RMS_NORM_SOURCE,
        H3_SDPA_SOFTMAX_SOURCE,
        QWEN_ATTENTION_SCALE_SOURCE,
        QWEN_GELU_SOURCE,
        QWEN_GELU_PTX,
        QWEN_HEAD_MEAN_SOURCE,
        QWEN_LAYER_NORM_SOURCE,
    ] {
        println!(
            "cargo:rerun-if-changed={}",
            workspace_root.join(source).display()
        );
    }
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=PTXAS");
    println!("cargo:rerun-if-env-changed={ARCHITECTURE_ENVIRONMENT_VARIABLE}");

    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    std::env::set_current_dir(&workspace_root).expect("cannot select stable H3 CUDA source root");

    let nvcc = std::env::var_os("NVCC").unwrap_or_else(|| "nvcc".into());
    let nvcc_version = Command::new(&nvcc)
        .arg("--version")
        .output()
        .unwrap_or_else(|error| panic!("failed to query {nvcc:?} version: {error}"));
    if !nvcc_version.status.success() {
        panic!(
            "{nvcc:?} --version failed:\n{}",
            String::from_utf8_lossy(&nvcc_version.stderr)
        );
    }
    let nvcc_version = String::from_utf8_lossy(&nvcc_version.stdout);
    const PINNED_NVCC_VERSION: &str = "13.2.86";
    let compiled_nvcc_version = nvcc_version
        .split_whitespace()
        .find_map(|token| token.strip_prefix('V'))
        .map(|version| version.trim_end_matches(',').to_owned())
        .unwrap_or_else(|| {
            panic!("could not read a version from {nvcc:?} --version output:\n{nvcc_version}")
        });
    if compiled_nvcc_version != PINNED_NVCC_VERSION {
        println!(
            "cargo:warning=building H3 CUDA kernels with nvcc {compiled_nvcc_version}, not the \
             pinned {PINNED_NVCC_VERSION}; this build selects the portable CUDA backend"
        );
    }
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
    compile_cuda(&nvcc, &out_dir, H3_RMS_NORM_SOURCE, "h3_rms_norm_bf16.ptx");
    compile_cuda(
        &nvcc,
        &out_dir,
        H3_SDPA_SOFTMAX_SOURCE,
        "h3_sdpa_softmax_f32.ptx",
    );
    compile_cuda(
        &nvcc,
        &out_dir,
        QWEN_ATTENTION_SCALE_SOURCE,
        "qwen_attention_scale_bf16.ptx",
    );
    compile_cuda(
        &nvcc,
        &out_dir,
        QWEN_HEAD_MEAN_SOURCE,
        "qwen_head_mean_f32.ptx",
    );
    compile_cuda(
        &nvcc,
        &out_dir,
        QWEN_LAYER_NORM_SOURCE,
        "qwen_layer_norm_bf16.ptx",
    );
    stage_pinned_qwen_gelu(&out_dir);

    let ptxas = resolve_ptxas(&nvcc);
    let architectures = resolve_target_architectures();
    let mut manifest_entries = Vec::new();
    for kernel in KERNELS {
        let translated = architectures
            .iter()
            .copied()
            .filter(|&architecture| assemble_cubin(&ptxas, &out_dir, kernel, architecture))
            .collect();
        manifest_entries.push(ManifestEntry {
            kernel,
            architectures: translated,
        });
    }
    let translated = architectures
        .iter()
        .map(|architecture| format!("sm_{architecture}"))
        .collect::<Vec<_>>()
        .join(",");
    println!("cargo:rustc-env=FLYINGFISH_CUDA_TARGET_ARCHITECTURES={translated}");
    write_kernel_manifest(&out_dir, &manifest_entries);

    println!("cargo:rustc-env=FLYINGFISH_H3_RMS_NORM_PTX_ARCH={PTX_ARCHITECTURE}");
    println!("cargo:rustc-env=FLYINGFISH_CUDA_NVCC_VERSION={compiled_nvcc_version}");

    println!("cargo:rustc-env=FLYINGFISH_QWEN_GELU_NVRTC_VERSION=13.0.88");
    println!(
        "cargo:rustc-env=FLYINGFISH_QWEN_GELU_NVRTC_SHA256=a49e67e8e74590f1e98de55c39c6287efd3f59e3c3797464d7bbe0fe01349b11"
    );
    println!(
        "cargo:rustc-env=FLYINGFISH_QWEN_GELU_NVRTC_BUILTINS_SHA256=91dcd944d01da9c0f08fff5d779db136a47f6d62bdb63bae900d2f481e92c3a2"
    );
}
