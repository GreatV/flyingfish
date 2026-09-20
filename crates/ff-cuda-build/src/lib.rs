//! The shared CUDA kernel pipeline: one compute_80 PTX per kernel, plus the
//! cubins `ptxas` translates it to ahead of time.
//!
//! The PTX is the numerical contract and it does not vary by device. A cubin
//! is that same PTX translated for one real architecture; loading it skips the
//! driver's own translation, which also makes the kernel loadable on drivers
//! whose PTX ISA predates the build toolkit's.

use std::{fmt::Write as _, path::Path, path::PathBuf, process::Command};

/// The one virtual architecture every kernel's PTX is generated for.
pub const PTX_ARCHITECTURE: &str = "compute_80";

/// Architectures to translate the PTX to ahead of time.
///
/// Naming one is a build-time performance choice, never a correctness one: a
/// device with no matching cubin loads the `compute_80` PTX and the driver
/// compiles it, which is the same instructions by a different route.
pub const ARCHITECTURE_ENVIRONMENT_VARIABLE: &str = "FF_CUDA_ARCHS";

/// The floor, and the architecture the PTX itself is generated for.
pub const BASELINE_ARCHITECTURE: u32 = 80;

/// Every architecture a shipped binary is expected to load on.
///
/// A cubin is the only image immune to the toolkit-newer-than-driver PTX ISA
/// rejection, so the fleet — not the build host's own GPU — decides what a
/// release carries. `ptxas` skips architectures it cannot target, so a toolkit
/// older than the newest entry simply emits fewer cubins.
pub const FLEET_ARCHITECTURES: [u32; 6] = [80, 86, 89, 90, 100, 120];

/// One kernel's build inputs.
pub struct KernelSpec {
    /// Artifact stem: `<stem>.ptx`, `<stem>.sm_NN.cubin`, manifest static name.
    pub stem: &'static str,
    /// Kernel source, relative to the caller's source root.
    pub source: Option<&'static str>,
    /// Checked-in PTX to stage instead of compiling, relative to the source root.
    pub staged_ptx: Option<&'static str>,
    /// Extra nvcc flags, inserted between the shared flags and `-o`.
    pub extra_flags: &'static [&'static str],
}

/// What a build actually produced.
pub struct BuildReport {
    pub nvcc_version: String,
    /// Requested architectures, baseline included, sorted and deduped.
    pub target_architectures: Vec<u32>,
    /// Architectures ptxas actually translated.
    pub translated_architectures: Vec<u32>,
}

struct ManifestEntry {
    stem: &'static str,
    architectures: Vec<u32>,
}

/// The workspace-style root nvcc can be run from.
///
/// `canonicalize` returns a verbatim `\\?\` path on Windows, and parts of the
/// `nvcc` pipeline run through `cmd.exe`, which refuses to adopt one as its
/// working directory. Only a verbatim *disk* path is unwrapped; a genuine UNC
/// share keeps its prefix.
pub fn stable_source_root(path: &Path) -> PathBuf {
    let canonical = path
        .canonicalize()
        .expect("missing CUDA kernel source root");
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
pub fn resolve_ptxas(nvcc: &std::ffi::OsStr) -> std::ffi::OsString {
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
/// emits the union of the fleet list, `CUDA_COMPUTE_CAP` (the documented
/// GPU-less-build override, written 86 or 8.6), and whatever the build host
/// has, the way a `CMAKE_CUDA_ARCHITECTURES`-less build does. A device with
/// no matching cubin takes the driver's own translation of the same PTX.
pub fn resolve_target_architectures() -> Vec<u32> {
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
        let mut detected = detect_local_architectures();
        if let Some(cap) = std::env::var_os("CUDA_COMPUTE_CAP") {
            let cap = cap.to_string_lossy().into_owned();
            let parsed = if let Some((major, minor)) = cap.trim().split_once('.') {
                (
                    major.trim().parse::<u32>().ok(),
                    minor.trim().parse::<u32>().ok(),
                )
            } else {
                cap.trim()
                    .parse::<u32>()
                    .ok()
                    .map(|plain| (plain / 10, plain % 10))
                    .unzip()
            };
            if let (Some(major), Some(minor)) = parsed {
                detected.push(major * 10 + minor);
            }
        }
        detected.extend_from_slice(&FLEET_ARCHITECTURES);
        detected
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

/// The `V13.2.86`-style token from `nvcc --version`.
pub fn nvcc_version(nvcc: &std::ffi::OsStr) -> String {
    let output = Command::new(nvcc)
        .arg("--version")
        .output()
        .unwrap_or_else(|error| panic!("failed to query {nvcc:?} version: {error}"));
    if !output.status.success() {
        panic!(
            "{nvcc:?} --version failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .split_whitespace()
        .find_map(|token| token.strip_prefix('V'))
        .map(|version| version.trim_end_matches(',').to_owned())
        .unwrap_or_else(|| {
            panic!("could not read a version from {nvcc:?} --version output:\n{stdout}")
        })
}

fn compile_cuda(nvcc: &std::ffi::OsStr, out_dir: &Path, spec: &KernelSpec) {
    let source = spec.source.expect("compile_cuda needs a source");
    let output_path = out_dir.join(format!("{}.ptx", spec.stem));
    let mut command = Command::new(nvcc);
    command
        .args(["--ptx", "--std=c++17", "-O2"])
        .arg(format!("--gpu-architecture={PTX_ARCHITECTURE}"))
        .args(["--prec-div=true", "--prec-sqrt=true", "--ftz=false"])
        .args(spec.extra_flags)
        .arg("-o")
        .arg(&output_path)
        .arg(source);
    let output = command.output().unwrap_or_else(|error| {
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

fn stage_ptx(source_root: &Path, out_dir: &Path, spec: &KernelSpec) {
    let staged = spec.staged_ptx.expect("stage_ptx needs a staged PTX");
    std::fs::copy(
        source_root.join(staged),
        out_dir.join(format!("{}.ptx", spec.stem)),
    )
    .unwrap_or_else(|error| panic!("failed to stage {staged}: {error}"));
}

/// Translate one already-compiled PTX to one real architecture.
///
/// Returns `false` when this toolkit cannot target the architecture, which is
/// a reason to emit fewer cubins and not a reason to fail the build: the PTX
/// remains loadable on that device through the driver.
pub fn assemble_cubin(
    ptxas: &std::ffi::OsStr,
    out_dir: &Path,
    stem: &str,
    architecture: u32,
) -> bool {
    let input = out_dir.join(format!("{stem}.ptx"));
    let output_path = out_dir.join(format!("{stem}.sm_{architecture}.cubin"));
    let output = match Command::new(ptxas)
        .arg(format!("-arch=sm_{architecture}"))
        .arg("-O3")
        .arg(&input)
        .arg("-o")
        .arg(&output_path)
        .output()
    {
        Ok(output) => output,
        Err(error) => {
            println!(
                "cargo:warning=failed to execute {ptxas:?} for {stem} sm_{architecture}; that \
                 device will load the {PTX_ARCHITECTURE} PTX instead: {error}"
            );
            return false;
        }
    };
    if !output.status.success() {
        println!(
            "cargo:warning=ptxas cannot target sm_{architecture} for {stem}; that device will \
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
/// list of `env!` constants the way a single architecture could. The manifest
/// names `KernelAssets` and `Cubin`, which the including module must provide.
fn write_kernel_manifest(out_dir: &Path, entries: &[ManifestEntry]) {
    let mut generated = String::new();
    generated.push_str(
        "// Generated by build.rs. The PTX is the numerical contract; each cubin is that\n\
         // same PTX translated to one architecture by ptxas.\n",
    );
    for ManifestEntry {
        stem,
        architectures,
    } in entries
    {
        let identifier = stem.to_uppercase();
        writeln!(
            generated,
            "pub(crate) static {identifier}: KernelAssets = KernelAssets {{\n    \
                 name: \"{stem}\",\n    \
                 ptx: include_str!(concat!(env!(\"OUT_DIR\"), \"/{stem}.ptx\")),\n    \
                 cubins: &["
        )
        .expect("string write");
        for architecture in architectures {
            writeln!(
                generated,
                "        Cubin {{ architecture: {architecture}, image: include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{stem}.sm_{architecture}.cubin\")) }},"
            )
            .expect("string write");
        }
        generated.push_str("    ],\n};\n");
    }
    std::fs::write(out_dir.join("cuda_kernel_manifest.rs"), generated)
        .expect("failed to write the CUDA kernel manifest");
}

/// Compile or stage every kernel's PTX, translate it to the target
/// architectures, and write the manifest. Returns `None` when the caller's
/// `cuda` feature is off; rerun directives are printed either way.
pub fn run(specs: &[KernelSpec], source_root: &Path) -> Option<BuildReport> {
    let source_root = stable_source_root(source_root);
    for spec in specs {
        if let Some(source) = spec.source {
            println!(
                "cargo:rerun-if-changed={}",
                source_root.join(source).display()
            );
        }
        if let Some(staged) = spec.staged_ptx {
            println!(
                "cargo:rerun-if-changed={}",
                source_root.join(staged).display()
            );
        }
    }
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=PTXAS");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo:rerun-if-env-changed={ARCHITECTURE_ENVIRONMENT_VARIABLE}");

    std::env::var_os("CARGO_FEATURE_CUDA")?;

    std::env::set_current_dir(&source_root).expect("cannot select the CUDA kernel source root");

    let nvcc = std::env::var_os("NVCC").unwrap_or_else(|| "nvcc".into());
    let nvcc_version = nvcc_version(&nvcc);
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
    for spec in specs {
        match spec.source {
            Some(_) => compile_cuda(&nvcc, &out_dir, spec),
            None => stage_ptx(&source_root, &out_dir, spec),
        }
    }

    let ptxas = resolve_ptxas(&nvcc);
    let architectures = resolve_target_architectures();
    let mut manifest_entries = Vec::new();
    let mut translated_all = Vec::new();
    for spec in specs {
        let translated: Vec<u32> = architectures
            .iter()
            .copied()
            .filter(|&architecture| assemble_cubin(&ptxas, &out_dir, spec.stem, architecture))
            .collect();
        translated_all.extend(translated.iter().copied());
        manifest_entries.push(ManifestEntry {
            stem: spec.stem,
            architectures: translated,
        });
    }
    let requested = architectures
        .iter()
        .map(|architecture| format!("sm_{architecture}"))
        .collect::<Vec<_>>()
        .join(",");
    println!("cargo:rustc-env=FLYINGFISH_CUDA_TARGET_ARCHITECTURES={requested}");
    write_kernel_manifest(&out_dir, &manifest_entries);
    translated_all.sort_unstable();
    translated_all.dedup();
    Some(BuildReport {
        nvcc_version,
        target_architectures: architectures,
        translated_architectures: translated_all,
    })
}
