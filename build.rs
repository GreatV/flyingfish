use std::{env, fs, path::PathBuf, process::Command};

#[path = "src/build_arch.rs"]
mod build_arch;

fn architectures() -> Vec<u32> {
    let mut selected: Option<(String, Vec<u32>)> = None;
    for key in [
        "CUDA_ARCH_LIST",
        "TORCH_CUDA_ARCH_LIST",
        "CMAKE_CUDA_ARCHITECTURES",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
        let value = match env::var(key) {
            Ok(value) => Some(value),
            Err(env::VarError::NotPresent) => None,
            Err(error) => panic!("{key}: {error}"),
        };
        if let Some(value) = value {
            let archs = build_arch::parse(&value).unwrap_or_else(|e| panic!("{key}: {e}"));
            if let Some((previous, values)) = &selected {
                assert_eq!(
                    *values, archs,
                    "CUDA architecture variables disagree: {previous} and {key}"
                );
            } else {
                selected = Some((key.into(), archs));
            }
        }
    }
    let (source, archs) = if let Some(value) = selected {
        value
    } else {
        let output = Command::new("nvidia-smi")
            .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
            .output()
            .expect("cannot query CUDA architectures; set CUDA_ARCH_LIST");
        assert!(
            output.status.success(),
            "nvidia-smi architecture query failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text =
            String::from_utf8(output.stdout).expect("nvidia-smi returned non-UTF8 architectures");
        (
            "nvidia-smi".into(),
            build_arch::parse(&text)
                .unwrap_or_else(|e| panic!("nvidia-smi architecture query: {e}")),
        )
    };
    println!("cargo:warning=CUDA architectures={archs:?} source={source} native cubins only");
    archs
}

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    println!("cargo:rerun-if-changed=src/build_arch.rs");
    let archs = architectures();
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=PATH");
    let nvcc = if let Some(home) = env::var_os("CUDA_HOME") {
        PathBuf::from(home).join("bin/nvcc")
    } else {
        env::split_paths(&env::var_os("PATH").expect("PATH is missing"))
            .map(|p| p.join("nvcc"))
            .find(|p| p.is_file())
            .expect("nvcc is missing: set CUDA_HOME or PATH")
    };
    let nvcc = nvcc.canonicalize().expect("cannot resolve nvcc path");
    let version = Command::new(&nvcc)
        .arg("--version")
        .output()
        .expect("cannot execute nvcc");
    assert!(version.status.success(), "nvcc --version failed");
    println!(
        "cargo:warning=nvcc={} {}",
        nvcc.display(),
        String::from_utf8_lossy(&version.stdout).replace('\n', " | ")
    );
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is missing"));
    let temp = out.join("nvcc-tmp");
    fs::create_dir_all(&temp).expect("cannot create nvcc temporary directory");
    fn collect(dir: &std::path::Path, files: &mut Vec<PathBuf>) {
        for item in fs::read_dir(dir).expect("cannot read kernel directory") {
            let item = item.expect("cannot read kernel entry");
            let kind = item.file_type().expect("cannot read kernel type");
            assert!(!kind.is_symlink(), "kernel symlinks are unsupported");
            if kind.is_dir() {
                collect(&item.path(), files);
            } else if item.path().extension().is_some_and(|e| e == "cu") {
                files.push(item.path());
            }
        }
    }
    let mut files = Vec::new();
    collect(std::path::Path::new("kernels"), &mut files);
    files.sort();
    let mut index = format!(
        "pub const ARCHES: &[u32] = &{archs:?};\npub fn cubin(name: &str, sm: u32) -> Option<&'static [u8]> {{\nmatch (name, sm) {{\n"
    );
    for file in files {
        println!("cargo:rerun-if-changed={}", file.display());
        let name = file
            .strip_prefix("kernels")
            .expect("invalid kernel path")
            .with_extension("");
        let name = name
            .to_str()
            .expect("invalid kernel filename")
            .replace('\\', "/");
        for sm in &archs {
            let filename = format!("{name}_sm{sm}.cubin");
            let cubin = out.join(&filename);
            fs::create_dir_all(cubin.parent().expect("cubin has no parent"))
                .expect("cannot create cubin directory");
            let result = Command::new(&nvcc)
                .env("TMPDIR", &temp)
                .args(["--cubin", "--threads=1", "-O3", "--std=c++17"])
                .arg(format!("-arch=sm_{sm}"))
                .arg(&file)
                .arg("-o")
                .arg(&cubin)
                .output()
                .expect("cannot compile CUDA kernel");
            assert!(
                result.status.success(),
                "nvcc failed for {} sm_{sm}:\n{}\n{}",
                file.display(),
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
            index.push_str(&format!("({name:?}, {sm}) => Some(include_bytes!(concat!(env!(\"OUT_DIR\"), \"/\", {filename:?}))),\n"));
        }
    }
    index.push_str("_ => None,\n}\n}\n");
    fs::write(out.join("kernels.rs"), index).expect("cannot write cubin index");
    println!("cargo:rerun-if-changed=third_party/flash-attention/adapter.cu");
    println!("cargo:rerun-if-changed=third_party/flash-attention/src");
    println!("cargo:rerun-if-changed=third_party/cutlass/include");
    let archive = out.join("libff_fa2.a");
    let mut command = Command::new(&nvcc);
    command.env("TMPDIR", &temp).args([
        "--lib",
        "--threads=1",
        "-O3",
        "--std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-Xcompiler=-fPIC",
        "--use_fast_math",
        "-DFLASHATTENTION_DISABLE_DROPOUT",
        "-DFLASHATTENTION_DISABLE_ALIBI",
        "-DFLASHATTENTION_DISABLE_UNEVEN_K",
        "-DFLASHATTENTION_DISABLE_SOFTCAP",
        "-DFLASHATTENTION_DISABLE_LOCAL",
        "-Ithird_party/cutlass/include",
    ]);
    for sm in &archs {
        command
            .arg("--generate-code")
            .arg(format!("arch=compute_{sm},code=sm_{sm}"));
    }
    let result = command
        .args(["third_party/flash-attention/adapter.cu", "-o"])
        .arg(&archive)
        .output()
        .expect("cannot compile FA2");
    assert!(
        result.status.success(),
        "FA2 nvcc failed:\n{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let home = nvcc
        .parent()
        .and_then(|p| p.parent())
        .expect("nvcc toolkit directory missing");
    let lib = [home.join("lib64"), home.join("targets/x86_64-linux/lib")]
        .into_iter()
        .find(|p| p.join("libcudart_static.a").is_file())
        .expect("CUDA static runtime library missing");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=static=ff_fa2");
    println!("cargo:rustc-link-lib=static=cudart_static");
    for name in ["stdc++", "dl", "rt", "pthread"] {
        println!("cargo:rustc-link-lib={name}");
    }
}
