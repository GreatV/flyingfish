//! Exact CUDA runtime profile shared by all H3 components.

use anyhow::Result;
use candle_core::Device;

pub const DRIVER_VERSION: &str = "595.84";
pub const CUBLAS_VERSION: i32 = 130_401;
pub const CUBLASLT_VERSION: usize = 130_401;
pub const COMPUTE_CAPABILITY: (i32, i32) = (8, 9);
pub const MULTIPROCESSOR_COUNT: i32 = 128;
pub const CUBLAS_ENVIRONMENT_VARIABLES: [&str; 3] = [
    "CUBLAS_WORKSPACE_CONFIG",
    "CUBLASLT_WORKSPACE_SIZE",
    "TORCH_CUBLASLT_UNIFIED_WORKSPACE",
];
#[cfg(feature = "cuda")]
pub const NVCC_VERSION: &str = env!("FLYINGFISH_CUDA_NVCC_VERSION");
pub const CANDLE_KERNELS_CRATE_VERSION: &str = "0.11.0";
pub const CANDLE_KERNELS_CRATE_CHECKSUM: &str =
    "67450168a281bbb195a14cc85cf164c7a12023a54a03409a3324f476346e0789";
pub const CANDLE_KERNELS_BUILD_FLAGS: &str =
    "cudaforge-build-ptx:--expt-relaxed-constexpr,-std=c++17,-O3;ug-disabled";
pub const CANDLE_KERNELS_EMBEDDED_PTX_SHA256: &str =
    "76ab472f7e3cf5feba6bc10f8379cc8ad3936555c0efd4ee54c92548f43d9000";
pub const CANDLE_KERNELS_BACKEND: &str = "candle-kernels-0.11.0/crate-sha256:67450168a281bbb195a14cc85cf164c7a12023a54a03409a3324f476346e0789/nvcc-13.2.86/O3/embedded-ptx-manifest-sha256:76ab472f7e3cf5feba6bc10f8379cc8ad3936555c0efd4ee54c92548f43d9000/ug-disabled";
pub const CANDLE_GEMM_REDUCED_PRECISION_F32: bool = false;
pub const CANDLE_GEMM_REDUCED_PRECISION_F16: bool = false;
pub const CANDLE_GEMM_REDUCED_PRECISION_BF16: bool = false;
pub const CUBLAS_LIBRARY_BASENAME: &str = "libcublas.so.13.4.1.3";
pub const CUBLAS_LIBRARY_BYTES: u64 = 54_198_912;
pub const CUBLAS_LIBRARY_SHA256: &str =
    "d089edf0a70ba75f0f422927da121f59018247f491741a3bf0b20637ed7c7b61";
pub const CUBLASLT_LIBRARY_BASENAME: &str = "libcublasLt.so.13.4.1.3";
pub const CUBLASLT_LIBRARY_BYTES: u64 = 508_260_928;
pub const CUBLASLT_LIBRARY_SHA256: &str =
    "8d702c94d90bd4bd0c032fd4207ae655b4a2d036bd31920c1fbf4ecaf3547c59";
pub const CUDNN_VERSION: usize = 92_101;
pub const CUDNN_CUDART_VERSION: usize = 13_020;
pub const QWEN_PATCH_CUDNN_DSO_COUNT: usize = 8;
pub const QWEN_PATCH_CUDNN_DSO_MANIFEST_SHA256: &str =
    "e9a3eef4b024a94ef9469136df4ad8b8aba74bba7f195945e30ac99b317c2930";
pub const QWEN_PATCH_CUDNN_DSOS: [(&str, u64, &str); QWEN_PATCH_CUDNN_DSO_COUNT] = [
    (
        "libcudnn.so.9.21.1",
        129_240,
        "29693d4cb390d13c48fa800b585b6f524678680c3ab918f9af5575efff68766a",
    ),
    (
        "libcudnn_cnn.so.9.21.1",
        2_468_752,
        "58473cb364769723736cef291db78045dcb35defd9da945391beff94debe8dfd",
    ),
    (
        "libcudnn_engines_precompiled.so.9.21.1",
        246_182_656,
        "0e42d76d4931da6e394a2956544f61f0b36527c0c6fdb5f4fee894f87f3808cb",
    ),
    (
        "libcudnn_engines_runtime_compiled.so.9.21.1",
        30_031_888,
        "b4e5ecad078e0a394df888eb1aa8ffed1c9cd0261150de93ce37c0b2d6bf3049",
    ),
    (
        "libcudnn_engines_tensor_ir.so.9.21.1",
        242_382_936,
        "9149d5cbcf55b25dceacd1919a5623cd6640498214733cb6a3fc1e07bbdf7d4b",
    ),
    (
        "libcudnn_graph.so.9.21.1",
        123_829_576,
        "406445a88232425093da4c3f158d6343022bbbdb6584f571dad56071fd3baa03",
    ),
    (
        "libcudnn_heuristic.so.9.21.1",
        62_142_480,
        "70c929e776269cf25ecbc2f95d5127361f7c95b6ba85d299a02df18c712e9d57",
    ),
    (
        "libcudnn_ops.so.9.21.1",
        38_010_464,
        "adccaa0e31f49e551e8bf25c55f5d732c742b4cd598a136d734a89af52f75444",
    ),
];

#[cfg(feature = "cuda")]
fn mapped_library_path(basename: &str) -> std::result::Result<std::path::PathBuf, String> {
    use std::collections::BTreeSet;

    let maps = std::fs::read_to_string("/proc/self/maps")
        .map_err(|error| format!("failed to read /proc/self/maps: {error}"))?;
    let mut paths = BTreeSet::new();
    for line in maps.lines() {
        let Some(raw_path) = line.split_whitespace().last() else {
            continue;
        };
        let path = std::path::Path::new(raw_path);
        if path.file_name().and_then(|name| name.to_str()) == Some(basename) {
            paths.insert(
                path.canonicalize()
                    .map_err(|error| format!("failed to resolve mapped {basename}: {error}"))?,
            );
        }
    }
    if paths.len() != 1 {
        return Err(format!(
            "expected exactly one mapped {basename}, observed {:?}",
            paths
        ));
    }
    Ok(paths.into_iter().next().expect("one mapped path"))
}

#[cfg(feature = "cuda")]
fn validate_mapped_library(basename: &str, expected_bytes: u64) -> std::result::Result<(), String> {
    let path = mapped_library_path(basename)?;
    validate_library_path(&path, basename, expected_bytes)
}

#[cfg(any(feature = "cuda", test))]
fn validate_library_path(
    path: &std::path::Path,
    basename: &str,
    expected_bytes: u64,
) -> std::result::Result<(), String> {
    let metadata = path
        .metadata()
        .map_err(|error| format!("failed to stat mapped {basename}: {error}"))?;
    if metadata.len() != expected_bytes {
        return Err(format!(
            "mapped {basename} has {} bytes, expected {expected_bytes}",
            metadata.len()
        ));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn validate_qwen_patch_cudnn_preflight() -> candle_core::Result<()> {
    use std::sync::OnceLock;

    static VALIDATED: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    VALIDATED
        .get_or_init(|| {
            let version = unsafe { cudarc::cudnn::sys::cudnnGetVersion() };
            if version != CUDNN_VERSION {
                return Err(format!(
                    "exact Qwen patch profile requires cuDNN {CUDNN_VERSION}, got {version}"
                ));
            }
            let cudart_version = unsafe { cudarc::cudnn::sys::cudnnGetCudartVersion() };
            if cudart_version != CUDNN_CUDART_VERSION {
                return Err(format!(
                    "exact Qwen patch profile requires cuDNN CUDART {CUDNN_CUDART_VERSION}, got {cudart_version}"
                ));
            }
            let core = mapped_library_path(QWEN_PATCH_CUDNN_DSOS[0].0)?;
            let parent = core
                .parent()
                .ok_or_else(|| "mapped cuDNN core library has no parent".to_owned())?;
            for (basename, bytes, _) in QWEN_PATCH_CUDNN_DSOS {
                let path = parent.join(basename);
                validate_library_path(&path, basename, bytes)?;
            }
            Ok(())
        })
        .clone()
        .map_err(candle_core::Error::Msg)
}

#[cfg(feature = "cuda")]
pub(crate) fn validate_qwen_patch_loaded_cudnn() -> candle_core::Result<()> {
    use std::{collections::BTreeSet, sync::OnceLock};

    static VALIDATED: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    VALIDATED
        .get_or_init(|| {
            let expected = QWEN_PATCH_CUDNN_DSOS
                .iter()
                .map(|(name, _, _)| (*name).to_owned())
                .collect::<BTreeSet<_>>();
            let maps = std::fs::read_to_string("/proc/self/maps")
                .map_err(|error| format!("failed to read /proc/self/maps: {error}"))?;
            let observed = maps
                .lines()
                .filter_map(|line| line.split_whitespace().last())
                .filter_map(|path| std::path::Path::new(path).file_name()?.to_str())
                .filter(|name| name.starts_with("libcudnn") && name.contains(".so"))
                .map(ToOwned::to_owned)
                .collect::<BTreeSet<_>>();
            if observed != expected {
                return Err(format!(
                    "mapped Qwen patch cuDNN libraries differ: observed={observed:?}, expected={expected:?}"
                ));
            }
            for (basename, bytes, _) in QWEN_PATCH_CUDNN_DSOS {
                validate_mapped_library(basename, bytes)?;
            }
            Ok(())
        })
        .clone()
        .map_err(candle_core::Error::Msg)
}

#[cfg(any(feature = "cuda", test))]
fn validate_workspace_environment_values(
    values: [Option<&std::ffi::OsStr>; 3],
) -> std::result::Result<(), String> {
    for (name, value) in CUBLAS_ENVIRONMENT_VARIABLES.into_iter().zip(values) {
        if let Some(value) = value {
            return Err(format!(
                "exact H3/Qwen CUDA profile requires {name} to be unset, got {value:?}"
            ));
        }
    }
    Ok(())
}

#[cfg(any(feature = "cuda", test))]
fn validate_gemm_reduced_precision_values(
    f32_reduced: bool,
    f16_reduced: bool,
    bf16_reduced: bool,
) -> std::result::Result<(), String> {
    let observed = (f32_reduced, f16_reduced, bf16_reduced);
    let expected = (
        CANDLE_GEMM_REDUCED_PRECISION_F32,
        CANDLE_GEMM_REDUCED_PRECISION_F16,
        CANDLE_GEMM_REDUCED_PRECISION_BF16,
    );
    if observed != expected {
        return Err(format!(
            "exact H3/Qwen CUDA profile requires Candle GEMM reduced-precision flags f32/f16/bf16={expected:?}, got {observed:?}"
        ));
    }
    Ok(())
}

#[cfg(any(feature = "cuda", test))]
fn validate_profile_values(
    major: i32,
    minor: i32,
    multiprocessors: i32,
    cublas_version: i32,
    cublaslt_version: usize,
    driver_version_text: Option<&str>,
) -> std::result::Result<(), String> {
    if (major, minor) != COMPUTE_CAPABILITY {
        return Err(format!(
            "exact H3 CUDA profile requires compute capability {}.{}, got {major}.{minor}",
            COMPUTE_CAPABILITY.0, COMPUTE_CAPABILITY.1
        ));
    }
    if multiprocessors != MULTIPROCESSOR_COUNT {
        return Err(format!(
            "exact H3 CUDA profile requires {MULTIPROCESSOR_COUNT} multiprocessors, got {multiprocessors}"
        ));
    }
    if cublas_version != CUBLAS_VERSION {
        return Err(format!(
            "exact H3 CUDA profile requires cuBLAS {CUBLAS_VERSION}, got {cublas_version}"
        ));
    }
    if cublaslt_version != CUBLASLT_VERSION {
        return Err(format!(
            "exact H3 CUDA profile requires cuBLASLt {CUBLASLT_VERSION}, got {cublaslt_version}"
        ));
    }
    let Some(driver_version_text) = driver_version_text else {
        return Err(format!(
            "exact H3 CUDA profile requires NVIDIA driver {DRIVER_VERSION} identified through /proc/driver/nvidia/version, which this platform does not provide"
        ));
    };
    if !driver_version_text.contains(&format!("  {DRIVER_VERSION}  ")) {
        return Err(format!(
            "exact H3 CUDA profile requires NVIDIA driver {DRIVER_VERSION}"
        ));
    }
    Ok(())
}

/// Numerical state of this process, which no device attribute can supply.
///
/// Environment overrides and reduced-precision flags affect the selected
/// arithmetic on every host. Kernel and library digests do not gate execution.
#[cfg(feature = "cuda")]
pub(crate) fn validate_process_numerics() -> candle_core::Result<()> {
    use std::sync::OnceLock;

    static VALIDATED: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    VALIDATED
        .get_or_init(|| {
            let environment = CUBLAS_ENVIRONMENT_VARIABLES.map(std::env::var_os);
            validate_workspace_environment_values(environment.each_ref().map(Option::as_deref))?;
            validate_gemm_reduced_precision_values(
                candle_core::cuda_backend::gemm_reduced_precision_f32(),
                candle_core::cuda_backend::gemm_reduced_precision_f16(),
                candle_core::cuda_backend::gemm_reduced_precision_bf16(),
            )?;
            Ok(())
        })
        .clone()
        .map_err(candle_core::Error::Msg)
}

/// Whether the reference vendor-library profile covers this device.
///
/// Unlike [`validate_process_numerics`] this really is a question about one
/// machine, and it governs exactly the operators that call into a vendor
/// library: the cuBLASLt biased linear and QK/PV matmuls, and the cuDNN patch
/// convolution. Those pick different kernels on different architectures and
/// different library builds, so their recorded identity is a claim about a
/// host. The kernels compiled from this repository's own sources make no such
/// claim and do not consult this.
#[cfg(feature = "cuda")]
pub(crate) fn validate_cuda_device(
    device: &candle_core::cuda_backend::CudaDevice,
) -> candle_core::Result<()> {
    use candle_core::cuda_backend::DeviceId;
    use cudarc::driver::sys::CUdevice_attribute;
    use std::{
        collections::HashSet,
        sync::{Mutex, OnceLock},
    };

    validate_process_numerics()?;

    static VALIDATED: OnceLock<Mutex<HashSet<DeviceId>>> = OnceLock::new();
    let validated = VALIDATED.get_or_init(|| Mutex::new(HashSet::new()));
    let mut validated = validated
        .lock()
        .map_err(|_| candle_core::Error::Msg("H3 CUDA profile cache lock is poisoned".into()))?;
    if validated.contains(&device.id()) {
        return Ok(());
    }

    let stream = device.cuda_stream();
    let attribute = |attribute| {
        stream.context().attribute(attribute).map_err(|error| {
            candle_core::Error::Msg(format!(
                "failed to inspect CUDA device attribute: {error:?}"
            ))
        })
    };
    let major = attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
    let minor = attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
    let multiprocessors = attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
    let blas = cudarc::cublas::CudaBlas::new(stream.clone()).map_err(|error| {
        candle_core::Error::Msg(format!("failed to create cuBLAS handle: {error:?}"))
    })?;
    let mut cublas_version = 0i32;
    let cublas_status =
        unsafe { cudarc::cublas::sys::cublasGetVersion_v2(*blas.handle(), &mut cublas_version) };
    if cublas_status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        candle_core::bail!("failed to query cuBLAS version: {cublas_status:?}")
    }
    let cublaslt_version = unsafe { cudarc::cublaslt::sys::cublasLtGetVersion() };
    #[cfg(target_os = "linux")]
    let driver_version_text = Some(
        std::fs::read_to_string("/proc/driver/nvidia/version").map_err(|error| {
            candle_core::Error::Msg(format!("failed to read NVIDIA driver version: {error}"))
        })?,
    );
    #[cfg(not(target_os = "linux"))]
    let driver_version_text: Option<String> = None;
    validate_profile_values(
        major,
        minor,
        multiprocessors,
        cublas_version,
        cublaslt_version,
        driver_version_text.as_deref(),
    )
    .map_err(candle_core::Error::Msg)?;
    validate_mapped_library(CUBLAS_LIBRARY_BASENAME, CUBLAS_LIBRARY_BYTES)
        .map_err(candle_core::Error::Msg)?;
    validate_mapped_library(CUBLASLT_LIBRARY_BASENAME, CUBLASLT_LIBRARY_BYTES)
        .map_err(candle_core::Error::Msg)?;
    validated.insert(device.id());
    Ok(())
}

/// Environment switch that turns the transcribed kernels off.
///
/// It exists so the Candle composition stays reachable for comparison on a
/// machine the kernels do cover, which is the only way to measure what they are
/// worth. It is not a correctness escape: both compositions execute.
pub const DISABLE_TUNED_KERNELS_ENVIRONMENT_VARIABLE: &str = "FF_CUDA_TUNED_KERNELS";

/// Whether the operator has asked for Candle's kernels outright.
///
/// Device-independent, so commands that model a run without opening a device —
/// `h3 plan` and the solver — can honour the same switch as execution does.
pub fn tuned_kernels_disabled() -> bool {
    std::env::var_os(DISABLE_TUNED_KERNELS_ENVIRONMENT_VARIABLE).is_some_and(|value| value == "0")
}

/// Whether this host's cuBLASLt and cuDNN are the reference builds.
///
/// A `false` answer is not an error and is not a statement about the device's
/// capability: it says the operators that call those libraries take Candle's
/// composition instead, because reproducing a recorded vendor-library result
/// is a claim about one host. The kernels compiled from this repository do not
/// consult this.
#[cfg(feature = "cuda")]
pub fn reference_libraries_available(device: &Device) -> bool {
    use candle_core::cuda_backend::DeviceId;
    use std::{
        collections::HashMap,
        sync::{Mutex, OnceLock},
    };

    if tuned_kernels_disabled() {
        return false;
    }
    let Device::Cuda(cuda) = device else {
        return false;
    };
    static PROBED: OnceLock<Mutex<HashMap<DeviceId, bool>>> = OnceLock::new();
    let probed = PROBED.get_or_init(|| Mutex::new(HashMap::new()));
    let Ok(mut probed) = probed.lock() else {
        return false;
    };
    if let Some(available) = probed.get(&cuda.id()) {
        return *available;
    }
    let available = validate_cuda_device(cuda).is_ok();
    probed.insert(cuda.id(), available);
    available
}

#[cfg(not(feature = "cuda"))]
pub fn reference_libraries_available(_device: &Device) -> bool {
    false
}

/// Entry-point check for a device a command is about to use.
///
/// Only the process-level numerical state, which applies to every CUDA run
/// whatever the device. The reference-library identities are checked by the
/// operators that actually call those libraries, so a host without them runs
/// the rest rather than being refused here.
pub fn validate_selected_device(device: &Device) -> Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    #[cfg(feature = "cuda")]
    validate_process_numerics()?;
    Ok(())
}

#[cfg(feature = "cuda")]
pub fn validate_exact_profile(device: &Device) -> Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    if let Device::Cuda(device) = device {
        validate_cuda_device(device).map_err(anyhow::Error::from)?;
    }
    Ok(())
}

pub fn validate_tensor_runtime_artifacts(device: &Device) -> Result<()> {
    validate_exact_profile(device)
}

#[cfg(not(feature = "cuda"))]
pub fn validate_exact_profile(device: &Device) -> Result<()> {
    anyhow::ensure!(
        !device.is_cuda(),
        "H3 exact CUDA profile validation requires the cuda build feature"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_profile_rejects_unverified_hardware_and_runtime() {
        let driver = "NVRM version: NVIDIA UNIX Open Kernel Module  595.84  Release Build";
        validate_profile_values(8, 9, 128, 130_401, 130_401, Some(driver)).unwrap();
        for (major, minor, sms, cublas, cublaslt, driver, expected) in [
            (
                8,
                6,
                128,
                130_401,
                130_401,
                Some(driver),
                "compute capability",
            ),
            (8, 9, 114, 130_401, 130_401, Some(driver), "multiprocessors"),
            (8, 9, 128, 130_101, 130_401, Some(driver), "cuBLAS 130401"),
            (8, 9, 128, 130_401, 130_101, Some(driver), "cuBLASLt"),
            (
                8,
                9,
                128,
                130_401,
                130_401,
                Some("  600.00  "),
                "driver 595.84",
            ),
            (
                8,
                9,
                128,
                130_401,
                130_401,
                None,
                "which this platform does not provide",
            ),
        ] {
            let error =
                validate_profile_values(major, minor, sms, cublas, cublaslt, driver).unwrap_err();
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn exact_profile_rejects_workspace_environment_overrides() {
        validate_workspace_environment_values([None, None, None]).unwrap();
        for (index, name) in CUBLAS_ENVIRONMENT_VARIABLES.into_iter().enumerate() {
            let mut values = [None, None, None];
            values[index] = Some(std::ffi::OsStr::new("override"));
            let error = validate_workspace_environment_values(values).unwrap_err();
            assert!(error.contains(name));
            assert!(error.contains("unset"));
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            let non_utf8 = std::ffi::OsString::from_vec(vec![0xff]);
            assert!(
                validate_workspace_environment_values([Some(non_utf8.as_os_str()), None, None,])
                    .is_err()
            );
        }
    }

    #[test]
    fn library_validation_checks_size_without_reading_contents() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("library.so");
        std::fs::write(&path, b"first").unwrap();
        validate_library_path(&path, "library.so", 5).unwrap();
        std::fs::write(&path, b"other").unwrap();
        validate_library_path(&path, "library.so", 5).unwrap();
        assert!(validate_library_path(&path, "library.so", 6).is_err());
        std::fs::remove_file(&path).unwrap();
        assert!(validate_library_path(&path, "library.so", 5).is_err());
    }

    #[test]
    fn exact_profile_rejects_any_candle_gemm_reduced_precision_flag() {
        validate_gemm_reduced_precision_values(false, false, false).unwrap();
        for (f32_reduced, f16_reduced, bf16_reduced) in [
            (true, false, false),
            (false, true, false),
            (false, false, true),
        ] {
            let error =
                validate_gemm_reduced_precision_values(f32_reduced, f16_reduced, bf16_reduced)
                    .unwrap_err();
            assert!(error.contains("f32/f16/bf16"));
        }
    }
}
