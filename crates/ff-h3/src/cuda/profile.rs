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
pub const CANDLE_KERNELS_BUILD_FLAGS: &str =
    "cudaforge-build-ptx:--expt-relaxed-constexpr,-std=c++17,-O3;ug-disabled";
pub const CANDLE_GEMM_REDUCED_PRECISION_F32: bool = false;
pub const CANDLE_GEMM_REDUCED_PRECISION_F16: bool = false;
pub const CANDLE_GEMM_REDUCED_PRECISION_BF16: bool = false;
pub const CUBLAS_LIBRARY_BASENAME: &str = "libcublas.so.13.4.1.3";
pub const CUBLAS_LIBRARY_BYTES: u64 = 54_198_912;
pub const CUBLASLT_LIBRARY_BASENAME: &str = "libcublasLt.so.13.4.1.3";
pub const CUBLASLT_LIBRARY_BYTES: u64 = 508_260_928;
pub const CUDNN_VERSION: usize = 92_101;
pub const CUDNN_CUDART_VERSION: usize = 13_020;
pub const QWEN_PATCH_CUDNN_DSO_COUNT: usize = 8;
pub const QWEN_PATCH_CUDNN_DSOS: [&str; QWEN_PATCH_CUDNN_DSO_COUNT] = [
    "libcudnn.so.9.21.1",
    "libcudnn_cnn.so.9.21.1",
    "libcudnn_engines_precompiled.so.9.21.1",
    "libcudnn_engines_runtime_compiled.so.9.21.1",
    "libcudnn_engines_tensor_ir.so.9.21.1",
    "libcudnn_graph.so.9.21.1",
    "libcudnn_heuristic.so.9.21.1",
    "libcudnn_ops.so.9.21.1",
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
            // The loader resolves libcudnn.so.9.
            let version = unsafe { cudarc::cudnn::sys::cudnnGetVersion() };
            if version / 10_000 != 9 {
                return Err(format!("Qwen patch conv requires cuDNN 9.x, got {version}"));
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
            let stem = |name: &str| name.split(".so").next().unwrap_or(name).to_owned();
            let expected = QWEN_PATCH_CUDNN_DSOS
                .iter()
                .map(|name| stem(name))
                .collect::<BTreeSet<_>>();
            let maps = std::fs::read_to_string("/proc/self/maps")
                .map_err(|error| format!("failed to read /proc/self/maps: {error}"))?;
            let observed = maps
                .lines()
                .filter_map(|line| line.split_whitespace().last())
                .filter_map(|path| std::path::Path::new(path).file_name()?.to_str())
                .filter(|name| name.starts_with("libcudnn") && name.contains(".so"))
                .map(stem)
                .collect::<BTreeSet<_>>();
            if observed != expected {
                return Err(format!(
                    "mapped Qwen patch cuDNN components differ: observed={observed:?}, expected={expected:?}"
                ));
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

/// Validate process-wide workspace overrides and reduced-precision flags.
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

/// Validate the device and library profile for reference cuBLASLt and cuDNN operators.
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

/// Environment switch that disables the transcribed kernels.
pub const DISABLE_TUNED_KERNELS_ENVIRONMENT_VARIABLE: &str = "FF_CUDA_TUNED_KERNELS";

/// Whether the operator selected Candle kernels.
pub fn tuned_kernels_disabled() -> bool {
    static VALUE: std::sync::OnceLock<std::result::Result<usize, String>> =
        std::sync::OnceLock::new();
    ff_core::probe::cached_env_usize(&VALUE, DISABLE_TUNED_KERNELS_ENVIRONMENT_VARIABLE, 1, 0, 1)
        .expect("FF_CUDA_TUNED_KERNELS must be 0 or 1; supply a valid value and rerun ff")
        == 0
}

/// Whether this host provides the reference cuBLASLt and cuDNN builds.
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

/// Validate process-wide numerical settings for CUDA execution.
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
