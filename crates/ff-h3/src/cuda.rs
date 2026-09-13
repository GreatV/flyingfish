//! CUDA execution with separate device, build, and reference-library checks.
//!
//! Repository kernels require compatible device instructions; the reference
//! vendor-library paths additionally check `profile`. FlashAttention requires
//! its own compiled feature and compatible CUDA device. Dispatch in `core` and
//! `multimodal_text_encoder` records the selected composition without claiming
//! reference-library results on other hosts.
//!
//! The CUDA sources these compile from stay at their established workspace
//! `src/cuda/*.cu` paths, which `build.rs` owns. Moving those files changes
//! NVCC's anonymous shared-memory symbol names and therefore the sealed PTX
//! identities; this Rust-side grouping deliberately leaves them untouched.

pub mod profile;

#[cfg(feature = "cuda")]
pub(crate) mod device;
#[cfg(feature = "cuda")]
pub(crate) mod kernel_assets;

#[cfg(feature = "cuda")]
pub(crate) mod linear;
#[cfg(feature = "cuda")]
pub(crate) mod qwen;
#[cfg(feature = "cuda")]
pub(crate) mod rms_norm;
#[cfg(feature = "cuda")]
pub(crate) mod sdpa_softmax;

/// Whether this device can execute the kernels compiled from `src/cuda/*.cu`.
///
/// Feature-independent so the execution policy, which is compiled without CUDA
/// as well, can record what a device afforded without knowing how this binary
/// was built.
#[cfg(feature = "cuda")]
pub fn tuned_kernels_available(device: &candle_core::Device) -> bool {
    device::tuned_kernels_available(device)
}

#[cfg(not(feature = "cuda"))]
pub fn tuned_kernels_available(_device: &candle_core::Device) -> bool {
    false
}

/// FlashAttention uses its own compiled CUDA kernels, independently of the
/// reference cuBLAS/cuDNN profile and the tuned-kernel comparison switch.
pub fn validate_flash_attention_device(device: &candle_core::Device) -> anyhow::Result<()> {
    anyhow::ensure!(device.is_cuda(), "FlashAttention requires a CUDA device");
    #[cfg(not(feature = "flash-attn"))]
    anyhow::bail!("FlashAttention was not compiled; rebuild with --features flash-attn");
    #[cfg(feature = "flash-attn")]
    {
        let cuda = device.as_cuda_device()?;
        let capability = device::compute_capability(cuda)?;
        anyhow::ensure!(
            capability >= (8, 0),
            "FlashAttention requires compute capability 8.0 or later, got {}.{}",
            capability.0,
            capability.1
        );
        profile::validate_selected_device(device)
    }
}
