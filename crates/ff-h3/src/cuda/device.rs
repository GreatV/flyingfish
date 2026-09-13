//! What a device can run, as distinct from which host it happens to be.
//!
//! The pinned profile in [`super::profile`] answers "is this the machine the
//! recorded numbers were produced on". That is a question about evidence. This
//! module answers "can this GPU execute these instructions", which is a
//! question about hardware, and the two have different answers on every host
//! but one.

use candle_core::cuda_backend::CudaDevice;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Compute capability of a CUDA device, cached per device.
///
/// Cached because it is queried on the first launch of every kernel and the
/// driver call is not free, and because it cannot change for a live device.
pub(crate) fn compute_capability(device: &CudaDevice) -> candle_core::Result<(i32, i32)> {
    use candle_core::cuda_backend::DeviceId;
    use cudarc::driver::sys::CUdevice_attribute;

    static PROBED: OnceLock<Mutex<HashMap<DeviceId, (i32, i32)>>> = OnceLock::new();
    let probed = PROBED.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let probed = probed
            .lock()
            .map_err(|_| candle_core::Error::Msg("CUDA capability cache is poisoned".into()))?;
        if let Some(capability) = probed.get(&device.id()) {
            return Ok(*capability);
        }
    }
    let context = device.cuda_stream().context().clone();
    let attribute = |attribute| {
        context.attribute(attribute).map_err(|error| {
            candle_core::Error::Msg(format!(
                "failed to inspect CUDA device attribute: {error:?}"
            ))
        })
    };
    let capability = (
        attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
        attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
    );
    let mut probed = probed
        .lock()
        .map_err(|_| candle_core::Error::Msg("CUDA capability cache is poisoned".into()))?;
    probed.insert(device.id(), capability);
    Ok(capability)
}

/// The lowest compute capability the transcribed kernels can run on.
///
/// It is the virtual architecture their PTX is generated for, and it is set by
/// the instructions the sources actually use: BF16 conversion, and the inline
/// `cvt.rn.bf16.f32` in the GELU and attention-scale kernels. Nothing in them
/// needs a later architecture, so nothing later is required.
pub(crate) const MINIMUM_COMPUTE_CAPABILITY: (i32, i32) = (8, 0);

/// Whether this device can execute the transcribed kernels.
///
/// This is the whole requirement for them. It deliberately says nothing about
/// the driver build, the multiprocessor count, or which vendor libraries are
/// installed: those govern the operators that call into cuBLASLt and cuDNN,
/// not the ones compiled from this repository's own sources.
pub(crate) fn tuned_kernels_available(device: &candle_core::Device) -> bool {
    if super::profile::tuned_kernels_disabled() {
        return false;
    }
    let candle_core::Device::Cuda(cuda) = device else {
        return false;
    };
    compute_capability(cuda).is_ok_and(|capability| capability >= MINIMUM_COMPUTE_CAPABILITY)
}

/// Refuse a device that cannot execute the transcribed kernels.
///
/// Called at the head of every kernel compiled from `src/cuda/*.cu`.
pub(crate) fn require_tuned_kernel(device: &CudaDevice) -> candle_core::Result<()> {
    super::profile::validate_process_numerics()?;
    let capability = compute_capability(device)?;
    if capability < MINIMUM_COMPUTE_CAPABILITY {
        candle_core::bail!(
            "the transcribed H3 CUDA kernels require compute capability {}.{} or later, got {}.{}",
            MINIMUM_COMPUTE_CAPABILITY.0,
            MINIMUM_COMPUTE_CAPABILITY.1,
            capability.0,
            capability.1
        )
    }
    Ok(())
}
