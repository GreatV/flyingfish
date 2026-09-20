//! The compiled images of a crate's CUDA kernels, and the choice between them.
//!
//! One PTX per kernel is the numerical contract: the crate's `build.rs`
//! generates it at `compute_80` and nothing about the running device changes
//! it. A cubin is the same PTX already translated by `ptxas` for one
//! architecture; loading it skips the driver's own translation, which rejects
//! a PTX whose ISA version postdates the driver. When no cubin matches, the
//! PTX is loaded and the driver translates it.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use candle_core::cuda_backend::cudarc::driver::{CudaFunction, CudaModule};
use candle_core::cuda_backend::{CudaDevice, DeviceId};

/// One architecture's translation of a kernel's PTX.
pub struct Cubin {
    pub architecture: u32,
    pub image: &'static [u8],
}

/// Every compiled form of one kernel.
pub struct KernelAssets {
    pub name: &'static str,
    pub ptx: &'static str,
    pub cubins: &'static [Cubin],
}

/// Which compiled form a device will load.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageSelection {
    /// A cubin built for exactly this architecture.
    Cubin { architecture: u32 },
    /// The `compute_80` PTX, translated by the driver for this device.
    Ptx,
}

impl KernelAssets {
    /// Pick the image for a device of this compute capability.
    ///
    /// Only an exact architecture match is used. A cubin is not forward
    /// compatible — `sm_89` code does not load on `sm_90` — so a near miss has
    /// to fall through to the PTX rather than to the closest cubin.
    pub fn select(&self, compute_capability: (i32, i32)) -> ImageSelection {
        let architecture = compute_capability
            .0
            .saturating_mul(10)
            .saturating_add(compute_capability.1);
        let Ok(architecture) = u32::try_from(architecture) else {
            return ImageSelection::Ptx;
        };
        match self
            .cubins
            .iter()
            .find(|cubin| cubin.architecture == architecture)
        {
            Some(cubin) => ImageSelection::Cubin {
                architecture: cubin.architecture,
            },
            None => ImageSelection::Ptx,
        }
    }

    /// The image bytes for a selection.
    fn image(&self, selection: ImageSelection) -> candle_core::Result<&'static [u8]> {
        match selection {
            ImageSelection::Ptx => Ok(self.ptx.as_bytes()),
            ImageSelection::Cubin { architecture } => self
                .cubins
                .iter()
                .find(|cubin| cubin.architecture == architecture)
                .map(|cubin| cubin.image)
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "kernel {} has no cubin for sm_{architecture}",
                        self.name
                    ))
                }),
        }
    }
}

/// Compute capability of a CUDA device, cached per device.
///
/// Cached because it is queried on the first launch of every kernel and the
/// driver call is not free, and because it cannot change for a live device.
pub fn compute_capability(device: &CudaDevice) -> candle_core::Result<(i32, i32)> {
    use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute;

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

type ModuleKey = (DeviceId, &'static str);

fn module_cache() -> &'static Mutex<HashMap<ModuleKey, Arc<CudaModule>>> {
    static CACHE: OnceLock<Mutex<HashMap<ModuleKey, Arc<CudaModule>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Load one function out of a kernel's best available image for this device.
///
/// Modules are cached per device, as Candle caches its own, so the translation
/// happens once per process rather than once per launch.
pub fn load_function(
    device: &CudaDevice,
    assets: &'static KernelAssets,
    module_name: &'static str,
    function_name: &str,
) -> candle_core::Result<CudaFunction> {
    use candle_core::cuda_backend::WrapErr;

    let key = (device.id(), module_name);
    let cached = {
        let cache = module_cache()
            .lock()
            .map_err(|_| candle_core::Error::Msg("CUDA kernel module cache is poisoned".into()))?;
        cache.get(&key).cloned()
    };
    let module = match cached {
        Some(module) => module,
        None => {
            let selection = assets.select(compute_capability(device)?);
            let bytes = assets.image(selection)?;
            let image = match selection {
                ImageSelection::Cubin { .. } => {
                    candle_core::cuda_backend::cudarc::nvrtc::Ptx::from_binary(bytes.to_vec())
                }
                ImageSelection::Ptx => {
                    candle_core::cuda_backend::cudarc::nvrtc::Ptx::from_src(assets.ptx)
                }
            };
            let module = device.cuda_stream().context().load_module(image).w()?;
            let mut cache = module_cache().lock().map_err(|_| {
                candle_core::Error::Msg("CUDA kernel module cache is poisoned".into())
            })?;
            cache.entry(key).or_insert(module).clone()
        }
    };
    module.load_function(function_name).w()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assets(cubins: &'static [Cubin]) -> KernelAssets {
        KernelAssets {
            name: "synthetic",
            ptx: ".version 8.0",
            cubins,
        }
    }

    #[test]
    fn exact_architecture_selects_the_cubin() {
        static CUBINS: [Cubin; 1] = [Cubin {
            architecture: 89,
            image: b"elf",
        }];
        assert_eq!(
            assets(&CUBINS).select((8, 9)),
            ImageSelection::Cubin { architecture: 89 }
        );
    }

    #[test]
    fn a_near_miss_falls_through_to_the_ptx() {
        static CUBINS: [Cubin; 1] = [Cubin {
            architecture: 89,
            image: b"elf",
        }];
        assert_eq!(assets(&CUBINS).select((9, 0)), ImageSelection::Ptx);
        assert_eq!(assets(&CUBINS).select((8, 10)), ImageSelection::Ptx);
    }

    #[test]
    fn no_cubins_and_out_of_range_capabilities_take_the_ptx() {
        assert_eq!(assets(&[]).select((8, 9)), ImageSelection::Ptx);
        static CUBINS: [Cubin; 1] = [Cubin {
            architecture: 80,
            image: b"elf",
        }];
        assert_eq!(assets(&CUBINS).select((-1, 700)), ImageSelection::Ptx);
    }
}
