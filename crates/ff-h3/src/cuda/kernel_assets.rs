//! The compiled images of each transcribed kernel, and the choice between them.
//!
//! One PTX per kernel is the numerical contract: `build.rs` generates it at
//! `compute_80` and nothing about the running device changes it. What the
//! device does change is how that PTX reaches the hardware. A cubin is the same
//! PTX already translated by `ptxas` for one architecture; loading it skips the
//! driver's own translation of the identical instructions. When no cubin
//! matches, the PTX is loaded and the driver translates it, which is the path
//! every device had before any cubin existed.
//!
//! So the selection here is a load-time choice, not a numerical one. The
//! architecture that was selected is still recorded, because "the same PTX"
//! is a claim about the instructions and not a proof about the SASS a given
//! `ptxas` emitted from them.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::cuda_backend::CudaDevice;
use candle_core::cuda_backend::cudarc::driver::{CudaFunction, CudaModule};
use std::sync::Arc;

/// One architecture's translation of a kernel's PTX.
pub(crate) struct Cubin {
    pub(crate) architecture: u32,
    pub(crate) image: &'static [u8],
}

/// Every compiled form of one kernel.
pub(crate) struct KernelAssets {
    pub(crate) name: &'static str,
    pub(crate) ptx: &'static str,
    pub(crate) cubins: &'static [Cubin],
}

include!(concat!(env!("OUT_DIR"), "/cuda_kernel_manifest.rs"));

/// Which compiled form a device will load.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ImageSelection {
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
    pub(crate) fn select(&self, compute_capability: (i32, i32)) -> ImageSelection {
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

type ModuleKey = (candle_core::cuda_backend::DeviceId, &'static str);

fn module_cache() -> &'static Mutex<HashMap<ModuleKey, Arc<CudaModule>>> {
    static CACHE: OnceLock<Mutex<HashMap<ModuleKey, Arc<CudaModule>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Load one function out of a kernel's best available image for this device.
///
/// Modules are cached per device, as Candle caches its own, so the translation
/// happens once per process rather than once per launch.
pub(crate) fn load_function(
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
            let selection = assets.select(super::device::compute_capability(device)?);
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

    /// Every kernel this crate compiles, so the structural gate cannot silently
    /// stop covering one that was added later.
    const ALL: [&KernelAssets; 6] = [
        &H3_RMS_NORM_BF16,
        &H3_SDPA_SOFTMAX_F32,
        &QWEN_ATTENTION_SCALE_BF16,
        &QWEN_HEAD_MEAN_F32,
        &QWEN_LAYER_NORM_BF16,
        &QWEN_GELU_BF16,
    ];

    /// The baseline is what makes every device runnable, so it must exist for
    /// each kernel whatever the build was asked to target.
    #[test]
    fn every_kernel_carries_the_portable_baseline() {
        for assets in ALL {
            assert!(
                !assets.ptx.is_empty(),
                "{} has no portable PTX",
                assets.name
            );
            assert_eq!(
                assets.select((7, 5)),
                ImageSelection::Ptx,
                "{} should fall back to PTX below the compiled architectures",
                assets.name
            );
        }
    }

    /// A cubin is only ever selected for its exact architecture.
    #[test]
    fn cubins_are_never_selected_for_another_architecture() {
        for assets in ALL {
            for cubin in assets.cubins {
                let major = i32::try_from(cubin.architecture / 10).expect("architecture major");
                let minor = i32::try_from(cubin.architecture % 10).expect("architecture minor");
                assert_eq!(
                    assets.select((major, minor)),
                    ImageSelection::Cubin {
                        architecture: cubin.architecture
                    },
                    "{} did not select its own sm_{} cubin",
                    assets.name,
                    cubin.architecture
                );
                assert_eq!(
                    assets.select((major, minor + 1)),
                    if assets
                        .cubins
                        .iter()
                        .any(|other| other.architecture == cubin.architecture + 1)
                    {
                        ImageSelection::Cubin {
                            architecture: cubin.architecture + 1,
                        }
                    } else {
                        ImageSelection::Ptx
                    },
                    "{} reused a cubin across architectures",
                    assets.name
                );
            }
        }
    }
}
