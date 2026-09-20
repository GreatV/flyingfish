//! The compiled images of each kernel, and the choice between them.
//!
//! One PTX per kernel is the numerical contract: `build.rs` generates it at
//! `compute_80` and nothing about the running device changes it. A cubin is
//! the same PTX already translated by `ptxas` for one architecture; loading it
//! skips the driver's own translation, which rejects a PTX whose ISA version
//! postdates the driver. When no cubin matches, the PTX is loaded and the
//! driver translates it.

use std::sync::Arc;

use anyhow::{Context, Result};
use cudarc::driver::safe::{CudaContext, CudaModule};

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
}

include!(concat!(env!("OUT_DIR"), "/cuda_kernel_manifest.rs"));

/// Load one kernel's selected image for this context's device.
pub fn load_module(
    context: &Arc<CudaContext>,
    assets: &'static KernelAssets,
) -> Result<Arc<CudaModule>> {
    let selection = assets.select(
        context
            .compute_capability()
            .context("compute capability query")?,
    );
    let image = match selection {
        ImageSelection::Cubin { architecture } => cudarc::nvrtc::Ptx::from_binary(
            assets
                .cubins
                .iter()
                .find(|cubin| cubin.architecture == architecture)
                .expect("select only names an embedded cubin")
                .image
                .to_vec(),
        ),
        ImageSelection::Ptx => cudarc::nvrtc::Ptx::from_src(assets.ptx),
    };
    context
        .load_module(image)
        .with_context(|| format!("failed to load {} module", assets.name))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every kernel this crate compiles, so the structural gate cannot silently
    /// miss one.
    const ALL: [&KernelAssets; 7] = [
        &EDGE0_GEMV,
        &EDGE0_SILU_MUL,
        &EDGE0_BATCHED_GEMV,
        &LORA_ADD,
        &EDGE0_GDN,
        &EDGE0_GLUE,
        &EDGE0_MEGA,
    ];

    #[test]
    fn every_kernel_carries_the_portable_baseline() {
        for assets in ALL {
            assert!(!assets.ptx.is_empty(), "{} has no PTX", assets.name);
            assert_eq!(
                assets.select((7, 5)),
                ImageSelection::Ptx,
                "{} must keep the PTX for pre-baseline devices",
                assets.name
            );
        }
    }

    #[test]
    fn cubins_are_never_selected_for_another_architecture() {
        for assets in ALL {
            for cubin in assets.cubins {
                let major = cubin.architecture / 10;
                let minor = cubin.architecture % 10;
                assert_eq!(
                    assets.select((major as i32, (minor + 1) as i32)),
                    ImageSelection::Ptx,
                    "sm_{} cubin of {} must not serve a near miss",
                    cubin.architecture,
                    assets.name
                );
                assert_eq!(
                    assets.select((major as i32, minor as i32)),
                    ImageSelection::Cubin {
                        architecture: cubin.architecture
                    },
                    "sm_{} cubin of {} must serve its exact architecture",
                    cubin.architecture,
                    assets.name
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn every_kernel_loads_on_this_device() {
        let Ok(context) = CudaContext::new(0) else {
            eprintln!("no CUDA device available, skipping");
            return;
        };
        for assets in ALL {
            load_module(&context, assets).expect("kernel module must load on this device");
        }
    }
}
