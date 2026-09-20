//! Compiled kernel images for this crate; the types come from ff-edge0.

pub(crate) use ff_edge0::kernel_assets::{Cubin, KernelAssets};

include!(concat!(env!("OUT_DIR"), "/cuda_kernel_manifest.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use ff_edge0::kernel_assets::ImageSelection;

    const ALL: [&KernelAssets; 2] = [&QWEN_BATCH2, &WIDE_GEMV];

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
            }
        }
    }
}
