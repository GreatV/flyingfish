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
                for probe in [(major + 1, 0), (major, minor + 1), (7, 5)] {
                    let architecture = probe.0 * 10 + probe.1;
                    let selected = assets.select((probe.0 as i32, probe.1 as i32));
                    assert!(
                        selected == ImageSelection::Ptx
                            || selected == ImageSelection::Cubin { architecture },
                        "select({probe:?}) on {} returned {selected:?}, not its own architecture",
                        assets.name
                    );
                }
            }
        }
    }
}
