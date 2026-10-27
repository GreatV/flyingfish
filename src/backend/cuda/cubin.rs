use crate::backend::gpu::{CudaContext, CudaModule};
use anyhow::{Result, ensure};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

mod binaries {
    include!(concat!(env!("OUT_DIR"), "/kernels.rs"));
}

/// The native cubin bytes for one kernel module, or None when this build has no
/// cubin for that architecture.
#[cfg(feature = "cuda")]
pub fn bytes(name: &str, sm: u32) -> Option<&'static [u8]> {
    binaries::cubin(name, sm)
}

#[cfg(feature = "cuda")]
pub fn check(sm: u32) -> Result<()> {
    ensure!(
        binaries::ARCHES.contains(&sm),
        "missing native cubins for sm_{sm}; built architectures: {:?}",
        binaries::ARCHES
    );
    Ok(())
}

#[cfg(feature = "cuda")]
pub fn capability(ctx: &CudaContext) -> Result<u32> {
    use cudarc::driver::sys::CUdevice_attribute as A;
    let major = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
    let minor = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
    ensure!(
        major > 0 && (0..=9).contains(&minor),
        "invalid CUDA compute capability {major}.{minor}"
    );
    Ok((major * 10 + minor) as u32)
}

#[cfg(feature = "cuda")]
pub fn module(ctx: &Arc<CudaContext>, name: &str) -> Result<Arc<CudaModule>> {
    let sm = capability(ctx)?;
    let bytes = binaries::cubin(name, sm).ok_or_else(|| {
        anyhow::anyhow!(
            "missing native cubin {name} for sm_{sm}; built architectures: {:?}",
            binaries::ARCHES
        )
    })?;
    Ok(ctx.load_module(Ptx::from_binary(bytes.to_vec()))?)
}

#[cfg(feature = "rocm")]
pub fn module(ctx: &Arc<CudaContext>, name: &str) -> Result<Arc<CudaModule>> {
    let architecture = ctx.architecture();
    ensure!(
        binaries::ARCHES.contains(&architecture),
        "missing native HIP code objects for {architecture}"
    );
    let bytes = binaries::hsaco(name, architecture).ok_or_else(|| {
        anyhow::anyhow!(
            "unsupported HIP kernel module {name} for {architecture}: no compiled implementation"
        )
    })?;
    ctx.require_simt(name)?;
    ctx.load_binary(bytes)
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::{binaries, check};

    #[test]
    fn exact_architecture_and_elf_only() {
        for &sm in binaries::ARCHES {
            check(sm).unwrap();
            for name in ["embed", "norm", "rope", "act", "sample", "attention"] {
                let bytes = binaries::cubin(name, sm).unwrap();
                assert_eq!(&bytes[..4], b"\x7fELF");
            }
        }
        assert!(check(0).unwrap_err().to_string().contains("sm_0"));
        assert!(binaries::cubin("norm", 0).is_none());
    }
}
