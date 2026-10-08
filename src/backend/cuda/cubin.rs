use anyhow::{Result, ensure};
use cudarc::{
    driver::{CudaContext, CudaModule},
    nvrtc::Ptx,
};
use std::sync::Arc;

mod binaries {
    include!(concat!(env!("OUT_DIR"), "/kernels.rs"));
}

pub fn check(sm: u32) -> Result<()> {
    ensure!(
        binaries::ARCHES.contains(&sm),
        "missing native cubins for sm_{sm}; built architectures: {:?}",
        binaries::ARCHES
    );
    Ok(())
}

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

#[cfg(test)]
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
