pub mod blas;
pub mod calibrate;
mod closure;
pub mod copy;
pub mod cubin;
pub mod decode;
pub mod draft;
pub mod engine;
pub mod flash;
mod linear_calibrate;
mod markov;
mod markov_calibrate;
pub mod multi_calibrate;
pub mod ops;
pub mod profile;
pub(crate) mod tree;
pub mod verification;
pub mod weights;

use crate::backend::setup::DeviceInfo;
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaContext, CudaStream};
use std::sync::Arc;

pub struct Device {
    pub info: DeviceInfo,
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub upload: Arc<CudaStream>,
}

impl Device {
    pub fn new(ordinal: usize) -> Result<Self> {
        ensure!(
            unsafe { cudarc::driver::sys::is_culib_present() },
            "CUDA backend unavailable: CUDA driver library is missing"
        );
        ensure!(
            unsafe { cudarc::cublas::sys::is_culib_present() },
            "CUDA backend unavailable: cuBLAS library is missing"
        );
        ensure!(
            unsafe { cudarc::cublaslt::sys::is_culib_present() },
            "CUDA backend unavailable: cuBLASLt library is missing"
        );
        let ctx = CudaContext::new(ordinal)
            .with_context(|| format!("CUDA backend unavailable: initialize device {ordinal}"))?;
        let sm = cubin::capability(&ctx)?;
        cubin::check(sm)?;
        eprintln!("cuda_sm={sm} native cubins selected");
        use cudarc::driver::sys::CUdevice_attribute as A;
        let info = DeviceInfo {
            name: ctx.name()?,
            arch: sm,
            sms: ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as usize,
            l2_bytes: ctx.attribute(A::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)? as usize,
            cooperative: ctx.attribute(A::CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH)? != 0,
            optin_shared_bytes: ctx
                .attribute(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)?
                as usize,
            warp: ctx.attribute(A::CU_DEVICE_ATTRIBUTE_WARP_SIZE)? as usize,
            max_threads: ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)? as u32,
            shared_bytes: ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?
                as usize,
            sm_shared_bytes: ctx
                .attribute(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)?
                as usize,
        };
        let stream = ctx.new_stream()?;
        let upload = ctx.new_stream()?;
        Ok(Self {
            info,
            ctx,
            stream,
            upload,
        })
    }

    pub fn finish_upload(&self) -> Result<()> {
        let event = self.upload.record_event(Some(
            cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
        ))?;
        self.stream.wait(&event)?;
        self.stream.synchronize()?;
        unsafe {
            self.ctx.disable_event_tracking();
        }
        Ok(())
    }
}
