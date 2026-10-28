#[cfg(feature = "cuda")]
pub(crate) mod attention_d256;
#[cfg(feature = "cuda")]
pub mod blas;
#[cfg(feature = "rocm")]
#[path = "../hip_blas.rs"]
pub mod blas;
#[cfg(feature = "cuda")]
pub mod calibrate;
mod closure;
pub mod contracts;
pub mod copy;
pub mod cubin;
pub mod decode;
#[cfg(feature = "cuda")]
pub mod draft;
pub mod engine;
#[cfg(feature = "cuda")]
pub mod flash;
#[cfg(feature = "cuda")]
pub(crate) mod gdn;
#[cfg(test)]
#[cfg(feature = "cuda")]
mod kv_cache_tests;
#[cfg(feature = "cuda")]
pub(crate) mod linear;
#[cfg(feature = "cuda")]
mod linear_calibrate;
#[cfg(feature = "cuda")]
mod markov;
#[cfg(feature = "cuda")]
mod markov_calibrate;
#[cfg(feature = "cuda")]
pub(crate) mod moe;
#[cfg(feature = "cuda")]
pub mod multi_calibrate;
pub mod ops;
#[cfg(feature = "cuda")]
pub mod profile;
pub mod state;
#[cfg(feature = "cuda")]
pub(crate) mod tree;
pub mod verification;
#[cfg(feature = "cuda")]
pub(crate) mod vision;
pub mod weights;

use crate::backend::gpu::{CudaContext, CudaStream};
#[cfg(feature = "cuda")]
use crate::backend::setup::CacheKey;
use crate::backend::setup::DeviceInfo;
#[cfg(feature = "cuda")]
use anyhow::ensure;
use anyhow::{Context, Result};
use std::sync::Arc;

#[derive(Clone)]
pub struct Device {
    #[cfg(feature = "cuda")]
    key: CacheKey,
    pub info: DeviceInfo,
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub upload: Arc<CudaStream>,
}

impl Device {
    #[cfg(feature = "cuda")]
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
        let memory_clock_khz = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)?;
        let memory_bus_bits = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)?;
        let memory_bytes_per_second =
            2.0 * (f64::from(memory_clock_khz) * 1000.0) * (f64::from(memory_bus_bits) / 8.0);
        ensure!(
            memory_clock_khz > 0
                && memory_bus_bits > 0
                && memory_bytes_per_second.is_finite()
                && memory_bytes_per_second > 0.0,
            "CUDA device {ordinal} reported invalid memory clock={memory_clock_khz} kHz bus={memory_bus_bits} bits"
        );
        let info = DeviceInfo {
            name: ctx.name()?,
            arch: crate::backend::setup::DeviceArchitecture::Cuda(sm),
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
            memory_bytes_per_second,
        };
        let stream = ctx.new_stream()?;
        let upload = ctx.new_stream()?;
        let key = calibrate::key(&ctx)?;
        Ok(Self {
            key,
            info,
            ctx,
            stream,
            upload,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn peer(&self) -> Result<Option<Self>> {
        Ok(self.peer_with_unsupported_sm()?.0)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn peer_with_unsupported_sm(&self) -> Result<(Option<Self>, Option<u32>)> {
        #[cfg(test)]
        if DISABLE_PEER.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok((None, None));
        }
        if self.ctx.ordinal() != 0 || CudaContext::device_count()? <= 1 {
            return Ok((None, None));
        }
        use cudarc::driver::sys::CUdevice_attribute as A;
        let dev = cudarc::driver::result::device::get(1)
            .context("CUDA backend unavailable: get device 1")?;
        // SAFETY: `dev` is returned by `cuDeviceGet(1)` on an initialized CUDA driver.
        let capability = unsafe {
            cudarc::driver::result::device::get_attribute(
                dev,
                A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )
            .and_then(|major| {
                let minor = cudarc::driver::result::device::get_attribute(
                    dev,
                    A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                )?;
                Ok((major, minor))
            })
            .context("CUDA backend unavailable: query device 1 capability")
        };
        if let Err(sm) = check_peer_capability(capability)? {
            return Ok((None, Some(sm)));
        }
        let peer = Self::new(1)?;
        let mut can_0_to_1 = 0i32;
        let mut can_1_to_0 = 0i32;
        unsafe {
            cudarc::driver::sys::cuDeviceCanAccessPeer(
                &mut can_0_to_1,
                self.ctx.cu_device(),
                peer.ctx.cu_device(),
            )
            .result()?;
            cudarc::driver::sys::cuDeviceCanAccessPeer(
                &mut can_1_to_0,
                peer.ctx.cu_device(),
                self.ctx.cu_device(),
            )
            .result()?;
        }
        let can_access_peer = [can_0_to_1 != 0, can_1_to_0 != 0];
        verify_peer_transport(self, &peer, can_access_peer, None)?;
        eprintln!(
            "{}",
            serde_json::json!({
                "cuda_peer_topology": {
                    "can_access_peer_0_to_1": can_access_peer[0],
                    "can_access_peer_1_to_0": can_access_peer[1],
                    "peer_access_enabled": false,
                    "decode_transport": "captured_portable_bounce",
                    "prefill_transport": "driver_staged_peer_memcpy",
                    "verified_bounce_bytes": PEER_VERIFY_BOUNCE_BYTES,
                    "verified_peer_memcpy_bytes": PEER_VERIFY_MEMCPY_BYTES,
                }
            })
        );
        self.ctx.bind_to_thread()?;
        Ok((Some(peer), None))
    }

    #[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
const PEER_VERIFY_BOUNCE_BYTES: usize = 65_536;
#[cfg(feature = "cuda")]
const PEER_VERIFY_MEMCPY_BYTES: usize = 4 * 1024 * 1024;

#[cfg(feature = "cuda")]
fn check_peer_capability(capability: Result<(i32, i32)>) -> Result<std::result::Result<u32, u32>> {
    let (major, minor) = capability?;
    ensure!(
        (1..=99).contains(&major) && (0..=9).contains(&minor),
        "invalid CUDA compute capability {major}.{minor}"
    );
    let sm = (major * 10 + minor) as u32;
    if cubin::check(sm).is_ok() {
        Ok(Ok(sm))
    } else {
        Ok(Err(sm))
    }
}

#[cfg(feature = "cuda")]
struct PortableVerifyBounce(*mut u8);

#[cfg(feature = "cuda")]
impl PortableVerifyBounce {
    fn new(bytes: usize) -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: CU_MEMHOSTALLOC_PORTABLE maps the pinned host region into all CUDA contexts.
        unsafe {
            cudarc::driver::sys::cuMemHostAlloc(
                &mut raw,
                bytes,
                cudarc::driver::sys::CU_MEMHOSTALLOC_PORTABLE,
            )
            .result()
            .context("cuMemHostAlloc peer transport verifier")?;
            std::ptr::write_bytes(raw.cast::<u8>(), 0, bytes);
        }
        Ok(Self(raw.cast::<u8>()))
    }

    fn send_ptr(&self) -> *mut std::ffi::c_void {
        self.0.cast()
    }

    fn recv_ptr(&self, offset: usize) -> *mut std::ffi::c_void {
        // SAFETY: caller allocates 2 * offset bytes.
        unsafe { self.0.add(offset).cast() }
    }
}

#[cfg(feature = "cuda")]
impl Drop for PortableVerifyBounce {
    fn drop(&mut self) {
        // SAFETY: pointer comes from cuMemHostAlloc and streams are synchronized before drop.
        unsafe {
            let _ = cudarc::driver::sys::cuMemFreeHost(self.0.cast());
        }
    }
}

#[cfg(feature = "cuda")]
fn peer_verify_pattern(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| ((i.wrapping_mul(131).wrapping_add(17)) ^ (i >> 7) ^ (i >> 15) ^ (i >> 23)) as u8)
        .collect()
}

#[cfg(feature = "cuda")]
fn check_peer_roundtrip_bytes(
    form: &str,
    expected: &[u8],
    actual: &[u8],
    can_access_peer: [bool; 2],
) -> Result<()> {
    ensure!(
        expected.len() == actual.len(),
        "peer transport verification length mismatch form={form} expected={} actual={}",
        expected.len(),
        actual.len()
    );
    let mut first = None;
    let mut mismatches = 0usize;
    for (idx, (&exp, &got)) in expected.iter().zip(actual).enumerate() {
        if exp != got {
            mismatches += 1;
            if first.is_none() {
                first = Some((idx, exp, got));
            }
        }
    }
    if let Some((offset, exp, got)) = first {
        anyhow::bail!(
            "peer transport verification failed form={form} first_offset={offset} expected={exp:#04x} got={got:#04x} mismatch_bytes={mismatches}/{} can_access_peer={can_access_peer:?} peer_access_enabled=false",
            expected.len()
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn assert_peer_access_disabled(
    dev0: &Device,
    dev1: &Device,
    can_access_peer: [bool; 2],
) -> Result<()> {
    use cudarc::driver::sys::cudaError_enum as E;
    let mut enabled_dir = None;
    for (src_ord, src, dst_ord, dst) in [
        (0usize, &dev0.ctx, 1usize, &dev1.ctx),
        (1usize, &dev1.ctx, 0usize, &dev0.ctx),
    ] {
        src.bind_to_thread()?;
        // SAFETY: both contexts are live and bound on this thread as required by cuCtxDisablePeerAccess.
        let status = unsafe { cudarc::driver::sys::cuCtxDisablePeerAccess(dst.cu_ctx()) };
        match status {
            E::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED => {}
            E::CUDA_SUCCESS => {
                if enabled_dir.is_none() {
                    enabled_dir = Some((src_ord, dst_ord));
                }
            }
            other => {
                let _ = dev0.ctx.bind_to_thread();
                return other.result().with_context(|| {
                    format!("cuCtxDisablePeerAccess guard failed on {src_ord}->{dst_ord}")
                });
            }
        }
    }
    dev0.ctx.bind_to_thread()?;
    if let Some((src_ord, dst_ord)) = enabled_dir {
        anyhow::bail!(
            "direct peer access was enabled on context {src_ord}->{dst_ord} (can_access_peer={can_access_peer:?}); direct peer access is forbidden"
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn verify_peer_transport(
    dev0: &Device,
    dev1: &Device,
    can_access_peer: [bool; 2],
    plant_flip: Option<usize>,
) -> Result<()> {
    use cudarc::driver::DevicePtr;
    assert_peer_access_disabled(dev0, dev1, can_access_peer)?;
    let expected = peer_verify_pattern(PEER_VERIFY_MEMCPY_BYTES);

    {
        let bounce = PortableVerifyBounce::new(PEER_VERIFY_BOUNCE_BYTES * 2)?;
        let d0 = dev0
            .stream
            .clone_htod(&expected[..PEER_VERIFY_BOUNCE_BYTES])?;
        let d1 = dev1.stream.alloc_zeros::<u8>(PEER_VERIFY_BOUNCE_BYTES)?;
        let fork = dev0.ctx.new_event(Some(
            cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
        ))?;
        let join = dev1.ctx.new_event(Some(
            cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
        ))?;
        dev0.stream.synchronize()?;
        dev1.stream.synchronize()?;
        dev0.ctx.bind_to_thread()?;
        let d0_ptr = d0.device_ptr(&dev0.stream).0;
        let d1_ptr = d1.device_ptr(&dev1.stream).0;
        dev0.stream
            .begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .context("cuStreamBeginCapture peer bounce verifier")?;
        let captured_launch = (|| -> Result<()> {
            // SAFETY: d0, d1, and portable bounce buffers cover PEER_VERIFY_BOUNCE_BYTES across capture and launch.
            unsafe {
                cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                    bounce.send_ptr(),
                    d0_ptr,
                    PEER_VERIFY_BOUNCE_BYTES,
                    dev0.stream.cu_stream(),
                )
                .result()
                .context("cuMemcpyDtoHAsync verifier 0->h")?;
                fork.record(&dev0.stream)?;
                dev1.stream.wait(&fork)?;
                cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                    d1_ptr,
                    bounce.send_ptr(),
                    PEER_VERIFY_BOUNCE_BYTES,
                    dev1.stream.cu_stream(),
                )
                .result()
                .context("cuMemcpyHtoDAsync verifier h->1")?;
                cudarc::driver::sys::cuMemsetD8Async(
                    d0_ptr,
                    0,
                    PEER_VERIFY_BOUNCE_BYTES,
                    dev0.stream.cu_stream(),
                )
                .result()
                .context("cuMemsetD8Async verifier clear d0")?;
                cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                    bounce.recv_ptr(PEER_VERIFY_BOUNCE_BYTES),
                    d1_ptr,
                    PEER_VERIFY_BOUNCE_BYTES,
                    dev1.stream.cu_stream(),
                )
                .result()
                .context("cuMemcpyDtoHAsync verifier 1->h")?;
                join.record(&dev1.stream)?;
                dev0.stream.wait(&join)?;
                cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                    d0_ptr,
                    bounce.recv_ptr(PEER_VERIFY_BOUNCE_BYTES),
                    PEER_VERIFY_BOUNCE_BYTES,
                    dev0.stream.cu_stream(),
                )
                .result()
                .context("cuMemcpyHtoDAsync verifier h->0")?;
            }
            Ok(())
        })();
        let graph_res = dev0
            .stream
            .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))
            .context("cuStreamEndCapture peer bounce verifier");
        captured_launch?;
        let graph = graph_res?.context("empty peer bounce verifier graph")?;
        graph
            .upload()
            .context("cuGraphUpload peer bounce verifier")?;
        graph
            .launch()
            .context("cuGraphLaunch peer bounce verifier")?;
        dev0.stream.synchronize()?;
        dev1.stream.synchronize()?;
        let actual_bounce = dev0.stream.clone_dtoh(&d0)?;
        check_peer_roundtrip_bytes(
            "captured_portable_bounce",
            &expected[..PEER_VERIFY_BOUNCE_BYTES],
            &actual_bounce,
            can_access_peer,
        )?;
    }

    {
        let m0 = dev0.stream.clone_htod(&expected)?;
        let m1 = dev1.stream.alloc_zeros::<u8>(PEER_VERIFY_MEMCPY_BYTES)?;
        dev0.stream.synchronize()?;
        dev1.stream.synchronize()?;
        let m0_ptr = m0.device_ptr(&dev0.stream).0;
        let m1_ptr = m1.device_ptr(&dev1.stream).0;
        // SAFETY: m0 on dev0 and m1 on dev1 each hold PEER_VERIFY_MEMCPY_BYTES and contexts stay live.
        unsafe {
            cudarc::driver::sys::cuMemcpyPeerAsync(
                m1_ptr,
                dev1.ctx.cu_ctx(),
                m0_ptr,
                dev0.ctx.cu_ctx(),
                PEER_VERIFY_MEMCPY_BYTES,
                dev1.stream.cu_stream(),
            )
            .result()
            .context("cuMemcpyPeerAsync verifier 0->1")?;
        }
        dev1.stream.synchronize()?;
        if let Some(flip_offset) = plant_flip {
            ensure!(
                flip_offset < PEER_VERIFY_MEMCPY_BYTES,
                "planted flip offset={flip_offset} outside {PEER_VERIFY_MEMCPY_BYTES}"
            );
            let corrupted = expected[flip_offset] ^ 0x5a;
            // SAFETY: flip_offset lies inside owned m1 buffer on dev1.
            unsafe {
                cudarc::driver::sys::cuMemsetD8Async(
                    m1_ptr + flip_offset as u64,
                    corrupted,
                    1,
                    dev1.stream.cu_stream(),
                )
                .result()
                .context("cuMemsetD8Async planted verifier byte flip")?;
            }
            dev1.stream.synchronize()?;
        }
        // SAFETY: m0 on dev0 holds PEER_VERIFY_MEMCPY_BYTES.
        unsafe {
            cudarc::driver::sys::cuMemsetD8Async(
                m0_ptr,
                0,
                PEER_VERIFY_MEMCPY_BYTES,
                dev0.stream.cu_stream(),
            )
            .result()
            .context("cuMemsetD8Async verifier clear m0")?;
        }
        dev0.stream.synchronize()?;
        // SAFETY: m1 on dev1 and m0 on dev0 each hold PEER_VERIFY_MEMCPY_BYTES.
        unsafe {
            cudarc::driver::sys::cuMemcpyPeerAsync(
                m0_ptr,
                dev0.ctx.cu_ctx(),
                m1_ptr,
                dev1.ctx.cu_ctx(),
                PEER_VERIFY_MEMCPY_BYTES,
                dev0.stream.cu_stream(),
            )
            .result()
            .context("cuMemcpyPeerAsync verifier 1->0")?;
        }
        dev0.stream.synchronize()?;
        let actual_memcpy = dev0.stream.clone_dtoh(&m0)?;
        dev0.ctx.bind_to_thread()?;
        check_peer_roundtrip_bytes(
            "driver_staged_peer_memcpy",
            &expected,
            &actual_memcpy,
            can_access_peer,
        )?;
    }

    Ok(())
}

/// γ_n = n·u / (1 − n·u).
pub(crate) fn gamma(n: usize, u: f64) -> f64 {
    let nu = n as f64 * u;
    nu / (1.0 - nu)
}

/// Max |a − b| relative to the compared magnitude, for two chains that each apply `roundings`
/// BF16 roundings and FP32 reductions of the given lengths: 2·(roundings·2^-8 + Σ γ_K(2^-24)).
pub(crate) fn bf16_chain_bound(roundings: u32, reductions: &[usize]) -> f64 {
    let accumulation: f64 = reductions.iter().map(|&k| gamma(k, 2f64.powi(-24))).sum();
    2.0 * (f64::from(roundings) * 2f64.powi(-8) + accumulation)
}

#[cfg(test)]
#[cfg(feature = "cuda")]
pub(crate) fn time_prepared_operation(
    stream: &Arc<CudaStream>,
    mut reset: impl FnMut() -> Result<()>,
    mut operation: impl FnMut() -> Result<()>,
) -> Result<f64> {
    for _ in 0..16 {
        reset()?;
        operation()?;
    }
    stream.synchronize()?;
    let flags = cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
    let start = stream.context().new_event(Some(flags))?;
    let end = stream.context().new_event(Some(flags))?;
    let mut total = 0.0;
    for _ in 0..32 {
        reset()?;
        start.record(stream)?;
        operation()?;
        end.record(stream)?;
        stream.synchronize()?;
        let elapsed = f64::from(start.elapsed_ms(&end)?);
        ensure!(
            elapsed.is_finite() && elapsed > 0.0,
            "invalid prepared operation elapsed_ms={elapsed}"
        );
        total += elapsed;
    }
    Ok(total / 32.0)
}

#[cfg(test)]
#[cfg(feature = "cuda")]
pub(crate) static DISABLE_PEER: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
#[cfg(feature = "cuda")]
mod q38_tests;

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    #[test]
    fn peer_copy_verifier_detects_single_byte_corruption_cpu() -> Result<()> {
        let pattern = peer_verify_pattern(PEER_VERIFY_MEMCPY_BYTES);
        assert_ne!(&pattern[..4096], &pattern[4096..8192]);
        check_peer_roundtrip_bytes(
            "driver_staged_peer_memcpy",
            &pattern,
            &pattern,
            [true, true],
        )?;
        let mut corrupted = pattern.clone();
        corrupted[2_345_679] ^= 0x5a;
        let err = check_peer_roundtrip_bytes(
            "driver_staged_peer_memcpy",
            &pattern,
            &corrupted,
            [true, true],
        )
        .expect_err("single-byte corruption must fail verifier");
        let msg = format!("{err:#}");
        ensure!(
            msg.contains("first_offset=2345679") && msg.contains("mismatch_bytes=1/4194304"),
            "unexpected verifier error: {msg}"
        );
        Ok(())
    }

    #[test]
    fn peer_capability_predicate_skips_unsupported_sm_and_propagates_errors_cpu() -> Result<()> {
        let supported_sm = (50..=120)
            .find(|&sm| cubin::check(sm).is_ok())
            .context("built binary must include at least one native cubin architecture")?;
        let supported = (
            i32::try_from(supported_sm / 10)?,
            i32::try_from(supported_sm % 10)?,
        );
        assert_eq!(check_peer_capability(Ok(supported))?, Ok(supported_sm));

        let unsupported_sm = [75u32, 86, 89, 90, 61, 52, 10]
            .into_iter()
            .find(|&sm| cubin::check(sm).is_err())
            .context("must find an architecture absent from CUDA_ARCH_LIST")?;
        let unsupported = (
            i32::try_from(unsupported_sm / 10)?,
            i32::try_from(unsupported_sm % 10)?,
        );
        assert_eq!(check_peer_capability(Ok(unsupported))?, Err(unsupported_sm));

        let driver_err = check_peer_capability(Err(anyhow::anyhow!("CUDA_ERROR_INVALID_DEVICE")))
            .expect_err("driver failure must propagate");
        ensure!(
            format!("{driver_err:#}").contains("CUDA_ERROR_INVALID_DEVICE"),
            "unexpected driver error: {driver_err:#}"
        );
        for invalid in [(0, 0), (-1, 5), (8, -1), (8, 10), (100, 0)] {
            assert!(
                check_peer_capability(Ok(invalid)).is_err(),
                "invalid capability {invalid:?} must error"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires 2 GPUs"]
    fn peer_transport_rejects_peer_access_and_detects_corrupt_p2p_gpu() -> Result<()> {
        const MODE_ENV: &str = "FF_PEER_TRANSPORT_GATE_MODE";
        const TEST_NAME: &str =
            "backend::cuda::tests::peer_transport_rejects_peer_access_and_detects_corrupt_p2p_gpu";

        if let Ok(mode) = std::env::var(MODE_ENV) {
            ensure!(
                CudaContext::device_count()? >= 2,
                "Runner FAIL peer transport gate requires >= 2 CUDA devices"
            );
            match mode.as_str() {
                "production_default" => {
                    let dev0 = Device::new(0)?;
                    let peer = dev0
                        .peer()?
                        .context("Runner FAIL Device::peer returned None on 2-GPU host")?;
                    assert_peer_access_disabled(&dev0, &peer, [true, true])?;
                    return Ok(());
                }
                "peer_access_enabled_rejected" => {
                    let dev0 = Device::new(0)?;
                    let dev1 = Device::new(1)?;
                    let mut can_0_to_1 = 0i32;
                    let mut can_1_to_0 = 0i32;
                    unsafe {
                        cudarc::driver::sys::cuDeviceCanAccessPeer(
                            &mut can_0_to_1,
                            dev0.ctx.cu_device(),
                            dev1.ctx.cu_device(),
                        )
                        .result()?;
                        cudarc::driver::sys::cuDeviceCanAccessPeer(
                            &mut can_1_to_0,
                            dev1.ctx.cu_device(),
                            dev0.ctx.cu_device(),
                        )
                        .result()?;
                    }
                    ensure!(
                        can_0_to_1 != 0 || can_1_to_0 != 0,
                        "Runner FAIL neither direction reports cuDeviceCanAccessPeer != 0"
                    );
                    if can_0_to_1 != 0 {
                        dev0.ctx.bind_to_thread()?;
                        unsafe {
                            cudarc::driver::sys::cuCtxEnablePeerAccess(dev1.ctx.cu_ctx(), 0)
                                .result()
                                .context("cuCtxEnablePeerAccess 0->1")?;
                        }
                    } else {
                        dev1.ctx.bind_to_thread()?;
                        unsafe {
                            cudarc::driver::sys::cuCtxEnablePeerAccess(dev0.ctx.cu_ctx(), 0)
                                .result()
                                .context("cuCtxEnablePeerAccess 1->0")?;
                        }
                    }
                    let rejected = verify_peer_transport(
                        &dev0,
                        &dev1,
                        [can_0_to_1 != 0, can_1_to_0 != 0],
                        None,
                    );
                    assert_peer_access_disabled(&dev0, &dev1, [can_0_to_1 != 0, can_1_to_0 != 0])?;
                    let err = rejected.expect_err("enabled peer access must be rejected");
                    let msg = format!("{err:#}");
                    ensure!(
                        msg.contains("direct peer access was enabled on context"),
                        "Runner FAIL unexpected guard error: {msg}"
                    );
                    std::process::exit(3);
                }
                "planted_byte_flip_detected" => {
                    let dev0 = Device::new(0)?;
                    let dev1 = Device::new(1)?;
                    let rejected =
                        verify_peer_transport(&dev0, &dev1, [true, true], Some(3_141_593));
                    assert_peer_access_disabled(&dev0, &dev1, [true, true])?;
                    let err = rejected.expect_err("planted 1-byte device corruption must fail");
                    let msg = format!("{err:#}");
                    ensure!(
                        msg.contains("driver_staged_peer_memcpy")
                            && msg.contains("first_offset=3141593")
                            && msg.contains("mismatch_bytes=1/4194304"),
                        "Runner FAIL unexpected corruption error: {msg}"
                    );
                    std::process::exit(3);
                }
                other => anyhow::bail!("unknown {MODE_ENV}={other}"),
            }
        }

        let exe = std::env::current_exe()?;
        for (mode, expected_rc) in [
            ("production_default", 0),
            ("peer_access_enabled_rejected", 3),
            ("planted_byte_flip_detected", 3),
        ] {
            let output = std::process::Command::new(&exe)
                .env(MODE_ENV, mode)
                .args(["--exact", TEST_NAME, "--include-ignored", "--nocapture"])
                .output()
                .with_context(|| format!("spawn child mode={mode}"))?;
            let rc = output.status.code().unwrap_or(-1);
            ensure!(
                rc == expected_rc,
                "Runner FAIL peer_transport_gate child={mode} rc={rc} expected={expected_rc} stdout={} stderr={}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            println!("PASS peer_transport_child mode={mode} rc={rc}");
        }
        println!(
            "PASS peer_transport_gate production_default=ok peer_access_enabled_rejected=true planted_byte_flip_detected=true"
        );
        Ok(())
    }
}

#[cfg(feature = "rocm")]
impl Device {
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal)
            .with_context(|| format!("HIP initialize device {ordinal}"))?;
        let info = ctx.device_info()?;
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
        self.stream.wait(&self.upload.record_event(Some(2))?)?;
        self.stream.synchronize()
    }
}
