//! HIP allocations retain their device and synchronize before releasing storage.
use anyhow::{Context as _, Result, ensure};
use std::{
    ffi::{CStr, CString, c_char, c_void},
    marker::PhantomData,
    ops::{Bound, RangeBounds},
    sync::Arc,
};

type Handle = *mut c_void;

unsafe extern "C" {
    fn hipInit(flags: u32) -> i32;
    fn hipSetDevice(device: i32) -> i32;
    fn hipGetDeviceCount(count: *mut i32) -> i32;
    fn hipGetErrorString(status: i32) -> *const c_char;
    fn hipDeviceSynchronize() -> i32;
    fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    fn hipStreamCreateWithFlags(stream: *mut Handle, flags: u32) -> i32;
    fn hipStreamDestroy(stream: Handle) -> i32;
    fn hipStreamSynchronize(stream: Handle) -> i32;
    fn hipStreamWaitEvent(stream: Handle, event: Handle, flags: u32) -> i32;
    fn hipEventCreateWithFlags(event: *mut Handle, flags: u32) -> i32;
    fn hipEventDestroy(event: Handle) -> i32;
    fn hipEventRecord(event: Handle, stream: Handle) -> i32;
    fn hipEventSynchronize(event: Handle) -> i32;
    fn hipEventElapsedTime(ms: *mut f32, start: Handle, end: Handle) -> i32;
    fn hipMalloc(ptr: *mut Handle, bytes: usize) -> i32;
    fn hipFree(ptr: Handle) -> i32;
    fn hipHostMalloc(ptr: *mut Handle, bytes: usize, flags: u32) -> i32;
    fn hipHostFree(ptr: Handle) -> i32;
    fn hipMemcpyAsync(
        dst: Handle,
        src: *const c_void,
        bytes: usize,
        kind: i32,
        stream: Handle,
    ) -> i32;
    fn hipMemcpy2DAsync(
        dst: Handle,
        dpitch: usize,
        src: *const c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: i32,
        stream: Handle,
    ) -> i32;
    fn hipMemsetAsync(dst: Handle, value: i32, bytes: usize, stream: Handle) -> i32;
    fn hipMemsetD32Async(dst: Handle, value: i32, count: usize, stream: Handle) -> i32;
    fn hipModuleLoadData(module: *mut Handle, bytes: *const c_void) -> i32;
    fn hipModuleUnload(module: Handle) -> i32;
    fn hipModuleGetFunction(function: *mut Handle, module: Handle, name: *const c_char) -> i32;
    fn hipModuleLaunchKernel(
        function: Handle,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
        shared: u32,
        stream: Handle,
        args: *mut Handle,
        extra: *mut Handle,
    ) -> i32;
    fn hipFuncGetAttribute(value: *mut i32, attribute: i32, function: Handle) -> i32;
    fn hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
        blocks: *mut i32,
        function: Handle,
        threads: i32,
        shared: usize,
    ) -> i32;
    fn ff_hip_device_properties(device: i32, output: *mut Properties) -> i32;
}

pub(super) fn check(status: i32, operation: &str) -> Result<()> {
    if status != 0 {
        let text = unsafe { CStr::from_ptr(hipGetErrorString(status)) }.to_string_lossy();
        anyhow::bail!("{operation}: HIP status {status}: {text}");
    }
    Ok(())
}

pub(super) fn drop_result(result: Result<()>) {
    if let Err(error) = result {
        eprintln!("HIP resource teardown failed: {error:#}");
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
pub struct Properties {
    pub name: [u8; 256],
    pub architecture: [u8; 256],
    pub uuid: [u8; 16],
    pub optin_shared_bytes: u64,
    pub warp: i32,
    pub compute_units: i32,
    pub max_threads: i32,
    pub shared_bytes: i32,
    pub unit_shared_bytes: i32,
    pub l2_bytes: i32,
    pub memory_clock_khz: i32,
    pub memory_bus_bits: i32,
    pub cooperative: i32,
    pub driver_version: i32,
    pub runtime_version: i32,
}

pub struct Context {
    ordinal: usize,
    pub properties: Properties,
    architecture: String,
}

impl Context {
    pub fn device_info(&self) -> Result<super::setup::DeviceInfo> {
        let p = &self.properties;
        Ok(super::setup::DeviceInfo {
            name: self.name()?,
            arch: super::setup::DeviceArchitecture::Hip(self.architecture.clone()),
            sms: usize::try_from(p.compute_units)?,
            l2_bytes: usize::try_from(p.l2_bytes)?,
            warp: usize::try_from(p.warp)?,
            max_threads: u32::try_from(p.max_threads)?,
            shared_bytes: usize::try_from(p.shared_bytes)?,
            sm_shared_bytes: usize::try_from(p.unit_shared_bytes)?,
            cooperative: p.cooperative != 0,
            optin_shared_bytes: usize::try_from(p.optin_shared_bytes)?,
            memory_bytes_per_second: 2.0
                * f64::from(p.memory_clock_khz)
                * 1000.0
                * f64::from(p.memory_bus_bits)
                / 8.0,
        })
    }
    pub fn new(ordinal: usize) -> Result<Arc<Self>> {
        unsafe {
            check(hipInit(0), "hipInit")?;
        }
        let device = i32::try_from(ordinal)?;
        unsafe {
            check(hipSetDevice(device), "hipSetDevice")?;
        }
        let mut properties = std::mem::MaybeUninit::<Properties>::uninit();
        unsafe {
            check(
                ff_hip_device_properties(device, properties.as_mut_ptr()),
                "HIP device properties",
            )?;
        }
        let properties = unsafe { properties.assume_init() };
        let architecture = CStr::from_bytes_until_nul(&properties.architecture)?
            .to_str()?
            .split(':')
            .next()
            .context("HIP architecture missing")?
            .to_owned();
        let digits = architecture
            .strip_prefix("gfx")
            .context("HIP architecture lacks gfx prefix")?;
        ensure!(
            digits.len() == 4
                && digits.bytes().all(|v| v.is_ascii_digit())
                && matches!(&digits[..2], "10" | "11" | "12")
                && properties.warp == 32,
            "unsupported HIP architecture {architecture} wave{}: wave32 RDNA required",
            properties.warp
        );
        ensure!(
            properties.compute_units > 0
                && properties.max_threads >= 32
                && properties.shared_bytes > 0
                && properties.unit_shared_bytes > 0
                && properties.l2_bytes >= 0
                && properties.memory_clock_khz > 0
                && properties.memory_bus_bits > 0,
            "invalid HIP device resource attributes"
        );
        Ok(Arc::new(Self {
            ordinal,
            properties,
            architecture,
        }))
    }
    pub fn architecture(&self) -> &str {
        &self.architecture
    }
    pub fn name(&self) -> Result<String> {
        Ok(CStr::from_bytes_until_nul(&self.properties.name)?
            .to_str()?
            .to_owned())
    }
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }
    pub fn device_count() -> Result<usize> {
        let mut count = 0;
        unsafe {
            check(hipGetDeviceCount(&mut count), "hipGetDeviceCount")?;
        }
        Ok(usize::try_from(count)?)
    }
    pub fn bind_to_thread(&self) -> Result<()> {
        unsafe { check(hipSetDevice(self.ordinal as i32), "hipSetDevice") }
    }
    pub fn synchronize(&self) -> Result<()> {
        self.bind_to_thread()?;
        unsafe { check(hipDeviceSynchronize(), "hipDeviceSynchronize") }
    }
    pub fn mem_get_info(&self) -> Result<(usize, usize)> {
        self.bind_to_thread()?;
        let (mut free, mut total) = (0, 0);
        unsafe {
            check(hipMemGetInfo(&mut free, &mut total), "hipMemGetInfo")?;
        }
        Ok((free, total))
    }
    pub fn new_stream(self: &Arc<Self>) -> Result<Arc<Stream>> {
        self.bind_to_thread()?;
        let mut handle = std::ptr::null_mut();
        unsafe {
            check(
                hipStreamCreateWithFlags(&mut handle, 1),
                "hipStreamCreateWithFlags",
            )?;
        }
        Ok(Arc::new(Stream {
            handle: handle as usize,
            context: self.clone(),
        }))
    }
    pub fn new_event(self: &Arc<Self>, flags: Option<u32>) -> Result<Event> {
        self.bind_to_thread()?;
        let mut handle = std::ptr::null_mut();
        unsafe {
            check(
                hipEventCreateWithFlags(&mut handle, flags.unwrap_or(0)),
                "hipEventCreateWithFlags",
            )?;
        }
        Ok(Event {
            handle: handle as usize,
            context: self.clone(),
        })
    }
    /// # Safety
    /// The caller must initialize every element before reading it.
    pub unsafe fn alloc_pinned_with_flags<T: DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
        flags: u32,
    ) -> Result<PinnedHostSlice<T>> {
        self.bind_to_thread()?;
        let bytes = extent::<T>(len)?;
        ensure!(bytes > 0, "empty HIP pinned allocation");
        let mut ptr = std::ptr::null_mut();
        unsafe {
            check(hipHostMalloc(&mut ptr, bytes, flags), "hipHostMalloc")?;
        }
        Ok(PinnedHostSlice {
            ptr: ptr as usize,
            len,
            context: self.clone(),
            marker: PhantomData,
        })
    }
    pub fn load_binary(self: &Arc<Self>, bytes: &[u8]) -> Result<Arc<Module>> {
        self.bind_to_thread()?;
        ensure!(!bytes.is_empty(), "empty HIP module");
        let mut handle = std::ptr::null_mut();
        unsafe {
            check(
                hipModuleLoadData(&mut handle, bytes.as_ptr().cast()),
                "hipModuleLoadData",
            )?;
        }
        Ok(Arc::new(Module {
            handle: handle as usize,
            context: self.clone(),
        }))
    }
    pub fn require_simt(&self, operation: &str) -> Result<()> {
        ensure!(self.properties.warp == 32, "{operation} requires wave32");
        Ok(())
    }
}

pub struct Stream {
    handle: usize,
    context: Arc<Context>,
}
impl Stream {
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }
    pub fn raw(&self) -> Handle {
        self.handle as Handle
    }
    pub fn synchronize(&self) -> Result<()> {
        self.context.bind_to_thread()?;
        unsafe { check(hipStreamSynchronize(self.raw()), "hipStreamSynchronize") }
    }
    pub fn wait(&self, event: &Event) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.context, &event.context),
            "HIP event device differs from stream"
        );
        self.context.bind_to_thread()?;
        unsafe {
            check(
                hipStreamWaitEvent(self.raw(), event.handle as Handle, 0),
                "hipStreamWaitEvent",
            )
        }
    }
    pub fn record_event(self: &Arc<Self>, flags: Option<u32>) -> Result<Event> {
        let event = self.context.new_event(flags)?;
        event.record(self)?;
        Ok(event)
    }
    /// # Safety
    /// Contents must be initialized before a device or host read.
    pub unsafe fn alloc<T: DeviceRepr>(self: &Arc<Self>, len: usize) -> Result<Slice<T>> {
        self.context.bind_to_thread()?;
        let bytes = extent::<T>(len)?;
        let mut ptr = std::ptr::null_mut();
        if bytes > 0 {
            unsafe {
                check(hipMalloc(&mut ptr, bytes), "hipMalloc")?;
            }
        }
        Ok(Slice {
            allocation: Arc::new(Allocation {
                ptr: ptr as usize,
                stream: self.clone(),
            }),
            len,
            marker: PhantomData,
        })
    }
    pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<Slice<T>> {
        let mut output = unsafe { self.alloc(len)? };
        self.memset_zeros(&mut output)?;
        Ok(output)
    }
    pub fn clone_htod<T: DeviceRepr>(self: &Arc<Self>, input: &[T]) -> Result<Slice<T>> {
        let mut output = unsafe { self.alloc(input.len())? };
        self.memcpy_htod(input, &mut output)?;
        Ok(output)
    }
    pub fn memcpy_htod<T: DeviceRepr>(
        self: &Arc<Self>,
        input: &[T],
        output: &mut impl DevicePtrMut<T>,
    ) -> Result<()> {
        ensure!(input.len() <= output.len(), "HIP H2D exceeds allocation");
        output.check_stream(self)?;
        let (ptr, _guard) = output.device_ptr_mut(self);
        unsafe {
            self.copy_raw(ptr, input.as_ptr() as u64, extent::<T>(input.len())?, 1)?;
        }
        self.synchronize()
    }
    pub fn clone_dtoh<T: DeviceRepr>(
        self: &Arc<Self>,
        input: &impl DevicePtr<T>,
    ) -> Result<Vec<T>> {
        input.check_stream(self)?;
        let mut output = Vec::<T>::with_capacity(input.len());
        let (ptr, _guard) = input.device_ptr(self);
        unsafe {
            self.copy_raw(
                output.as_mut_ptr() as u64,
                ptr,
                extent::<T>(input.len())?,
                2,
            )?;
        }
        self.synchronize()?;
        unsafe {
            output.set_len(input.len());
        }
        Ok(output)
    }
    pub fn memcpy_dtod<T: DeviceRepr>(
        self: &Arc<Self>,
        input: &impl DevicePtr<T>,
        output: &mut impl DevicePtrMut<T>,
    ) -> Result<()> {
        ensure!(input.len() <= output.len(), "HIP D2D exceeds allocation");
        input.check_stream(self)?;
        output.check_stream(self)?;
        let (src, _read) = input.device_ptr(self);
        let (dst, _write) = output.device_ptr_mut(self);
        unsafe { self.copy_raw(dst, src, extent::<T>(input.len())?, 3) }
    }
    pub fn memset_zeros<T: DeviceRepr + ValidAsZeroBits>(
        self: &Arc<Self>,
        output: &mut impl DevicePtrMut<T>,
    ) -> Result<()> {
        self.context.bind_to_thread()?;
        let bytes = extent::<T>(output.len())?;
        output.check_stream(self)?;
        let (ptr, _guard) = output.device_ptr_mut(self);
        if bytes > 0 {
            unsafe {
                check(
                    hipMemsetAsync(ptr as Handle, 0, bytes, self.raw()),
                    "hipMemsetAsync",
                )?;
            }
        }
        Ok(())
    }
    /// # Safety
    /// Both spans must be valid for `bytes` and ordered on this stream through completion.
    pub unsafe fn copy_raw(&self, dst: u64, src: u64, bytes: usize, kind: i32) -> Result<()> {
        self.context.bind_to_thread()?;
        if bytes > 0 {
            unsafe {
                check(
                    hipMemcpyAsync(dst as Handle, src as *const c_void, bytes, kind, self.raw()),
                    "hipMemcpyAsync",
                )?;
            }
        }
        Ok(())
    }
    /// # Safety
    /// The pitched spans must be live, disjoint and ordered on this stream.
    pub unsafe fn copy_2d(
        &self,
        dst: u64,
        src: u64,
        pitch: (usize, usize),
        extent: (usize, usize),
    ) -> Result<()> {
        self.context.bind_to_thread()?;
        ensure!(
            extent.0 <= pitch.0 && extent.0 <= pitch.1,
            "HIP copy width exceeds pitch"
        );
        unsafe {
            check(
                hipMemcpy2DAsync(
                    dst as Handle,
                    pitch.0,
                    src as *const c_void,
                    pitch.1,
                    extent.0,
                    extent.1,
                    3,
                    self.raw(),
                ),
                "hipMemcpy2DAsync",
            )
        }
    }
    /// # Safety
    /// `ptr` must address `count` writable u32 words ordered on this stream.
    pub unsafe fn memset_words(&self, ptr: u64, value: u32, count: usize) -> Result<()> {
        self.context.bind_to_thread()?;
        unsafe {
            check(
                hipMemsetD32Async(ptr as Handle, value as i32, count, self.raw()),
                "hipMemsetD32Async",
            )
        }
    }
    pub fn launch_builder<'a>(self: &'a Arc<Self>, function: &'a Function) -> Launch<'a> {
        Launch {
            stream: self,
            function,
            arguments: [[0; 8]; 32],
            allocations: std::array::from_fn(|_| None),
            count: 0,
        }
    }
}
impl Drop for Stream {
    fn drop(&mut self) {
        drop_result(
            self.synchronize()
                .and_then(|()| unsafe { check(hipStreamDestroy(self.raw()), "hipStreamDestroy") }),
        );
    }
}

pub struct Event {
    handle: usize,
    context: Arc<Context>,
}
impl Event {
    pub fn record(&self, stream: &Arc<Stream>) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.context, &stream.context),
            "HIP event/stream context mismatch"
        );
        self.context.bind_to_thread()?;
        unsafe {
            check(
                hipEventRecord(self.handle as Handle, stream.raw()),
                "hipEventRecord",
            )
        }
    }
    pub fn synchronize(&self) -> Result<()> {
        self.context.bind_to_thread()?;
        unsafe {
            check(
                hipEventSynchronize(self.handle as Handle),
                "hipEventSynchronize",
            )
        }
    }
    pub fn elapsed_ms(&self, end: &Self) -> Result<f32> {
        ensure!(
            Arc::ptr_eq(&self.context, &end.context),
            "HIP elapsed event context mismatch"
        );
        self.context.bind_to_thread()?;
        let mut ms = 0.0;
        unsafe {
            check(
                hipEventElapsedTime(&mut ms, self.handle as Handle, end.handle as Handle),
                "hipEventElapsedTime",
            )?;
        }
        Ok(ms)
    }
}
impl Drop for Event {
    fn drop(&mut self) {
        drop_result(self.context.bind_to_thread().and_then(|()| unsafe {
            check(hipEventDestroy(self.handle as Handle), "hipEventDestroy")
        }));
    }
}

/// # Safety
/// Every bit pattern written by the device must be valid for this plain-data type.
pub unsafe trait DeviceRepr: Copy {}
/// # Safety
/// All-zero bytes must represent a valid value.
pub unsafe trait ValidAsZeroBits: DeviceRepr {}
macro_rules! plain {
    ($($t:ty),*) => { $(unsafe impl DeviceRepr for $t {} unsafe impl ValidAsZeroBits for $t {})* };
}
plain!(u8, u16, u32, u64, i32, f32, half::bf16);

fn extent<T>(len: usize) -> Result<usize> {
    len.checked_mul(size_of::<T>())
        .filter(|&v| v <= isize::MAX as usize)
        .context("HIP allocation extent overflow")
}
struct Allocation {
    ptr: usize,
    stream: Arc<Stream>,
}
impl Drop for Allocation {
    fn drop(&mut self) {
        if self.ptr != 0 {
            drop_result(
                self.stream
                    .context
                    .synchronize()
                    .and_then(|()| unsafe { check(hipFree(self.ptr as Handle), "hipFree") }),
            );
        }
    }
}
pub struct Slice<T> {
    allocation: Arc<Allocation>,
    len: usize,
    marker: PhantomData<T>,
}
pub struct View<'a, T> {
    allocation: &'a Arc<Allocation>,
    offset: usize,
    len: usize,
    marker: PhantomData<&'a T>,
}
pub struct ViewMut<'a, T> {
    allocation: &'a Arc<Allocation>,
    offset: usize,
    len: usize,
    marker: PhantomData<&'a mut T>,
}

fn bounds(range: impl RangeBounds<usize>, len: usize) -> (usize, usize) {
    let start = match range.start_bound() {
        Bound::Included(&v) => v,
        Bound::Excluded(&v) => v.checked_add(1).expect("HIP slice start overflow"),
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&v) => v.checked_add(1).expect("HIP slice end overflow"),
        Bound::Excluded(&v) => v,
        Bound::Unbounded => len,
    };
    assert!(start <= end && end <= len, "HIP slice outside allocation");
    (start, end - start)
}
impl<T> Slice<T> {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn stream(&self) -> &Arc<Stream> {
        &self.allocation.stream
    }
    /// Transfers exclusive stream ownership after fencing all prior owner-stream work.
    pub fn handoff(&mut self, stream: &Arc<Stream>) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&stream.context, &self.allocation.stream.context),
            "HIP handoff context mismatch"
        );
        if Arc::ptr_eq(stream, &self.allocation.stream) {
            return Ok(());
        }
        let allocation = Arc::get_mut(&mut self.allocation)
            .context("HIP handoff requires exclusive allocation ownership")?;
        let completed = allocation.stream.record_event(Some(2))?;
        stream.wait(&completed)?;
        allocation.stream = stream.clone();
        Ok(())
    }
    pub fn slice(&self, range: impl RangeBounds<usize>) -> View<'_, T> {
        let (offset, len) = bounds(range, self.len);
        View {
            allocation: &self.allocation,
            offset,
            len,
            marker: PhantomData,
        }
    }
    pub fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> ViewMut<'_, T> {
        let (offset, len) = bounds(range, self.len);
        ViewMut {
            allocation: &self.allocation,
            offset,
            len,
            marker: PhantomData,
        }
    }
}
pub struct PointerGuard<'a>(PhantomData<&'a Allocation>);
impl<T> View<'_, T> {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}
impl<T> ViewMut<'_, T> {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}
/// # Safety
/// The returned pointer must cover `len` elements and remain live for the guard's borrow.
pub unsafe trait DevicePtr<T> {
    fn len(&self) -> usize;
    fn check_stream(&self, stream: &Arc<Stream>) -> Result<()>;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn device_ptr<'a>(&'a self, stream: &'a Arc<Stream>) -> (u64, PointerGuard<'a>);
}
/// # Safety
/// The returned span must be exclusively writable during the mutable borrow.
pub unsafe trait DevicePtrMut<T>: DevicePtr<T> {
    fn device_ptr_mut<'a>(&'a mut self, stream: &'a Arc<Stream>) -> (u64, PointerGuard<'a>);
}
macro_rules! pointer {
    ($ty:ty, $offset:expr) => {
        unsafe impl<T> DevicePtr<T> for $ty {
            fn len(&self) -> usize {
                self.len
            }
            fn check_stream(&self, stream: &Arc<Stream>) -> Result<()> {
                ensure!(
                    Arc::ptr_eq(stream, &self.allocation.stream),
                    "HIP cross-stream access requires Slice::handoff with an explicit fence"
                );
                Ok(())
            }
            fn device_ptr<'a>(&'a self, stream: &'a Arc<Stream>) -> (u64, PointerGuard<'a>) {
                self.check_stream(stream)
                    .expect("HIP pointer stream ownership");
                (
                    (self.allocation.ptr + $offset(self) * size_of::<T>()) as u64,
                    PointerGuard(PhantomData),
                )
            }
        }
    };
}
pointer!(Slice<T>, |_: &Slice<T>| 0);
pointer!(View<'_, T>, |v: &View<'_, T>| v.offset);
pointer!(ViewMut<'_, T>, |v: &ViewMut<'_, T>| v.offset);
unsafe impl<T> DevicePtrMut<T> for Slice<T> {
    fn device_ptr_mut<'a>(&'a mut self, s: &'a Arc<Stream>) -> (u64, PointerGuard<'a>) {
        self.device_ptr(s)
    }
}
unsafe impl<T> DevicePtrMut<T> for ViewMut<'_, T> {
    fn device_ptr_mut<'a>(&'a mut self, s: &'a Arc<Stream>) -> (u64, PointerGuard<'a>) {
        self.device_ptr(s)
    }
}

pub struct PinnedHostSlice<T> {
    ptr: usize,
    len: usize,
    context: Arc<Context>,
    marker: PhantomData<T>,
}
impl<T> PinnedHostSlice<T> {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn as_slice(&self) -> Result<&[T]> {
        self.context.synchronize()?;
        Ok(unsafe { std::slice::from_raw_parts(self.ptr as *const T, self.len) })
    }
    pub fn as_mut_slice(&mut self) -> Result<&mut [T]> {
        self.context.synchronize()?;
        Ok(unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, self.len) })
    }
    pub fn as_ptr(&self) -> Result<*const T> {
        Ok(self.ptr as *const T)
    }
    pub fn as_mut_ptr(&mut self) -> Result<*mut T> {
        Ok(self.ptr as *mut T)
    }
}
impl<T> Drop for PinnedHostSlice<T> {
    fn drop(&mut self) {
        drop_result(
            self.context
                .synchronize()
                .and_then(|()| unsafe { check(hipHostFree(self.ptr as Handle), "hipHostFree") }),
        );
    }
}

pub struct Module {
    handle: usize,
    context: Arc<Context>,
}
impl Module {
    pub fn load_function(self: &Arc<Self>, name: &str) -> Result<Function> {
        self.context.bind_to_thread()?;
        let name = CString::new(name)?;
        let mut handle = std::ptr::null_mut();
        unsafe {
            check(
                hipModuleGetFunction(&mut handle, self.handle as Handle, name.as_ptr()),
                "hipModuleGetFunction",
            )?;
        }
        Ok(Function {
            handle: handle as usize,
            module: self.clone(),
        })
    }
}
impl Drop for Module {
    fn drop(&mut self) {
        drop_result(self.context.synchronize().and_then(|()| unsafe {
            check(hipModuleUnload(self.handle as Handle), "hipModuleUnload")
        }));
    }
}
#[derive(Clone)]
pub struct Function {
    handle: usize,
    module: Arc<Module>,
}
impl Function {
    pub fn max_threads_per_block(&self) -> Result<i32> {
        self.module.context.bind_to_thread()?;
        let mut value = 0;
        unsafe {
            check(
                hipFuncGetAttribute(&mut value, 0, self.handle as Handle),
                "hipFuncGetAttribute(maxThreads)",
            )?;
        }
        Ok(value)
    }
    pub fn occupancy_max_active_blocks_per_multiprocessor(
        &self,
        threads: u32,
        shared: usize,
        flags: Option<u32>,
    ) -> Result<i32> {
        ensure!(flags.is_none(), "HIP occupancy flags are unsupported");
        self.module.context.bind_to_thread()?;
        let mut count = 0;
        unsafe {
            check(
                hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
                    &mut count,
                    self.handle as Handle,
                    i32::try_from(threads)?,
                    shared,
                ),
                "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",
            )?;
        }
        Ok(count)
    }
}
#[derive(Clone, Copy)]
pub struct LaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}
pub struct Launch<'a> {
    stream: &'a Arc<Stream>,
    function: &'a Function,
    arguments: [[u64; 8]; 32],
    allocations: [Option<Arc<Allocation>>; 32],
    count: usize,
}
pub trait PushKernelArg<A> {
    fn arg(&mut self, value: A) -> &mut Self;
}
impl Launch<'_> {
    fn push<T: Copy>(&mut self, value: &T) {
        assert!(
            self.count < self.arguments.len(),
            "HIP_KERNEL_ARGUMENT_LIMIT=32"
        );
        assert!(
            size_of::<T>() <= 64 && align_of::<T>() <= 8,
            "HIP_ARGUMENT_BYTES_LIMIT=64 alignment_limit=8"
        );
        let bytes = &mut self.arguments[self.count];
        unsafe {
            std::ptr::copy_nonoverlapping(
                (value as *const T).cast::<u8>(),
                bytes.as_mut_ptr().cast(),
                size_of::<T>(),
            );
        }
        self.count += 1;
    }
    /// # Safety
    /// Raw pointer arguments, aliasing, launch shape and inter-stream dependencies must match the kernel contract.
    pub unsafe fn launch(&mut self, config: LaunchConfig) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.stream.context, &self.function.module.context),
            "HIP launch context mismatch"
        );
        ensure!(
            [
                config.grid_dim.0,
                config.grid_dim.1,
                config.grid_dim.2,
                config.block_dim.0,
                config.block_dim.1,
                config.block_dim.2
            ]
            .iter()
            .all(|&v| v > 0),
            "empty HIP launch dimension"
        );
        self.stream.context.bind_to_thread()?;
        let mut pointers = [std::ptr::null_mut(); 32];
        for (pointer, argument) in pointers
            .iter_mut()
            .zip(&mut self.arguments)
            .take(self.count)
        {
            *pointer = argument.as_mut_ptr().cast();
        }
        unsafe {
            check(
                hipModuleLaunchKernel(
                    self.function.handle as Handle,
                    config.grid_dim.0,
                    config.grid_dim.1,
                    config.grid_dim.2,
                    config.block_dim.0,
                    config.block_dim.1,
                    config.block_dim.2,
                    config.shared_mem_bytes,
                    self.stream.raw(),
                    pointers.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "hipModuleLaunchKernel",
            )
        }
    }
}
impl<T: DeviceRepr> PushKernelArg<&T> for Launch<'_> {
    fn arg(&mut self, value: &T) -> &mut Self {
        self.push(value);
        self
    }
}
macro_rules! argument {
    ($ty:ty) => {
        impl<T> PushKernelArg<&$ty> for Launch<'_> {
            fn arg(&mut self, value: &$ty) -> &mut Self {
                self.push(&value.device_ptr(self.stream).0);
                self.allocations[self.count - 1] = Some(value.allocation.clone());
                self
            }
        }
        impl<T> PushKernelArg<&mut $ty> for Launch<'_> {
            fn arg(&mut self, value: &mut $ty) -> &mut Self {
                self.push(&value.device_ptr(self.stream).0);
                self.allocations[self.count - 1] = Some(value.allocation.clone());
                self
            }
        }
    };
}
argument!(Slice<T>);
argument!(View<'_, T>);
argument!(ViewMut<'_, T>);
