//! TPU execution backend built directly on the PJRT C API and StableHLO.

pub mod dit_block;
pub mod hlo;
pub mod minimax_h3;
pub mod pjrt;
pub mod plan;

use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use pjrt::{
    AttrValue, BufferType, CompiledMemoryStats, HostBufferSpec, HostSemantics, MemoryKindInfo,
    MemoryStats, PjrtApi,
};
use serde::Serialize;
use std::{collections::BTreeMap, ffi::c_void, path::Path, ptr, rc::Rc, slice, sync::Arc};

#[derive(Clone, Debug, Serialize)]
pub struct DeviceAttributes {
    pub pjrt_major: i32,
    pub pjrt_minor: i32,
    pub pjrt_api_struct_size: usize,
    pub platform_name: String,
    pub platform_version: String,
    pub plugin_attributes: BTreeMap<String, AttrValue>,
    pub total_devices: usize,
    pub addressable_devices: usize,
    pub device_id: i32,
    pub local_hardware_id: i32,
    pub chip_kind: String,
    pub debug_string: String,
    pub device_attributes: BTreeMap<String, AttrValue>,
    pub default_memory_kind: String,
    pub addressable_memories: Vec<MemoryKindInfo>,
    pub memory_stats: MemoryStats,
}

struct ClientInner {
    api: Arc<PjrtApi>,
    client: *mut c_void,
}

impl Drop for ClientInner {
    fn drop(&mut self) {
        self.api.destroy_client(self.client);
    }
}

pub struct TpuDevice {
    inner: Rc<ClientInner>,
    device: *mut c_void,
    default_memory: *mut c_void,
    pub attributes: DeviceAttributes,
}

impl TpuDevice {
    /// Opens `FF_TPU_LIB` when set, otherwise loads `libtpu.so` from the dynamic linker search path.
    pub fn open_default(device_index: usize) -> Result<Self> {
        let lib_path = std::env::var_os("FF_TPU_LIB")
            .filter(|v| !v.is_empty())
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("libtpu.so"));
        Self::open(&lib_path, device_index)
    }

    pub fn open(library_path: &Path, device_index: usize) -> Result<Self> {
        let api = PjrtApi::load(library_path)?;
        let plugin_attributes = api.plugin_attributes()?;
        let client = api.create_client()?;
        let inner = Rc::new(ClientInner {
            api: Arc::clone(&api),
            client,
        });
        let platform_name = api.client_platform_name(client)?;
        let platform_version = api.client_platform_version(client)?;
        let all_devices = api.client_devices(client, false)?;
        let addressable = api.client_devices(client, true)?;
        ensure!(
            device_index < addressable.len(),
            "TPU device index {device_index} out of range (addressable={})",
            addressable.len()
        );
        let device = addressable[device_index];
        ensure!(
            !device.is_null(),
            "TPU addressable device {device_index} handle is null"
        );
        let desc = api.device_description(device)?;
        let device_id = api.description_id(desc)?;
        let local_hardware_id = api.device_local_hardware_id(device)?;
        let chip_kind = api.description_kind(desc)?;
        let debug_string = api.description_debug_string(desc)?;
        let device_attributes = api.description_attributes(desc)?;
        let default_memory = api.device_default_memory(device)?;
        let default_memory_kind = api.memory_kind(default_memory)?;
        let addressable_memories = api.device_addressable_memories(device)?;
        let memory_stats = api.device_memory_stats(device)?;

        let attributes = DeviceAttributes {
            pjrt_major: api.major_version,
            pjrt_minor: api.minor_version,
            pjrt_api_struct_size: api.struct_size,
            platform_name,
            platform_version,
            plugin_attributes,
            total_devices: all_devices.len(),
            addressable_devices: addressable.len(),
            device_id,
            local_hardware_id,
            chip_kind,
            debug_string,
            device_attributes,
            default_memory_kind,
            addressable_memories,
            memory_stats,
        };
        Ok(Self {
            inner,
            device,
            default_memory,
            attributes,
        })
    }

    pub fn architecture(&self) -> crate::backend::setup::DeviceArchitecture {
        crate::backend::setup::DeviceArchitecture::Tpu(self.attributes.chip_kind.clone())
    }

    pub fn memory_stats(&self) -> Result<MemoryStats> {
        self.inner.api.device_memory_stats(self.device)
    }

    pub fn compile_mlir(&self, mlir: &str) -> Result<TpuExecutable> {
        let handle =
            self.inner
                .api
                .compile_mlir(self.inner.client, self.attributes.device_id, mlir)?;
        let (num_outputs, compiled_memory_stats) =
            self.inner.api.loaded_executable_metadata(handle)?;
        Ok(TpuExecutable {
            inner: Rc::clone(&self.inner),
            device: self.device,
            handle,
            num_outputs,
            compiled_memory_stats: compiled_memory_stats.ok(),
        })
    }

    pub fn pinned_host_memory(&self) -> Option<&MemoryKindInfo> {
        self.attributes
            .addressable_memories
            .iter()
            .find(|m| m.kind == "pinned_host")
    }

    pub fn upload_bf16(
        &self,
        data: &[bf16],
        dims: &[i64],
        semantics: HostSemantics,
    ) -> Result<TpuBuffer> {
        self.upload_bf16_to_memory(data, dims, semantics, None)
    }

    pub fn upload_bf16_to_memory(
        &self,
        data: &[bf16],
        dims: &[i64],
        semantics: HostSemantics,
        target_memory: Option<&MemoryKindInfo>,
    ) -> Result<TpuBuffer> {
        let elems = checked_elements(dims)?;
        ensure!(
            data.len() == elems,
            "upload_bf16 length {} != dims product {elems}",
            data.len()
        );
        let (device, memory) = match target_memory {
            Some(info) => (ptr::null_mut(), info.handle),
            None => (self.device, ptr::null_mut()),
        };
        let handle = unsafe {
            self.inner.api.buffer_from_host_buffer(
                self.inner.client,
                HostBufferSpec {
                    data: data.as_ptr() as *const c_void,
                    buffer_type: BufferType::Bf16,
                    dims,
                    semantics,
                    device,
                    memory,
                },
            )?
        };
        Ok(TpuBuffer {
            inner: Rc::clone(&self.inner),
            handle,
            byte_len: elems * BufferType::Bf16.byte_width(),
        })
    }

    pub fn upload_i32(
        &self,
        data: &[i32],
        dims: &[i64],
        semantics: HostSemantics,
    ) -> Result<TpuBuffer> {
        let elems = checked_elements(dims)?;
        ensure!(
            data.len() == elems,
            "upload_i32 length {} != dims product {elems}",
            data.len()
        );
        let handle = unsafe {
            self.inner.api.buffer_from_host_buffer(
                self.inner.client,
                HostBufferSpec {
                    data: data.as_ptr() as *const c_void,
                    buffer_type: BufferType::S32,
                    dims,
                    semantics,
                    device: self.device,
                    memory: ptr::null_mut(),
                },
            )?
        };
        Ok(TpuBuffer {
            inner: Rc::clone(&self.inner),
            handle,
            byte_len: elems * BufferType::S32.byte_width(),
        })
    }

    pub fn upload_u8(
        &self,
        data: &[u8],
        semantics: HostSemantics,
        target_memory: Option<&MemoryKindInfo>,
    ) -> Result<TpuBuffer> {
        let dims = [i64::try_from(data.len()).context("u8 buffer length overflows i64")?];
        let (device, memory) = match target_memory {
            Some(info) => (ptr::null_mut(), info.handle),
            None => (self.device, ptr::null_mut()),
        };
        let handle = unsafe {
            self.inner.api.buffer_from_host_buffer(
                self.inner.client,
                HostBufferSpec {
                    data: data.as_ptr() as *const c_void,
                    buffer_type: BufferType::U8,
                    dims: &dims,
                    semantics,
                    device,
                    memory,
                },
            )?
        };
        Ok(TpuBuffer {
            inner: Rc::clone(&self.inner),
            handle,
            byte_len: data.len(),
        })
    }

    pub fn upload_f32(
        &self,
        data: &[f32],
        dims: &[i64],
        semantics: HostSemantics,
    ) -> Result<TpuBuffer> {
        self.upload_f32_to_memory(data, dims, semantics, None)
    }

    pub fn upload_f32_to_memory(
        &self,
        data: &[f32],
        dims: &[i64],
        semantics: HostSemantics,
        target_memory: Option<&MemoryKindInfo>,
    ) -> Result<TpuBuffer> {
        let elems = checked_elements(dims)?;
        ensure!(
            data.len() == elems,
            "upload_f32 length {} != dims product {elems}",
            data.len()
        );
        let (device, memory) = match target_memory {
            Some(info) => (ptr::null_mut(), info.handle),
            None => (self.device, ptr::null_mut()),
        };
        let handle = unsafe {
            self.inner.api.buffer_from_host_buffer(
                self.inner.client,
                HostBufferSpec {
                    data: data.as_ptr() as *const c_void,
                    buffer_type: BufferType::F32,
                    dims,
                    semantics,
                    device,
                    memory,
                },
            )?
        };
        Ok(TpuBuffer {
            inner: Rc::clone(&self.inner),
            handle,
            byte_len: elems * BufferType::F32.byte_width(),
        })
    }

    pub fn copy_buffer_to_default_hbm(&self, src: &TpuBuffer) -> Result<TpuBuffer> {
        let handle = self
            .inner
            .api
            .buffer_copy_to_memory(src.handle, self.default_memory)?;
        Ok(TpuBuffer {
            inner: Rc::clone(&self.inner),
            handle,
            byte_len: src.byte_len,
        })
    }

    pub fn copy_buffer_to_default_hbm_async(&self, src: &TpuBuffer) -> Result<TpuBuffer> {
        let handle = self
            .inner
            .api
            .buffer_copy_to_memory_async(src.handle, self.default_memory)?;
        Ok(TpuBuffer {
            inner: Rc::clone(&self.inner),
            handle,
            byte_len: src.byte_len,
        })
    }

    pub fn alloc_dma_mapped(&self, len: usize) -> Result<DmaMappedHostBuffer> {
        ensure!(len > 0, "DMA-mapped buffer length must be positive");
        let mut raw: *mut c_void = ptr::null_mut();
        // SAFETY: 4096-byte page alignment is a power of two and multiple of sizeof(void*).
        let rc = unsafe { libc::posix_memalign(&mut raw, 4096, len) };
        if rc != 0 || raw.is_null() {
            bail!("posix_memalign(4096, {len}) failed: errno={rc}");
        }
        // Touch pages and attempt mlock (non-fatal if RLIMIT_MEMLOCK denies it; status is reported).
        unsafe { ptr::write_bytes(raw as *mut u8, 0, len) };
        let mlock_rc = unsafe { libc::mlock(raw, len) };
        let mlocked = mlock_rc == 0;
        let mlock_errno = if mlocked {
            0
        } else {
            std::io::Error::last_os_error().raw_os_error().unwrap_or(-1)
        };
        if let Err(err) = unsafe { self.inner.api.dma_map(self.inner.client, raw, len) } {
            if mlocked {
                unsafe { libc::munlock(raw, len) };
            }
            unsafe { libc::free(raw) };
            return Err(err);
        }
        Ok(DmaMappedHostBuffer {
            inner: Rc::clone(&self.inner),
            ptr: raw,
            len,
            mlocked,
            mlock_errno,
        })
    }
}

pub struct DmaMappedHostBuffer {
    inner: Rc<ClientInner>,
    ptr: *mut c_void,
    len: usize,
    pub mlocked: bool,
    pub mlock_errno: i32,
}

impl DmaMappedHostBuffer {
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr as *mut u8, self.len) }
    }
}

impl Drop for DmaMappedHostBuffer {
    fn drop(&mut self) {
        let _ = unsafe { self.inner.api.dma_unmap(self.inner.client, self.ptr) };
        if self.mlocked {
            unsafe { libc::munlock(self.ptr, self.len) };
        }
        unsafe { libc::free(self.ptr) };
    }
}

pub struct TpuBuffer {
    inner: Rc<ClientInner>,
    handle: *mut c_void,
    byte_len: usize,
}

impl TpuBuffer {
    pub fn on_device_size_in_bytes(&self) -> Result<usize> {
        self.inner.api.buffer_on_device_size(self.handle)
    }

    pub fn download_bytes(&self) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.byte_len];
        self.inner.api.buffer_to_host(self.handle, &mut out)?;
        Ok(out)
    }

    pub fn await_ready(&self) -> Result<()> {
        self.inner.api.await_buffer_ready(self.handle)
    }

    pub fn download_bf16(&self) -> Result<Vec<bf16>> {
        ensure!(
            self.byte_len.is_multiple_of(2),
            "BF16 buffer byte length {} is odd",
            self.byte_len
        );
        let bytes = self.download_bytes()?;
        Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| bf16::from_le_bytes(c))
            .collect())
    }

    pub fn download_f32(&self) -> Result<Vec<f32>> {
        ensure!(
            self.byte_len.is_multiple_of(4),
            "F32 buffer byte length {} is not divisible by 4",
            self.byte_len
        );
        let bytes = self.download_bytes()?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect())
    }
}

impl Drop for TpuBuffer {
    fn drop(&mut self) {
        self.inner.api.destroy_buffer(self.handle);
    }
}

pub struct TpuEvent {
    inner: Rc<ClientInner>,
    handle: *mut c_void,
}

impl TpuEvent {
    pub fn wait(mut self) -> Result<()> {
        let h = self.handle;
        self.handle = ptr::null_mut();
        self.inner
            .api
            .await_and_destroy_event(h, "PJRT_LoadedExecutable_Execute")
    }
}

impl Drop for TpuEvent {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let h = self.handle;
            self.handle = ptr::null_mut();
            let _ = self.inner.api.await_and_destroy_event(h, "TpuEvent::drop");
        }
    }
}

pub struct TpuExecutable {
    inner: Rc<ClientInner>,
    device: *mut c_void,
    handle: *mut c_void,
    pub num_outputs: usize,
    pub compiled_memory_stats: Option<CompiledMemoryStats>,
}

impl TpuExecutable {
    pub fn execute(
        &self,
        inputs: &[&TpuBuffer],
        output_byte_lens: &[usize],
    ) -> Result<Vec<TpuBuffer>> {
        let (outputs, event) = self.execute_async(inputs, output_byte_lens)?;
        event.wait()?;
        Ok(outputs)
    }

    pub fn execute_async(
        &self,
        inputs: &[&TpuBuffer],
        output_byte_lens: &[usize],
    ) -> Result<(Vec<TpuBuffer>, TpuEvent)> {
        ensure!(
            output_byte_lens.len() == self.num_outputs,
            "executable produces {} outputs, caller supplied {} byte lengths",
            self.num_outputs,
            output_byte_lens.len()
        );
        let handles: Vec<*mut c_void> = inputs.iter().map(|b| b.handle).collect();
        let (raw_outs, event_handle) = self.inner.api.execute_single_device_async(
            self.handle,
            self.device,
            &handles,
            self.num_outputs,
        )?;
        let outputs = raw_outs
            .into_iter()
            .zip(output_byte_lens.iter().copied())
            .map(|(handle, byte_len)| TpuBuffer {
                inner: Rc::clone(&self.inner),
                handle,
                byte_len,
            })
            .collect();
        Ok((
            outputs,
            TpuEvent {
                inner: Rc::clone(&self.inner),
                handle: event_handle,
            },
        ))
    }
}

impl Drop for TpuExecutable {
    fn drop(&mut self) {
        self.inner.api.destroy_loaded_executable(self.handle);
    }
}

fn checked_elements(dims: &[i64]) -> Result<usize> {
    let mut total: usize = 1;
    for &d in dims {
        let u = usize::try_from(d).with_context(|| format!("negative dimension {d}"))?;
        total = total
            .checked_mul(u)
            .with_context(|| format!("dimension product overflows usize: {dims:?}"))?;
    }
    Ok(total)
}
