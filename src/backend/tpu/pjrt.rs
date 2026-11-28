//! Minimal FFI bindings to the official OpenXLA PJRT C API (`xla/pjrt/c/pjrt_c_api.h`).

use anyhow::{Context, Result, bail, ensure};
use std::{
    collections::BTreeMap,
    ffi::{CStr, CString, c_char, c_int, c_void},
    mem::{offset_of, size_of},
    path::Path,
    ptr, slice,
    sync::{Arc, Mutex},
};

pub(crate) const PJRT_API_MAJOR_EXPECTED: c_int = 0;

const SLOT_ERROR_DESTROY: usize = 3;
const SLOT_ERROR_MESSAGE: usize = 4;
const SLOT_ERROR_GET_CODE: usize = 5;
const SLOT_PLUGIN_INITIALIZE: usize = 6;
const SLOT_PLUGIN_ATTRIBUTES: usize = 7;
const SLOT_EVENT_DESTROY: usize = 8;
const SLOT_EVENT_ERROR: usize = 10;
const SLOT_EVENT_AWAIT: usize = 11;
const SLOT_CLIENT_CREATE: usize = 13;
const SLOT_CLIENT_DESTROY: usize = 14;
const SLOT_CLIENT_PLATFORM_NAME: usize = 15;
const SLOT_CLIENT_PLATFORM_VERSION: usize = 17;
const SLOT_CLIENT_DEVICES: usize = 18;
const SLOT_CLIENT_ADDRESSABLE_DEVICES: usize = 19;
const SLOT_CLIENT_COMPILE: usize = 23;
const SLOT_CLIENT_BUFFER_FROM_HOST_BUFFER: usize = 25;
const SLOT_DEVICE_DESCRIPTION_ID: usize = 26;
const SLOT_DEVICE_DESCRIPTION_ATTRIBUTES: usize = 28;
const SLOT_DEVICE_DESCRIPTION_KIND: usize = 29;
const SLOT_DEVICE_DESCRIPTION_DEBUG_STRING: usize = 30;
const SLOT_DEVICE_GET_DESCRIPTION: usize = 32;
const SLOT_DEVICE_LOCAL_HARDWARE_ID: usize = 34;
const SLOT_DEVICE_ADDRESSABLE_MEMORIES: usize = 35;
const SLOT_DEVICE_DEFAULT_MEMORY: usize = 36;
const SLOT_DEVICE_MEMORY_STATS: usize = 37;
const SLOT_MEMORY_KIND: usize = 39;
const SLOT_EXECUTABLE_DESTROY: usize = 43;
const SLOT_EXECUTABLE_NUM_OUTPUTS: usize = 47;
const SLOT_LOADED_EXECUTABLE_DESTROY: usize = 53;
const SLOT_LOADED_EXECUTABLE_GET_EXECUTABLE: usize = 54;
const SLOT_LOADED_EXECUTABLE_EXECUTE: usize = 58;
const SLOT_BUFFER_DESTROY: usize = 61;
const SLOT_BUFFER_DIMENSIONS: usize = 63;
const SLOT_BUFFER_ON_DEVICE_SIZE_IN_BYTES: usize = 67;
const SLOT_BUFFER_TO_HOST_BUFFER: usize = 73;
const SLOT_BUFFER_READY_EVENT: usize = 75;
const SLOT_BUFFER_COPY_TO_MEMORY: usize = 95;
const SLOT_EXECUTABLE_GET_COMPILED_MEMORY_STATS: usize = 99;
const SLOT_CLIENT_DMA_MAP: usize = 113;
const SLOT_CLIENT_DMA_UNMAP: usize = 114;
const PJRT_BUFFER_MEMORY_LAYOUT_TYPE_TILED: c_int = 0;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct PjrtApiVersion {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub major_version: c_int,
    pub minor_version: c_int,
}

#[repr(C)]
struct PjrtApiHeader {
    struct_size: usize,
    extension_start: *mut c_void,
    version: PjrtApiVersion,
}

#[repr(C)]
#[derive(Clone, Copy)]
union PjrtNamedValueUnion {
    string_value: *const c_char,
    int64_value: i64,
    int64_array_value: *const i64,
    float_value: f32,
    bool_value: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PjrtNamedValue {
    struct_size: usize,
    extension_start: *mut c_void,
    name: *const c_char,
    name_size: usize,
    value_type: c_int,
    value: PjrtNamedValueUnion,
    value_size: usize,
}

#[repr(C)]
struct PjrtErrorDestroyArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    error: *mut c_void,
}

#[repr(C)]
struct PjrtErrorMessageArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    error: *const c_void,
    message: *const c_char,
    message_size: usize,
}

#[repr(C)]
struct PjrtErrorGetCodeArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    error: *const c_void,
    code: c_int,
}

#[repr(C)]
struct PjrtPluginInitializeArgs {
    struct_size: usize,
    extension_start: *mut c_void,
}

#[repr(C)]
struct PjrtPluginAttributesArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    attributes: *const PjrtNamedValue,
    num_attributes: usize,
}

#[repr(C)]
struct PjrtEventArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    event: *mut c_void,
}

#[repr(C)]
struct PjrtClientCreateArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    create_options: *const PjrtNamedValue,
    num_options: usize,
    kv_get_callback: *const c_void,
    kv_get_user_arg: *mut c_void,
    kv_put_callback: *const c_void,
    kv_put_user_arg: *mut c_void,
    client: *mut c_void,
    kv_try_get_callback: *const c_void,
    kv_try_get_user_arg: *mut c_void,
}

#[repr(C)]
struct PjrtClientDestroyArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    client: *mut c_void,
}

#[repr(C)]
struct PjrtClientStringArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    client: *mut c_void,
    data: *const c_char,
    size: usize,
}

#[repr(C)]
struct PjrtClientDevicesArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    client: *mut c_void,
    devices: *const *mut c_void,
    num_devices: usize,
}

#[repr(C)]
struct PjrtProgram {
    struct_size: usize,
    extension_start: *mut c_void,
    code: *mut c_char,
    code_size: usize,
    format: *const c_char,
    format_size: usize,
}

#[repr(C)]
struct PjrtClientCompileArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    client: *mut c_void,
    program: *const PjrtProgram,
    compile_options: *const c_char,
    compile_options_size: usize,
    executable: *mut c_void,
}

#[repr(C)]
struct PjrtBufferFromHostBufferArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    client: *mut c_void,
    data: *const c_void,
    buffer_type: c_int,
    dims: *const i64,
    num_dims: usize,
    byte_strides: *const i64,
    num_byte_strides: usize,
    host_buffer_semantics: c_int,
    device: *mut c_void,
    memory: *mut c_void,
    device_layout: *mut c_void,
    done_with_host_buffer: *mut c_void,
    buffer: *mut c_void,
}

#[repr(C)]
struct PjrtClientDmaMapArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    client: *mut c_void,
    data: *mut c_void,
    size: usize,
}

#[repr(C)]
struct PjrtClientDmaUnmapArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    client: *mut c_void,
    data: *mut c_void,
}

#[repr(C)]
struct PjrtDeviceGetDescriptionArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    device: *mut c_void,
    device_description: *mut c_void,
}

#[repr(C)]
struct PjrtDeviceLocalHardwareIdArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    device: *mut c_void,
    local_hardware_id: c_int,
}

#[repr(C)]
struct PjrtDeviceAddressableMemoriesArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    device: *mut c_void,
    memories: *const *mut c_void,
    num_memories: usize,
}

#[repr(C)]
struct PjrtDeviceDefaultMemoryArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    device: *mut c_void,
    memory: *mut c_void,
}

#[repr(C)]
struct PjrtDeviceMemoryStatsArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    device: *mut c_void,
    bytes_in_use: i64,
    peak_bytes_in_use: i64,
    peak_bytes_in_use_is_set: bool,
    num_allocs: i64,
    num_allocs_is_set: bool,
    largest_alloc_size: i64,
    largest_alloc_size_is_set: bool,
    bytes_limit: i64,
    bytes_limit_is_set: bool,
    bytes_reserved: i64,
    bytes_reserved_is_set: bool,
    peak_bytes_reserved: i64,
    peak_bytes_reserved_is_set: bool,
    bytes_reservable_limit: i64,
    bytes_reservable_limit_is_set: bool,
    largest_free_block_bytes: i64,
    largest_free_block_bytes_is_set: bool,
    pool_bytes: i64,
    pool_bytes_is_set: bool,
    peak_pool_bytes: i64,
    peak_pool_bytes_is_set: bool,
    peak_allocated_bytes: i64,
    peak_allocated_bytes_is_set: bool,
}

#[repr(C)]
struct PjrtDeviceDescriptionIdArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    device_description: *mut c_void,
    id: c_int,
}

#[repr(C)]
struct PjrtDeviceDescriptionStringArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    device_description: *mut c_void,
    data: *const c_char,
    size: usize,
}

#[repr(C)]
struct PjrtDeviceDescriptionAttributesArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    device_description: *mut c_void,
    num_attributes: usize,
    attributes: *const PjrtNamedValue,
}

#[repr(C)]
struct PjrtMemoryKindArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    memory: *mut c_void,
    kind: *const c_char,
    kind_size: usize,
}

#[repr(C)]
struct PjrtExecutableDestroyArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    executable: *mut c_void,
}

#[repr(C)]
struct PjrtExecutableNumOutputsArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    executable: *mut c_void,
    num_outputs: usize,
}

#[repr(C)]
struct PjrtExecutableCompiledMemoryStatsArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    executable: *mut c_void,
    generated_code_size_in_bytes: i64,
    argument_size_in_bytes: i64,
    output_size_in_bytes: i64,
    alias_size_in_bytes: i64,
    temp_size_in_bytes: i64,
    host_generated_code_size_in_bytes: i64,
    host_argument_size_in_bytes: i64,
    host_output_size_in_bytes: i64,
    host_alias_size_in_bytes: i64,
    host_temp_size_in_bytes: i64,
    peak_memory_in_bytes: i64,
    total_size_in_bytes: i64,
    total_allocation_bytes: i64,
    indefinite_allocations: i64,
    peak_unpadded_heap_bytes: i64,
}

#[repr(C)]
struct PjrtLoadedExecutableGetExecutableArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    loaded_executable: *mut c_void,
    executable: *mut c_void,
}

#[repr(C)]
struct PjrtExecuteOptions {
    struct_size: usize,
    extension_start: *mut c_void,
    send_callbacks: *mut *mut c_void,
    recv_callbacks: *mut *mut c_void,
    num_send_ops: usize,
    num_recv_ops: usize,
    launch_id: c_int,
    non_donatable_input_indices: *const i64,
    num_non_donatable_input_indices: usize,
    context: *mut c_void,
    call_location: *const c_char,
    num_tasks: usize,
    task_ids: *mut c_int,
    incarnation_ids: *mut i64,
    multi_slice_config: *mut c_void,
    use_major_to_minor_data_layout_for_callbacks: bool,
    hlo_output_callbacks: *mut c_void,
    num_hlo_output_callbacks: usize,
    custom_options: *const PjrtNamedValue,
    num_custom_options: usize,
}

#[repr(C)]
struct PjrtLoadedExecutableExecuteArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    executable: *mut c_void,
    options: *mut PjrtExecuteOptions,
    argument_lists: *const *const *mut c_void,
    num_devices: usize,
    num_args: usize,
    output_lists: *const *mut *mut c_void,
    device_complete_events: *mut *mut c_void,
    execute_device: *mut c_void,
}

#[repr(C)]
struct PjrtBufferDestroyArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    buffer: *mut c_void,
}

#[repr(C)]
struct PjrtBufferOnDeviceSizeArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    buffer: *mut c_void,
    on_device_size_in_bytes: usize,
}

#[repr(C)]
struct PjrtBufferDimensionsArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    buffer: *mut c_void,
    dims: *const i64,
    num_dims: usize,
}

#[repr(C)]
struct PjrtBufferMemoryLayoutTiled {
    struct_size: usize,
    extension_start: *mut c_void,
    minor_to_major: *const i64,
    minor_to_major_size: usize,
    tile_dims: *const i64,
    tile_dim_sizes: *const usize,
    num_tiles: usize,
}

// `PJRT_Buffer_MemoryLayout` holds a C union of `tiled` (56 B, align 8) and `strides` (32 B, align 8),
// so embedding `tiled` directly has identical size, alignment, and `layout_type` offset (72).
#[repr(C)]
struct PjrtBufferMemoryLayout {
    struct_size: usize,
    extension_start: *mut c_void,
    tiled: PjrtBufferMemoryLayoutTiled,
    layout_type: c_int,
}

#[repr(C)]
struct PjrtBufferToHostArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    src: *mut c_void,
    host_layout: *mut PjrtBufferMemoryLayout,
    dst: *mut c_void,
    dst_size: usize,
    event: *mut c_void,
}

#[repr(C)]
struct PjrtBufferReadyEventArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    buffer: *mut c_void,
    event: *mut c_void,
}

#[repr(C)]
struct PjrtBufferCopyToMemoryArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    buffer: *mut c_void,
    dst_memory: *mut c_void,
    dst_buffer: *mut c_void,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum BufferType {
    S32 = 4,
    U8 = 6,
    F32 = 11,
    Bf16 = 13,
}

impl BufferType {
    pub const fn byte_width(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::Bf16 => 2,
            Self::S32 | Self::F32 => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum HostSemantics {
    ImmutableOnlyDuringCall = 0,
    ImmutableUntilTransferCompletes = 1,
}

pub(crate) struct HostBufferSpec<'a> {
    pub data: *const c_void,
    pub buffer_type: BufferType,
    pub dims: &'a [i64],
    pub semantics: HostSemantics,
    pub device: *mut c_void,
    pub memory: *mut c_void,
}

#[derive(Clone, Debug, serde::Serialize)]
#[serde(untagged)]
pub enum AttrValue {
    String(String),
    Int64(i64),
    Int64List(Vec<i64>),
    Float(f32),
    Bool(bool),
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct MemoryStats {
    pub bytes_in_use: i64,
    pub peak_bytes_in_use: Option<i64>,
    pub num_allocs: Option<i64>,
    pub largest_alloc_size: Option<i64>,
    pub bytes_limit: Option<i64>,
    pub bytes_reserved: Option<i64>,
    pub peak_bytes_reserved: Option<i64>,
    pub bytes_reservable_limit: Option<i64>,
    pub largest_free_block_bytes: Option<i64>,
    pub pool_bytes: Option<i64>,
    pub peak_pool_bytes: Option<i64>,
    pub peak_allocated_bytes: Option<i64>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct CompiledMemoryStats {
    pub generated_code_size_in_bytes: i64,
    pub argument_size_in_bytes: i64,
    pub output_size_in_bytes: i64,
    pub alias_size_in_bytes: i64,
    pub temp_size_in_bytes: i64,
    pub host_generated_code_size_in_bytes: i64,
    pub host_argument_size_in_bytes: i64,
    pub host_output_size_in_bytes: i64,
    pub host_alias_size_in_bytes: i64,
    pub host_temp_size_in_bytes: i64,
    pub peak_memory_in_bytes: i64,
    pub total_size_in_bytes: i64,
    pub total_allocation_bytes: i64,
    pub indefinite_allocations: i64,
    pub peak_unpadded_heap_bytes: i64,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct MemoryKindInfo {
    pub kind: String,
    #[serde(skip)]
    pub(crate) handle: *mut c_void,
}

pub(crate) struct PjrtApi {
    header: *const PjrtApiHeader,
    pub struct_size: usize,
    pub major_version: c_int,
    pub minor_version: c_int,
}

// `PJRT_Api` is a process-static immutable function table returned by `GetPjrtApi`.
unsafe impl Send for PjrtApi {}
unsafe impl Sync for PjrtApi {}

static LOADED_PLUGINS: Mutex<BTreeMap<String, Arc<PjrtApi>>> = Mutex::new(BTreeMap::new());

impl PjrtApi {
    pub(crate) fn load(path: &Path) -> Result<Arc<Self>> {
        let key = path.to_string_lossy().into_owned();
        let mut guard = LOADED_PLUGINS
            .lock()
            .map_err(|_| anyhow::anyhow!("PJRT plugin table mutex poisoned"))?;
        if let Some(existing) = guard.get(&key) {
            return Ok(Arc::clone(existing));
        }

        let c_path = CString::new(key.as_bytes())
            .with_context(|| format!("PJRT library path contains NUL byte: {key}"))?;
        // SAFETY: `c_path` is a valid NUL-terminated path; handle stays mapped for process life.
        let handle = unsafe { libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if handle.is_null() {
            bail!("dlopen({key}) failed: {}", dlerror_string());
        }

        let sym_name = c"GetPjrtApi";
        // SAFETY: `handle` is a valid dlopen handle and `sym_name` is NUL-terminated.
        let sym = unsafe { libc::dlsym(handle, sym_name.as_ptr()) };
        if sym.is_null() {
            bail!("dlsym(GetPjrtApi) in {key} failed: {}", dlerror_string());
        }

        // SAFETY: `GetPjrtApi` has C signature `const PJRT_Api* GetPjrtApi()`.
        let get_api = unsafe {
            std::mem::transmute::<*mut c_void, unsafe extern "C" fn() -> *const PjrtApiHeader>(sym)
        };
        let header = unsafe { get_api() };
        ensure!(!header.is_null(), "GetPjrtApi() in {key} returned null");

        // SAFETY: `header` points to the plugin's static `PJRT_Api` table.
        let (struct_size, version) = unsafe { ((*header).struct_size, (*header).version) };
        ensure!(
            struct_size >= size_of::<PjrtApiHeader>(),
            "PJRT_Api struct_size {struct_size} is smaller than header {}",
            size_of::<PjrtApiHeader>()
        );
        ensure!(
            version.major_version == PJRT_API_MAJOR_EXPECTED,
            "unsupported PJRT major version {} (expected {})",
            version.major_version,
            PJRT_API_MAJOR_EXPECTED
        );

        let api = Arc::new(Self {
            header,
            struct_size,
            major_version: version.major_version,
            minor_version: version.minor_version,
        });
        api.plugin_initialize()?;
        guard.insert(key, Arc::clone(&api));
        Ok(api)
    }

    fn slot_offset(slot: usize) -> usize {
        debug_assert!(slot >= 3);
        size_of::<PjrtApiHeader>() + (slot - 3) * size_of::<usize>()
    }

    fn raw_fn_ptr(&self, slot: usize, name: &str) -> Result<*const c_void> {
        let offset = Self::slot_offset(slot);
        let needed = offset + size_of::<usize>();
        ensure!(
            self.struct_size >= needed,
            "PJRT_Api struct_size {} too small for {name} (slot {slot}, needs {needed})",
            self.struct_size
        );
        // SAFETY: `offset + size_of::<usize>() <= self.struct_size` and 8-byte aligned.
        let fp = unsafe { *((self.header as *const u8).add(offset) as *const *const c_void) };
        ensure!(!fp.is_null(), "PJRT_Api function pointer {name} is null");
        Ok(fp)
    }

    unsafe fn call_status<A>(&self, slot: usize, name: &str, args: &mut A) -> Result<()> {
        let fp = self.raw_fn_ptr(slot, name)?;
        // SAFETY: caller matches `A` to the PJRT C API signature for `slot`.
        let f = unsafe {
            std::mem::transmute::<*const c_void, unsafe extern "C" fn(*mut A) -> *mut c_void>(fp)
        };
        let err = unsafe { f(args as *mut A) };
        self.check_error(err, name)
    }

    pub(crate) fn check_error(&self, err: *mut c_void, context: &str) -> Result<()> {
        if err.is_null() {
            return Ok(());
        }
        let message = self.error_message(err);
        let code = self.error_code(err);
        self.destroy_error(err);
        bail!("PJRT {context} failed (code={code}): {message}");
    }

    fn error_message(&self, err: *const c_void) -> String {
        let Ok(fp) = self.raw_fn_ptr(SLOT_ERROR_MESSAGE, "PJRT_Error_Message") else {
            return "<PJRT_Error_Message unavailable>".into();
        };
        let f = unsafe {
            std::mem::transmute::<*const c_void, unsafe extern "C" fn(*mut PjrtErrorMessageArgs)>(
                fp,
            )
        };
        let mut args = PjrtErrorMessageArgs {
            struct_size: offset_of!(PjrtErrorMessageArgs, message_size) + size_of::<usize>(),
            extension_start: ptr::null_mut(),
            error: err,
            message: ptr::null(),
            message_size: 0,
        };
        unsafe { f(&mut args) };
        bytes_to_string(args.message, args.message_size)
    }

    fn error_code(&self, err: *const c_void) -> c_int {
        let Ok(fp) = self.raw_fn_ptr(SLOT_ERROR_GET_CODE, "PJRT_Error_GetCode") else {
            return -1;
        };
        let f = unsafe {
            std::mem::transmute::<
                *const c_void,
                unsafe extern "C" fn(*mut PjrtErrorGetCodeArgs) -> *mut c_void,
            >(fp)
        };
        let mut args = PjrtErrorGetCodeArgs {
            struct_size: offset_of!(PjrtErrorGetCodeArgs, code) + size_of::<c_int>(),
            extension_start: ptr::null_mut(),
            error: err,
            code: -1,
        };
        let sub_err = unsafe { f(&mut args) };
        if !sub_err.is_null() {
            self.destroy_error(sub_err);
            return -1;
        }
        args.code
    }

    fn destroy_error(&self, err: *mut c_void) {
        if let Ok(fp) = self.raw_fn_ptr(SLOT_ERROR_DESTROY, "PJRT_Error_Destroy") {
            let f = unsafe {
                std::mem::transmute::<*const c_void, unsafe extern "C" fn(*mut PjrtErrorDestroyArgs)>(
                    fp,
                )
            };
            let mut args = PjrtErrorDestroyArgs {
                struct_size: offset_of!(PjrtErrorDestroyArgs, error) + size_of::<*mut c_void>(),
                extension_start: ptr::null_mut(),
                error: err,
            };
            unsafe { f(&mut args) };
        }
    }

    fn plugin_initialize(&self) -> Result<()> {
        let mut args = PjrtPluginInitializeArgs {
            struct_size: offset_of!(PjrtPluginInitializeArgs, extension_start)
                + size_of::<*mut c_void>(),
            extension_start: ptr::null_mut(),
        };
        unsafe { self.call_status(SLOT_PLUGIN_INITIALIZE, "PJRT_Plugin_Initialize", &mut args) }
    }

    pub(crate) fn plugin_attributes(&self) -> Result<BTreeMap<String, AttrValue>> {
        let mut args = PjrtPluginAttributesArgs {
            struct_size: offset_of!(PjrtPluginAttributesArgs, num_attributes) + size_of::<usize>(),
            extension_start: ptr::null_mut(),
            attributes: ptr::null(),
            num_attributes: 0,
        };
        unsafe { self.call_status(SLOT_PLUGIN_ATTRIBUTES, "PJRT_Plugin_Attributes", &mut args)? };
        Ok(decode_named_values(args.attributes, args.num_attributes))
    }

    pub(crate) fn await_and_destroy_event(&self, event: *mut c_void, context: &str) -> Result<()> {
        if event.is_null() {
            return Ok(());
        }
        let mut await_args = PjrtEventArgs {
            struct_size: offset_of!(PjrtEventArgs, event) + size_of::<*mut c_void>(),
            extension_start: ptr::null_mut(),
            event,
        };
        let await_res =
            unsafe { self.call_status(SLOT_EVENT_AWAIT, "PJRT_Event_Await", &mut await_args) };

        let mut err_args = PjrtEventArgs {
            struct_size: offset_of!(PjrtEventArgs, event) + size_of::<*mut c_void>(),
            extension_start: ptr::null_mut(),
            event,
        };
        let event_err =
            unsafe { self.call_status(SLOT_EVENT_ERROR, "PJRT_Event_Error", &mut err_args) };

        let mut destroy_args = PjrtEventArgs {
            struct_size: offset_of!(PjrtEventArgs, event) + size_of::<*mut c_void>(),
            extension_start: ptr::null_mut(),
            event,
        };
        let _ = unsafe {
            self.call_status(SLOT_EVENT_DESTROY, "PJRT_Event_Destroy", &mut destroy_args)
        };

        await_res.with_context(|| format!("await {context}"))?;
        event_err.with_context(|| format!("event status {context}"))
    }

    pub(crate) fn create_client(&self) -> Result<*mut c_void> {
        let mut args = PjrtClientCreateArgs {
            struct_size: offset_of!(PjrtClientCreateArgs, kv_try_get_user_arg)
                + size_of::<*mut c_void>(),
            extension_start: ptr::null_mut(),
            create_options: ptr::null(),
            num_options: 0,
            kv_get_callback: ptr::null(),
            kv_get_user_arg: ptr::null_mut(),
            kv_put_callback: ptr::null(),
            kv_put_user_arg: ptr::null_mut(),
            client: ptr::null_mut(),
            kv_try_get_callback: ptr::null(),
            kv_try_get_user_arg: ptr::null_mut(),
        };
        unsafe { self.call_status(SLOT_CLIENT_CREATE, "PJRT_Client_Create", &mut args)? };
        ensure!(
            !args.client.is_null(),
            "PJRT_Client_Create returned null client"
        );
        Ok(args.client)
    }

    pub(crate) fn destroy_client(&self, client: *mut c_void) {
        if client.is_null() {
            return;
        }
        let mut args = PjrtClientDestroyArgs {
            struct_size: offset_of!(PjrtClientDestroyArgs, client) + size_of::<*mut c_void>(),
            extension_start: ptr::null_mut(),
            client,
        };
        let _ = unsafe { self.call_status(SLOT_CLIENT_DESTROY, "PJRT_Client_Destroy", &mut args) };
    }

    pub(crate) fn client_platform_name(&self, client: *mut c_void) -> Result<String> {
        let mut args = PjrtClientStringArgs {
            struct_size: offset_of!(PjrtClientStringArgs, size) + size_of::<usize>(),
            extension_start: ptr::null_mut(),
            client,
            data: ptr::null(),
            size: 0,
        };
        unsafe {
            self.call_status(
                SLOT_CLIENT_PLATFORM_NAME,
                "PJRT_Client_PlatformName",
                &mut args,
            )?
        };
        Ok(bytes_to_string(args.data, args.size))
    }

    pub(crate) fn client_platform_version(&self, client: *mut c_void) -> Result<String> {
        let mut args = PjrtClientStringArgs {
            struct_size: offset_of!(PjrtClientStringArgs, size) + size_of::<usize>(),
            extension_start: ptr::null_mut(),
            client,
            data: ptr::null(),
            size: 0,
        };
        unsafe {
            self.call_status(
                SLOT_CLIENT_PLATFORM_VERSION,
                "PJRT_Client_PlatformVersion",
                &mut args,
            )?
        };
        Ok(bytes_to_string(args.data, args.size))
    }

    pub(crate) fn client_devices(
        &self,
        client: *mut c_void,
        addressable_only: bool,
    ) -> Result<Vec<*mut c_void>> {
        let (slot, name) = if addressable_only {
            (
                SLOT_CLIENT_ADDRESSABLE_DEVICES,
                "PJRT_Client_AddressableDevices",
            )
        } else {
            (SLOT_CLIENT_DEVICES, "PJRT_Client_Devices")
        };
        let mut args = PjrtClientDevicesArgs {
            struct_size: offset_of!(PjrtClientDevicesArgs, num_devices) + size_of::<usize>(),
            extension_start: ptr::null_mut(),
            client,
            devices: ptr::null(),
            num_devices: 0,
        };
        unsafe { self.call_status(slot, name, &mut args)? };
        if args.num_devices == 0 {
            return Ok(Vec::new());
        }
        ensure!(!args.devices.is_null(), "{name} returned null device slice");
        Ok(unsafe { slice::from_raw_parts(args.devices, args.num_devices) }.to_vec())
    }

    pub(crate) fn device_description(&self, device: *mut c_void) -> Result<*mut c_void> {
        let mut args = PjrtDeviceGetDescriptionArgs {
            struct_size: offset_of!(PjrtDeviceGetDescriptionArgs, device_description)
                + size_of::<*mut c_void>(),
            extension_start: ptr::null_mut(),
            device,
            device_description: ptr::null_mut(),
        };
        unsafe {
            self.call_status(
                SLOT_DEVICE_GET_DESCRIPTION,
                "PJRT_Device_GetDescription",
                &mut args,
            )?
        };
        ensure!(
            !args.device_description.is_null(),
            "PJRT_Device_GetDescription returned null"
        );
        Ok(args.device_description)
    }

    pub(crate) fn device_local_hardware_id(&self, device: *mut c_void) -> Result<i32> {
        let mut args = PjrtDeviceLocalHardwareIdArgs {
            struct_size: offset_of!(PjrtDeviceLocalHardwareIdArgs, local_hardware_id)
                + size_of::<c_int>(),
            extension_start: ptr::null_mut(),
            device,
            local_hardware_id: -1,
        };
        unsafe {
            self.call_status(
                SLOT_DEVICE_LOCAL_HARDWARE_ID,
                "PJRT_Device_LocalHardwareId",
                &mut args,
            )?
        };
        Ok(args.local_hardware_id)
    }

    pub(crate) fn device_default_memory(&self, device: *mut c_void) -> Result<*mut c_void> {
        let mut args = PjrtDeviceDefaultMemoryArgs {
            struct_size: offset_of!(PjrtDeviceDefaultMemoryArgs, memory) + size_of::<*mut c_void>(),
            extension_start: ptr::null_mut(),
            device,
            memory: ptr::null_mut(),
        };
        unsafe {
            self.call_status(
                SLOT_DEVICE_DEFAULT_MEMORY,
                "PJRT_Device_DefaultMemory",
                &mut args,
            )?
        };
        ensure!(
            !args.memory.is_null(),
            "PJRT_Device_DefaultMemory returned null"
        );
        Ok(args.memory)
    }

    pub(crate) fn device_addressable_memories(
        &self,
        device: *mut c_void,
    ) -> Result<Vec<MemoryKindInfo>> {
        let mut args = PjrtDeviceAddressableMemoriesArgs {
            struct_size: offset_of!(PjrtDeviceAddressableMemoriesArgs, num_memories)
                + size_of::<usize>(),
            extension_start: ptr::null_mut(),
            device,
            memories: ptr::null(),
            num_memories: 0,
        };
        unsafe {
            self.call_status(
                SLOT_DEVICE_ADDRESSABLE_MEMORIES,
                "PJRT_Device_AddressableMemories",
                &mut args,
            )?
        };
        if args.num_memories == 0 {
            return Ok(Vec::new());
        }
        ensure!(
            !args.memories.is_null(),
            "PJRT_Device_AddressableMemories returned null"
        );
        let handles = unsafe { slice::from_raw_parts(args.memories, args.num_memories) };
        let mut out = Vec::with_capacity(handles.len());
        for &handle in handles {
            let kind = self.memory_kind(handle)?;
            out.push(MemoryKindInfo { kind, handle });
        }
        Ok(out)
    }

    pub(crate) fn memory_kind(&self, memory: *mut c_void) -> Result<String> {
        let mut args = PjrtMemoryKindArgs {
            struct_size: offset_of!(PjrtMemoryKindArgs, kind_size) + size_of::<usize>(),
            extension_start: ptr::null_mut(),
            memory,
            kind: ptr::null(),
            kind_size: 0,
        };
        unsafe { self.call_status(SLOT_MEMORY_KIND, "PJRT_Memory_Kind", &mut args)? };
        Ok(bytes_to_string(args.kind, args.kind_size))
    }

    pub(crate) fn device_memory_stats(&self, device: *mut c_void) -> Result<MemoryStats> {
        let mut args = PjrtDeviceMemoryStatsArgs {
            struct_size: offset_of!(PjrtDeviceMemoryStatsArgs, peak_allocated_bytes_is_set)
                + size_of::<bool>(),
            extension_start: ptr::null_mut(),
            device,
            bytes_in_use: 0,
            peak_bytes_in_use: 0,
            peak_bytes_in_use_is_set: false,
            num_allocs: 0,
            num_allocs_is_set: false,
            largest_alloc_size: 0,
            largest_alloc_size_is_set: false,
            bytes_limit: 0,
            bytes_limit_is_set: false,
            bytes_reserved: 0,
            bytes_reserved_is_set: false,
            peak_bytes_reserved: 0,
            peak_bytes_reserved_is_set: false,
            bytes_reservable_limit: 0,
            bytes_reservable_limit_is_set: false,
            largest_free_block_bytes: 0,
            largest_free_block_bytes_is_set: false,
            pool_bytes: 0,
            pool_bytes_is_set: false,
            peak_pool_bytes: 0,
            peak_pool_bytes_is_set: false,
            peak_allocated_bytes: 0,
            peak_allocated_bytes_is_set: false,
        };
        unsafe {
            self.call_status(
                SLOT_DEVICE_MEMORY_STATS,
                "PJRT_Device_MemoryStats",
                &mut args,
            )?
        };
        Ok(MemoryStats {
            bytes_in_use: args.bytes_in_use,
            peak_bytes_in_use: args
                .peak_bytes_in_use_is_set
                .then_some(args.peak_bytes_in_use),
            num_allocs: args.num_allocs_is_set.then_some(args.num_allocs),
            largest_alloc_size: args
                .largest_alloc_size_is_set
                .then_some(args.largest_alloc_size),
            bytes_limit: args.bytes_limit_is_set.then_some(args.bytes_limit),
            bytes_reserved: args.bytes_reserved_is_set.then_some(args.bytes_reserved),
            peak_bytes_reserved: args
                .peak_bytes_reserved_is_set
                .then_some(args.peak_bytes_reserved),
            bytes_reservable_limit: args
                .bytes_reservable_limit_is_set
                .then_some(args.bytes_reservable_limit),
            largest_free_block_bytes: args
                .largest_free_block_bytes_is_set
                .then_some(args.largest_free_block_bytes),
            pool_bytes: args.pool_bytes_is_set.then_some(args.pool_bytes),
            peak_pool_bytes: args.peak_pool_bytes_is_set.then_some(args.peak_pool_bytes),
            peak_allocated_bytes: args
                .peak_allocated_bytes_is_set
                .then_some(args.peak_allocated_bytes),
        })
    }

    pub(crate) fn description_id(&self, desc: *mut c_void) -> Result<i32> {
        let mut args = PjrtDeviceDescriptionIdArgs {
            struct_size: offset_of!(PjrtDeviceDescriptionIdArgs, id) + size_of::<c_int>(),
            extension_start: ptr::null_mut(),
            device_description: desc,
            id: -1,
        };
        unsafe {
            self.call_status(
                SLOT_DEVICE_DESCRIPTION_ID,
                "PJRT_DeviceDescription_Id",
                &mut args,
            )?
        };
        Ok(args.id)
    }

    pub(crate) fn description_kind(&self, desc: *mut c_void) -> Result<String> {
        let mut args = PjrtDeviceDescriptionStringArgs {
            struct_size: offset_of!(PjrtDeviceDescriptionStringArgs, size) + size_of::<usize>(),
            extension_start: ptr::null_mut(),
            device_description: desc,
            data: ptr::null(),
            size: 0,
        };
        unsafe {
            self.call_status(
                SLOT_DEVICE_DESCRIPTION_KIND,
                "PJRT_DeviceDescription_Kind",
                &mut args,
            )?
        };
        Ok(bytes_to_string(args.data, args.size))
    }

    pub(crate) fn description_debug_string(&self, desc: *mut c_void) -> Result<String> {
        let mut args = PjrtDeviceDescriptionStringArgs {
            struct_size: offset_of!(PjrtDeviceDescriptionStringArgs, size) + size_of::<usize>(),
            extension_start: ptr::null_mut(),
            device_description: desc,
            data: ptr::null(),
            size: 0,
        };
        unsafe {
            self.call_status(
                SLOT_DEVICE_DESCRIPTION_DEBUG_STRING,
                "PJRT_DeviceDescription_DebugString",
                &mut args,
            )?
        };
        Ok(bytes_to_string(args.data, args.size))
    }

    pub(crate) fn description_attributes(
        &self,
        desc: *mut c_void,
    ) -> Result<BTreeMap<String, AttrValue>> {
        let mut args = PjrtDeviceDescriptionAttributesArgs {
            struct_size: offset_of!(PjrtDeviceDescriptionAttributesArgs, attributes)
                + size_of::<*const PjrtNamedValue>(),
            extension_start: ptr::null_mut(),
            device_description: desc,
            num_attributes: 0,
            attributes: ptr::null(),
        };
        unsafe {
            self.call_status(
                SLOT_DEVICE_DESCRIPTION_ATTRIBUTES,
                "PJRT_DeviceDescription_Attributes",
                &mut args,
            )?
        };
        Ok(decode_named_values(args.attributes, args.num_attributes))
    }

    pub(crate) fn compile_mlir(
        &self,
        client: *mut c_void,
        device_id: i32,
        mlir: &str,
    ) -> Result<*mut c_void> {
        let mut code = mlir.as_bytes().to_vec();
        let format = b"mlir";
        let compile_options = single_device_compile_options(device_id)?;
        let program = PjrtProgram {
            struct_size: offset_of!(PjrtProgram, format_size) + size_of::<usize>(),
            extension_start: ptr::null_mut(),
            code: code.as_mut_ptr() as *mut c_char,
            code_size: code.len(),
            format: format.as_ptr() as *const c_char,
            format_size: format.len(),
        };
        let mut args = client_compile_args(client, &program, &compile_options);
        unsafe { self.call_status(SLOT_CLIENT_COMPILE, "PJRT_Client_Compile", &mut args)? };
        ensure!(
            !args.executable.is_null(),
            "PJRT_Client_Compile returned null executable"
        );
        Ok(args.executable)
    }

    pub(crate) fn destroy_loaded_executable(&self, executable: *mut c_void) {
        if executable.is_null() {
            return;
        }
        let mut args = PjrtExecutableDestroyArgs {
            struct_size: offset_of!(PjrtExecutableDestroyArgs, executable)
                + size_of::<*mut c_void>(),
            extension_start: ptr::null_mut(),
            executable,
        };
        let _ = unsafe {
            self.call_status(
                SLOT_LOADED_EXECUTABLE_DESTROY,
                "PJRT_LoadedExecutable_Destroy",
                &mut args,
            )
        };
    }

    pub(crate) fn loaded_executable_metadata(
        &self,
        loaded: *mut c_void,
    ) -> Result<(usize, Result<CompiledMemoryStats>)> {
        let mut get_args = PjrtLoadedExecutableGetExecutableArgs {
            struct_size: offset_of!(PjrtLoadedExecutableGetExecutableArgs, executable)
                + size_of::<*mut c_void>(),
            extension_start: ptr::null_mut(),
            loaded_executable: loaded,
            executable: ptr::null_mut(),
        };
        unsafe {
            self.call_status(
                SLOT_LOADED_EXECUTABLE_GET_EXECUTABLE,
                "PJRT_LoadedExecutable_GetExecutable",
                &mut get_args,
            )?
        };
        ensure!(
            !get_args.executable.is_null(),
            "PJRT_LoadedExecutable_GetExecutable returned null"
        );
        let exec = get_args.executable;
        let num_outputs = self.executable_num_outputs(exec);
        let stats = self.executable_compiled_memory_stats(exec);
        self.destroy_executable(exec);
        Ok((num_outputs?, stats))
    }

    fn executable_num_outputs(&self, executable: *mut c_void) -> Result<usize> {
        let mut args = PjrtExecutableNumOutputsArgs {
            struct_size: offset_of!(PjrtExecutableNumOutputsArgs, num_outputs) + size_of::<usize>(),
            extension_start: ptr::null_mut(),
            executable,
            num_outputs: 0,
        };
        unsafe {
            self.call_status(
                SLOT_EXECUTABLE_NUM_OUTPUTS,
                "PJRT_Executable_NumOutputs",
                &mut args,
            )?
        };
        Ok(args.num_outputs)
    }

    fn executable_compiled_memory_stats(
        &self,
        executable: *mut c_void,
    ) -> Result<CompiledMemoryStats> {
        let mut args = PjrtExecutableCompiledMemoryStatsArgs {
            struct_size: offset_of!(
                PjrtExecutableCompiledMemoryStatsArgs,
                peak_unpadded_heap_bytes
            ) + size_of::<i64>(),
            extension_start: ptr::null_mut(),
            executable,
            generated_code_size_in_bytes: 0,
            argument_size_in_bytes: 0,
            output_size_in_bytes: 0,
            alias_size_in_bytes: 0,
            temp_size_in_bytes: 0,
            host_generated_code_size_in_bytes: 0,
            host_argument_size_in_bytes: 0,
            host_output_size_in_bytes: 0,
            host_alias_size_in_bytes: 0,
            host_temp_size_in_bytes: 0,
            peak_memory_in_bytes: 0,
            total_size_in_bytes: 0,
            total_allocation_bytes: 0,
            indefinite_allocations: 0,
            peak_unpadded_heap_bytes: 0,
        };
        unsafe {
            self.call_status(
                SLOT_EXECUTABLE_GET_COMPILED_MEMORY_STATS,
                "PJRT_Executable_GetCompiledMemoryStats",
                &mut args,
            )?
        };
        Ok(CompiledMemoryStats {
            generated_code_size_in_bytes: args.generated_code_size_in_bytes,
            argument_size_in_bytes: args.argument_size_in_bytes,
            output_size_in_bytes: args.output_size_in_bytes,
            alias_size_in_bytes: args.alias_size_in_bytes,
            temp_size_in_bytes: args.temp_size_in_bytes,
            host_generated_code_size_in_bytes: args.host_generated_code_size_in_bytes,
            host_argument_size_in_bytes: args.host_argument_size_in_bytes,
            host_output_size_in_bytes: args.host_output_size_in_bytes,
            host_alias_size_in_bytes: args.host_alias_size_in_bytes,
            host_temp_size_in_bytes: args.host_temp_size_in_bytes,
            peak_memory_in_bytes: args.peak_memory_in_bytes,
            total_size_in_bytes: args.total_size_in_bytes,
            total_allocation_bytes: args.total_allocation_bytes,
            indefinite_allocations: args.indefinite_allocations,
            peak_unpadded_heap_bytes: args.peak_unpadded_heap_bytes,
        })
    }

    fn destroy_executable(&self, executable: *mut c_void) {
        if executable.is_null() {
            return;
        }
        let mut args = PjrtExecutableDestroyArgs {
            struct_size: offset_of!(PjrtExecutableDestroyArgs, executable)
                + size_of::<*mut c_void>(),
            extension_start: ptr::null_mut(),
            executable,
        };
        let _ = unsafe {
            self.call_status(
                SLOT_EXECUTABLE_DESTROY,
                "PJRT_Executable_Destroy",
                &mut args,
            )
        };
    }

    pub(crate) fn execute_single_device_async(
        &self,
        executable: *mut c_void,
        execute_device: *mut c_void,
        inputs: &[*mut c_void],
        num_outputs: usize,
    ) -> Result<(Vec<*mut c_void>, *mut c_void)> {
        let mut options = PjrtExecuteOptions {
            struct_size: offset_of!(PjrtExecuteOptions, num_custom_options) + size_of::<usize>(),
            extension_start: ptr::null_mut(),
            send_callbacks: ptr::null_mut(),
            recv_callbacks: ptr::null_mut(),
            num_send_ops: 0,
            num_recv_ops: 0,
            launch_id: 0,
            non_donatable_input_indices: ptr::null(),
            num_non_donatable_input_indices: 0,
            context: ptr::null_mut(),
            call_location: ptr::null(),
            num_tasks: 0,
            task_ids: ptr::null_mut(),
            incarnation_ids: ptr::null_mut(),
            multi_slice_config: ptr::null_mut(),
            use_major_to_minor_data_layout_for_callbacks: false,
            hlo_output_callbacks: ptr::null_mut(),
            num_hlo_output_callbacks: 0,
            custom_options: ptr::null(),
            num_custom_options: 0,
        };
        let arg_list_ptr = inputs.as_ptr();
        let arg_lists = [arg_list_ptr];
        let mut out_storage = vec![ptr::null_mut(); num_outputs];
        let out_lists = [out_storage.as_mut_ptr()];
        let mut complete_event: *mut c_void = ptr::null_mut();

        let mut args = single_device_execute_args(
            (executable, execute_device),
            &mut options,
            (arg_lists.as_ptr(), inputs.len()),
            out_lists.as_ptr(),
            &mut complete_event,
        )?;
        unsafe {
            self.call_status(
                SLOT_LOADED_EXECUTABLE_EXECUTE,
                "PJRT_LoadedExecutable_Execute",
                &mut args,
            )?
        };
        for (idx, &buf) in out_storage.iter().enumerate() {
            if buf.is_null() {
                let _ = self.await_and_destroy_event(
                    complete_event,
                    "PJRT_LoadedExecutable_Execute cleanup",
                );
                for &out in &out_storage {
                    self.destroy_buffer(out);
                }
                bail!("PJRT_LoadedExecutable_Execute output[{idx}] is null");
            }
        }
        Ok((out_storage, complete_event))
    }

    pub(crate) unsafe fn buffer_from_host_buffer(
        &self,
        client: *mut c_void,
        spec: HostBufferSpec<'_>,
    ) -> Result<*mut c_void> {
        let mut args = PjrtBufferFromHostBufferArgs {
            struct_size: offset_of!(PjrtBufferFromHostBufferArgs, buffer)
                + size_of::<*mut c_void>(),
            extension_start: ptr::null_mut(),
            client,
            data: spec.data,
            buffer_type: spec.buffer_type as c_int,
            dims: spec.dims.as_ptr(),
            num_dims: spec.dims.len(),
            byte_strides: ptr::null(),
            num_byte_strides: 0,
            host_buffer_semantics: spec.semantics as c_int,
            device: spec.device,
            memory: spec.memory,
            device_layout: ptr::null_mut(),
            done_with_host_buffer: ptr::null_mut(),
            buffer: ptr::null_mut(),
        };
        unsafe {
            self.call_status(
                SLOT_CLIENT_BUFFER_FROM_HOST_BUFFER,
                "PJRT_Client_BufferFromHostBuffer",
                &mut args,
            )?
        };
        let done_res = self.await_and_destroy_event(
            args.done_with_host_buffer,
            "BufferFromHostBuffer.done_with_host_buffer",
        );
        if let Err(err) = done_res {
            self.destroy_buffer(args.buffer);
            return Err(err);
        }
        ensure!(
            !args.buffer.is_null(),
            "PJRT_Client_BufferFromHostBuffer returned null buffer"
        );
        if let Err(err) = self.await_buffer_ready(args.buffer) {
            self.destroy_buffer(args.buffer);
            return Err(err);
        }
        Ok(args.buffer)
    }

    pub(crate) fn await_buffer_ready(&self, buffer: *mut c_void) -> Result<()> {
        let mut args = PjrtBufferReadyEventArgs {
            struct_size: offset_of!(PjrtBufferReadyEventArgs, event) + size_of::<*mut c_void>(),
            extension_start: ptr::null_mut(),
            buffer,
            event: ptr::null_mut(),
        };
        unsafe { self.call_status(SLOT_BUFFER_READY_EVENT, "PJRT_Buffer_ReadyEvent", &mut args)? };
        self.await_and_destroy_event(args.event, "PJRT_Buffer_ReadyEvent")
    }

    pub(crate) fn buffer_on_device_size(&self, buffer: *mut c_void) -> Result<usize> {
        let mut args = PjrtBufferOnDeviceSizeArgs {
            struct_size: offset_of!(PjrtBufferOnDeviceSizeArgs, on_device_size_in_bytes)
                + size_of::<usize>(),
            extension_start: ptr::null_mut(),
            buffer,
            on_device_size_in_bytes: 0,
        };
        unsafe {
            self.call_status(
                SLOT_BUFFER_ON_DEVICE_SIZE_IN_BYTES,
                "PJRT_Buffer_OnDeviceSizeInBytes",
                &mut args,
            )?
        };
        Ok(args.on_device_size_in_bytes)
    }

    pub(crate) fn buffer_dimensions(&self, buffer: *mut c_void) -> Result<Vec<i64>> {
        let mut args = PjrtBufferDimensionsArgs {
            struct_size: offset_of!(PjrtBufferDimensionsArgs, num_dims) + size_of::<usize>(),
            extension_start: ptr::null_mut(),
            buffer,
            dims: ptr::null(),
            num_dims: 0,
        };
        unsafe { self.call_status(SLOT_BUFFER_DIMENSIONS, "PJRT_Buffer_Dimensions", &mut args)? };
        if args.num_dims == 0 {
            return Ok(Vec::new());
        }
        ensure!(
            !args.dims.is_null(),
            "PJRT_Buffer_Dimensions returned null dims with num_dims={}",
            args.num_dims
        );
        // SAFETY: `args.dims` points to `args.num_dims` `i64` elements valid for `buffer`'s lifetime.
        let slice = unsafe { std::slice::from_raw_parts(args.dims, args.num_dims) };
        Ok(slice.to_vec())
    }

    /// Copies a device buffer to `dst` in explicit C row-major (`[rank-1, ..., 0]`) host layout.
    pub(crate) fn buffer_to_host(&self, buffer: *mut c_void, dst: &mut [u8]) -> Result<()> {
        let dims = self.buffer_dimensions(buffer)?;
        let minor_to_major = row_major_minor_to_major(dims.len());
        let mut host_layout = PjrtBufferMemoryLayout {
            struct_size: offset_of!(PjrtBufferMemoryLayout, layout_type) + size_of::<c_int>(),
            extension_start: ptr::null_mut(),
            tiled: PjrtBufferMemoryLayoutTiled {
                struct_size: offset_of!(PjrtBufferMemoryLayoutTiled, num_tiles)
                    + size_of::<usize>(),
                extension_start: ptr::null_mut(),
                minor_to_major: minor_to_major.as_ptr(),
                minor_to_major_size: minor_to_major.len(),
                tile_dims: ptr::null(),
                tile_dim_sizes: ptr::null(),
                num_tiles: 0,
            },
            layout_type: PJRT_BUFFER_MEMORY_LAYOUT_TYPE_TILED,
        };
        let host_layout_ptr = if minor_to_major.is_empty() {
            ptr::null_mut()
        } else {
            &mut host_layout as *mut PjrtBufferMemoryLayout
        };
        let mut args = PjrtBufferToHostArgs {
            struct_size: offset_of!(PjrtBufferToHostArgs, event) + size_of::<*mut c_void>(),
            extension_start: ptr::null_mut(),
            src: buffer,
            host_layout: host_layout_ptr,
            dst: dst.as_mut_ptr() as *mut c_void,
            dst_size: dst.len(),
            event: ptr::null_mut(),
        };
        unsafe {
            self.call_status(
                SLOT_BUFFER_TO_HOST_BUFFER,
                "PJRT_Buffer_ToHostBuffer",
                &mut args,
            )?
        };
        let res = self.await_and_destroy_event(args.event, "PJRT_Buffer_ToHostBuffer");
        drop((host_layout, minor_to_major));
        res
    }

    pub(crate) fn buffer_copy_to_memory(
        &self,
        buffer: *mut c_void,
        dst_memory: *mut c_void,
    ) -> Result<*mut c_void> {
        let dst = self.buffer_copy_to_memory_async(buffer, dst_memory)?;
        if let Err(err) = self.await_buffer_ready(dst) {
            self.destroy_buffer(dst);
            return Err(err);
        }
        Ok(dst)
    }

    pub(crate) fn buffer_copy_to_memory_async(
        &self,
        buffer: *mut c_void,
        dst_memory: *mut c_void,
    ) -> Result<*mut c_void> {
        let mut args = PjrtBufferCopyToMemoryArgs {
            struct_size: offset_of!(PjrtBufferCopyToMemoryArgs, dst_buffer)
                + size_of::<*mut c_void>(),
            extension_start: ptr::null_mut(),
            buffer,
            dst_memory,
            dst_buffer: ptr::null_mut(),
        };
        unsafe {
            self.call_status(
                SLOT_BUFFER_COPY_TO_MEMORY,
                "PJRT_Buffer_CopyToMemory",
                &mut args,
            )?
        };
        ensure!(
            !args.dst_buffer.is_null(),
            "PJRT_Buffer_CopyToMemory returned null buffer"
        );
        Ok(args.dst_buffer)
    }

    pub(crate) fn destroy_buffer(&self, buffer: *mut c_void) {
        if buffer.is_null() {
            return;
        }
        let mut args = PjrtBufferDestroyArgs {
            struct_size: offset_of!(PjrtBufferDestroyArgs, buffer) + size_of::<*mut c_void>(),
            extension_start: ptr::null_mut(),
            buffer,
        };
        let _ = unsafe { self.call_status(SLOT_BUFFER_DESTROY, "PJRT_Buffer_Destroy", &mut args) };
    }

    pub(crate) unsafe fn dma_map(
        &self,
        client: *mut c_void,
        data: *mut c_void,
        size: usize,
    ) -> Result<()> {
        let mut args = PjrtClientDmaMapArgs {
            struct_size: offset_of!(PjrtClientDmaMapArgs, size) + size_of::<usize>(),
            extension_start: ptr::null_mut(),
            client,
            data,
            size,
        };
        unsafe { self.call_status(SLOT_CLIENT_DMA_MAP, "PJRT_Client_DmaMap", &mut args) }
    }

    pub(crate) unsafe fn dma_unmap(&self, client: *mut c_void, data: *mut c_void) -> Result<()> {
        let mut args = PjrtClientDmaUnmapArgs {
            struct_size: offset_of!(PjrtClientDmaUnmapArgs, data) + size_of::<*mut c_void>(),
            extension_start: ptr::null_mut(),
            client,
            data,
        };
        unsafe { self.call_status(SLOT_CLIENT_DMA_UNMAP, "PJRT_Client_DmaUnmap", &mut args) }
    }
}

fn dlerror_string() -> String {
    // SAFETY: `dlerror` returns null or a static/thread-local C string.
    let ptr = unsafe { libc::dlerror() };
    if ptr.is_null() {
        "unknown dlerror".into()
    } else {
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}

fn bytes_to_string(ptr: *const c_char, len: usize) -> String {
    if ptr.is_null() || len == 0 {
        return String::new();
    }
    let bytes = unsafe { slice::from_raw_parts(ptr as *const u8, len) };
    String::from_utf8_lossy(bytes).into_owned()
}

fn decode_named_values(ptr: *const PjrtNamedValue, count: usize) -> BTreeMap<String, AttrValue> {
    let mut out = BTreeMap::new();
    if ptr.is_null() || count == 0 {
        return out;
    }
    let values = unsafe { slice::from_raw_parts(ptr, count) };
    for item in values {
        let key = bytes_to_string(item.name, item.name_size);
        let val = match item.value_type {
            0 => AttrValue::String(bytes_to_string(
                unsafe { item.value.string_value },
                item.value_size,
            )),
            1 => AttrValue::Int64(unsafe { item.value.int64_value }),
            2 => {
                let p = unsafe { item.value.int64_array_value };
                let list = if p.is_null() || item.value_size == 0 {
                    Vec::new()
                } else {
                    unsafe { slice::from_raw_parts(p, item.value_size) }.to_vec()
                };
                AttrValue::Int64List(list)
            }
            3 => AttrValue::Float(unsafe { item.value.float_value }),
            4 => AttrValue::Bool(unsafe { item.value.bool_value }),
            other => AttrValue::String(format!("<unknown PJRT_NamedValue_Type={other}>")),
        };
        out.insert(key, val);
    }
    out
}

/// Minor-to-major physical dimension order for C row-major storage (`[rank-1, ..., 0]`).
fn row_major_minor_to_major(rank: usize) -> Vec<i64> {
    (0..rank as i64).rev().collect()
}

fn push_varint_u64(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte);
            break;
        }
        buf.push(byte | 0x80);
    }
}

/// Proto3 wire encoding of `xla::CompileOptionsProto` for a 1-replica, 1-partition
/// executable assigned to `device_id` (`xla/pjrt/proto/compile_options.proto`, `xla/xla_data.proto`).
pub(crate) fn single_device_compile_options(device_id: i32) -> Result<Vec<u8>> {
    ensure!(
        device_id >= 0,
        "TPU compile device_id must be non-negative, got {device_id}"
    );
    let mut id_varint = Vec::with_capacity(5);
    push_varint_u64(&mut id_varint, device_id as u64);

    let mut comp_dev = Vec::with_capacity(2 + id_varint.len());
    comp_dev.push(0x0a);
    push_varint_u64(&mut comp_dev, id_varint.len() as u64);
    comp_dev.extend_from_slice(&id_varint);

    let mut dev_assign = Vec::with_capacity(6 + comp_dev.len());
    dev_assign.extend_from_slice(&[0x08, 0x01, 0x10, 0x01, 0x1a]);
    push_varint_u64(&mut dev_assign, comp_dev.len() as u64);
    dev_assign.extend_from_slice(&comp_dev);

    // `device_ordinal = -1` leaves StreamExecutor ordinal unset in ExecutableBuildOptionsFromProto.
    let mut build_opts = Vec::with_capacity(17 + dev_assign.len());
    build_opts.push(0x08);
    push_varint_u64(&mut build_opts, (-1i64) as u64);
    build_opts.extend_from_slice(&[0x20, 0x01, 0x28, 0x01, 0x4a]);
    push_varint_u64(&mut build_opts, dev_assign.len() as u64);
    build_opts.extend_from_slice(&dev_assign);

    let mut out = Vec::with_capacity(2 + build_opts.len());
    out.push(0x1a);
    push_varint_u64(&mut out, build_opts.len() as u64);
    out.extend_from_slice(&build_opts);
    Ok(out)
}

fn client_compile_args(
    client: *mut c_void,
    program: *const PjrtProgram,
    compile_options: &[u8],
) -> PjrtClientCompileArgs {
    PjrtClientCompileArgs {
        struct_size: offset_of!(PjrtClientCompileArgs, executable) + size_of::<*mut c_void>(),
        extension_start: ptr::null_mut(),
        client,
        program,
        compile_options: compile_options.as_ptr() as *const c_char,
        compile_options_size: compile_options.len(),
        executable: ptr::null_mut(),
    }
}

fn single_device_execute_args(
    (executable, execute_device): (*mut c_void, *mut c_void),
    options: *mut PjrtExecuteOptions,
    (argument_lists, num_args): (*const *const *mut c_void, usize),
    output_lists: *const *mut *mut c_void,
    device_complete_events: *mut *mut c_void,
) -> Result<PjrtLoadedExecutableExecuteArgs> {
    ensure!(
        !execute_device.is_null(),
        "PJRT_LoadedExecutable_Execute execute_device is null"
    );
    Ok(PjrtLoadedExecutableExecuteArgs {
        struct_size: offset_of!(PjrtLoadedExecutableExecuteArgs, execute_device)
            + size_of::<*mut c_void>(),
        extension_start: ptr::null_mut(),
        executable,
        options,
        argument_lists,
        num_devices: 1,
        num_args,
        output_lists,
        device_complete_events,
        execute_device,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_usize_at<T>(val: &T, offset: usize) -> usize {
        let bytes =
            unsafe { slice::from_raw_parts(ptr::from_ref(val).cast::<u8>(), size_of::<T>()) };
        let mut raw = [0u8; size_of::<usize>()];
        raw.copy_from_slice(&bytes[offset..offset + size_of::<usize>()]);
        usize::from_ne_bytes(raw)
    }

    #[test]
    fn pjrt_abi_struct_sizes_and_slot_offsets_match_header() {
        assert_eq!(size_of::<PjrtApiVersion>(), 24);
        assert_eq!(size_of::<PjrtApiHeader>(), 40);
        assert_eq!(PjrtApi::slot_offset(SLOT_ERROR_DESTROY), 40);
        assert_eq!(PjrtApi::slot_offset(SLOT_CLIENT_CREATE), 120);
        assert_eq!(PjrtApi::slot_offset(SLOT_CLIENT_COMPILE), 200);
        assert_eq!(
            PjrtApi::slot_offset(SLOT_CLIENT_BUFFER_FROM_HOST_BUFFER),
            216
        );
        assert_eq!(PjrtApi::slot_offset(SLOT_DEVICE_MEMORY_STATS), 312);
        assert_eq!(PjrtApi::slot_offset(SLOT_LOADED_EXECUTABLE_EXECUTE), 480);
        assert_eq!(PjrtApi::slot_offset(SLOT_BUFFER_DIMENSIONS), 520);
        assert_eq!(PjrtApi::slot_offset(SLOT_BUFFER_COPY_TO_MEMORY), 776);
        assert_eq!(
            PjrtApi::slot_offset(SLOT_EXECUTABLE_GET_COMPILED_MEMORY_STATS),
            808
        );
        assert_eq!(PjrtApi::slot_offset(SLOT_CLIENT_DMA_MAP), 920);
        assert_eq!(PjrtApi::slot_offset(SLOT_CLIENT_DMA_UNMAP), 928);

        // Trailing-padding checks matching PJRT_STRUCT_SIZE(T, last_field).
        assert_eq!(
            offset_of!(PjrtNamedValue, value_size) + size_of::<usize>(),
            56
        );
        assert_eq!(
            offset_of!(PjrtDeviceMemoryStatsArgs, peak_allocated_bytes_is_set) + size_of::<bool>(),
            201
        );
        assert_eq!(
            offset_of!(PjrtDeviceLocalHardwareIdArgs, local_hardware_id) + size_of::<c_int>(),
            28
        );
        assert_eq!(
            offset_of!(PjrtErrorGetCodeArgs, code) + size_of::<c_int>(),
            28
        );
        assert_eq!(
            offset_of!(PjrtBufferDimensionsArgs, num_dims) + size_of::<usize>(),
            40
        );
        assert_eq!(
            offset_of!(PjrtBufferMemoryLayoutTiled, num_tiles) + size_of::<usize>(),
            56
        );
        assert_eq!(
            offset_of!(PjrtBufferMemoryLayout, layout_type) + size_of::<c_int>(),
            76
        );
        assert_eq!(size_of::<PjrtBufferMemoryLayout>(), 80);
        assert_eq!(row_major_minor_to_major(0), Vec::<i64>::new());
        assert_eq!(row_major_minor_to_major(1), vec![0]);
        assert_eq!(row_major_minor_to_major(2), vec![1, 0]);
        assert_eq!(row_major_minor_to_major(3), vec![2, 1, 0]);
        assert_eq!(row_major_minor_to_major(4), vec![3, 2, 1, 0]);
        assert_eq!(BufferType::U8 as i32, 6);
        assert_eq!(BufferType::F32 as i32, 11);
        assert_eq!(BufferType::Bf16 as i32, 13);
        assert_eq!(HostSemantics::ImmutableOnlyDuringCall as i32, 0);
        assert_eq!(HostSemantics::ImmutableUntilTransferCompletes as i32, 1);
    }

    #[test]
    fn single_device_compile_options_and_execute_args_carry_selected_device() -> Result<()> {
        let opts_dev0 = single_device_compile_options(0)?;
        let opts_dev3 = single_device_compile_options(3)?;
        let opts_dev130 = single_device_compile_options(130)?;

        // CompileOptionsProto(3) -> ExecutableBuildOptionsProto(1=-1, 4=1, 5=1, 9=DeviceAssignmentProto(1=1, 2=1, 3=[3])).
        let expected_dev3: [u8; 28] = [
            0x1a, 0x1a, 0x08, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01, 0x20,
            0x01, 0x28, 0x01, 0x4a, 0x09, 0x08, 0x01, 0x10, 0x01, 0x1a, 0x03, 0x0a, 0x01, 0x03,
        ];
        assert_eq!(opts_dev3, expected_dev3);
        assert_eq!(opts_dev0[..27], expected_dev3[..27]);
        assert_eq!(opts_dev0[27], 0x00);
        assert_ne!(opts_dev3, opts_dev0);
        assert_eq!(opts_dev130.len(), 29);
        assert_eq!(&opts_dev130[25..], &[0x0a, 0x02, 0x82, 0x01]);
        assert!(single_device_compile_options(-1).is_err());

        // Inspect raw C struct bytes for PJRT_Client_Compile_Args.
        let mut fake_client = 0x1111usize;
        let program = PjrtProgram {
            struct_size: offset_of!(PjrtProgram, format_size) + size_of::<usize>(),
            extension_start: ptr::null_mut(),
            code: ptr::null_mut(),
            code_size: 0,
            format: ptr::null(),
            format_size: 0,
        };
        let compile_args = client_compile_args(
            ptr::from_mut(&mut fake_client).cast::<c_void>(),
            &program,
            &opts_dev3,
        );
        assert_eq!(
            read_usize_at(
                &compile_args,
                offset_of!(PjrtClientCompileArgs, struct_size)
            ),
            offset_of!(PjrtClientCompileArgs, executable) + size_of::<*mut c_void>()
        );
        let raw_opts_ptr = read_usize_at(
            &compile_args,
            offset_of!(PjrtClientCompileArgs, compile_options),
        );
        let raw_opts_len = read_usize_at(
            &compile_args,
            offset_of!(PjrtClientCompileArgs, compile_options_size),
        );
        assert_eq!(raw_opts_ptr, opts_dev3.as_ptr() as usize);
        assert_eq!(raw_opts_len, opts_dev3.len());
        let raw_opts_slice =
            unsafe { slice::from_raw_parts(raw_opts_ptr as *const u8, raw_opts_len) };
        assert_eq!(raw_opts_slice, expected_dev3);

        // Inspect raw C struct bytes for PJRT_LoadedExecutable_Execute_Args.
        let mut fake_exec = 0x2222usize;
        let mut fake_dev0 = 0x3000usize;
        let mut fake_dev3 = 0x3003usize;
        let addressable = [
            ptr::from_mut(&mut fake_dev0).cast::<c_void>(),
            ptr::from_mut(&mut fake_dev3).cast::<c_void>(),
        ];
        let mut options = PjrtExecuteOptions {
            struct_size: offset_of!(PjrtExecuteOptions, num_custom_options) + size_of::<usize>(),
            extension_start: ptr::null_mut(),
            send_callbacks: ptr::null_mut(),
            recv_callbacks: ptr::null_mut(),
            num_send_ops: 0,
            num_recv_ops: 0,
            launch_id: 0,
            non_donatable_input_indices: ptr::null(),
            num_non_donatable_input_indices: 0,
            context: ptr::null_mut(),
            call_location: ptr::null(),
            num_tasks: 0,
            task_ids: ptr::null_mut(),
            incarnation_ids: ptr::null_mut(),
            multi_slice_config: ptr::null_mut(),
            use_major_to_minor_data_layout_for_callbacks: false,
            hlo_output_callbacks: ptr::null_mut(),
            num_hlo_output_callbacks: 0,
            custom_options: ptr::null(),
            num_custom_options: 0,
        };
        let mut event: *mut c_void = ptr::null_mut();
        let exec_args = single_device_execute_args(
            (
                ptr::from_mut(&mut fake_exec).cast::<c_void>(),
                addressable[1],
            ),
            &mut options,
            (ptr::null(), 0),
            ptr::null(),
            &mut event,
        )?;
        assert_eq!(
            read_usize_at(
                &exec_args,
                offset_of!(PjrtLoadedExecutableExecuteArgs, struct_size)
            ),
            offset_of!(PjrtLoadedExecutableExecuteArgs, execute_device) + size_of::<*mut c_void>()
        );
        assert_eq!(
            read_usize_at(
                &exec_args,
                offset_of!(PjrtLoadedExecutableExecuteArgs, num_devices)
            ),
            1
        );
        let raw_exec_device = read_usize_at(
            &exec_args,
            offset_of!(PjrtLoadedExecutableExecuteArgs, execute_device),
        );
        assert_eq!(raw_exec_device, addressable[1] as usize);
        assert_ne!(raw_exec_device, 0);
        assert_ne!(raw_exec_device, addressable[0] as usize);

        assert!(
            single_device_execute_args(
                (
                    ptr::from_mut(&mut fake_exec).cast::<c_void>(),
                    ptr::null_mut()
                ),
                &mut options,
                (ptr::null(), 0),
                ptr::null(),
                &mut event,
            )
            .is_err()
        );
        Ok(())
    }
}
