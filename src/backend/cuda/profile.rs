use std::ffi::CStr;
#[cfg(feature = "cuda")]
use std::ffi::c_char;

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn ff_profile_push(name: *const c_char);
    fn ff_profile_pop();
}

pub struct Range {
    #[cfg(feature = "cuda")]
    enabled: bool,
}

impl Range {
    pub fn new(enabled: bool, name: &CStr) -> Self {
        #[cfg(feature = "rocm")]
        {
            let _ = name;
            assert!(!enabled, "HIP NVTX profiling is unsupported");
        }
        #[cfg(feature = "cuda")]
        if enabled {
            unsafe {
                ff_profile_push(name.as_ptr());
            }
        }
        Self {
            #[cfg(feature = "cuda")]
            enabled,
        }
    }
}

impl Drop for Range {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        if self.enabled {
            unsafe {
                ff_profile_pop();
            }
        }
    }
}
