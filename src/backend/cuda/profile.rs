use std::ffi::{CStr, c_char};

unsafe extern "C" {
    fn ff_profile_push(name: *const c_char);
    fn ff_profile_pop();
}

pub struct Range(bool);

impl Range {
    pub fn new(enabled: bool, name: &CStr) -> Self {
        if enabled {
            unsafe {
                ff_profile_push(name.as_ptr());
            }
        }
        Self(enabled)
    }
}

impl Drop for Range {
    fn drop(&mut self) {
        if self.0 {
            unsafe {
                ff_profile_pop();
            }
        }
    }
}
