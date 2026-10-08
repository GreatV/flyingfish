use anyhow::{Result, ensure};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use std::{
    ffi::{CStr, c_char, c_void},
    sync::Arc,
};

unsafe extern "C" {
    fn ff_fa2_tile() -> i32;
    fn ff_fa2_error() -> *const c_char;
    fn ff_fa2_run(
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        out: *mut c_void,
        lse: *mut c_void,
        rows: i32,
        start: i32,
        q_heads: i32,
        kv_heads: i32,
        capacity: i32,
        causal: i32,
        stream: *mut c_void,
    ) -> i32;
}

pub struct Shape {
    pub rows: usize,
    pub start: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub capacity: usize,
    pub causal: bool,
}

pub struct Flash {
    stream: Arc<CudaStream>,
    lse: CudaSlice<f32>,
}

impl Flash {
    pub fn tile() -> Result<usize> {
        let tile = unsafe { ff_fa2_tile() };
        ensure!(tile > 0, "FA2 reported invalid query tile {tile}");
        Ok(tile as usize)
    }

    pub fn new(stream: Arc<CudaStream>, rows: usize, heads: usize) -> Result<Self> {
        let lse = stream.alloc_zeros(rows * heads)?;
        Ok(Self { stream, lse })
    }

    pub fn run(
        &mut self,
        qkv: &CudaSlice<bf16>,
        k: &CudaSlice<bf16>,
        v: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        shape: Shape,
    ) -> Result<()> {
        ensure!(
            shape.rows * shape.q_heads <= self.lse.len(),
            "FA2 LSE buffer is too small"
        );
        ensure!(
            shape.rows * (shape.q_heads + 2 * shape.kv_heads) * 128 <= qkv.len(),
            "FA2 Q buffer is too small"
        );
        ensure!(
            shape.capacity * shape.kv_heads * 128 <= k.len()
                && shape.capacity * shape.kv_heads * 128 <= v.len(),
            "FA2 KV buffers are too small"
        );
        ensure!(
            shape.rows * shape.q_heads * 128 <= out.len(),
            "FA2 output buffer is too small"
        );
        ensure!(
            shape.start + shape.rows <= shape.capacity,
            "FA2 range exceeds cache"
        );
        self.stream.context().bind_to_thread()?;
        let (q, _q) = qkv.device_ptr(&self.stream);
        let (k, _k) = k.device_ptr(&self.stream);
        let (v, _v) = v.device_ptr(&self.stream);
        let (out, _out) = out.device_ptr_mut(&self.stream);
        let (lse, _lse) = self.lse.device_ptr_mut(&self.stream);
        let status = unsafe {
            ff_fa2_run(
                q as _,
                k as _,
                v as _,
                out as _,
                lse as _,
                shape.rows as i32,
                shape.start as i32,
                shape.q_heads as i32,
                shape.kv_heads as i32,
                shape.capacity as i32,
                i32::from(shape.causal),
                self.stream.cu_stream() as _,
            )
        };
        if status != 0 {
            let error = unsafe { CStr::from_ptr(ff_fa2_error()) }.to_string_lossy();
            anyhow::bail!("FA2 launch failed: {error}");
        }
        Ok(())
    }
}
