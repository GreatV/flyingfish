use anyhow::{Result, ensure};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use std::{
    ffi::{CStr, c_char, c_void},
    sync::Arc,
};

unsafe extern "C" {
    fn ff_fa2_tile(head_dim: i32) -> i32;
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
        head_dim: i32,
        q_row_stride: i64,
        causal: i32,
        stream: *mut c_void,
    ) -> i32;
}

/// One attention problem: `rows` query rows at absolute positions
/// `[start, start + rows)` against the head-major KV caches of `capacity`
/// positions. Q rows are `q_heads * head_dim` wide with head stride `head_dim`;
/// the row pitch is stated at the call site.
pub struct Shape {
    pub rows: usize,
    pub start: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub capacity: usize,
    pub head_dim: usize,
    pub causal: bool,
}

pub struct Flash {
    stream: Arc<CudaStream>,
    lse: CudaSlice<f32>,
}

impl Flash {
    /// FA2's `kBlockM` for a head dim on the current device: 64 for d128 and
    /// 128 or 64 for d256, per FA2's own hdim256 shared-memory query.
    pub fn tile(head_dim: usize) -> Result<usize> {
        let tile = unsafe { ff_fa2_tile(head_dim as i32) };
        ensure!(
            tile > 0,
            "FA2 binds head dims 128 and 256 (asked for {head_dim})"
        );
        Ok(tile as usize)
    }

    /// Holds FA2's LSE, `rows * heads` f32 (FA2 writes `(b, h, seqlen_q)`),
    /// allocated once at setup so `run` never allocates.
    pub fn new(stream: Arc<CudaStream>, rows: usize, heads: usize) -> Result<Self> {
        let lse = stream.alloc_zeros(rows * heads)?;
        Ok(Self { stream, lse })
    }

    /// Runs FA2 on one chunk. `q` is the normed, RoPE'd `[rows, q_heads,
    /// head_dim]` view (head stride `head_dim`); `q_row_stride` is its element
    /// pitch between rows — 6144 for the Q38 split Q, not a fused qkv. `k` and
    /// `v` are the head-major caches `[kv_heads, capacity, head_dim]` that
    /// `tcmqa_d256_rope_k_scatter` / `_v_scatter` write, holding every position
    /// `[0, start + rows)`.
    pub fn run(
        &mut self,
        q: &CudaSlice<bf16>,
        q_row_stride: usize,
        k: &CudaSlice<bf16>,
        v: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        shape: Shape,
    ) -> Result<()> {
        ensure!(
            shape.head_dim == 128 || shape.head_dim == 256,
            "FA2 binds head dims 128 and 256 (got {})",
            shape.head_dim
        );
        ensure!(shape.rows > 0, "FA2 needs at least one query row");
        ensure!(
            shape.kv_heads > 0 && shape.q_heads.is_multiple_of(shape.kv_heads),
            "FA2 query heads must be a multiple of the KV heads"
        );
        ensure!(
            shape.start + shape.rows <= shape.capacity,
            "FA2 range exceeds cache"
        );
        ensure!(
            q_row_stride >= shape.q_heads * shape.head_dim,
            "FA2 Q row pitch {q_row_stride} is below one head row"
        );
        ensure!(
            shape.rows * shape.q_heads <= self.lse.len(),
            "FA2 LSE buffer is too small"
        );
        let q_span = shape.rows.saturating_sub(1) * q_row_stride + shape.q_heads * shape.head_dim;
        ensure!(q_span <= q.len(), "FA2 Q buffer is too small");
        let cache_span = shape
            .capacity
            .checked_mul(shape.kv_heads)
            .and_then(|x| x.checked_mul(shape.head_dim))
            .ok_or_else(|| anyhow::anyhow!("FA2 cache span overflow"))?;
        ensure!(
            cache_span <= k.len() && cache_span <= v.len(),
            "FA2 KV buffers are too small"
        );
        ensure!(
            shape.rows * shape.q_heads * shape.head_dim <= out.len(),
            "FA2 output buffer is too small"
        );
        self.stream.context().bind_to_thread()?;
        let (q, _q) = q.device_ptr(&self.stream);
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
                shape.head_dim as i32,
                q_row_stride as i64,
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
