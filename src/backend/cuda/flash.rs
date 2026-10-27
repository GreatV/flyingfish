use anyhow::Context;
use anyhow::{Result, ensure};
use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr};
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
    fn ff_fa2_run_window(
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
        window_left: i32,
        stream: *mut c_void,
    ) -> i32;
}

/// One attention problem: `rows` query rows at absolute positions
/// `[start, start + rows)` against the head-major KV caches of `capacity`
/// positions. Q rows are `q_heads * head_dim` wide with head stride `head_dim`;
/// the row pitch is stated at the call site.
pub use super::verification::Shape;

pub struct Flash {
    stream: Arc<CudaStream>,
    lse: CudaSlice<f32>,
    sink_rescale: CudaFunction,
    native: Option<CudaFunction>,
    geometry: [usize; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Attention {
    q: u64,
    k: u64,
    v: u64,
    out: u64,
    lse: u64,
    rows: u32,
    capacity: u32,
    q_stride: u32,
    q_heads: u32,
    group: u32,
    start: u32,
}
unsafe impl DeviceRepr for Attention {}

impl Flash {
    /// Query tile rows for the bound prefill family.
    pub fn tile(head_dim: usize) -> Result<usize> {
        if head_dim == 256 {
            return Ok(64);
        }
        let tile = unsafe { ff_fa2_tile(head_dim as i32) };
        ensure!(
            tile > 0,
            "FA2 binds head dims 128 and 256 (asked for {head_dim})"
        );
        Ok(tile as usize)
    }

    /// LSE is F32 [heads, rows], allocated once at setup.
    pub fn new(
        stream: Arc<CudaStream>,
        rows: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        ensure!(
            rows > 0 && heads > 0 && kv_heads > 0 && heads.is_multiple_of(kv_heads),
            "invalid prefill attention geometry"
        );
        ensure!(
            matches!(head_dim, 128 | 256),
            "unsupported prefill attention head dimension {head_dim}"
        );
        let elements = rows
            .checked_mul(heads)
            .context("prefill LSE extent overflow")?;
        ensure!(
            elements <= u32::MAX as usize,
            "prefill LSE indexing exceeds u32"
        );
        let native = if head_dim == 256 {
            use cudarc::driver::sys::{CUdevice_attribute as A, CUfunction_attribute as F};
            let ctx = stream.context();
            ensure!(
                heads <= ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)? as usize,
                "prefill heads exceed grid limit"
            );
            let function = super::cubin::module(ctx, "attention_native")?
                .load_function("attention_prefill_d256")?;
            let shared = function.get_attribute(F::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)?;
            let local = function.get_attribute(F::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?;
            let registers = function.get_attribute(F::CU_FUNC_ATTRIBUTE_NUM_REGS)?;
            ensure!(
                local == 0 && registers <= 255,
                "native prefill resource limit: registers={registers} local={local}"
            );
            ensure!(
                98304 + shared
                    <= ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)?,
                "native prefill shared memory does not fit"
            );
            function.set_attribute(F::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 98304)?;
            ensure!(
                function.occupancy_max_active_blocks_per_multiprocessor(128, 98304, None)? > 0,
                "native prefill has zero resident blocks"
            );
            Some(function)
        } else {
            None
        };
        let lse = stream.alloc_zeros(elements)?;
        let sink_rescale = super::cubin::module(stream.context(), "attention")?
            .load_function("attention_sink_rescale")?;
        Ok(Self {
            stream,
            lse,
            sink_rescale,
            native,
            geometry: [heads, kv_heads, head_dim],
        })
    }

    /// Runs prefill attention on one chunk. `q` is the normed, RoPE'd `[rows, q_heads,
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
            [shape.q_heads, shape.kv_heads, shape.head_dim] == self.geometry,
            "prefill geometry differs from setup binding"
        );
        self.run_with_window(q, q_row_stride, k, v, out, shape)
    }

    fn run_with_window(
        &mut self,
        q: &CudaSlice<bf16>,
        q_row_stride: usize,
        k: &CudaSlice<bf16>,
        v: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        shape: Shape,
    ) -> Result<()> {
        ensure!(
            [
                shape.rows,
                shape.start,
                shape.capacity,
                shape.q_heads,
                shape.kv_heads
            ]
            .iter()
            .all(|&x| x <= i32::MAX as usize),
            "FA2 dimensions exceed i32"
        );
        ensure!(
            shape
                .window_left
                .is_none_or(|window| window <= i32::MAX as usize),
            "FA2 window exceeds i32"
        );
        ensure!(
            shape.causal,
            "FA2 prefill is causal-only; the non-causal path selects a zero-width local window"
        );
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
            shape
                .start
                .checked_add(shape.rows)
                .is_some_and(|end| end <= shape.capacity),
            "FA2 range exceeds cache"
        );
        ensure!(
            q_row_stride >= shape.q_heads * shape.head_dim
                && q_row_stride <= u32::MAX as usize
                && q_row_stride.is_multiple_of(8),
            "FA2 Q row pitch {q_row_stride} is below one head row"
        );
        ensure!(
            shape.rows * shape.q_heads <= self.lse.len(),
            "FA2 LSE buffer is too small"
        );
        let q_span = shape
            .rows
            .saturating_sub(1)
            .checked_mul(q_row_stride)
            .and_then(|span| span.checked_add(shape.q_heads * shape.head_dim))
            .context("prefill Q extent overflow")?;
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
        if let Some(native) = &self.native {
            ensure!(
                shape.window_left.is_none(),
                "D256 prefill does not bind a sliding window"
            );
            let args = Attention {
                q,
                k,
                v,
                out,
                lse,
                rows: shape.rows as u32,
                capacity: shape.capacity as u32,
                q_stride: q_row_stride as u32,
                q_heads: shape.q_heads as u32,
                group: (shape.q_heads / shape.kv_heads) as u32,
                start: shape.start as u32,
            };
            unsafe {
                self.stream
                    .launch_builder(native)
                    .arg(&args)
                    .launch(LaunchConfig {
                        grid_dim: (shape.rows.div_ceil(64) as u32, shape.q_heads as u32, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 98304,
                    })?;
            }
            return Ok(());
        }
        let status = if let Some(window_left) = shape.window_left {
            unsafe {
                ff_fa2_run_window(
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
                    window_left as i32,
                    self.stream.cu_stream() as _,
                )
            }
        } else {
            unsafe {
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
            }
        };
        if status != 0 {
            let error = unsafe { CStr::from_ptr(ff_fa2_error()) }.to_string_lossy();
            anyhow::bail!("FA2 launch failed: {error}");
        }
        Ok(())
    }

    pub fn apply_sinks(
        &mut self,
        sinks: &CudaSlice<bf16>,
        output: &mut CudaSlice<bf16>,
        rows: usize,
        heads: usize,
    ) -> Result<()> {
        ensure!(
            rows > 0 && heads > 0 && rows <= i32::MAX as usize && heads <= i32::MAX as usize,
            "attention sink rescale dimensions exceed i32"
        );
        ensure!(
            sinks.len() == heads && self.lse.len() >= rows * heads,
            "attention sink extents differ: sinks={}, lse={}, rows={rows}, heads={heads}",
            sinks.len(),
            self.lse.len()
        );
        ensure!(
            output.len() >= rows * heads * 128,
            "attention sink output extent is below rows*heads*128"
        );
        self.stream.context().bind_to_thread()?;
        let (lse, _lse) = self.lse.device_ptr(&self.stream);
        let (sinks, _sinks) = sinks.device_ptr(&self.stream);
        let (output, _output) = output.device_ptr_mut(&self.stream);
        let elements = rows
            .checked_mul(heads)
            .and_then(|count| count.checked_mul(128))
            .context("attention sink output extent overflow")?;
        let blocks = elements.div_ceil(256).min(65_535);
        unsafe {
            self.stream
                .launch_builder(&self.sink_rescale)
                .arg(&lse)
                .arg(&sinks)
                .arg(&output)
                .arg(&(rows as i32))
                .arg(&(heads as i32))
                .launch(LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Attention;

    #[test]
    fn native_prefill_parameter_layout() {
        assert_eq!(std::mem::size_of::<Attention>(), 64);
        assert_eq!(std::mem::align_of::<Attention>(), 8);
        assert_eq!(std::mem::offset_of!(Attention, rows), 40);
        assert_eq!(std::mem::offset_of!(Attention, q_stride), 48);
        assert_eq!(std::mem::offset_of!(Attention, start), 60);
    }
}
