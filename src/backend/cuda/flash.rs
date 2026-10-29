use crate::backend::gpu::{CudaFunction, LaunchConfig, PushKernelArg};
use crate::backend::gpu::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr};
use anyhow::Context;
use anyhow::{Result, ensure};
use half::bf16;
#[cfg(feature = "cuda")]
use std::ffi::{CStr, c_char, c_void};
use std::sync::Arc;

#[cfg(feature = "cuda")]
unsafe extern "C" {
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

pub struct Query<'a> {
    pub q: &'a CudaSlice<bf16>,
    pub q_row_stride: usize,
    pub k: &'a CudaSlice<bf16>,
    pub v: &'a CudaSlice<bf16>,
    pub out: &'a mut CudaSlice<bf16>,
    pub shape: Shape,
    pub keys: usize,
    pub prefix: Option<usize>,
}

#[derive(Clone, Copy)]
pub struct Score {
    pub scale: f32,
    pub window_left: Option<usize>,
}

impl Score {
    pub fn from_tokens(scale: f32, window_tokens: Option<usize>) -> Result<Self> {
        let window_left = window_tokens
            .map(|tokens| {
                tokens
                    .checked_sub(1)
                    .context("attention window must contain a token")
            })
            .transpose()?;
        Ok(Self { scale, window_left })
    }
}

pub struct Flash {
    stream: Arc<CudaStream>,
    lse: CudaSlice<f32>,
    sink_rescale: CudaFunction,
    native: Vec<CudaFunction>,
    geometry: [usize; 3],
    score: Option<Score>,
    tile: usize,
    shared: u32,
    split: Option<Split>,
}

struct Split {
    partial: CudaFunction,
    merge: CudaFunction,
    workspace: CudaSlice<f32>,
    capacity: usize,
    parts: u32,
    chunk: u32,
}

#[cfg(any(feature = "cuda", test))]
fn split_extent(capacity: usize, heads: usize, resident: usize) -> Result<(usize, usize)> {
    ensure!(
        capacity > 0 && capacity <= i32::MAX as usize,
        "invalid split attention capacity"
    );
    ensure!(
        heads > 0 && resident > 0,
        "invalid split attention occupancy"
    );
    let wave = resident.div_ceil(heads).min(capacity.div_ceil(32));
    let chunk = capacity
        .div_ceil(wave)
        .div_ceil(32)
        .checked_mul(32)
        .context("split attention chunk overflow")?;
    Ok((capacity.div_ceil(chunk), chunk))
}

#[cfg(feature = "cuda")]
impl Split {
    fn new(stream: &Arc<CudaStream>, rows: usize, heads: usize, capacity: usize) -> Result<Self> {
        use cudarc::driver::sys::{CUdevice_attribute as A, CUfunction_attribute as F};
        let ctx = stream.context();
        let module = super::cubin::module(ctx, "attention_native")?;
        let partial = module.load_function("attention_partial_d512")?;
        let merge = module.load_function("attention_merge_d512")?;
        for (function, shared) in [(&partial, 90112), (&merge, 0)] {
            let local = function.get_attribute(F::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?;
            let regs = function.get_attribute(F::CU_FUNC_ATTRIBUTE_NUM_REGS)?;
            ensure!(
                local == 0 && regs <= 255,
                "D512 split resource limit: regs={regs} local={local}"
            );
            ensure!(
                shared + function.get_attribute(F::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)?
                    <= ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)?,
                "D512 split shared memory does not fit"
            );
            function.set_attribute(F::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared)?;
        }
        let resident =
            partial.occupancy_max_active_blocks_per_multiprocessor(128, 90112, None)? as usize;
        let sms = usize::try_from(ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?)?;
        let (parts, chunk) = split_extent(
            capacity,
            heads,
            sms.checked_mul(resident)
                .context("D512 resident grid overflow")?,
        )?;
        ensure!(
            parts <= ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)? as usize,
            "D512 split grid exceeds device limit"
        );
        let elements = rows
            .min(16)
            .checked_mul(heads)
            .and_then(|v| v.checked_mul(parts))
            .and_then(|v| v.checked_mul(514))
            .context("D512 partial workspace overflow")?;
        eprintln!(
            "{}",
            serde_json::json!({"native_attention_split":{
                "head_dim":512,"capacity":capacity,"heads":heads,"max_rows":rows.min(16),
                "sms":sms,"resident_blocks_per_sm":resident,"parts":parts,"chunk":chunk,
                "workspace_bytes":elements.checked_mul(4).context("D512 workspace bytes overflow")?
            }})
        );
        Ok(Self {
            partial,
            merge,
            workspace: stream.alloc_zeros(elements)?,
            capacity,
            parts: u32::try_from(parts)?,
            chunk: u32::try_from(chunk)?,
        })
    }
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
        ensure!(
            matches!(head_dim, 128 | 256 | 512),
            "prefill binds head dims 128, 256 and 512 (asked for {head_dim})"
        );
        #[cfg(feature = "cuda")]
        {
            Ok(if head_dim == 512 { 16 } else { 64 })
        }
        #[cfg(feature = "rocm")]
        {
            ensure!(head_dim != 512, "HIP D512 attention is not bound");
            Ok(if head_dim == 256 { 16 } else { 32 })
        }
    }

    /// LSE is F32 [heads, rows], allocated once at setup.
    pub fn new(
        stream: Arc<CudaStream>,
        rows: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        ensure!(head_dim != 512, "D512 requires an explicit score binding");
        let scale = if head_dim == 256 {
            0.0625
        } else {
            std::f32::consts::FRAC_1_SQRT_2 / 8.0
        };
        Self::new_with_score(
            stream,
            rows,
            heads,
            kv_heads,
            head_dim,
            Score::from_tokens(scale, None)?,
            rows,
        )
    }

    pub fn new_with_score(
        stream: Arc<CudaStream>,
        rows: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        score: Score,
        capacity: usize,
    ) -> Result<Self> {
        ensure!(
            matches!(head_dim, 128 | 256 | 512),
            "score binding requires D128, D256 or D512"
        );
        ensure!(
            score.scale.is_finite() && score.scale > 0.0,
            "invalid attention score multiplier"
        );
        ensure!(
            score.window_left.is_none_or(|w| w <= i32::MAX as usize),
            "attention window exceeds indexing range"
        );
        ensure!(
            head_dim != 512 || score.window_left.is_none(),
            "D512 binds full causal attention"
        );
        let legacy_scale = if head_dim == 256 {
            0.0625
        } else {
            std::f32::consts::FRAC_1_SQRT_2 / 8.0
        };
        let explicit =
            head_dim == 512 || score.scale != legacy_scale || score.window_left.is_some();
        ensure!(
            head_dim != 128 || !explicit,
            "D128 explicit score is not bound"
        );
        Self::bind(
            stream,
            rows,
            heads,
            kv_heads,
            head_dim,
            explicit.then_some(score),
            capacity,
        )
    }

    fn bind(
        stream: Arc<CudaStream>,
        rows: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        score: Option<Score>,
        capacity: usize,
    ) -> Result<Self> {
        let tile = Self::tile(head_dim)?;
        #[cfg(feature = "cuda")]
        let dynamic_shared = if head_dim == 512 {
            90112
        } else {
            (384 * head_dim) as u32
        };
        ensure!(
            rows > 0 && heads > 0 && kv_heads > 0 && heads.is_multiple_of(kv_heads),
            "invalid prefill attention geometry"
        );
        ensure!(
            matches!(head_dim, 128 | 256 | 512),
            "unsupported prefill attention head dimension {head_dim}"
        );
        #[cfg(feature = "rocm")]
        let dynamic_shared = 0;
        #[cfg(feature = "rocm")]
        ensure!(score.is_none(), "HIP explicit attention score is not bound");
        #[cfg(feature = "rocm")]
        ensure!(
            capacity >= rows,
            "HIP attention capacity is below query workspace"
        );
        let elements = rows
            .checked_mul(heads)
            .context("prefill LSE extent overflow")?;
        ensure!(
            elements <= u32::MAX as usize,
            "prefill LSE indexing exceeds u32"
        );
        #[cfg(feature = "cuda")]
        let native = if matches!(head_dim, 256 | 512)
            || matches!(
                (heads, kv_heads, head_dim),
                (16, 2, 128) | (16, 8, 128) | (12, 12, 128) | (32, 4, 128)
            ) {
            use cudarc::driver::sys::{CUdevice_attribute as A, CUfunction_attribute as F};
            let ctx = stream.context();
            ensure!(
                heads <= ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)? as usize,
                "prefill heads exceed grid limit"
            );
            let names: &[&str] = if head_dim == 512 {
                &["attention_prefill_d512"]
            } else if score.is_some() {
                &["attention_prefill_d256_window"]
            } else if head_dim == 256 {
                &["attention_prefill_d256"]
            } else if (heads, kv_heads) == (32, 4) {
                &["attention_prefill_d128", "attention_prefill_d128_window"]
            } else {
                &[
                    "attention_prefill_d128",
                    "attention_prefill_d128_bidirectional",
                    "attention_prefill_d128_prefix",
                ]
            };
            let module = super::cubin::module(ctx, "attention_native")?;
            let mut functions = Vec::with_capacity(names.len());
            for name in names {
                let function = module.load_function(name)?;
                let shared = function.get_attribute(F::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)?;
                let local = function.get_attribute(F::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?;
                let registers = function.get_attribute(F::CU_FUNC_ATTRIBUTE_NUM_REGS)?;
                ensure!(
                    local == 0 && registers <= 255,
                    "native prefill {name} resource limit: registers={registers} local={local}"
                );
                ensure!(
                    dynamic_shared as i32 + shared
                        <= ctx
                            .attribute(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)?,
                    "native prefill {name} shared memory does not fit"
                );
                function.set_attribute(
                    F::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    dynamic_shared as i32,
                )?;
                ensure!(
                    function.occupancy_max_active_blocks_per_multiprocessor(
                        128,
                        dynamic_shared as usize,
                        None
                    )? > 0,
                    "native prefill {name} has zero resident blocks"
                );
                functions.push(function);
            }
            functions
        } else {
            Vec::new()
        };
        #[cfg(feature = "rocm")]
        let native = {
            let ctx = stream.context();
            ctx.require_wmma().context("native prefill attention")?;
            ensure!(
                heads <= ctx.properties.max_grid_y as usize,
                "prefill heads exceed grid limit"
            );
            ensure!(
                rows.div_ceil(Self::tile(head_dim)?) <= ctx.properties.max_grid_x as usize,
                "prefill query tiles exceed grid limit"
            );
            let names: &[&str] = if head_dim == 256 {
                &["attention_prefill_d256"]
            } else if (heads, kv_heads) == (32, 4) {
                &["attention_prefill_d128", "attention_prefill_d128_window"]
            } else {
                ensure!(
                    matches!((heads, kv_heads), (16, 2) | (16, 8) | (12, 12)),
                    "HIP native D128 prefill geometry is not bound"
                );
                &[
                    "attention_prefill_d128",
                    "attention_prefill_d128_bidirectional",
                    "attention_prefill_d128_prefix",
                ]
            };
            let module = super::cubin::module(ctx, "attention")?;
            names
                .iter()
                .map(|name| {
                    let function = module.load_function(name)?;
                    ensure!(
                        function.max_threads_per_block()? >= 128
                            && function.shared_size_bytes()? as usize
                                <= ctx.properties.optin_shared_bytes as usize
                            && function
                                .occupancy_max_active_blocks_per_multiprocessor(128, 0, None)?
                                > 0,
                        "HIP native prefill {name} resources exceed device limits"
                    );
                    Ok(function)
                })
                .collect::<Result<Vec<_>>>()?
        };
        #[cfg(feature = "rocm")]
        let split = None;
        #[cfg(feature = "cuda")]
        let split = if head_dim == 512 {
            Some(Split::new(&stream, rows, heads, capacity)?)
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
            score,
            tile,
            shared: dynamic_shared,
            split,
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
        let keys = shape
            .start
            .checked_add(shape.rows)
            .context("prefill key extent overflow")?;
        self.run_masked(Query {
            q,
            q_row_stride,
            k,
            v,
            out,
            shape,
            keys,
            prefix: None,
        })
    }

    /// `keys` is the populated sequence extent; `prefix` enables bidirectional prefix rows.
    pub fn run_masked(&mut self, query: Query<'_>) -> Result<()> {
        let Query {
            q,
            q_row_stride,
            k,
            v,
            out,
            shape,
            keys,
            prefix,
        } = query;
        ensure!(
            [shape.q_heads, shape.kv_heads, shape.head_dim] == self.geometry,
            "prefill geometry differs from setup binding"
        );
        let end = shape
            .start
            .checked_add(shape.rows)
            .context("prefill query extent overflow")?;
        ensure!(
            end <= keys && keys <= shape.capacity && keys <= i32::MAX as usize,
            "prefill populated key extent does not cover queries or exceeds capacity"
        );
        ensure!(
            prefix.is_none_or(|p| p <= keys),
            "prefill prefix exceeds populated keys"
        );
        ensure!(
            prefix.is_none() || !shape.causal,
            "prefix-LM uses its own causal suffix mask"
        );
        let mode = if prefix.is_some() {
            2
        } else if shape.causal {
            0
        } else {
            1
        };
        ensure!(
            mode == 0 || (shape.window_left.is_none() && self.native.len() == 3),
            "prefill mask is not bound for this geometry/window"
        );
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
            "prefill dimensions exceed i32"
        );
        ensure!(
            shape
                .window_left
                .is_none_or(|window| window <= i32::MAX as usize),
            "prefill window exceeds i32"
        );
        ensure!(
            matches!(shape.head_dim, 128 | 256 | 512),
            "prefill binds head dims 128, 256 and 512 (got {})",
            shape.head_dim
        );
        ensure!(shape.rows > 0, "prefill needs at least one query row");
        ensure!(
            shape.kv_heads > 0 && shape.q_heads.is_multiple_of(shape.kv_heads),
            "prefill query heads must be a multiple of the KV heads"
        );
        ensure!(
            shape
                .start
                .checked_add(shape.rows)
                .is_some_and(|end| end <= shape.capacity),
            "prefill range exceeds cache"
        );
        ensure!(
            q_row_stride >= shape.q_heads * shape.head_dim
                && q_row_stride <= u32::MAX as usize
                && q_row_stride.is_multiple_of(8),
            "prefill Q row pitch {q_row_stride} is below one head row"
        );
        ensure!(
            shape.rows * shape.q_heads <= self.lse.len(),
            "prefill LSE buffer is too small"
        );
        let q_span = shape
            .rows
            .saturating_sub(1)
            .checked_mul(q_row_stride)
            .and_then(|span| span.checked_add(shape.q_heads * shape.head_dim))
            .context("prefill Q extent overflow")?;
        ensure!(q_span <= q.len(), "prefill Q buffer is too small");
        let cache_span = shape
            .capacity
            .checked_mul(shape.kv_heads)
            .and_then(|x| x.checked_mul(shape.head_dim))
            .ok_or_else(|| anyhow::anyhow!("prefill cache span overflow"))?;
        ensure!(
            cache_span <= k.len() && cache_span <= v.len(),
            "prefill KV buffers are too small"
        );
        ensure!(
            shape.rows * shape.q_heads * shape.head_dim <= out.len(),
            "prefill output buffer is too small"
        );
        self.stream.context().bind_to_thread()?;
        let (q, _q) = q.device_ptr(&self.stream);
        let (k, _k) = k.device_ptr(&self.stream);
        let (v, _v) = v.device_ptr(&self.stream);
        let (out, _out) = out.device_ptr_mut(&self.stream);
        let (lse, _lse) = self.lse.device_ptr_mut(&self.stream);
        if !self.native.is_empty() {
            ensure!(
                shape.window_left.is_none()
                    || self.geometry == [32, 4, 128]
                    || (self.geometry[2] == 256 && self.score.is_some()),
                "native prefill window is not bound for this geometry"
            );
            let window = shape.window_left.is_some();
            let native = self
                .native
                .get(if window && self.score.is_none() {
                    1
                } else {
                    mode
                })
                .context("native prefill mask is not bound")?;
            ensure!(
                self.score
                    .is_none_or(|score| shape.window_left == score.window_left),
                "native prefill window differs from setup binding"
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
            if let Some(split) = &mut self.split {
                ensure!(
                    shape.capacity == split.capacity,
                    "D512 capacity differs from setup binding"
                );
                if shape.rows <= 16 && split.parts > 1 {
                    let (partials, _partials) = split.workspace.device_ptr_mut(&self.stream);
                    let scale = self.score.context("D512 score is not bound")?.scale;
                    unsafe {
                        self.stream
                            .launch_builder(&split.partial)
                            .arg(&args)
                            .arg(&scale)
                            .arg(&partials)
                            .arg(&split.parts)
                            .arg(&split.chunk)
                            .launch(LaunchConfig {
                                grid_dim: (1, shape.q_heads as u32, split.parts),
                                block_dim: (128, 1, 1),
                                shared_mem_bytes: 90112,
                            })?;
                        self.stream
                            .launch_builder(&split.merge)
                            .arg(&args)
                            .arg(&partials)
                            .arg(&split.parts)
                            .launch(LaunchConfig {
                                grid_dim: (shape.rows as u32, shape.q_heads as u32, 1),
                                block_dim: (128, 1, 1),
                                shared_mem_bytes: 0,
                            })?;
                    }
                    return Ok(());
                }
            }
            let extent = shape.window_left.or(prefix).unwrap_or(keys) as u32;
            unsafe {
                let mut launch = self.stream.launch_builder(native);
                launch.arg(&args);
                if let Some(score) = &self.score {
                    let window = score.window_left.map_or(u32::MAX, |w| w as u32);
                    if shape.head_dim == 256 {
                        launch.arg(&window);
                    }
                    launch.arg(&score.scale);
                    launch.launch(LaunchConfig {
                        grid_dim: (
                            shape.rows.div_ceil(self.tile) as u32,
                            shape.q_heads as u32,
                            1,
                        ),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: self.shared,
                    })?;
                    return Ok(());
                }
                if mode != 0 || window {
                    launch.arg(&extent);
                }
                launch.launch(LaunchConfig {
                    grid_dim: (
                        shape.rows.div_ceil(self.tile) as u32,
                        shape.q_heads as u32,
                        1,
                    ),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: self.shared,
                })?;
            }
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        {
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
        #[cfg(feature = "rocm")]
        anyhow::bail!("HIP native prefill binding is missing")
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
    use super::*;

    #[test]
    fn split_chunks_cover_each_key_once() {
        for capacity in [1usize, 31, 32, 33, 129, 131072, 262144] {
            for heads in [8, 16, 32] {
                for resident in [1, 48, 128] {
                    let (parts, chunk) = split_extent(capacity, heads, resident).unwrap();
                    assert!(chunk.is_multiple_of(32));
                    assert_eq!(parts, capacity.div_ceil(chunk));
                    assert!((parts - 1) * chunk < capacity);
                    assert!(parts * chunk >= capacity);
                    for key in [0, capacity / 2, capacity - 1] {
                        let part = key / chunk;
                        assert!(part < parts && key < (part + 1) * chunk);
                    }
                }
            }
        }
        assert!(split_extent(0, 8, 128).is_err());
        assert!(split_extent(1, 0, 128).is_err());
        assert!(split_extent(1, 8, 0).is_err());
        assert!(split_extent(usize::MAX, 8, 128).is_err());
    }

    #[test]
    fn unit_window_excludes_first_expired_key() {
        assert!(Score::from_tokens(1.0, Some(0)).is_err());
        for tokens in [1usize, 512, 1024] {
            let score = Score::from_tokens(1.0, Some(tokens)).unwrap();
            let left = score.window_left.unwrap();
            for query in [0usize, tokens - 1, tokens, tokens + 1, 4095] {
                for key in 0..=query {
                    assert_eq!(key >= query.saturating_sub(left), key + tokens > query);
                }
            }
        }
    }

    #[test]
    fn native_prefill_parameter_layout() {
        assert_eq!(std::mem::size_of::<Attention>(), 64);
        assert_eq!(std::mem::align_of::<Attention>(), 8);
        assert_eq!(std::mem::offset_of!(Attention, rows), 40);
        assert_eq!(std::mem::offset_of!(Attention, q_stride), 48);
        assert_eq!(std::mem::offset_of!(Attention, start), 60);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "GPU lease; shared window and explicit score dispatch"]
    fn score_window_binding_gpu() -> Result<()> {
        let ctx = crate::backend::gpu::CudaContext::new(0)?;
        let stream = ctx.new_stream()?;
        for (d, hq, hkv, window) in [
            (128, 32, 4, Some(31)),
            (256, 8, 2, Some(31)),
            (512, 8, 2, None),
        ] {
            let case = Case {
                name: "score-binding".into(),
                d,
                hq,
                hkv,
                mode: if window.is_some() { "window" } else { "causal" }.into(),
                seed: 47,
                rows: 3,
                start: 77,
                keys: 80,
                extent: window.unwrap_or(0),
                capacity: 128,
                pitch: hq * d + 128,
            };
            let values: Vec<_> = (0..hkv * case.capacity * d)
                .map(|i| {
                    let head = i / (case.capacity * d);
                    let key = i / d % case.capacity;
                    bf16::from_f32(
                        ((key * 29 + head * 7 + i % d * 3) % 127) as f32 / 64.0 - 63.0 / 64.0,
                    )
                })
                .collect();
            let mut b = Buffers {
                q: stream.alloc_zeros(case.rows * case.pitch)?,
                k: stream.alloc_zeros(values.len())?,
                v: stream.clone_htod(&values)?,
                out: stream.alloc_zeros(case.rows * hq * d)?,
            };
            let mut op = if d == 128 {
                Flash::new(stream.clone(), case.rows, hq, hkv, d)?
            } else {
                Flash::new_with_score(
                    stream.clone(),
                    case.rows,
                    hq,
                    hkv,
                    d,
                    Score {
                        scale: 1.0,
                        window_left: window,
                    },
                    case.capacity,
                )?
            };
            case.run(&mut op, &mut b)?;
            let actual = stream.clone_dtoh(&b.out)?;
            let expected = uniform_reference(&case, &values);
            ensure!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| (a.to_f64() - b).abs() <= 0.0078125),
                "shared score/window output differs D={d}"
            );
            let lse = stream.clone_dtoh(&op.lse)?;
            for head in 0..hq {
                for row in 0..case.rows {
                    let count = (0..case.keys)
                        .filter(|&key| visible(&case, row, key))
                        .count();
                    ensure!(
                        (f64::from(lse[head * case.rows + row]) - (count as f64).ln()).abs()
                            <= 5.142_130_660_829_025e-6,
                        "shared score/window LSE differs D={d}"
                    );
                }
            }
            println!(
                "PASS score_window_binding D={d} heads={hq} kv_heads={hkv} window={window:?} rows=3"
            );
        }
        Ok(())
    }
    #[derive(Clone, serde::Deserialize)]
    struct Case {
        name: String,
        d: usize,
        hq: usize,
        hkv: usize,
        mode: String,
        seed: usize,
        rows: usize,
        start: usize,
        keys: usize,
        extent: usize,
        capacity: usize,
        pitch: usize,
    }
    struct Buffers {
        q: CudaSlice<bf16>,
        k: CudaSlice<bf16>,
        v: CudaSlice<bf16>,
        out: CudaSlice<bf16>,
    }
    impl Case {
        fn pattern(&self, count: usize, seed: usize) -> Vec<bf16> {
            (0..count)
                .map(|i| {
                    bf16::from_f32(
                        ((i * 17 + seed * 31) % 257) as f32 / 64.0 - 2.0
                            + (((i * 29 + seed * 13) % 263) as f32 - 131.0) / 128.0,
                    )
                })
                .collect()
        }
        fn buffers(&self, stream: &Arc<CudaStream>) -> Result<Buffers> {
            let width = self.hq * self.d;
            let mut q = vec![bf16::from_bits(0x7fc1); self.rows * self.pitch];
            for (row, data) in q.chunks_exact_mut(self.pitch).enumerate() {
                for (d, value) in data[..width].iter_mut().enumerate() {
                    let i = row * width + d;
                    *value = bf16::from_f32(
                        ((i * 17 + self.seed * 31) % 257) as f32 / 64.0 - 2.0
                            + (((i * 29 + self.seed * 13) % 263) as f32 - 131.0) / 128.0,
                    );
                }
            }
            let q = stream.clone_htod(&q)?;
            let elements = self.hkv * self.capacity * self.d;
            Ok(Buffers {
                q,
                k: stream.clone_htod(&self.pattern(elements, self.seed + 7))?,
                v: stream.clone_htod(&self.pattern(elements, self.seed + 19))?,
                out: stream.clone_htod(&vec![bf16::from_bits(0x7fc1); self.rows * width + 32])?,
            })
        }
        fn run(&self, op: &mut Flash, b: &mut Buffers) -> Result<()> {
            op.run_masked(Query {
                q: &b.q,
                q_row_stride: self.pitch,
                k: &b.k,
                v: &b.v,
                out: &mut b.out,
                shape: Shape {
                    rows: self.rows,
                    start: self.start,
                    q_heads: self.hq,
                    kv_heads: self.hkv,
                    capacity: self.capacity,
                    head_dim: self.d,
                    causal: matches!(self.mode.as_str(), "causal" | "window"),
                    window_left: (self.mode == "window").then_some(self.extent),
                },
                keys: self.keys,
                prefix: (self.mode == "prefix").then_some(self.extent),
            })
        }
    }
    fn reference_f64(tensors: &safetensors::SafeTensors<'_>, name: &str) -> Result<Vec<f64>> {
        let view = tensors.tensor(name)?;
        ensure!(
            view.dtype() == safetensors::Dtype::F64,
            "reference dtype must be F64"
        );
        Ok(view
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&v| f64::from_le_bytes(v))
            .collect())
    }
    fn errors(actual: &[f64], expected: &[f64]) -> Result<(f64, f64)> {
        ensure!(
            actual.len() == expected.len() && !actual.is_empty(),
            "reference extent"
        );
        ensure!(
            actual.iter().chain(expected).all(|x| x.is_finite()),
            "nonfinite attention output"
        );
        let mut maximum = 0f64;
        let mut square = 0f64;
        let mut norm = 0f64;
        for (&a, &b) in actual.iter().zip(expected) {
            maximum = maximum.max((a - b).abs() / (1.0 + b.abs()));
            square += (a - b).powi(2);
            norm += b * b;
        }
        Ok((
            maximum,
            (square / actual.len() as f64).sqrt() / (1.0 + (norm / actual.len() as f64).sqrt()),
        ))
    }
    fn cell(
        stream: &Arc<CudaStream>,
        case: &Case,
        root: &std::path::Path,
        bound: &serde_json::Value,
    ) -> Result<usize> {
        let data = std::fs::read(root.join(format!("{}.safetensors", case.name)))?;
        let tensors = safetensors::SafeTensors::deserialize(&data)?;
        let selected = tensors
            .tensor("selected")?
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&v| usize::try_from(i64::from_le_bytes(v)))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        ensure!(
            selected.iter().all(|&r| r < case.rows),
            "selected reference rows"
        );
        let reference = reference_f64(&tensors, "output")?;
        let reference_lse = reference_f64(&tensors, "lse")?;
        let mut buffers = case.buffers(stream)?;
        let mut op = Flash::new(stream.clone(), case.rows, case.hq, case.hkv, case.d)?;
        case.run(&mut op, &mut buffers)?;
        let first = stream.clone_dtoh(&buffers.out)?;
        let first_lse = stream.clone_dtoh(&op.lse)?;
        let mut defects = 0;
        ensure!(
            first[..case.rows * case.hq * case.d]
                .iter()
                .all(|x| x.is_finite()),
            "nonfinite prefill output"
        );
        defects += usize::from(
            !first[case.rows * case.hq * case.d..]
                .iter()
                .all(|x| x.to_bits() == 0x7fc1),
        );
        #[cfg(feature = "cuda")]
        stream.begin_capture(
            cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )?;
        #[cfg(feature = "rocm")]
        let capture = stream.begin_capture()?;
        case.run(&mut op, &mut buffers)?;
        #[cfg(feature = "cuda")]
        let graph = stream
            .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))?
            .context("attention graph missing")?;
        #[cfg(feature = "rocm")]
        let graph = capture.finish()?;
        for repeat in 0..3 {
            if repeat < 2 {
                graph.launch()?;
            } else {
                case.run(&mut op, &mut buffers)?;
            }
            defects += usize::from(
                !stream
                    .clone_dtoh(&buffers.out)?
                    .iter()
                    .zip(&first)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
            );
            defects += usize::from(
                !stream
                    .clone_dtoh(&op.lse)?
                    .iter()
                    .zip(&first_lse)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
            );
        }
        let actual = selected
            .iter()
            .flat_map(|&row| {
                first[row * case.hq * case.d..(row + 1) * case.hq * case.d]
                    .iter()
                    .map(|v| v.to_f64())
            })
            .collect::<Vec<_>>();
        let lse = selected
            .iter()
            .flat_map(|&row| {
                let values = &first_lse;
                (0..case.hq).map(move |h| f64::from(values[h * case.rows + row]))
            })
            .collect::<Vec<_>>();
        let (maximum, rms) = errors(&actual, &reference)?;
        let (lse_error, _) = errors(&lse, &reference_lse)?;
        let limit = |name| {
            bound[name]
                .as_f64()
                .context("missing frozen attention limit")
        };
        if lse_error > limit("lse_max")? {
            let (index, (&actual, &expected)) = lse
                .iter()
                .zip(&reference_lse)
                .enumerate()
                .max_by(|(_, (a, b)), (_, (c, d))| {
                    ((**a - **b).abs() / (1.0 + b.abs()))
                        .total_cmp(&((**c - **d).abs() / (1.0 + d.abs())))
                })
                .context("LSE reference empty")?;
            println!(
                "FFA_LSE_DEFECT name={} row={} head={} actual={actual:.17e} expected={expected:.17e}",
                case.name,
                selected[index / case.hq],
                index % case.hq
            );
        }
        defects += usize::from(
            maximum > limit("output_max")?
                || rms > limit("output_rms")?
                || lse_error > limit("lse_max")?,
        );
        println!(
            "FFA_HIP_CELL name={} d={} hq={} hkv={} mode={} rows={} start={} max={maximum:e} rms={rms:e} lse={lse_error:e} defects={defects}",
            case.name, case.d, case.hq, case.hkv, case.mode, case.rows, case.start
        );
        Ok(defects)
    }
    fn visible(case: &Case, row: usize, key: usize) -> bool {
        let query = case.start + row;
        match case.mode.as_str() {
            "bidirectional" => key < case.keys,
            "prefix" => key < case.keys && key < case.extent.max(query + 1),
            "window" => key <= query && query - key <= case.extent,
            _ => key <= query,
        }
    }
    fn uniform_reference(case: &Case, values: &[bf16]) -> Vec<f64> {
        (0..case.rows)
            .flat_map(|row| {
                let count = (0..case.keys)
                    .filter(|&key| visible(case, row, key))
                    .count();
                (0..case.hq * case.d).map(move |column| {
                    let kv = column / case.d / (case.hq / case.hkv);
                    (0..case.keys)
                        .filter(|&key| visible(case, row, key))
                        .map(|key| {
                            values[(kv * case.capacity + key) * case.d + column % case.d].to_f64()
                        })
                        .sum::<f64>()
                        / count as f64
                })
            })
            .collect()
    }
    fn mask_control(
        stream: &Arc<CudaStream>,
        case: &Case,
        limits: &serde_json::Value,
    ) -> Result<usize> {
        let mut values = vec![bf16::ZERO; case.hkv * case.capacity * case.d];
        let spike = match case.mode.as_str() {
            "prefix" => case.extent - 1,
            "window" => case.start - case.extent,
            _ => case.keys - 1,
        };
        for kv in 0..case.hkv {
            for key in 0..case.capacity {
                let value = kv as f32 * 4.0 + if key == spike { 64.0 } else { 0.0 }
                    - if case.mode == "prefix" && key == case.keys - 1 {
                        128.0
                    } else {
                        0.0
                    };
                values
                    [(kv * case.capacity + key) * case.d..(kv * case.capacity + key + 1) * case.d]
                    .fill(bf16::from_f32(value));
            }
        }
        let mut b = Buffers {
            q: stream.alloc_zeros(case.rows * case.pitch)?,
            k: stream.alloc_zeros(values.len())?,
            v: stream.clone_htod(&values)?,
            out: stream.alloc_zeros(case.rows * case.hq * case.d)?,
        };
        let mut op = Flash::new(stream.clone(), case.rows, case.hq, case.hkv, case.d)?;
        let expected = uniform_reference(case, &values);
        let limit = limits["limits"][format!("{}-{}", case.d, case.mode)]["output_max"]
            .as_f64()
            .context("mask control limit")?;
        case.run(&mut op, &mut b)?;
        let actual: Vec<_> = stream
            .clone_dtoh(&b.out)?
            .iter()
            .map(|v| v.to_f64())
            .collect();
        let (maximum, _) = errors(&actual, &expected)?;
        let mut defects = usize::from(maximum > limit);
        let mut mutant = case.clone();
        match case.mode.as_str() {
            "prefix" | "bidirectional" => mutant.mode = "causal".into(),
            "window" => mutant.extent -= 1,
            _ => {
                mutant.start += 1;
                mutant.keys += 1;
            }
        }
        mutant.run(&mut op, &mut b)?;
        let wrong: Vec<_> = stream
            .clone_dtoh(&b.out)?
            .iter()
            .map(|v| v.to_f64())
            .collect();
        let (wrong_error, _) = errors(&wrong, &expected)?;
        defects += usize::from(wrong_error <= limit);
        values.rotate_left(case.capacity * case.d);
        stream.memcpy_htod(&values, &mut b.v)?;
        case.run(&mut op, &mut b)?;
        let wrong: Vec<_> = stream
            .clone_dtoh(&b.out)?
            .iter()
            .map(|v| v.to_f64())
            .collect();
        let (head_error, _) = errors(&wrong, &expected)?;
        defects += usize::from(head_error <= limit);
        println!(
            "FFA_HIP_CONTROL d={} mode={} actual={maximum:e} mask_mutant={wrong_error:e} head_mutant={head_error:e} limit={limit:e} defects={defects}",
            case.d, case.mode
        );
        if case.mode == "window" {
            values.rotate_right(case.capacity * case.d);
            stream.memcpy_htod(&values, &mut b.v)?;
            case.run(&mut op, &mut b)?;
            let sinks = (0..case.hq)
                .map(|h| bf16::from_f32([0.0, 4.0, -4.0][h % 3]))
                .collect::<Vec<_>>();
            op.apply_sinks(&stream.clone_htod(&sinks)?, &mut b.out, case.rows, case.hq)?;
            let wanted = expected
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let row = i / (case.hq * case.d);
                    let head = i / case.d % case.hq;
                    let count = (0..case.keys)
                        .filter(|&key| visible(case, row, key))
                        .count() as f64;
                    bf16::from_f64(v).to_f64() * count / (count + sinks[head].to_f64().exp())
                })
                .collect::<Vec<_>>();
            let actual: Vec<_> = stream
                .clone_dtoh(&b.out)?
                .iter()
                .map(|v| v.to_f64())
                .collect();
            let (sink_error, _) = errors(&actual, &wanted)?;
            let (omitted, _) = errors(&expected, &wanted)?;
            defects += usize::from(sink_error > limit || omitted <= limit);
            println!(
                "FFA_HIP_SINK actual={sink_error:e} omitted_mutant={omitted:e} limit={limit:e} defects={defects}"
            );
        }
        Ok(defects)
    }
    fn mask_controls(stream: &Arc<CudaStream>, limits: &serde_json::Value) -> Result<usize> {
        let mut defects = 0;
        for (d, hq, hkv, mode, rows, start, keys, extent) in [
            (256, 24, 4, "causal", 33, 17, 50, 0),
            (128, 16, 2, "causal", 33, 17, 50, 0),
            (128, 16, 8, "bidirectional", 17, 0, 33, 0),
            (128, 12, 12, "prefix", 65, 0, 65, 33),
            (128, 32, 4, "window", 17, 127, 144, 31),
        ] {
            let case = Case {
                name: "control".into(),
                d,
                hq,
                hkv,
                mode: mode.into(),
                rows,
                start,
                keys,
                extent,
                seed: 31,
                capacity: keys + 23,
                pitch: hq * d + 128,
            };
            defects += mask_control(stream, &case, limits)?;
        }
        Ok(defects)
    }
    #[test]
    #[ignore = "GPU lease and independent attention reference fixtures"]
    fn native_prefill_reference_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_DENSE_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window missing"
        );
        let root = std::path::PathBuf::from(std::env::var("FF_DENSE_REFERENCE")?);
        let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(root.join("cases.json"))?)?;
        let limits: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/attention_reference.json"
        ))?;
        ensure!(!cases.is_empty(), "attention fixtures empty");
        let ctx = crate::backend::gpu::CudaContext::new(0)?;
        #[cfg(feature = "cuda")]
        // SAFETY: every allocation, launch and copy uses the single test stream.
        unsafe {
            ctx.disable_event_tracking();
        }
        let stream = ctx.new_stream()?;
        let mut defects = mask_controls(&stream, &limits)?;
        for case in &cases {
            let bound = &limits["limits"][format!("{}-{}", case.d, case.mode)];
            defects += cell(&stream, case, &root, bound)?;
        }
        println!(
            "FFA_HIP_RESULT cells={} defects={defects} frozen_limits=true",
            cases.len()
        );
        ensure!(defects == 0, "native prefill attention DEFECT");
        Ok(())
    }
}
