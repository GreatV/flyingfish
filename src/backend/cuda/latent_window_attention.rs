//! Latent window attention: batch-1 decode over the per-layer ring of FP8
//! latent K rows with one attention sink per head
//! (kernels/latent_window_attention.cu). Semantics per the DeepSeek-V4.1-Flash
//! model definition (publisher inference/model.py Attention.forward,
//! _window_kv and get_window_topk_idxs): the visible window is the ring in
//! oldest-first order, the sink enters the softmax denominator only, the value
//! is the K content itself, and the output tail gets the inverse RoPE.
//! The same module stages the ring and the attention output: the ring write
//! (per-32 E4M3/E8M0 quantization placed at the position's slot), the visible
//! slot list, the adjacent-pair rotary on the tail, and the block-diagonal
//! wo_a group projection of the quantized attention output.
use crate::backend::cuda::contracts::{Tensor, TensorMut, dense_layout, no_overlap};
use crate::backend::cuda::cubin;
use anyhow::{Result, ensure};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use half::bf16;
use std::sync::Arc;

#[derive(Clone, Copy)]
pub(crate) struct IndexedInputs {
    pub window: LwaInputs,
    pub packed: Tensor<u8, 2>,
    pub scales: Tensor<u8, 2>,
    pub selected: Tensor<i32, 2>,
    pub committed: Tensor<u32, 1>,
    pub sequence: crate::backend::cuda::contracts::SeqRef,
    pub ratio: u32,
}

pub(crate) struct IndexedAttentionPlan {
    kernel: CudaFunction,
    inputs: IndexedInputs,
    output: Tensor<bf16, 2>,
    shape: [i32; 8],
    scale: f32,
    grid: LaunchConfig,
}

impl IndexedAttentionPlan {
    pub(crate) fn new(
        ctx: &Arc<CudaContext>,
        inputs: IndexedInputs,
        output: Tensor<bf16, 2>,
        query: u32,
    ) -> Result<Self> {
        let dims = validate(inputs.window, &output)?;
        let capacity = inputs.packed.shape[0];
        require_dense(&inputs.packed, "compressed codes")?;
        require_dense(&inputs.scales, "compressed scales")?;
        require_dense(&inputs.selected, "compressed selection")?;
        require_dense(&inputs.committed, "compressed count")?;
        inputs.sequence.validate()?;
        ensure!(
            dims.head_dim == 512
                && dims.window == 128
                && inputs.packed.shape == [capacity, 256]
                && inputs.scales.shape == [capacity, 32]
                && inputs.selected.shape[1] == 512
                && query < inputs.selected.shape[0]
                && inputs.committed.shape == [1]
                && matches!(inputs.ratio, 1 | 2),
            "indexed attention requires D512/W128, native FP4-per16, and a valid query row"
        );
        for (ptr, bytes) in [
            (inputs.packed.ptr, inputs.packed.span_bytes),
            (inputs.scales.ptr, inputs.scales.span_bytes),
            (inputs.selected.ptr, inputs.selected.span_bytes),
            (inputs.committed.ptr, inputs.committed.span_bytes),
            (inputs.sequence.ptr, 32),
            (inputs.window.q.ptr, inputs.window.q.span_bytes),
            (inputs.window.k_codes.ptr, inputs.window.k_codes.span_bytes),
            (
                inputs.window.k_scales.ptr,
                inputs.window.k_scales.span_bytes,
            ),
        ] {
            ensure!(
                crate::backend::cuda::contracts::spans_disjoint(
                    ptr,
                    bytes,
                    output.ptr,
                    output.span_bytes
                )?,
                "indexed attention output aliases an input"
            );
        }
        let shape = [
            dims.heads,
            dims.head_dim,
            dims.window,
            capacity,
            inputs.ratio,
            query,
            dims.rope_tail,
            inputs.selected.shape[0],
        ];
        ensure!(
            shape.iter().all(|&v| v <= i32::MAX as u32),
            "indexed attention extent overflow"
        );
        Ok(Self {
            kernel: cubin::module(ctx, "latent_window_attention")?
                .load_function("latent_indexed_attention")?,
            inputs,
            output,
            shape: shape.map(|v| v as i32),
            scale: (dims.head_dim as f32).sqrt().recip(),
            grid: LaunchConfig {
                grid_dim: (dims.heads, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: (2 * dims.head_dim + dims.window + 512) * 4,
            },
        })
    }

    pub(crate) fn run(&self, stream: &Arc<CudaStream>) -> Result<()> {
        let v = self.inputs;
        unsafe {
            stream
                .launch_builder(&self.kernel)
                .arg(&v.window.q.ptr)
                .arg(&v.window.k_codes.ptr)
                .arg(&v.window.k_scales.ptr)
                .arg(&v.packed.ptr)
                .arg(&v.scales.ptr)
                .arg(&v.selected.ptr)
                .arg(&v.committed.ptr)
                .arg(&v.window.sink.ptr)
                .arg(&v.window.inv_rotary.ptr)
                .arg(&v.sequence.ptr)
                .arg(&self.output.ptr)
                .arg(&self.shape[0])
                .arg(&self.shape[1])
                .arg(&self.shape[2])
                .arg(&self.shape[3])
                .arg(&self.shape[4])
                .arg(&self.shape[5])
                .arg(&self.shape[6])
                .arg(&self.scale)
                .launch(self.grid)?;
        }
        Ok(())
    }
}

/// Threads per latent window attention block.
const LWA_THREADS: u32 = 256;
/// E4M3 codes per E8M0 scale along the head vector.
const SCALE_BLOCK: u32 = 32;
/// Dynamic shared bytes one block may request below the 48 KB limit.
const LWA_SHARED_LIMIT: u64 = 48 * 1024 - 1024;

/// Inputs of one latent window attention call.
#[derive(Clone, Copy)]
pub struct LwaInputs {
    /// `[heads, head_dim]` bf16 post-RoPE query.
    pub q: Tensor<bf16, 2>,
    /// `[window, head_dim]` E4M3 codes of the ring rows.
    pub k_codes: Tensor<u8, 2>,
    /// `[window, head_dim/32]` E8M0 scales of the ring rows.
    pub k_scales: Tensor<u8, 2>,
    /// `[slots]` visible ring slot ids, oldest first.
    pub slot_ids: Tensor<i32, 1>,
    /// `[heads]` f32 per-head attention sink.
    pub sink: Tensor<f32, 1>,
    /// `[rope_tail]` f32 interleaved (cos, sin) of the inverse RoPE angle.
    pub inv_rotary: Tensor<f32, 1>,
}

/// Derived extents of one call.
#[derive(Debug)]
struct LwaDims {
    heads: u32,
    head_dim: u32,
    window: u32,
    slots: u32,
    rope_tail: u32,
}

fn require_dense<T: Copy, const R: usize>(tensor: &Tensor<T, R>, name: &str) -> Result<()> {
    tensor.validate()?;
    let (stride, _) = dense_layout(tensor.shape)?;
    ensure!(
        tensor.stride == stride,
        "latent window attention {name} must be dense: stride {:?} != {:?}",
        tensor.stride,
        stride
    );
    Ok(())
}

/// Validate the call extents against each other.
fn validate(inputs: LwaInputs, out: &Tensor<bf16, 2>) -> Result<LwaDims> {
    require_dense(&inputs.q, "q")?;
    require_dense(&inputs.k_codes, "k_codes")?;
    require_dense(&inputs.k_scales, "k_scales")?;
    require_dense(&inputs.slot_ids, "slot_ids")?;
    require_dense(&inputs.sink, "sink")?;
    require_dense(&inputs.inv_rotary, "inv_rotary")?;
    require_dense(out, "out")?;
    let heads = inputs.q.shape[0];
    let head_dim = inputs.q.shape[1];
    ensure!(
        heads >= 1,
        "latent window attention heads {heads} is below 1"
    );
    ensure!(
        head_dim >= 1,
        "latent window attention head_dim {head_dim} is below 1"
    );
    ensure!(
        head_dim.is_multiple_of(SCALE_BLOCK),
        "latent window attention head_dim {head_dim} must be a multiple of {SCALE_BLOCK} for one E8M0 scale per {SCALE_BLOCK} codes"
    );
    let window = inputs.k_codes.shape[0];
    ensure!(
        inputs.k_codes.shape[1] == head_dim,
        "latent window attention k_codes shape {:?} != expected [{}, {}]",
        inputs.k_codes.shape,
        window,
        head_dim
    );
    let nblocks = head_dim / SCALE_BLOCK;
    ensure!(
        inputs.k_scales.shape == [window, nblocks],
        "latent window attention k_scales shape {:?} != expected [{}, {}]",
        inputs.k_scales.shape,
        window,
        nblocks
    );
    let slots = inputs.slot_ids.shape[0];
    ensure!(
        slots >= 1 && slots <= window,
        "latent window attention slots {slots} outside 1..={window}"
    );
    ensure!(
        inputs.sink.shape[0] == heads,
        "latent window attention sink len {} != heads {heads}",
        inputs.sink.shape[0]
    );
    let rope_tail = inputs.inv_rotary.shape[0];
    ensure!(
        rope_tail.is_multiple_of(2),
        "latent window attention rope_tail {rope_tail} must be even"
    );
    ensure!(
        rope_tail <= head_dim,
        "latent window attention rope_tail {rope_tail} exceeds head_dim {head_dim}"
    );
    ensure!(
        out.shape == [heads, head_dim],
        "latent window attention out shape {:?} != expected [{}, {}]",
        out.shape,
        heads,
        head_dim
    );
    Ok(LwaDims {
        heads,
        head_dim,
        window,
        slots,
        rope_tail,
    })
}

/// Ring write extents against each other; the quant parameters (one E8M0
/// scale per 32 codes) are checked here.
fn validate_ring_write(
    kv: &Tensor<bf16, 2>,
    position: i32,
    codes: &Tensor<u8, 2>,
    scales: &Tensor<u8, 2>,
) -> Result<()> {
    require_dense(kv, "ring kv")?;
    require_dense(codes, "ring codes")?;
    require_dense(scales, "ring scales")?;
    let rows = kv.shape[0];
    let head_dim = kv.shape[1];
    let window = codes.shape[0];
    ensure!(rows >= 1, "latent ring write rows {rows} is below 1");
    // Rows `window` apart share a slot; one launch writes each slot at most once.
    ensure!(
        rows <= window,
        "latent ring write rows {rows} exceed the {window}-slot window"
    );
    ensure!(
        head_dim.is_multiple_of(SCALE_BLOCK),
        "latent ring write head_dim {head_dim} must be a multiple of {SCALE_BLOCK} for one E8M0 scale per {SCALE_BLOCK} codes"
    );
    let nblocks = head_dim / SCALE_BLOCK;
    ensure!(
        codes.shape == [window, head_dim],
        "latent ring write codes shape {:?} != expected [{}, {}]",
        codes.shape,
        window,
        head_dim
    );
    ensure!(
        scales.shape == [window, nblocks],
        "latent ring write scales shape {:?} != expected [{}, {}]",
        scales.shape,
        window,
        nblocks
    );
    ensure!(
        position >= 0,
        "latent ring write position {position} is negative"
    );
    no_overlap(kv, codes, "ring kv", "ring codes")?;
    no_overlap(kv, scales, "ring kv", "ring scales")?;
    Ok(())
}

/// Block-diagonal group projection extents against each other, including the
/// 32-divisibility that keeps one weight scale block inside one group.
fn validate_group_project(
    act_codes: &Tensor<u8, 3>,
    act_scales: &Tensor<u8, 3>,
    w_codes: &Tensor<u8, 2>,
    w_scales: &Tensor<u8, 2>,
    out: &Tensor<bf16, 2>,
) -> Result<()> {
    require_dense(act_codes, "group act codes")?;
    require_dense(act_scales, "group act scales")?;
    require_dense(w_codes, "group weight codes")?;
    require_dense(w_scales, "group weight scales")?;
    require_dense(out, "group out")?;
    let rows = act_codes.shape[0];
    let groups = act_codes.shape[1];
    let width = act_codes.shape[2];
    let rank_rows = w_codes.shape[0];
    ensure!(
        rows >= 1 && groups >= 1,
        "latent group project rows {rows} or groups {groups} are below 1"
    );
    ensure!(
        width.is_multiple_of(SCALE_BLOCK * 8),
        "latent group project width {width} must be a multiple of {} so that every stride lane owns a contiguous partial (one E8M0 scale per {SCALE_BLOCK} codes)",
        SCALE_BLOCK * 8
    );
    let rank_rows_u32 = rank_rows / groups;
    let rank = rank_rows_u32 as usize;
    ensure!(
        rank_rows_u32 >= 1 && rank_rows == rank_rows_u32 * groups,
        "latent group project weight rows {rank_rows} are not a multiple of groups {groups}"
    );
    ensure!(
        rank.is_multiple_of(SCALE_BLOCK as usize),
        "latent group project rank {rank} must be a multiple of {SCALE_BLOCK} so one 32x32 weight scale block never spans two groups"
    );
    let nblocks = width / SCALE_BLOCK;
    ensure!(
        act_scales.shape == [rows, groups, nblocks],
        "latent group project act scales shape {:?} != expected [{rows}, {groups}, {nblocks}]",
        act_scales.shape
    );
    ensure!(
        w_codes.shape[1] == width,
        "latent group project weight codes width {} != activation width {width}",
        w_codes.shape[1]
    );
    ensure!(
        w_scales.shape == [rank_rows / SCALE_BLOCK, nblocks],
        "latent group project weight scales shape {:?} != expected [{}, {nblocks}]",
        w_scales.shape,
        rank_rows / SCALE_BLOCK
    );
    ensure!(
        out.shape == [rows, rank_rows],
        "latent group project out shape {:?} != expected [{rows}, {rank_rows}]",
        out.shape
    );
    no_overlap(act_codes, out, "group act codes", "group out")?;
    no_overlap(act_scales, out, "group act scales", "group out")?;
    no_overlap(w_codes, out, "group weight codes", "group out")?;
    no_overlap(w_scales, out, "group weight scales", "group out")?;
    Ok(())
}

/// Latent window attention over a ring of FP8 latent K rows.
pub struct LatentWindowAttention {
    kernel: CudaFunction,
    ring_write: CudaFunction,
    act_quant: CudaFunction,
    window_slots: CudaFunction,
    group_project: CudaFunction,
}

impl LatentWindowAttention {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let module = cubin::module(ctx, "latent_window_attention")?;
        Ok(Self {
            kernel: module.load_function("latent_window_attention")?,
            ring_write: module.load_function("latent_ring_write")?,
            act_quant: module.load_function("latent_act_quant")?,
            window_slots: module.load_function("latent_window_slots")?,
            group_project: module.load_function("latent_group_project")?,
        })
    }

    /// Attention of one query over the visible ring slots.
    ///
    /// `slot_ids` holds the visible ring slot ids oldest first, `inv_rotary`
    /// the interleaved (cos, sin) factors of the inverse RoPE angle.
    pub fn run(
        &self,
        stream: &Arc<CudaStream>,
        inputs: LwaInputs,
        out: &mut TensorMut<bf16, 2>,
        p_mode: i32,
    ) -> Result<()> {
        let dims = validate(inputs, &out.0)?;
        no_overlap(&inputs.q, &out.0, "q", "out")?;
        no_overlap(&inputs.k_codes, &out.0, "k_codes", "out")?;
        no_overlap(&inputs.k_scales, &out.0, "k_scales", "out")?;
        no_overlap(&inputs.slot_ids, &out.0, "slot_ids", "out")?;
        no_overlap(&inputs.sink, &out.0, "sink", "out")?;
        no_overlap(&inputs.inv_rotary, &out.0, "inv_rotary", "out")?;
        let shared = (2 * u64::from(dims.head_dim) + u64::from(dims.window)) * 4;
        ensure!(
            shared <= LWA_SHARED_LIMIT,
            "latent window attention shared staging {shared} B exceeds the {LWA_SHARED_LIMIT} B limit (head_dim {}, window {})",
            dims.head_dim,
            dims.window
        );
        // softmax_scale = head_dim^-0.5 (model definition, Attention.__init__).
        let scale = 1.0f32 / (dims.head_dim as f32).sqrt();
        let (heads, head_dim, window, slots, rope_tail) = (
            i32::try_from(dims.heads)?,
            i32::try_from(dims.head_dim)?,
            i32::try_from(dims.window)?,
            i32::try_from(dims.slots)?,
            i32::try_from(dims.rope_tail)?,
        );
        unsafe {
            stream
                .launch_builder(&self.kernel)
                .arg(&inputs.q.ptr)
                .arg(&inputs.k_codes.ptr)
                .arg(&inputs.k_scales.ptr)
                .arg(&inputs.slot_ids.ptr)
                .arg(&inputs.sink.ptr)
                .arg(&inputs.inv_rotary.ptr)
                .arg(&out.0.ptr)
                .arg(&heads)
                .arg(&head_dim)
                .arg(&window)
                .arg(&slots)
                .arg(&rope_tail)
                .arg(&scale)
                .arg(&p_mode)
                .launch(LaunchConfig {
                    grid_dim: (dims.heads, 1, 1),
                    block_dim: (LWA_THREADS, 1, 1),
                    shared_mem_bytes: shared as u32,
                })?;
        }
        Ok(())
    }

    /// Ring write: quantize the post-RoPE K rows per 32 (E4M3 codes, one E8M0
    /// scale per block) and place row `r` at ring slot `(position + r) % window`.
    pub fn ring_write(
        &self,
        stream: &Arc<CudaStream>,
        kv: Tensor<bf16, 2>,
        position: i32,
        codes: &mut TensorMut<u8, 2>,
        scales: &mut TensorMut<u8, 2>,
    ) -> Result<()> {
        validate_ring_write(&kv, position, &codes.0, &scales.0)?;
        let rows = kv.shape[0];
        let head_dim = kv.shape[1];
        let window = codes.0.shape[0];
        let groups = (head_dim / SCALE_BLOCK) as usize;
        let warps = rows as usize * groups;
        unsafe {
            stream
                .launch_builder(&self.ring_write)
                .arg(&kv.ptr)
                .arg(&codes.0.ptr)
                .arg(&scales.0.ptr)
                .arg(&position)
                .arg(&i32::try_from(rows)?)
                .arg(&i32::try_from(head_dim)?)
                .arg(&i32::try_from(window)?)
                .launch(LaunchConfig {
                    grid_dim: (u32::try_from(warps.div_ceil(8))?, 1, 1),
                    block_dim: (LWA_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// Per-32 E4M3/E8M0 quantization of `x` rows into linear codes and scales.
    pub fn act_quant(
        &self,
        stream: &Arc<CudaStream>,
        x: Tensor<bf16, 2>,
        codes: &mut TensorMut<u8, 2>,
        scales: &mut TensorMut<u8, 2>,
    ) -> Result<()> {
        require_dense(&x, "quant input")?;
        require_dense(&codes.0, "quant codes")?;
        require_dense(&scales.0, "quant scales")?;
        let rows = x.shape[0];
        let width = x.shape[1];
        ensure!(rows >= 1, "latent act quant rows {rows} is below 1");
        ensure!(
            width.is_multiple_of(SCALE_BLOCK),
            "latent act quant width {width} must be a multiple of {SCALE_BLOCK} for one E8M0 scale per {SCALE_BLOCK} codes"
        );
        let nblocks = width / SCALE_BLOCK;
        ensure!(
            codes.0.shape == [rows, width],
            "latent act quant codes shape {:?} != expected [{}, {}]",
            codes.0.shape,
            rows,
            width
        );
        ensure!(
            scales.0.shape == [rows, nblocks],
            "latent act quant scales shape {:?} != expected [{}, {}]",
            scales.0.shape,
            rows,
            nblocks
        );
        no_overlap(&x, &codes.0, "quant input", "quant codes")?;
        no_overlap(&x, &scales.0, "quant input", "quant scales")?;
        let warps = rows as usize * nblocks as usize;
        unsafe {
            stream
                .launch_builder(&self.act_quant)
                .arg(&x.ptr)
                .arg(&codes.0.ptr)
                .arg(&scales.0.ptr)
                .arg(&(rows as i64))
                .arg(&i32::try_from(width)?)
                .launch(LaunchConfig {
                    grid_dim: (u32::try_from(warps.div_ceil(8))?, 1, 1),
                    block_dim: (LWA_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// The visible ring slots of the query at `position`, oldest first; the
    /// caller exposes `min(position + 1, window)` of the buffer.
    pub fn window_slots(
        &self,
        stream: &Arc<CudaStream>,
        slots: &mut TensorMut<i32, 1>,
        position: i32,
    ) -> Result<()> {
        require_dense(&slots.0, "window slots")?;
        let window = slots.0.shape[0];
        ensure!(
            window >= 1,
            "latent window slots window {window} is below 1"
        );
        ensure!(
            position >= 0,
            "latent window slots position {position} is negative"
        );
        unsafe {
            stream
                .launch_builder(&self.window_slots)
                .arg(&slots.0.ptr)
                .arg(&position)
                .arg(&i32::try_from(window)?)
                .launch(crate::backend::cuda::ops::flat(window as usize))?;
        }
        Ok(())
    }

    /// Block-diagonal group projection of the quantized attention output:
    /// group `g` of each row is projected by weight rows `[g*rank, (g+1)*rank)`
    /// with one E8M0 scale per 32x32 weight block (model definition einsum
    /// `bsgd,grd->bsgr` over the per-32 quantized o). The dot accumulates exact
    /// 16-element sums into eight contiguous FP32 partials, the definition's
    /// mma k-step, then reduces the partials in a five-level tree.
    pub fn group_project(
        &self,
        stream: &Arc<CudaStream>,
        act_codes: Tensor<u8, 3>,
        act_scales: Tensor<u8, 3>,
        w_codes: Tensor<u8, 2>,
        w_scales: Tensor<u8, 2>,
        out: &mut TensorMut<bf16, 2>,
    ) -> Result<()> {
        validate_group_project(&act_codes, &act_scales, &w_codes, &w_scales, &out.0)?;
        let rows = act_codes.shape[0];
        let groups = act_codes.shape[1];
        let rank_rows = w_codes.shape[0];
        let rank = (rank_rows / groups) as usize;
        let width = act_codes.shape[2];
        let warps = rows as usize * rank_rows as usize;
        unsafe {
            stream
                .launch_builder(&self.group_project)
                .arg(&act_codes.ptr)
                .arg(&act_scales.ptr)
                .arg(&w_codes.ptr)
                .arg(&w_scales.ptr)
                .arg(&out.0.ptr)
                .arg(&i32::try_from(rows)?)
                .arg(&i32::try_from(groups)?)
                .arg(&i32::try_from(rank)?)
                .arg(&i32::try_from(width)?)
                .launch(LaunchConfig {
                    grid_dim: (u32::try_from(warps.div_ceil(8))?, 1, 1),
                    block_dim: (LWA_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cuda::Device;
    use crate::backend::cuda::moe::testing;
    use crate::fp8::decode_e4m3;
    use anyhow::ensure;
    use half::bf16;
    use std::mem::size_of;

    const HEADS: usize = 64;
    const HEAD_DIM: usize = 512;
    const WINDOW: usize = 128;
    const ROPE_TAIL: usize = 64;

    /// BF16 ULP distance of two outputs; 1 is the accepted BF16 rounding step.
    const OUTPUT_ULP_TOL: u32 = 1;
    /// FP32-vs-FP64 output difference for |q| <= 1 and |k| <= 0.5: the 512-term
    /// dot over 12 rounding levels gives 12*2^-24*256 = 1.8e-4, times the
    /// 512^-0.5 score scale 8.1e-6, times three softmax-weight terms 1.2e-5,
    /// plus the 128-term weighted sum 3.8e-6: 1.6e-5 < 3e-5.
    const OUTPUT_ABS_TOL: f64 = 3e-5;

    /// Golden-ratio stride: the phase sequence never repeats.
    fn pattern(seed: usize, phase: f64) -> f64 {
        (seed as f64 * 0.6180339887498949 + phase).sin() * 0.75
    }

    /// E4M3 codes of the fixture, including the subnormal range.
    const CODES: [u8; 23] = [
        0x00, 0x01, 0x07, 0x08, 0x0c, 0x38, 0x3c, 0x44, 0x7e, 0x7d, 0xfe, 0x80, 0x2e, 0x50, 0x68,
        0x71, 0x04, 0x10, 0x20, 0x30, 0x48, 0x58, 0x6c,
    ];

    /// E8M0FN scale: the value is 2^(code - 127); 0xff is NaN.
    fn e8m0_value(code: u8) -> f64 {
        if code == 0xff {
            return f64::NAN;
        }
        let exponent = i64::from(code) - 127;
        f64::from_bits(((exponent + 1023) as u64) << 52)
    }

    /// Distance in representable BF16 values between two outputs.
    fn bf16_ulp_distance(a: bf16, b: bf16) -> u32 {
        fn key(value: bf16) -> u32 {
            let bits = u32::from(value.to_bits());
            if bits & 0x8000 != 0 {
                (!bits) & 0xffff
            } else {
                bits | 0x8000
            }
        }
        key(a).abs_diff(key(b))
    }

    /// One call of the op.
    #[derive(Clone, Debug)]
    struct Fixture {
        head_dim: usize,
        nblocks: usize,
        q: Vec<bf16>,
        codes: Vec<u8>,
        scales: Vec<u8>,
        slot_ids: Vec<i32>,
        sink: Vec<f32>,
        inv_rotary: Vec<f32>,
    }

    impl Fixture {
        /// Dequantized ring value of one code.
        fn value(&self, row: usize, dim: usize) -> f64 {
            let code = self.codes[row * self.head_dim + dim];
            let scale = self.scales[row * self.nblocks + dim / 32];
            f64::from(decode_e4m3(code)) * e8m0_value(scale)
        }
    }

    /// Fixture whose strides are coprime with the shape strides in play.
    fn fixture(heads: usize, head_dim: usize, window: usize, slots: usize, seed: usize) -> Fixture {
        let nblocks = head_dim / 32;
        let q: Vec<bf16> = (0..heads * head_dim)
            .map(|i| bf16::from_f32(pattern(seed + i, 0.11) as f32))
            .collect();
        let codes: Vec<u8> = (0..window * head_dim)
            .map(|i| CODES[(i / head_dim * 29 + i % head_dim * 7 + seed) % CODES.len()])
            .collect();
        let scales: Vec<u8> = (0..window * nblocks)
            .map(|i| 115 + ((i / nblocks * 5 + i % nblocks * 7 + seed) % 3) as u8)
            .collect();
        // The ring of a decode step at position 300 holds 128 valid slots; the
        // visible order is oldest first (model definition, get_window_topk_idxs).
        let oldest = 300 % window + 1;
        let slot_ids: Vec<i32> = (0..slots).map(|i| ((oldest + i) % window) as i32).collect();
        let sink: Vec<f32> = (0..heads)
            .map(|h| (pattern(seed + h * 13, 2.7) * 2.0) as f32)
            .collect();
        let inv_rotary: Vec<f32> = (0..ROPE_TAIL)
            .map(|i| {
                let angle = -((i / 2) as f64) * 0.37 - 0.11;
                if i % 2 == 0 {
                    angle.cos() as f32
                } else {
                    angle.sin() as f32
                }
            })
            .collect();
        let fixture = Fixture {
            head_dim,
            nblocks,
            q,
            codes,
            scales,
            slot_ids,
            sink,
            inv_rotary,
        };
        // The frozen output tolerance assumes these magnitudes.
        for slot in 0..window {
            for dim in 0..head_dim {
                let value = fixture.value(slot, dim);
                assert!(
                    value.abs() <= 0.5,
                    "latent window attention fixture value {value} exceeds 0.5"
                );
            }
        }
        fixture
    }

    /// FP64 reference of the latent window attention.
    struct Oracle {
        heads: usize,
        head_dim: usize,
        slots: usize,
        rope_tail: usize,
        scale: f64,
    }

    impl Oracle {
        fn new(heads: usize, head_dim: usize, slots: usize, rope_tail: usize) -> Self {
            Self {
                heads,
                head_dim,
                slots,
                rope_tail,
                scale: (head_dim as f64).powf(-0.5),
            }
        }

        /// FP64 output before the BF16 rounding.
        fn forward_f64(&self, fx: &Fixture) -> Vec<f64> {
            let mut out = vec![0.0f64; self.heads * self.head_dim];
            for head in 0..self.heads {
                let q: Vec<f64> = (0..self.head_dim)
                    .map(|d| f64::from(fx.q[head * self.head_dim + d].to_f32()))
                    .collect();
                let mut scores = Vec::with_capacity(self.slots);
                for s in 0..self.slots {
                    let row = fx.slot_ids[s] as usize;
                    let mut dot = 0.0f64;
                    for (d, qv) in q.iter().enumerate() {
                        dot += qv * fx.value(row, d);
                    }
                    scores.push(dot * self.scale);
                }
                let sink = f64::from(fx.sink[head]);
                let max = scores.iter().copied().fold(sink, f64::max);
                let mut denom = (sink - max).exp();
                for score in scores.iter().take(self.slots) {
                    denom += (score - max).exp();
                }
                let mut acc = vec![0.0f64; self.head_dim];
                for (s, score) in scores.iter().take(self.slots).enumerate() {
                    let row = fx.slot_ids[s] as usize;
                    let weight = (score - max).exp() / denom;
                    for (d, a) in acc.iter_mut().enumerate() {
                        *a += weight * fx.value(row, d);
                    }
                }
                let tail = self.head_dim - self.rope_tail;
                for i in 0..self.rope_tail / 2 {
                    let c = f64::from(fx.inv_rotary[2 * i]);
                    let sn = f64::from(fx.inv_rotary[2 * i + 1]);
                    let a = acc[tail + 2 * i];
                    let b = acc[tail + 2 * i + 1];
                    acc[tail + 2 * i] = a * c - b * sn;
                    acc[tail + 2 * i + 1] = a * sn + b * c;
                }
                out[head * self.head_dim..(head + 1) * self.head_dim].copy_from_slice(&acc);
            }
            out
        }

        /// BF16 output of the op.
        fn forward(&self, fx: &Fixture) -> Vec<bf16> {
            self.forward_f64(fx)
                .iter()
                .map(|value| bf16::from_f32(*value as f32))
                .collect()
        }
    }

    /// Tensor view over host-side storage, for the validation test.
    fn fake<T: Copy, const R: usize>(shape: [u32; R], ptr: u64) -> Tensor<T, R> {
        let (stride, count) = dense_layout(shape).expect("dense layout");
        unsafe {
            Tensor::new(ptr, shape, stride, count * size_of::<T>() as u64).expect("fake tensor")
        }
    }

    #[test]
    fn lwa_codec_decodes_e4m3_times_e8m0() {
        // E4M3 value = 2^(e-7)*(1+m/8) for e > 0, m/512 for e == 0.
        let cases = [
            (0x00u8, 127u8, 0.0f64),
            (0x01, 127, 1.0 / 512.0),
            (0x07, 127, 7.0 / 512.0),
            (0x04, 127, 4.0 / 512.0),
            (0x08, 127, 1.0 / 64.0),
            (0x0c, 127, 1.5 / 64.0),
            (0x38, 127, 1.0),
            (0x3c, 127, 1.5),
            (0x44, 127, 3.0),
            (0x7e, 127, 448.0),
            (0x7d, 127, 416.0),
            (0xfe, 127, -448.0),
            (0x80, 127, 0.0),
            (0x30, 127, 0.5),
            (0x38, 117, 1.0 / 1024.0),
            (0x7e, 117, 448.0 / 1024.0),
            (0x01, 137, 1024.0 / 512.0),
            (0x38, 137, 1024.0),
        ];
        for (code, scale, want) in cases {
            let got = f64::from(decode_e4m3(code)) * e8m0_value(scale);
            assert_eq!(
                got, want,
                "latent window attention codec code {code:#04x} scale {scale} gave {got} != {want}"
            );
        }
        // The E8M0 scale is an exact power of two.
        assert_eq!(e8m0_value(127), 1.0);
        assert_eq!(e8m0_value(117), 2.0f64.powi(-10));
        assert_eq!(e8m0_value(137), 1024.0);
        assert!(e8m0_value(0xff).is_nan());
    }

    #[test]
    fn lwa_validation_names_every_failed_extent() {
        let q = fake::<bf16, 2>([HEADS as u32, HEAD_DIM as u32], 0x1000);
        let codes = fake::<u8, 2>([WINDOW as u32, HEAD_DIM as u32], 0x2000);
        let scales = fake::<u8, 2>([WINDOW as u32, (HEAD_DIM / 32) as u32], 0x3000);
        let ids = fake::<i32, 1>([64], 0x4000);
        let sink = fake::<f32, 1>([HEADS as u32], 0x5000);
        let rotary = fake::<f32, 1>([ROPE_TAIL as u32], 0x6000);
        let out = fake::<bf16, 2>([HEADS as u32, HEAD_DIM as u32], 0x7000);
        let inputs = LwaInputs {
            q,
            k_codes: codes,
            k_scales: scales,
            slot_ids: ids,
            sink,
            inv_rotary: rotary,
        };
        validate(inputs, &out).expect("valid latent window attention extents");
        let check = |error: Result<LwaDims>, expected: &str| {
            let error = error
                .expect_err("latent window attention validation accepted a bad extent")
                .to_string();
            assert!(
                error.contains(expected),
                "latent window attention validation error {error:?} does not name {expected}"
            );
        };
        let mut bad = inputs;
        bad.k_codes = fake::<u8, 2>([WINDOW as u32, (HEAD_DIM / 32) as u32], 0x2000);
        check(validate(bad, &out), "k_codes shape");
        let mut bad = inputs;
        bad.k_scales = fake::<u8, 2>([WINDOW as u32, (HEAD_DIM / 64) as u32], 0x3000);
        check(validate(bad, &out), "k_scales shape");
        let mut bad = inputs;
        bad.slot_ids = fake::<i32, 1>([WINDOW as u32 + 1], 0x4000);
        check(validate(bad, &out), "slots");
        let mut bad = inputs;
        bad.sink = fake::<f32, 1>([HEADS as u32 + 1], 0x5000);
        check(validate(bad, &out), "sink len");
        let mut bad = inputs;
        bad.inv_rotary = fake::<f32, 1>([ROPE_TAIL as u32 + 1], 0x6000);
        check(validate(bad, &out), "rope_tail");
        let bad_out = fake::<bf16, 2>([HEADS as u32, HEAD_DIM as u32 + 32], 0x7000);
        check(validate(inputs, &bad_out), "out shape");
    }

    #[test]
    fn lwa_ring_write_validation_names_every_failed_extent() {
        let kv = fake::<bf16, 2>([3, HEAD_DIM as u32], 0x1000);
        let codes = fake::<u8, 2>([WINDOW as u32, HEAD_DIM as u32], 0x2000);
        let scales = fake::<u8, 2>([WINDOW as u32, (HEAD_DIM / 32) as u32], 0x3000);
        validate_ring_write(&kv, 131, &codes, &scales).expect("valid latent ring write extents");
        let check = |error: Result<()>, expected: &str| {
            let error = error
                .expect_err("latent ring write validation accepted a bad extent")
                .to_string();
            assert!(
                error.contains(expected),
                "latent ring write validation error {error:?} does not name {expected}"
            );
        };
        check(
            validate_ring_write(
                &fake::<bf16, 2>([3, HEAD_DIM as u32 - 16], 0x1000),
                131,
                &codes,
                &scales,
            ),
            "must be a multiple of 32",
        );
        check(
            validate_ring_write(
                &kv,
                131,
                &fake::<u8, 2>([WINDOW as u32, (HEAD_DIM / 32) as u32], 0x2000),
                &scales,
            ),
            "codes shape",
        );
        check(
            validate_ring_write(
                &kv,
                131,
                &codes,
                &fake::<u8, 2>([WINDOW as u32, (HEAD_DIM / 64) as u32], 0x3000),
            ),
            "scales shape",
        );
        check(validate_ring_write(&kv, -1, &codes, &scales), "position");
        validate_ring_write(
            &fake::<bf16, 2>([WINDOW as u32, HEAD_DIM as u32], 0x100000),
            131,
            &codes,
            &scales,
        )
        .expect("a full-window latent ring write is valid");
        check(
            validate_ring_write(
                &fake::<bf16, 2>([WINDOW as u32 + 1, HEAD_DIM as u32], 0x100000),
                131,
                &codes,
                &scales,
            ),
            "exceed the 128-slot window",
        );
    }

    #[test]
    fn lwa_group_project_validation_names_every_failed_extent() {
        const GROUPS: usize = 8;
        const RANK: usize = 1024;
        const WIDTH: usize = 4096;
        let act_codes = fake::<u8, 3>([3, GROUPS as u32, WIDTH as u32], 0x100000);
        let act_scales = fake::<u8, 3>([3, GROUPS as u32, (WIDTH / 32) as u32], 0x200000);
        let w_codes = fake::<u8, 2>([(GROUPS * RANK) as u32, WIDTH as u32], 0x300000);
        let w_scales = fake::<u8, 2>([(GROUPS * RANK / 32) as u32, (WIDTH / 32) as u32], 0x400000);
        let out = fake::<bf16, 2>([3, (GROUPS * RANK) as u32], 0x8000000);
        validate_group_project(&act_codes, &act_scales, &w_codes, &w_scales, &out)
            .expect("valid latent group project extents");
        let check = |error: Result<()>, expected: &str| {
            let error = error
                .expect_err("latent group project validation accepted a bad extent")
                .to_string();
            assert!(
                error.contains(expected),
                "latent group project validation error {error:?} does not name {expected}"
            );
        };
        check(
            validate_group_project(
                &act_codes,
                &act_scales,
                &fake::<u8, 2>([(GROUPS * RANK + 8) as u32, WIDTH as u32], 0x300000),
                &fake::<u8, 2>(
                    [((GROUPS * RANK + 8) / 32) as u32, (WIDTH / 32) as u32],
                    0x400000,
                ),
                &fake::<bf16, 2>([3, (GROUPS * RANK + 8) as u32], 0x8000000),
            ),
            "rank",
        );
        check(
            validate_group_project(
                &act_codes,
                &fake::<u8, 3>([3, GROUPS as u32, (WIDTH / 64) as u32], 0x200000),
                &w_codes,
                &w_scales,
                &out,
            ),
            "act scales shape",
        );
        check(
            validate_group_project(
                &act_codes,
                &act_scales,
                &fake::<u8, 2>([(GROUPS * RANK) as u32, WIDTH as u32 + 32], 0x300000),
                &w_scales,
                &out,
            ),
            "weight codes width",
        );
        check(
            validate_group_project(
                &act_codes,
                &act_scales,
                &w_codes,
                &fake::<u8, 2>(
                    [(GROUPS * RANK / 32) as u32 + 1, (WIDTH / 32) as u32],
                    0x400000,
                ),
                &out,
            ),
            "weight scales shape",
        );
        check(
            validate_group_project(
                &act_codes,
                &act_scales,
                &w_codes,
                &w_scales,
                &fake::<bf16, 2>([3, (GROUPS * RANK - 32) as u32], 0x8000000),
            ),
            "out shape",
        );
    }

    #[test]
    fn lwa_sink_enters_the_softmax_denominator() {
        let fx = fixture(4, 64, 16, 8, 3);
        let oracle = Oracle::new(4, 64, 8, 16);
        let with_sink = oracle.forward(&fx);
        let mut zero_sink = fx.clone();
        zero_sink.sink = vec![0.0; fx.sink.len()];
        let without_sink = oracle.forward(&zero_sink);
        assert_ne!(
            with_sink, without_sink,
            "latent window attention sink control escaped"
        );
        // The sink is a denominator term: the output of every head stays
        // proportional to the output without the sink.
        let raw_with = oracle.forward_f64(&fx);
        let raw_without = oracle.forward_f64(&zero_sink);
        for head in 0..4 {
            let start = head * 64;
            let peak = (start..start + 64)
                .max_by(|a, b| raw_without[*a].abs().total_cmp(&raw_without[*b].abs()))
                .expect("non-empty head");
            let factor = raw_with[peak] / raw_without[peak];
            assert!(
                factor.is_finite() && factor != 0.0,
                "latent window attention sink factor {factor} is degenerate"
            );
            for d in start..start + 64 {
                let want = factor * raw_without[d];
                assert!(
                    (raw_with[d] - want).abs() <= 1e-9 * raw_without[peak].abs().max(1e-6),
                    "latent window attention sink is not a denominator factor at dim {d}: {} != {want}",
                    raw_with[d]
                );
            }
        }
    }

    #[test]
    fn lwa_wrong_slot_exclusion_changes_the_output() {
        let fx = fixture(4, 64, 16, 8, 5);
        let oracle = Oracle::new(4, 64, 8, 16);
        let straight = oracle.forward(&fx);
        let mut excluded = fx.clone();
        let last = excluded.slot_ids.len() - 1;
        excluded.slot_ids[last] = excluded.slot_ids[0];
        let wrong = oracle.forward(&excluded);
        assert_ne!(
            straight, wrong,
            "latent window attention slot exclusion control escaped"
        );
    }

    #[test]
    fn lwa_wrong_scale_order_changes_the_output() {
        let fx = fixture(4, 64, 16, 8, 9);
        let oracle = Oracle::new(4, 64, 8, 16);
        let straight = oracle.forward(&fx);
        let mut shifted = fx.clone();
        for row in 0..16 {
            for block in 0..shifted.nblocks {
                let from = row * shifted.nblocks + (block + 1) % shifted.nblocks;
                let to = row * shifted.nblocks + block;
                shifted.scales[to] = fx.scales[from];
            }
        }
        let wrong = oracle.forward(&shifted);
        assert_ne!(
            straight, wrong,
            "latent window attention scale order control escaped"
        );
    }

    #[test]
    fn lwa_inverse_rope_changes_only_the_output_tail() {
        let fx = fixture(4, 64, 16, 8, 13);
        let oracle = Oracle::new(4, 64, 8, 16);
        let rotated = oracle.forward(&fx);
        let mut identity = fx.clone();
        identity.inv_rotary = (0..ROPE_TAIL)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let straight = oracle.forward(&identity);
        assert_ne!(
            rotated, straight,
            "latent window attention inverse rope control escaped"
        );
        let tail = 64 - ROPE_TAIL;
        for head in 0..4 {
            for d in 0..tail {
                let index = head * 64 + d;
                assert_eq!(
                    rotated[index].to_bits(),
                    straight[index].to_bits(),
                    "latent window attention inverse rope changed dim {d} of head {head} outside the tail"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_Q38_GPU_WINDOW=granted"]
    fn latent_window_attention_matches_oracle() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let device = Device::new(0)?;
        let s = &device.stream;
        let ops = LatentWindowAttention::new(&device.ctx)?;
        for &slots in &[1usize, 64, 128] {
            let fx = fixture(HEADS, HEAD_DIM, WINDOW, slots, 7);
            let oracle = Oracle::new(HEADS, HEAD_DIM, slots, ROPE_TAIL);
            let expected = oracle.forward(&fx);
            let qb = testing::upload(&device.upload, fx.q.clone())?;
            let cb = testing::upload(&device.upload, fx.codes.clone())?;
            let sb = testing::upload(&device.upload, fx.scales.clone())?;
            let idb = testing::upload(&device.upload, fx.slot_ids.clone())?;
            let skb = testing::upload(&device.upload, fx.sink.clone())?;
            let rb = testing::upload(&device.upload, fx.inv_rotary.clone())?;
            // Discriminating control: one visible slot replaced by the next ring
            // slot must change the output.
            let mut wrong_ids = fx.slot_ids.clone();
            let last = wrong_ids.len() - 1;
            wrong_ids[last] = (wrong_ids[last] + 1) % WINDOW as i32;
            let widb = testing::upload(&device.upload, wrong_ids)?;
            device.finish_upload()?;
            let out = testing::zeros::<bf16>(s, HEADS * HEAD_DIM)?;
            let wrong_out = testing::zeros::<bf16>(s, HEADS * HEAD_DIM)?;
            let q = testing::view(
                &qb,
                s,
                [HEADS as u32, HEAD_DIM as u32],
                [HEAD_DIM as u64, 1],
            )?;
            let codes = testing::view(
                &cb,
                s,
                [WINDOW as u32, HEAD_DIM as u32],
                [HEAD_DIM as u64, 1],
            )?;
            let scales = testing::view(
                &sb,
                s,
                [WINDOW as u32, (HEAD_DIM / 32) as u32],
                [HEAD_DIM as u64 / 32, 1],
            )?;
            let ids = testing::view(&idb, s, [slots as u32], [1])?;
            let wrong_ids = testing::view(&widb, s, [slots as u32], [1])?;
            let sink = testing::view(&skb, s, [HEADS as u32], [1])?;
            let rotary = testing::view(&rb, s, [ROPE_TAIL as u32], [1])?;
            let mut out_view = testing::mut_view(
                &out,
                s,
                [HEADS as u32, HEAD_DIM as u32],
                [HEAD_DIM as u64, 1],
            )?;
            let mut wrong_view = testing::mut_view(
                &wrong_out,
                s,
                [HEADS as u32, HEAD_DIM as u32],
                [HEAD_DIM as u64, 1],
            )?;
            let inputs = LwaInputs {
                q,
                k_codes: codes,
                k_scales: scales,
                slot_ids: ids,
                sink,
                inv_rotary: rotary,
            };
            ops.run(s, inputs, &mut out_view, 0)?;
            let wrong_inputs = LwaInputs {
                slot_ids: wrong_ids,
                ..inputs
            };
            ops.run(s, wrong_inputs, &mut wrong_view, 0)?;
            s.synchronize()?;
            let host = s.clone_dtoh(&out)?;
            let wrong_host = s.clone_dtoh(&wrong_out)?;
            for head in 0..HEADS {
                for d in 0..HEAD_DIM {
                    let got = host[head * HEAD_DIM + d];
                    let want = expected[head * HEAD_DIM + d];
                    let distance = bf16_ulp_distance(got, want);
                    let abs = (f64::from(got.to_f32()) - f64::from(want.to_f32())).abs();
                    ensure!(
                        distance <= OUTPUT_ULP_TOL || abs <= OUTPUT_ABS_TOL,
                        "latent window attention head {head} dim {d} bf16 ulp distance {distance} abs {abs}"
                    );
                }
            }
            let mut caught = 0usize;
            for head in 0..HEADS {
                for d in 0..HEAD_DIM {
                    let got = host[head * HEAD_DIM + d];
                    let wrong = wrong_host[head * HEAD_DIM + d];
                    let distance = bf16_ulp_distance(got, wrong);
                    let abs = (f64::from(got.to_f32()) - f64::from(wrong.to_f32())).abs();
                    if distance > OUTPUT_ULP_TOL && abs > OUTPUT_ABS_TOL {
                        caught += 1;
                    }
                }
            }
            ensure!(
                caught > 0,
                "latent window attention planted wrong slot exclusion was not caught (slots {slots})"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "latent_attention_certificate.rs"]
pub(super) mod certificate;

#[cfg(test)]
mod indexed_tests {
    use super::*;
    use crate::backend::cuda::contracts::{SeqRef, view};
    use anyhow::Context;
    use cudarc::driver::DevicePtr;

    #[derive(serde::Deserialize)]
    struct Cell {
        layer: usize,
        position: usize,
        ratio: u32,
        capacity: u32,
        query: Vec<u16>,
        raw_codes: Vec<u8>,
        raw_scales: Vec<u8>,
        compressed_codes: Vec<u8>,
        compressed_scales: Vec<u8>,
        selected: Vec<i32>,
        sink: Vec<f32>,
        output: Vec<u16>,
        spread: f64,
        bound: f64,
        reference_stores: String,
    }
    #[derive(serde::Deserialize)]
    struct Fixture {
        reference_only: bool,
        candidate_seen: bool,
        cells: Vec<Cell>,
    }

    struct Outcome {
        raw: usize,
        exempt: usize,
        defects: usize,
        maximum: f64,
        score_control: bool,
    }

    fn cell(
        ctx: &Arc<CudaContext>,
        record: &Cell,
        graph: bool,
        control: bool,
        publisher: &[f32],
    ) -> Result<Outcome> {
        let s = ctx.new_stream()?;
        let q = s.clone_htod(
            &record
                .query
                .iter()
                .copied()
                .map(bf16::from_bits)
                .collect::<Vec<_>>(),
        )?;
        let raw = s.clone_htod(&record.raw_codes)?;
        let raw_scales = s.clone_htod(&record.raw_scales)?;
        let packed = s.clone_htod(&record.compressed_codes)?;
        let packed_scales = s.clone_htod(&record.compressed_scales)?;
        let mut ids = record.selected.clone();
        if control && !ids.is_empty() {
            ids.pop();
        }
        ids.resize(512, -1);
        let selected = s.clone_htod(&ids)?;
        let count = s.clone_htod(&[((record.position + 1) / record.ratio as usize) as u32])?;
        let sink = s.clone_htod(&record.sink)?;
        let inverse = s.clone_htod(&[1.0f32, 0.0].repeat(32))?;
        let slots = s.alloc_zeros::<i32>(128)?;
        let meta = s.clone_htod(&[
            record.position as i32,
            1,
            record.position as i32 + 1,
            (record.capacity as i32 * record.ratio as i32).max(record.position as i32 + 1),
            1,
            0,
            0,
            0,
        ])?;
        let out = s.clone_htod(&vec![bf16::from_f32(127.0); 64 * 512 + 16])?;
        let plan = IndexedAttentionPlan::new(
            ctx,
            IndexedInputs {
                window: LwaInputs {
                    q: view(&q, &s, 0, [64, 512], [512, 1])?,
                    k_codes: view(&raw, &s, 0, [128, 512], [512, 1])?,
                    k_scales: view(&raw_scales, &s, 0, [128, 16], [16, 1])?,
                    slot_ids: view(&slots, &s, 0, [128], [1])?,
                    sink: view(&sink, &s, 0, [64], [1])?,
                    inv_rotary: view(&inverse, &s, 0, [64], [1])?,
                },
                packed: view(&packed, &s, 0, [record.capacity, 256], [256, 1])?,
                scales: view(&packed_scales, &s, 0, [record.capacity, 32], [32, 1])?,
                selected: view(&selected, &s, 0, [1, 512], [512, 1])?,
                committed: view(&count, &s, 0, [1], [1])?,
                sequence: SeqRef {
                    ptr: meta.device_ptr(&s).0,
                },
                ratio: record.ratio,
            },
            view(&out, &s, 0, [64, 512], [512, 1])?,
            0,
        )?;
        if graph {
            s.begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )?;
            plan.run(&s)?;
            let captured = s
                .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))?
                .context("indexed graph missing")?;
            captured.launch()?;
        } else {
            plan.run(&s)?;
        }
        let got = s.clone_dtoh(&out)?;
        ensure!(
            got[64 * 512..].iter().all(|v| *v == bf16::from_f32(127.0)),
            "indexed attention output guard overwritten"
        );
        ensure!(
            record.output.len() == 64 * 512 && record.bound == 3.0 * record.spread,
            "indexed reference extent/freeze formula"
        );
        let mut defects = 0;
        let mut maximum = 0.0f64;
        for (actual, reference) in got.iter().zip(&record.output) {
            ensure!(actual.is_finite(), "indexed attention nonfinite output");
            let difference =
                f64::from((actual.to_f32() - bf16::from_bits(*reference).to_f32()).abs());
            maximum = maximum.max(difference);
            defects += usize::from(difference > record.bound);
        }
        let trace = s.clone_htod(&vec![f32::NAN; certificate::TRACE_LEN])?;
        let kernel = cubin::module(ctx, "latent_window_attention")?
            .load_function("latent_indexed_attention_trace")?;
        let v = plan.inputs;
        unsafe {
            s.launch_builder(&kernel)
                .arg(&v.window.q.ptr)
                .arg(&v.window.k_codes.ptr)
                .arg(&v.window.k_scales.ptr)
                .arg(&v.packed.ptr)
                .arg(&v.scales.ptr)
                .arg(&v.selected.ptr)
                .arg(&v.committed.ptr)
                .arg(&v.window.sink.ptr)
                .arg(&v.window.inv_rotary.ptr)
                .arg(&v.sequence.ptr)
                .arg(&plan.output.ptr)
                .arg(&plan.shape[0])
                .arg(&plan.shape[1])
                .arg(&plan.shape[2])
                .arg(&plan.shape[3])
                .arg(&plan.shape[4])
                .arg(&plan.shape[5])
                .arg(&plan.shape[6])
                .arg(&plan.scale)
                .arg(&trace.device_ptr(&s).0)
                .launch(plan.grid)?;
        }
        let traced = s.clone_dtoh(&out)?;
        ensure!(
            got.iter()
                .zip(&traced)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "indexed instrumentation changed output bits"
        );
        let mut stores = s.clone_dtoh(&trace)?;
        let prove = |candidate: &[f32]| {
            certificate::prove(certificate::Inputs {
                layer: record.layer,
                position: record.position,
                query: [&record.query, &record.query],
                sink: &record.sink,
                candidate,
                publisher,
                output: &got[..64 * 512],
                expected: &record.output,
            })
        };
        let proof = prove(&stores)?;
        let chain_defects = proof.store_defects + proof.operand_defects;
        let exempt = if chain_defects == 0 { defects } else { 0 };
        let mut score_control = false;
        if record.layer == 2 && record.position == 1 && !graph && !control {
            stores[19 * 64] += 1.0;
            let mutant = prove(&stores)?;
            score_control = mutant.store_defects > 0;
            println!(
                "MIXED_ATTENTION_SCORE_CONTROL planted_qk_plus_one detected={score_control} store_defects={}",
                mutant.store_defects
            );
            ensure!(score_control, "indexed planted score control escaped");
        }
        ensure!(
            proof.output_differences >= defects,
            "indexed certificate output count mismatch"
        );
        Ok(Outcome {
            raw: defects,
            exempt,
            defects: if chain_defects == 0 {
                0
            } else {
                defects.max(chain_defects)
            },
            maximum,
            score_control,
        })
    }

    #[test]
    #[ignore]
    fn indexed_attention_publisher_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_DS41_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let path = std::env::var_os("D1_MIXED_REFERENCE").context("D1_MIXED_REFERENCE required")?;
        let path = std::path::PathBuf::from(path);
        let reference: Fixture = serde_json::from_slice(&std::fs::read(&path)?)?;
        ensure!(
            reference.reference_only && !reference.candidate_seen && !reference.cells.is_empty(),
            "mixed floor not reference-only"
        );
        let ctx = CudaContext::new(0)?;
        let mut defects = 0;
        let mut controls = std::collections::BTreeSet::new();
        let mut cases = 0;
        let mut raw = 0;
        let mut exempt = 0;
        let mut score_controls = 0;
        // 2026-10-29: rule-B chain certificates use actual stores; the reference floor is unchanged.
        for record in &reference.cells {
            let bytes = std::fs::read(
                path.parent()
                    .context("mixed fixture parent")?
                    .join(&record.reference_stores),
            )?;
            ensure!(
                bytes.len() == certificate::TRACE_LEN * 4,
                "publisher store record extent"
            );
            let publisher: Vec<_> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            for graph in [false, true] {
                let result = cell(&ctx, record, graph, false, &publisher)?;
                println!(
                    "MIXED_ATTENTION layer={} position={} graph={graph} max={:e} limit={:e} raw={} exempt={} defects={}",
                    record.layer,
                    record.position,
                    result.maximum,
                    record.bound,
                    result.raw,
                    result.exempt,
                    result.defects
                );
                raw += result.raw;
                exempt += result.exempt;
                defects += result.defects;
                score_controls += usize::from(result.score_control);
                cases += 1;
            }
            if !record.selected.is_empty() && !controls.contains(&record.layer) {
                let result = cell(&ctx, record, false, true, &publisher)?;
                let bad = result.defects;
                let max = result.maximum;
                println!(
                    "MIXED_ATTENTION_CONTROL layer={} position={} dropped_selected_position max={max:e} detected={}",
                    record.layer,
                    record.position,
                    bad > 0
                );
                if bad > 0 {
                    controls.insert(record.layer);
                }
            }
        }
        let owners: std::collections::BTreeSet<_> =
            reference.cells.iter().map(|c| c.layer).collect();
        println!(
            "MIXED_ATTENTION_RESULT cases={cases} raw={raw} exempt={exempt} defects={defects} score_controls={score_controls} controls={} owners={}",
            controls.len(),
            owners.len()
        );
        ensure!(
            controls == owners && score_controls == 1,
            "mixed attention wrong-selection control escaped"
        );
        ensure!(defects == 0, "mixed attention candidate DEFECT");
        Ok(())
    }
}
