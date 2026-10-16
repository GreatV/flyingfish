// Q38 ABI v1 section 6, GDN conv: depthwise causal conv1d, k=4, groups == channels, no
// bias, over CONV_DIM = 10240 channels, with the exact shared SiLU applied in the same
// thread. ABI section 5's BF16 rounding points are kept: the conv result is cast to BF16
// and back before silu(), and the silu result is cast to BF16 for conv_out.
//
// HISTORY is written by a SEPARATE kernel, gdn_conv_hist_update, not by this one. They used
// to be fused and that was a cross-block race: the block owning the last row overwrote
// history[] while blocks with blockIdx.y==0 were still reading it to initialise their window.
// For valid_rows > 32 the result was nondeterministic.
//
// TAPS are read directly, one load per tap, rather than carried in a shared ring. A ring
// needs rotation state and got the tap ORDER wrong after the first row: slot (i&3) held
// x[t+1] while the dot product still paired ring[j] with wgt[j]. Reading x[r-3..r] directly
// makes both the ordering and the row-block boundary mistakes impossible by construction.
//
// CONV WEIGHTS are per-thread registers, not shared memory. They were `__shared__ float
// wgt[4]` filled by threads 0..3 with the weights of THEIR OWN channels, then read by all
// 256 threads in the block -- so 252 of every 256 channels convolved with the wrong weights.
//
// CPU emulation of exactly this indexing lives in the P1-C gate
// (p1c/gdn_gate_cpu.cpp, section (f)); it caught all of the above without a GPU.
#include "../q38_contract.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define GDN_CONV_DIM  10240
#define GDN_CONV_CHAN 256
#define GDN_CONV_ROWS 32

struct GdnMeta {
    int prefix, valid_rows, key_count, capacity, causal, r20, r24, r28;
};

__device__ __forceinline__ bool aligned16(const void* p) {
    return p && (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}

// One thread per channel; the block's 256 threads cover 256 contiguous channels, so every
// global access along the channel axis is coalesced. Each thread walks a strip of rows.
//
// launch: grid (conv_dim/GDN_CONV_CHAN, ceil(valid_rows/GDN_CONV_ROWS)), block 256.
extern "C" __global__ void gdn_conv(
        const __nv_bfloat16* __restrict__ qkv,     // [M, conv_dim] pre-conv, strides [conv_dim,1]
        const __nv_bfloat16* __restrict__ conv_w,  // [conv_dim, 4],   strides [4,1]
        const float* __restrict__ history,         // [conv_dim, 3],   read-only here
        __nv_bfloat16* __restrict__ out,           // [M, conv_dim] BF16 conv result
        const GdnMeta* __restrict__ meta,
        int conv_dim) {
    const int ch = blockIdx.x * GDN_CONV_CHAN + threadIdx.x;

    if (conv_dim <= 0 || blockDim.x != GDN_CONV_CHAN || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.z != 1 || !aligned16(qkv) || !aligned16(conv_w) || !aligned16(history) ||
        !aligned16(out) || ch >= conv_dim) {
        __trap(); return;
    }
    const int prefix = meta->prefix, valid_rows = meta->valid_rows;
    if (valid_rows <= 0 || prefix < 0 || meta->causal != 1 ||
        meta->key_count != prefix + valid_rows || meta->capacity < meta->key_count) {
        __trap(); return;
    }

    const int row_base = blockIdx.y * GDN_CONV_ROWS;
    if (row_base >= valid_rows) return;

    // PER-THREAD weights. Each channel has its own four taps.
    float wgt[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) wgt[j] = __bfloat162float(conv_w[ch * 4 + j]);

    const int rows_here = min(GDN_CONV_ROWS, valid_rows - row_base);
    for (int i = 0; i < rows_here; ++i) {
        // LOCAL row this thread writes. qkv and out are this call's [M, conv_dim] buffers,
        // so every row index into them is 0-based within the call. prefix rows PRECEDE the
        // call and are reachable only through history[]; indexing these buffers by the
        // absolute row (prefix + ...) reads and writes out of bounds as soon as prefix != 0.
        // Every existing fixture has prefix == 0, where the two coincide and the defect
        // hides. Fixtures conv_pfx* carry prefix != 0 and a -DGDN_ABS_ROWS control build
        // restores the absolute form; that control must FAIL them.
        const int lr = row_base + i;
        float acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int sl = lr - 3 + j;               // LOCAL row of tap j
// Negative-control build; never defined by build.rs.
#ifdef GDN_ABS_ROWS
            // NEGATIVE CONTROL BUILD (never shipped): the pre-fix absolute-row indexing.
            // The point of the prefix gate is that this is a real kernel build, so the
            // control is the same source with absolute rows restored, not a perturbation of
            // the outputs. It must FAIL every conv_pfx* case and pass the prefix==0 ones,
            // which is what makes a clean conv run distinguishable from a broken one.
            const int s = prefix + sl;
            const float x = (s < prefix)
                ? history[ch * 3 + (s - prefix + 3)]
                : __bfloat162float(qkv[(size_t)s * conv_dim + ch]);
#else
            // sl < 0 -> the row is in history, at index sl - (-3) = sl + 3
            const float x = (sl < 0)
                ? history[ch * 3 + (sl + 3)]
                : __bfloat162float(qkv[(size_t)sl * conv_dim + ch]);
#endif
            acc = fmaf(x, wgt[j], acc);
        }
// Negative-control build; never defined by build.rs.
#ifdef GDN_ABS_ROWS
        const __nv_bfloat16 vis = __float2bfloat16_rn(acc);
        out[(size_t)(prefix + lr) * conv_dim + ch] = __float2bfloat16_rn(silu(__bfloat162float(vis)));
#else
        const __nv_bfloat16 vis = __float2bfloat16_rn(acc);   // the visible cast (ABI §5)
        out[(size_t)lr * conv_dim + ch] = __float2bfloat16_rn(silu(__bfloat162float(vis)));
#endif
    }
}

// New history = the last 3 rows of (old history ++ this call's valid rows). The sequence has
// 3 + valid_rows entries, so its last 3 are seq[valid_rows + j]; that also covers
// valid_rows < 3 (valid_rows==1 gives history[1], history[2], the one new row).
//
// SEPARATE KERNEL so it cannot race the readers in gdn_conv. One thread per channel; every
// write is thread-local, so there is no cross-block ordering to get wrong.
//
// launch: grid (conv_dim/GDN_CONV_CHAN), block 256.
extern "C" __global__ void gdn_conv_hist_update(
        const __nv_bfloat16* __restrict__ qkv,
        float* __restrict__ history,
        const GdnMeta* __restrict__ meta,
        int conv_dim) {
    const int ch = blockIdx.x * GDN_CONV_CHAN + threadIdx.x;
    if (conv_dim <= 0 || blockDim.x != GDN_CONV_CHAN || blockDim.y != 1 || blockDim.z != 1 ||
        !aligned16(qkv) || !aligned16(history) || ch >= conv_dim) {
        __trap(); return;
    }
    const int prefix = meta->prefix, valid_rows = meta->valid_rows;
    if (valid_rows <= 0 || prefix < 0 || meta->causal != 1 ||
        meta->key_count != prefix + valid_rows || meta->capacity < meta->key_count) {
        __trap(); return;
    }
    #pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int idx = valid_rows + j;             // index into (history ++ qkv)
        // qkv is this call's [M, conv_dim] buffer, so its row is the LOCAL index idx-3.
        // The pre-fix form added prefix to make an absolute row, which reads past the end
        // of the buffer whenever prefix != 0. prefix is not needed here at all: the new
        // history is the last 3 of (history ++ this call's rows), which is independent of
        // how many rows preceded the call.
// Negative-control build; never defined by build.rs.
#ifdef GDN_ABS_ROWS
        // NEGATIVE CONTROL BUILD (never shipped) -- the pre-fix absolute row.
        history[ch * 3 + j] = (idx < 3)
            ? history[ch * 3 + idx]
            : __bfloat162float(qkv[(size_t)(prefix + idx - 3) * conv_dim + ch]);
#else
        history[ch * 3 + j] = (idx < 3)
            ? history[ch * 3 + idx]
            : __bfloat162float(qkv[(size_t)(idx - 3) * conv_dim + ch]);
#endif
    }
}

extern "C" __global__ void gdn_conv_hist_snapshot(
        const __nv_bfloat16* __restrict__ qkv, float* __restrict__ history,
        const GdnMeta* __restrict__ meta, int conv_dim, float* __restrict__ snapshots) {
    const int ch = blockIdx.x * GDN_CONV_CHAN + threadIdx.x;
    const int rows = meta->valid_rows;
    if (conv_dim != GDN_CONV_DIM || blockDim.x != GDN_CONV_CHAN || blockDim.y != 1 ||
        blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1 || ch >= conv_dim ||
        !aligned16(qkv) || !aligned16(history) || !aligned16(snapshots) ||
        rows < 1 || rows > 4 || meta->prefix < 0 || meta->causal != 1 ||
        meta->key_count != meta->prefix + rows || meta->capacity < meta->key_count) {
        __trap(); return;
    }
    const float old[3] = { history[ch * 3], history[ch * 3 + 1], history[ch * 3 + 2] };
    for (int r = 0; r < rows; ++r) {
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int idx = r + 1 + j;
            const float x = idx < 3 ? old[idx] : __bfloat162float(qkv[(size_t)(idx - 3) * conv_dim + ch]);
            // The last row is not snapshotted; snapshot_row reads rows < live only.
            if (r + 1 < rows) snapshots[((size_t)r * conv_dim + ch) * 3 + j] = x;
            if (r == rows - 1) history[ch * 3 + j] = x;
        }
    }
}
