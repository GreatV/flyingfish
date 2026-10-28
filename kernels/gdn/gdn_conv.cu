// GDN conv: depthwise causal conv1d, k=4, groups == channels, no bias, over
// GDN_CONV_DIM = 8192 or 10240 channels, with the exact shared SiLU applied in the same thread. The
// conv result is cast to BF16 and back before silu(), and the silu result is cast to
// BF16 for conv_out.
//
// history[] is read-only here and written by gdn_conv_hist_update. Taps x[r-3..r] are
// read directly, one load per tap; conv weights are per-thread registers.
#include "../q38_contract.cuh"
#include "../gpu_compat.cuh"
#include <stdint.h>

#ifndef GDN_HEADS
#define GDN_HEADS 48
#endif
#define GDN_CONV_DIM ((2 * 16 + GDN_HEADS) * 128)
#define GDN_CONV_CHAN 256
#define GDN_CONV_ROWS 32

// One thread per channel; the block's 256 threads cover 256 contiguous channels. Each
// thread walks a strip of rows.
//
// launch: grid (conv_dim/GDN_CONV_CHAN, ceil(valid_rows/RowTile)), block 256.
template<bool Normalize, int RowTile = GDN_CONV_ROWS>
__device__ __forceinline__ void gdn_conv_impl(
        const __nv_bfloat16* __restrict__ qkv,     // [M, conv_dim] pre-conv, strides [conv_dim,1]
        const __nv_bfloat16* __restrict__ conv_w,  // [conv_dim, 4],   strides [4,1]
        const float* __restrict__ history,         // [conv_dim, 3],   read-only here
        __nv_bfloat16* __restrict__ out,           // [M, conv_dim] BF16 conv result
        const Q38SequenceFields* __restrict__ meta,
        int conv_dim, float* q_fp32, float* k_fp32,
        __nv_bfloat16* q_bf16, __nv_bfloat16* k_bf16, float l2_eps) {
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

    const int row_base = blockIdx.y * RowTile;
    if (row_base >= valid_rows) return;

    float wgt[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) wgt[j] = __bfloat162float(conv_w[ch * 4 + j]);

    const int rows_here = min(RowTile, valid_rows - row_base);
    __shared__ float sums[2][4];
    for (int i = 0; i < rows_here; ++i) {
        // Local row: qkv and out are this call's [M, conv_dim] buffers, indexed 0-based
        // within the call; prefix rows are reachable only through history[].
        const int lr = row_base + i;
        float acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int sl = lr - 3 + j;               // LOCAL row of tap j
            // sl < 0 -> the row is in history, at index sl - (-3) = sl + 3
            const float x = (sl < 0)
                ? history[ch * 3 + (sl + 3)]
                : __bfloat162float(qkv[(size_t)sl * conv_dim + ch]);
            acc = fmaf(x, wgt[j], acc);
        }
        const __nv_bfloat16 vis = __float2bfloat16_rn(acc);   // the visible BF16 cast
        const __nv_bfloat16 value = __float2bfloat16_rn(silu(__bfloat162float(vis)));
        out[(size_t)lr * conv_dim + ch] = value;
        if constexpr (Normalize) {
            if (ch < 2 * 16 * 128) {
                const int lane = threadIdx.x & 31, warp = (threadIdx.x & 127) >> 5;
                const int group = threadIdx.x >> 7;
                const float source = __bfloat162float(value);
                float sum = source * source;
                #pragma unroll
                for (int delta = 16; delta > 0; delta >>= 1)
                    sum += __shfl_down_sync(FF_WARP_MASK, sum, delta);
                if (lane == 0) sums[group][warp] = sum;
                __syncthreads();
                if (warp == 0) {
                    sum = lane < 4 ? sums[group][lane] : 0.0f;
                    #pragma unroll
                    for (int delta = 16; delta > 0; delta >>= 1)
                        sum += __shfl_down_sync(FF_WARP_MASK, sum, delta);
                    if (lane == 0) sums[group][0] = rsqrtf(sum + l2_eps);
                }
                __syncthreads();
                float normalized = source * sums[group][0];
                const bool query = ch < 16 * 128;
                if (query) normalized *= 1.0f / sqrtf(128.0f);
                const size_t index = (size_t)lr * 16 * 128 + ch % (16 * 128);
                (query ? q_fp32 : k_fp32)[index] = normalized;
                (query ? q_bf16 : k_bf16)[index] = __float2bfloat16_rn(normalized);
                __syncthreads();
            }
        }
    }
}

extern "C" __global__ void gdn_conv(
        const __nv_bfloat16* __restrict__ qkv, const __nv_bfloat16* __restrict__ conv_w,
        const float* __restrict__ history, __nv_bfloat16* __restrict__ out,
        const Q38SequenceFields* __restrict__ meta, int conv_dim) {
    gdn_conv_impl<false>(qkv, conv_w, history, out, meta, conv_dim,
                         nullptr, nullptr, nullptr, nullptr, 0.0f);
}

extern "C" __global__ void gdn_conv_l2_compact(
        const __nv_bfloat16* __restrict__ qkv, const __nv_bfloat16* __restrict__ conv_w,
        const float* __restrict__ history, __nv_bfloat16* __restrict__ out,
        const Q38SequenceFields* __restrict__ meta, int conv_dim,
        float* __restrict__ q_fp32, float* __restrict__ k_fp32,
        __nv_bfloat16* __restrict__ q_bf16, __nv_bfloat16* __restrict__ k_bf16, float l2_eps) {
    if (conv_dim != 10240 || !aligned16(q_fp32) || !aligned16(k_fp32) ||
        !aligned16(q_bf16) || !aligned16(k_bf16)) { __trap(); return; }
    gdn_conv_impl<true>(qkv, conv_w, history, out, meta, conv_dim,
                        q_fp32, k_fp32, q_bf16, k_bf16, l2_eps);
}

extern "C" __global__ void gdn_conv_l2_compact_rows8(
        const __nv_bfloat16* __restrict__ qkv, const __nv_bfloat16* __restrict__ conv_w,
        const float* __restrict__ history, __nv_bfloat16* __restrict__ out,
        const Q38SequenceFields* __restrict__ meta, int conv_dim,
        float* __restrict__ q_fp32, float* __restrict__ k_fp32,
        __nv_bfloat16* __restrict__ q_bf16, __nv_bfloat16* __restrict__ k_bf16, float l2_eps) {
    if (conv_dim != 10240 || !aligned16(q_fp32) || !aligned16(k_fp32) ||
        !aligned16(q_bf16) || !aligned16(k_bf16)) { __trap(); return; }
    gdn_conv_impl<true, 8>(qkv, conv_w, history, out, meta, conv_dim,
                         q_fp32, k_fp32, q_bf16, k_bf16, l2_eps);
}

// New history = the last 3 rows of (old history ++ this call's valid rows). The sequence has
// 3 + valid_rows entries, so its last 3 are seq[valid_rows + j]; that also covers
// valid_rows < 3 (valid_rows==1 gives history[1], history[2], the one new row).
// Runs after gdn_conv. One thread per channel; every write is thread-local.
//
// launch: grid (conv_dim/GDN_CONV_CHAN), block 256.
extern "C" __global__ void gdn_conv_hist_update(
        const __nv_bfloat16* __restrict__ qkv,
        float* __restrict__ history,
        const Q38SequenceFields* __restrict__ meta,
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
        // qkv is this call's [M, conv_dim] buffer; its row is the local index idx-3.
        history[ch * 3 + j] = (idx < 3)
            ? history[ch * 3 + idx]
            : __bfloat162float(qkv[(size_t)(idx - 3) * conv_dim + ch]);
    }
}

extern "C" __global__ void gdn_conv_hist_snapshot(
        const __nv_bfloat16* __restrict__ qkv, float* __restrict__ history,
        const Q38SequenceFields* __restrict__ meta, int conv_dim, float* __restrict__ snapshots) {
    const int ch = blockIdx.x * GDN_CONV_CHAN + threadIdx.x;
    const int rows = meta->valid_rows;
    constexpr int row_limit = GDN_HEADS == 32 ? 16 : 5;
    if (conv_dim != GDN_CONV_DIM || blockDim.x != GDN_CONV_CHAN || blockDim.y != 1 ||
        blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1 || ch >= conv_dim ||
        !aligned16(qkv) || !aligned16(history) || !aligned16(snapshots) ||
        rows < 1 || rows > row_limit || meta->prefix < 0 || meta->causal != 1 ||
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
