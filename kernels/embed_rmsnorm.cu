// embed_rmsnorm.cu -- embedding gather + RMSNorm as a single launch.
//   embed_rmsnorm            decode/chain paths
//   tree_prepare_embed_norm  tree path: tree_prepare's metadata writes plus
//                            the same gather+norm body
// Each replaces a back-to-back (embed, rms_norm [, tree_prepare]) sequence
// with identical per-element math, reduction order and rounding points, so
// the result is bit-identical. x receives the exact table bits (it is the
// residual stream add_rmsnorm consumes and updates).
//
// Semantics as embed.cu + norm.cu rms_norm: fp32 ssq per row (butterfly warp
// reduction, one value per warp in smem), y = bf16(x * rsqrt(mean+eps) * w),
// one rounding at the store.
//
// Layout / contract:
//   * table bf16 [vocab][hidden]; x, y bf16 [rows][hidden]; w bf16 [hidden];
//     ids uint32[rows] (tree: tokens).
//   * grid = (rows,1,1), block = (256,1,1), smem 128 B static; blocks past
//     the valid rows return before any write.
//   * FD_GUARD (__trap__): hidden != 2048, rows < 1, eps negative or not
//     finite, any null buffer, blockDim.x not a multiple of 32 or outside
//     [32,1024]. The tree entry additionally keeps tree_prepare's checks:
//     valid_rows in [1,64], budget in [1,64], valid_rows <= budget,
//     *prefix in [0, capacity-budget], *prefix == *expected, depth[row] in
//     [0,7] for valid rows, gridDim.x >= valid_rows.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace embed_norm {

using bf16 = __nv_bfloat16;

constexpr int kHidden = 2048;      // hidden_size
constexpr int kVec = 8;            // bf16 elements per 16 B access
constexpr int kMaxPerThread = 8;   // elements cached per thread in registers

__device__ __forceinline__ void shape(int rows, int hidden, float eps, unsigned nthr) {
    if (hidden != kHidden) __trap();
    if (rows < 1) __trap();
    if (!(eps >= 0.f) || !isfinite(eps)) __trap();
    if (nthr < 32u || nthr > 1024u || (nthr & 31u)) __trap();
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
    const int tid = threadIdx.x;
    const int nw = blockDim.x >> 5;
    v = warp_reduce_sum(v);
    if ((tid & 31) == 0) smem[tid >> 5] = v;
    __syncthreads();
    if (tid < 32) {
        float t = (tid < nw) ? smem[tid] : 0.0f;
        t = warp_reduce_sum(t);
        if (tid == 0) smem[0] = t;
    }
    __syncthreads();
    return smem[0];
}

// Load 8 contiguous bf16 -> float[8] with one 16 B access; the raw vector is
// returned so the caller can store it to x without a rounding step.
__device__ __forceinline__ float4 ld_vec8(const bf16* __restrict__ p, float* __restrict__ f) {
    union U { float4 v; __nv_bfloat162 h[4]; } u;
    u.v = *reinterpret_cast<const float4*>(p);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        float2 t = __bfloat1622float2(u.h[k]);
        f[2 * k] = t.x;
        f[2 * k + 1] = t.y;
    }
    return u.v;
}

__device__ __forceinline__ void st_vec8(bf16* __restrict__ p, const float* __restrict__ f) {
    union U { float4 v; __nv_bfloat162 h[4]; } u;
#pragma unroll
    for (int k = 0; k < 4; ++k) u.h[k] = __floats2bfloat162_rn(f[2 * k], f[2 * k + 1]);
    *reinterpret_cast<float4*>(p) = u.v;
}

// One block per row: gather the table row at `src` into x[row] and write
// rmsnorm(x[row], w) into y[row]. n == hidden (the guard pins it to 2048).
__device__ __forceinline__ void gather_norm_row(const bf16* __restrict__ src,
                                                const bf16* __restrict__ w,
                                                bf16* __restrict__ x, bf16* __restrict__ y,
                                                int n, float eps) {
    __shared__ float smem[32];
    const int nthr = blockDim.x;
    const int gid = threadIdx.x;
    const int nv = n / kVec;
    const bool cached = (nv <= nthr * (kMaxPerThread / kVec));   // block-uniform

    float r[kMaxPerThread];
    float ssq = 0.0f;
    if (cached) {
#pragma unroll
        for (int i = 0; i < kMaxPerThread / kVec; ++i) {
            const int c = gid + i * nthr;
            if (c < nv) {
                const size_t idx = (size_t)c * kVec;
                *reinterpret_cast<float4*>(x + idx) = ld_vec8(src + idx, r + i * kVec);
#pragma unroll
                for (int j = 0; j < kVec; ++j) ssq += r[i * kVec + j] * r[i * kVec + j];
            }
        }
    } else {
        for (int c = gid; c < nv; c += nthr) {
            const size_t idx = (size_t)c * kVec;
            *reinterpret_cast<float4*>(x + idx) = ld_vec8(src + idx, r);
#pragma unroll
            for (int j = 0; j < kVec; ++j) ssq += r[j] * r[j];
        }
    }
    const float rstd = rsqrtf(block_reduce_sum(ssq, smem) / (float)n + eps);

    if (cached) {
#pragma unroll
        for (int i = 0; i < kMaxPerThread / kVec; ++i) {
            const int c = gid + i * nthr;
            if (c < nv) {
                const size_t idx = (size_t)c * kVec;
                float ww[kVec], out[kVec];
                ld_vec8(w + idx, ww);
#pragma unroll
                for (int j = 0; j < kVec; ++j) out[j] = r[i * kVec + j] * rstd * ww[j];
                st_vec8(y + idx, out);
            }
        }
    } else {
        for (int c = gid; c < nv; c += nthr) {
            const size_t idx = (size_t)c * kVec;
            float ww[kVec], out[kVec];
            ld_vec8(src + idx, r);
            ld_vec8(w + idx, ww);
#pragma unroll
            for (int j = 0; j < kVec; ++j) out[j] = r[j] * rstd * ww[j];
            st_vec8(y + idx, out);
        }
    }
}

// tree_prepare's check conditions.
__device__ __forceinline__ void tree_rows(int n) {
    if (n < 1 || n > 64) __trap();
}
__device__ __forceinline__ void tree_extent(int start, int n, int capacity) {
    tree_rows(n);
    if (start < 0 || capacity <= 0 || start > capacity - n) __trap();
}

}  // namespace embed_norm

extern "C" {

// x[row] = table[ids[row]]; y[row] = rmsnorm(x[row], w).
__global__ __launch_bounds__(1024) void embed_rmsnorm(const __nv_bfloat16* __restrict__ table,
                                                      const uint32_t* __restrict__ ids,
                                                      __nv_bfloat16* __restrict__ x,
                                                      const __nv_bfloat16* __restrict__ w,
                                                      __nv_bfloat16* __restrict__ y, int rows,
                                                      int hidden, float eps) {
    using namespace embed_norm;
    shape(rows, hidden, eps, blockDim.x);
    if (!table || !ids || !x || !w || !y) __trap();
    const int row = blockIdx.x;
    if (row >= rows) return;
    gather_norm_row(table + (size_t)ids[row] * hidden, w, x + (size_t)row * hidden,
                    y + (size_t)row * hidden, hidden, eps);
}

// tree_prepare's metadata writes (all budget entries, every check) plus the
// same per-row gather+norm, for blockIdx.x < valid_rows.
__global__ __launch_bounds__(1024) void tree_prepare_embed_norm(
    const __nv_bfloat16* __restrict__ table, const uint32_t* __restrict__ tokens,
    const int* __restrict__ depth, const int* __restrict__ valid_rows,
    const int* __restrict__ expected, const int* __restrict__ prefix, int* __restrict__ snapshot,
    uint32_t* __restrict__ ids, int* __restrict__ positions, int* __restrict__ slots,
    const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y, int budget, int capacity, int hidden, float eps) {
    using namespace embed_norm;
    const int n = *valid_rows, start = *prefix;
    tree_rows(n);
    tree_extent(start, budget, capacity);
    if (n > budget || start != *expected) __trap();
    shape(n, hidden, eps, blockDim.x);
    if (!table || !tokens || !depth || !ids || !positions || !slots || !w || !x || !y) __trap();
    if (gridDim.x < n) __trap();

    // same index map as tree_prepare; budget <= 64 puts every entry in block 0
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < budget) {
        if (!m) *snapshot = start;
        if (m < n) {
            if (depth[m] < 0 || depth[m] > 7) __trap();
            ids[m] = tokens[m];
            positions[m] = start + depth[m];
        } else {
            ids[m] = 0;
            positions[m] = 0;
        }
        slots[m] = start + m;
    }

    const int row = blockIdx.x;
    if (row >= n) return;
    // reads tokens[row], not ids[row]: no dependence on block 0's store
    gather_norm_row(table + (size_t)tokens[row] * hidden, w, x + (size_t)row * hidden,
                    y + (size_t)row * hidden, hidden, eps);
}

}  // extern "C"
