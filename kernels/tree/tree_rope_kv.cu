// tree_rope_kv.cu -- fused tree RoPE + KV-cache write; one launch,
// bitwise-identical to the (tree_rope, tree_kv_write) pair it replaces,
// including the padding zeroing of qkv rows and cache slots.
// FD_GUARD: __trap() unless valid_rows in [1,64], budget in [1,64],
// valid_rows <= budget, q_heads == 16, kv_heads == 2, dim == 128, theta
// finite and > 0, 0 <= *prefix <= capacity - budget, and positions[row] >= 0
// for every valid row.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace tree_meta {
using bf16 = __nv_bfloat16;
__device__ __forceinline__ void rows(int n) {
    if (n < 1 || n > 64) __trap();
}

__device__ __forceinline__ void extent(int start, int n, int capacity) {
    rows(n);
    if (start < 0 || capacity <= 0 || start > capacity - n) __trap();
}
}

extern "C" __global__ void tree_rope_kv(__nv_bfloat16* __restrict__ qkv,
                                        __nv_bfloat16* __restrict__ k,
                                        __nv_bfloat16* __restrict__ v,
                                        const int* __restrict__ positions,
                                        const int* __restrict__ valid_rows,
                                        const int* __restrict__ prefix, int budget, int capacity,
                                        int q_heads, int kv_heads, int dim, float theta) {
    using bf16 = __nv_bfloat16;
    const int n = *valid_rows, start = *prefix;
    tree_meta::rows(n);
    tree_meta::extent(start, budget, capacity);
    if (n > budget) __trap();
    if (q_heads != 16 || kv_heads != 2 || dim != 128 || !isfinite(theta) || theta <= 0.f) __trap();

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= budget * 20 * 64) return;
    const int d = i % 64, head = (i / 64) % 20, row = i / (20 * 64);
    const int off = row * 2560 + head * 128 + d;
    if (row >= n) {
        qkv[off] = __float2bfloat16(0.f);
        qkv[off + 64] = __float2bfloat16(0.f);
        if (head >= 16) {
            if (head < 18) {
                const size_t dst = ((size_t)(head - 16) * capacity + start + row) * 128 + d;
                k[dst] = __float2bfloat16(0.f);
                k[dst + 64] = __float2bfloat16(0.f);
            } else {
                const size_t dst = ((size_t)(head - 18) * capacity + start + row) * 128 + d;
                v[dst] = __float2bfloat16(0.f);
                v[dst + 64] = __float2bfloat16(0.f);
            }
        }
        return;
    }
    if (head >= 18) {
        const size_t dst = ((size_t)(head - 18) * capacity + start + row) * 128 + d;
        v[dst] = qkv[off];
        v[dst + 64] = qkv[off + 64];
        return;
    }
    if (positions[row] < 0) __trap();
    const float angle = float(positions[row]) * (1.f / powf(theta, float(2 * d) / 128));
    const __nv_bfloat16 c = __float2bfloat16(cosf(angle)), s = __float2bfloat16(sinf(angle));
    const __nv_bfloat16 a = qkv[off], b = qkv[off + 64];
    const __nv_bfloat16 x = __hsub(__hmul(a, c), __hmul(b, s));
    const __nv_bfloat16 y = __hadd(__hmul(b, c), __hmul(a, s));
    qkv[off] = x;
    qkv[off + 64] = y;
    if (head >= 16) {
        const size_t dst = ((size_t)(head - 16) * capacity + start + row) * 128 + d;
        k[dst] = x;
        k[dst + 64] = y;
    }
}
