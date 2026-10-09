// rope_kv.cu -- fused RoPE rotation + KV-cache write (chain/decode); one
// launch, bitwise-identical to the (rope, kv_write) pair it replaces.
// FD_GUARD: __trap() unless q_heads == 16, kv_heads == 2, dim == 128, theta
// finite and > 0, rows >= 1, and 0 <= *start <= capacity - rows.
#include <cuda_bf16.h>

namespace rope_kv_guard {
__device__ __forceinline__ void shape(int rows, int q_heads, int kv_heads,
                                      int dim, float theta) {
    if (q_heads != 16 || kv_heads != 2 || dim != 128) __trap();
    if (!(theta > 0.f) || !isfinite(theta)) __trap();
    if (rows < 1) __trap();
}
__device__ __forceinline__ void extent(int start, int rows, int capacity) {
    if (start < 0 || capacity <= 0 || start > capacity - rows) __trap();
}
}

extern "C" __global__ void rope_kv(__nv_bfloat16* __restrict__ qkv,
                                   __nv_bfloat16* __restrict__ k,
                                   __nv_bfloat16* __restrict__ v,
                                   const int* __restrict__ start, int rows,
                                   int q_heads, int kv_heads, int dim,
                                   float theta, int capacity) {
    rope_kv_guard::shape(rows, q_heads, kv_heads, dim, theta);
    rope_kv_guard::extent(*start, rows, capacity);
    const int half = dim / 2;
    const int heads = q_heads + 2 * kv_heads;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * heads * half) return;
    const int d = i % half;
    const int head = (i / half) % heads;
    const int row = i / (half * heads);
    const int off = row * heads * dim + head * dim + d;
    if (head < q_heads + kv_heads) {
        float inv_freq = 1.0f / powf(theta, float(2 * d) / dim);
        float angle = float(*start + row) * inv_freq;
        __nv_bfloat16 c = __float2bfloat16(cosf(angle));
        __nv_bfloat16 s = __float2bfloat16(sinf(angle));
        __nv_bfloat16 a = qkv[off];
        __nv_bfloat16 b = qkv[off + half];
        __nv_bfloat16 x = __hsub(__hmul(a, c), __hmul(b, s));
        __nv_bfloat16 y = __hadd(__hmul(b, c), __hmul(a, s));
        qkv[off] = x;
        qkv[off + half] = y;
        if (head >= q_heads) {
            size_t dst = ((size_t)(head - q_heads) * capacity + *start + row) * dim + d;
            k[dst] = x;
            k[dst + half] = y;
        }
    } else {
        size_t dst = ((size_t)(head - q_heads - kv_heads) * capacity + *start + row) * dim + d;
        v[dst] = qkv[off];
        v[dst + half] = qkv[off + half];
    }
}
