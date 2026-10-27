#include "gpu_compat.cuh"

extern "C" __global__ void rope_table_kv(__nv_bfloat16* qkvg,
    __nv_bfloat16* k, __nv_bfloat16* v, const __nv_bfloat16* table,
    const int* start, int rows, int q_heads, int kv_heads, int dim,
    int capacity, int row_stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int half = dim / 2;
    const int heads = q_heads + kv_heads;
    if (i >= rows * heads * half) return;
    const int d = i % half;
    const int head = (i / half) % heads;
    const int row = i / (half * heads);
    const int position = *start + row;
    if (position < 0 || position >= capacity) { __trap(); return; }
    const int off = row * row_stride + head * dim + d;
    const __nv_bfloat16 c = table[position * dim + d];
    const __nv_bfloat16 s = table[position * dim + half + d];
    const __nv_bfloat16 a = qkvg[off];
    const __nv_bfloat16 b = qkvg[off + half];
    const __nv_bfloat16 lo = __hsub(__hmul(a,c), __hmul(b,s));
    const __nv_bfloat16 hi = __hadd(__hmul(b,c), __hmul(a,s));
    if (head < q_heads) {
        qkvg[off] = lo;
        qkvg[off + half] = hi;
    } else {
        const size_t dst = ((size_t)(head - q_heads) * capacity + position) * dim + d;
        k[dst] = lo;
        k[dst + half] = hi;
        v[dst] = qkvg[off + kv_heads * dim];
        v[dst + half] = qkvg[off + kv_heads * dim + half];
    }
}

extern "C" __global__ void sequence_positions(const int* prefix, int* metadata,
                                                int* axes, int rows, int capacity,
                                                int rotary_shift) {
    const int p = *prefix;
    if (p < 0 || rows < 1 || rows > 64 || p > capacity - rows || p + rotary_shift < 0)
        __trap();
    const int row = threadIdx.x;
    if (row == 0) {
        metadata[0] = p;
        metadata[1] = rows;
        metadata[2] = p + rows;
        metadata[3] = capacity;
        metadata[4] = 0;
        metadata[5] = metadata[6] = metadata[7] = 0;
    }
    if (row < rows) {
        const int position = p + row + rotary_shift;
        axes[row * 3] = axes[row * 3 + 1] = axes[row * 3 + 2] = position;
    }
}

extern "C" __global__ void rope(__nv_bfloat16* qkv, const int* start,
                               int rows, int q_heads, int kv_heads, int dim, float theta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int heads = q_heads + kv_heads;
    if (i >= rows * heads * (dim / 2)) return;
    int d = i % (dim / 2);
    int head = (i / (dim / 2)) % heads;
    int row = i / ((dim / 2) * heads);
    int stride = (q_heads + 2 * kv_heads) * dim;
    int off = row * stride + head * dim + d;
    float inv_freq = 1.0f / powf(theta, float(2 * d) / dim);
    float angle = float(*start + row) * inv_freq;
    __nv_bfloat16 c = __float2bfloat16(cosf(angle));
    __nv_bfloat16 s = __float2bfloat16(sinf(angle));
    __nv_bfloat16 a = qkv[off];
    __nv_bfloat16 b = qkv[off + dim / 2];
    qkv[off] = __hsub(__hmul(a, c), __hmul(b, s));
    qkv[off + dim / 2] = __hadd(__hmul(b, c), __hmul(a, s));
}

extern "C" __global__ void kv_write(const __nv_bfloat16* qkv, __nv_bfloat16* k,
                                   __nv_bfloat16* v, const int* start,
                                   int rows, int q_heads, int kv_heads, int dim, int capacity) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * kv_heads * dim) return;
    int d = i % dim;
    int head = (i / dim) % kv_heads;
    int row = i / (dim * kv_heads);
    int src = row * (q_heads + 2 * kv_heads) * dim + q_heads * dim + head * dim + d;
    size_t dst = ((size_t)head * capacity + *start + row) * dim + d;
    k[dst] = qkv[src];
    v[dst] = qkv[src + kv_heads * dim];
}

extern "C" __global__ void rope_float(__nv_bfloat16* qkv, const int* start,
                                     int rows, int q_heads, int kv_heads, int dim, float theta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int heads = q_heads + kv_heads;
    if (i >= rows * heads * (dim / 2)) return;
    int d = i % (dim / 2);
    int head = (i / (dim / 2)) % heads;
    int row = i / ((dim / 2) * heads);
    int off = row * (q_heads + 2 * kv_heads) * dim + head * dim + d;
    float inv = 1.0f / powf(theta, float(2 * d) / dim);
    float angle = float(*start + row) * inv;
    float c = __bfloat162float(__float2bfloat16(cosf(angle)));
    float s = __bfloat162float(__float2bfloat16(sinf(angle)));
    float a = __bfloat162float(qkv[off]);
    float b = __bfloat162float(qkv[off + dim / 2]);
    qkv[off] = __float2bfloat16(a * c - b * s);
    qkv[off + dim / 2] = __float2bfloat16(b * c + a * s);
}

template <bool ROTARY, bool PLANAR>
__device__ __forceinline__ void qknorm_rope_kv_impl(__nv_bfloat16* qkv,
                                                    __nv_bfloat16* k, __nv_bfloat16* v,
                                                    const __nv_bfloat16* q_norm_w,
                                                    const __nv_bfloat16* k_norm_w,
                                                    const int* start,
                                                    int cache_start,
                                                    int rows, int q_heads, int kv_heads,
                                                    int dim, int capacity,
                                                    int ring_tokens, float theta, float eps,
                                                    size_t q_row_stride, size_t k_offset,
                                                    size_t k_row_stride, size_t v_offset,
                                                    size_t v_row_stride) {
    const bool shape = (q_heads == 16 && kv_heads == 2) ||
                       (q_heads == 32 && kv_heads == 4);
    if (!shape || dim != 128 || rows < 1 || rows > 65535 ||
        *start < 0 || cache_start < -1 ||
        (ring_tokens == 0 && cache_start < 0 && *start + rows > capacity) ||
        (ring_tokens == 0 && cache_start >= 0 && cache_start + rows > capacity) ||
        (ring_tokens > 0 && cache_start >= 0))
        __trap();
    __shared__ float smem[32];
    const int heads = q_heads + 2 * kv_heads;
    const int row = blockIdx.x / heads;
    const int head = blockIdx.x % heads;
    if (row >= rows) return;
    const int gid = threadIdx.x;
    const int nthr = blockDim.x;
    size_t base;
    if constexpr (PLANAR) {
        if (head < q_heads) {
            base = (size_t)row * q_row_stride + (size_t)head * dim;
        } else if (head < q_heads + kv_heads) {
            const int kv_head = head - q_heads;
            base = k_offset + (size_t)row * k_row_stride + (size_t)kv_head * dim;
        } else {
            const int v_head = head - q_heads - kv_heads;
            base = v_offset + (size_t)row * v_row_stride + (size_t)v_head * dim;
        }
    } else {
        base = (size_t)row * heads * dim + (size_t)head * dim;
    }
    const int position = *start + row;
    const int cache_pos = ring_tokens > 0
                              ? position % ring_tokens
                              : (cache_start < 0 ? position : cache_start + row);

    if (head < q_heads + kv_heads) {
        const __nv_bfloat16* w = (head < q_heads) ? q_norm_w : k_norm_w;
        float r[8];
        float ssq = 0.0f;
        if (gid < dim / 8) {
            const __nv_bfloat16* p = qkv + base + (size_t)gid * 8;
            union U { float4 x; __nv_bfloat162 h[4]; } u;
            u.x = *reinterpret_cast<const float4*>(p);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float2 t = __bfloat1622float2(u.h[j >> 1]);
                r[j] = (j & 1) ? t.y : t.x;
                ssq += r[j] * r[j];
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(FF_WARP_MASK, ssq, o);
        if ((gid & 31) == 0) smem[gid >> 5] = ssq;
        __syncthreads();
        if (gid < 32) {
            int nw = nthr >> 5;
            float t = (gid < nw) ? smem[gid] : 0.0f;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(FF_WARP_MASK, t, o);
            if (gid == 0) smem[0] = t;
        }
        __syncthreads();
        const float rstd = rsqrtf(smem[0] / (float)dim + eps);

        if (gid < dim / 8) {
            float ww[8], out[8];
            const __nv_bfloat16* wp = w + (size_t)gid * 8;
            union U { float4 x; __nv_bfloat162 h[4]; } u;
            u.x = *reinterpret_cast<const float4*>(wp);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float2 t = __bfloat1622float2(u.h[j >> 1]);
                ww[j] = (j & 1) ? t.y : t.x;
                out[j] = r[j] * rstd * ww[j];
            }
            __nv_bfloat16* p = qkv + base + (size_t)gid * 8;
            union O { float4 x; __nv_bfloat162 h[4]; } o;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                o.h[j] = __floats2bfloat162_rn(out[2 * j], out[2 * j + 1]);
            *reinterpret_cast<float4*>(p) = o.x;
        }
        __syncthreads();

        if constexpr (ROTARY) {
          if (gid < dim / 2) {
            const float inv = 1.0f / powf(theta, float(2 * gid) / dim);
            const float angle = float(position) * inv;
            const float c = __bfloat162float(__float2bfloat16(cosf(angle)));
            const float s = __bfloat162float(__float2bfloat16(sinf(angle)));
            const float a = __bfloat162float(qkv[base + gid]);
            const float b = __bfloat162float(qkv[base + gid + dim / 2]);
            const __nv_bfloat16 lo = __float2bfloat16(a * c - b * s);
            const __nv_bfloat16 hi = __float2bfloat16(b * c + a * s);
            qkv[base + gid] = lo;
            qkv[base + gid + dim / 2] = hi;
            if (head >= q_heads) {
                const int kv_head = head - q_heads;
                const size_t dst = ((size_t)kv_head * capacity + cache_pos) * dim;
                k[dst + gid] = lo;
                k[dst + gid + dim / 2] = hi;
            }
          }
        } else if (head >= q_heads) {
            const int kv_head = head - q_heads;
            const size_t dst = ((size_t)kv_head * capacity + cache_pos) * dim;
            for (int d = gid; d < dim; d += nthr) k[dst + d] = qkv[base + d];
        }
    } else {
        const int kv_head = head - q_heads - kv_heads;
        const size_t dst = ((size_t)kv_head * capacity + cache_pos) * dim;
#pragma unroll
        for (int d = gid; d < dim; d += nthr) v[dst + d] = qkv[base + d];
    }
}

// draft q/k RMSNorm + RoPE + KV write in one launch,
// bit-identical to columns+norm+columns (q,k) + rope_float + kv_write.
extern "C" __global__ void draft_qknorm_rope_kv(__nv_bfloat16* qkv,
                                                __nv_bfloat16* k, __nv_bfloat16* v,
                                                const __nv_bfloat16* q_norm_w,
                                                const __nv_bfloat16* k_norm_w,
                                                const int* start,
                                                int rows, int q_heads, int kv_heads,
                                                int dim, int capacity,
                                                int ring_tokens, float theta, float eps) {
    qknorm_rope_kv_impl<true, false>(qkv, k, v, q_norm_w, k_norm_w, start, -1, rows,
                                     q_heads, kv_heads, dim, capacity, ring_tokens, theta, eps,
                                     0, 0, 0, 0, 0);
}

extern "C" __global__ void qknorm_kv_nope(__nv_bfloat16* qkv,
                                          __nv_bfloat16* k, __nv_bfloat16* v,
                                          const __nv_bfloat16* q_norm_w,
                                          const __nv_bfloat16* k_norm_w,
                                          const int* start,
                                          int rows, int q_heads, int kv_heads,
                                          int dim, int capacity,
                                          int ring_tokens, float, float eps) {
    qknorm_rope_kv_impl<false, false>(qkv, k, v, q_norm_w, k_norm_w, start, -1, rows,
                                      q_heads, kv_heads, dim, capacity, ring_tokens, 1.0f, eps,
                                      0, 0, 0, 0, 0);
}

extern "C" __global__ void flint_qknorm_rope_kv(__nv_bfloat16* qkv,
                                                 __nv_bfloat16* k, __nv_bfloat16* v,
                                                 const __nv_bfloat16* q_norm_w,
                                                 const __nv_bfloat16* k_norm_w,
                                                 const int* start,
                                                 int cache_start,
                                                 int rows, int q_heads, int kv_heads,
                                                 int dim, int capacity,
                                                 int ring_tokens, float theta, float eps,
                                                 unsigned long long q_row_stride,
                                                 unsigned long long k_offset,
                                                 unsigned long long k_row_stride,
                                                 unsigned long long v_offset,
                                                 unsigned long long v_row_stride) {
    qknorm_rope_kv_impl<true, true>(qkv, k, v, q_norm_w, k_norm_w, start, cache_start, rows,
                                     q_heads, kv_heads, dim, capacity, ring_tokens, theta, eps,
                                     q_row_stride, k_offset, k_row_stride, v_offset, v_row_stride);
}

extern "C" __global__ void flint_qknorm_kv_nope(__nv_bfloat16* qkv,
                                                 __nv_bfloat16* k, __nv_bfloat16* v,
                                                 const __nv_bfloat16* q_norm_w,
                                                 const __nv_bfloat16* k_norm_w,
                                                 const int* start,
                                                 int cache_start,
                                                 int rows, int q_heads, int kv_heads,
                                                 int dim, int capacity,
                                                 int ring_tokens, float, float eps,
                                                 unsigned long long q_row_stride,
                                                 unsigned long long k_offset,
                                                 unsigned long long k_row_stride,
                                                 unsigned long long v_offset,
                                                 unsigned long long v_row_stride) {
    qknorm_rope_kv_impl<false, true>(qkv, k, v, q_norm_w, k_norm_w, start, cache_start, rows,
                                      q_heads, kv_heads, dim, capacity, ring_tokens, 1.0f, eps,
                                      q_row_stride, k_offset, k_row_stride, v_offset, v_row_stride);
}
