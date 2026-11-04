#include "gpu_compat.cuh"

template<typename T>
__device__ __forceinline__ void rotary_prefix(T* q,
    const T* key, const T* value,
    T* k, T* v, const T* table,
    int start, int rows, int q_heads, int kv_heads, int dim, int rotary,
    int capacity) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int heads = q_heads + kv_heads;
    if (i >= rows * heads * dim) return;
    const int d = i % dim;
    const int head = (i / dim) % heads;
    const int row = i / (heads * dim);
    const int position = start + row;
    if (position < 0 || position >= capacity) { __trap(); return; }
    const bool query = head < q_heads;
    const int h = query ? head : head - q_heads;
    const size_t src = ((size_t)row * (query ? q_heads : kv_heads) + h) * dim;
    const size_t dst = ((size_t)h * capacity + position) * dim;
    if (!query) v[dst + d] = value[src + d];
    if (d >= rotary) {
        if (!query) k[dst + d] = key[src + d];
        return;
    }
    const int half = rotary / 2;
    if (d >= half) return;
    const T* input = query ? q : key;
    const T a = input[src + d];
    const T b = input[src + d + half];
    const T c = table[(size_t)position * rotary + d];
    const T s = table[(size_t)position * rotary + d + half];
    T lo, hi;
    if constexpr (sizeof(T) == sizeof(float)) {
        lo = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
        hi = __fadd_rn(__fmul_rn(b, c), __fmul_rn(a, s));
    } else {
#if defined(__HIP__)
        lo = __hsub_rn(__hmul_rn(a, c), __hmul_rn(b, s));
        hi = __hadd_rn(__hmul_rn(b, c), __hmul_rn(a, s));
#else
        lo = __hsub(__hmul(a, c), __hmul(b, s));
        hi = __hadd(__hmul(b, c), __hmul(a, s));
#endif
    }
    T* output = query ? q + src : k + dst;
    output[d] = lo;
    output[d + half] = hi;
}

extern "C" __global__ void rotary_prefix_kv_f32(float* q,
    const float* key, const float* value, float* k, float* v, const float* table,
    int start, int rows, int q_heads, int kv_heads, int dim, int rotary, int capacity) {
    rotary_prefix(q, key, value, k, v, table, start, rows, q_heads, kv_heads, dim, rotary, capacity);
}

extern "C" __global__ void rotary_prefix_kv(__nv_bfloat16* q,
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    __nv_bfloat16* k, __nv_bfloat16* v, const __nv_bfloat16* table,
    int start, int rows, int q_heads, int kv_heads, int dim, int rotary,
    int capacity) {
    rotary_prefix(q, key, value, k, v, table, start, rows, q_heads, kv_heads, dim, rotary, capacity);
}

extern "C" __global__ void rotary_prefix_kv_device(__nv_bfloat16* q,
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    __nv_bfloat16* k, __nv_bfloat16* v, const __nv_bfloat16* table,
    const int* start, int rows, int q_heads, int kv_heads, int dim, int rotary,
    int capacity) {
    rotary_prefix(q, key, value, k, v, table, *start, rows, q_heads, kv_heads, dim, rotary, capacity);
}

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
    const __nv_bfloat16 c0 = table[position * 2 * dim + d];
    const __nv_bfloat16 s0 = table[position * 2 * dim + half + d];
    const __nv_bfloat16 c1 = table[position * 2 * dim + dim + d];
    const __nv_bfloat16 s1 = table[position * 2 * dim + dim + half + d];
    const __nv_bfloat16 a = qkvg[off];
    const __nv_bfloat16 b = qkvg[off + half];
#if defined(__HIP__)
    const __nv_bfloat16 lo = __hsub_rn(__hmul_rn(a,c0), __hmul_rn(b,s0));
    const __nv_bfloat16 hi = __hadd_rn(__hmul_rn(b,c1), __hmul_rn(a,s1));
#else
    const __nv_bfloat16 lo = __hsub(__hmul(a,c0), __hmul(b,s0));
    const __nv_bfloat16 hi = __hadd(__hmul(b,c1), __hmul(a,s1));
#endif
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

extern "C" __global__ void rope_2d_qk(__nv_bfloat16* qkv, const int* positions,
    int rows, int q_heads, int kv_heads, int dim, int row_stride, float theta) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int half = dim / 2;
    const int heads = q_heads + kv_heads;
    if (!qkv || !positions || rows < 1 || heads < 1 || dim < 2 || (dim & 1) ||
        row_stride < (q_heads + 2 * kv_heads) * dim || i >= rows * heads * half) return;
    const int d = i % half;
    const int head = (i / half) % heads;
    const int row = i / (half * heads);
    const int off = row * row_stride + head * dim + d;
    const float inv = 1.0f / powf(theta, float(2 * d) / float(dim));
    const float angle0 = float(positions[row]) * inv;
    const float angle1 = float(positions[rows + row]) * inv;
    const __nv_bfloat16 c0 = __float2bfloat16_rn(cosf(angle0));
    const __nv_bfloat16 s0 = __float2bfloat16_rn(sinf(angle0));
    const __nv_bfloat16 c1 = __float2bfloat16_rn(cosf(angle1));
    const __nv_bfloat16 s1 = __float2bfloat16_rn(sinf(angle1));
    const __nv_bfloat16 a = qkv[off];
    const __nv_bfloat16 b = qkv[off + half];
#if defined(__HIP__)
    qkv[off] = __hsub_rn(__hmul_rn(a, c0), __hmul_rn(b, s0));
    qkv[off + half] = __hadd_rn(__hmul_rn(b, c1), __hmul_rn(a, s1));
#else
    qkv[off] = __hsub(__hmul(a, c0), __hmul(b, s0));
    qkv[off + half] = __hadd(__hmul(b, c1), __hmul(a, s1));
#endif
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

template <bool ROTARY, bool PLANAR, bool TABLE = false, bool SHARED_KV = false>
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
                                                    size_t v_row_stride,
                                                    const __nv_bfloat16* table) {
    const bool shape = TABLE ? (q_heads > 0 && kv_heads >= 0 &&
                                (dim == 256 || dim == 512))
                             : ((q_heads == 16 && kv_heads == 2) ||
                                (q_heads == 32 && kv_heads == 4));
    if (!shape || (!TABLE && dim != 128) || (TABLE && !table) || rows < 1 || rows > 65535 ||
        *start < 0 || cache_start < -1 ||
        (ring_tokens == 0 && cache_start < 0 && *start + rows > capacity) ||
        (ring_tokens == 0 && cache_start >= 0 && cache_start + rows > capacity) ||
        (ring_tokens > 0 && cache_start >= 0))
        __trap();
    __shared__ float smem[32];
    __shared__ __align__(16) __nv_bfloat16 s_norm[512];
    const int heads = q_heads + (SHARED_KV ? 1 : 2) * kv_heads;
    const int row = blockIdx.x / heads;
    const int head = blockIdx.x % heads;
    if (row >= rows) return;
    const int gid = threadIdx.x;
    const int nthr = blockDim.x;
    if constexpr (TABLE) {
        if (dim == 256 && gid >= 32) return;
    }
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
    const int cache_pos = ring_tokens > 0 ? position % ring_tokens
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
        float total_ssq = ssq;
        if (!(TABLE && dim == 256)) {
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
            total_ssq = smem[0];
        }
        const float rstd = powf(total_ssq / (float)dim + eps, -0.5f);

        if (gid < dim / 8) {
            if constexpr (SHARED_KV) {
                if (head >= q_heads) {
                    const size_t dst = ((size_t)(head - q_heads) * capacity + cache_pos) * dim;
                    union V { float4 x; __nv_bfloat162 h[4]; } vo;
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        vo.h[j] = __floats2bfloat162_rn(r[2 * j] * rstd, r[2 * j + 1] * rstd);
                    *reinterpret_cast<float4*>(v + dst + (size_t)gid * 8) = vo.x;
                }
            }
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
            union O { float4 x; __nv_bfloat162 h[4]; } o;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                o.h[j] = __floats2bfloat162_rn(out[2 * j], out[2 * j + 1]);
            if constexpr (TABLE) {
                *reinterpret_cast<float4*>(s_norm + (size_t)gid * 8) = o.x;
            } else {
                __nv_bfloat16* p = qkv + base + (size_t)gid * 8;
                *reinterpret_cast<float4*>(p) = o.x;
            }
        }
        if constexpr (TABLE) {
            if (dim == 256) {
                __syncwarp();
            } else {
                __syncthreads();
            }
            if (gid < dim / 8) {
                const int d = gid * 4;
                union P { uint2 u; __nv_bfloat16 v[4]; } ca, sa, aa, ba, loa, hia;
                ca.u = *reinterpret_cast<const uint2*>(table + (size_t)position * dim + d);
                sa.u = *reinterpret_cast<const uint2*>(table + (size_t)position * dim + dim / 2 + d);
                aa.u = *reinterpret_cast<const uint2*>(s_norm + d);
                ba.u = *reinterpret_cast<const uint2*>(s_norm + dim / 2 + d);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const __nv_bfloat16 c = ca.v[j];
                    const __nv_bfloat16 s = sa.v[j];
                    const __nv_bfloat16 a = aa.v[j];
                    const __nv_bfloat16 b = ba.v[j];
                    loa.v[j] = __hsub_rn(__hmul_rn(a, c), __hmul_rn(b, s));
                    hia.v[j] = __hadd_rn(__hmul_rn(b, c), __hmul_rn(a, s));
                }
                if (head < q_heads) {
                    *reinterpret_cast<uint2*>(qkv + base + d) = loa.u;
                    *reinterpret_cast<uint2*>(qkv + base + dim / 2 + d) = hia.u;
                } else {
                    const int kv_head = head - q_heads;
                    const size_t dst = ((size_t)kv_head * capacity + cache_pos) * dim;
                    *reinterpret_cast<uint2*>(k + dst + d) = loa.u;
                    *reinterpret_cast<uint2*>(k + dst + dim / 2 + d) = hia.u;
                }
            }
        } else {
        __syncthreads();

        if constexpr (ROTARY) {
          for (int d = gid; d < dim / 2; d += nthr) {
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
        }
    } else {
        const int kv_head = head - q_heads - kv_heads;
        const size_t dst = ((size_t)kv_head * capacity + cache_pos) * dim;
        if constexpr (TABLE) {
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
        float total_ssq = ssq;
        if (dim != 256) {
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
            total_ssq = smem[0];
        }
        const float rstd = powf(total_ssq / (float)dim + eps, -0.5f);
        if (gid < dim / 8) {
            union O { float4 x; __nv_bfloat162 h[4]; } o;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                o.h[j] = __floats2bfloat162_rn(r[2 * j] * rstd, r[2 * j + 1] * rstd);
            *reinterpret_cast<float4*>(v + dst + (size_t)gid * 8) = o.x;
        }
        } else {
#pragma unroll
            for (int d = gid; d < dim; d += nthr) v[dst + d] = qkv[base + d];
        }
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
                                     0, 0, 0, 0, 0, nullptr);
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
                                      0, 0, 0, 0, 0, nullptr);
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
                                     q_row_stride, k_offset, k_row_stride, v_offset, v_row_stride, nullptr);
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
                                      q_row_stride, k_offset, k_row_stride, v_offset, v_row_stride, nullptr);
}

extern "C" __global__ void qknorm_rope_table_kv(__nv_bfloat16* qkv,
                                                 __nv_bfloat16* k, __nv_bfloat16* v,
                                                 const __nv_bfloat16* q_norm_w,
                                                 const __nv_bfloat16* k_norm_w,
                                                 const __nv_bfloat16* table,
                                                 const int* start,
                                                 int cache_start,
                                                 int rows, int q_heads, int kv_heads,
                                                 int dim, int capacity,
                                                 int ring_tokens, float eps,
                                                 unsigned long long q_row_stride,
                                                 unsigned long long k_offset,
                                                 unsigned long long k_row_stride,
                                                 unsigned long long v_offset,
                                                 unsigned long long v_row_stride) {
    qknorm_rope_kv_impl<true, true, true>(qkv, k, v, q_norm_w, k_norm_w, start, cache_start,
                                          rows, q_heads, kv_heads, dim, capacity, ring_tokens,
                                          1.0f, eps, q_row_stride, k_offset, k_row_stride,
                                          v_offset, v_row_stride, table);
}

extern "C" __global__ void qknorm_rope_table_shared_kv(__nv_bfloat16* qkv,
                                                 __nv_bfloat16* k, __nv_bfloat16* v,
                                                 const __nv_bfloat16* q_norm_w,
                                                 const __nv_bfloat16* k_norm_w,
                                                 const __nv_bfloat16* table,
                                                 const int* start,
                                                 int cache_start,
                                                 int rows, int q_heads, int kv_heads,
                                                 int dim, int capacity,
                                                 int ring_tokens, float eps,
                                                 unsigned long long q_row_stride,
                                                 unsigned long long k_offset,
                                                 unsigned long long k_row_stride,
                                                 unsigned long long v_offset,
                                                 unsigned long long v_row_stride) {
    qknorm_rope_kv_impl<true, true, true, true>(qkv, k, v, q_norm_w, k_norm_w, start,
                                             cache_start, rows, q_heads, kv_heads, dim,
                                             capacity, ring_tokens, 1.0f, eps, q_row_stride,
                                             k_offset, k_row_stride, v_offset, v_row_stride, table);
}

// Adjacent-pair layout mode of the rope op (the model definition's complex
// convention: pairs (x[2i], x[2i+1]), in-place, FP32 math, one BF16 round).
// rope_pair_rows emits the interleaved (cos, sin) rows of one position range
// from per-layer inverse frequencies; the second buffer holds the conjugate
// the attention output needs.
// rows*tail/2 threads: fwd[r][2i]=cos(a), fwd[r][2i+1]=sin(a),
// inv[r][2i+1]=-sin(a), a = (start + r) * inv_freq[i].
extern "C" __global__ void rope_pair_rows(
    const float* __restrict__ inv_freq, int start, int rows, int tail,
    float* __restrict__ fwd, float* __restrict__ inv) {
  if (start < 0 || rows < 1 || tail < 2 || (tail % 2) != 0) __trap();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int half = tail / 2;
  if (i >= rows * half) return;
  const int r = i / half, d = i % half;
  const float angle = float(start + r) * inv_freq[d];
  const float c = cosf(angle), s = sinf(angle);
  fwd[(size_t)r * tail + 2 * d] = c;
  fwd[(size_t)r * tail + 2 * d + 1] = s;
  inv[(size_t)r * tail + 2 * d] = c;
  inv[(size_t)r * tail + 2 * d + 1] = -s;
}

// rows*n*(tail/2) threads: rotate the last `tail` elements of every n-vector
// of x[r] in place by the row's (cos, sin) pair.
namespace {

__device__ __forceinline__ float rope_to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float rope_to_f32(__half v) { return __half2float(v); }

template <typename T>
__device__ __forceinline__ T rope_from_f32(float v);
template <>
__device__ __forceinline__ __nv_bfloat16 rope_from_f32<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}
template <>
__device__ __forceinline__ __half rope_from_f32<__half>(float v) {
  return __float2half_rn(v);
}

template <typename T>
__device__ __forceinline__ void rope_pair_tail_impl(
    T* x, const float* __restrict__ rotary, int rows, int n,
    int width, int tail, int rotary_heads, bool head_major = false) {
  if (rows < 1 || n < 1 || tail < 2 || (tail % 2) != 0 || tail > width ||
      (rotary_heads != 1 && rotary_heads != n))
    __trap();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int half = tail / 2;
  if (i >= rows * n * half) return;
  const int pair = i % half;
  const int v = (i / half) % n;
  const int r = i / (half * n);
  const size_t vector = head_major ? (size_t)v * rows + r : (size_t)r * n + v;
  const size_t d = vector * width + (width - tail) + 2 * pair;
  const size_t rot_row = (rotary_heads == 1) ? (size_t)r : ((size_t)r * n + v);
  const float c = rotary[rot_row * tail + 2 * pair];
  const float s = rotary[rot_row * tail + 2 * pair + 1];
  const float a = rope_to_f32(x[d]);
  const float b = rope_to_f32(x[d + 1]);
  x[d] = rope_from_f32<T>(__fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s)));
  x[d + 1] = rope_from_f32<T>(__fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c)));
}

}  // namespace

extern "C" __global__ void rope_pair_rows_3d(
    const float* __restrict__ delta_pos, const float* __restrict__ freqs,
    int rows, int heads, int tail, float* __restrict__ fwd, int phase_form) {
  if (!delta_pos || !freqs || !fwd || rows < 1 || heads < 1 || tail < 6 ||
      (tail % 2) != 0 || phase_form < 0 || phase_form > 1)
    __trap();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int half = tail / 2;
  const int total = rows * heads * half;
  if (i >= total) return;
  const int d = i % half;
  const int rh = i / half;
  const int h0 = tail / 6;
  if (phase_form == 1 && d >= 3 * h0) {
    fwd[(size_t)rh * tail + 2 * d] = 1.0f;
    fwd[(size_t)rh * tail + 2 * d + 1] = 0.0f;
    return;
  }
  const int axis = (d < h0) ? 0 : ((d < 2 * h0) ? 1 : 2);
  const float x = delta_pos[(size_t)rh * 3 + axis];
  const float f = freqs[d];
  const float f_t = tanhf(f);
  const float cm = __fadd_rn(__fmul_rn(x, f_t), __fmul_rn(x, __fsub_rn(f, f_t)));
  const float angle = phase_form == 0
      ? __fmul_rn(cm, 3.14159265358979323846f) : __fmul_rn(x, f);
  fwd[(size_t)rh * tail + 2 * d] = cosf(angle);
  fwd[(size_t)rh * tail + 2 * d + 1] = sinf(angle);
}

extern "C" __global__ void rope_pair_tail(
    __nv_bfloat16* x, const float* __restrict__ rotary, int rows, int n,
    int width, int tail) {
  rope_pair_tail_impl(x, rotary, rows, n, width, tail, 1);
}

extern "C" __global__ void rope_pair_tail_hm(
    __nv_bfloat16* x, const float* __restrict__ rotary, int rows, int heads,
    int width, int tail) {
  rope_pair_tail_impl(x, rotary, rows, heads, width, tail, 1, true);
}

// Partial split-half rotary apply in F32: q [rows, heads, head_dim], only the
// leading rotary_dim dims rotate against per-row cos/sin tables (already
// sliced to the row positions, layout [rows, rotary_dim]).
// Partial split-half rotary apply in F32: q [rows, heads, head_dim]; each
// thread owns one (d, d + rotary_dim/2) pair, reads both, writes both.
extern "C" __global__ void partial_rotary_f32(float* __restrict__ q,
    const float* __restrict__ cos, const float* __restrict__ sin,
    int rows, int heads, int head_dim, int rotary_dim) {
    if (!q || !cos || !sin || !rows || !heads || head_dim < rotary_dim ||
        rotary_dim % 2 || blockDim.x != 256 || blockDim.y != 1 ||
        blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1) { __trap(); return; }
    const int half = rotary_dim / 2;
    const long long pairs = (long long)rows * heads * half;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < pairs;
         i += (long long)gridDim.x * blockDim.x) {
        const int d = (int)(i % half);
        const long long base = (i / half) * head_dim;
        const long long row = i / ((long long)heads * half);
        const float c0 = cos[row * rotary_dim + d];
        const float s0 = sin[row * rotary_dim + d];
        const float c1 = cos[row * rotary_dim + half + d];
        const float s1 = sin[row * rotary_dim + half + d];
        const float a = q[base + d];
        const float b = q[base + half + d];
        q[base + d] = a * c0 - b * s0;
        q[base + half + d] = a * s1 + b * c1;
    }
}
