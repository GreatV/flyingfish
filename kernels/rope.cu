#include <cuda_bf16.h>

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

// draft_qknorm_rope_kv: draft q/k RMSNorm + RoPE + KV write in one launch,
// bit-identical to columns+norm+columns (q,k) + rope_float + kv_write.
extern "C" __global__ void draft_qknorm_rope_kv(__nv_bfloat16* qkv,
                                                __nv_bfloat16* k, __nv_bfloat16* v,
                                                const __nv_bfloat16* q_norm_w,
                                                const __nv_bfloat16* k_norm_w,
                                                const int* start,
                                                int rows, int q_heads, int kv_heads,
                                                int dim, int capacity,
                                                float theta, float eps) {
    if (q_heads != 16 || kv_heads != 2 || dim != 128 || rows > 8 ||
        *start < 0 || *start + rows > capacity)
        __trap();
    __shared__ float smem[32];
    const int heads = q_heads + 2 * kv_heads;
    const int row = blockIdx.x / heads;
    const int head = blockIdx.x % heads;
    if (row >= rows) return;
    const int gid = threadIdx.x;
    const int nthr = blockDim.x;
    const size_t base = (size_t)row * heads * dim + (size_t)head * dim;
    const int position = *start + row;

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
        for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffffu, ssq, o);
        if ((gid & 31) == 0) smem[gid >> 5] = ssq;
        __syncthreads();
        if (gid < 32) {
            int nw = nthr >> 5;
            float t = (gid < nw) ? smem[gid] : 0.0f;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
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
                const size_t dst = ((size_t)kv_head * capacity + position) * dim;
                k[dst + gid] = lo;
                k[dst + gid + dim / 2] = hi;
            }
        }
    } else {
        const int kv_head = head - q_heads - kv_heads;
        const size_t dst = ((size_t)kv_head * capacity + position) * dim;
#pragma unroll
        for (int d = gid; d < dim; d += nthr) v[dst + d] = qkv[base + d];
    }
}
