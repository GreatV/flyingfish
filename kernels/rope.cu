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
