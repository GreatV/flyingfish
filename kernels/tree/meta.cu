#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <limits.h>

namespace tree_meta {
using bf16 = __nv_bfloat16;
__device__ __forceinline__ void rows(int n) {
    if (n < 1 || n > 64) __trap();
}

__device__ __forceinline__ void extent(int start, int n, int capacity) {
    rows(n);
    if (start < 0 || capacity <= 0 || start > capacity - n) __trap();
}
__device__ __forceinline__ int input_row(const int* path, int row, int count, int n) {
    rows(n);
    if (count < 1 || count > n || path[0] != 0) __trap();
    const int src = path[row];
    if (src < row || src >= n || (row && path[row - 1] >= src)) __trap();
    return src;
}
}

extern "C" __global__ void tree_prepare(const uint32_t* tokens, const int* depth,
    const int* rows, const int* expected, const int* prefix, int* snapshot,
    uint32_t* ids, int* positions, int* slots, int budget, int capacity) {
    const int n = *rows, start = *prefix;
    tree_meta::rows(n);
    tree_meta::extent(start,budget,capacity);
    if (n > budget || start != *expected) __trap();
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (!row) *snapshot = start;
    if (row >= budget) return;
    if (row < n) {
        if (depth[row] < 0 || depth[row] > 7) __trap();
        ids[row] = tokens[row];
        positions[row] = start + depth[row];
    } else {
        ids[row] = 0;
        positions[row] = 0;
    }
    slots[row] = start + row;
}

extern "C" __global__ void tree_qkv_pad(__nv_bfloat16* qkv,
    const int* rows, int budget) {
    const int n = *rows;
    tree_meta::rows(n);
    if (budget < n || budget > 64) __trap();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (budget - n) * 2560) qkv[n*2560+i] = __float2bfloat16(0.f);
}

extern "C" __global__ void tree_kv_pad(__nv_bfloat16* k, __nv_bfloat16* v,
    const int* prefix, const int* rows, int budget, int capacity) {
    const int n = *rows, start = *prefix;
    tree_meta::extent(start,budget,capacity);
    if (n < 1 || n > budget) __trap();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (budget - n) * 256) return;
    const int row = i / 256, head = (i / 128) % 2, d = i % 128;
    const size_t dst = (size_t(head)*capacity+start+n+row)*128+d;
    k[dst] = __float2bfloat16(0.f);
    v[dst] = __float2bfloat16(0.f);
}

extern "C" __global__ void tree_logits(const __nv_bfloat16* source,
    __nv_bfloat16* destination, const int* row, const int* rows, int width) {
    tree_meta::rows(*rows);
    if (*row < 0 || *row >= *rows || width < 1) __trap();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) destination[i] = source[size_t(*row)*width+i];
}

extern "C" __global__ void tree_rope(__nv_bfloat16* qkv, const int* positions,
    const int* rows, int q_heads, int kv_heads, int dim, float theta) {
    const int n = *rows;
    tree_meta::rows(n);
    if (q_heads != 16 || kv_heads != 2 || dim != 128 || !isfinite(theta) || theta <= 0.f) __trap();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 18 * 64) return;
    const int d = i % 64, head = (i / 64) % 18, row = i / (18 * 64);
    if (positions[row] < 0) __trap();
    const int off = row * 2560 + head * 128 + d;
    const float angle = float(positions[row]) * (1.f / powf(theta, float(2*d)/128));
    const __nv_bfloat16 c = __float2bfloat16(cosf(angle)), s = __float2bfloat16(sinf(angle));
    const __nv_bfloat16 a = qkv[off], b = qkv[off + 64];
    qkv[off] = __hsub(__hmul(a,c),__hmul(b,s));
    qkv[off+64] = __hadd(__hmul(b,c),__hmul(a,s));
}

extern "C" __global__ void tree_kv_write(const __nv_bfloat16* qkv, __nv_bfloat16* k,
    __nv_bfloat16* v, const int* prefix, const int* rows, int capacity) {
    const int n = *rows, start = *prefix;
    tree_meta::extent(start,n,capacity);
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 256) return;
    const int d = i % 128, head = (i / 128) % 2, row = i / 256;
    const size_t dst = (size_t(head) * capacity + start + row) * 128 + d;
    k[dst] = qkv[row*2560 + 2048 + head*128 + d];
    v[dst] = qkv[row*2560 + 2304 + head*128 + d];
}

extern "C" __global__ void tree_kv_gather(const uint64_t* k, const uint64_t* v,
    __nv_bfloat16* compact, const int* path, const int* count, const int* rows,
    const int* prefix, int capacity, int layers) {
    const int n = *rows, take = *count, start = *prefix;
    tree_meta::extent(start,n,capacity);
    if (take < 1 || take > n || layers < 1 || layers > 42) __trap();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= layers * 2 * 2 * take * 128) return;
    const int d = i % 128, row = (i / 128) % take, head = (i / (128*take)) % 2;
    const int kind = (i / (256*take)) % 2, layer = i / (512*take);
    const int src = tree_meta::input_row(path,row,take,n);
    const auto* data = reinterpret_cast<const __nv_bfloat16*>(kind ? v[layer] : k[layer]);
    if (!data) __trap();
    compact[i] = data[(size_t(head)*capacity + start + src)*128+d];
}

extern "C" __global__ void tree_kv_scatter(const __nv_bfloat16* compact,
    const uint64_t* k, const uint64_t* v, const int* count, const int* prefix,
    int capacity, int layers) {
    const int take = *count, start = *prefix;
    tree_meta::extent(start,take,capacity);
    if (layers < 1 || layers > 42) __trap();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= layers * 2 * 2 * take * 128) return;
    const int d = i % 128, row = (i / 128) % take, head = (i / (128*take)) % 2;
    const int kind = (i / (256*take)) % 2, layer = i / (512*take);
    auto* data = reinterpret_cast<__nv_bfloat16*>(kind ? v[layer] : k[layer]);
    if (!data) __trap();
    data[(size_t(head)*capacity+start+row)*128+d] = compact[i];
}

extern "C" __global__ void tree_hidden_gather(const __nv_bfloat16* hidden,
    __nv_bfloat16* compact, const int* path, const int* count, const int* rows, int width) {
    const int n = *rows, take = *count;
    tree_meta::rows(n);
    if (take < 1 || take > n || (width != 2048 && width != 10240)) __trap();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= take * width) return;
    const int src = tree_meta::input_row(path,i/width,take,n);
    compact[i] = hidden[size_t(src)*width+i%width];
}

extern "C" __global__ void tree_hidden_scatter(const __nv_bfloat16* compact,
    __nv_bfloat16* hidden, const int* count, const int* start, int capacity, int width) {
    const int n = *count, base = *start;
    tree_meta::extent(base,n,capacity);
    if (width != 2048 && width != 10240) __trap();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * width) return;
    hidden[size_t(base)*width+i] = compact[i];
}
