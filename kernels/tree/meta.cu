#include "../gpu_compat.cuh"
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

extern "C" __global__ void tree_logits(const __nv_bfloat16* source,
    __nv_bfloat16* destination, const int* row, const int* rows, int width) {
    tree_meta::rows(*rows);
    if (*row < 0 || *row >= *rows || width < 1) __trap();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) destination[i] = source[size_t(*row)*width+i];
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

