#include "gpu_compat.cuh"
#include <stdint.h>

extern "C" __global__ void embed(const __nv_bfloat16* w, const uint32_t* ids,
                                __nv_bfloat16* y, int rows, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * dim) y[i] = w[(uint64_t)ids[i / dim] * dim + i % dim];
}

extern "C" __global__ void embed_scale_broadcast(const __nv_bfloat16* w,
    const uint32_t* ids, const __nv_bfloat16* initial,
    __nv_bfloat16* embedded, __nv_bfloat16* broadcast, int rows, int dim, float scale,
    const __nv_bfloat16* visual, int visual_begin, int visual_rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * dim) return;
    const int row = i / dim;
    const __nv_bfloat16 value = row >= visual_begin && row < visual_begin + visual_rows
        ? visual[(row - visual_begin) * dim + i % dim]
        : w[(uint64_t)ids[row] * dim + i % dim];
    embedded[i] = __float2bfloat16(__bfloat162float(value) * scale);
    broadcast[i] = initial[i % dim];
}
