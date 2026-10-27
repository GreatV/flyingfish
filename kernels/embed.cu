#include <cuda_bf16.h>
#include <stdint.h>

extern "C" __global__ void embed(const __nv_bfloat16* w, const uint32_t* ids,
                                __nv_bfloat16* y, int rows, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * dim) y[i] = w[(uint64_t)ids[i / dim] * dim + i % dim];
}

extern "C" __global__ void embed_scale_broadcast(const __nv_bfloat16* w,
    const uint32_t* ids, const __nv_bfloat16* initial,
    __nv_bfloat16* embedded, __nv_bfloat16* broadcast, int rows, int dim, float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * dim) return;
    embedded[i] = __float2bfloat16(__bfloat162float(w[(uint64_t)ids[i / dim] * dim + i % dim]) * scale);
    broadcast[i] = initial[i % dim];
}
