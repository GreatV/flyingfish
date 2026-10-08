#include <cuda_bf16.h>
#include <stdint.h>

extern "C" __global__ void embed(const __nv_bfloat16* w, const uint32_t* ids,
                                __nv_bfloat16* y, int rows, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * dim) y[i] = w[(uint64_t)ids[i / dim] * dim + i % dim];
}
