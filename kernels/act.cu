#include <cuda_bf16.h>

extern "C" __global__ void silu_mul(const __nv_bfloat16* gu, __nv_bfloat16* y,
                                   int rows, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * dim) return;
    int row = i / dim;
    int d = i % dim;
    float g = __bfloat162float(gu[row * 2 * dim + d]);
    __nv_bfloat16 s = __float2bfloat16(g / (1.0f + expf(-g)));
    y[i] = __hmul(s, gu[row * 2 * dim + dim + d]);
}
