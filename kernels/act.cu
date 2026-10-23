#include <cuda_bf16.h>

extern "C" __global__ void silu_mul(const __nv_bfloat16* gu, __nv_bfloat16* y,
                                   int rows, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * dim) return;
    int row = i / dim;
    int d = i % dim;
    float g = __bfloat162float(gu[row * 2 * dim + d]);
  float u = __bfloat162float(gu[row * 2 * dim + dim + d]);
  y[i] = __float2bfloat16(g / (1.0f + expf(-g)) * u);
}

// Gate and up are dense [rows,dim] regions; y is disjoint from both inputs.
extern "C" __global__ void silu_mul_split(const __nv_bfloat16* gate,
                                         const __nv_bfloat16* up,
                                         __nv_bfloat16* y, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float g = __bfloat162float(gate[i]);
    const float u = __bfloat162float(up[i]);
    y[i] = __float2bfloat16_rn(g / (1.0f + expf(-g)) * u);
}
