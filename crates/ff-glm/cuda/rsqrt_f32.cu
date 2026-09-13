#include <cuda_runtime.h>

// Keep native rsqrtf distinct from sqrt followed by division: these are not
// bitwise equivalent. Surrounding RMSNorm operations remain separate kernels.
extern "C" __global__ void glm_rsqrt_f32(
    int count, const float* __restrict__ input, float* __restrict__ output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) output[index] = rsqrtf(input[index]);
}
