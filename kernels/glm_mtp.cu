#include "gpu_compat.cuh"

using bf16 = __nv_bfloat16;

extern "C" {

__global__ void glm_mtp_concat_eh(
    const bf16* enorm,
    const bf16* hnorm,
    bf16* out,
    int dim,
    int swap
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    if (swap == 0) {
        out[i] = enorm[i];
        out[dim + i] = hnorm[i];
    } else {
        out[i] = hnorm[i];
        out[dim + i] = enorm[i];
    }
}

__global__ void glm_mtp_add(
    bf16* x,
    const bf16* res,
    int count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float a = __bfloat162float(x[i]);
    float b = __bfloat162float(res[i]);
    x[i] = __float2bfloat16_rn(a + b);
}

}
