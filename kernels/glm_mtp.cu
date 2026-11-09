#include "gpu_compat.cuh"
#include <cuda_fp8.h>

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

__global__ void glm_mtp_accum_scaled(
    bf16* accum,
    const bf16* val,
    float scale,
    int count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float cur = __bfloat162float(accum[i]);
    float v = __bfloat162float(val[i]);
    accum[i] = __float2bfloat16_rn(cur + scale * v);
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

__global__ void glm_mtp_extract_dsa_value(
    const bf16* kv_expanded,
    bf16* attn_out,
    int heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = heads * head_dim;
    if (idx >= total) return;
    int h = idx / head_dim;
    int d = idx % head_dim;
    // kv_expanded has shape [heads, 2 * head_dim], value is the second half
    attn_out[idx] = kv_expanded[h * (2 * head_dim) + head_dim + d];
}

__global__ void glm_mtp_dequant_fp8_block128(
    const __nv_fp8_e4m3* codes,
    const float* scale_inv,
    bf16* out,
    int rows,
    int cols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || col >= cols) return;

    int scale_cols = (cols + 127) / 128;
    int scale_row = row / 128;
    int scale_col = col / 128;
    float scale = scale_inv[scale_row * scale_cols + scale_col];

    float val = (float)codes[row * cols + col];
    out[row * cols + col] = __float2bfloat16_rn(val * scale);
}

}
