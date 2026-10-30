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

// One mHC stream row batch: out holds rows*mult embedded rows, row r the
// mult-fold repeat of token ids[r/mult].
extern "C" __global__ void embed_streams(const __nv_bfloat16* w, const uint32_t* ids,
                                         __nv_bfloat16* y, int rows, int mult, int dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * mult * dim)
        y[i] = w[(uint64_t)ids[i / (mult * dim)] * dim + i % dim];
}

// out = bf16(bf16(rms(bf16(context*context_scale))*wn + bf16(table[id]*identity_scale))*combine_scale).
extern "C" __global__ void per_layer_input(const uint32_t* ids,
    const __nv_bfloat16* table, const __nv_bfloat16* context,
    const __nv_bfloat16* wn, __nv_bfloat16* out, int rows, int layers, int width,
    float identity_scale, float context_scale, float combine_scale, float eps) {
    if (width <= 0 || layers <= 0 || rows <= 0 || blockDim.x != 128) { __trap(); return; }
    const int row = blockIdx.x / layers;
    const int layer = blockIdx.x % layers;
    if (row >= rows) return;
    const size_t context_base = ((size_t)row * layers + layer) * width;
    const size_t out_base = ((size_t)layer * rows + row) * width;
    const size_t token = (size_t)ids[row] * layers * width + (size_t)layer * width;
    const __nv_bfloat16 is = __float2bfloat16_rn(identity_scale);
    const __nv_bfloat16 cs = __float2bfloat16_rn(context_scale);
    __shared__ float partial[4];
    float sum = 0.0f;
    for (int d = threadIdx.x; d < width; d += blockDim.x) {
        out[out_base + d] = __hmul(table[token + d], is);
        const float x = __bfloat162float(__hmul(context[context_base + d], cs));
        sum += x * x;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(FF_WARP_MASK, sum, offset);
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 4) {
        float x = partial[threadIdx.x];
        for (int offset = 2; offset > 0; offset >>= 1) x += __shfl_xor_sync(FF_WARP_MASK & 0xf, x, offset);
        if (threadIdx.x == 0) partial[0] = x;
    }
    __syncthreads();
    const float rstd = rsqrtf(partial[0] / width + eps);
    const __nv_bfloat16 combine = __float2bfloat16_rn(combine_scale);
    for (int d = threadIdx.x; d < width; d += blockDim.x) {
        const float x = __bfloat162float(__hmul(context[context_base + d], cs));
        const __nv_bfloat16 n = __float2bfloat16_rn(x * rstd * __bfloat162float(wn[d]));
        out[out_base + d] = __hmul(__hadd(n, out[out_base + d]), combine);
    }
}
