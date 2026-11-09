#include "gpu_compat.cuh"
#include <stdint.h>

extern "C" __global__ void embed_code_broadcast(const __nv_bfloat16* table,
    const uint32_t* code, __nv_bfloat16* out, int offset, int vocab, int dim, int branches) {
    if (*code >= (unsigned)vocab || offset < 0) { __trap(); return; }
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < branches * dim)
        out[i] = table[(size_t)(*code + offset) * dim + i % dim];
}

// Semantic code is absolute; residual table is [codebooks*vocab, dim].
extern "C" __global__ void codebook_embedding_sum(
    const __nv_bfloat16* semantic, const __nv_bfloat16* residual,
    const uint32_t* codes, __nv_bfloat16* out, int semantic_begin,
    int semantic_count, int codebooks, int vocab, int dim, int branches, float scale) {
    if (codes[0] < (unsigned)semantic_begin ||
        codes[0] >= (unsigned)(semantic_begin + semantic_count)) { __trap(); return; }
    for (int book = 0; book < codebooks; ++book)
        if (codes[book + 1] >= (unsigned)vocab) { __trap(); return; }
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim * branches) return;
    const int channel = i % dim;
    float sum = 0.0f;
    for (int book = 0; book < codebooks; ++book)
        sum += __bfloat162float(residual[((size_t)book * vocab + codes[book + 1]) * dim + channel]);
    const float rounded = __bfloat162float(__float2bfloat16_rn(sum));
    const float combined = __bfloat162float(__float2bfloat16_rn(
        __bfloat162float(semantic[(size_t)codes[0] * dim + channel]) + rounded));
    out[i] = __float2bfloat16_rn(combined * scale);
}

extern "C" __global__ void embed(const __nv_bfloat16* w, const uint32_t* ids,
                                __nv_bfloat16* y, int rows, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * dim) y[i] = w[(uint64_t)ids[i / dim] * dim + i % dim];
}

// ids are [rows, books]. out[row, d] is one bf16 of the f32 sum of books rows.
extern "C" __global__ void embed_rows_sum(const __nv_bfloat16* table, const int32_t* ids,
                                          __nv_bfloat16* out, int rows, int books, int vocab,
                                          int dim) {
    if (rows < 0 || books < 1 || vocab < 1 || dim < 1 || !table || !ids || !out) {
        __trap();
        return;
    }
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * dim) return;
    const int row = i / dim;
    const int col = i % dim;
    float sum = 0.0f;
    for (int book = 0; book < books; ++book) {
        const int id = ids[(size_t)row * books + book];
        if ((unsigned)id >= (unsigned)vocab) {
            __trap();
            return;
        }
        sum += __bfloat162float(table[((size_t)id + (size_t)book * (unsigned)vocab) * dim + col]);
    }
    out[i] = __float2bfloat16_rn(sum);
}

extern "C" __global__ void embed_scale_broadcast(const __nv_bfloat16* w,
    const uint32_t* ids, const __nv_bfloat16* initial,
    __nv_bfloat16* embedded, __nv_bfloat16* broadcast, int rows, int dim, float scale,
    const __nv_bfloat16* visual, int visual_begin, int visual_rows) {
    if (broadcast && !initial) { __trap(); return; }
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * dim) return;
    const int row = i / dim;
    const __nv_bfloat16 value = row >= visual_begin && row < visual_begin + visual_rows
        ? visual[(row - visual_begin) * dim + i % dim]
        : w[(uint64_t)ids[row] * dim + i % dim];
    embedded[i] = __float2bfloat16(__bfloat162float(value) * scale);
    if (broadcast) broadcast[i] = initial[i % dim];
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
    if (width <= 256) {
        const int d0 = threadIdx.x;
        const int d1 = threadIdx.x + 128;
        __nv_bfloat16 id0 = __float2bfloat16_rn(0.0f);
        __nv_bfloat16 id1 = __float2bfloat16_rn(0.0f);
        float x0 = 0.0f;
        float x1 = 0.0f;
        if (d0 < width) {
            id0 = __hmul(table[token + d0], is);
            x0 = __bfloat162float(__hmul(context[context_base + d0], cs));
            sum += x0 * x0;
        }
        if (d1 < width) {
            id1 = __hmul(table[token + d1], is);
            x1 = __bfloat162float(__hmul(context[context_base + d1], cs));
            sum += x1 * x1;
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
        if (d0 < width) {
            const __nv_bfloat16 n = __float2bfloat16_rn(x0 * rstd * __bfloat162float(wn[d0]));
            out[out_base + d0] = __hmul(__hadd(n, id0), combine);
        }
        if (d1 < width) {
            const __nv_bfloat16 n = __float2bfloat16_rn(x1 * rstd * __bfloat162float(wn[d1]));
            out[out_base + d1] = __hmul(__hadd(n, id1), combine);
        }
        return;
    }
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

// out = bf16(rms(bf16(context*context_scale))*wn).
extern "C" __global__ void per_layer_projection(const __nv_bfloat16* context,
    const __nv_bfloat16* wn, __nv_bfloat16* out, int rows, int layers, int width,
    float context_scale, float eps) {
    if (width <= 0 || layers <= 0 || rows <= 0 || blockDim.x != 128) { __trap(); return; }
    const int row = blockIdx.x / layers;
    const int layer = blockIdx.x % layers;
    if (row >= rows) return;
    const size_t context_base = ((size_t)row * layers + layer) * width;
    const size_t out_base = ((size_t)layer * rows + row) * width;
    const __nv_bfloat16 cs = __float2bfloat16_rn(context_scale);
    __shared__ float partial[4];
    float sum = 0.0f;
    for (int d = threadIdx.x; d < width; d += blockDim.x) {
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
    for (int d = threadIdx.x; d < width; d += blockDim.x) {
        const float x = __bfloat162float(__hmul(context[context_base + d], cs));
        out[out_base + d] = __float2bfloat16_rn(x * rstd * __bfloat162float(wn[d]));
    }
}

// out accumulates valid_count * mean when accumulate is true, otherwise it stores mean.
extern "C" __global__ void mean_pool_l2(const __nv_bfloat16* input, const bool* mask,
    float* out, int rows, int width, bool accumulate) {
    if (rows <= 0 || width <= 0 || blockDim.x != 128) { __trap(); return; }
    unsigned valid = 0;
    for (int row = 0; row < rows; ++row) valid += mask[row] ? 1u : 0u;
    if (valid == 0) { __trap(); return; }
    for (int d = threadIdx.x; d < width; d += blockDim.x) {
        float value = 0.0f;
        for (int row = 0; row < rows; ++row) {
            if (mask[row]) value += __bfloat162float(input[row * width + d]);
        }
        value /= (float)valid;
        if (accumulate) out[d] += value * (float)valid;
        else out[d] = value;
    }
}

// out = out / sqrt(sum(out^2) + 1e-12).
extern "C" __global__ void l2_normalize(float* out, int width) {
    if (width <= 0 || blockDim.x != 128) { __trap(); return; }
    __shared__ float partial[4];
    float sum = 0.0f;
    for (int d = threadIdx.x; d < width; d += blockDim.x) sum += out[d] * out[d];
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
    const float scale = rsqrtf(partial[0] + 1e-12f);
    for (int d = threadIdx.x; d < width; d += blockDim.x) out[d] *= scale;
}
