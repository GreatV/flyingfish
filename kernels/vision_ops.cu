#include "gpu_compat.cuh"
#include <stdint.h>
#include <math.h>

__device__ __forceinline__ bool vision_flat_launch() {
    return blockDim.x == 256 && blockDim.y == 1 && blockDim.z == 1 &&
           gridDim.y == 1 && gridDim.z == 1;
}

// NaViT pack `[channels, side, rows*side]` (order c, ky, patch, kx) -> per-patch rows
// `[rows, channels*side*side]` in (c, ky, kx) order, rounding each value to BF16.
extern "C" __global__ void vision_patch_rows(
    const float* input, __nv_bfloat16* output,
    uint32_t rows, uint32_t channels, uint32_t side) {
    if (!vision_flat_launch() || !input || !output || !rows || !channels || !side) { __trap(); return; }
    uint32_t width = channels * side * side;
    uint64_t count = uint64_t(rows) * width;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        uint32_t patch = i / width, j = i % width;
        uint32_t c = j / (side * side), r = j % (side * side), ky = r / side, kx = r % side;
        uint64_t src = (uint64_t(c * side + ky)) * rows * side + uint64_t(patch) * side + kx;
        float x = input[src];
        __nv_bfloat16 y = __float2bfloat16_rn(x);
        if (!isfinite(x) || !isfinite(__bfloat162float(y))) { __trap(); return; }
        output[i] = y;
    }
}

// Three token-major `[rows, hq*d]` projections -> head-major Q, padded head-major K and V.
// K and V both land at the image's padded key offsets; Q stays at the real-row offsets.
extern "C" __global__ void vision_pack_split(
    const __nv_bfloat16* q_in, const __nv_bfloat16* k_in, const __nv_bfloat16* v_in,
    __nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* v,
    uint32_t rows, uint32_t padded, uint32_t first, uint32_t key_first, uint32_t n,
    uint32_t hidden, uint32_t hq, uint32_t d) {
    if (!vision_flat_launch() || !q_in || !k_in || !v_in || !q || !k || !v || !n ||
        !hidden || !hq || !d || hidden != hq * d ||
        uint64_t(first) + n > rows || uint64_t(key_first) + n > padded) { __trap(); return; }
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < uint64_t(n) * hidden;
         i += uint64_t(gridDim.x) * 256) {
        uint32_t r = i / hidden, head = (i % hidden) / d, j = i % d;
        uint64_t src = uint64_t(first + r) * hidden + head * d + j;
        q[(uint64_t(head) * rows + first + r) * d + j] = q_in[src];
        k[(uint64_t(head) * padded + key_first + r) * d + j] = k_in[src];
        v[(uint64_t(head) * padded + key_first + r) * d + j] = v_in[src];
    }
}

// Row gather: gathered[r] = input[indices[r]] over `width`-wide BF16 rows.
extern "C" __global__ void vision_row_gather(
    const __nv_bfloat16* input, const int32_t* indices, __nv_bfloat16* gathered,
    uint32_t rows, uint32_t in_rows, uint32_t width) {
    if (!vision_flat_launch() || !input || !indices || !gathered || !rows || !width) { __trap(); return; }
    uint64_t count = uint64_t(rows) * width;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        int32_t source = indices[i / width];
        if (source < 0 || uint32_t(source) >= in_rows) { __trap(); return; }
        gathered[i] = input[uint64_t(source) * width + i % width];
    }
}

// Mean of each consecutive `k`-row group over `width` BF16 columns: f32 sum, one BF16 round.
extern "C" __global__ void vision_mean_rows(
    const __nv_bfloat16* input, __nv_bfloat16* out, uint32_t rows, uint32_t width, uint32_t k) {
    if (!vision_flat_launch() || !input || !out || !rows || !width || !k || rows % k) { __trap(); return; }
    uint64_t count = uint64_t(rows / k) * width;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        uint64_t group = i / width, column = i % width;
        float sum = 0.0f;
        for (uint32_t j = 0; j < k; ++j) sum += __bfloat162float(input[(group * k + j) * width + column]);
        out[i] = __float2bfloat16_rn(sum / float(k));
    }
}

// Learned nearest-bucket position add: output[i] = patches[i] + table[indices[row] * width + column].
extern "C" __global__ void vision_bucket_position_add(
    const __nv_bfloat16* table, const int32_t* indices, const __nv_bfloat16* patches,
    __nv_bfloat16* output, uint32_t rows, uint32_t table_rows, uint32_t width) {
    if (!vision_flat_launch() || !table || !indices || !patches || !output || !rows || !width) { __trap(); return; }
    uint64_t count = uint64_t(rows) * width;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        int32_t index = indices[i / width];
        if (index < 0 || uint32_t(index) >= table_rows) { __trap(); return; }
        output[i] = __hadd(patches[i], table[uint64_t(index) * width + i % width]);
    }
}

extern "C" __global__ void vision_upcast_f32(
    const __nv_bfloat16* input, float* output, uint32_t count) {
    if (!vision_flat_launch() || !input || !output || !count) { __trap(); return; }
    for (uint32_t i = blockIdx.x * 256 + threadIdx.x; i < count; i += gridDim.x * 256) {
        output[i] = __bfloat162float(input[i]);
    }
}

extern "C" __global__ void vision_bias_f32(
    const float* input, const float* bias, float* output, uint32_t rows, uint32_t width) {
    if (!vision_flat_launch() || !input || !bias || !output || !rows || !width) { __trap(); return; }
    uint32_t count = rows * width;
    for (uint32_t i = blockIdx.x * 256 + threadIdx.x; i < count; i += gridDim.x * 256) {
        output[i] = __fadd_rn(input[i], bias[i % width]);
    }
}

extern "C" __global__ void vision_bias_gelu_f32(
    const float* input, const float* bias, float* output, uint32_t rows, uint32_t width) {
    if (!vision_flat_launch() || !input || !bias || !output || !rows || !width) { __trap(); return; }
    uint32_t count = rows * width;
    const float scale = 0.70710678118654752440f;
    for (uint32_t i = blockIdx.x * 256 + threadIdx.x; i < count; i += gridDim.x * 256) {
        float x = __fadd_rn(input[i], bias[i % width]);
        output[i] = __fmul_rn(0.5f * x, __fadd_rn(1.0f, erff(__fmul_rn(x, scale))));
    }
}

// One FP32 RMS normalization and scale, with independent FP32 and BF16 outputs.
extern "C" __global__ void vision_rms_scale_f32(
    const float* input, const float* scale, float* output_f32,
    __nv_bfloat16* output_bf16, uint32_t width, float eps) {
    if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.y != 1 || gridDim.z != 1 || !input || !scale ||
        !output_f32 || !output_bf16 || !width || !(eps > 0.0f)) { __trap(); return; }
    uint32_t row = blockIdx.x;
    __shared__ float partial[8];
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < width; i += 256) {
        float x = input[uint64_t(row) * width + i];
        sum = __fadd_rn(sum, __fmul_rn(x, x));
    }
    for (int delta = 16; delta; delta >>= 1)
        sum = __fadd_rn(sum, __shfl_down_sync(FF_WARP_MASK, sum, delta));
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < 8 ? partial[threadIdx.x] : 0.0f;
        for (int delta = 16; delta; delta >>= 1)
            sum = __fadd_rn(sum, __shfl_down_sync(FF_WARP_MASK, sum, delta));
        if (threadIdx.x == 0) partial[0] = rsqrtf(__fadd_rn(sum / float(width), eps));
    }
    __syncthreads();
    float rstd = partial[0];
    for (uint32_t i = threadIdx.x; i < width; i += 256) {
        float normalized = __fmul_rn(input[uint64_t(row) * width + i], rstd);
        float scaled = __fmul_rn(normalized, *scale);
        output_f32[uint64_t(row) * width + i] = scaled;
        output_bf16[uint64_t(row) * width + i] = __float2bfloat16_rn(scaled);
    }
}

extern "C" __global__ void vision_add_row_silu_f32(
    const float* input, const float* row, float* output, uint32_t count, uint32_t width) {
    if (!vision_flat_launch() || !input || !row || !output || !count || !width ||
        count % width) { __trap(); return; }
    for (uint32_t i = blockIdx.x * 256 + threadIdx.x; i < count; i += gridDim.x * 256) {
        float x = __fadd_rn(input[i], row[i % width]);
        output[i] = __fdiv_rn(x, __fadd_rn(1.0f, expf(-x)));
    }
}

extern "C" __global__ void vision_gate_round_bf16(
    const float* score, __nv_bfloat16* gate, uint32_t count) {
    if (!vision_flat_launch() || !score || !gate || !count) { __trap(); return; }
    for (uint32_t i = blockIdx.x * 256 + threadIdx.x; i < count; i += gridDim.x * 256) {
        float x = score[i];
        float bounded = fminf(fmaxf(x, -8.0f), 8.0f);
        float surrogate_score = __fadd_rn(x, __fsub_rn(bounded, x));
        float raw = __fmul_rn(2.0f, __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-x))));
        float surrogate = __fmul_rn(
            2.0f, __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-surrogate_score))));
        gate[i] = __float2bfloat16_rn(__fsub_rn(__fadd_rn(raw, surrogate), surrogate));
    }
}

// BF16 sum and denominator, followed by one BF16 division per instruction column.
extern "C" __global__ void vision_masked_mean_bf16(
    const __nv_bfloat16* states, __nv_bfloat16* output,
    int begin, int end, int width) {
    if (!vision_flat_launch() || !states || !output || begin < 0 || end <= begin ||
        width != 1536) { __trap(); return; }
    for (int column = blockIdx.x * 256 + threadIdx.x; column < width;
         column += gridDim.x * 256) {
        float sum = 0.0f;
        for (int row = begin; row < end; ++row)
            sum = __fadd_rn(sum, __bfloat162float(states[uint64_t(row) * width + column]));
        __nv_bfloat16 total = __float2bfloat16_rn(sum);
        __nv_bfloat16 denominator = __float2bfloat16_rn(float(end - begin));
        output[column] = __float2bfloat16_rn(
            __fdiv_rn(__bfloat162float(total), __bfloat162float(denominator)));
    }
}

// alpha and both products round to BF16 before the BF16 residual addition.
extern "C" __global__ void vision_anchor_reinject_bf16(
    __nv_bfloat16* states, const __nv_bfloat16* anchor,
    const __nv_bfloat16* gates, const float* raw_alpha,
    int visual_rows, int width, int cycle) {
    if (!vision_flat_launch() || !states || !anchor || !gates || !raw_alpha ||
        visual_rows < 1 || width != 1536 || cycle < 0 || cycle > 1) { __trap(); return; }
    float alpha_f32 = __fmul_rn(
        0.5f, __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-raw_alpha[cycle]))));
    __nv_bfloat16 alpha = __float2bfloat16_rn(alpha_f32);
    uint32_t count = uint32_t(visual_rows) * width;
    for (uint32_t i = blockIdx.x * 256 + threadIdx.x; i < count; i += gridDim.x * 256) {
#if defined(__HIP__)
        __nv_bfloat16 scaled_gate = __hmul_rn(alpha, gates[i / width]);
        __nv_bfloat16 delta = __hmul_rn(scaled_gate, anchor[i]);
        states[i] = __hadd_rn(states[i], delta);
#else
        __nv_bfloat16 scaled_gate = __hmul(alpha, gates[i / width]);
        __nv_bfloat16 delta = __hmul(scaled_gate, anchor[i]);
        states[i] = __hadd(states[i], delta);
#endif
    }
}
