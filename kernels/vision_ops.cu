#include "gpu_compat.cuh"
#include <stdint.h>
#include <math.h>

// NHWC 2D im2col: [height, width, channels] -> [out_y_count * out_w, channels * kh * kw] in (c, ky, kx) order.
// `mask` may be null; when non-null, rows with mask[in_y] == 0 read zero.
extern "C" __global__ void masked_strided_patch_bf16(
    const __nv_bfloat16* input, const uint8_t* mask, __nv_bfloat16* output,
    uint32_t height, uint32_t width, uint32_t channels,
    uint32_t kh, uint32_t kw, uint32_t sh, uint32_t sw,
    uint32_t pad_top, uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right,
    uint32_t out_y_start, uint32_t out_y_count) {
    if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.y != 1 || gridDim.z != 1 || !input || !output || input == output ||
        !height || !width || !channels || !kh || !kw || !sh || !sw || !out_y_count) {
        __trap(); return;
    }
    const uint64_t padded_h = uint64_t(height) + pad_top + pad_bottom;
    const uint64_t padded_w = uint64_t(width) + pad_left + pad_right;
    if (padded_h < kh || padded_w < kw) { __trap(); return; }
    const uint32_t out_h = uint32_t((padded_h - kh) / sh + 1);
    const uint32_t out_w = uint32_t((padded_w - kw) / sw + 1);
    if (uint64_t(out_y_start) + out_y_count > out_h) { __trap(); return; }
    const uint32_t k_area = kh * kw;
    const uint32_t row_width = channels * k_area;
    const uint64_t count = uint64_t(out_y_count) * out_w * row_width;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        const uint64_t row = i / row_width;
        const uint32_t k = uint32_t(i % row_width);
        const uint32_t c = k / k_area;
        const uint32_t rem = k % k_area;
        const uint32_t ky = rem / kw;
        const uint32_t kx = rem % kw;
        const uint32_t oy = out_y_start + uint32_t(row / out_w);
        const uint32_t ox = uint32_t(row % out_w);
        const int in_y = int(oy * sh + ky) - int(pad_top);
        const int in_x = int(ox * sw + kx) - int(pad_left);
        const bool valid = in_y >= 0 && in_y < int(height) &&
                           in_x >= 0 && in_x < int(width) &&
                           (!mask || mask[in_y]);
        output[i] = valid
            ? input[(uint64_t(in_y) * width + uint64_t(in_x)) * channels + c]
            : __float2bfloat16_rn(0.0f);
    }
}

// mode 0: ReLU; mode 1: SiLU; mode 2: GLU; mode 3: softplus.
extern "C" __global__ void activation_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output, uint32_t rows, uint32_t width, int mode) {
    if (blockDim.x != 256 || !input || !output || !rows || !width || mode < 0 || mode > 3) {
        __trap(); return;
    }
    const uint64_t count = uint64_t(rows) * width;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        const uint64_t offset = mode == 2 ? i / width * (2 * width) + i % width : i;
        const float x = __bfloat162float(input[offset]);
        const float gate = mode == 2 ? __bfloat162float(input[offset + width]) : x;
        const float y = mode == 0 ? fmaxf(x, 0.0f) : mode == 3 ?
            (x > 20.0f ? x : log1pf(expf(x))) : x / (1.0f + expf(-gate));
        output[i] = __float2bfloat16_rn(y);
    }
}

// Native weights[channel,1,kernel]; BF16 input/output[time,channel], zero left padding.
extern "C" __global__ void causal_depthwise_conv_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight, __nv_bfloat16* output,
    uint32_t rows, uint32_t channels, uint32_t kernel) {
    if (blockDim.x != 256 || !input || !weight || !output || input == output ||
        !rows || !channels || !kernel) { __trap(); return; }
    const uint64_t count = uint64_t(rows) * channels;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        const uint32_t row = i / channels, c = i % channels;
        float sum = 0.0f;
        for (uint32_t k = 0; k < kernel; ++k) {
            const int t = int(row) + int(k) - int(kernel) + 1;
            if (t >= 0) sum = __fmaf_rn(__bfloat162float(input[uint64_t(t) * channels + c]),
                __bfloat162float(weight[uint64_t(c) * kernel + k]), sum);
        }
        output[i] = __float2bfloat16_rn(sum);
    }
}

__device__ __forceinline__ bool vision_flat_launch() {
    return blockDim.x == 256 && blockDim.y == 1 && blockDim.z == 1 &&
           gridDim.y == 1 && gridDim.z == 1;
}

extern "C" __global__ void vision_affine_cast_bf16(
    const float* input, __nv_bfloat16* output, uint64_t count, float shift, float scale) {
    if (!vision_flat_launch() || !input || !output || !count ||
        !isfinite(shift) || !isfinite(scale)) { __trap(); return; }
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        const float value = __fmul_rn(__fadd_rn(input[i], shift), scale);
        const __nv_bfloat16 rounded = __float2bfloat16_rn(value);
        if (!isfinite(value) || !isfinite(__bfloat162float(rounded))) { __trap(); return; }
        output[i] = rounded;
    }
}

extern "C" __global__ void vision_clamp_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output, uint64_t count,
    const __nv_bfloat16* lower, const __nv_bfloat16* upper) {
    if (!vision_flat_launch() || !input || !output || !count || !lower || !upper) { __trap(); return; }
    const float lo = __bfloat162float(*lower), hi = __bfloat162float(*upper);
    if (!isfinite(lo) || !isfinite(hi) || lo > hi) { __trap(); return; }
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        const float value = __bfloat162float(input[i]);
        if (!isfinite(value)) { __trap(); return; }
        output[i] = __float2bfloat16_rn(fminf(hi, fmaxf(lo, value)));
    }
}

// The two axis embeddings add in BF16 before the patch residual addition.
extern "C" __global__ void vision_axis_position_add_bf16(
    const __nv_bfloat16* patches, const __nv_bfloat16* table, __nv_bfloat16* output,
    uint32_t grid_h, uint32_t grid_w, uint32_t width, uint32_t positions,
    uint64_t axis_stride, uint64_t position_stride) {
    if (!vision_flat_launch() || !patches || !table || !output || !grid_h || !grid_w ||
        !width || grid_h > positions || grid_w > positions ||
        !((axis_stride == uint64_t(positions) * width && position_stride == width) ||
          (axis_stride == width && position_stride == 2 * uint64_t(width)))) { __trap(); return; }
    const uint64_t count = uint64_t(grid_h) * grid_w * width;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        const uint64_t row = i / width, column = i % width;
        const __nv_bfloat16 position = __hadd(table[(row % grid_w) * position_stride + column],
            table[axis_stride + row / grid_w * position_stride + column]);
        output[i] = __hadd(patches[i], position);
    }
}

// Spatial mean rounds to BF16 before F32 scale and optional F32 standardization.
extern "C" __global__ void vision_spatial_mean_scale_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    const __nv_bfloat16* bias, const __nv_bfloat16* multiplier,
    uint32_t grid_h, uint32_t grid_w, uint32_t width, uint32_t merge, float scale) {
    if (!vision_flat_launch() || !input || !output || !grid_h || !grid_w || !width || !merge ||
        grid_h % merge || grid_w % merge || bool(bias) != bool(multiplier) || !isfinite(scale)) { __trap(); return; }
    const uint64_t columns = grid_w / merge, rows = grid_h / merge;
    const float weight = 1.0f / (float(merge) * float(merge));
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < rows * columns * width;
         i += uint64_t(gridDim.x) * 256) {
        const uint64_t group = i / width, channel = i % width;
        const uint64_t first = (group / columns * merge * grid_w + group % columns * merge) * width + channel;
        float sum = 0.0f;
        for (uint32_t y = 0; y < merge; ++y)
            for (uint32_t x = 0; x < merge; ++x)
                sum = __fmaf_rn(weight, __bfloat162float(input[first + (uint64_t(y) * grid_w + x) * width]), sum);
        float value = __fmul_rn(__bfloat162float(__float2bfloat16_rn(sum)), scale);
        if (bias) value = __fmul_rn(__fsub_rn(value, __bfloat162float(bias[channel])),
            __bfloat162float(multiplier[channel]));
        output[i] = __float2bfloat16_rn(value);
    }
}

// NaViT pack `[channels, side, rows*side]` (order c, ky, patch, kx) -> per-patch rows
// `[rows, channels*side*side]` in (c, ky, kx) order, rounding each value to BF16.
template<typename T>
__device__ __forceinline__ void patch_rows(
    const float* input, T* output,
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
        T y = T(x);
        if (!isfinite(x) || !isfinite(float(y))) { __trap(); return; }
        output[i] = y;
    }
}

extern "C" __global__ void vision_patch_rows(const float* input, __nv_bfloat16* output,
    uint32_t rows, uint32_t channels, uint32_t side) {
    patch_rows(input, output, rows, channels, side);
}
extern "C" __global__ void vision_patch_rows_f32(const float* input, float* output,
    uint32_t rows, uint32_t channels, uint32_t side) {
    patch_rows(input, output, rows, channels, side);
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
    if (!vision_flat_launch() || !input || !output || !rows || !width) { __trap(); return; }
    uint32_t count = rows * width;
    const float scale = 0.70710678118654752440f;
    for (uint32_t i = blockIdx.x * 256 + threadIdx.x; i < count; i += gridDim.x * 256) {
        float x = bias ? __fadd_rn(input[i], bias[i % width]) : input[i];
        output[i] = __fmul_rn(0.5f * x, __fadd_rn(1.0f, erff(__fmul_rn(x, scale))));
    }
}

// One FP32 RMS normalization per row times scale[i * scale_stride] (stride 0: one scalar, 1: per element), with a
// BF16 output and an optional (null-skipped) FP32 output.
extern "C" __global__ void vision_rms_scale_f32(
    const float* input, const float* scale, uint32_t scale_stride, float* output_f32,
    __nv_bfloat16* output_bf16, uint32_t width, float eps) {
    if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.y != 1 || gridDim.z != 1 || !input || !scale || scale_stride > 1 ||
        !output_bf16 || !width || !(eps > 0.0f)) { __trap(); return; }
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
        float scaled = __fmul_rn(normalized, scale[i * scale_stride]);
        if (output_f32) output_f32[uint64_t(row) * width + i] = scaled;
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

// residual[i] = residual[i] + f32(delta[i]) * scale[i % width], rounded after the product and after the sum.
extern "C" __global__ void scaled_residual_f32(
    float* residual, const __nv_bfloat16* delta, const float* scale, uint64_t count, uint32_t width) {
    if (!vision_flat_launch() || !residual || !delta || !scale || !count || !width ||
        count % width) { __trap(); return; }
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256)
        residual[i] = __fadd_rn(residual[i], __fmul_rn(__bfloat162float(delta[i]), scale[i % width]));
}

// Token rows [frames * height * width, channels * pt * p * p] -> pixels [channels, frames * pt, height * p, width * p].
extern "C" __global__ void unpatchify_bf16(
    const __nv_bfloat16* rows, __nv_bfloat16* pixels, uint32_t frames, uint32_t height,
    uint32_t width, uint32_t channels, uint32_t pt, uint32_t p) {
    if (!vision_flat_launch() || !rows || !pixels || !frames || !height || !width || !channels ||
        !pt || !p) { __trap(); return; }
    const uint64_t out_t = uint64_t(frames) * pt, out_h = uint64_t(height) * p,
                   out_w = uint64_t(width) * p;
    const uint64_t count = uint64_t(channels) * out_t * out_h * out_w;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        const uint64_t x = i % out_w, y = (i / out_w) % out_h, t = (i / (out_w * out_h)) % out_t,
                       c = i / (out_w * out_h * out_t);
        const uint64_t token = ((t / pt) * height + y / p) * width + x / p;
        const uint64_t column = ((c * pt + t % pt) * p + y % p) * p + x % p;
        pixels[i] = rows[token * (uint64_t(channels) * pt * p * p) + column];
    }
}

// Weight w = position * (1 / extent) of a linear blend.
__device__ __forceinline__ float blend_weight(uint32_t position, uint32_t extent) {
    return __fmul_rn(float(position), __frcp_rn(float(extent)));
}

// bf16(bf16(a * bf16(1 - w)) + bf16(b * w)) with w = bf16(blend_weight): one element of a linear tile blend.
__device__ __forceinline__ __nv_bfloat16 linear_blend(
    __nv_bfloat16 a, __nv_bfloat16 b, uint32_t position, uint32_t extent) {
    const float w = __bfloat162float(__float2bfloat16_rn(blend_weight(position, extent)));
    const float wa = __bfloat162float(__float2bfloat16_rn(__fsub_rn(1.0f, w)));
    const float pa = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__bfloat162float(a), wa)));
    const float pb = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__bfloat162float(b), w)));
    return __float2bfloat16_rn(__fadd_rn(pa, pb));
}

// a * (1 - w) + b * w in F32, each operation rounded, with w = blend_weight.
__device__ __forceinline__ float linear_blend(float a, float b, uint32_t position, uint32_t extent) {
    const float w = blend_weight(position, extent);
    return __fadd_rn(__fmul_rn(a, __fsub_rn(1.0f, w)), __fmul_rn(b, w));
}

// Raw tiles [rows * cols, planes, tile_h, tile_w] -> canvas [planes, canvas_h, canvas_w]. grid holds the row starts,
// row overlaps, column starts and column overlaps (rows, rows - 1, cols, cols - 1 values). A tile blends its first
// overlap rows with the raw tile above, then its first overlap columns with the raw tile to its left.
template <typename T>
__device__ __forceinline__ void tile_stitch(
    const T* tiles, T* canvas, const uint32_t* grid, uint32_t rows, uint32_t cols, uint32_t planes,
    uint32_t tile_h, uint32_t tile_w, uint32_t canvas_h, uint32_t canvas_w) {
    if (!vision_flat_launch() || !tiles || !canvas || !grid || !rows || !cols || !planes ||
        !tile_h || !tile_w || !canvas_h || !canvas_w) { __trap(); return; }
    const uint32_t* row_start = grid;
    const uint32_t* row_overlap = grid + rows;
    const uint32_t* col_start = row_overlap + rows - 1;
    const uint32_t* col_overlap = col_start + cols;
    const uint64_t tile_plane = uint64_t(tile_h) * tile_w, tile_size = uint64_t(planes) * tile_plane;
    const uint64_t count = uint64_t(planes) * canvas_h * canvas_w;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        const uint32_t X = uint32_t(i % canvas_w), Y = uint32_t((i / canvas_w) % canvas_h);
        const uint64_t plane = i / (uint64_t(canvas_w) * canvas_h);
        uint32_t r = 0, c = 0;
        while (r + 1 < rows && row_start[r + 1] <= Y) ++r;
        while (c + 1 < cols && col_start[c + 1] <= X) ++c;
        const uint32_t y = Y - row_start[r], x = X - col_start[c];
        if (y >= tile_h || x >= tile_w) { __trap(); return; }
        const uint64_t base = plane * tile_plane;
        T v = tiles[uint64_t(r * cols + c) * tile_size + base + uint64_t(y) * tile_w + x];
        if (r > 0 && y < row_overlap[r - 1]) {
            const uint32_t above_y = y + tile_h - row_overlap[r - 1];
            const T a = tiles[uint64_t((r - 1) * cols + c) * tile_size + base + uint64_t(above_y) * tile_w + x];
            v = linear_blend(a, v, y, row_overlap[r - 1]);
        }
        if (c > 0 && x < col_overlap[c - 1]) {
            const uint32_t left_x = x + tile_w - col_overlap[c - 1];
            const T a = tiles[uint64_t(r * cols + c - 1) * tile_size + base + uint64_t(y) * tile_w + left_x];
            v = linear_blend(a, v, x, col_overlap[c - 1]);
        }
        canvas[i] = v;
    }
}

extern "C" __global__ void tile_stitch_bf16(
    const __nv_bfloat16* tiles, __nv_bfloat16* canvas, const uint32_t* grid, uint32_t rows,
    uint32_t cols, uint32_t planes, uint32_t tile_h, uint32_t tile_w, uint32_t canvas_h,
    uint32_t canvas_w) {
    tile_stitch(tiles, canvas, grid, rows, cols, planes, tile_h, tile_w, canvas_h, canvas_w);
}

extern "C" __global__ void tile_stitch_f32(
    const float* tiles, float* canvas, const uint32_t* grid, uint32_t rows, uint32_t cols,
    uint32_t planes, uint32_t tile_h, uint32_t tile_w, uint32_t canvas_h, uint32_t canvas_w) {
    tile_stitch(tiles, canvas, grid, rows, cols, planes, tile_h, tile_w, canvas_h, canvas_w);
}

// out[c, out_first + k, :] = linear blend of a[c, a_first + k, :] and b[c, b_first + k, :] at position k of extent
// frames; a, b and out are [channels, frames, plane] with their own frame counts.
extern "C" __global__ void frame_blend_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, uint32_t channels,
    uint32_t extent, uint64_t plane, uint32_t a_frames, uint32_t a_first, uint32_t b_frames,
    uint32_t b_first, uint32_t out_frames, uint32_t out_first) {
    if (!vision_flat_launch() || !a || !b || !out || !channels || !extent || !plane ||
        a_first + extent > a_frames || b_first + extent > b_frames ||
        out_first + extent > out_frames) { __trap(); return; }
    const uint64_t count = uint64_t(channels) * extent * plane;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        const uint64_t px = i % plane, k = (i / plane) % extent, c = i / (plane * extent);
        const __nv_bfloat16 va = a[(c * a_frames + a_first + k) * plane + px];
        const __nv_bfloat16 vb = b[(c * b_frames + b_first + k) * plane + px];
        out[(c * out_frames + out_first + k) * plane + px] = linear_blend(va, vb, uint32_t(k), extent);
    }
}

// F32 latents [channels, frames, height, width] -> BF16 rows [n_frames * n_height * n_width, channels] of the window
// starting at (first_frame, first_row, first_col).
extern "C" __global__ void latent_rows_bf16(
    const float* latents, __nv_bfloat16* rows, uint32_t channels, uint32_t frames, uint32_t height,
    uint32_t width, uint32_t first_frame, uint32_t first_row, uint32_t first_col, uint32_t n_frames,
    uint32_t n_height, uint32_t n_width) {
    if (!vision_flat_launch() || !latents || !rows || !channels || !n_frames || !n_height ||
        !n_width || first_frame + n_frames > frames || first_row + n_height > height ||
        first_col + n_width > width) { __trap(); return; }
    const uint64_t count = uint64_t(n_frames) * n_height * n_width * channels;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        const uint64_t c = i % channels, token = i / channels;
        const uint64_t x = token % n_width, y = (token / n_width) % n_height,
                       t = token / (uint64_t(n_width) * n_height);
        const uint64_t source =
            ((c * frames + first_frame + t) * height + first_row + y) * uint64_t(width) + first_col + x;
        rows[i] = __float2bfloat16_rn(latents[source]);
    }
}

#if defined(__HIP__)
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#endif

__device__ __forceinline__ int reflect_index(int i, int n) {
    return i < 0 ? -i : (i >= n ? 2 * (n - 1) - i : i);
}

// NCHW reflection padding, with pad < min(height,width).
extern "C" __global__ void reflect_pad_2d_f32(const float* input, float* output,
    uint32_t channels, uint32_t height, uint32_t width, uint32_t pad) {
    if (!vision_flat_launch() || !input || !output || !channels || pad >= height || pad >= width) { __trap(); return; }
    const uint64_t ph = uint64_t(height) + 2 * pad, pw = uint64_t(width) + 2 * pad;
    const uint64_t count = uint64_t(channels) * ph * pw;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count; i += uint64_t(gridDim.x) * 256) {
        const uint32_t c = i / (ph * pw);
        const int y = reflect_index(int((i / pw) % ph) - int(pad), int(height));
        const int x = reflect_index(int(i % pw) - int(pad), int(width));
        output[i] = input[(uint64_t(c) * height + y) * width + x];
    }
}

// One frame x [height * width, channels] -> im2col rows a [rows, channels * kt * k * k] of a causal 3D convolution
// with weight [out, channels, kt, k, k] and spatial stride, for output pixels first + r of an out_w-wide output.
// Column (c * kt + t) * k * k + ky * k + kx is 0 for t < kt - 1 (the causal zero frames), else x at
// (oy * stride + ky - pad, ox * stride + kx - pad) with reflected borders.
extern "C" __global__ void causal_frame_im2col_f32(
    const float* x, float* a, uint32_t channels, uint32_t height, uint32_t width, uint32_t out_w,
    uint32_t kt, uint32_t k, uint32_t stride, uint32_t pad, uint64_t first, uint32_t rows) {
    if (!vision_flat_launch() || !x || !a || !channels || height < 2 || width < 2 || !out_w || !kt ||
        !k || !stride || pad >= height || pad >= width) { __trap(); return; }
    const uint32_t taps = k * k, frame_cols = kt * taps;
    const uint64_t cols = uint64_t(channels) * frame_cols, count = uint64_t(rows) * cols;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        const uint64_t r = i / cols;
        const uint32_t col = uint32_t(i - r * cols), c = col / frame_cols, rem = col % frame_cols;
        if (rem / taps + 1 < kt) { a[i] = 0.0f; continue; }
        const uint32_t j = rem % taps;
        const uint64_t p = first + r;
        const int oy = int(p / out_w), ox = int(p % out_w);
        const int iy = reflect_index(oy * int(stride) + int(j / k) - int(pad), int(height));
        const int ix = reflect_index(ox * int(stride) + int(j % k) - int(pad), int(width));
        a[i] = x[(uint64_t(iy) * width + ix) * channels + c];
    }
}

// GroupNorm/SiLU: layout 0 NHWC, 1 NCHW; 512-thread Welford or 1024-thread two-pass moments.
extern "C" __global__ __launch_bounds__(1024) void group_norm_silu_f32(
    const float* x, const float* gamma, const float* beta, float* y, uint64_t pixels,
    uint32_t channels, uint32_t groups, float eps, uint32_t layout) {
    if ((blockDim.x != 512 && blockDim.x != 1024) || blockDim.y != 1 || gridDim.x != groups || gridDim.y != 1 || !x ||
        !gamma || !beta || !y || !pixels || !groups || channels % groups || layout > 1) { __trap(); return; }
    __shared__ float means[32], moments[32], counts[32];
    __shared__ float result[2], total;
    const uint32_t width = channels / groups, first = blockIdx.x * width;
    const uint64_t count = pixels * width;
    float mean, rstd;
    if (blockDim.x == 1024) {
        const auto reduce = [&](float v) {
            for (int delta = 16; delta; delta /= 2) v = __fadd_rn(v, __shfl_down_sync(FF_WARP_MASK, v, delta));
            if (threadIdx.x % 32 == 0) means[threadIdx.x / 32] = v;
            __syncthreads();
            if (threadIdx.x < 32) {
                v = means[threadIdx.x];
                for (int delta = 16; delta; delta /= 2) v = __fadd_rn(v, __shfl_down_sync(FF_WARP_MASK, v, delta));
                if (threadIdx.x == 0) total = v;
            }
            __syncthreads();
            const float out = total;
            __syncthreads();
            return out;
        };
        const auto at = [&](uint64_t e) { return layout ? (first + e % width) * pixels + e / width
                                                      : (e / width) * channels + first + e % width; };
        float sum = 0.0f;
        for (uint64_t e = threadIdx.x; e < count; e += 1024) sum = __fadd_rn(sum, x[at(e)]);
        mean = __fdiv_rn(reduce(sum), float(count));
        float squares = 0.0f;
        for (uint64_t e = threadIdx.x; e < count; e += 1024) {
            const float d = __fsub_rn(x[at(e)], mean);
            squares = __fmaf_rn(d, d, squares);
        }
        rstd = __frcp_rn(__fsqrt_rn(__fadd_rn(__fdiv_rn(reduce(squares), float(count)), eps)));
    } else {
    const auto at = [&](uint64_t e) { return layout ? uint64_t(first) * pixels + e
                                                  : (e % pixels) * channels + first + e / pixels; };
    float m = 0.0f, moment = 0.0f, n = 0.0f;
    for (uint64_t e = threadIdx.x; e < count; e += blockDim.x) {
        const float value = x[at(e)];
        n += 1.0f;
        const float delta = value - m;
        m += delta / n;
        moment += delta * (value - m);
    }
    const auto warp_reduce = [&]() {
        for (int offset = 16; offset; offset /= 2) {
            const float other_m = __shfl_down_sync(FF_WARP_MASK, m, offset);
            const float other_moment = __shfl_down_sync(FF_WARP_MASK, moment, offset);
            const float other_n = __shfl_down_sync(FF_WARP_MASK, n, offset);
            if (threadIdx.x % 32 + offset < 32 && other_n > 0.0f) {
                const float combined = n + other_n;
                const float ratio = other_n / combined;
                const float delta = other_m - m;
                moment = moment + other_moment + delta * delta * n * ratio;
                m += delta * ratio;
                n = combined;
            }
        }
    };
    warp_reduce();
    if (threadIdx.x % 32 == 0) {
        means[threadIdx.x / 32] = m;
        moments[threadIdx.x / 32] = moment;
        counts[threadIdx.x / 32] = n;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        const bool active = threadIdx.x < blockDim.x / 32;
        m = active ? means[threadIdx.x] : 0.0f;
        moment = active ? moments[threadIdx.x] : 0.0f;
        n = active ? counts[threadIdx.x] : 0.0f;
        warp_reduce();
        if (threadIdx.x == 0) {
            result[0] = m;
            result[1] = rsqrtf(moment / n + eps);
        }
    }
    __syncthreads();
    mean = result[0];
    rstd = result[1];
    }
    for (uint64_t e = threadIdx.x; e < count; e += blockDim.x) {
        const uint64_t i = layout ? uint64_t(first) * pixels + e : (e / width) * channels + first + e % width;
        const uint32_t c = first + uint32_t(layout ? e / pixels : e % width);
        const float a = __fmul_rn(rstd, gamma[c]);
        const float b = blockDim.x == 512 ? __fsub_rn(beta[c], __fmul_rn(a, mean))
                                        : __fmaf_rn(-a, mean, beta[c]);
        const float v = __fmaf_rn(x[i], a, b);
        y[i] = __fdiv_rn(v, __fadd_rn(1.0f, expf(-v)));
    }
}

// y[i] = residual[i] + (y[i] + bias[(i / inner) % channels]); without residual y[i] + bias, F32.
extern "C" __global__ void bias_residual_f32(
    float* y, const float* bias, const float* residual, uint64_t count, uint32_t channels,
    uint64_t inner) {
    if (!vision_flat_launch() || !y || !bias || !channels || !inner) { __trap(); return; }
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        const float v = __fadd_rn(y[i], bias[(i / inner) % channels]);
        y[i] = residual ? __fadd_rn(residual[i], v) : v;
    }
}

// u8 planes [channels, height, width] -> F32 [h * w, channels] of the h x w window at (y0, x0):
// (u * (1 / 255) - mean[c]) / std[c], table = [mean | std].
extern "C" __global__ void normalize_u8_window_f32(
    const uint8_t* planes, const float* table, float* out, uint32_t channels, uint32_t height,
    uint32_t width, uint32_t y0, uint32_t x0, uint32_t h, uint32_t w) {
    if (!vision_flat_launch() || !planes || !table || !out || !channels || y0 + h > height ||
        x0 + w > width) { __trap(); return; }
    const uint64_t count = uint64_t(h) * w * channels;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        const uint32_t c = uint32_t(i % channels);
        const uint64_t p = i / channels;
        const uint64_t src = (uint64_t(c) * height + y0 + p / w) * width + x0 + p % w;
        const float v = __fmul_rn(float(planes[src]), __frcp_rn(255.0f));
        out[i] = __fdiv_rn(__fsub_rn(v, table[c]), table[channels + c]);
    }
}

// Moments [2 * channels, plane] (mean, then log-variance) and eps [channels, plane] -> mean, std =
// exp(0.5 * clamp(logvar, -30, 20)) and latents = (f32(f16(mean + std * eps)) - table[c]) / table[channels + c],
// table = [latents_mean | latents_std], all [channels, plane] F32.
extern "C" __global__ void gaussian_posterior_f32(
    const float* moments, const float* eps, const float* table, float* mean, float* stdev,
    float* latents, uint32_t channels, uint64_t plane) {
    if (!vision_flat_launch() || !moments || !eps || !table || !mean || !stdev || !latents ||
        !channels || !plane) { __trap(); return; }
    const uint64_t count = uint64_t(channels) * plane;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        const uint32_t c = uint32_t(i / plane);
        const float m = moments[i];
        const float logvar = fminf(fmaxf(moments[count + i], -30.0f), 20.0f);
        const float s = expf(__fmul_rn(0.5f, logvar));
        const float sample = __half2float(__float2half_rn(__fadd_rn(m, __fmul_rn(s, eps[i]))));
        mean[i] = m;
        stdev[i] = s;
        latents[i] = __fdiv_rn(__fsub_rn(sample, table[c]), table[channels + c]);
    }
}

// F32 CHW [channels, height, width] -> BF16 NHWC [height, width, channels] with
// y = bf16(bf16(x) * bf16(scale) + bf16(shift)).
extern "C" __global__ void chw_to_nhwc_scale_shift_bf16(
    const float* input, __nv_bfloat16* output,
    uint32_t channels, uint32_t height, uint32_t width,
    float scale, float shift) {
    if (!vision_flat_launch() || !input || !output || !channels || !height || !width ||
        !isfinite(scale) || !isfinite(shift)) { __trap(); return; }
    const __nv_bfloat16 sc = __float2bfloat16_rn(scale);
    const __nv_bfloat16 sh = __float2bfloat16_rn(shift);
    const uint64_t plane = uint64_t(height) * width;
    const uint64_t count = plane * channels;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        const uint64_t hw = i / channels;
        const uint32_t c = uint32_t(i % channels);
        const float v = input[uint64_t(c) * plane + hw];
        const __nv_bfloat16 x = __float2bfloat16_rn(v);
        if (!isfinite(v) || !isfinite(__bfloat162float(x))) { __trap(); return; }
        output[i] = __hadd(__hmul(x, sc), sh);
    }
}

// Diagonal-Gaussian sample from NHWC moments [height, width, 2*latent_channels],
// 2x2 space-to-depth rearrange, non-affine BF16 BatchNorm2d, and upcast to F32
// token-major [height/2 * width/2, 4*latent_channels]. Optionally writes NCHW moments.
extern "C" __global__ void diagonal_gaussian_patch_bn_f32(
    const __nv_bfloat16* moments_nhwc, const __nv_bfloat16* noise_nchw,
    const __nv_bfloat16* running_mean, const __nv_bfloat16* running_var,
    __nv_bfloat16* moments_nchw, float* output,
    uint32_t latent_channels, uint32_t height, uint32_t width, float eps) {
    if (!vision_flat_launch() || !moments_nhwc || !noise_nchw || !running_mean ||
        !running_var || !output || !latent_channels || !height || !width ||
        (height & 1u) || (width & 1u) || !(eps > 0.0f)) { __trap(); return; }
    const uint32_t out_h = height >> 1;
    const uint32_t out_w = width >> 1;
    const uint32_t bn_channels = latent_channels << 2;
    const uint32_t moment_channels = latent_channels << 1;
    const uint64_t tokens = uint64_t(out_h) * out_w;
    const uint64_t count = tokens * bn_channels;
    const __nv_bfloat16 half_bf16 = __float2bfloat16_rn(0.5f);
    const __nv_bfloat16 eps_bf16 = __float2bfloat16_rn(eps);
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        const uint64_t token = i / bn_channels;
        const uint32_t c_bn = uint32_t(i % bn_channels);
        const uint32_t c = c_bn >> 2;
        const uint32_t ph = (c_bn >> 1) & 1u;
        const uint32_t pw = c_bn & 1u;
        const uint32_t h_out = uint32_t(token / out_w);
        const uint32_t w_out = uint32_t(token % out_w);
        const uint32_t h_in = (h_out << 1) + ph;
        const uint32_t w_in = (w_out << 1) + pw;
        const uint64_t hw_in = uint64_t(h_in) * width + w_in;
        const uint64_t nhwc_base = hw_in * moment_channels;
        const __nv_bfloat16 mean = moments_nhwc[nhwc_base + c];
        const __nv_bfloat16 logvar = moments_nhwc[nhwc_base + latent_channels + c];
        if (moments_nchw) {
            const uint64_t plane = uint64_t(height) * width;
            moments_nchw[uint64_t(c) * plane + hw_in] = mean;
            moments_nchw[uint64_t(latent_channels + c) * plane + hw_in] = logvar;
        }
        const float lv_f = fminf(fmaxf(__bfloat162float(logvar), -30.0f), 20.0f);
        const __nv_bfloat16 lv_clamped = __float2bfloat16_rn(lv_f);
        const __nv_bfloat16 half_lv = __hmul(half_bf16, lv_clamped);
        const __nv_bfloat16 std_val = __float2bfloat16_rn(expf(__bfloat162float(half_lv)));
        const __nv_bfloat16 u = noise_nchw[uint64_t(c) * uint64_t(height) * width + hw_in];
        const __nv_bfloat16 z = __hadd(mean, __hmul(std_val, u));
        const __nv_bfloat16 var_eps = __hadd(running_var[c_bn], eps_bf16);
        const __nv_bfloat16 bn_std = __float2bfloat16_rn(sqrtf(__bfloat162float(var_eps)));
        const __nv_bfloat16 z_centered = __hsub(z, running_mean[c_bn]);
        const __nv_bfloat16 z_norm = __float2bfloat16_rn(
            __fdiv_rn(__bfloat162float(z_centered), __bfloat162float(bn_std)));
        output[i] = __bfloat162float(z_norm);
    }
}

// NHWC adaptive average pooling into a channel span of NHWC output.
extern "C" __global__ void adaptive_avg_pool2d_f32(
    const float* input, float* output, uint32_t ih, uint32_t iw, uint32_t oh, uint32_t ow,
    uint32_t channels, uint32_t output_channels, uint32_t offset) {
    if (!vision_flat_launch() || !input || !output || !ih || !iw || !oh || !ow ||
        !channels || channels > output_channels || offset > output_channels - channels) { __trap(); return; }
    const uint64_t count = uint64_t(oh) * ow * channels;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count; i += uint64_t(gridDim.x) * 256) {
        const uint32_t c = i % channels, x = (i / channels) % ow, y = i / (uint64_t(channels) * ow);
        const uint32_t y0 = uint64_t(y) * ih / oh, y1 = (uint64_t(y + 1) * ih + oh - 1) / oh;
        const uint32_t x0 = uint64_t(x) * iw / ow, x1 = (uint64_t(x + 1) * iw + ow - 1) / ow;
        float sum = 0.0f;
        for (uint32_t yy = y0; yy < y1; ++yy)
            for (uint32_t xx = x0; xx < x1; ++xx)
                sum = __fadd_rn(sum, input[(uint64_t(yy) * iw + xx) * channels + c]);
        output[(uint64_t(y) * ow + x) * output_channels + offset + c] = __fdiv_rn(sum, float((y1-y0)*(x1-x0)));
    }
}

// Axial split-half RoPE on NHWC, with a [height * width, head_dim] cos/sin table.
extern "C" __global__ void rotary_spatial_f32(float* x, const float* table,
    uint32_t pixels, uint32_t heads, uint32_t dim) {
    if (!vision_flat_launch() || !x || !table || !pixels || !heads || !dim || dim % 4) { __trap(); return; }
    const uint32_t half = dim / 2;
    const uint64_t count = uint64_t(pixels) * heads * half;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count; i += uint64_t(gridDim.x) * 256) {
        const uint32_t d = i % half;
        const uint64_t row = i / half, pixel = row / heads;
        const float c = table[pixel * dim + d], s = table[pixel * dim + half + d];
        const float a = x[row * dim + d], b = x[row * dim + half + d];
        x[row * dim + d] = __fsub_rn(__fmul_rn(a,c), __fmul_rn(b,s));
        x[row * dim + half + d] = __fadd_rn(__fmul_rn(b,c), __fmul_rn(a,s));
    }
}

// Q [H,W,heads,64], pooled K [h,w,heads,64], V [h,w,heads,256], output [H,W,heads,256].
// Integer nearest-exact expansion and dilated shifted 9x9 windows share source indices.
extern "C" __global__ __launch_bounds__(128) void neighborhood_attention_2d_f32(
    const float* q, const float* k, const float* v, float* output,
    uint32_t height, uint32_t width, uint32_t source_h, uint32_t source_w, uint32_t heads,
    uint32_t dilation_h, uint32_t dilation_w) {
    if (blockDim.x != 128 || blockDim.y != 1 || gridDim.y != 1 || !q || !k || !v || !output ||
        !heads || source_h < 9 || source_w < 9 || height % source_h || width % source_w ||
        !dilation_h || !dilation_w || height % dilation_h || width % dilation_w ||
        height/dilation_h < 9 || width/dilation_w < 9) { __trap(); return; }
    __shared__ float scores[4][81];
    const uint32_t lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const uint64_t query = uint64_t(blockIdx.x) * 4 + warp;
    if (query >= uint64_t(height) * width * heads) return;
    const uint32_t head = query % heads, y = query / (uint64_t(heads) * width), x = (query / heads) % width;
    const int top = max(0, min(int(y / dilation_h)-4, int(height/dilation_h)-9));
    const int left = max(0, min(int(x / dilation_w)-4, int(width/dilation_w)-9));
    const float qa = q[query * 64 + lane], qb = q[query * 64 + 32 + lane];
    float maximum = -INFINITY;
    for (uint32_t tap = 0; tap < 81; ++tap) {
        const uint32_t sy = ((top + tap/9)*dilation_h + y%dilation_h)/(height/source_h);
        const uint32_t sx = ((left + tap%9)*dilation_w + x%dilation_w)/(width/source_w);
        const uint64_t source = (uint64_t(sy)*source_w + sx)*heads + head;
        float score = __fadd_rn(__fmul_rn(qa, k[source*64+lane]), __fmul_rn(qb, k[source*64+32+lane]));
        for (uint32_t shift = 16; shift; shift /= 2) score = __fadd_rn(score, __shfl_xor_sync(FF_WARP_MASK, score, shift));
        score = __fmul_rn(score, 0.125f);
        if (lane == 0) scores[warp][tap] = score;
        maximum = fmaxf(maximum, score);
    }
    __syncwarp();
    float sum = 0.0f;
    for (uint32_t tap = lane; tap < 81; tap += 32) {
        const float weight = expf(__fsub_rn(scores[warp][tap], maximum));
        scores[warp][tap] = weight;
        sum = __fadd_rn(sum, weight);
    }
    for (uint32_t shift = 16; shift; shift /= 2) sum = __fadd_rn(sum, __shfl_xor_sync(FF_WARP_MASK, sum, shift));
    if (!(sum > 0.0f) || !isfinite(sum)) { __trap(); return; }
    __syncwarp();
    float accum[8] = {};
    for (uint32_t tap = 0; tap < 81; ++tap) {
        const uint32_t sy = ((top + tap/9)*dilation_h + y%dilation_h)/(height/source_h);
        const uint32_t sx = ((left + tap%9)*dilation_w + x%dilation_w)/(width/source_w);
        const uint64_t source = (uint64_t(sy)*source_w + sx)*heads + head;
        const float weight = __fdiv_rn(scores[warp][tap],sum);
        #pragma unroll
        for (uint32_t j = 0; j < 8; ++j) accum[j] = __fmaf_rn(weight,v[source*256+j*32+lane],accum[j]);
    }
    #pragma unroll
    for (uint32_t j = 0; j < 8; ++j) output[query*256+j*32+lane] = accum[j];
}

// Camera [12] changes from horizontal FOV to focal pixels; [14] is image resolution.
extern "C" __global__ void perspective_focal_f32(float* camera) {
    if (blockDim.x != 1 || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.x != 1 || gridDim.y != 1 || gridDim.z != 1 || !camera ||
        !isfinite(camera[12]) || !isfinite(camera[14]) || !(camera[12] > 0.0f) ||
        !(camera[12] < 3.14159265358979323846f) || !(camera[14] > 0.0f)) { __trap(); return; }
    const float tangent = tanf(__fmul_rn(camera[12], 0.5f));
    const float focal = __fdiv_rn(__fmul_rn(__fdiv_rn(16.0f, tangent), camera[14]), 32.0f);
    if (!isfinite(focal) || !(focal > 0.0f)) { __trap(); return; }
    camera[12] = focal;
}

// Inclusive XYZ grid, row-major camera inverse [3,4], border bilinear align_corners=false.
// Camera parameters [12 matrix values, focal_pixels, mesh_scale, image_resolution].
extern "C" __global__ void project_bilinear_f32(const float* input, float* output, const float* camera,
    uint32_t grid_side, uint32_t height, uint32_t width, uint32_t channels,
    uint32_t output_channels, uint32_t offset, uint64_t first, uint32_t rows,
    uint32_t add, float divisor) {
    if (!vision_flat_launch() || !input || !output || !camera || grid_side < 2 || !height || !width ||
        !channels || channels > output_channels || offset > output_channels-channels || add > 1 ||
        !(divisor > 0.0f)) { __trap(); return; }
    const uint64_t count = uint64_t(rows)*channels;
    for (uint64_t i = uint64_t(blockIdx.x)*256+threadIdx.x; i < count; i += uint64_t(gridDim.x)*256) {
        const uint64_t row = first + i/channels;
        const uint32_t c = i%channels, ix = row/(uint64_t(grid_side)*grid_side), iy = (row/grid_side)%grid_side, iz=row%grid_side;
        const auto coord = [&](uint32_t j) {
            const float step = __fdiv_rn(2.0f,float(grid_side-1));
            const float v = j < grid_side/2 ? __fmaf_rn(float(j),step,-1.0f) : __fmaf_rn(-float(grid_side-1-j),step,1.0f);
            return __fdiv_rn(__fdiv_rn(v,camera[13]),2.0f);
        };
        const float a=coord(ix), b=-coord(iz), d=coord(iy);
        const auto transform = [&](uint32_t r) {
            return __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(a,camera[r*4]),__fmul_rn(b,camera[r*4+1])),__fmul_rn(d,camera[r*4+2])),camera[r*4+3]);
        };
        const float xx=transform(0), yy=transform(1), zz=transform(2);
        const float resolution=camera[14];
        const float focal=camera[12];
        const float denom=__fadd_rn(-zz,1.0e-8f);
        const auto pixel = [&](float camera_coord, uint32_t size) {
            const float projected=__fadd_rn(__fdiv_rn(__fmul_rn(focal,camera_coord),denom),resolution/2.0f);
            const float normalized=__fsub_rn(__fmul_rn(__fdiv_rn(__fadd_rn(projected,0.5f),resolution),2.0f),1.0f);
            return fminf(fmaxf(__fdiv_rn(__fsub_rn(__fmul_rn(__fadd_rn(normalized,1.0f),float(size)),1.0f),2.0f),0.0f),float(size-1));
        };
        const float px=pixel(xx,width), py=pixel(-yy,height);
        const uint32_t x0=uint32_t(px), y0=uint32_t(py), x1=min(x0+1,width-1), y1=min(y0+1,height-1);
        const float dx=px-float(x0), dy=py-float(y0);
        float val=__fmul_rn(input[(uint64_t(y0)*width+x0)*channels+c],__fmul_rn(1.0f-dx,1.0f-dy));
        val=__fadd_rn(val,__fmul_rn(input[(uint64_t(y0)*width+x1)*channels+c],__fmul_rn(dx,1.0f-dy)));
        val=__fadd_rn(val,__fmul_rn(input[(uint64_t(y1)*width+x0)*channels+c],__fmul_rn(1.0f-dx,dy)));
        val=__fadd_rn(val,__fmul_rn(input[(uint64_t(y1)*width+x1)*channels+c],__fmul_rn(dx,dy)));
        const uint64_t out=row*output_channels+offset+c;
        output[out]=__fdiv_rn(add ? __fadd_rn(output[out],val) : val,divisor);
    }
}

// Pairwise sum of at most 32 dense [heads, rows, columns] parts into a strided tensor.
extern "C" __global__ void sum_parts_f32(const float* input, float* output,
    uint32_t parts, uint32_t heads, uint32_t rows, uint32_t columns,
    uint64_t head_stride, uint64_t row_stride) {
    if (!vision_flat_launch() || !input || !output || !parts || parts > 32 ||
        !heads || !rows || !columns) { __trap(); return; }
    const uint64_t count = uint64_t(heads) * rows * columns;
    for (uint64_t i = uint64_t(blockIdx.x)*256+threadIdx.x; i<count; i+=uint64_t(gridDim.x)*256) {
        float values[32];
        #pragma unroll
        for (uint32_t p=0; p<32; ++p) values[p] = p<parts ? input[uint64_t(p)*count+i] : 0.0f;
        #pragma unroll
        for (uint32_t step=1; step<32; step*=2) {
            #pragma unroll
            for (uint32_t p=0; p<32; p+=2*step) values[p] = __fadd_rn(values[p],values[p+step]);
        }
        const uint64_t row = i/columns;
        output[(row/rows)*head_stride+(row%rows)*row_stride+i%columns] = values[0];
    }
}

// Ordered F32 sum with division on the last view.
extern "C" __global__ void accumulate_mean_f32(const float* input, float* output,
    uint64_t count, uint32_t add, float divisor) {
    if (!vision_flat_launch() || !input || !output || add > 1 || !(divisor > 0.0f)) { __trap(); return; }
    for (uint64_t i = uint64_t(blockIdx.x)*256+threadIdx.x; i<count; i+=uint64_t(gridDim.x)*256)
        output[i]=__fdiv_rn(add ? __fadd_rn(output[i],input[i]) : input[i],divisor);
}

// Row-major [rows, inner] times transposed [cols, inner].
extern "C" __global__ __launch_bounds__(256) void linear_ordered_f32(
    const float* x, const float* weight, const float* bias, float* y,
    uint32_t rows, uint32_t cols, uint32_t inner) {
    if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1 || gridDim.z != 1 ||
        !x || !weight || !y || !rows || !cols || !inner) { __trap(); return; }
    __shared__ float a[16][33];
    __shared__ float b[32][65];
    const uint32_t row0 = blockIdx.y * 16, col0 = blockIdx.x * 64;
    const uint32_t lane = threadIdx.x % 64, group = threadIdx.x / 64;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t first = 0; first < inner; first += 32) {
        for (uint32_t j = threadIdx.x; j < 16 * 32; j += 256) {
            const uint32_t r = j / 32, k = j % 32;
            a[r][k] = row0 + r < rows && first + k < inner ? x[uint64_t(row0 + r) * inner + first + k] : 0.0f;
        }
        for (uint32_t j = threadIdx.x; j < 64 * 32; j += 256) {
            const uint32_t c = j / 32, k = j % 32;
            b[k][c] = col0 + c < cols && first + k < inner ? weight[uint64_t(col0 + c) * inner + first + k] : 0.0f;
        }
        __syncthreads();
        const uint32_t count = min(32u, inner - first);
        for (uint32_t k = 0; k < count; ++k) {
            const float w = b[k][lane];
            #pragma unroll
            for (uint32_t r = 0; r < 4; ++r) acc[r] = __fmaf_rn(a[group + r * 4][k], w, acc[r]);
        }
        __syncthreads();
    }
    if (col0 + lane < cols) {
        #pragma unroll
        for (uint32_t r = 0; r < 4; ++r) {
            const uint32_t row = row0 + group + r * 4;
            if (row < rows) y[uint64_t(row) * cols + col0 + lane] = bias ? __fadd_rn(acc[r], bias[col0 + lane]) : acc[r];
        }
    }
}
