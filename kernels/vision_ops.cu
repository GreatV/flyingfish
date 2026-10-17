#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

__device__ __forceinline__ bool vision_flat_launch() {
    return blockDim.x == 256 && blockDim.y == 1 && blockDim.z == 1 &&
           gridDim.y == 1 && gridDim.z == 1;
}

// f32 pixels -> bf16 rows of `width`; both sides finite.
extern "C" __global__ void vision_cast_patches(
    const float* input, __nv_bfloat16* output, uint64_t count, uint32_t width) {
    if (!vision_flat_launch() || !input || !output || !count || !width || count % width) { __trap(); return; }
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        float x = input[i];
        __nv_bfloat16 y = __float2bfloat16_rn(x);
        if (!isfinite(x) || !isfinite(__bfloat162float(y))) { __trap(); return; }
        output[i] = y;
    }
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

// Three token-major `[rows, hq*d]` projections -> head-major Q/K/V for the attention core.
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
        uint64_t dst = (uint64_t(head) * rows + first + r) * d + j;
        q[dst] = q_in[src];
        k[dst] = k_in[src];
        v[(uint64_t(head) * padded + key_first + r) * d + j] = v_in[src];
    }
}

// Row copy of `width` BF16 elements: output[destination_first + r] = features[source_first + r].
extern "C" __global__ void vision_splice_rows(
    const __nv_bfloat16* features, __nv_bfloat16* output,
    uint32_t feature_rows, uint32_t output_rows,
    uint32_t source_first, uint32_t destination_first, uint32_t rows, uint32_t width) {
    if (!vision_flat_launch() || !features || !output || !rows || !width ||
        uint64_t(source_first) + rows > feature_rows ||
        uint64_t(destination_first) + rows > output_rows) { __trap(); return; }
    uint64_t count = uint64_t(rows) * width;
    for (uint64_t i = uint64_t(blockIdx.x) * 256 + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * 256) {
        output[uint64_t(destination_first) * width + i] =
            features[uint64_t(source_first) * width + i];
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
