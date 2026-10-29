#include "q38_contract.cuh"

// One block normalizes one stream; projection stores [copies * width keys, width value].
extern "C" __global__ void normalized_gate_residual(
    const __nv_bfloat16* input, const __nv_bfloat16* projection,
    const __nv_bfloat16* q_weight, const __nv_bfloat16* k_weight,
    const uint8_t* mask, const Q38SequenceMeta* sequence,
    __nv_bfloat16* output, float* gates,
    int rows, int copies, int width, float epsilon, float clamp_value, float dot_scale) {
    q38_meta_guard(sequence);
    if (sequence->valid_rows > rows) __trap();
    const int row = blockIdx.x / copies, copy = blockIdx.x % copies;
    if (row >= sequence->valid_rows) return;
    if (mask[row] > 1) __trap();
    const int64_t base = (int64_t(row) * copies + copy) * width;
    const int64_t kv = int64_t(row) * (copies + 1) * width;
    if (!mask[row]) {
        for (int d = threadIdx.x; d < width; d += blockDim.x) output[base + d] = input[base + d];
        if (threadIdx.x == 0) gates[int64_t(row) * copies + copy] = 0.0f;
        return;
    }
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    for (int d = threadIdx.x; d < width; d += blockDim.x) {
        const float h = __bfloat162float(input[base + d]);
        const float k = __bfloat162float(projection[kv + int64_t(copy) * width + d]);
        const int64_t w = int64_t(copy) * width + d;
        const float weight = __fmul_rn(__bfloat162float(q_weight[w]), __bfloat162float(k_weight[w]));
        h2 = __fadd_rn(h2, __fmul_rn(h, h));
        k2 = __fadd_rn(k2, __fmul_rn(k, k));
        dot = __fadd_rn(dot, __fmul_rn(__fmul_rn(h, weight), k));
    }
    for (int offset = 16; offset; offset >>= 1) {
        h2 = __fadd_rn(h2, __shfl_down_sync(0xffffffffu, h2, offset));
        k2 = __fadd_rn(k2, __shfl_down_sync(0xffffffffu, k2, offset));
        dot = __fadd_rn(dot, __shfl_down_sync(0xffffffffu, dot, offset));
    }
    __shared__ float sums[3][8];
    __shared__ float gate;
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    if (lane == 0) { sums[0][warp] = h2; sums[1][warp] = k2; sums[2][warp] = dot; }
    __syncthreads();
    if (warp == 0) {
        h2 = lane < 8 ? sums[0][lane] : 0.0f;
        k2 = lane < 8 ? sums[1][lane] : 0.0f;
        dot = lane < 8 ? sums[2][lane] : 0.0f;
        for (int offset = 16; offset; offset >>= 1) {
            h2 = __fadd_rn(h2, __shfl_down_sync(0xffffffffu, h2, offset));
            k2 = __fadd_rn(k2, __shfl_down_sync(0xffffffffu, k2, offset));
            dot = __fadd_rn(dot, __shfl_down_sync(0xffffffffu, dot, offset));
        }
        if (lane == 0) {
            const float h_inv = rsqrtf(__fadd_rn(__fdiv_rn(h2, float(width)), epsilon));
            const float k_inv = rsqrtf(__fadd_rn(__fdiv_rn(k2, float(width)), epsilon));
            dot = __fmul_rn(__fmul_rn(dot, __fmul_rn(h_inv, k_inv)), dot_scale);
            if (!isfinite(dot)) __trap();
            gate = sigmoid(copysignf(sqrtf(fmaxf(fabsf(dot), clamp_value)), dot));
            gates[int64_t(row) * copies + copy] = gate;
        }
    }
    __syncthreads();
    for (int d = threadIdx.x; d < width; d += blockDim.x) {
        const float h = __bfloat162float(input[base + d]);
        const float value = __bfloat162float(projection[kv + int64_t(copies) * width + d]);
        const float result = __fadd_rn(h, __fmul_rn(gate, value));
        if (!isfinite(result)) __trap();
        output[base + d] = __float2bfloat16_rn(result);
    }
}
