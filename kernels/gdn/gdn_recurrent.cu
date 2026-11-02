// GDN recurrent decode entry: M = 1, value_split = 2.
#include "gdn_recurrent.cuh"

extern "C" __global__ __launch_bounds__(GDN_THREADS, 1)
void gdn_recurrent_masked_commit(const float* q, const float* k, const float* v,
        const float* beta, const float* g, float* state, __nv_bfloat16* out,
        const int* valid, int rows) {
    gdn_recurrent_t<2, false, false, false, true>(q, k, v, beta, g, state, out, nullptr, rows, nullptr, valid);
}

extern "C" __global__ __launch_bounds__(128, 2)
void gdn_block_softmax_history_preview(
        const float* q, const float* k, const float* v, const float* g,
        const float* state, const __nv_bfloat16* logit_scale,
        const __nv_bfloat16* history_bias, const __nv_bfloat16* history_c,
        const int* valid, __nv_bfloat16* out, int rows) {
    if (rows < 1 || rows > 32 || blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.x != (rows + 3) / 4 || gridDim.y != GDN_HEADS || gridDim.z != 1 ||
        !aligned16(q) || !aligned16(k) || !aligned16(v) || !aligned16(g) ||
        !aligned16(state) || !aligned16(logit_scale) || !aligned16(history_bias) ||
        !aligned16(history_c) || !aligned16(out) || !valid) { __trap(); return; }
    const int head = blockIdx.y, first = blockIdx.x * 4;
    const int d = threadIdx.x, lane = d & 31, warp = d >> 5;
    __shared__ float keys[128][33], queries[4][128], weights[4][33];
    for (int row = 0; row < rows; ++row)
        keys[d][row] = k[((size_t)row * GDN_HEADS + head) * 128 + d];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        queries[i][d] = first + i < rows ? q[((size_t)(first + i) * GDN_HEADS + head) * 128 + d] : 0.0f;
    __syncthreads();
    const int query = first + warp;
    const bool key_valid = lane < rows && valid[lane];
    const int count = __popc(__ballot_sync(FF_WARP_MASK, key_valid));
    float score = -INFINITY;
    if (key_valid) {
        score = 0.0f;
        for (int j = 0; j < 128; ++j) score = fmaf(queries[warp][j], keys[j][lane], score);
        score *= __bfloat162float(logit_scale[head]);
    }
    const float history_logit = __fadd_rn(__bfloat162float(history_bias[head]),
        __fmul_rn(__bfloat162float(history_c[head]), logf(float(max(count, 1)) / 32.0f)));
    float maximum = score;
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        maximum = fmaxf(maximum, __shfl_down_sync(FF_WARP_MASK, maximum, delta));
    maximum = fmaxf(__shfl_sync(FF_WARP_MASK, maximum, 0), history_logit);
    const float probability = key_valid ? expf(score - maximum) : 0.0f;
    const float history_probability = expf(history_logit - maximum);
    float denominator = probability;
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        denominator += __shfl_down_sync(FF_WARP_MASK, denominator, delta);
    denominator = __shfl_sync(FF_WARP_MASK, denominator, 0) + history_probability;
    weights[warp][lane] = probability / denominator;
    if (lane == 0) weights[warp][32] = history_probability / denominator;
    __syncthreads();
    float history[4] = {}, local[4] = {};
    for (int key = 0; key < 128; ++key) {
        const float value = state[((size_t)head * 128 + key) * 128 + d];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            history[i] = fmaf(queries[i][key] * (1.0f / sqrtf(128.0f)), value, history[i]);
    }
    for (int key = 0; key < rows; ++key) {
        const float value = v[((size_t)key * GDN_HEADS + head) * 128 + d];
        #pragma unroll
        for (int i = 0; i < 4; ++i) local[i] = fmaf(weights[i][key], value, local[i]);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = first + i;
        if (row < rows) {
            const float result = valid[row] ? __fadd_rn(local[i], __fmul_rn(weights[i][32],
                __fmul_rn(history[i], expf(g[row * GDN_HEADS + head])))) : 0.0f;
            out[((size_t)row * GDN_HEADS + head) * 128 + d] = __float2bfloat16_rn(result);
        }
    }
}

extern "C" __global__ __launch_bounds__(GDN_THREADS, 1)
void gdn_recurrent_core_v2(const float* q, const float* k, const float* v,
                           const float* beta, const float* g, float* S,
                           __nv_bfloat16* out, const Q38SequenceFields* meta) {
    gdn_recurrent_t<2, true>(q, k, v, beta, g, S, out, meta, 1, nullptr);
}

extern "C" __global__ __launch_bounds__(GDN_THREADS, 1)
void gdn_recurrent_core_per_channel_v2(const float* q, const float* k, const float* v,
                                       const float* beta, const float* g, float* S,
                                       __nv_bfloat16* out, const Q38SequenceFields* meta) {
    gdn_recurrent_t<2, true, false, true>(q, k, v, beta, g, S, out, meta, 1, nullptr);
}
