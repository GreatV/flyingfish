#include "q38_contract.cuh"

// One warp scores one compressed position. Stored Q/K are E2M1-per32/E8M0.
template<bool Candidates>
__device__ __forceinline__ void score_positions(
    const uint8_t* query, const uint8_t* query_scales,
    const uint8_t* keys, const uint8_t* key_scales,
    const __nv_bfloat16* weights, const Q38SequenceMeta* sequence,
    const uint32_t* committed, const int32_t* candidates, float* scores, int32_t* positions,
    int rows, int heads, int capacity, int score_columns, int ratio, float weight_scale) {
    q38_meta_guard(sequence);
    if (sequence->valid_rows > rows || *committed != uint32_t(sequence->key_count / ratio)
        || *committed > uint32_t(capacity)) __trap();
    const int tiles = (score_columns + 3) / 4;
    const int row = blockIdx.x / tiles;
    const int column = (blockIdx.x % tiles) * 4 + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (column >= score_columns) return;
    const int64_t output = int64_t(row) * score_columns + column;
    if (row >= sequence->valid_rows) {
        if (lane == 0) { scores[output] = -INFINITY; positions[output] = -1; }
        return;
    }
    int position = column;
    if constexpr (Candidates) {
        const int block = candidates[int64_t(row) * 2048 + column / 8];
        if (block < -1 || (block >= 0 && int64_t(block) * 8 >= capacity)) __trap();
        position = block < 0 ? -1 : block * 8 + column % 8;
    }
    if (position < 0 || position >= capacity
        || position >= (sequence->prefix + row + 1) / ratio) {
        if (lane == 0) { scores[output] = -INFINITY; positions[output] = -1; }
        return;
    }
    float key[4];
#pragma unroll
    for (int group = 0; group < 4; ++group) {
        const int column = group * 32 + lane;
        const uint8_t packed = keys[int64_t(position) * 64 + column / 2];
        const uint8_t nibble = (column & 1) ? packed >> 4 : packed & 15;
        key[group] = bround(__fmul_rn(e2m1_decode(nibble),
            e8m0_decode(key_scales[int64_t(position) * 4 + group])));
        if (!isfinite(key[group])) __trap();
    }
    float total = 0.0f;
    for (int head = 0; head < heads; ++head) {
        const int64_t qrow = int64_t(row) * heads + head;
        float dot = 0.0f;
#pragma unroll
        for (int group = 0; group < 4; ++group) {
            const int column = group * 32 + lane;
            const uint8_t packed = query[qrow * 64 + column / 2];
            const uint8_t nibble = (column & 1) ? packed >> 4 : packed & 15;
            const float value = bround(__fmul_rn(e2m1_decode(nibble),
                e8m0_decode(query_scales[qrow * 4 + group])));
            if (!isfinite(value)) __trap();
            dot = fmaf(value, key[group], dot);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            dot = __fadd_rn(dot, __shfl_down_sync(0xffffffffu, dot, offset));
        if (lane == 0) {
            const float weight = bround(__fmul_rn(__bfloat162float(weights[qrow]), weight_scale));
            const float product = bround(__fmul_rn(fmaxf(bround(dot), 0.0f), weight));
            total = __fadd_rn(total, product);
        }
    }
    if (lane == 0) {
        if (!isfinite(total)) __trap();
        scores[output] = bround(total);
        positions[output] = position;
    }
}

#define SCORE_ENTRY(Name, Candidates) \
extern "C" __global__ void Name( \
    const uint8_t* query, const uint8_t* query_scales, const uint8_t* keys, const uint8_t* key_scales, \
    const __nv_bfloat16* weights, const Q38SequenceMeta* sequence, const uint32_t* committed, \
    const int32_t* candidates, float* scores, int32_t* positions, \
    int rows, int heads, int capacity, int score_columns, int ratio, float weight_scale) { \
    score_positions<Candidates>(query, query_scales, keys, key_scales, weights, sequence, \
        committed, candidates, scores, positions, rows, heads, capacity, score_columns, ratio, weight_scale); \
}
SCORE_ENTRY(index_score_fp4_128, false)
SCORE_ENTRY(index_score_fp4_128_candidates, true)
#undef SCORE_ENTRY

// The newest reachable block is excluded here and appended after the bounded 2047 selection.
extern "C" __global__ void index_block_scores(
    const float* scores, float* blocks, int32_t* ids, const Q38SequenceMeta* sequence,
    int rows, int capacity, int ratio) {
    q38_meta_guard(sequence);
    if (sequence->valid_rows > rows || sequence->key_count / ratio > capacity) __trap();
    const int block_count = (capacity + 7) / 8;
    const int64_t item = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (item >= int64_t(rows) * block_count) return;
    const int row = item / block_count, block = item % block_count;
    const int visible = row < sequence->valid_rows ? (sequence->prefix + row + 1) / ratio : 0;
    const int newest = visible ? (visible - 1) / 8 : -1;
    float best = -INFINITY;
    if (block != newest) {
#pragma unroll
        for (int offset = 0; offset < 8; ++offset) {
            const int position = block * 8 + offset;
            if (position < visible) {
                const float value = scores[int64_t(row) * capacity + position];
                if (isnan(value) || value == INFINITY) __trap();
                best = fmaxf(best, value);
            }
        }
    }
    blocks[item] = best;
    ids[item] = best == -INFINITY ? -1 : block;
}

extern "C" __global__ void index_block_publish(
    const int32_t* selected, int32_t* candidates, const Q38SequenceMeta* sequence,
    int rows, int ratio) {
    q38_meta_guard(sequence);
    if (sequence->valid_rows > rows) __trap();
    const int64_t item = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (item >= int64_t(rows) * 2048) return;
    const int row = item / 2048, column = item % 2048;
    if (column < 2047) candidates[item] = selected[int64_t(row) * 2047 + column];
    else {
        const int visible = row < sequence->valid_rows ? (sequence->prefix + row + 1) / ratio : 0;
        candidates[item] = visible ? (visible - 1) / 8 : -1;
    }
}

extern "C" __global__ void index_position_order(const int32_t* input, int32_t* output) {
    __shared__ int32_t order[512];
    const int64_t base = int64_t(blockIdx.x) * 512;
    for (int i = threadIdx.x; i < 512; i += 256) {
        const int value = input[base + i];
        if (value < -1) __trap();
        order[i] = value < 0 ? INT32_MAX : value;
    }
    __syncthreads();
    for (int size = 2; size <= 512; size <<= 1) {
        for (int stride = size / 2; stride; stride >>= 1) {
            for (int i = threadIdx.x; i < 512; i += 256) {
                const int other = i ^ stride;
                if (other > i && ((order[i] > order[other]) == ((i & size) == 0))) {
                    const int value = order[i]; order[i] = order[other]; order[other] = value;
                }
            }
            __syncthreads();
        }
    }
    for (int i = threadIdx.x; i < 512; i += 256)
        output[base + i] = order[i] == INT32_MAX ? -1 : order[i];
}
