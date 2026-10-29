#include "q38_contract.cuh"

// Cursor is the next absolute input position; tails retain the latest row of each parity.
__device__ __forceinline__ void compress_guard(
    const Q38SequenceMeta* sequence, const uint32_t* cursor, int rows) {
    q38_meta_guard(sequence);
    if (sequence->valid_rows > rows || *cursor != uint32_t(sequence->prefix)) __trap();
}

extern "C" __global__ void latent_compress_pool(
    const float* values, const float* scores, const float* tail_values,
    const float* tail_scores, const uint32_t* cursor, const Q38SequenceMeta* sequence,
    __nv_bfloat16* pooled, int rows, int channels) {
    compress_guard(sequence, cursor, rows);
    const int tiles = (channels + 255) / 256;
    const int group = blockIdx.x / tiles;
    const int channel = (blockIdx.x % tiles) * 256 + threadIdx.x;
    const int first = 2 * group - (sequence->prefix & 1);
    if (channel >= channels || first + 1 >= sequence->valid_rows) return;
    const float v0 = first < 0 ? tail_values[channel] : values[int64_t(first) * channels + channel];
    const float s0 = first < 0 ? tail_scores[channel] : scores[int64_t(first) * channels + channel];
    const float v1 = values[int64_t(first + 1) * channels + channel];
    const float s1 = scores[int64_t(first + 1) * channels + channel];
    if (!isfinite(v0) || !isfinite(v1) || !isfinite(s0) || !isfinite(s1)) __trap();
    const float maximum = fmaxf(s0, s1);
    const float e0 = expf(s0 - maximum), e1 = expf(s1 - maximum);
    const float sum = __fadd_rn(e0, e1);
    const float p0 = __fdiv_rn(e0, sum), p1 = __fdiv_rn(e1, sum);
    pooled[int64_t(group) * channels + channel] = __float2bfloat16_rn(
        __fadd_rn(__fmul_rn(v0, p0), __fmul_rn(v1, p1)));
}

extern "C" __global__ void latent_compress_tail(
    const float* values, const float* scores, float* tail_values, float* tail_scores,
    uint32_t* cursor, const Q38SequenceMeta* sequence, int rows, int channels) {
    compress_guard(sequence, cursor, rows);
    __syncthreads();
    for (int item = threadIdx.x; item < 2 * channels; item += blockDim.x) {
        const int parity = item / channels;
        int row = sequence->valid_rows - 1;
        row -= ((sequence->prefix + row) & 1) != parity;
        if (row >= 0) {
            const int64_t source = int64_t(row) * channels + item % channels;
            const float value = values[source], score = scores[source];
            if (!isfinite(value) || !isfinite(score)) __trap();
            tail_values[item] = value;
            tail_scores[item] = score;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) *cursor = sequence->key_count;
}

// Tickets are [owner, generation, next input position, committed rows].
extern "C" __global__ void latent_publication_begin(
    const Q38SequenceMeta* sequence, const uint32_t* generation,
    const uint32_t* ticket, const uint32_t* cursor, const uint32_t* count,
    Q38SequenceMeta* groups, const float* frequencies, float* rotary,
    int owner, int ratio, int rows, int rotary_dim) {
    q38_meta_guard(sequence);
    const int prefix = sequence->prefix, end = sequence->key_count;
    if (sequence->valid_rows > rows || (ratio != 1 && ratio != 2)
        || *cursor != uint32_t(prefix) || *count != uint32_t(prefix / ratio)
        || !*generation || (prefix && (ticket[0] != uint32_t(owner)
            || ticket[1] != *generation || ticket[2] != uint32_t(prefix)
            || ticket[3] != *count))) __trap();
    const int completed = end / ratio - prefix / ratio;
    if (threadIdx.x == 0) {
        groups->prefix = 0;
        groups->valid_rows = completed ? completed : 1;
        groups->key_count = groups->valid_rows;
        groups->capacity = rows;
        groups->causal = 1;
        for (int i = 0; i < 3; ++i) groups->reserved[i] = 0;
    }
    for (int i = threadIdx.x; i < ((rows + ratio - 1) / ratio) * rotary_dim / 2;
         i += blockDim.x) {
        const int row = i / (rotary_dim / 2), column = i % (rotary_dim / 2);
        const float angle = float((prefix / ratio + row) * ratio) * frequencies[column];
        rotary[2 * i] = cosf(angle);
        rotary[2 * i + 1] = sinf(angle);
    }
}

// Payload stores precede count and ticket publication within one CTA.
extern "C" __global__ void latent_publication_commit(
    const Q38SequenceMeta* sequence, const uint32_t* generation,
    uint32_t* ticket, uint32_t* cursor, uint32_t* count,
    const uint8_t* codes, const uint8_t* scales,
    const uint8_t* index_codes, const uint8_t* index_scales,
    uint8_t* cache_codes, uint8_t* cache_scales,
    uint8_t* cache_index_codes, uint8_t* cache_index_scales,
    int owner, int ratio, int capacity) {
    q38_meta_guard(sequence);
    const int first = sequence->prefix / ratio, end = sequence->key_count / ratio;
    if (end > capacity || *count != uint32_t(first)
        || *cursor != uint32_t(ratio == 2 ? sequence->key_count : sequence->prefix)) __trap();
    const int rows = end - first;
    for (int i = threadIdx.x; i < rows * 256; i += blockDim.x)
        cache_codes[int64_t(first) * 256 + i] = codes[i];
    for (int i = threadIdx.x; i < rows * 32; i += blockDim.x)
        cache_scales[int64_t(first) * 32 + i] = scales[i];
    for (int i = threadIdx.x; i < rows * 64; i += blockDim.x)
        cache_index_codes[int64_t(first) * 64 + i] = index_codes[i];
    for (int i = threadIdx.x; i < rows * 4; i += blockDim.x)
        cache_index_scales[int64_t(first) * 4 + i] = index_scales[i];
    __syncthreads();
    if (threadIdx.x == 0) {
        *cursor = sequence->key_count;
        *count = end;
        ticket[0] = owner; ticket[1] = *generation;
        ticket[2] = sequence->key_count; ticket[3] = end;
    }
}

// Selection tickets are [index owner, KV owner, generation, prefix, end].
extern "C" __global__ void latent_selection_ticket(
    const Q38SequenceMeta* sequence, const uint32_t* generation,
    const uint32_t* kv_ticket, uint32_t* selection_ticket,
    int kv_owner, int index_owner, int ratio, int publish) {
    q38_meta_guard(sequence);
    if (kv_ticket[0] != uint32_t(kv_owner) || kv_ticket[1] != *generation
        || kv_ticket[2] != uint32_t(sequence->key_count)
        || kv_ticket[3] != uint32_t(sequence->key_count / ratio)) __trap();
    if (publish == 2) return;
    if (publish == 1) {
        selection_ticket[0] = index_owner; selection_ticket[1] = kv_owner;
        selection_ticket[2] = *generation; selection_ticket[3] = sequence->prefix;
        selection_ticket[4] = sequence->key_count;
    } else if (selection_ticket[0] != uint32_t(index_owner)
        || selection_ticket[1] != uint32_t(kv_owner) || selection_ticket[2] != *generation
        || selection_ticket[3] != uint32_t(sequence->prefix)
        || selection_ticket[4] != uint32_t(sequence->key_count)) __trap();
}
