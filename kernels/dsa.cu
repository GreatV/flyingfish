#include "q38_contract.cuh"
#include "dsa_topk.cuh"
#include <climits>

#define DSA_INDEX_HEADS 32
#define DSA_INDEX_DIM 128
#define DSA_POOL 4
#define DSA_POOL_TOPK 512
#define DSA_RAW_TOPK 2048
#define DSA_OUTPUT_WIDTH (DSA_RAW_TOPK + DSA_POOL - 1)
#define DSA_PACKED_WIDTH 257
#define DSA_Q_HEADS 64
#define DSA_HEAD_DIM 256
#define DSA_LATENT_DIM 512

struct alignas(16) DsaSequenceMeta {
    int prefix, valid_rows, key_count, capacity, first_key, reserved[3];
};
static_assert(sizeof(DsaSequenceMeta) == 32, "DSA metadata extent");

template<int Capacity>
__device__ void dsa_select_scores(const float* scores, const int* ids, int candidates,
                                  int keep, int* selected, float* values, float* margins) {
    if (threadIdx.x != 0) return;
    if (keep <= 0 || keep > Capacity || candidates <= 0) __trap();
    __shared__ DsaTopK<Capacity> heap;
    heap.size = 0;
    float rejected = -INFINITY;
    size_t row = blockIdx.x;
    for (int candidate = 0; candidate < candidates; ++candidate) {
        float score = scores[row * candidates + candidate];
        int id = ids[row * candidates + candidate];
        if (isnan(score) || score == INFINITY || id < -1) __trap();
        if (id >= 0 && score != -INFINITY) dsa_topk_offer(&heap, keep, score, id, &rejected);
    }
    int count = heap.size;
    for (int slot = count - 1; slot >= 0; --slot) {
        selected[row * keep + slot] = heap.id[0];
        values[row * keep + slot] = heap.score[0];
        dsa_topk_pop(&heap);
    }
    for (int slot = count; slot < keep; ++slot) {
        selected[row * keep + slot] = -1;
        values[row * keep + slot] = -INFINITY;
    }
    margins[row] = count > 0 && isfinite(rejected) ? values[row * keep + count - 1] - rejected : INFINITY;
}

extern "C" __global__ __launch_bounds__(32, 1)
void dsa_bounded_topk512(const float* scores, const int* ids, int candidates,
                         int keep, int* selected, float* values, float* margins) {
    dsa_select_scores<512>(scores, ids, candidates, keep, selected, values, margins);
}

extern "C" __global__ __launch_bounds__(32, 1)
void dsa_bounded_topk2048(const float* scores, const int* ids, int candidates,
                          int keep, int* selected, float* values, float* margins) {
    dsa_select_scores<2048>(scores, ids, candidates, keep, selected, values, margins);
}

__device__ __forceinline__ void dsa_meta_guard(const DsaSequenceMeta* meta, int capacity, int rows) {
    if (!meta || (reinterpret_cast<uintptr_t>(meta) & 15) || meta->prefix < 0 ||
        meta->valid_rows <= 0 || meta->valid_rows > rows || meta->valid_rows > meta->capacity ||
        meta->capacity != capacity ||
        meta->key_count <= 0 || meta->capacity <= 0 ||
        meta->first_key < 0 || meta->first_key > meta->key_count ||
        (int64_t)meta->prefix + meta->valid_rows != meta->key_count || meta->reserved[0] ||
        meta->reserved[1] || meta->reserved[2]) {
        __trap();
    }
}

__device__ __forceinline__ int dsa_live_begin(const DsaSequenceMeta* meta) {
    return max(meta->first_key, meta->key_count - meta->capacity);
}

extern "C" __global__ __launch_bounds__(256, 1)
void dsa_append(const __nv_bfloat16* input_latent, const __nv_bfloat16* input_packed,
                __nv_bfloat16* latent, __nv_bfloat16* packed, int* token_ids,
                const DsaSequenceMeta* meta, int latent_width, int capacity, int rows) {
    dsa_meta_guard(meta, capacity, rows);
    int row = blockIdx.x;
    if (row >= meta->valid_rows) return;
    int logical = meta->prefix + row;
    int physical = logical % meta->capacity;
    for (int d = threadIdx.x; d < latent_width; d += blockDim.x) {
        latent[(size_t)physical * latent_width + d] =
            input_latent[(size_t)row * latent_width + d];
    }
    for (int d = threadIdx.x; d < DSA_PACKED_WIDTH; d += blockDim.x) {
        const __nv_bfloat16 value = d == 2 * DSA_INDEX_DIM
            ? __float2bfloat16_rn(1.0f)
            : input_packed[(size_t)row * DSA_PACKED_WIDTH + d];
        packed[(size_t)physical * DSA_PACKED_WIDTH + d] = value;
    }
    if (threadIdx.x == 0) token_ids[physical] = logical;
}

extern "C" __global__ __launch_bounds__(128, 1)
void dsa_pool(const __nv_bfloat16* packed, __nv_bfloat16* pool_keys,
              const int* token_ids, int* pool_ids, const __nv_bfloat16* ape,
              const DsaSequenceMeta* meta, int pool_capacity, int capacity, int rows) {
    dsa_meta_guard(meta, capacity, rows);
    int pool = max(0, meta->prefix - meta->first_key) / DSA_POOL + blockIdx.x;
    int total = (meta->key_count - meta->first_key) / DSA_POOL +
                ((meta->key_count - meta->first_key) % DSA_POOL != 0);
    if (pool >= total) return;
    int channel = threadIdx.x;
    bool complete = true;
    float logits[DSA_POOL];
    float keys[DSA_POOL];
    for (int j = 0; j < DSA_POOL; ++j) {
        int64_t logical = (int64_t)meta->first_key + (int64_t)pool * DSA_POOL + j;
        bool inside = logical < meta->key_count;
        int physical = inside ? (int)(logical % meta->capacity) : 0;
        bool valid = inside && token_ids[physical] == logical &&
                     __bfloat162float(packed[(size_t)physical * DSA_PACKED_WIDTH +
                                             2 * DSA_INDEX_DIM]) != 0.0f;
        complete = complete && valid;
        size_t at = (size_t)physical * DSA_PACKED_WIDTH;
        logits[j] = valid ? __bfloat162float(packed[at + DSA_INDEX_DIM + channel]) +
                                __bfloat162float(ape[j * DSA_INDEX_DIM + channel])
                          : -INFINITY;
        keys[j] = __bfloat162float(packed[at + channel]);
    }
    int output = pool % pool_capacity;
    if (threadIdx.x == 0) pool_ids[output] = complete ? pool : -1;
    if (!complete) {
        pool_keys[(size_t)output * DSA_INDEX_DIM + channel] = __float2bfloat16_rn(0.0f);
        return;
    }
    float largest = fmaxf(fmaxf(logits[0], logits[1]), fmaxf(logits[2], logits[3]));
    float sum = 0.0f;
    for (int j = 0; j < DSA_POOL; ++j) {
        logits[j] = expf(logits[j] - largest);
        sum += logits[j];
    }
    float value = 0.0f;
    for (int j = 0; j < DSA_POOL; ++j) {
        float probability = bround(logits[j] / sum);
        value += bround(probability * keys[j]);
    }
    pool_keys[(size_t)output * DSA_INDEX_DIM + channel] = __float2bfloat16_rn(value);
}

extern "C" __global__ __launch_bounds__(1024, 1)
void dsa_index_score_topk(const __nv_bfloat16* q, const __nv_bfloat16* weights,
                          const __nv_bfloat16* packed, const __nv_bfloat16* pool_keys,
                          const int* token_ids, const int* pool_ids,
                          const DsaSequenceMeta* meta, int pool_capacity,
                          int* selected, float* selected_scores, float* margins,
                          int capacity, int rows) {
    dsa_meta_guard(meta, capacity, rows);
    int row = blockIdx.x;
    if (row >= meta->valid_rows) return;
    int tid = threadIdx.x, lane = tid & 31, head = tid >> 5;
    int position = meta->prefix + row;
    int live_begin = dsa_live_begin(meta);
    __shared__ float sq[DSA_INDEX_HEADS * DSA_INDEX_DIM];
    __shared__ float sw[DSA_INDEX_HEADS], sk[DSA_INDEX_DIM], sh[DSA_INDEX_HEADS];
    __shared__ DsaTopK<DSA_POOL_TOPK> heap;
    __shared__ float rejected;
    __shared__ bool query_valid;
    for (int i = tid; i < DSA_INDEX_HEADS * DSA_INDEX_DIM; i += blockDim.x) {
        sq[i] = __bfloat162float(q[(size_t)row * DSA_INDEX_HEADS * DSA_INDEX_DIM + i]);
    }
    if (tid < DSA_INDEX_HEADS) {
        sw[tid] = __bfloat162float(weights[(size_t)row * DSA_INDEX_HEADS + tid]) *
                  0.1767766922712326f;
    }
    if (tid == 0) {
        int physical = position % meta->capacity;
        query_valid = position >= live_begin && position < meta->key_count &&
                      token_ids[physical] == position &&
                      __bfloat162float(packed[(size_t)physical * DSA_PACKED_WIDTH +
                                              2 * DSA_INDEX_DIM]) != 0.0f;
        heap.size = 0;
        rejected = -INFINITY;
    }
    __syncthreads();
    int first_pool = (live_begin - meta->first_key) / DSA_POOL +
                     ((live_begin - meta->first_key) % DSA_POOL != 0);
    int total = (meta->key_count - meta->first_key) / DSA_POOL +
                ((meta->key_count - meta->first_key) % DSA_POOL != 0);
    int k = min(DSA_POOL_TOPK, total - first_pool);
    if (query_valid) {
        for (int pool = first_pool; pool < total; ++pool) {
            int64_t end = (int64_t)meta->first_key + (int64_t)pool * DSA_POOL + DSA_POOL - 1;
            int physical_pool = pool % pool_capacity;
            bool candidate = end <= position && end >= live_begin &&
                             pool_ids[physical_pool] == pool;
            if (tid < DSA_INDEX_DIM) {
                sk[tid] = candidate ? __bfloat162float(
                              pool_keys[(size_t)physical_pool * DSA_INDEX_DIM + tid])
                                    : 0.0f;
            }
            __syncthreads();
            float dot = 0.0f;
            for (int d = lane; d < DSA_INDEX_DIM; d += 32) {
                dot = fmaf(sq[head * DSA_INDEX_DIM + d], sk[d], dot);
            }
            for (int delta = 16; delta > 0; delta >>= 1) {
                dot += __shfl_down_sync(FF_WARP_MASK, dot, delta);
            }
            if (lane == 0) {
                sh[head] = fmaxf(dot * 0.0883883461356163f, 0.0f) * sw[head];
            }
            __syncthreads();
            if (tid == 0 && candidate) {
                float score = 0.0f;
                for (int h = 0; h < DSA_INDEX_HEADS; ++h) score += sh[h];
                if (!isfinite(score)) __trap();
                dsa_topk_offer(&heap, k, score, pool, &rejected);
            }
            __syncthreads();
        }
    }
    if (tid == 0) {
        int count = heap.size;
        for (int slot = count - 1; slot >= 0; --slot) {
            selected[(size_t)row * DSA_POOL_TOPK + slot] = heap.id[0];
            selected_scores[(size_t)row * DSA_POOL_TOPK + slot] = heap.score[0];
            dsa_topk_pop(&heap);
        }
        for (int slot = count; slot < DSA_POOL_TOPK; ++slot) {
            selected[(size_t)row * DSA_POOL_TOPK + slot] = -1;
            selected_scores[(size_t)row * DSA_POOL_TOPK + slot] = -INFINITY;
        }
        margins[row] = count > 0 && isfinite(rejected)
                           ? selected_scores[(size_t)row * DSA_POOL_TOPK + count - 1] - rejected
                           : INFINITY;
    }
}

extern "C" __global__ __launch_bounds__(128, 1)
void dsa_expand_indices(const int* selected, const __nv_bfloat16* packed,
                        const int* token_ids, const int* pool_ids, int pool_capacity,
                        const DsaSequenceMeta* meta, int* indices, int capacity, int rows) {
    dsa_meta_guard(meta, capacity, rows);
    int row = blockIdx.x;
    if (row >= meta->valid_rows) return;
    int position = meta->prefix + row;
    int live_begin = dsa_live_begin(meta);
    __shared__ int tail_count, tail_start, selected_width;
    __shared__ bool query_valid;
    if (threadIdx.x == 0) {
        int physical = position % meta->capacity;
        query_valid = position >= live_begin && position < meta->key_count &&
                      token_ids[physical] == position &&
                      __bfloat162float(packed[(size_t)physical * DSA_PACKED_WIDTH +
                                              2 * DSA_INDEX_DIM]) != 0.0f;
        int visible = live_begin - meta->first_key;
        if (query_valid) {
            for (int logical = live_begin; logical <= position; ++logical) {
                int slot = logical % meta->capacity;
                visible += token_ids[slot] == logical &&
                           __bfloat162float(packed[(size_t)slot * DSA_PACKED_WIDTH +
                                                   2 * DSA_INDEX_DIM]) != 0.0f;
            }
        }
        tail_count = query_valid ? visible % DSA_POOL : 0;
        tail_start = meta->first_key + visible - tail_count;
        int pools = 0;
        int total = (meta->key_count - meta->first_key) / DSA_POOL;
        for (int pool = (live_begin - meta->first_key) / DSA_POOL +
                        ((live_begin - meta->first_key) % DSA_POOL != 0);
             pool < total; ++pool) {
            pools += pool_ids[pool % pool_capacity] == pool;
        }
        selected_width = min(DSA_POOL_TOPK, pools) * DSA_POOL;
    }
    __syncthreads();
    for (int slot = threadIdx.x; slot < DSA_OUTPUT_WIDTH; slot += blockDim.x) {
        int logical = -1;
        if (query_valid) {
            if (slot < selected_width) {
                int pool = selected[(size_t)row * DSA_POOL_TOPK + slot / DSA_POOL];
                if (pool >= 0) logical = meta->first_key + pool * DSA_POOL + slot % DSA_POOL;
            } else {
                int offset = slot - selected_width;
                if (offset < tail_count) logical = tail_start + offset;
            }
            if (logical >= 0) {
                int physical = logical % meta->capacity;
                bool valid = logical >= live_begin && logical <= position &&
                             token_ids[physical] == logical &&
                             __bfloat162float(packed[(size_t)physical * DSA_PACKED_WIDTH +
                                                     2 * DSA_INDEX_DIM]) != 0.0f;
                if (!valid) logical = -1;
            }
        }
        indices[(size_t)row * DSA_OUTPUT_WIDTH + slot] = logical;
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void dsa_sparse_attention(const __nv_bfloat16* q, const __nv_bfloat16* latent,
                          const __nv_bfloat16* kv_b, const int* indices,
                          const int* token_ids, const float* sinks,
                          const DsaSequenceMeta* meta, __nv_bfloat16* out, int capacity, int rows) {
    dsa_meta_guard(meta, capacity, rows);
    int row = blockIdx.x, head = blockIdx.y, tid = threadIdx.x;
    if (row >= meta->valid_rows) return;
    int lane = tid & 31, warp = tid >> 5;
    int live_begin = dsa_live_begin(meta);
    __shared__ float sq[DSA_HEAD_DIM], sl[DSA_LATENT_DIM], sr[8], sp[4];
    for (int d = tid; d < DSA_HEAD_DIM; d += blockDim.x) {
        sq[d] = __bfloat162float(q[((size_t)row * DSA_Q_HEADS + head) * DSA_HEAD_DIM + d]);
    }
    if (tid == 0) {
        sp[0] = sinks ? sinks[head] : -INFINITY;
        sp[1] = sinks ? 1.0f : 0.0f;
    }
    __syncthreads();
    float accumulator = 0.0f;
    for (int slot = 0; slot < DSA_OUTPUT_WIDTH; ++slot) {
        int logical = indices[(size_t)row * DSA_OUTPUT_WIDTH + slot];
        bool valid = logical >= live_begin && logical < meta->key_count;
        int physical = valid ? logical % meta->capacity : 0;
        valid = valid && token_ids[physical] == logical;
        if (!valid) continue;
        for (int d = tid; d < DSA_LATENT_DIM; d += blockDim.x) {
            sl[d] = __bfloat162float(latent[(size_t)physical * DSA_LATENT_DIM + d]);
        }
        __syncthreads();
        float key = 0.0f;
        const __nv_bfloat16* key_weight = kv_b + ((size_t)head * 512 + tid) * 512;
        for (int d = 0; d < DSA_LATENT_DIM; ++d) {
            key = fmaf(__bfloat162float(key_weight[d]), sl[d], key);
        }
        key = bround(key);
        float dot = sq[tid] * key;
        for (int delta = 16; delta > 0; delta >>= 1) {
            dot += __shfl_down_sync(FF_WARP_MASK, dot, delta);
        }
        if (lane == 0) sr[warp] = dot;
        __syncthreads();
        if (tid == 0) {
            float score = 0.0f;
            for (int w = 0; w < 8; ++w) score += sr[w];
            score *= 0.0625f;
            float next_m = fmaxf(sp[0], score);
            float scale = expf(sp[0] - next_m);
            float probability = expf(score - next_m);
            sp[1] = sp[1] * scale + probability;
            sp[0] = next_m;
            sp[2] = scale;
            sp[3] = probability;
        }
        __syncthreads();
        accumulator *= sp[2];
        float value = 0.0f;
        const __nv_bfloat16* value_weight = kv_b + ((size_t)head * 512 + 256 + tid) * 512;
        for (int d = 0; d < DSA_LATENT_DIM; ++d) {
            value = fmaf(__bfloat162float(value_weight[d]), sl[d], value);
        }
        value = bround(value);
        accumulator = fmaf(sp[3], value, accumulator);
        __syncthreads();
    }
    float value = sp[1] > 0.0f ? accumulator / sp[1] : 0.0f;
    out[((size_t)row * DSA_Q_HEADS + head) * DSA_HEAD_DIM + tid] =
        __float2bfloat16_rn(value);
}

extern "C" __global__ __launch_bounds__(512, 1)
void dsa_sparse_attention_direct(const __nv_bfloat16* q, const __nv_bfloat16* latent,
                                  const int* indices, int width, const int* token_ids,
                                  const float* sinks, const DsaSequenceMeta* meta,
                                  __nv_bfloat16* out, int capacity, int rows) {
    dsa_meta_guard(meta, capacity, rows);
    int row = blockIdx.x, head = blockIdx.y, tid = threadIdx.x;
    if (row >= meta->valid_rows) return;
    int lane = tid & 31, warp = tid >> 5;
    __shared__ float query[512], score[64], reduction[16], stats[4];
    __shared__ int physical[64];
    query[tid] = __bfloat162float(q[((size_t)row * 64 + head) * 512 + tid]);
    if (tid == 0) { stats[0] = -1e30f; stats[1] = 0.0f; }
    __syncthreads();
    float accumulator = 0.0f;
    for (int64_t begin = 0; begin < width; begin += 64) {
        if (tid < 64) {
            int id = begin + tid < width ? indices[(size_t)row * width + begin + tid] : -1;
            bool valid = id >= dsa_live_begin(meta) && id < meta->key_count;
            int at = valid ? id % meta->capacity : 0;
            physical[tid] = valid && token_ids[at] == id ? at : -1;
        }
        __syncthreads();
        for (int slot = 0; slot < 64; ++slot) {
            int at = physical[slot];
            float dot = at >= 0 ? query[tid] * __bfloat162float(latent[(size_t)at * 512 + tid]) : 0.0f;
            for (int delta = 16; delta > 0; delta >>= 1) dot += __shfl_down_sync(FF_WARP_MASK, dot, delta);
            if (lane == 0) reduction[warp] = dot;
            __syncthreads();
            if (tid == 0) {
                float sum = 0.0f;
                for (int w = 0; w < 16; ++w) sum += reduction[w];
                score[slot] = at >= 0 ? sum * 0.04419417382415922f : -INFINITY;
            }
            __syncthreads();
        }
        if (tid == 0) {
            float largest = stats[0];
            for (int slot = 0; slot < 64; ++slot) largest = fmaxf(largest, score[slot]);
            stats[2] = expf(stats[0] - largest);
            stats[1] *= stats[2];
            for (int slot = 0; slot < 64; ++slot) {
                float probability = expf(score[slot] - largest);
                stats[1] += probability;
                score[slot] = bround(probability);
            }
            stats[0] = largest;
        }
        __syncthreads();
        accumulator *= stats[2];
        for (int slot = 0; slot < 64; ++slot) {
            int at = physical[slot];
            if (at >= 0) accumulator = fmaf(score[slot], __bfloat162float(latent[(size_t)at * 512 + tid]), accumulator);
        }
        __syncthreads();
    }
    if (tid == 0) stats[3] = stats[1] + (sinks ? expf(sinks[head] - stats[0]) : 0.0f);
    __syncthreads();
    out[((size_t)row * 64 + head) * 512 + tid] = __float2bfloat16_rn(stats[3] > 0.0f ? accumulator / stats[3] : 0.0f);
}

extern "C" __global__ __launch_bounds__(256, 1)
void dsa_checkpoint(__nv_bfloat16* latent, __nv_bfloat16* packed,
                    __nv_bfloat16* pool_keys, int* token_ids, int* pool_ids,
                    DsaSequenceMeta* meta, __nv_bfloat16* saved_latent,
                    __nv_bfloat16* saved_packed, __nv_bfloat16* saved_pool_keys,
                    int* saved_token_ids, int* saved_pool_ids, DsaSequenceMeta* saved_meta,
                    int rows, int pool_capacity, int latent_width, int mode, int capacity, int pool_size) {
    if (mode != 0 && mode != 1) __trap();
    int tid = threadIdx.x;
    if (mode == 0 && tid < 8) ((int*)saved_meta)[tid] = ((const int*)meta)[tid];
    __syncthreads();
    int base = saved_meta->key_count;
    if (saved_meta->capacity != capacity || base < 0 || rows <= 0 || rows > capacity ||
        (int64_t)base + rows > INT_MAX || saved_meta->first_key < 0 || saved_meta->first_key > base) __trap();
    const __nv_bfloat16* source_latent = mode == 0 ? latent : saved_latent;
    const __nv_bfloat16* source_packed = mode == 0 ? packed : saved_packed;
    const __nv_bfloat16* source_pools = mode == 0 ? pool_keys : saved_pool_keys;
    __nv_bfloat16* target_latent = mode == 0 ? saved_latent : latent;
    __nv_bfloat16* target_packed = mode == 0 ? saved_packed : packed;
    __nv_bfloat16* target_pools = mode == 0 ? saved_pool_keys : pool_keys;
    for (int row = 0; row < rows; ++row) {
        int logical = base + row;
        int physical = logical % capacity;
        size_t source_at = (mode == 0 ? (size_t)physical : (size_t)row) * latent_width;
        size_t target_at = (mode == 0 ? (size_t)row : (size_t)physical) * latent_width;
        for (int d = tid; d < latent_width; d += blockDim.x) {
            target_latent[target_at + d] = source_latent[source_at + d];
        }
        size_t source_packed_at = (mode == 0 ? (size_t)physical : (size_t)row) * DSA_PACKED_WIDTH;
        size_t target_packed_at = (mode == 0 ? (size_t)row : (size_t)physical) * DSA_PACKED_WIDTH;
        for (int d = tid; d < DSA_PACKED_WIDTH; d += blockDim.x) {
            target_packed[target_packed_at + d] = source_packed[source_packed_at + d];
        }
        if (tid == 0) {
            if (mode == 0) saved_token_ids[row] = token_ids[physical];
            else token_ids[physical] = saved_token_ids[row];
        }
    }
    int pool_base = max(0, base - saved_meta->first_key) / pool_size;
    int pool_count = rows / pool_size + (rows % pool_size != 0) + 1;
    for (int row = 0; row < pool_count; ++row) {
        int physical = (pool_base + row) % pool_capacity;
        size_t source_at = (mode == 0 ? (size_t)physical : (size_t)row) * DSA_INDEX_DIM;
        size_t target_at = (mode == 0 ? (size_t)row : (size_t)physical) * DSA_INDEX_DIM;
        for (int d = tid; d < DSA_INDEX_DIM; d += blockDim.x) {
            target_pools[target_at + d] = source_pools[source_at + d];
        }
        if (tid == 0) {
            if (mode == 0) saved_pool_ids[row] = pool_ids[physical];
            else pool_ids[physical] = saved_pool_ids[row];
        }
    }
    __syncthreads();
    if (mode == 1 && tid < 8) ((int*)meta)[tid] = ((const int*)saved_meta)[tid];
}
