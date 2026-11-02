// Token sampling kernels over BF16 logits.
//
// argmax: launch <<<1, T>>>, T <= 1024. Comparator: strictly greater value wins;
// ties resolve to the lower index, including all-equal and -inf logits. 16B loads
// (8 bf16 per uint4) when vocab % 8 == 0 and the pointer is 16B-aligned; scalar
// loads otherwise, same semantics.
//
// advance: thread 0 increments the device position scalar.
#include "gpu_compat.cuh"
#include <stdint.h>

extern "C" __global__ void argmax(const __nv_bfloat16* logits, uint32_t* token, int vocab) {
    const int tid  = threadIdx.x;
    const int nthr = blockDim.x;
    float best = -INFINITY;
    uint32_t id = 0xFFFFFFFFu;
    const bool vec_ok = (vocab % 8 == 0) &&
                        ((reinterpret_cast<uintptr_t>(logits) & 15) == 0);
    if (vec_ok) {
        const int nv = vocab >> 3;
        const uint4* p = reinterpret_cast<const uint4*>(logits);
        for (int i = tid; i < nv; i += nthr) {
            const uint4 r = p[i];
            const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&r);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __bfloat1622float2(h[j]);
                const uint32_t base = (uint32_t)i * 8 + (uint32_t)j * 2;
                if (f.x > best || (f.x == best && base < id))     { best = f.x; id = base; }
                if (f.y > best || (f.y == best && base + 1 < id)) { best = f.y; id = base + 1; }
            }
        }
    } else {
        for (int i = tid; i < vocab; i += nthr) {
            const float v = __bfloat162float(logits[i]);
            if (v > best || (v == best && (uint32_t)i < id)) { best = v; id = i; }
        }
    }
    // warp reduce with the same comparator
    for (int off = 16; off > 0; off >>= 1) {
        const float ob    = __shfl_down_sync(FF_WARP_MASK, best, off);
        const uint32_t oi = __shfl_down_sync(FF_WARP_MASK, id, off);
        if (ob > best || (ob == best && oi < id)) { best = ob; id = oi; }
    }
    __shared__ float wv[32];
    __shared__ uint32_t wi[32];
    const int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) { wv[warp] = best; wi[warp] = id; }
    __syncthreads();
    if (warp == 0) {
        const int nw = (nthr + 31) >> 5;
        best = -INFINITY; id = 0xFFFFFFFFu;
        for (int w = lane; w < nw; w += 32) {
            const float v = wv[w];
            const uint32_t j = wi[w];
            if (v > best || (v == best && j < id)) { best = v; id = j; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ob    = __shfl_down_sync(FF_WARP_MASK, best, off);
            const uint32_t oi = __shfl_down_sync(FF_WARP_MASK, id, off);
            if (ob > best || (ob == best && oi < id)) { best = ob; id = oi; }
        }
        if (lane == 0) *token = id;
    }
}

extern "C" __global__ void lm_head_f32(const __nv_bfloat16* __restrict__ x,
                                        const __nv_bfloat16* __restrict__ weight,
                                        float* __restrict__ logits, int vocab,
                                        int hidden) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const __nv_bfloat16* xr = x;
  const __nv_bfloat16* wr = weight + (size_t)row * hidden;
  float part = 0.0f;
  for (int c = tid; c < hidden; c += blockDim.x)
    part = fmaf(__bfloat162float(xr[c]), __bfloat162float(wr[c]), part);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    part += __shfl_down_sync(FF_WARP_MASK, part, off);
  __shared__ float red[8];
  if ((tid & 31) == 0) red[tid >> 5] = part;
  __syncthreads();
  if (tid < 32) {
    float value = tid < (blockDim.x >> 5) ? red[tid] : 0.0f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      value += __shfl_xor_sync(FF_WARP_MASK, value, off);
    if (tid == 0) logits[row] = value;
  }
}

extern "C" __global__ void argmax_f32(const float* logits, uint32_t* token, int vocab) {
  const int tid = threadIdx.x;
  const int nthr = blockDim.x;
  float best = -INFINITY;
  uint32_t id = 0xFFFFFFFFu;
  for (int v = tid; v < vocab; v += nthr) {
    const float value = logits[v];
    if (value > best) {
      best = value;
      id = (uint32_t)v;
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float other_best = __shfl_down_sync(FF_WARP_MASK, best, off);
    const uint32_t other_id = __shfl_down_sync(FF_WARP_MASK, id, off);
    if (other_best > best || (other_best == best && other_id < id)) {
      best = other_best;
      id = other_id;
    }
  }
  __shared__ float best_red[32];
  __shared__ uint32_t id_red[32];
  if ((tid & 31) == 0) {
    best_red[tid >> 5] = best;
    id_red[tid >> 5] = id;
  }
  __syncthreads();
  if (tid < 32) {
    best = tid < (nthr >> 5) ? best_red[tid] : -INFINITY;
    id = tid < (nthr >> 5) ? id_red[tid] : 0xFFFFFFFFu;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float other_best = __shfl_xor_sync(FF_WARP_MASK, best, off);
      const uint32_t other_id = __shfl_xor_sync(FF_WARP_MASK, id, off);
      if (other_best > best || (other_best == best && other_id < id)) {
        best = other_best;
        id = other_id;
      }
    }
    if (tid == 0) *token = id;
  }
}

// Per-row argmax, launch <<<budget, 1024>>>: block b reduces row b of a
// [budget][vocab] bf16 logits slab into tokens[b]. Same comparator as argmax.
// Rows b >= *valid_rows_dev return before any write.
extern "C" __global__ void __launch_bounds__(1024) argmax_rows_bf16(
    const __nv_bfloat16* __restrict__ logits, uint32_t* __restrict__ tokens,
    int vocab, const int* __restrict__ valid_rows_dev) {
    const int row = blockIdx.x;
    if (row >= *valid_rows_dev) return;

    const __nv_bfloat16* __restrict__ p = logits + (size_t)row * vocab;
    const int tid  = threadIdx.x;
    const int nthr = blockDim.x;
    float best = -INFINITY;
    uint32_t id = 0xFFFFFFFFu;
    const bool vec_ok = (vocab % 8 == 0) &&
                        ((reinterpret_cast<uintptr_t>(p) & 15) == 0);
    if (vec_ok) {
        const int nv = vocab >> 3;
        const uint4* pv = reinterpret_cast<const uint4*>(p);
        for (int i = tid; i < nv; i += nthr) {
            const uint4 r = pv[i];
            const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&r);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __bfloat1622float2(h[j]);
                const uint32_t base = (uint32_t)i * 8 + (uint32_t)j * 2;
                if (f.x > best || (f.x == best && base < id))     { best = f.x; id = base; }
                if (f.y > best || (f.y == best && base + 1 < id)) { best = f.y; id = base + 1; }
            }
        }
    } else {
        for (int i = tid; i < vocab; i += nthr) {
            const float v = __bfloat162float(p[i]);
            if (v > best || (v == best && (uint32_t)i < id)) { best = v; id = i; }
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float ob    = __shfl_down_sync(FF_WARP_MASK, best, off);
        const uint32_t oi = __shfl_down_sync(FF_WARP_MASK, id, off);
        if (ob > best || (ob == best && oi < id)) { best = ob; id = oi; }
    }
    __shared__ float wv[32];
    __shared__ uint32_t wi[32];
    const int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) { wv[warp] = best; wi[warp] = id; }
    __syncthreads();
    if (warp == 0) {
        const int nw = (nthr + 31) >> 5;
        best = -INFINITY; id = 0xFFFFFFFFu;
        for (int w = lane; w < nw; w += 32) {
            const float v = wv[w];
            const uint32_t j = wi[w];
            if (v > best || (v == best && j < id)) { best = v; id = j; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ob    = __shfl_down_sync(FF_WARP_MASK, best, off);
            const uint32_t oi = __shfl_down_sync(FF_WARP_MASK, id, off);
            if (ob > best || (ob == best && oi < id)) { best = ob; id = oi; }
        }
        if (lane == 0) tokens[row] = id;
    }
}

__device__ __forceinline__ __nv_bfloat16 softcap_logit(__nv_bfloat16 x, float cap) {
    const float z = __bfloat162float(__float2bfloat16_rn(__bfloat162float(x) / cap));
    const float t = __bfloat162float(__float2bfloat16_rn(tanhf(z)));
    return __float2bfloat16_rn(t * cap);
}

// Softcap over [vocab] BF16 logits fused with single-row argmax<<<1, T>>>, T <= 1024.
// Same rounding as act.cu:softcap (90-97) and same tie comparator as sample.cu:argmax (12-65):
// strictly greater post-softcap BF16 value wins; ties resolve to the lower index.
extern "C" __global__ void softcap_argmax(const __nv_bfloat16* logits, uint32_t* token,
                                          int vocab, float cap) {
    if (blockDim.x > 1024 || gridDim.x != 1 || vocab < 1 || !(cap > 0.0f) || !isfinite(cap)) {
        __trap();
        return;
    }
    const int tid  = threadIdx.x;
    const int nthr = blockDim.x;
    float best = -INFINITY;
    uint32_t id = 0xFFFFFFFFu;
    const bool vec_ok = (vocab % 8 == 0) &&
                        ((reinterpret_cast<uintptr_t>(logits) & 15) == 0);
    if (vec_ok) {
        const int nv = vocab >> 3;
        const uint4* p = reinterpret_cast<const uint4*>(logits);
        for (int i = tid; i < nv; i += nthr) {
            const uint4 r = p[i];
            const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&r);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const __nv_bfloat16 s = softcap_logit(e[j], cap);
                const float v = __bfloat162float(s);
                const uint32_t idx = (uint32_t)i * 8 + (uint32_t)j;
                if (v > best || (v == best && idx < id)) { best = v; id = idx; }
            }
        }
    } else {
        for (int i = tid; i < vocab; i += nthr) {
            const __nv_bfloat16 s = softcap_logit(logits[i], cap);
            const float v = __bfloat162float(s);
            if (v > best || (v == best && (uint32_t)i < id)) { best = v; id = (uint32_t)i; }
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float ob    = __shfl_down_sync(FF_WARP_MASK, best, off);
        const uint32_t oi = __shfl_down_sync(FF_WARP_MASK, id, off);
        if (ob > best || (ob == best && oi < id)) { best = ob; id = oi; }
    }
    __shared__ float wv[32];
    __shared__ uint32_t wi[32];
    const int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) { wv[warp] = best; wi[warp] = id; }
    __syncthreads();
    if (warp == 0) {
        const int nw = (nthr + 31) >> 5;
        best = -INFINITY; id = 0xFFFFFFFFu;
        for (int w = lane; w < nw; w += 32) {
            const float v = wv[w];
            const uint32_t j = wi[w];
            if (v > best || (v == best && j < id)) { best = v; id = j; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ob    = __shfl_down_sync(FF_WARP_MASK, best, off);
            const uint32_t oi = __shfl_down_sync(FF_WARP_MASK, id, off);
            if (ob > best || (ob == best && oi < id)) { best = ob; id = oi; }
        }
        if (lane == 0) *token = id;
    }
}

extern "C" __global__ void advance(int* position) {
    if (threadIdx.x == 0) ++*position;
}

extern "C" __global__ void flint_decode_length(const int* position, int* length, int ring) {
  if (threadIdx.x == 0) {
    const int logical = *position + 1;
    *length = logical < ring ? logical : ring;
  }
}

// A call-scoped sticky error flag. The caller clears it before the first scan.
extern "C" __global__ void finite_bf16(
    const __nv_bfloat16* values, uint32_t* nonfinite, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && !isfinite(__bfloat162float(values[i]))) {
        atomicOr(nonfinite, 1u);
    }
}

// Per-row argmax over FP32 logits: strictly greater wins, ties to the lower
// index (torch.argmax semantics). Grid = rows, block = vocab_threads. Two
// passes: block max of the row, then the lowest index holding exactly it.
extern "C" __global__ void argmax_rows_f32(
    const float* __restrict__ logits, int* __restrict__ out, int rows,
    long long vocab, int vocab_threads) {
  if (rows < 1 || vocab < 1 || vocab_threads < 1 || (blockDim.x & 31) != 0 ||
      (int)blockDim.x != vocab_threads || vocab_threads > 1024)
    __trap();
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const float* l = logits + (long long)row * vocab;
  const long long chunk = (vocab + vocab_threads - 1) / vocab_threads;
  const long long lo = (long long)tid * chunk;
  const long long hi = lo + chunk < vocab ? lo + chunk : vocab;

  float best = -INFINITY;
  for (long long c = lo; c < hi; ++c) best = fmaxf(best, l[c]);
  __shared__ float s_v[32];
  for (int off = 16; off > 0; off >>= 1)
    best = fmaxf(best, __shfl_down_sync(FF_WARP_MASK, best, off));
  if (lane == 0) s_v[warp] = best;
  __syncthreads();
  __shared__ float s_best;
  if (tid == 0) {
    const int nw = (int)blockDim.x >> 5;
    float m = -INFINITY;
    for (int w = 0; w < nw; ++w) m = fmaxf(m, s_v[w]);
    s_best = m;
  }
  __syncthreads();  // every warp reads s_best in the second pass
  __syncthreads();

  int mine = 0x7fffffff;
  for (long long c = lo; c < hi; ++c) {
    if (l[c] == s_best) { mine = (int)c; break; }
  }
  __shared__ int s_pick[32];
  const int nw = (int)blockDim.x >> 5;
  for (int off = 16; off > 0; off >>= 1) {
    const int other = __shfl_down_sync(FF_WARP_MASK, mine, off);
    if (other < mine) mine = other;
  }
  if (lane == 0) s_pick[warp] = mine;
  __syncthreads();
  if (warp == 0) {
    mine = lane < nw ? s_pick[lane] : 0x7fffffff;
    for (int off = 16; off > 0; off >>= 1) {
      const int other = __shfl_down_sync(FF_WARP_MASK, mine, off);
      if (other < mine) mine = other;
    }
    if (lane == 0)
      out[row] = mine == 0x7fffffff ? (int)(vocab - 1) : mine;
  }
}

// Row softmax in FP32 with a parallel BF16 copy: probs = exp(x-max)/sum.
extern "C" __global__ void softmax_rows_dual(const float* __restrict__ x,
    float* __restrict__ probs, __nv_bfloat16* __restrict__ probs_bf16, int rows,
    long long vocab) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const float* row = x + (long long)r * vocab;
    float m = -INFINITY;
    for (long long c = threadIdx.x; c < vocab; c += blockDim.x)
        m = fmaxf(m, row[c]);
    __shared__ float s_m[32];
    __shared__ float s_s[32];
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_xor_sync(FF_WARP_MASK, m, off));
    if ((threadIdx.x & 31) == 0) s_m[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < (blockDim.x >> 5) ? s_m[threadIdx.x] : -INFINITY;
        for (int off = 16; off > 0; off >>= 1)
            t = fmaxf(t, __shfl_xor_sync(FF_WARP_MASK, t, off));
        if (threadIdx.x == 0) s_m[0] = t;
    }
    __syncthreads();
    m = s_m[0];
    float s = 0.0f;
    for (long long c = threadIdx.x; c < vocab; c += blockDim.x)
        s += expf(row[c] - m);
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_xor_sync(FF_WARP_MASK, s, off);
    if ((threadIdx.x & 31) == 0) s_s[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < (blockDim.x >> 5) ? s_s[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            t += __shfl_xor_sync(FF_WARP_MASK, t, off);
        if (threadIdx.x == 0) s_s[0] = t;
    }
    __syncthreads();
    const float inv = 1.0f / s_s[0];
    float* p = probs + (long long)r * vocab;
    __nv_bfloat16* pb = probs_bf16 + (long long)r * vocab;
    for (long long c = threadIdx.x; c < vocab; c += blockDim.x) {
        const float e = expf(row[c] - m) * inv;
        p[c] = e;
        pb[c] = __float2bfloat16_rn(e);
    }
}

// state = [committed response length, returned rows, stopped, pending rows].
extern "C" __global__ void masked_block_init(int* block, unsigned char* pending,
    unsigned int* state, int rows, int mask, unsigned int capacity) {
    const int r = threadIdx.x;
    if (state[0] > capacity || (unsigned int)rows > capacity - state[0] || state[2]) {
        __trap(); return;
    }
    if (r < rows) { block[r] = mask; pending[r] = 1; }
    if (r == 0) { state[1] = 0; state[3] = rows; }
}

// p = [presence, frequency, repetition, reciprocal, length strength/start/reciprocal window,
//      close strength/start/reciprocal window].
extern "C" __global__ void token_history_bias(float* logits,
    const unsigned int* counts, const int* block, const unsigned int* state,
    const unsigned int* stops, const float* p, int rows, int vocab,
    int mask, int pad, int close, int stop_count) {
    __shared__ bool close_enabled;
    if (threadIdx.x == 0) {
        bool visible = counts[close] > 0;
        for (int r = 0; r < rows; ++r)
            visible |= block[r] != mask && block[r] == close;
        close_enabled = !visible;
    }
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * vocab) return;
    const int token = i % vocab;
    if (token == mask) { logits[i] = -INFINITY; return; }
    float x = logits[i];
    if (isnan(x) || x == INFINITY) { __trap(); return; }
    const unsigned int count = counts[token];
    if (p[1] != 0.0f) x = __fsub_rn(x, __fmul_rn((float)count, p[1]));
    if (p[0] != 0.0f) x = __fsub_rn(x, __fmul_rn((float)(count > 0), p[0]));
    if (count && p[2] != 1.0f) x = __fmul_rn(x, x < 0.0f ? p[2] : p[3]);
    const float position = (float)(state[0] + i / vocab + 1);
    bool stop = token == pad;
    for (int j = 0; j < stop_count; ++j) stop |= token == (int)stops[j];
    if (stop && p[4] != 0.0f) {
        const float t = fminf(1.0f, fmaxf(0.0f, __fmul_rn(__fsub_rn(position, p[5]), p[6])));
        x = __fadd_rn(x, __fmul_rn(t, p[4]));
    }
    if (token == close && p[7] != 0.0f) {
        const float t = fminf(1.0f, fmaxf(0.0f, __fmul_rn(__fsub_rn(position, p[8]), p[9])));
        const float bias = __fmul_rn(t, p[7]);
        x = __fadd_rn(x, __fmul_rn((float)close_enabled, bias));
    }
    logits[i] = x;
}

extern "C" __global__ void confidence_threshold_fill(const float* probabilities,
    const int* picks, int* block, unsigned char* pending, unsigned int* state,
    int rows, int vocab, float threshold) {
    const int r = threadIdx.x;
    const bool active = r < rows && pending[r];
    float confidence = -INFINITY;
    if (active) {
        const int pick = picks[r];
        if (pick < 0 || pick >= vocab) { __trap(); return; }
        confidence = probabilities[(long long)r * vocab + pick];
        if (!isfinite(confidence)) { __trap(); return; }
    }
    const unsigned selected = __ballot_sync(FF_WARP_MASK, active && confidence > threshold);
    const unsigned live = __ballot_sync(FF_WARP_MASK, active);
    if (!live) { __trap(); return; }
    float best_value = confidence;
    int best = active ? r : 0x7fffffff;
    for (int off = 16; off > 0; off >>= 1) {
        const float value = __shfl_down_sync(FF_WARP_MASK, best_value, off);
        const int index = __shfl_down_sync(FF_WARP_MASK, best, off);
        if (value > best_value || (value == best_value && index < best)) {
            best_value = value; best = index;
        }
    }
    best = __shfl_sync(FF_WARP_MASK, best, 0);
    const bool take = active && (confidence > threshold || (!selected && r == best));
    if (take) { block[r] = picks[r]; pending[r] = 0; }
    const unsigned left = __ballot_sync(FF_WARP_MASK, active && !take);
    if (r == 0) state[3] = __popc(left);
}

extern "C" __global__ void response_count_commit(const int* block,
    const unsigned char* pending, unsigned int* counts, unsigned int* state,
    const unsigned int* stops, int stop_count, int rows, int vocab,
    int remaining, unsigned int capacity) {
    if (threadIdx.x != 0) return;
    for (int r = 0; r < rows; ++r) {
        if (pending[r]) { __trap(); return; }
    }
    if (state[1] || state[3] || state[2] || remaining < 1 || remaining > rows ||
        state[0] > capacity || (unsigned int)remaining > capacity - state[0]) {
        __trap(); return;
    }
    unsigned int returned = 0;
    for (int r = 0; r < remaining; ++r) {
        const int token = block[r];
        if (token < 0 || token >= vocab) { __trap(); return; }
        ++counts[token]; ++returned;
        bool stop = false;
        for (int j = 0; j < stop_count; ++j) stop |= token == (int)stops[j];
        if (stop) { state[2] = 1; break; }
    }
    state[0] += returned;
    state[1] = returned;
}
