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
#include <type_traits>

__device__ __forceinline__ float policy_sum(float value, float* shared) {
    for (int delta = 16; delta; delta >>= 1) {
        const float other = __shfl_down_sync(FF_WARP_MASK, value, delta);
        value += other;
    }
    if ((threadIdx.x & 31) == 0) shared[threadIdx.x >> 5] = value;
    __syncthreads();
    if (threadIdx.x < 32) {
        value = threadIdx.x < 8 ? shared[threadIdx.x] : 0.0f;
        for (int delta = 16; delta; delta >>= 1) {
            const float other = __shfl_down_sync(FF_WARP_MASK, value, delta);
            value += other;
        }
        if (threadIdx.x == 0) shared[0] = value;
    }
    __syncthreads();
    value = shared[0];
    __syncthreads();
    return value;
}

template<bool ACCURATE, bool BF16_COPY>
__device__ __forceinline__ void softmax_row_f32(const float* row, float* probs,
    __nv_bfloat16* probs_bf16, long long vocab) {
    using Sum = std::conditional_t<ACCURATE, double, float>;
    float m = -INFINITY;
    for (long long c = threadIdx.x; c < vocab; c += blockDim.x)
        m = fmaxf(m, row[c]);
    __shared__ float s_m[32];
    __shared__ Sum s_s[32];
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
    Sum s = 0;
    for (long long c = threadIdx.x; c < vocab; c += blockDim.x) {
        if constexpr (ACCURATE) {
            if (row[c] != -INFINITY) s += exp(double(row[c]) - double(m));
        } else {
            s += expf(row[c] - m);
        }
    }
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_xor_sync(FF_WARP_MASK, s, off);
    if ((threadIdx.x & 31) == 0) s_s[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        Sum t = threadIdx.x < (blockDim.x >> 5) ? s_s[threadIdx.x] : Sum(0);
        for (int off = 16; off > 0; off >>= 1)
            t += __shfl_xor_sync(FF_WARP_MASK, t, off);
        if (threadIdx.x == 0) s_s[0] = t;
    }
    __syncthreads();
    const Sum inv = Sum(1) / s_s[0];
    for (long long c = threadIdx.x; c < vocab; c += blockDim.x) {
        float e;
        if constexpr (ACCURATE) {
            e = row[c] == -INFINITY ? 0.0f : float(exp(double(row[c]) - double(m)) * inv);
        } else {
            e = expf(row[c] - m) * inv;
        }
        probs[c] = e;
        if constexpr (BF16_COPY) probs_bf16[c] = __float2bfloat16_rn(e);
    }
}

__device__ unsigned policy_key(float value) {
    const unsigned bits = __float_as_uint(value);
    return bits ^ ((bits >> 31) ? 0xffffffffu : 0x80000000u);
}

__device__ float policy_kth(const float* values, int count, int rank, float* shared) {
    unsigned prefix = 0;
    for (int bit = 31; bit >= 0; --bit) {
        const unsigned candidate = prefix | (1u << bit);
        float above = 0;
        for (int i = threadIdx.x; i < count; i += 256)
            above += policy_key(values[i]) >= candidate;
        if (policy_sum(above, shared) >= rank) prefix = candidate;
    }
    return __uint_as_float(prefix ^ ((prefix >> 31) ? 0x80000000u : 0xffffffffu));
}

// Full-vocabulary probabilities; interval and optional extra token define the mask.
extern "C" __global__ void guided_topk_bf16(
    const __nv_bfloat16* logits, float* probabilities, float* scratch,
    float* thresholds, unsigned char* keep, int vocab, int begin, int count,
    int extra, int conditional_k, int sample_k, float scale) {
    if (blockDim.x != 256 || gridDim.x != 1 || begin < 0 || count < sample_k ||
        begin + count > vocab || extra >= vocab || conditional_k < 0 ||
        conditional_k > count || sample_k < 1 || !isfinite(scale)) {
        __trap(); return;
    }
    __shared__ float shared[8];
    const int size = count + (extra >= 0);
    float* guided = scratch + size;
    for (int i = threadIdx.x; i < vocab; i += 256) {
        if (!isfinite(__bfloat162float(logits[i])) || !isfinite(__bfloat162float(logits[vocab + i]))) __trap();
        probabilities[i] = -INFINITY;
        keep[i] = 0;
    }
    for (int i = threadIdx.x; i < size; i += 256) {
        const int id = i < count ? begin + i : extra;
        const float cond = __bfloat162float(logits[id]);
        const float uncond = __bfloat162float(logits[vocab + id]);
        if (!isfinite(cond) || !isfinite(uncond)) __trap();
        scratch[i] = cond;
        guided[i] = __fadd_rn(uncond, __fmul_rn(__fsub_rn(cond, uncond), scale));
    }
    __syncthreads();
    const float conditional = conditional_k ? policy_kth(scratch, size, conditional_k, shared) : -INFINITY;
    for (int i = threadIdx.x; i < size; i += 256)
        if (scratch[i] < conditional) guided[i] = -1e9f;
    __syncthreads();
    const float threshold = policy_kth(guided, size, sample_k, shared);
    if (!(threshold > -1e9f)) { __trap(); return; }
    for (int i = threadIdx.x; i < size; i += 256) {
        const bool selected = guided[i] >= threshold;
        const int id = i < count ? begin + i : extra;
        keep[id] = selected;
        probabilities[id] = selected ? guided[i] : -INFINITY;
    }
    __syncthreads();
    softmax_row_f32<true, false>(probabilities, probabilities, nullptr, vocab);
    if (threadIdx.x == 0) {
        thresholds[0] = conditional;
        thresholds[1] = threshold;
    }
}

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

// Text-diffusion canvas sampler ops (definition: DiffusionGemma generation loop).
// All logits are FP32 post-softcap rows of [rows, vocab].

// out[i] = logits[i] / t (definition: scores / temperature, a division).
extern "C" __global__ void temperature_scale_f32(
    const float* __restrict__ logits, float* __restrict__ out,
    long long count, float t) {
  if (!(t > 0.0f) || !isfinite(t)) __trap();
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) out[i] = logits[i] / t;
}

// Per-row categorical entropy in FP32: H = -sum(softmax(l) * log_softmax(l)),
// two passes (row max, then exp/sum and the weighted deviation sum), matching
// torch.distributions.Categorical.entropy over FP32 logits.
extern "C" __global__ void row_entropy_f32(
    const float* __restrict__ logits, float* __restrict__ entropy, int rows,
    long long vocab, int vocab_threads) {
  if (rows < 1 || vocab < 1 || vocab_threads < 1 || (blockDim.x & 31) != 0 ||
      (int)blockDim.x < vocab_threads || vocab % 16 != 0)
    __trap();
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const float* l = logits + (long long)row * vocab;
  float mx = -INFINITY;
  for (long long c = tid; c < vocab; c += vocab_threads)
    mx = fmaxf(mx, l[c]);
  __shared__ float s_red[32];
  for (int off = 16; off > 0; off >>= 1)
    mx = fmaxf(mx, __shfl_down_sync(FF_WARP_MASK, mx, off));
  if (lane == 0) s_red[warp] = mx;
  __syncthreads();
  mx = warp == 0 && lane < ((int)blockDim.x >> 5) ? s_red[lane] : -INFINITY;
  for (int off = 16; off > 0; off >>= 1)
    mx = fmaxf(mx, __shfl_down_sync(FF_WARP_MASK, mx, off));
  // Broadcast the block max to every warp (see moe_route_topk_softmax_scale_f32).
  if (warp == 0 && lane == 0) s_red[0] = mx;
  __syncthreads();
  mx = s_red[0];
  __syncthreads();  // s_red is reused for the sum stage below
  float acc = 0.0f;
  float dev = 0.0f;
  for (long long c = tid; c < vocab; c += vocab_threads) {
    const float e = expf(l[c] - mx);
    acc += e;
    dev += e * (l[c] - mx);
  }
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(FF_WARP_MASK, acc, off);
    dev += __shfl_down_sync(FF_WARP_MASK, dev, off);
  }
  if (lane == 0) { s_red[warp] = acc; }
  __shared__ float s_dev[32];
  if (lane == 0) s_dev[warp] = dev;
  __syncthreads();
  const int nw = (int)blockDim.x >> 5;
  if (warp == 0) {
    acc = lane < nw ? s_red[lane] : 0.0f;
    dev = lane < nw ? s_dev[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      acc += __shfl_down_sync(FF_WARP_MASK, acc, off);
      dev += __shfl_down_sync(FF_WARP_MASK, dev, off);
    }
    if (lane == 0) {
      const float lse = logf(acc);
      entropy[row] = lse - dev / acc;
    }
  }
}

// Entropy-bound acceptance (definition EntropyBoundSampler.accept_canvas) over
// one canvas row set: sort entropies ascending, accept the prefix while
// cumsum(sorted) - sorted[k] <= bound, scatter back. n must be a power of two
// (canvas 256); one block, n threads, bitonic sort of (entropy, index).
extern "C" __global__ void entropy_bound_accept(
    const float* __restrict__ entropy, unsigned char* __restrict__ accept,
    int n, float bound) {
  if (n < 2 || (n & (n - 1)) != 0 || (int)blockDim.x != n ||
      !isfinite(bound) || bound <= 0.0f)
    __trap();
  __shared__ float s_e[1024];
  __shared__ int s_i[1024];
  __shared__ float s_cum[1024];
  __shared__ unsigned char s_mask[1024];
  const int i = threadIdx.x;
  s_e[i] = entropy[i];
  s_i[i] = i;
  __syncthreads();
  for (int k = 2; k <= n; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      const int a = i ^ j;
      if (a > i) {
        const bool up = (i & k) == 0;
        const float ei = s_e[i], ea = s_e[a];
        const int ii = s_i[i], ia = s_i[a];
        const bool swap = up ? (ea < ei || (ea == ei && ia < ii))
                             : (ea > ei || (ea == ei && ia > ii));
        if (swap) {
          s_e[i] = ea; s_e[a] = ei;
          s_i[i] = ia; s_i[a] = ii;
        }
      }
      __syncthreads();
    }
  }
  // Sequential inclusive cumsum on thread 0 in ascending sorted order: the
  // definition (torch.cumsum) adds in exactly this order, so the boundary
  // decision is bit-stable.
  if (i == 0) {
    float run = 0.0f;
    for (int k = 0; k < n; ++k) {
      run += s_e[k];
      s_cum[k] = run;
    }
  }
  __syncthreads();
  s_mask[i] = (s_cum[i] - s_e[i]) <= bound;
  accept[s_i[i]] = s_mask[i];
}

// Philox4_32-10 counter RNG (public algorithm, independent implementation).
// One thread produces 4 draws; out = floor(u01 * vocab) with u01 = word / 2^32.
__device__ __forceinline__ uint4 philox4_32_10(uint4 counter, uint2 key) {
  const unsigned mask = 0x9E3779B9u, round = 0xBB67AE85u;
#pragma unroll
  for (int r = 0; r < 10; ++r) {
    if (r > 0) {
      key.x += mask;
      key.y += round;
    }
    const unsigned hi0 = __umulhi(0xD2511F53u, counter.x);
    const unsigned lo0 = 0xD2511F53u * counter.x;
    const unsigned hi1 = __umulhi(0xCD9E8D57u, counter.z);
    const unsigned lo1 = 0xCD9E8D57u * counter.z;
    counter = make_uint4(hi1 ^ counter.y ^ key.x, lo1, hi0 ^ counter.w ^ key.y,
                         lo0);
  }
  return counter;
}

extern "C" __global__ void philox_uniform_ids(
    unsigned long long seed, unsigned long long stream, int* __restrict__ out,
    long long count, int vocab) {
  if (vocab < 1 || count < 0) __trap();
  const long long base = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const uint2 key = make_uint2((unsigned)(seed & 0xffffffffu),
                               (unsigned)(seed >> 32));
  const uint4 counter = make_uint4((unsigned)(stream & 0xffffffffu),
                                   (unsigned)(stream >> 32),
                                   (unsigned)(base & 0xffffffffu),
                                   (unsigned)(base >> 32));
  const uint4 r = philox4_32_10(counter, key);
  for (int j = 0; j < 4; ++j) {
    const long long idx = base * 4 + j;
    if (idx < count) {
      const unsigned word = j == 0 ? r.x : j == 1 ? r.y : j == 2 ? r.z : r.w;
      // id = floor(word * vocab / 2^32), always < vocab.
      out[idx] = (int)(((unsigned long long)word * (unsigned long long)vocab) >> 32);
    }
  }
}

// Inverse-CDF categorical sample per row: one uniform draw from the same
// Philox stream, then the first prefix that reaches it (two passes: row sum,
// then scan). Deterministic given (seed, stream).
extern "C" __global__ void categorical_sample_rows(
    const float* __restrict__ probs, int* __restrict__ out, int rows,
    long long vocab, unsigned long long seed, unsigned long long stream,
    int vocab_threads) {
  if (rows < 1 || vocab < 1 || vocab_threads < 1 || (blockDim.x & 31) != 0 ||
      (int)blockDim.x != vocab_threads || vocab_threads > 1024)
    __trap();
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const uint2 key =
      make_uint2((unsigned)(seed & 0xffffffffu), (unsigned)(seed >> 32));
  const uint4 counter =
      make_uint4((unsigned)(stream & 0xffffffffu), (unsigned)(stream >> 32),
                 (unsigned)row, 0u);
  const float u = fminf(
      (float)(philox4_32_10(counter, key).x * (1.0 / 4294967296.0)),
      0x1.fffffep-1f);
  const float* p = probs + (long long)row * vocab;
  const long long chunk = (vocab + vocab_threads - 1) / vocab_threads;
  const long long lo = (long long)tid * chunk;
  const long long hi = lo + chunk < vocab ? lo + chunk : vocab;

  __shared__ float s_stage[32];
  __shared__ float s_scan[1024];
  float sum = 0.0f;
  for (long long c = lo; c < hi; ++c) {
    const float probability = p[c];
    if (!isfinite(probability) || probability < 0.0f) { __trap(); return; }
    sum += probability;
  }

  // Block total of the row probabilities.
  float t = sum;
  for (int off = 16; off > 0; off >>= 1)
    t += __shfl_down_sync(FF_WARP_MASK, t, off);
  if (lane == 0) s_stage[warp] = t;
  __syncthreads();
  if (warp == 0) {
    const int nw = (int)blockDim.x >> 5;
    t = lane < nw ? s_stage[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1)
      t += __shfl_down_sync(FF_WARP_MASK, t, off);
    if (lane == 0) s_stage[0] = t;
  }
  __syncthreads();
  if (!isfinite(s_stage[0]) || !(s_stage[0] > 0.0f)) { __trap(); return; }
  const float norm = 1.0f / s_stage[0];

  // Exclusive block scan over the per-thread chunk sums (s_scan).
  s_scan[tid] = sum;
  __syncthreads();
  for (int d = 1; d < vocab_threads; d <<= 1) {
    const float add = tid >= d ? s_scan[tid - d] : 0.0f;
    __syncthreads();
    s_scan[tid] += add;
    __syncthreads();
  }
  float base = s_scan[tid] - sum;

  // First prefix reaching u inside this thread's chunk.
  int mine = 0x7fffffff;
  float run = base * norm;
  for (long long c = lo; c < hi && mine == 0x7fffffff; ++c) {
    const float next = run + p[c] * norm;
    if (u >= run && u < next) mine = (int)c;
    run = next;
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
    if (lane == 0) {
      if (mine == 0x7fffffff) { __trap(); return; }
      out[row] = mine;
    }
  }
}

// Adaptive stopping (definition StableAndConfidentStoppingCriteria):
// stable = the argmax canvas is unchanged from the previous step on every
// position; confident = mean entropy below the threshold. One block.
extern "C" __global__ void canvas_stability_stop(
    const int* __restrict__ argmax_prev, const int* __restrict__ argmax_cur,
    const float* __restrict__ entropy, unsigned char* __restrict__ stop, int n,
    float confidence) {
  if (n < 1 || (blockDim.x & 31) != 0 || !isfinite(confidence) ||
      confidence <= 0.0f)
    __trap();
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  int same = 1;
  float h = 0.0f;
  if (tid < n) {
    same = argmax_prev[tid] == argmax_cur[tid];
    h = entropy[tid];
  } else {
    h = 0.0f;
  }
  for (int off = 16; off > 0; off >>= 1) {
    same &= __shfl_down_sync(FF_WARP_MASK, same, off);
    h += __shfl_down_sync(FF_WARP_MASK, h, off);
  }
  __shared__ int s_same[32];
  __shared__ float s_h[32];
  if (lane == 0) { s_same[warp] = same; s_h[warp] = h; }
  __syncthreads();
  const int nw = (int)blockDim.x >> 5;
  if (warp == 0) {
    same = lane < nw ? s_same[lane] : 1;
    h = lane < nw ? s_h[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      same &= __shfl_down_sync(FF_WARP_MASK, same, off);
      h += __shfl_down_sync(FF_WARP_MASK, h, off);
    }
    if (lane == 0) *stop = (unsigned char)(same && h / n < confidence);
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
    if (lane == 0) {
      if (mine == 0x7fffffff) { __trap(); return; }
      out[row] = mine;
    }
  }
}

// Row softmax in FP32 with a parallel BF16 copy: probs = exp(x-max)/sum.
extern "C" __global__ void softmax_rows_dual(const float* __restrict__ x,
    float* __restrict__ probs, __nv_bfloat16* __restrict__ probs_bf16, int rows,
    long long vocab) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const long long offset = (long long)r * vocab;
    softmax_row_f32<false, true>(x + offset, probs + offset, probs_bf16 + offset, vocab);
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

// Canvas iterate update: accepted rows take the sampled pick, the rest take a
// fresh uniform renoise draw.
extern "C" __global__ void canvas_accept_scatter(const unsigned char* __restrict__ accept,
    const int* __restrict__ picks, const int* __restrict__ renoise,
    unsigned int* __restrict__ canvas_ids, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    canvas_ids[i] =
        (unsigned int)(accept[i] ? picks[i] : renoise[i]);
}
