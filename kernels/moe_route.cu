// MoE router: softmax or sigmoid scoring over BF16 expert logits, deterministic
// top-k (descending, exact ties by lower expert id), FP32 renormalised weights
// stored as BF16.
// moe_route_decode_bf16 publishes BF16 logits from disjoint expert-owner CTAs;
// the last completing CTA selects top-k and resets the row completion counter.
//   moe_route_logits_bf16(x, router, logits, rows, experts, hidden):
//     x [rows, hidden] bf16 row-major, router W [experts, hidden] bf16
//     row-major, logits [rows, experts] bf16 row-major; each expert logit is
//     one thread's FP32 fmaf chain over BF16 products in ascending element
//     order, rounded to BF16 (F.linear boundary). Grid (ceil(experts/32),
//     rows), block 128; the block stages 32 router rows x 512 elements and the
//     matching x elements in static shared memory (34,304 B) per step.
//   moe_route_topk(logits, ids, weights, rows, experts, top_k):
//     logits [rows, experts] bf16 row-major from moe_route_logits_bf16 or a
//     BF16 GEMM with FP32 accumulation; no rounding is applied here.
//   ids [rows, top_k] i32, weights [rows, top_k] bf16.
// moe_route_topk: one block per row; thread e < experts owns expert e. Softmax
// or sigmoid runs in FP32 over the BF16 logits. top_k rounds each take the block
// maximum under the comparator (higher probability, then lower expert id); a
// selected thread retires with probability -1.0f, below every softmax output.
// The top-k probabilities are renormalised in FP32 accumulated in selection
// order and stored as BF16.
// __trap() guards: grid beyond rows, experts outside [1,1024]. moe_route_topk:
// blockDim.x not a warp multiple or below experts, top_k outside [1,32] or
// >= experts, logits base pointer not 16-byte aligned. moe_route_logits_bf16:
// hidden < 8 or % 8 != 0, blockDim.x != 128, gridDim.x != ceil(experts/32),
// x/router base pointers not 16-byte aligned.
// moe_route_topk static shared memory: two 32-word reduction stages plus 32
// selected (id, probability) slots; launch grid = rows, block = experts
// rounded up to a warp multiple (<= 1024).
#include "gpu_compat.cuh"
#include <math.h>
#include <stdint.h>

using bf16 = __nv_bfloat16;

template<bool MASS>
__device__ void moe_route_block_select(
    const bf16* logits, const int* valid, const unsigned* allowed,
    int* ids, float* output, int rows, int experts, int top_k) {
  const int e = threadIdx.x, row = blockIdx.x;
  if (row >= rows || experts > 1024 || experts != blockDim.x ||
      top_k < 1 || top_k > 32 || top_k >= experts) __trap();
  __shared__ float values[1024], chosen[32], sum;
  __shared__ int indices[1024];
  float logit = __bfloat162float(logits[(size_t)row * experts + e]);
  if (!isfinite(logit)) __trap();
  if constexpr (!MASS) {
    if (allowed && valid[row] && !allowed[e]) logit = -INFINITY;
  }
  values[e] = logit;
  __syncthreads();
  for (int stride = experts / 2; stride; stride >>= 1) {
    if (e < stride) values[e] = fmaxf(values[e], values[e + stride]);
    __syncthreads();
  }
  const float maximum = values[0];
  __syncthreads();
  const float exponential = expf(logit - maximum);
  values[e] = exponential;
  __syncthreads();
  for (int stride = experts / 2; stride; stride >>= 1) {
    if (e < stride) values[e] += values[e + stride];
    __syncthreads();
  }
  const float probability = exponential / values[0];
  if constexpr (MASS) output[(size_t)row * experts + e] = 0.0f;
  bool taken = false;
  for (int rank = 0; rank < top_k; ++rank) {
    __syncthreads();
    values[e] = logit;
    indices[e] = taken ? 0x7fffffff : e;
    if (taken) values[e] = -INFINITY;
    __syncthreads();
    for (int stride = experts / 2; stride; stride >>= 1) {
      if (e < stride) {
        const float other = values[e + stride];
        const int id = indices[e + stride];
        if (other > values[e] || (other == values[e] && id < indices[e])) {
          values[e] = other;
          indices[e] = id;
        }
      }
      __syncthreads();
    }
    if (e == indices[0]) {
      taken = true;
      if constexpr (MASS) output[(size_t)row * experts + e] = valid[row] ? probability : 0.0f;
      else {
        ids[(size_t)row * top_k + rank] = e;
        chosen[rank] = probability;
      }
    }
  }
  if constexpr (!MASS) {
    __syncthreads();
    if (e == 0) {
      float total = 0.0f;
      for (int rank = 0; rank < top_k; ++rank) total += chosen[rank];
      sum = fmaxf(total, 1e-12f);
    }
    __syncthreads();
    if (e < top_k) output[(size_t)row * top_k + e] = chosen[e] / sum;
  }
}

extern "C" __global__ void moe_route_topk_mass(
    const bf16* logits, const int* valid, float* mass, int rows, int experts, int top_k) {
  moe_route_block_select<true>(logits, valid, nullptr, nullptr, mass, rows, experts, top_k);
}

extern "C" __global__ void moe_route_topk_masked_f32(
    const bf16* logits, const int* valid, unsigned* allowed, int* ids,
    float* weights, int rows, int experts, int top_k, int masked) {
  if (!masked && blockIdx.x == 0) allowed[threadIdx.x] = 1;
  moe_route_block_select<false>(logits, valid, masked ? allowed : nullptr, ids, weights, rows, experts, top_k);
}

extern "C" __global__ void moe_route_mass_coreset(
    const float* mass, unsigned* allowed, int rows, int experts, float top_p) {
  const int e = threadIdx.x;
  if (blockIdx.x || blockDim.x != experts || experts > 1024 ||
      experts < 32 || (experts & (experts - 1)) || !(top_p > 0 && top_p <= 1)) __trap();
  __shared__ float probability[1024], reduction[1024];
  __shared__ int order[1024];
  float total = 0.0f;
  for (int row = 0; row < rows; ++row) total += mass[(size_t)row * experts + e];
  reduction[e] = total;
  __syncthreads();
  for (int stride = experts / 2; stride; stride >>= 1) {
    if (e < stride) reduction[e] += reduction[e + stride];
    __syncthreads();
  }
  probability[e] = total / fmaxf(reduction[0], 1e-9f);
  order[e] = e;
  __syncthreads();
  for (int width = 2; width <= experts; width <<= 1) {
    for (int distance = width >> 1; distance; distance >>= 1) {
      const int other = e ^ distance;
      const float a = probability[e], b = probability[other];
      const int ai = order[e], bi = order[other];
      const bool first = a > b || (a == b && ai < bi);
      const bool want_first = ((e & width) == 0) == ((e & distance) == 0);
      __syncthreads();
      if (first != want_first) { probability[e] = b; order[e] = bi; }
      __syncthreads();
    }
  }
  float cumulative = 0.0f;
  for (int i = 0; i <= e; ++i) cumulative += probability[i];
  allowed[order[e]] = e == 0 || __fsub_rn(cumulative, probability[e]) < top_p;
}

template<bool reverse_ties = false>
__device__ void moe_route_select(
    float logit, int* __restrict__ ids, bf16* __restrict__ weights,
    int experts, int top_k, int row) {
  const int tid = threadIdx.x;
  const int nthr = (int)blockDim.x;
  const int nw = nthr >> 5;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int e = tid;
  const bool live = e < experts;

  __shared__ float s_red[32];
  __shared__ int s_rid[32];
  __shared__ float s_sel_p[32];
  __shared__ int s_sel_id[32];
  __shared__ float s_tsum;

  float v = logit;
  for (int off = 16; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_down_sync(FF_WARP_MASK, v, off));
  if (lane == 0) s_red[warp] = v;
  __syncthreads();
  v = warp == 0 && lane < nw ? s_red[lane] : -INFINITY;
  if (warp == 0) {
    for (int off = 16; off > 0; off >>= 1)
      v = fmaxf(v, __shfl_down_sync(FF_WARP_MASK, v, off));
    if (lane == 0) s_red[0] = v;
  }
  __syncthreads();
  const float max_logit = s_red[0];
  // max_logit reads s_red[0]; the sum stage scatters into s_red[warp].
  __syncthreads();

  const float ev = live ? expf(logit - max_logit) : 0.0f;
  v = ev;
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_down_sync(FF_WARP_MASK, v, off);
  if (lane == 0) s_red[warp] = v;
  __syncthreads();
  v = warp == 0 && lane < nw ? s_red[lane] : 0.0f;
  if (warp == 0) {
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(FF_WARP_MASK, v, off);
    if (lane == 0) s_red[0] = v;
  }
  __syncthreads();
  const float prob = ev / s_red[0];

  if (tid == 0) s_tsum = 0.0f;
  __syncthreads();
  bool taken = false;
  for (int r = 0; r < top_k; ++r) {
    // comparator: higher probability wins, exact ties go to the lower id
    v = !taken && live ? prob : -1.0f;
    int id = live ? e : 0x7fffffff;
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_down_sync(FF_WARP_MASK, v, off);
      const int oi = __shfl_down_sync(FF_WARP_MASK, id, off);
      if (ov > v || (ov == v && (reverse_ties ? oi > id : oi < id))) { v = ov; id = oi; }
    }
    if (lane == 0) { s_red[warp] = v; s_rid[warp] = id; }
    __syncthreads();
    if (warp == 0) {
      v = lane < nw ? s_red[lane] : -INFINITY;
      id = lane < nw ? s_rid[lane] : 0x7fffffff;
      for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(FF_WARP_MASK, v, off);
        const int oi = __shfl_down_sync(FF_WARP_MASK, id, off);
        if (ov > v || (ov == v && (reverse_ties ? oi > id : oi < id))) { v = ov; id = oi; }
      }
      if (lane == 0) { s_red[0] = v; s_rid[0] = id; }
    }
    __syncthreads();
    if (live && !taken && prob == s_red[0] && e == s_rid[0]) {
      s_sel_id[r] = e;
      s_sel_p[r] = prob;
      s_tsum += prob;
      taken = true;
    }
    __syncthreads();
  }

  if (tid < top_k) {
    ids[(size_t)row * top_k + tid] = s_sel_id[tid];
    weights[(size_t)row * top_k + tid] =
        __float2bfloat16(s_sel_p[tid] / s_tsum);
  }
}

template<bool reverse_ties = false>
__device__ void moe_route_select_sigmoid(
    float logit, int* __restrict__ ids, bf16* __restrict__ weights,
    int experts, int top_k, int row) {
  const int tid = threadIdx.x;
  const int nthr = (int)blockDim.x;
  const int nw = nthr >> 5;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int e = tid;
  const bool live = e < experts;
  const float score = live ? 1.0f / (1.0f + expf(-logit)) : -1.0f;

  __shared__ float s_sel_p[32];
  __shared__ int s_sel_id[32];
  __shared__ float s_red[32];
  __shared__ int s_rid[32];
  __shared__ float s_tsum;
  if (tid == 0) s_tsum = 0.0f;
  __syncthreads();

  bool taken = false;
  for (int r = 0; r < top_k; ++r) {
    float v = !taken && live ? score : -1.0f;
    int id = live ? e : 0x7fffffff;
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_down_sync(FF_WARP_MASK, v, off);
      const int oi = __shfl_down_sync(FF_WARP_MASK, id, off);
      if (ov > v || (ov == v && (reverse_ties ? oi > id : oi < id))) { v = ov; id = oi; }
    }
    if (lane == 0) { s_red[warp] = v; s_rid[warp] = id; }
    __syncthreads();
    if (warp == 0) {
      v = lane < nw ? s_red[lane] : -1.0f;
      id = lane < nw ? s_rid[lane] : 0x7fffffff;
      for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(FF_WARP_MASK, v, off);
        const int oi = __shfl_down_sync(FF_WARP_MASK, id, off);
        if (ov > v || (ov == v && (reverse_ties ? oi > id : oi < id))) { v = ov; id = oi; }
      }
      if (lane == 0) { s_red[0] = v; s_rid[0] = id; }
    }
    __syncthreads();
    if (live && !taken && score == s_red[0] && e == s_rid[0]) {
      s_sel_id[r] = e;
      s_sel_p[r] = score;
      s_tsum += score;
      taken = true;
    }
    __syncthreads();
  }

  if (tid < top_k) {
    ids[(size_t)row * top_k + tid] = s_sel_id[tid];
    weights[(size_t)row * top_k + tid] =
        __float2bfloat16(s_sel_p[tid] / (s_tsum + 1.0e-20f));
  }
}

// The routed weight's storage type: FP32 where the definition multiplies by an
// FP32 weight (model.py Expert.forward), BF16 for the paths that round it.
template <class Weight>
__device__ __forceinline__ void moe_route_weight_store(
    Weight* __restrict__ weights, size_t index, float value) {
  if constexpr (sizeof(Weight) == sizeof(float)) {
    weights[index] = value;
  } else {
    weights[index] = __float2bfloat16(value);
  }
}

template <class Weight>
__device__ __forceinline__ void moe_route_select_noaux(
    float logit, float bias, int* __restrict__ ids, Weight* __restrict__ weights,
    int experts, int top_k, int score_mode, float routed_scaling) {
  const int tid = threadIdx.x;
  const int nthr = (int)blockDim.x;
  const int nw = nthr >> 5;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int row = blockIdx.x;
  const int e = tid;
  const bool live = e < experts;
  float score = -1.0f;
  if (live) {
    if (score_mode == 0) {
      score = 1.0f / (1.0f + expf(-logit));
    } else {
      const float softplus = fmaxf(logit, 0.0f) + log1pf(expf(-fabsf(logit)));
      score = sqrtf(softplus);
    }
  }
  const float select_score = score + bias;

  __shared__ float s_sel_p[32];
  __shared__ int s_sel_id[32];
  __shared__ float s_red[32];
  __shared__ int s_rid[32];
  __shared__ float s_tsum;
  if (tid == 0) s_tsum = 0.0f;
  __syncthreads();

  bool taken = false;
  for (int r = 0; r < top_k; ++r) {
    float v = !taken && live ? select_score : -INFINITY;
    int id = live ? e : 0x7fffffff;
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_down_sync(FF_WARP_MASK, v, off);
      const int oi = __shfl_down_sync(FF_WARP_MASK, id, off);
      if (ov > v || (ov == v && oi < id)) { v = ov; id = oi; }
    }
    if (lane == 0) { s_red[warp] = v; s_rid[warp] = id; }
    __syncthreads();
    if (warp == 0) {
      v = lane < nw ? s_red[lane] : -INFINITY;
      id = lane < nw ? s_rid[lane] : 0x7fffffff;
      for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(FF_WARP_MASK, v, off);
        const int oi = __shfl_down_sync(FF_WARP_MASK, id, off);
        if (ov > v || (ov == v && oi < id)) { v = ov; id = oi; }
      }
      if (lane == 0) { s_red[0] = v; s_rid[0] = id; }
    }
    __syncthreads();
    if (live && !taken && select_score == s_red[0] && e == s_rid[0]) {
      s_sel_id[r] = e;
      s_sel_p[r] = score;
      s_tsum += score;
      taken = true;
    }
    __syncthreads();
  }

  if (tid < top_k) {
    ids[(size_t)row * top_k + tid] = s_sel_id[tid];
    moe_route_weight_store(
        weights, (size_t)row * top_k + tid,
        (s_sel_p[tid] / (s_tsum + 1.0e-20f)) * routed_scaling);
  }
}

__device__ __forceinline__ float moe_route_dot8(uint4 a, uint4 b, float acc) {
    const __nv_bfloat162* ha = reinterpret_cast<const __nv_bfloat162*>(&a);
    const __nv_bfloat162* hb = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 fa = __bfloat1622float2(ha[j]);
      const float2 fb = __bfloat1622float2(hb[j]);
      acc = fmaf(fa.x, fb.x, acc);
      acc = fmaf(fa.y, fb.y, acc);
    }
    return acc;
}

#define MOE_ROUTE_EPB 32
#define MOE_ROUTE_WORDS 64
#define MOE_ROUTE_THREADS 128
extern "C" __global__ __launch_bounds__(MOE_ROUTE_THREADS) void moe_route_logits_bf16(
    const bf16* __restrict__ x, const bf16* __restrict__ router,
    bf16* __restrict__ logits, int rows, int experts, int hidden) {
  if ((int)blockIdx.y >= rows || experts < 1 || experts > 1024 || hidden < 8 ||
      (hidden & 7) != 0 || blockDim.x != MOE_ROUTE_THREADS ||
      (int)gridDim.x != (experts + MOE_ROUTE_EPB - 1) / MOE_ROUTE_EPB ||
      (reinterpret_cast<uintptr_t>(x) & 15) != 0 ||
      (reinterpret_cast<uintptr_t>(router) & 15) != 0)
    __trap();

  __shared__ uint4 s_w[MOE_ROUTE_EPB][MOE_ROUTE_WORDS + 1];
  __shared__ uint4 s_x[MOE_ROUTE_WORDS];
  const int tid = threadIdx.x;
  const int row = blockIdx.y;
  const int e0 = blockIdx.x * MOE_ROUTE_EPB;
  const int words = hidden >> 3;
  const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)row * hidden);
  const uint4* wr = reinterpret_cast<const uint4*>(router);
  const bool owner = tid < MOE_ROUTE_EPB && e0 + tid < experts;
  float acc = 0.0f;
  for (int c = 0; c < words; c += MOE_ROUTE_WORDS) {
    const int n = min(MOE_ROUTE_WORDS, words - c);
    if (n == MOE_ROUTE_WORDS && e0 + MOE_ROUTE_EPB <= experts) {
      constexpr int per = MOE_ROUTE_EPB * MOE_ROUTE_WORDS / MOE_ROUTE_THREADS;
      uint4 v[per];
#pragma unroll
      for (int p = 0; p < per; ++p) {
        const int i = tid + p * MOE_ROUTE_THREADS;
        v[p] = wr[(size_t)(e0 + i / MOE_ROUTE_WORDS) * words + c + i % MOE_ROUTE_WORDS];
      }
#pragma unroll
      for (int p = 0; p < per; ++p) {
        const int i = tid + p * MOE_ROUTE_THREADS;
        s_w[i / MOE_ROUTE_WORDS][i % MOE_ROUTE_WORDS] = v[p];
      }
    } else {
      for (int i = tid; i < MOE_ROUTE_EPB * MOE_ROUTE_WORDS; i += MOE_ROUTE_THREADS) {
        const int r = i / MOE_ROUTE_WORDS, w = i % MOE_ROUTE_WORDS;
        if (w < n && e0 + r < experts)
          s_w[r][w] = wr[(size_t)(e0 + r) * words + c + w];
      }
    }
    if (tid < n) s_x[tid] = xr[c + tid];
    __syncthreads();
    if (owner)
      for (int w = 0; w < n; ++w) acc = moe_route_dot8(s_x[w], s_w[tid][w], acc);
    __syncthreads();
  }
  if (owner) logits[(size_t)row * experts + e0 + tid] = __float2bfloat16(acc);
}

extern "C" __global__ void moe_route_topk(
    const bf16* __restrict__ logits, int* __restrict__ ids,
    bf16* __restrict__ weights, int rows, int experts, int top_k) {
  if ((int)blockIdx.x >= rows || experts < 1 || experts > 1024 || top_k < 1 ||
      top_k > 32 || top_k >= experts || (blockDim.x & 31) != 0 ||
      (int)blockDim.x < experts ||
      (reinterpret_cast<uintptr_t>(logits) & 15) != 0)
    __trap();

  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  float logit = -INFINITY;
  if (tid < experts)
    logit = __bfloat162float(logits[(size_t)row * experts + tid]);
  moe_route_select(logit, ids, weights, experts, top_k, row);
}

extern "C" __global__ void moe_route_topk_sigmoid(
    const bf16* __restrict__ logits, int* __restrict__ ids,
    bf16* __restrict__ weights, int rows, int experts, int top_k) {
  if ((int)blockIdx.x >= rows || experts < 1 || experts > 1024 || top_k < 1 ||
      top_k > 32 || top_k >= experts || (blockDim.x & 31) != 0 ||
      (int)blockDim.x < experts ||
      (reinterpret_cast<uintptr_t>(logits) & 15) != 0)
    __trap();

  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  float logit = -INFINITY;
  if (tid < experts)
    logit = __bfloat162float(logits[(size_t)row * experts + tid]);
  moe_route_select_sigmoid(logit, ids, weights, experts, top_k, row);
}

// Each expert has one ascending FP32 FMA chain and one BF16 logit store.
// The last producer CTA for a row performs selection and resets its counter.
template<bool reverse_ties>
__device__ void moe_route_decode(
    const bf16* __restrict__ x, const bf16* __restrict__ router,
    bf16* __restrict__ logits, int* __restrict__ ids,
    bf16* __restrict__ weights, unsigned int* __restrict__ completed,
    int rows, int experts, int hidden, int top_k, int owners, int score_mode) {
  if ((int)blockIdx.y >= rows || experts < 1 || experts > 1024 ||
      hidden < 8 || (hidden & 7) != 0 || top_k < 1 || top_k > 32 ||
      top_k >= experts || owners < 1 || owners > 32 ||
      (owners & (owners - 1)) != 0 ||
      (int)blockDim.x != ((experts + 31) & ~31) ||
      (int)gridDim.x != (experts + owners - 1) / owners ||
      score_mode < 0 || score_mode > 1 ||
      (reinterpret_cast<uintptr_t>(x) & 15) != 0 ||
      (reinterpret_cast<uintptr_t>(router) & 15) != 0 ||
      (reinterpret_cast<uintptr_t>(logits) & 15) != 0)
    __trap();

  extern __shared__ uint4 tile[];
  uint4* sw = tile;
  uint4* sx = tile + owners * (MOE_ROUTE_WORDS + 1);
  __shared__ unsigned int last;
  const int tid = threadIdx.x;
  const int row = blockIdx.y;
  const int e0 = blockIdx.x * owners;
  const int words = hidden >> 3;
  const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)row * hidden);
  const uint4* wr = reinterpret_cast<const uint4*>(router);
  const bool owner = tid < owners && e0 + tid < experts;
  float acc = 0.0f;
  for (int c = 0; c < words; c += MOE_ROUTE_WORDS) {
    const int n = min(MOE_ROUTE_WORDS, words - c);
    for (int i = tid; i < owners * MOE_ROUTE_WORDS; i += blockDim.x) {
      const int r = i / MOE_ROUTE_WORDS;
      const int w = i % MOE_ROUTE_WORDS;
      if (w < n && e0 + r < experts)
        sw[r * (MOE_ROUTE_WORDS + 1) + w] =
            wr[(size_t)(e0 + r) * words + c + w];
    }
    for (int w = tid; w < n; w += blockDim.x) sx[w] = xr[c + w];
    __syncthreads();
    if (owner)
      for (int w = 0; w < n; ++w)
        acc = moe_route_dot8(sx[w], sw[tid * (MOE_ROUTE_WORDS + 1) + w], acc);
    __syncthreads();
  }
  volatile unsigned short* published = reinterpret_cast<volatile unsigned short*>(logits);
  if (owner) {
    published[(size_t)row * experts + e0 + tid] =
        __bfloat16_as_ushort(__float2bfloat16(acc));
    __threadfence();
  }
  __syncthreads();
  if (tid == 0) last = atomicAdd(completed + row, 1u) == gridDim.x - 1;
  __syncthreads();
  if (!last) return;
  float logit = -INFINITY;
  if (tid < experts)
    logit = __bfloat162float(__ushort_as_bfloat16(published[(size_t)row * experts + tid]));
  if (score_mode == 0)
    moe_route_select<reverse_ties>(logit, ids, weights, experts, top_k, row);
  else
    moe_route_select_sigmoid<reverse_ties>(logit, ids, weights, experts, top_k, row);
  __syncthreads();
  if (tid == 0) atomicExch(completed + row, 0u);
}

extern "C" __global__ void moe_route_decode_bf16(
    const bf16* x, const bf16* router, bf16* logits, int* ids, bf16* weights,
    unsigned int* completed, int rows, int experts, int hidden, int top_k,
    int owners, int score_mode) {
  moe_route_decode<false>(x, router, logits, ids, weights, completed, rows,
                         experts, hidden, top_k, owners, score_mode);
}

// Gate control: exact score ties select the higher global expert id.
extern "C" __global__ void moe_route_decode_reverse_ties(
    const bf16* x, const bf16* router, bf16* logits, int* ids, bf16* weights,
    unsigned int* completed, int rows, int experts, int hidden, int top_k,
    int owners, int score_mode) {
  moe_route_decode<true>(x, router, logits, ids, weights, completed, rows,
                        experts, hidden, top_k, owners, score_mode);
}

extern "C" __global__ void moe_route_topk_noaux(
    const float* __restrict__ logits, const float* __restrict__ bias,
    int* __restrict__ ids, bf16* __restrict__ weights, int rows, int experts,
    int top_k, int score_mode, float routed_scaling) {
  if ((int)blockIdx.x >= rows || experts < 1 || experts > 1024 || top_k < 1 ||
      top_k > 32 || top_k > experts || (blockDim.x & 31) != 0 ||
      (int)blockDim.x < experts || score_mode < 0 || score_mode > 1 ||
      !isfinite(routed_scaling) || routed_scaling <= 0.0f ||
      (reinterpret_cast<uintptr_t>(logits) & 3) != 0 ||
      (reinterpret_cast<uintptr_t>(bias) & 3) != 0)
    __trap();

  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  float logit = -INFINITY;
  float correction = 0.0f;
  if (tid < experts) {
    logit = logits[(size_t)row * experts + tid];
    correction = bias[(size_t)row * experts + tid];
  }
  moe_route_select_noaux(logit, correction, ids, weights, experts, top_k,
                         score_mode, routed_scaling);
}

// The same selection with FP32 routed weights (model.py Gate.forward returns
// FP32 scores; Expert.forward multiplies by them in FP32).
extern "C" __global__ void moe_route_topk_noaux_f32(
    const float* __restrict__ logits, const float* __restrict__ bias,
    int* __restrict__ ids, float* __restrict__ weights, int rows, int experts,
    int top_k, int score_mode, float routed_scaling) {
  if ((int)blockIdx.x >= rows || experts < 1 || experts > 1024 || top_k < 1 ||
      top_k > 32 || top_k > experts || (blockDim.x & 31) != 0 ||
      (int)blockDim.x < experts || score_mode < 0 || score_mode > 1 ||
      !isfinite(routed_scaling) || routed_scaling <= 0.0f ||
      (reinterpret_cast<uintptr_t>(logits) & 3) != 0 ||
      (reinterpret_cast<uintptr_t>(bias) & 3) != 0 ||
      (reinterpret_cast<uintptr_t>(weights) & 3) != 0)
    __trap();

  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  float logit = -INFINITY;
  float correction = 0.0f;
  if (tid < experts) {
    logit = logits[(size_t)row * experts + tid];
    correction = bias[(size_t)row * experts + tid];
  }
  moe_route_select_noaux(logit, correction, ids, weights, experts, top_k,
                         score_mode, routed_scaling);
}

// Stream-ordered publication of one native expert slot and its generation.
// victim is -1 or differs from incoming; a valid incoming epoch strictly increases.
extern "C" __global__ void moe_slot_publish(
    uint64_t* gate_up, uint64_t* down, uint32_t* valid, uint64_t* generation,
    int experts, int incoming, int victim, uint64_t gu, uint64_t dn,
    uint64_t epoch) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (experts <= 0 || incoming < 0 || incoming >= experts || victim < -1 ||
      victim >= experts || victim == incoming || gu == 0 || dn == 0 ||
      (gu & 15) != 0 || (dn & 15) != 0 || epoch == 0) __trap();
  if (valid[incoming] == 1 && generation[incoming] >= epoch) __trap();
  if (victim >= 0) {
    valid[victim] = 0;
    generation[victim] = 0;
    gate_up[victim] = 0;
    down[victim] = 0;
  }
  gate_up[incoming] = gu;
  down[incoming] = dn;
  generation[incoming] = epoch;
  valid[incoming] = 1;
}

extern "C" __global__ void moe_route_mask(
    const int* ids, const uint32_t* valid, const uint64_t* generation,
    uint32_t* active, uint64_t* assignment_gen, int assignments, int experts) {
  const int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= assignments) return;
  const int expert = ids[a];
  if (expert < 0 || expert >= experts || valid[expert] > 1 ||
      (valid[expert] == 1 && generation[expert] == 0)) __trap();
  active[a] = valid[expert];
  assignment_gen[a] = valid[expert] ? generation[expert] : 0;
}

