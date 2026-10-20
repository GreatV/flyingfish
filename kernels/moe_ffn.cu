// BF16 routed and shared contributions: legacy rows1..8, invariant rows1..16; caller-owned done bits.
// gate_up: grid(ceil(I/32),rows,K+1), block256, smem=H*2; down: grid(ceil(H/16),rows,K+1), block128.
// Active assignments require valid slots, matching generations and distinct expert ids.
// Expert row pointers and shared weights are 16-byte aligned.
#include <cuda_runtime.h>
#include "q38_contract.cuh"

using bf16 = __nv_bfloat16;

constexpr int LANES = 8;    // lanes cooperating on one dot product
constexpr int GU_ROWS = 32;  // gate/up output rows per block
constexpr int GU_NT = LANES * GU_ROWS;
constexpr int DN_ROWS = 16;  // down output rows per block
constexpr int DN_NT = LANES * DN_ROWS;

#define FD_GUARD_MOE_GATE_UP do {                                                  \
  if (blockDim.x != GU_NT || gridDim.y != rows || gridDim.z != top_k + 1 ||        \
      rows < 1 || rows > 8 || top_k < 1 || (hidden & 7) != 0 || inter < 1)         \
    __trap();                                                                     \
} while (0)

#define FD_GUARD_MOE_DOWN do {                                                     \
  if (blockDim.x != DN_NT || gridDim.y != rows || gridDim.z != top_k + 1 ||        \
      rows < 1 || rows > 8 || top_k < 1 || (inter & 7) != 0)                       \
    __trap();                                                                     \
} while (0)

__device__ __forceinline__ float moe_dot8(uint4 w, uint4 a) {
  const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&w);
  const __nv_bfloat162* ap = reinterpret_cast<const __nv_bfloat162*>(&a);
  float s = 0.0f;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const float2 wf = __bfloat1622float2(wp[q]);
    const float2 af = __bfloat1622float2(ap[q]);
    s = fmaf(wf.x, af.x, s);
    s = fmaf(wf.y, af.y, s);
  }
  return s;
}

// Sum one 8-lane group; the reduced value lands in lane 0 of the group.
__device__ __forceinline__ float moe_group_sum(float v) {
#pragma unroll
  for (int off = LANES / 2; off > 0; off >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, off, LANES);
  return v;
}

__device__ __forceinline__ float moe_silu(float v) { return v / (1.0f + expf(-v)); }

// Native FP8 slots contain gate/up/down E4M3 codes followed by BF16 channel scales.
__device__ __forceinline__ uint4 moe_dequant8(uint2 codes, float scale) {
  uint4 w;
  const unsigned char* in = reinterpret_cast<const unsigned char*>(&codes);
  unsigned short* out = reinterpret_cast<unsigned short*>(&w);
#pragma unroll
  for (int b = 0; b < 8; ++b)
    out[b] = __bfloat16_as_ushort(
        __float2bfloat16(e4m3_decode(in[b]) * scale));
  return w;
}

template <bool FP8>
__device__ __forceinline__ uint4 moe_load8(const bf16* base, const unsigned char* codes,
                                         float scale, int v) {
  if constexpr (FP8) {
    if (codes) return moe_dequant8(reinterpret_cast<const uint2*>(codes)[v], scale);
  }
  return reinterpret_cast<const uint4*>(base)[v];
}

// Sixteen scalar FMA lanes, paired in ascending 1/2/4/8 order.
__device__ __forceinline__ float moe_pairwise16(float sum, int tid) {
  const unsigned mask = 0xffffu << (tid & 16);
  for (int step = 1; step < 16; step *= 2)
    sum = __fadd_rn(sum, __shfl_down_sync(mask, sum, step, 16));
  return sum;
}

template<bool FP8>
__device__ __forceinline__ float moe_load1(const bf16* weight,
    const unsigned char* codes, float scale, int col) {
  if constexpr (FP8) {
    if (codes) return __bfloat162float(__float2bfloat16_rn(e4m3_decode(codes[col]) * scale));
  }
  return __bfloat162float(weight[col]);
}

template <bool FP8, bool INVARIANT = false, class Word>
__device__ __forceinline__ void moe_gate_up(
    const bf16* __restrict__ x, const int* __restrict__ ids,
    const unsigned* __restrict__ active,
    const Word* const* __restrict__ slots_gate_up,
    const unsigned* __restrict__ slot_valid,
    const unsigned long long* __restrict__ slot_generation,
    const unsigned long long* __restrict__ assignment_gen,
    const bf16* __restrict__ shared_gate, const bf16* __restrict__ shared_up,
    bf16* __restrict__ h, bf16* __restrict__ h_shared,
    int rows, int experts, int top_k, int hidden, int inter) {
  constexpr int lanes = INVARIANT ? 16 : LANES;
  if constexpr (INVARIANT) {
    if (blockDim.x != GU_NT || gridDim.y != rows || gridDim.z != top_k ||
        rows < 1 || rows > 16 || top_k < 1 || (hidden & 7) != 0 || inter < 1)
      __trap();
  } else {
    FD_GUARD_MOE_GATE_UP;
  }
  const int r = blockIdx.y;
  const int seg = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & (lanes - 1);
  const int j = blockIdx.x * (GU_NT / lanes) + (tid / lanes);
  const int nvec = hidden >> 3;
  const bool live_seg = seg >= top_k || active[(size_t)r * top_k + seg] != 0u;

  extern __shared__ __align__(16) bf16 xs[];
  {
    const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)r * hidden);
    uint4* dst = reinterpret_cast<uint4*>(xs);
    for (int v = tid; v < nvec; v += GU_NT) dst[v] = src[v];
  }
  __syncthreads();
  const uint4* xq = reinterpret_cast<const uint4*>(xs);

  const bf16* gate_base = shared_gate;
  const bf16* up_base = shared_up;
  const unsigned char* codes = nullptr;
  // Active router choices have distinct expert ids.
  if (seg == 0 && blockIdx.x == 0 && tid == 0) {
    for (int c0 = 0; c0 < top_k; ++c0) {
      if (active[(size_t)r * top_k + c0] == 0u) continue;
      for (int c1 = c0 + 1; c1 < top_k; ++c1) {
        if (active[(size_t)r * top_k + c1] == 0u) continue;
        if (ids[(size_t)r * top_k + c0] == ids[(size_t)r * top_k + c1]) __trap();
      }
    }
  }
  if (live_seg && seg < top_k) {
    const int e = ids[(size_t)r * top_k + seg];
    if ((unsigned)e >= (unsigned)experts || slot_valid[e] == 0u ||
        slot_generation[e] != assignment_gen[(size_t)r * top_k + seg])
      __trap();
    const Word* gu = slots_gate_up[e];
    if ((reinterpret_cast<unsigned long long>(gu) & 15) != 0) __trap();
    if constexpr (FP8) {
      codes = gu;
    } else {
      gate_base = gu;
      up_base = gu + (size_t)inter * hidden;
    }
  }

  float accg = 0.0f;
  float accu = 0.0f;
  if (j < inter && live_seg) {
    float gs = 0.0f, us = 0.0f;
    const unsigned char* gc = nullptr;
    const unsigned char* uc = nullptr;
    if constexpr (FP8) {
      if (codes) {
        const bf16* scales = reinterpret_cast<const bf16*>(codes + (size_t)3 * inter * hidden);
        gs = __bfloat162float(scales[j]);
        us = __bfloat162float(scales[inter + j]);
        gc = codes + (size_t)j * hidden;
        uc = codes + (size_t)(inter + j) * hidden;
      }
    }
    const bf16* gv = gate_base + (size_t)j * hidden;
    const bf16* uv = up_base + (size_t)j * hidden;
    if constexpr (INVARIANT) {
      for (int col = lane; col < hidden; col += 16) {
        const float a = __bfloat162float(xs[col]);
        accg = __fmaf_rn(a, moe_load1<FP8>(gv, gc, gs, col), accg);
        accu = __fmaf_rn(a, moe_load1<FP8>(uv, uc, us, col), accu);
      }
    } else {
#pragma unroll 4
      for (int v = lane; v < nvec; v += LANES) {
        accg += moe_dot8(moe_load8<FP8>(gv, gc, gs, v), xq[v]);
        accu += moe_dot8(moe_load8<FP8>(uv, uc, us, v), xq[v]);
      }
    }
  }
  accg = INVARIANT ? moe_pairwise16(accg, tid) : moe_group_sum(accg);
  accu = INVARIANT ? moe_pairwise16(accu, tid) : moe_group_sum(accu);
  if (lane == 0 && j < inter && live_seg) {
    const float g = __bfloat162float(__float2bfloat16(accg));
    const float u = __bfloat162float(__float2bfloat16(accu));
    const float activation = INVARIANT
        ? __fmul_rn(g, __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g))))
        : moe_silu(g);
    const float s = __bfloat162float(__float2bfloat16(activation));
    bf16* out = seg == top_k ? h_shared + (size_t)r * inter
                             : h + ((size_t)r * top_k + seg) * inter;
    out[j] = __float2bfloat16(s * u);
  }
}

template <bool FP8, bool INVARIANT = false, class Word>
__device__ __forceinline__ void moe_down(
    const bf16* __restrict__ h, const bf16* __restrict__ h_shared,
    const int* __restrict__ ids, const unsigned* __restrict__ active,
    const bf16* __restrict__ weights,
    const Word* const* __restrict__ slots_down,
    const unsigned* __restrict__ slot_valid,
    const unsigned long long* __restrict__ slot_generation,
    const unsigned long long* __restrict__ assignment_gen,
    const bf16* __restrict__ shared_down,
    bf16* __restrict__ contributions, bf16* __restrict__ shared_out,
    unsigned* __restrict__ done,
    int rows, int experts, int top_k, int hidden, int inter) {
  constexpr int lanes = INVARIANT ? 16 : LANES;
  if constexpr (INVARIANT) {
    if (blockDim.x != DN_NT || gridDim.y != rows || gridDim.z != top_k ||
        rows < 1 || rows > 16 || top_k < 1 || (inter & 7) != 0)
      __trap();
  } else {
    FD_GUARD_MOE_DOWN;
  }
  const int r = blockIdx.y;
  const int seg = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & (lanes - 1);
  const int d = blockIdx.x * (DN_NT / lanes) + (tid / lanes);
  const int ivec = inter >> 3;
  const bool routed = seg < top_k;
  const bool live = !routed || active[(size_t)r * top_k + seg] != 0u;
  const bf16* wrow = shared_down;
  const unsigned char* codes = nullptr;
  float wk = 0.0f;
  if (routed && live) {
    const int e = ids[(size_t)r * top_k + seg];
    if ((unsigned)e >= (unsigned)experts || slot_valid[e] == 0u ||
        slot_generation[e] != assignment_gen[(size_t)r * top_k + seg])
      __trap();
    const Word* slot = slots_down[e];
    if ((reinterpret_cast<unsigned long long>(slot) & 15) != 0) __trap();
    if constexpr (FP8) {
      codes = slot;
    } else {
      wrow = slot;
    }
    wk = __bfloat162float(weights[(size_t)r * top_k + seg]);
    if (blockIdx.x == 0 && tid == 0) {
      const size_t a = (size_t)r * top_k + seg;
      if (done[a] != 0u) __trap();
      done[a] = 1u;
    }
  }
  const bf16* hrow =
      routed ? h + ((size_t)r * top_k + seg) * inter : h_shared + (size_t)r * inter;
  float acc = 0.0f;
  if (d < hidden && live) {
    float scale = 0.0f;
    const unsigned char* wc = nullptr;
    if constexpr (FP8) {
      if (codes) {
        const bf16* scales = reinterpret_cast<const bf16*>(codes + (size_t)3 * inter * hidden);
        scale = __bfloat162float(scales[2 * inter + d]);
        wc = codes + (size_t)2 * inter * hidden + (size_t)d * inter;
      }
    }
    const bf16* wv = wrow + (size_t)d * inter;
    const uint4* hv = reinterpret_cast<const uint4*>(hrow);
    if constexpr (INVARIANT) {
      for (int col = lane; col < inter; col += 16)
        acc = __fmaf_rn(__bfloat162float(hrow[col]),
                       moe_load1<FP8>(wv, wc, scale, col), acc);
    } else {
#pragma unroll 4
      for (int v = lane; v < ivec; v += LANES)
        acc += moe_dot8(moe_load8<FP8>(wv, wc, scale, v), hv[v]);
    }
  }
  acc = INVARIANT ? moe_pairwise16(acc, tid) : moe_group_sum(acc);
  if (lane == 0 && d < hidden && live) {
    const float lin = __bfloat162float(__float2bfloat16(acc));
    bf16* out = routed ? contributions + ((size_t)r * top_k + seg) * (size_t)hidden
                       : shared_out + (size_t)r * (size_t)hidden;
    out[d] = routed ? __float2bfloat16(lin * wk) : __float2bfloat16(lin);
  }
}

extern "C" __global__ void __launch_bounds__(GU_NT) moe_ffn_gate_up(
    const bf16* __restrict__ x, const int* __restrict__ ids,
    const unsigned* __restrict__ active,
    const bf16* const* __restrict__ slots_gate_up,
    const unsigned* __restrict__ slot_valid,
    const unsigned long long* __restrict__ slot_generation,
    const unsigned long long* __restrict__ assignment_gen,
    const bf16* __restrict__ shared_gate, const bf16* __restrict__ shared_up,
    bf16* __restrict__ h, bf16* __restrict__ h_shared,
    int rows, int experts, int top_k, int hidden, int inter) {
  moe_gate_up<false>(x, ids, active, slots_gate_up, slot_valid, slot_generation, assignment_gen, shared_gate, shared_up, h, h_shared, rows, experts, top_k, hidden, inter);
}

extern "C" __global__ void __launch_bounds__(GU_NT) moe_ffn_gate_up_w8a16(
    const bf16* __restrict__ x, const int* __restrict__ ids,
    const unsigned* __restrict__ active,
    const unsigned char* const* __restrict__ slots_gate_up,
    const unsigned* __restrict__ slot_valid,
    const unsigned long long* __restrict__ slot_generation,
    const unsigned long long* __restrict__ assignment_gen,
    const bf16* __restrict__ shared_gate, const bf16* __restrict__ shared_up,
    bf16* __restrict__ h, bf16* __restrict__ h_shared,
    int rows, int experts, int top_k, int hidden, int inter) {
  moe_gate_up<true>(x, ids, active, slots_gate_up, slot_valid, slot_generation, assignment_gen, shared_gate, shared_up, h, h_shared, rows, experts, top_k, hidden, inter);
}

extern "C" __global__ void __launch_bounds__(DN_NT) moe_ffn_down(
    const bf16* __restrict__ h, const bf16* __restrict__ h_shared,
    const int* __restrict__ ids, const unsigned* __restrict__ active,
    const bf16* __restrict__ weights,
    const bf16* const* __restrict__ slots_down,
    const unsigned* __restrict__ slot_valid,
    const unsigned long long* __restrict__ slot_generation,
    const unsigned long long* __restrict__ assignment_gen,
    const bf16* __restrict__ shared_down,
    bf16* __restrict__ contributions, bf16* __restrict__ shared_out,
    unsigned* __restrict__ done,
    int rows, int experts, int top_k, int hidden, int inter) {
  moe_down<false>(h, h_shared, ids, active, weights, slots_down, slot_valid, slot_generation, assignment_gen, shared_down, contributions, shared_out, done, rows, experts, top_k, hidden, inter);
}

extern "C" __global__ void __launch_bounds__(DN_NT) moe_ffn_down_w8a16(
    const bf16* __restrict__ h, const bf16* __restrict__ h_shared,
    const int* __restrict__ ids, const unsigned* __restrict__ active,
    const bf16* __restrict__ weights,
    const unsigned char* const* __restrict__ slots_down,
    const unsigned* __restrict__ slot_valid,
    const unsigned long long* __restrict__ slot_generation,
    const unsigned long long* __restrict__ assignment_gen,
    const bf16* __restrict__ shared_down,
    bf16* __restrict__ contributions, bf16* __restrict__ shared_out,
    unsigned* __restrict__ done,
    int rows, int experts, int top_k, int hidden, int inter) {
  moe_down<true>(h, h_shared, ids, active, weights, slots_down, slot_valid, slot_generation, assignment_gen, shared_down, contributions, shared_out, done, rows, experts, top_k, hidden, inter);
}

#define MOE_INVARIANT_GU(NAME, WORD, FP8) \
extern "C" __global__ void __launch_bounds__(GU_NT) NAME( \
    const bf16* x, const int* ids, const unsigned* active, \
    const WORD* const* slots, const unsigned* valid, \
    const unsigned long long* generation, const unsigned long long* assigned, \
    const bf16* shared_gate, const bf16* shared_up, bf16* h, bf16* hs, \
    int rows, int experts, int top_k, int hidden, int inter) { \
  moe_gate_up<FP8, true>(x, ids, active, slots, valid, generation, assigned, \
      shared_gate, shared_up, h, hs, rows, experts, top_k, hidden, inter); \
}
#define MOE_INVARIANT_DOWN(NAME, WORD, FP8) \
extern "C" __global__ void __launch_bounds__(DN_NT) NAME( \
    const bf16* h, const bf16* hs, const int* ids, const unsigned* active, \
    const bf16* weights, const WORD* const* slots, const unsigned* valid, \
    const unsigned long long* generation, const unsigned long long* assigned, \
    const bf16* shared_down, bf16* contributions, bf16* shared_out, unsigned* done, \
    int rows, int experts, int top_k, int hidden, int inter) { \
  moe_down<FP8, true>(h, hs, ids, active, weights, slots, valid, generation, assigned, \
      shared_down, contributions, shared_out, done, rows, experts, top_k, hidden, inter); \
}
MOE_INVARIANT_GU(moe_ffn_gate_up_lane16, bf16, false)
MOE_INVARIANT_DOWN(moe_ffn_down_lane16, bf16, false)
MOE_INVARIANT_GU(moe_ffn_gate_up_w8a16_lane16, unsigned char, true)
MOE_INVARIANT_DOWN(moe_ffn_down_w8a16_lane16, unsigned char, true)
#undef MOE_INVARIANT_GU
#undef MOE_INVARIANT_DOWN

#undef FD_GUARD_MOE_GATE_UP
#undef FD_GUARD_MOE_DOWN
