// BF16 routed and shared contributions: legacy rows1..8, invariant rows1..16; caller-owned done bits.
// gate_up: grid(ceil(I/32),rows,K+1), block256, smem=H*2; down: grid(ceil(H/16),rows,K+1), block128.
// Active assignments require valid slots, matching generations and distinct expert ids.
// Expert row pointers and shared weights are 16-byte aligned.
#if !defined(__HIP__)
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#endif
#include "q38_contract.cuh"

using bf16 = __nv_bfloat16;

constexpr int LANES = 8;    // lanes cooperating on one dot product
constexpr int GU_ROWS = 32;  // gate/up output rows per block
constexpr int GU_NT = LANES * GU_ROWS;
constexpr int DN_ROWS = 16;  // down output rows per block
constexpr int DN_NT = LANES * DN_ROWS;

#define FD_GUARD_MOE_GATE_UP do {                                                  \
  if (blockDim.x != GU_NT || gridDim.y != rows ||                                  \
      (gridDim.z != top_k + 1 && gridDim.z != top_k) ||                            \
      rows < 1 || rows > 8 || top_k < 0 || gridDim.z < 1 ||                        \
      (hidden & 7) != 0 || inter < 1 || shared_inter < 1 ||                        \
      (shared_inter & 7) != 0)                                                     \
    __trap();                                                                      \
} while (0)

#define FD_GUARD_MOE_DOWN do {                                                     \
  if (blockDim.x != DN_NT || gridDim.y != rows ||                                  \
      (gridDim.z != top_k + 1 && gridDim.z != top_k) ||                            \
      rows < 1 || rows > 8 || top_k < 0 || gridDim.z < 1 ||                        \
      (inter & 7) != 0 || (shared_inter & 7) != 0)                                 \
    __trap();                                                                      \
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
    v += __shfl_down_sync(FF_WARP_MASK, v, off, LANES);
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
  const auto mask = (FF_WARP_MASK & 0xffffu) << (tid & 16);
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
    int rows, int experts, int top_k, int hidden, int inter, int shared_inter,
    int rounding_once) {
  if ((unsigned)rounding_once > 1u) __trap();
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
  const int out_inter = seg == top_k ? shared_inter : inter;
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
  if (j < out_inter && live_seg) {
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
  if (lane == 0 && j < out_inter && live_seg) {
    const float g = __bfloat162float(__float2bfloat16(accg));
    const float u = __bfloat162float(__float2bfloat16(accu));
    const float activation = INVARIANT
        ? __fmul_rn(g, __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g))))
        : moe_silu(g);
    bf16* out = seg == top_k ? h_shared + (size_t)r * shared_inter
                             : h + ((size_t)r * top_k + seg) * inter;
    out[j] = rounding_once
        ? __float2bfloat16(activation * u)
        : __float2bfloat16(__bfloat162float(__float2bfloat16(activation)) * u);
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
    int rows, int experts, int top_k, int hidden, int inter, int shared_inter,
    int routed_weight_fp32) {
  if ((unsigned)routed_weight_fp32 > 1u) __trap();
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
  const bool routed = seg < top_k;
  const bool live = !routed || active[(size_t)r * top_k + seg] != 0u;
  const int row_inter = routed ? inter : shared_inter;
  const int ivec = row_inter >> 3;
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
  const bf16* hrow = routed
      ? h + ((size_t)r * top_k + seg) * inter
      : h_shared + (size_t)r * shared_inter;
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
    const bf16* wv = wrow + (size_t)d * row_inter;
    const uint4* hv = reinterpret_cast<const uint4*>(hrow);
    if constexpr (INVARIANT) {
      for (int col = lane; col < row_inter; col += 16)
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
    bf16* out = routed ? contributions + ((size_t)r * top_k + seg) * (size_t)hidden
                       : shared_out + (size_t)r * (size_t)hidden;
    out[d] = routed
        ? __float2bfloat16(routed_weight_fp32
                               ? acc * wk
                               : __bfloat162float(__float2bfloat16(acc)) * wk)
        : __float2bfloat16(__bfloat162float(__float2bfloat16(acc)));
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
    int rows, int experts, int top_k, int hidden, int inter, int shared_inter,
    int rounding_once) {
  moe_gate_up<false>(x, ids, active, slots_gate_up, slot_valid, slot_generation, assignment_gen, shared_gate, shared_up, h, h_shared, rows, experts, top_k, hidden, inter, shared_inter, rounding_once);
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
    int rows, int experts, int top_k, int hidden, int inter, int shared_inter,
    int rounding_once) {
  moe_gate_up<true>(x, ids, active, slots_gate_up, slot_valid, slot_generation, assignment_gen, shared_gate, shared_up, h, h_shared, rows, experts, top_k, hidden, inter, shared_inter, rounding_once);
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
    int rows, int experts, int top_k, int hidden, int inter, int shared_inter,
    int routed_weight_fp32) {
  moe_down<false>(h, h_shared, ids, active, weights, slots_down, slot_valid, slot_generation, assignment_gen, shared_down, contributions, shared_out, done, rows, experts, top_k, hidden, inter, shared_inter, routed_weight_fp32);
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
    int rows, int experts, int top_k, int hidden, int inter, int shared_inter,
    int routed_weight_fp32) {
  moe_down<true>(h, h_shared, ids, active, weights, slots_down, slot_valid, slot_generation, assignment_gen, shared_down, contributions, shared_out, done, rows, experts, top_k, hidden, inter, shared_inter, routed_weight_fp32);
}

#define MOE_INVARIANT_GU(NAME, WORD, FP8) \
extern "C" __global__ void __launch_bounds__(GU_NT) NAME( \
    const bf16* x, const int* ids, const unsigned* active, \
    const WORD* const* slots, const unsigned* valid, \
    const unsigned long long* generation, const unsigned long long* assigned, \
    const bf16* shared_gate, const bf16* shared_up, bf16* h, bf16* hs, \
    int rows, int experts, int top_k, int hidden, int inter, int shared_inter, \
    int rounding_once) { \
  moe_gate_up<FP8, true>(x, ids, active, slots, valid, generation, assigned, \
      shared_gate, shared_up, h, hs, rows, experts, top_k, hidden, inter, \
      shared_inter, rounding_once); \
}
#define MOE_INVARIANT_DOWN(NAME, WORD, FP8) \
extern "C" __global__ void __launch_bounds__(DN_NT) NAME( \
    const bf16* h, const bf16* hs, const int* ids, const unsigned* active, \
    const bf16* weights, const WORD* const* slots, const unsigned* valid, \
    const unsigned long long* generation, const unsigned long long* assigned, \
    const bf16* shared_down, bf16* contributions, bf16* shared_out, unsigned* done, \
    int rows, int experts, int top_k, int hidden, int inter, int shared_inter, \
    int routed_weight_fp32) { \
  moe_down<FP8, true>(h, hs, ids, active, weights, slots, valid, generation, assigned, \
      shared_down, contributions, shared_out, done, rows, experts, top_k, hidden, inter, \
      shared_inter, routed_weight_fp32); \
}
MOE_INVARIANT_GU(moe_ffn_gate_up_lane16, bf16, false)
MOE_INVARIANT_DOWN(moe_ffn_down_lane16, bf16, false)
MOE_INVARIANT_GU(moe_ffn_gate_up_w8a16_lane16, unsigned char, true)
MOE_INVARIANT_DOWN(moe_ffn_down_w8a16_lane16, unsigned char, true)
#undef MOE_INVARIANT_GU
#undef MOE_INVARIANT_DOWN

#undef FD_GUARD_MOE_GATE_UP
#undef FD_GUARD_MOE_DOWN
#if !defined(__HIP__)

// ---------------------------------------------------------------------------
// W4A8 routed experts: E2M1 weights with one E8M0 scale per 32 elements of
// each row, E4M3 activations with one E8M0 scale per 32 elements along K.
// The packed byte at column j carries the element 2j in its low nibble and
// the element 2j+1 in its high nibble; the E2M1 code is one sign bit, two
// exponent bits and one mantissa bit.
// Semantics per the DeepSeek-V4.1-Flash model definition (publisher
// inference/model.py Expert.forward, kernel.py act_quant and fp4_gemm):
//   - the activation is quantized dynamically: the amax over 32 elements is
//     floored at 1e-4, the E8M0 scale is 2^ceil(log2(amax/448)) and the codes
//     are E4M3 round-to-nearest-even;
//   - the 32-element dot is scaled by the activation and the weight scale
//     outside the dot;
//   - the gate branch is clamped from above, the up branch on both sides, the
//     SiLU stays in FP32 and multiplies the up branch in FP32;
//   - the routed weight multiplies that FP32 product and the result takes one
//     BF16 cast, which is the down-input row.
// act quant: grid(ceil(groups/8)), block256, smem=0.
// gate_up: grid(ceil(I/8)), block256, smem=K+K/32; down: grid(ceil(H/8)),
// block256, smem=I+I/32. One warp owns one output row.

#define W4A8_SCALE_BLOCK 32  // elements per E8M0 scale
#define W4A8_ELEMS 8         // elements per chunk
#define W4A8_THREADS 256     // threads per W4A8 block
#define W4A8_WARPS 8         // output rows per W4A8 block

// E2M1 magnitude from the three low bits: 0, 0.5, 1, 1.5, 2, 3, 4 and 6.
__device__ __forceinline__ float moe_e2m1(unsigned nibble) {
  const unsigned u = nibble & 7u;
  unsigned bits = 0x3F000000u + u * 0x400000u;
  if (u == 0u) bits = 0u;
  else if (u == 1u) bits = 0x3F000000u;
  return __uint_as_float(bits | ((nibble & 8u) << 28));
}

// ceil(log2(v)) of a positive normal float and 2^exponent live in
// q38_contract.cuh (act_quant_log2_ceil / act_quant_pow2).

// Warp sum; every lane receives the total.
__device__ __forceinline__ float moe_warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// Quantize one BF16 row to E4M3 codes with one E8M0 scale per 32 elements.
__device__ __forceinline__ void moe_act_quant(
    const bf16* __restrict__ x, unsigned char* __restrict__ codes,
    unsigned char* __restrict__ scales, int k) {
  if (k < 1 || (k % W4A8_SCALE_BLOCK) != 0 ||
      blockDim.x != W4A8_THREADS || blockDim.x % 32 != 0)
    __trap();
  const int groups = k / W4A8_SCALE_BLOCK;
  const int group = int(blockIdx.x) * (int(blockDim.x) / 32) + int(threadIdx.x / 32);
  if (group >= groups) return;
  const int lane = int(threadIdx.x % 32);
  const size_t base = (size_t)group * W4A8_SCALE_BLOCK;
  const float value = __bfloat162float(x[base + lane]);
  unsigned char code, scale_code;
  act_quant32(value, &code, &scale_code);
  codes[base + lane] = code;
  if (lane == 0) scales[group] = scale_code;
}

extern "C" __global__ __launch_bounds__(W4A8_THREADS) void moe_act_quant_e4m0(
    const bf16* __restrict__ x, unsigned char* __restrict__ codes,
    unsigned char* __restrict__ scales, int k) {
  moe_act_quant(x, codes, scales, k);
}

// One warp per output row: the row of packed E2M1 weights is read in 4-byte
// chunks, the activation chunk is read from shared memory, and the block
// scales multiply the chunk sums.
__device__ __forceinline__ float moe_w4a8_dot(
    const unsigned char* __restrict__ weight_row,
    const unsigned char* __restrict__ weight_scales,
    const unsigned char* __restrict__ act_codes,
    const unsigned char* __restrict__ act_scales, int chunks) {
  float acc = 0.0f;
  for (int chunk = int(threadIdx.x % 32); chunk < chunks; chunk += 32) {
    const unsigned word =
        *reinterpret_cast<const unsigned*>(weight_row + (size_t)chunk * 4);
    const unsigned lo = *reinterpret_cast<const unsigned*>(act_codes + (size_t)chunk * 8);
    const unsigned hi = *reinterpret_cast<const unsigned*>(act_codes + (size_t)chunk * 8 + 4);
    unsigned bytes[8];
    bytes[0] = lo & 0xFFu;
    bytes[1] = (lo >> 8) & 0xFFu;
    bytes[2] = (lo >> 16) & 0xFFu;
    bytes[3] = lo >> 24;
    bytes[4] = hi & 0xFFu;
    bytes[5] = (hi >> 8) & 0xFFu;
    bytes[6] = (hi >> 16) & 0xFFu;
    bytes[7] = hi >> 24;
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const unsigned byte = (word >> (8 * i)) & 0xFFu;
      const float a_lo = e4m3_decode((unsigned char)bytes[2 * i]);
      const float a_hi = e4m3_decode((unsigned char)bytes[2 * i + 1]);
      sum = fmaf(moe_e2m1(byte & 0xFu), a_lo, sum);
      sum = fmaf(moe_e2m1(byte >> 4), a_hi, sum);
    }
    const int group = chunk / (W4A8_SCALE_BLOCK / W4A8_ELEMS);
    acc = fmaf(sum, e8m0_decode(act_scales[group]) * e8m0_decode(weight_scales[group]), acc);
  }
  return acc;
}

__device__ __forceinline__ void moe_w4a8_gate_up(
    const unsigned char* __restrict__ act_codes,
    const unsigned char* __restrict__ act_scales,
    const unsigned char* __restrict__ gate_codes,
    const unsigned char* __restrict__ gate_scales,
    const unsigned char* __restrict__ up_codes,
    const unsigned char* __restrict__ up_scales,
    bf16* __restrict__ h, int hidden, int inter, float swiglu_limit,
    float route_weight) {
  if (blockDim.x != W4A8_THREADS || hidden < 1 || inter < 1 ||
      (hidden % W4A8_SCALE_BLOCK) != 0 || !(swiglu_limit > 0.0f) ||
      !isfinite(route_weight) ||
      gridDim.x != (unsigned)((inter + W4A8_WARPS - 1) / W4A8_WARPS))
    __trap();
  const int groups = hidden / W4A8_SCALE_BLOCK;
  extern __shared__ __align__(16) unsigned char w4a8_gate_up_act[];
  unsigned char* s_codes = w4a8_gate_up_act;
  unsigned char* s_scales = w4a8_gate_up_act + hidden;
  for (int i = int(threadIdx.x); i < hidden; i += W4A8_THREADS)
    s_codes[i] = act_codes[i];
  for (int i = int(threadIdx.x); i < groups; i += W4A8_THREADS)
    s_scales[i] = act_scales[i];
  __syncthreads();
  const int row = int(blockIdx.x) * W4A8_WARPS + int(threadIdx.x / 32);
  if (row >= inter) return;
  const int chunks = hidden / W4A8_ELEMS;
  const float gate = moe_warp_sum(moe_w4a8_dot(
      gate_codes + (size_t)row * (size_t)(hidden / 2),
      gate_scales + (size_t)row * (size_t)groups, s_codes, s_scales, chunks));
  const float up = moe_warp_sum(moe_w4a8_dot(
      up_codes + (size_t)row * (size_t)(hidden / 2),
      up_scales + (size_t)row * (size_t)groups, s_codes, s_scales, chunks));
  if (int(threadIdx.x % 32) != 0) return;
  float g = __bfloat162float(__float2bfloat16_rn(gate));
  float u = __bfloat162float(__float2bfloat16_rn(up));
  g = fminf(g, swiglu_limit);
  u = fmaxf(fminf(u, swiglu_limit), -swiglu_limit);
  // The definition's boundary (model.py Expert.forward): FP32 silu times the up
  // branch, the route weight in FP32, then one BF16 cast before w2.
  const float silu = g / (1.0f + expf(-g));
  h[row] = __float2bfloat16_rn((silu * u) * route_weight);
}

__device__ __forceinline__ void moe_w4a8_down(
    const unsigned char* __restrict__ act_codes,
    const unsigned char* __restrict__ act_scales,
    const unsigned char* __restrict__ down_codes,
    const unsigned char* __restrict__ down_scales,
    bf16* __restrict__ out, int hidden, int inter) {
  if (blockDim.x != W4A8_THREADS || hidden < 1 || inter < 1 ||
      (inter % W4A8_SCALE_BLOCK) != 0 ||
      gridDim.x != (unsigned)((hidden + W4A8_WARPS - 1) / W4A8_WARPS))
    __trap();
  const int groups = inter / W4A8_SCALE_BLOCK;
  extern __shared__ __align__(16) unsigned char w4a8_down_act[];
  unsigned char* s_codes = w4a8_down_act;
  unsigned char* s_scales = w4a8_down_act + inter;
  for (int i = int(threadIdx.x); i < inter; i += W4A8_THREADS)
    s_codes[i] = act_codes[i];
  for (int i = int(threadIdx.x); i < groups; i += W4A8_THREADS)
    s_scales[i] = act_scales[i];
  __syncthreads();
  const int row = int(blockIdx.x) * W4A8_WARPS + int(threadIdx.x / 32);
  if (row >= hidden) return;
  const int chunks = inter / W4A8_ELEMS;
  const float acc = moe_warp_sum(moe_w4a8_dot(
      down_codes + (size_t)row * (size_t)(inter / 2),
      down_scales + (size_t)row * (size_t)groups, s_codes, s_scales, chunks));
  if (int(threadIdx.x % 32) == 0) out[row] = __float2bfloat16_rn(acc);
}

extern "C" __global__ __launch_bounds__(W4A8_THREADS) void moe_ffn_gate_up_e2m1_e8m0(
    const unsigned char* __restrict__ act_codes,
    const unsigned char* __restrict__ act_scales,
    const unsigned char* __restrict__ gate_codes,
    const unsigned char* __restrict__ gate_scales,
    const unsigned char* __restrict__ up_codes,
    const unsigned char* __restrict__ up_scales,
    bf16* __restrict__ h, int hidden, int inter, float swiglu_limit,
    float route_weight) {
  moe_w4a8_gate_up(act_codes, act_scales, gate_codes, gate_scales, up_codes,
                   up_scales, h, hidden, inter, swiglu_limit, route_weight);
}

extern "C" __global__ __launch_bounds__(W4A8_THREADS) void moe_ffn_down_e2m1_e8m0(
    const unsigned char* __restrict__ act_codes,
    const unsigned char* __restrict__ act_scales,
    const unsigned char* __restrict__ down_codes,
    const unsigned char* __restrict__ down_scales,
    bf16* __restrict__ out, int hidden, int inter) {
  moe_w4a8_down(act_codes, act_scales, down_codes, down_scales, out, hidden,
                inter);
}

// Device-side routed dispatch: the id and the routed weight are read from
// device memory; `slots` holds six addresses per expert in the order
// gate_codes, gate_scales, up_codes, up_scales, down_codes, down_scales. An
// id outside [0, experts) or a zero slot_valid entry traps the launch (the
// unresident-expert failure class; the launch grid stays host-known).
extern "C" __global__ __launch_bounds__(W4A8_THREADS) void moe_ffn_dispatch_gate_up_e2m1_e8m0(
    const unsigned char* __restrict__ act_codes,
    const unsigned char* __restrict__ act_scales,
    const unsigned long long* __restrict__ slots,
    const unsigned* __restrict__ slot_valid,
    const int* __restrict__ ids, const float* __restrict__ route_weights,
    unsigned long long id_index, int experts, bf16* __restrict__ h, int hidden,
    int inter, float swiglu_limit) {
  const int e = ids[id_index];
  if ((unsigned)e >= (unsigned)experts || slot_valid[e] == 0u) __trap();
  const unsigned long long* slot = slots + (size_t)e * 6u;
  moe_w4a8_gate_up(
      act_codes, act_scales,
      reinterpret_cast<const unsigned char*>(slot[0]),
      reinterpret_cast<const unsigned char*>(slot[1]),
      reinterpret_cast<const unsigned char*>(slot[2]),
      reinterpret_cast<const unsigned char*>(slot[3]), h, hidden, inter,
      swiglu_limit, route_weights[id_index]);
}

extern "C" __global__ __launch_bounds__(W4A8_THREADS) void moe_ffn_dispatch_down_e2m1_e8m0(
    const unsigned char* __restrict__ act_codes,
    const unsigned char* __restrict__ act_scales,
    const unsigned long long* __restrict__ slots,
    const unsigned* __restrict__ slot_valid,
    const int* __restrict__ ids, unsigned long long id_index, int experts,
    bf16* __restrict__ out, int hidden, int inter) {
  const int e = ids[id_index];
  if ((unsigned)e >= (unsigned)experts || slot_valid[e] == 0u) __trap();
  const unsigned long long* slot = slots + (size_t)e * 6u;
  moe_w4a8_down(act_codes, act_scales,
                reinterpret_cast<const unsigned char*>(slot[4]),
                reinterpret_cast<const unsigned char*>(slot[5]), out, hidden,
                inter);
}

// ---------------------------------------------------------------------------
// W8A8 shared expert: E4M3 weights with one E8M0 scale per 32x32 weight
// block and an E4M3 activation with one E8M0 scale per 32 elements along K.
// The weight scale grid is [ceil(rows/32), ceil(k/32)]; the scale of the
// block (row/32, k/32) multiplies the 32-element dot of that block.
// Semantics per the DeepSeek-V4.1-Flash model definition (publisher
// inference/kernel.py fp8_gemm and act_quant, model.py Expert.forward):
//   - the activation is quantized dynamically: the amax over 32 elements is
//     floored at 1e-4, the E8M0 scale is 2^ceil(log2(amax/448)) and the codes
//     are E4M3 round-to-nearest-even;
//   - the 32-element dot is scaled by the activation scale and the weight
//     block scale outside the dot;
//   - the gate branch is clamped from above, the up branch on both sides, the
//     SiLU stays in FP32 and multiplies the up branch in FP32, one BF16 cast;
//   - the down input is the BF16 hidden state and the output is BF16.
// act quant: grid(ceil(groups/8)), block256, smem=0.
// gate_up: grid(ceil(I/8)), block256, smem=K+K/32; down: grid(ceil(H/8)),
// block256, smem=I+I/32. One warp owns one output row.

#define W8A8_BLOCK 32     // rows and columns of one weight scale block
#define W8A8_LANE_ELEMS 8 // elements of one K block per lane

// One warp owns one output row. Each 32-element K block is split over four
// lanes of eight elements and the block scale multiplies the block sum.
__device__ __forceinline__ float moe_w8a8_dot(
    const unsigned char* __restrict__ weight_row,
    const unsigned char* __restrict__ weight_scales,
    const unsigned char* __restrict__ act_codes,
    const unsigned char* __restrict__ act_scales, int groups) {
  const int lane = int(threadIdx.x % 32);
  const int sub = lane & 3;
  const int stride = lane >> 2;
  float acc = 0.0f;
  for (int base = 0; base < groups; base += 8) {
    const int block = base + stride;
    float sum = 0.0f;
    if (block < groups) {
      const unsigned char* w =
          weight_row + (size_t)block * W8A8_BLOCK + sub * W8A8_LANE_ELEMS;
      const unsigned char* a =
          act_codes + (size_t)block * W8A8_BLOCK + sub * W8A8_LANE_ELEMS;
#pragma unroll
      for (int i = 0; i < W8A8_LANE_ELEMS; ++i)
        sum = fmaf(e4m3_decode(w[i]), e4m3_decode(a[i]), sum);
    }
    sum += __shfl_xor_sync(0xffffffffu, sum, 1, 4);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2, 4);
    if (sub == 0 && block < groups) {
      const float scale =
          e8m0_decode(act_scales[block]) * e8m0_decode(weight_scales[block]);
      acc = fmaf(sum, scale, acc);
    }
  }
  return acc;
}

extern "C" __global__ __launch_bounds__(W4A8_THREADS) void moe_ffn_gate_up_e4m3_block32(
    const unsigned char* __restrict__ act_codes,
    const unsigned char* __restrict__ act_scales,
    const unsigned char* __restrict__ gate_codes,
    const unsigned char* __restrict__ gate_scales,
    const unsigned char* __restrict__ up_codes,
    const unsigned char* __restrict__ up_scales,
    bf16* __restrict__ h, int hidden, int inter, float swiglu_limit) {
  if (blockDim.x != W4A8_THREADS || hidden < 1 || inter < 1 ||
      (hidden % W8A8_BLOCK) != 0 || !(swiglu_limit > 0.0f) ||
      gridDim.x != (unsigned)((inter + W4A8_WARPS - 1) / W4A8_WARPS))
    __trap();
  const int groups = hidden / W8A8_BLOCK;
  extern __shared__ __align__(16) unsigned char w8a8_gate_up_act[];
  unsigned char* s_codes = w8a8_gate_up_act;
  unsigned char* s_scales = w8a8_gate_up_act + hidden;
  for (int i = int(threadIdx.x); i < hidden; i += W4A8_THREADS)
    s_codes[i] = act_codes[i];
  for (int i = int(threadIdx.x); i < groups; i += W4A8_THREADS)
    s_scales[i] = act_scales[i];
  __syncthreads();
  const int row = int(blockIdx.x) * W4A8_WARPS + int(threadIdx.x / 32);
  if (row >= inter) return;
  const float gate = moe_warp_sum(moe_w8a8_dot(
      gate_codes + (size_t)row * (size_t)hidden,
      gate_scales + (size_t)(row >> 5) * (size_t)groups,
      s_codes, s_scales, groups));
  const float up = moe_warp_sum(moe_w8a8_dot(
      up_codes + (size_t)row * (size_t)hidden,
      up_scales + (size_t)(row >> 5) * (size_t)groups,
      s_codes, s_scales, groups));
  if (int(threadIdx.x % 32) != 0) return;
  float g = __bfloat162float(__float2bfloat16_rn(gate));
  float u = __bfloat162float(__float2bfloat16_rn(up));
  g = fminf(g, swiglu_limit);
  u = fmaxf(fminf(u, swiglu_limit), -swiglu_limit);
  // The definition's boundary (model.py Expert.forward): FP32 silu times the up
  // branch, then one BF16 cast before w2.
  const float silu = g / (1.0f + expf(-g));
  h[row] = __float2bfloat16_rn(silu * u);
}

extern "C" __global__ __launch_bounds__(W4A8_THREADS) void moe_ffn_down_e4m3_block32(
    const unsigned char* __restrict__ act_codes,
    const unsigned char* __restrict__ act_scales,
    const unsigned char* __restrict__ down_codes,
    const unsigned char* __restrict__ down_scales,
    bf16* __restrict__ out, int hidden, int inter) {
  if (blockDim.x != W4A8_THREADS || hidden < 1 || inter < 1 ||
      (inter % W8A8_BLOCK) != 0 ||
      gridDim.x != (unsigned)((hidden + W4A8_WARPS - 1) / W4A8_WARPS))
    __trap();
  const int groups = inter / W8A8_BLOCK;
  extern __shared__ __align__(16) unsigned char w8a8_down_act[];
  unsigned char* s_codes = w8a8_down_act;
  unsigned char* s_scales = w8a8_down_act + inter;
  for (int i = int(threadIdx.x); i < inter; i += W4A8_THREADS)
    s_codes[i] = act_codes[i];
  for (int i = int(threadIdx.x); i < groups; i += W4A8_THREADS)
    s_scales[i] = act_scales[i];
  __syncthreads();
  const int row = int(blockIdx.x) * W4A8_WARPS + int(threadIdx.x / 32);
  if (row >= hidden) return;
  const float acc = moe_warp_sum(moe_w8a8_dot(
      down_codes + (size_t)row * (size_t)inter,
      down_scales + (size_t)(row >> 5) * (size_t)groups,
      s_codes, s_scales, groups));
  if (int(threadIdx.x % 32) == 0) out[row] = __float2bfloat16_rn(acc);
}
#endif
