// BF16 routed and shared expert contributions for rows1..8; caller-owned done bits.
// gate_up: grid(ceil(I/32),rows,K+1), block256, smem=H*2; down: grid(ceil(H/16),rows,K+1), block128.
// Active assignments require valid slots, matching generations and distinct expert ids.
// Expert row pointers and shared weights are 16-byte aligned.
#include <cuda_runtime.h>
#include <cuda_bf16.h>

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
  FD_GUARD_MOE_GATE_UP;
  const int r = blockIdx.y;
  const int seg = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & (LANES - 1);
  const int j = blockIdx.x * GU_ROWS + (tid / LANES);
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
  // Router outputs carry K distinct experts per row (torch.topk); a
  // duplicate id among a row's active choices is an invalid input.
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
    const bf16* gu = slots_gate_up[e];
    if ((reinterpret_cast<unsigned long long>(gu) & 15) != 0) __trap();
    gate_base = gu;
    up_base = gu + (size_t)inter * hidden;
  }

  float accg = 0.0f;
  float accu = 0.0f;
  if (j < inter && live_seg) {
    const uint4* gv = reinterpret_cast<const uint4*>(gate_base + (size_t)j * hidden);
    const uint4* uv = reinterpret_cast<const uint4*>(up_base + (size_t)j * hidden);
#pragma unroll 4
    for (int v = lane; v < nvec; v += LANES) {
      accg += moe_dot8(gv[v], xq[v]);
      accu += moe_dot8(uv[v], xq[v]);
    }
  }
  accg = moe_group_sum(accg);
  accu = moe_group_sum(accu);
  if (lane == 0 && j < inter && live_seg) {
    const float g = __bfloat162float(__float2bfloat16(accg));
    const float u = __bfloat162float(__float2bfloat16(accu));
    const float s = __bfloat162float(__float2bfloat16(moe_silu(g)));
    bf16* out = seg == top_k ? h_shared + (size_t)r * inter
                             : h + ((size_t)r * top_k + seg) * inter;
    out[j] = __float2bfloat16(s * u);
  }
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
  FD_GUARD_MOE_DOWN;
  const int r = blockIdx.y;
  const int seg = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & (LANES - 1);
  const int d = blockIdx.x * DN_ROWS + (tid / LANES);
  const int ivec = inter >> 3;
  const bool routed = seg < top_k;
  const bool live = !routed || active[(size_t)r * top_k + seg] != 0u;
  const bf16* wrow = shared_down;
  float wk = 0.0f;
  if (routed && live) {
    const int e = ids[(size_t)r * top_k + seg];
    if ((unsigned)e >= (unsigned)experts || slot_valid[e] == 0u ||
        slot_generation[e] != assignment_gen[(size_t)r * top_k + seg])
      __trap();
    wrow = slots_down[e];
    if ((reinterpret_cast<unsigned long long>(wrow) & 15) != 0) __trap();
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
    const uint4* wv = reinterpret_cast<const uint4*>(wrow + (size_t)d * inter);
    const uint4* hv = reinterpret_cast<const uint4*>(hrow);
#pragma unroll 4
    for (int v = lane; v < ivec; v += LANES) acc += moe_dot8(wv[v], hv[v]);
  }
  acc = moe_group_sum(acc);
  if (lane == 0 && d < hidden && live) {
    const float lin = __bfloat162float(__float2bfloat16(acc));
    bf16* out = routed ? contributions + ((size_t)r * top_k + seg) * (size_t)hidden
                       : shared_out + (size_t)r * (size_t)hidden;
    out[d] = routed ? __float2bfloat16(lin * wk) : __float2bfloat16(lin);
  }
}

#undef FD_GUARD_MOE_GATE_UP
#undef FD_GUARD_MOE_DOWN
