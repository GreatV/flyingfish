// Decode MoE expert FFN, bf16, rows in [1,8]: top-k routed experts addressed
// through device slot tables plus one shared expert.
// Active assignment cohort: active[r][seg] == 0 skips the assignment entirely
// (no slot dereference, no h element written, no contribution accumulated);
// the shared expert (seg == K) always computes. An active assignment traps
// when slot_valid[id] == 0 or slot_generation[id] != assignment_gen[a],
// the per-assignment snapshot of the slot generation at routing time.
// moe_ffn_gate_up: grid (ceil(I/32), rows, K+1), block 256, dynamic smem
//   hidden*2 bytes. Each 8-lane group computes one intermediate element
//   h[r][seg][j] = bf16(bf16(silu(bf16(x_r.gate_j))) * bf16(x_r.up_j));
//   seg < K reads rows j and I+j of expert ids[r][seg]'s gate_up block,
//   seg == K reads shared_gate/shared_up. Block (0, r, K) lane group 0 also
//   stores scale[r] = f32(bf16(sigmoid(f32(bf16(x_r.shared_expert_gate))))).
// moe_ffn_down: grid (ceil(H/16), rows), block 128. Each 8-lane group computes
//   y[r][d] = bf16(sum over active k of
//                 f32(bf16(f32(bf16(h[r][k].down_e_k[d])) * w[r][k]))
//                 + f32(bf16(scale[r] * bf16(h[r][K].shared_down[d]))))
//   with the shared term (k == K) added last; y is written for every row.
// Weight bytes per row at E=256,K=8,H=2048,I=512: gate_up streams
// K*2I*H*2 + 2I*H*2 = 37.7 MB, down streams K*H*I*2 + H*I*2 = 18.9 MB;
// 56.6 MB total, no expert byte read twice within a launch. h makes one round
// trip through scratch: rows*(K+1)*I*2*2 = 147456 B at rows=8, 0.26% of the
// streamed bytes. 16-byte vector loads; every expert row is read once, by one
// 8-lane group sweeping it in 128-byte warp segments.
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
  if (blockDim.x != DN_NT || top_k < 1 || (inter & 7) != 0) __trap();              \
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
__device__ __forceinline__ float moe_sigmoid(float v) { return 1.0f / (1.0f + expf(-v)); }

extern "C" __global__ void __launch_bounds__(GU_NT) moe_ffn_gate_up(
    const bf16* __restrict__ x, const int* __restrict__ ids,
    const unsigned* __restrict__ active,
    const bf16* const* __restrict__ slots_gate_up,
    const unsigned* __restrict__ slot_valid,
    const unsigned long long* __restrict__ slot_generation,
    const unsigned long long* __restrict__ assignment_gen,
    const bf16* __restrict__ shared_gate, const bf16* __restrict__ shared_up,
    const bf16* __restrict__ shared_expert_gate,
    bf16* __restrict__ h, float* __restrict__ scale,
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
  // duplicate id within a row is an invalid input.
  if (seg == 0 && blockIdx.x == 0 && tid == 0) {
    for (int c0 = 0; c0 < top_k; ++c0)
      for (int c1 = c0 + 1; c1 < top_k; ++c1)
        if (ids[(size_t)r * top_k + c0] == ids[(size_t)r * top_k + c1]) __trap();
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

  float sacc = 0.0f;
  if (seg == top_k && blockIdx.x == 0 && tid < LANES) {
    const uint4* sg = reinterpret_cast<const uint4*>(shared_expert_gate);
#pragma unroll 4
    for (int v = lane; v < nvec; v += LANES) sacc += moe_dot8(sg[v], xq[v]);
  }
  sacc = moe_group_sum(sacc);
  if (tid == 0 && seg == top_k && blockIdx.x == 0) {
    const float dot = __bfloat162float(__float2bfloat16(sacc));
    scale[r] = __bfloat162float(__float2bfloat16(moe_sigmoid(dot)));
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
    h[((size_t)r * (top_k + 1) + seg) * inter + j] = __float2bfloat16(s * u);
  }
}

extern "C" __global__ void __launch_bounds__(DN_NT) moe_ffn_down(
    const bf16* __restrict__ h, const int* __restrict__ ids,
    const unsigned* __restrict__ active, const bf16* __restrict__ weights,
    const bf16* const* __restrict__ slots_down,
    const unsigned* __restrict__ slot_valid,
    const unsigned long long* __restrict__ slot_generation,
    const unsigned long long* __restrict__ assignment_gen,
    const bf16* __restrict__ shared_down, const float* __restrict__ scale,
    bf16* __restrict__ y, int experts, int top_k, int hidden, int inter) {
  FD_GUARD_MOE_DOWN;
  const int r = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & (LANES - 1);
  const int d = blockIdx.x * DN_ROWS + (tid / LANES);
  const int ivec = inter >> 3;
  const float gate_scale = scale[r];
  float total = 0.0f;
  for (int k = 0; k <= top_k; ++k) {
    const bool routed = k < top_k;
    const bool live = !routed || active[(size_t)r * top_k + k] != 0u;
    const bf16* wrow = shared_down;
    float wk = 0.0f;
    if (routed && live) {
      const int e = ids[(size_t)r * top_k + k];
      if ((unsigned)e >= (unsigned)experts || slot_valid[e] == 0u ||
          slot_generation[e] != assignment_gen[(size_t)r * top_k + k])
        __trap();
      wrow = slots_down[e];
      if ((reinterpret_cast<unsigned long long>(wrow) & 15) != 0) __trap();
      wk = __bfloat162float(weights[(size_t)r * top_k + k]);
    }
    float acc = 0.0f;
    if (d < hidden && live) {
      const uint4* wv = reinterpret_cast<const uint4*>(wrow + (size_t)d * inter);
      const uint4* hv = reinterpret_cast<const uint4*>(h + ((size_t)r * (top_k + 1) + k) * inter);
#pragma unroll 4
      for (int v = lane; v < ivec; v += LANES) acc += moe_dot8(wv[v], hv[v]);
    }
    acc = moe_group_sum(acc);
    if (lane == 0 && d < hidden && live) {
      const float lin = __bfloat162float(__float2bfloat16(acc));
      if (routed) {
        total += __bfloat162float(__float2bfloat16(lin * wk));
      } else {
        total += __bfloat162float(__float2bfloat16(gate_scale * lin));
      }
    }
  }
  if (lane == 0 && d < hidden) y[(size_t)r * hidden + d] = __float2bfloat16(total);
}

#undef FD_GUARD_MOE_GATE_UP
#undef FD_GUARD_MOE_DOWN
