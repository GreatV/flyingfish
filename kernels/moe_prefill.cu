// MoE prefill kernels: per-expert count, exclusive scan, assignment permute,
// SwiGLU and weighted combine. The two grouped GEMMs run through cuBLAS
// (cublasGemmGroupedBatchedEx) between permute/swiglu/combine.
// Assignment space is rows*top_k; every kernel takes the dims as runtime
// arguments and __trap()s on contract violations (experts > 1024, top_k > 32,
// active id outside [0,experts), slot_valid == 0 or a slot_generation mismatch
// on an active assignment, permute slot past the expert end, provenance join
// mismatch in the combine). Inactive assignments (active[a] == 0) are skipped
// without dereferencing ids, the slot tables or any guard.
#include <cuda_runtime.h>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

#define FD_MOE_GUARDS do {                                                              \
  if (blockDim.x != 1024 || experts < 1 || experts > 1024) __trap();                    \
} while (0)

// One block counts the ACTIVE assignments into per-expert counts with shared
// atomics and writes the active-assignment total; single block, so counts
// needs no pre-zeroed global state.
extern "C" __global__ void moe_count(const int* __restrict__ ids,
                                     const unsigned* __restrict__ active,
                                     int* __restrict__ counts,
                                     int* __restrict__ active_total,
                                     long long assignments, int experts,
                                     int top_k) {
  FD_MOE_GUARDS;
  __shared__ int sc[1024];
  __shared__ int sa;
  const int tid = threadIdx.x;
  sc[tid] = 0;
  if (tid == 0) sa = 0;
  __syncthreads();
  for (long long a = tid; a < assignments; a += blockDim.x) {
    if (active[a] == 0u) continue;
    const int e = ids[a];
    if (e < 0 || e >= experts) __trap();
    atomicAdd(&sc[e], 1);
    atomicAdd(&sa, 1);
  }
  __syncthreads();
  if (tid < experts) counts[tid] = sc[tid];
  if (tid == 0) *active_total = sa;
}

// Exclusive scan of counts into offsets[0..=experts] (single block,
// Hillis-Steele over blockDim-padded counts) and the matching permute
// cursors cursor[e] = offsets[e].
extern "C" __global__ void moe_scan(const int* __restrict__ counts,
                                    int* __restrict__ offsets, int* __restrict__ cursor,
                                    int experts) {
  FD_MOE_GUARDS;
  __shared__ int sh[1024];
  const int tid = threadIdx.x;
  sh[tid] = tid < experts ? counts[tid] : 0;
  __syncthreads();
  for (int d = 1; d < 1024; d <<= 1) {
    int t = 0;
    if (tid >= d) t = sh[tid - d];
    __syncthreads();
    sh[tid] += t;
    __syncthreads();
  }
  const int excl = tid == 0 ? 0 : sh[tid - 1];
  if (tid < experts) {
    offsets[tid] = excl;
    cursor[tid] = excl;
  }
  if (tid == 0) offsets[experts] = sh[experts - 1];
}

// One block per assignment: an inactive block returns before touching ids or
// the slot tables. An active assignment claims slot pos = atomicAdd(cursor[e],
// 1) inside [offsets[e], offsets[e+1]), stores the (row, choice) provenance
// pair, the reverse map and the gathered x row. __trap() on invalid id,
// slot_valid == 0, slot_generation != assignment_gen[a] (the per-assignment
// snapshot of the slot generation at routing time) or a claimed slot
// outside the expert segment.
extern "C" __global__ void moe_permute(
    const int* __restrict__ ids, const bf16* __restrict__ weights,
    const bf16* __restrict__ x, const int* __restrict__ offsets,
    int* __restrict__ cursor, const unsigned* __restrict__ valid,
    const unsigned long long* __restrict__ generation,
    const unsigned long long* __restrict__ assignment_gen,
    const unsigned* __restrict__ active,
    int* __restrict__ src_row, int* __restrict__ src_choice,
    bf16* __restrict__ w_row, int* __restrict__ inv_pos,
    bf16* __restrict__ x_perm, int rows, int top_k, int experts,
    long long hidden, long long x_stride) {
  if (blockDim.x != 256) __trap();
  const int a = blockIdx.x;
  if (active[a] == 0u) return;
  const int row = a / top_k;
  const int choice = a % top_k;
  const int e = ids[a];
  if (row < 0 || row >= rows || e < 0 || e >= experts) __trap();
  if (valid[e] == 0u || generation[e] != assignment_gen[a]) __trap();
  // Router outputs carry K distinct experts per row (torch.topk).
  if (threadIdx.x == 0) {
    for (int c0 = 0; c0 < top_k; ++c0)
      for (int c1 = c0 + 1; c1 < top_k; ++c1)
        if (ids[(size_t)row * top_k + c0] == ids[(size_t)row * top_k + c1]) __trap();
  }
  // One cursor claim per block: thread 0 atomics and broadcasts pos; a
  // per-thread atomicAdd would consume blockDim.x segment slots.
  __shared__ int spos;
  if (threadIdx.x == 0) {
    const int pos = atomicAdd(&cursor[e], 1);
    if (pos < offsets[e] || pos >= offsets[e + 1]) __trap();
    src_row[pos] = row;
    src_choice[pos] = choice;
    w_row[pos] = weights[a];
    inv_pos[a] = pos;
    spos = pos;
  }
  __syncthreads();
  const int pos = spos;
  const bf16* __restrict__ xr = x + (long long)row * x_stride;
  bf16* __restrict__ dst = x_perm + (long long)pos * hidden;
  for (long long c = threadIdx.x; c < hidden; c += blockDim.x) dst[c] = xr[c];
}

// act[r][j] = bf16(bf16(silu(g)) * u) with g = gu[r][j], u = gu[r][inter+j].
// grid.y is the row index, grid.x covers the inter columns.
extern "C" __global__ void moe_swiglu(const bf16* __restrict__ gu,
                                      bf16* __restrict__ act, long long rows,
                                      int inter) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= inter) return;
  const long long r = blockIdx.y;
  const bf16* grow = gu + r * 2 * (long long)inter;
  const float g = __bfloat162float(grow[j]);
  const bf16 s = __float2bfloat16(g / (1.0f + expf(-g)));
  act[r * inter + j] = __hmul(s, grow[inter + j]);
}

// One block per row. Combines the ACTIVE top_k weighted expert rows with the
// shared expert (always computed): y[row][d] = bf16( sum over active k of
// bf16(Y_perm[pos_k][d] * w_k) + bf16(scale * shared_y[row][d]) ) with FP32
// accumulation, scale = bf16(sigmoid(bf16(dot(x_row, gate_vec)))). The bf16
// boundaries match reference::moe_row_with Combine::Fp32. Each active term
// checks the stored (row, choice) provenance pair against its join slot.
extern "C" __global__ void moe_combine(
    const bf16* __restrict__ y_perm, const bf16* __restrict__ w_row,
    const int* __restrict__ inv_pos, const int* __restrict__ src_row,
    const int* __restrict__ src_choice, const unsigned* __restrict__ active,
    const bf16* __restrict__ x, const bf16* __restrict__ gate_vec,
    const bf16* __restrict__ shared_y, bf16* __restrict__ y, int rows,
    int top_k, long long hidden, long long x_stride) {
  if (blockDim.x != 256 || top_k < 1 || top_k > 32) __trap();
  const int row = blockIdx.x;
  if (row < 0 || row >= rows) __trap();
  const int tid = threadIdx.x;
  const int a0 = row * top_k;
  int pos[32];
  float w[32];
  int live = 0;
  for (int kk = 0; kk < top_k; ++kk) {
    if (active[a0 + kk] == 0u) continue;
    const int p = inv_pos[a0 + kk];
    if (src_row[p] != row || src_choice[p] != kk) __trap();
    pos[live] = p;
    w[live] = __bfloat162float(w_row[p]);
    ++live;
  }
  const bf16* __restrict__ xr = x + (long long)row * x_stride;
  float part = 0.0f;
  for (long long c = tid; c < hidden; c += blockDim.x)
    part += __bfloat162float(xr[c]) * __bfloat162float(gate_vec[c]);
  for (int off = 16; off > 0; off >>= 1)
    part += __shfl_down_sync(0xffffffffu, part, off);
  __shared__ float red[8];
  __shared__ float dot_sh;
  if ((tid & 31) == 0) red[tid >> 5] = part;
  __syncthreads();
  if (tid < 32) {
    float v = tid < (blockDim.x >> 5) ? red[tid] : 0.0f;
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(0xffffffffu, v, off);
    if (tid == 0) dot_sh = v;
  }
  __syncthreads();
  const float dot = dot_sh;
  const float sig = 1.0f / (1.0f + expf(-__bfloat162float(
      __float2bfloat16(dot))));
  const float scale = __bfloat162float(__float2bfloat16(sig));
  const bf16* __restrict__ sh = shared_y + (long long)row * hidden;
  bf16* __restrict__ yr = y + (long long)row * hidden;
  for (long long d = tid; d < hidden; d += blockDim.x) {
    float acc = __bfloat162float(__hmul(__float2bfloat16(scale), sh[d]));
    for (int kk = 0; kk < live; ++kk) {
      acc += __bfloat162float(__hmul(
          y_perm[(long long)pos[kk] * hidden + d], __float2bfloat16(w[kk])));
    }
    yr[d] = __float2bfloat16(acc);
  }
}

#undef FD_MOE_GUARDS
