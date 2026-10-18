// Active-assignment count, exclusive scan, row permutation and weighted BF16 scatter.
// Active expert ids, slot generations, done bits and segment positions are checked before use.
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

// One block per assignment; active choices claim one segment position and one done bit.
// Invalid ids, generations, duplicate choices and out-of-segment positions trap.
extern "C" __global__ void moe_permute(
    const int* __restrict__ ids, const bf16* __restrict__ x,
    const int* __restrict__ offsets,
    int* __restrict__ cursor, const unsigned* __restrict__ valid,
    const unsigned long long* __restrict__ generation,
    const unsigned long long* __restrict__ assignment_gen,
    const unsigned* __restrict__ active, unsigned* __restrict__ done,
    int* __restrict__ pos_of, bf16* __restrict__ x_perm, int rows, int top_k,
    int experts, long long hidden) {
  if (blockDim.x != 256) __trap();
  const int a = blockIdx.x;
  const int row = a / top_k;
  const int choice = a % top_k;
  if (choice == 0 && threadIdx.x == 0) {
    for (int c0 = 0; c0 < top_k; ++c0) {
      if (active[(size_t)row * top_k + c0] == 0u) continue;
      for (int c1 = c0 + 1; c1 < top_k; ++c1) {
        if (active[(size_t)row * top_k + c1] == 0u) continue;
        if (ids[(size_t)row * top_k + c0] == ids[(size_t)row * top_k + c1]) __trap();
      }
    }
  }
  if (active[a] == 0u) return;
  const int e = ids[a];
  if (row < 0 || row >= rows || e < 0 || e >= experts) __trap();
  if (valid[e] == 0u || generation[e] != assignment_gen[a]) __trap();
  // One cursor claim per block: thread 0 atomics and broadcasts pos; a
  // per-thread atomicAdd would consume blockDim.x segment slots.
  __shared__ int spos;
  if (threadIdx.x == 0) {
    if (done[a] != 0u) __trap();
    const int pos = atomicAdd(&cursor[e], 1);
    if (pos < offsets[e] || pos >= offsets[e + 1]) __trap();
    pos_of[a] = pos;
    done[a] = 1u;
    spos = pos;
  }
  __syncthreads();
  const int pos = spos;
  const bf16* __restrict__ xr = x + (long long)row * hidden;
  bf16* __restrict__ dst = x_perm + (long long)pos * hidden;
  for (long long c = threadIdx.x; c < hidden; c += blockDim.x) dst[c] = xr[c];
}

// contributions[a] = BF16(y_perm[pos_of[a]] * weights[a]) for active assignments.
extern "C" __global__ void moe_scatter(
    const bf16* __restrict__ y_perm, const bf16* __restrict__ weights,
    const int* __restrict__ pos_of, const unsigned* __restrict__ active,
    bf16* __restrict__ contributions, long long assignments,
    long long hidden) {
  if (blockDim.x != 256) __trap();
  const long long a = blockIdx.x;
  if (a >= assignments) __trap();
  if (active[a] == 0u) return;
  const long long pos = pos_of[a];
  if (pos < 0 || pos >= assignments) __trap();
  const float w = __bfloat162float(weights[a]);
  const bf16* __restrict__ src = y_perm + pos * hidden;
  bf16* __restrict__ dst = contributions + a * hidden;
  for (long long c = threadIdx.x; c < hidden; c += blockDim.x)
    dst[c] = __float2bfloat16(__bfloat162float(src[c]) * w);
}

#undef FD_MOE_GUARDS
