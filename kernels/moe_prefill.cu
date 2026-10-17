// MoE prefill kernels: per-expert count, exclusive scan, assignment permute
// and the router-weight scaling of the per-assignment contributions. The two
// grouped GEMMs run through cuBLAS (cublasGemmGroupedBatchedEx), the
// activation through silu_mul (kernels/act.cu) and the final combine through
// moe_combine (kernels/moe_combine.cu).
// Assignment space is rows*top_k; every kernel takes the dims as runtime
// arguments and __trap()s on contract violations (experts > 1024, top_k > 32,
// active id outside [0,experts), slot_valid == 0 or a slot_generation mismatch
// on an active assignment, duplicate expert ids among a row's active choices,
// a claimed slot outside the expert segment, done already set on an active
// assignment). Inactive assignments (active[a] == 0) are skipped without
// dereferencing ids, the slot tables or any guard.
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

// One block per assignment. The choice-0 block of a row checks its ACTIVE
// choices for duplicate expert ids once per row, then an inactive block
// returns before touching ids or the slot tables. An active assignment with
// done[a] != 0 traps; otherwise it claims slot pos = atomicAdd(cursor[e], 1)
// inside [offsets[e], offsets[e+1]), stores the (row, choice) provenance
// pair, gathers the x row and sets done[a] = 1. __trap() on invalid id,
// slot_valid == 0, slot_generation != assignment_gen[a] (the per-assignment
// snapshot of the slot generation at routing time) or a claimed slot outside
// the expert segment.
extern "C" __global__ void moe_permute(
    const int* __restrict__ ids, const bf16* __restrict__ x,
    const int* __restrict__ offsets,
    int* __restrict__ cursor, const unsigned* __restrict__ valid,
    const unsigned long long* __restrict__ generation,
    const unsigned long long* __restrict__ assignment_gen,
    const unsigned* __restrict__ active, unsigned* __restrict__ done,
    int* __restrict__ src_row, int* __restrict__ src_choice,
    bf16* __restrict__ x_perm, int rows, int top_k, int experts,
    long long hidden) {
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
    src_row[pos] = row;
    src_choice[pos] = choice;
    done[a] = 1u;
    spos = pos;
  }
  __syncthreads();
  const int pos = spos;
  const bf16* __restrict__ xr = x + (long long)row * hidden;
  bf16* __restrict__ dst = x_perm + (long long)pos * hidden;
  for (long long c = threadIdx.x; c < hidden; c += blockDim.x) dst[c] = xr[c];
}

// contributions[a] = bf16(contributions[a] * weights[a]) on the ACTIVE
// assignments; the grouped down-projection wrote the unweighted rows into
// contributions before this launch.
extern "C" __global__ void moe_scale(bf16* __restrict__ contributions,
                                     const bf16* __restrict__ weights,
                                     const unsigned* __restrict__ active,
                                     long long assignments, long long hidden) {
  if (blockDim.x != 256) __trap();
  const long long a = blockIdx.x;
  if (a >= assignments) __trap();
  if (active[a] == 0u) return;
  const float w = __bfloat162float(weights[a]);
  bf16* __restrict__ row = contributions + a * hidden;
  for (long long d = threadIdx.x; d < hidden; d += blockDim.x)
    row[d] = __float2bfloat16(__bfloat162float(row[d]) * w);
}

#undef FD_MOE_GUARDS
