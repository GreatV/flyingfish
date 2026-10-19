// Active-assignment count, exclusive scan, row permutation and weighted BF16 scatter.
// Active expert ids, slot generations, done bits and segment positions are checked before use.
#include <cuda_runtime.h>
#include "q38_contract.cuh"

using bf16 = __nv_bfloat16;

// Sixteen K-strided FMA accumulators and the CPU expert's pairwise tree.
template <bool FP8>
__device__ __forceinline__ void moe_lane16(
    const unsigned long long* __restrict__ matrices, int problems,
    int first, int columns, int inner, unsigned long long scale_offset) {
  if (blockDim.x != 256 || first < 0 || first >= problems ||
      columns < 1 || inner < 1) __trap();
  const int problem = first + blockIdx.z;
  if (problem >= problems) __trap();
  const int row = blockIdx.y;
  if (row >= matrices[3 * problems + problem]) return;
  const int column = blockIdx.x * 16 + threadIdx.x / 16;
  if (column >= columns) return;
  const int lane = threadIdx.x % 16;
  const bf16* __restrict__ weight =
      reinterpret_cast<const bf16*>(matrices[problem]) + (size_t)column * inner;
  const bf16* __restrict__ x =
      reinterpret_cast<const bf16*>(matrices[problems + problem]) + (size_t)row * inner;
  bf16* __restrict__ y =
      reinterpret_cast<bf16*>(matrices[2 * problems + problem]) + (size_t)row * columns;
  float sum = 0.0f;
  for (int k = lane; k < inner; k += 16) {
    bf16 value;
    if constexpr (FP8) {
      const unsigned char* codes = reinterpret_cast<const unsigned char*>(matrices[problem]);
      const bf16* scales = reinterpret_cast<const bf16*>(codes + scale_offset);
      value = __float2bfloat16_rn(e4m3_decode(codes[(size_t)column * inner + k]) *
                                __bfloat162float(scales[column]));
    } else {
      value = weight[k];
    }
    sum = __fmaf_rn(__bfloat162float(x[k]), __bfloat162float(value), sum);
  }
  const unsigned mask = 0xFFFFu << (threadIdx.x & 16);
  for (int step = 1; step < 16; step *= 2)
    sum = __fadd_rn(sum, __shfl_down_sync(mask, sum, step, 16));
  if (lane == 0) y[column] = __float2bfloat16_rn(sum);
}

extern "C" __global__ void moe_gemm_lane16(
    const unsigned long long* __restrict__ matrices, int problems,
    int first, int columns, int inner) {
  moe_lane16<false>(matrices, problems, first, columns, inner, 0);
}

extern "C" __global__ void moe_gemm_lane16_w8a16(
    const unsigned long long* __restrict__ matrices, int problems,
    int first, int columns, int inner, unsigned long long scale_offset) {
  moe_lane16<true>(matrices, problems, first, columns, inner, scale_offset);
}

// A128x64, B64x64; four warps own64x32 outputs, with increasing k16 MMA steps.
extern "C" __global__ void __launch_bounds__(128) moe_gemm_w8a16(
    const unsigned long long* __restrict__ matrices, int problems,
    int tensor_problems, int columns, int inner, unsigned long long scale_offset) {
  if (blockDim.x != 128 || tensor_problems < 1 || tensor_problems > problems ||
      columns < 1 || inner < 1 || (inner & 7) != 0 || blockIdx.z >= tensor_problems)
    __trap();
  constexpr int CK = 64, LDS = 72;
  const int problem = blockIdx.z;
  const int acts = matrices[3 * problems + problem];
  const int m0 = blockIdx.x * 128, n0 = blockIdx.y * 64;
  if (n0 >= acts) return;
  const unsigned char* codes = reinterpret_cast<const unsigned char*>(matrices[problem]);
  const bf16* scales = reinterpret_cast<const bf16*>(codes + scale_offset);
  const bf16* input = reinterpret_cast<const bf16*>(matrices[problems + problem]);
  bf16* output = reinterpret_cast<bf16*>(matrices[2 * problems + problem]);
  __shared__ __align__(16) bf16 weight_tile[128][LDS];
  __shared__ __align__(16) bf16 input_tile[64][LDS];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = (warp >> 1) * 64, wn = (warp & 1) * 32;
  float accum[4][4][4] = {};
  for (int k0 = 0; k0 < inner; k0 += CK) {
    for (int index = tid; index < 128 * CK; index += 128) {
      const int row = index / CK, col = index % CK;
      const bool live = m0 + row < columns && k0 + col < inner;
      weight_tile[row][col] = live
          ? __float2bfloat16_rn(e4m3_decode(codes[(size_t)(m0 + row) * inner + k0 + col]) *
                               __bfloat162float(scales[m0 + row]))
          : __float2bfloat16_rn(0.0f);
    }
    for (int index = tid; index < 64 * CK; index += 128) {
      const int row = index / CK, col = index % CK;
      input_tile[row][col] = n0 + row < acts && k0 + col < inner
          ? input[(size_t)(n0 + row) * inner + k0 + col]
          : __float2bfloat16_rn(0.0f);
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < CK; kk += 16) {
      unsigned a[4][4], b[4][2];
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const unsigned address = (unsigned)__cvta_generic_to_shared(
            &weight_tile[wm + m * 16 + (lane & 15)][kk + (lane >> 4) * 8]);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(a[m][0]), "=r"(a[m][1]), "=r"(a[m][2]), "=r"(a[m][3])
            : "r"(address));
      }
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        if (n0 + wn + n * 8 >= acts) continue;
        const unsigned address = (unsigned)__cvta_generic_to_shared(
            &input_tile[wn + n * 8 + (lane & 7)][kk + ((lane >> 3) & 1) * 8]);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
            : "=r"(b[n][0]), "=r"(b[n][1]) : "r"(address));
      }
#pragma unroll
      for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          if (n0 + wn + n * 8 >= acts) continue;
          float* d = accum[m][n];
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
              : "r"(a[m][0]), "r"(a[m][1]), "r"(a[m][2]), "r"(a[m][3]),
                "r"(b[n][0]), "r"(b[n][1]));
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int m = 0; m < 4; ++m) {
#pragma unroll
    for (int n = 0; n < 4; ++n) {
#pragma unroll
      for (int d = 0; d < 4; ++d) {
        const int row = m0 + wm + m * 16 + lane / 4 + (d / 2) * 8;
        const int col = n0 + wn + n * 8 + (lane & 3) * 2 + (d & 1);
        if (row < columns && col < acts)
          output[(size_t)col * columns + row] = __float2bfloat16_rn(accum[m][n][d]);
      }
    }
  }
}

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
