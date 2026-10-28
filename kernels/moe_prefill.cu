// Active-assignment count, exclusive scan, row permutation and weighted BF16 scatter.
// Active expert ids, slot generations, done bits and segment positions are checked before use.
#if !defined(__HIP__)
#include <cuda/atomic>
#endif
#include <type_traits>
#include "q38_contract.cuh"

using bf16 = __nv_bfloat16;

// Sixteen K-strided FMA accumulators and the CPU expert's pairwise tree.
template <bool FP8, class Out>
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
  Out* __restrict__ y =
      reinterpret_cast<Out*>(matrices[2 * problems + problem]) + (size_t)row * columns;
  float sum = 0.0f;
  for (int k = lane; k < inner; k += 16) {
    if constexpr (FP8) {
      const unsigned char* codes = reinterpret_cast<const unsigned char*>(matrices[problem]);
      const bf16* scales = reinterpret_cast<const bf16*>(codes + scale_offset);
      const bf16 value = __float2bfloat16_rn(
          e4m3_decode(codes[(size_t)column * inner + k]) * __bfloat162float(scales[column]));
      sum = __fmaf_rn(__bfloat162float(x[k]), __bfloat162float(value), sum);
    } else {
      sum = __fmaf_rn(__bfloat162float(x[k]), __bfloat162float(weight[k]), sum);
    }
  }
  const unsigned mask = 0xFFFFu << (threadIdx.x & 16);
  for (int step = 1; step < 16; step *= 2)
    sum = __fadd_rn(sum, __shfl_down_sync(mask, sum, step, 16));
  if (lane == 0) {
    if constexpr (std::is_same_v<Out, float>) {
      y[column] = sum;
    } else {
      y[column] = __float2bfloat16_rn(sum);
    }
  }
}

extern "C" __global__ void moe_gemm_lane16(
    const unsigned long long* __restrict__ matrices, int problems,
    int first, int columns, int inner) {
  moe_lane16<false, bf16>(matrices, problems, first, columns, inner, 0);
}

extern "C" __global__ void moe_gemm_lane16_w8a16(
    const unsigned long long* __restrict__ matrices, int problems,
    int first, int columns, int inner, unsigned long long scale_offset) {
  moe_lane16<true, bf16>(matrices, problems, first, columns, inner, scale_offset);
}

extern "C" __global__ void moe_gemm_lane16_f32(
    const unsigned long long* matrices, int problems,
    int first, int columns, int inner) {
  moe_lane16<false, float>(matrices, problems, first, columns, inner, 0);
}

__device__ __forceinline__ float decode_e4m3_byte(unsigned code) {
  const unsigned sign = (code & 0x80u) << 24;
  const unsigned mag = code & 0x7fu;
  unsigned bits = (mag << 20) + 0x3c000000u;
  if (mag < 8u) bits = __float_as_uint(__uint_as_float(bits + 0x00800000u) - 0.015625f);
  if (mag == 0x7fu) bits = 0x7fc00000u;
  return __uint_as_float(sign | bits);
}

__device__ __forceinline__ unsigned decode_e4m3x2(unsigned short pair, float scale) {
  const unsigned short lo = __bfloat16_as_ushort(
      __float2bfloat16_rn(decode_e4m3_byte(pair & 0xFFu) * scale));
  const unsigned short hi = __bfloat16_as_ushort(
      __float2bfloat16_rn(decode_e4m3_byte(pair >> 8) * scale));
  return (unsigned)lo | ((unsigned)hi << 16);
}

#if defined(__HIP__)
using MoeBf16 = short __attribute__((ext_vector_type(16)));
using MoeAcc = float __attribute__((ext_vector_type(8)));

// Eight wave32 groups own 16 output columns and four 16-row activation tiles each.
extern "C" __global__ void __launch_bounds__(256) moe_gemm_w8a16(
    const unsigned long long* __restrict__ matrices, int problems,
    int tensor_problems, int columns, int inner, unsigned long long scale_offset) {
  if (blockDim.x != 256 || tensor_problems < 1 || tensor_problems > problems ||
      columns < 1 || inner < 1 || (inner & 63) != 0 || blockIdx.z >= tensor_problems)
    __trap();
  const int problem = blockIdx.z;
  const int rows = matrices[3 * problems + problem];
  const int n0 = blockIdx.x * 128, m0 = blockIdx.y * 64;
  if (m0 >= rows) return;
  const unsigned char* codes = reinterpret_cast<const unsigned char*>(matrices[problem]);
  const bf16* scales = reinterpret_cast<const bf16*>(codes + scale_offset);
  const bf16* x = reinterpret_cast<const bf16*>(matrices[problems + problem]);
  bf16* y = reinterpret_cast<bf16*>(matrices[2 * problems + problem]);
  __shared__ bf16 weight[128][32], input[64][32];
  const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
  MoeAcc acc[4] = {};
  for (int k0 = 0; k0 < inner; k0 += 32) {
    for (int cell = tid; cell < 128 * 32; cell += 256) {
      const int n = n0 + cell / 32, k = k0 + cell % 32;
      const float boost = n < columns && __bfloat162float(scales[n]) < 0x1p-64f
          ? 0x1p64f : 1.0f;
      weight[cell / 32][cell % 32] = n < columns
          ? __float2bfloat16(__bfloat162float(__float2bfloat16_rn(
                e4m3_decode(codes[(size_t)n * inner + k]) * __bfloat162float(scales[n]))) * boost)
          : __float2bfloat16(0.0f);
    }
    for (int cell = tid; cell < 64 * 32; cell += 256) {
      const int m = m0 + cell / 32, k = k0 + cell % 32;
      input[cell / 32][cell % 32] = m < rows ? x[(size_t)m * inner + k]
                                                     : __float2bfloat16(0.0f);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < 2; ++part) {
      MoeBf16 a;
#pragma unroll
      for (int j = 0; j < 16; ++j)
        a[j] = reinterpret_cast<const unsigned short*>(weight[wave * 16 + lane % 16])[part * 16 + j];
#pragma unroll
      for (int tile = 0; tile < 4; ++tile) {
        MoeBf16 b;
#pragma unroll
        for (int j = 0; j < 16; ++j)
          b[j] = reinterpret_cast<const unsigned short*>(input[tile * 16 + lane % 16])[part * 16 + j];
        acc[tile] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[tile]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int tile = 0; tile < 4; ++tile) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int m = m0 + tile * 16 + lane % 16;
      const int n = n0 + wave * 16 + 2 * j + lane / 16;
      if (m < rows && n < columns) {
        const float unscale = __bfloat162float(scales[n]) < 0x1p-64f ? 0x1p-64f : 1.0f;
        y[(size_t)m * columns + n] = __float2bfloat16(acc[tile][j] * unscale);
      }
    }
  }
}
#else
// A 128x64, B 64x64; eight warps own 16x64 outputs, 2-stage cp.async with k16 MMA steps.
extern "C" __global__ void __launch_bounds__(256, 2) moe_gemm_w8a16(
    const unsigned long long* __restrict__ matrices, int problems,
    int tensor_problems, int columns, int inner, unsigned long long scale_offset) {
  if (blockDim.x != 256 || tensor_problems < 1 || tensor_problems > problems ||
      columns < 1 || inner < 1 || (inner & 63) != 0 || blockIdx.z >= tensor_problems)
    __trap();
  constexpr int CK = 64, LDS = 72, CODE_LDS = 80;
  const int problem = blockIdx.z;
  const int acts = matrices[3 * problems + problem];
  const int m0 = blockIdx.x * 128, n0 = blockIdx.y * 64;
  if (n0 >= acts) return;
  const int tile_acts = acts - n0 < 64 ? acts - n0 : 64;
  const unsigned char* codes = reinterpret_cast<const unsigned char*>(matrices[problem]);
  const bf16* scales = reinterpret_cast<const bf16*>(codes + scale_offset);
  const bf16* input = reinterpret_cast<const bf16*>(matrices[problems + problem]);
  bf16* output = reinterpret_cast<bf16*>(matrices[2 * problems + problem]);
  __shared__ __align__(16) unsigned char weight_codes[2][128][CODE_LDS];
  __shared__ __align__(16) bf16 input_tile[2][64][LDS];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp * 16;
  const int r0 = wm + (lane >> 2), r1 = r0 + 8, c_pair = (lane & 3) * 2;
  const float scale0 = m0 + r0 < columns ? __bfloat162float(scales[m0 + r0]) : 0.0f;
  const float scale1 = m0 + r1 < columns ? __bfloat162float(scales[m0 + r1]) : 0.0f;
  const int in_row0 = tid >> 3, in_row1 = in_row0 + 32, in_col = (tid & 7) * 8;
  const bool in_live0 = in_row0 < tile_acts, in_live1 = in_row1 < tile_acts;
  const bf16* in_src0 = in_live0 ? input + (size_t)(n0 + in_row0) * inner + in_col : input;
  const bf16* in_src1 = in_live1 ? input + (size_t)(n0 + in_row1) * inner + in_col : input;
  const int in_bytes0 = in_live0 ? 16 : 0, in_bytes1 = in_live1 ? 16 : 0;
  const int w_row0 = tid >> 2, w_row1 = w_row0 + 64, w_col = (tid & 3) * 16;
  const bool w_live0 = m0 + w_row0 < columns, w_live1 = m0 + w_row1 < columns;
  const unsigned char* w_src0 = w_live0 ? codes + (size_t)(m0 + w_row0) * inner + w_col : codes;
  const unsigned char* w_src1 = w_live1 ? codes + (size_t)(m0 + w_row1) * inner + w_col : codes;
  const int w_bytes0 = w_live0 ? 16 : 0, w_bytes1 = w_live1 ? 16 : 0;
  float accum[8][4] = {};
  auto stage_tile = [&](int stage, int k0) {
    const size_t k_off_in = in_live0 ? (size_t)k0 : 0;
    const size_t k_off_in1 = in_live1 ? (size_t)k0 : 0;
    const unsigned in_dst0 =
        (unsigned)__cvta_generic_to_shared(&input_tile[stage][in_row0][in_col]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
        :: "r"(in_dst0), "l"(in_src0 + k_off_in), "r"(in_bytes0) : "memory");
    if (tile_acts > 32) {
      const unsigned in_dst1 =
          (unsigned)__cvta_generic_to_shared(&input_tile[stage][in_row1][in_col]);
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
          :: "r"(in_dst1), "l"(in_src1 + k_off_in1), "r"(in_bytes1) : "memory");
    }
    const size_t k_off_w0 = w_live0 ? (size_t)k0 : 0;
    const size_t k_off_w1 = w_live1 ? (size_t)k0 : 0;
    const unsigned w_dst0 =
        (unsigned)__cvta_generic_to_shared(&weight_codes[stage][w_row0][w_col]);
    const unsigned w_dst1 =
        (unsigned)__cvta_generic_to_shared(&weight_codes[stage][w_row1][w_col]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
        :: "r"(w_dst0), "l"(w_src0 + k_off_w0), "r"(w_bytes0) : "memory");
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
        :: "r"(w_dst1), "l"(w_src1 + k_off_w1), "r"(w_bytes1) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
  };
  stage_tile(0, 0);
  for (int k0 = 0, stage = 0; k0 < inner; k0 += CK, stage ^= 1) {
    if (k0 + CK < inner) {
      stage_tile(stage ^ 1, k0 + CK);
      asm volatile("cp.async.wait_group 1;" ::: "memory");
    } else {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
    }
    __syncthreads();
#pragma unroll 2
    for (int kk = 0; kk < CK; kk += 16) {
      const int c0 = kk + c_pair;
      const unsigned short p0 =
          *reinterpret_cast<const unsigned short*>(&weight_codes[stage][r0][c0]);
      const unsigned short p1 =
          *reinterpret_cast<const unsigned short*>(&weight_codes[stage][r1][c0]);
      const unsigned short p2 =
          *reinterpret_cast<const unsigned short*>(&weight_codes[stage][r0][c0 + 8]);
      const unsigned short p3 =
          *reinterpret_cast<const unsigned short*>(&weight_codes[stage][r1][c0 + 8]);
      const unsigned a0 = decode_e4m3x2(p0, scale0);
      const unsigned a1 = decode_e4m3x2(p1, scale1);
      const unsigned a2 = decode_e4m3x2(p2, scale0);
      const unsigned a3 = decode_e4m3x2(p3, scale1);
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        unsigned b0, b1;
        const unsigned address = (unsigned)__cvta_generic_to_shared(
            &input_tile[stage][n * 8 + (lane & 7)][kk + ((lane >> 3) & 1) * 8]);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
            : "=r"(b0), "=r"(b1) : "r"(address));
        float* d = accum[n];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
      if (tile_acts > 32) {
#pragma unroll
        for (int n = 4; n < 8; ++n) {
          unsigned b0, b1;
          const unsigned address = (unsigned)__cvta_generic_to_shared(
              &input_tile[stage][n * 8 + (lane & 7)][kk + ((lane >> 3) & 1) * 8]);
          asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
              : "=r"(b0), "=r"(b1) : "r"(address));
          float* d = accum[n];
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
              : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    if (n * 8 >= tile_acts) break;
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      const int row = m0 + wm + (lane >> 2) + (d >> 1) * 8;
      const int col = n0 + n * 8 + (lane & 3) * 2 + (d & 1);
      if (row < columns && col < acts)
        output[(size_t)col * columns + row] = __float2bfloat16_rn(accum[n][d]);
    }
  }
}

#endif

__device__ __forceinline__ bool moe_assignment_active(
    const int* ids, const unsigned* active, const unsigned* valid,
    long long a, int experts, int mask_from_slots) {
  if (!mask_from_slots && active[a] == 0u) return false;
  const int e = ids[a];
  if (e < 0 || e >= experts) __trap();
  return mask_from_slots ? valid[e] != 0u : true;
}

// state[0..1024] accumulates counts; state[1024] retires CTAs; state[1025] reserves compact indices.
extern "C" __global__ void moe_count_scan(
    const int* __restrict__ ids, const unsigned* __restrict__ active,
    const unsigned* __restrict__ valid,
    const unsigned long long* __restrict__ generation,
    const unsigned long long* __restrict__ assignment_gen,
    const unsigned* __restrict__ done, unsigned* __restrict__ state,
    int* __restrict__ compact, int* __restrict__ counts,
    int* __restrict__ active_total, int* __restrict__ offsets,
    int* __restrict__ cursor, long long assignments, int experts,
    int top_k, int capacity, int expected_active, int mask_from_slots) {
  if (blockDim.x != 256 || experts < 1 || experts > 1024 ||
      top_k < 1 || top_k > 32 || capacity < 0 ||
      (unsigned)mask_from_slots > 1u) __trap();
  __shared__ int sc[1024];
  __shared__ unsigned last;
  const int tid = threadIdx.x;
  for (int e = tid; e < 1024; e += 256) sc[e] = 0;
  __syncthreads();
  for (long long base = (long long)blockIdx.x * 2048; base < assignments;
       base += (long long)gridDim.x * 2048) {
    for (int offset = tid; offset < 2048; offset += 256) {
      const long long a = base + offset;
      const bool live = a < assignments &&
          moe_assignment_active(ids, active, valid, a, experts, mask_from_slots);
      if (live) {
        const int e = ids[a];
        if (valid[e] == 0u || done[a] != 0u ||
            (mask_from_slots && (valid[e] > 1u || generation[e] == 0u)) ||
            (!mask_from_slots && generation[e] != assignment_gen[a])) __trap();
        const long long row = a / top_k;
        for (int c = 0; c < a % top_k; ++c) {
          const long long other = row * top_k + c;
          if (moe_assignment_active(ids, active, valid, other, experts, mask_from_slots) &&
              ids[other] == e) __trap();
        }
        atomicAdd(&sc[e], 1);
      }
      const unsigned ballot = __ballot_sync(FF_WARP_MASK, live);
      if (ballot != 0u) {
        const int lane = tid & 31;
        const int leader = __ffs(ballot) - 1;
        unsigned start = 0;
        if (lane == leader) start = atomicAdd(state + 1025, __popc(ballot));
        start = __shfl_sync(FF_WARP_MASK, start, leader);
        if (live) {
          const unsigned index = start + __popc(ballot & ((1u << lane) - 1u));
          if (index >= (unsigned)capacity) __trap();
          compact[index] = (int)a;
        }
      }
    }
  }
  __syncthreads();
  for (int e = tid; e < experts; e += 256) {
    if (sc[e] != 0) atomicAdd(state + e, sc[e]);
  }
  __syncthreads();
  // Every CTA publishes before its release RMW; the last RMW acquires the retirement chain.
  if (tid == 0) {
#if defined(__HIP__)
    last = __hip_atomic_fetch_add(state + 1024, 1u, __ATOMIC_ACQ_REL,
                                  __HIP_MEMORY_SCOPE_AGENT) + 1u == gridDim.x;
#else
    cuda::atomic_ref<unsigned, cuda::thread_scope_device> retired(state[1024]);
    last = retired.fetch_add(1u, cuda::memory_order_acq_rel) + 1u == gridDim.x;
#endif
  }
  __syncthreads();
  if (!last) return;
  for (int e = tid; e < experts; e += 256) {
    sc[e] = state[e];
    counts[e] = sc[e];
    state[e] = 0u;
  }
  __syncthreads();
  for (int d = 1; d < experts; d <<= 1) {
    int values[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int e = tid + j * 256;
      values[j] = e < experts && e >= d ? sc[e - d] : 0;
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int e = tid + j * 256;
      if (e < experts) sc[e] += values[j];
    }
    __syncthreads();
  }
  for (int e = tid; e < experts; e += 256) {
    offsets[e] = e == 0 ? 0 : sc[e - 1];
    cursor[e] = offsets[e];
  }
  if (tid == 0) {
    const int total = sc[experts - 1];
    if ((unsigned)total != state[1025] || total > capacity ||
        (expected_active >= 0 && total != expected_active)) __trap();
    *active_total = total;
    offsets[experts] = total;
    state[1025] = 0u;
#if defined(__HIP__)
    __hip_atomic_store(state + 1024, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
    cuda::atomic_ref<unsigned, cuda::thread_scope_device> retired(state[1024]);
    retired.store(0u, cuda::memory_order_relaxed);
#endif
  }
}

// One CTA copies one compact assignment into its expert segment.
extern "C" __global__ void moe_permute(
    const int* __restrict__ ids, const bf16* __restrict__ x,
    const int* __restrict__ offsets,
    int* __restrict__ cursor, const unsigned* __restrict__ valid,
    const unsigned long long* __restrict__ generation,
    const unsigned long long* __restrict__ assignment_gen,
    const int* __restrict__ compact, const int* __restrict__ active_total,
    unsigned* __restrict__ done, int* __restrict__ pos_of,
    bf16* __restrict__ x_perm, int rows, int top_k,
    int experts, long long hidden, int mask_from_slots) {
  if (blockDim.x != 256) __trap();
  if (blockIdx.x >= (unsigned)*active_total) return;
  const int a = compact[blockIdx.x];
  if (a < 0 || (long long)a >= (long long)rows * top_k) __trap();
  const int row = a / top_k;
  const int e = ids[a];
  if (e < 0 || e >= experts || valid[e] == 0u ||
      (!mask_from_slots && generation[e] != assignment_gen[a])) __trap();
  __shared__ int spos;
  if (threadIdx.x == 0) {
    if (atomicCAS(done + a, 0u, 1u) != 0u) __trap();
    const int pos = atomicAdd(cursor + e, 1);
    if (pos < offsets[e] || pos >= offsets[e + 1]) __trap();
    pos_of[a] = pos;
    spos = pos;
  }
  __syncthreads();
  const bf16* __restrict__ src = x + (long long)row * hidden;
  bf16* __restrict__ dst = x_perm + (long long)spos * hidden;
  if (hidden % 8 == 0) {
    for (long long c = threadIdx.x; c < hidden / 8; c += 256)
      reinterpret_cast<float4*>(dst)[c] = reinterpret_cast<const float4*>(src)[c];
  } else {
    for (long long c = threadIdx.x; c < hidden; c += 256) dst[c] = src[c];
  }
}

template <class T>
__device__ __forceinline__ void moe_scatter_rows(
    const T* __restrict__ y_perm, const bf16* __restrict__ weights,
    const int* __restrict__ pos_of, const int* __restrict__ compact,
    bf16* __restrict__ contributions, long long assignments,
    long long hidden, int active_total, int routed_weight_fp32) {
  if (blockDim.x != 256 || (unsigned)routed_weight_fp32 > 1u ||
      blockIdx.x >= (unsigned)active_total) __trap();
  const int a = compact[blockIdx.x];
  if (a < 0 || a >= assignments) __trap();
  const int pos = pos_of[a];
  if (pos < 0 || pos >= active_total) __trap();
  const float w = __bfloat162float(weights[a]);
  const T* __restrict__ src = y_perm + (long long)pos * hidden;
  bf16* __restrict__ dst = contributions + (long long)a * hidden;
  if (hidden % 8 == 0) {
    for (long long c = (long long)threadIdx.x * 8; c < hidden; c += 256 * 8) {
      alignas(16) T values[8];
      alignas(16) bf16 output[8];
#pragma unroll
      for (int v = 0; v < (int)(sizeof(T) * 8 / 16); ++v)
        reinterpret_cast<float4*>(values)[v] = reinterpret_cast<const float4*>(src + c)[v];
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        float value;
        if constexpr (std::is_same_v<T, bf16>) value = __bfloat162float(values[v]);
        else value = routed_weight_fp32 ? values[v] : __bfloat162float(__float2bfloat16(values[v]));
        output[v] = __float2bfloat16(value * w);
      }
      *reinterpret_cast<float4*>(dst + c) = *reinterpret_cast<float4*>(output);
    }
  } else {
    for (long long c = threadIdx.x; c < hidden; c += 256) {
      float value;
      if constexpr (std::is_same_v<T, bf16>) value = __bfloat162float(src[c]);
      else value = routed_weight_fp32 ? src[c] : __bfloat162float(__float2bfloat16(src[c]));
      dst[c] = __float2bfloat16(value * w);
    }
  }
}

extern "C" __global__ void moe_scatter(
    const bf16* __restrict__ y_perm, const bf16* __restrict__ weights,
    const int* __restrict__ pos_of, const int* __restrict__ compact,
    bf16* __restrict__ contributions, long long assignments,
    long long hidden, int active_total) {
  moe_scatter_rows(y_perm, weights, pos_of, compact, contributions,
                   assignments, hidden, active_total, 0);
}

extern "C" __global__ void moe_scatter_f32(
    const float* __restrict__ y_perm, const bf16* __restrict__ weights,
    const int* __restrict__ pos_of, const int* __restrict__ compact,
    bf16* __restrict__ contributions, long long assignments,
    long long hidden, int active_total, int routed_weight_fp32) {
  moe_scatter_rows(y_perm, weights, pos_of, compact, contributions,
                   assignments, hidden, active_total, routed_weight_fp32);
}
