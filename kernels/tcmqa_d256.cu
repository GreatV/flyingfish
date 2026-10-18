// HD256 full attention.
// Shape: (q_heads, kv_heads) = (24, 4) or (16, 2), head_dim=256, scale=1/16.
// Tile: MT=64 pair rows, KT=32 KV positions, 4 warps / 128 threads, dynamic smem SMEM_NARROW.
// Array extents and unrolled loop bounds are written in DH/HQ/HKV through
// NSLICE = DH/8 and KSLICE = DH/16; no dimension is a literal.
// Partials (fixed-Pmax SoA):
//   Pmax = ceil(capacity/key_chunk)   m_cap = min(max_rows, 64) = seg_rows
//   E = seg_rows * HQ * Pmax
//   e = ((kv_head*Pmax + part)*(GQA*seg_rows) + q_in_group*seg_rows + local_row)
//   p_max:  f32[E]      offset 0
//   p_sum:  f32[E]      offset align256(E*4)
//   p_out:  f32[E*DH]   offset p_sum + align256(E*4)
// Phase 1 writes only valid rows/parts; phase 2 reads only those.
// Query segmentation: segments of at most 64 rows. Segment local row 0 is global row
// row_base, absolute position L = prefix + row_base, m = min(seg_rows, valid_rows - row_base)
// live rows. Row r sees every key below L, and key L+jj iff bit jj of causal_mask[r] is
// set. causal_mask is u64[64]; mask[63] is UINT64_MAX.
// Kernels read prefix/valid_rows from the device SequenceMeta. Grids are sized from
// max_rows; segments with no live rows exit without writing.
#include <cuda_bf16.h>
#include <cstdint>
#include "kv_cache.cuh"

namespace tcmqa_d256 {

using bf16 = __nv_bfloat16;

#ifndef ATTENTION_Q_HEADS
#define ATTENTION_Q_HEADS 24
#endif
#ifndef ATTENTION_KV_HEADS
#define ATTENTION_KV_HEADS 4
#endif
constexpr int HQ      = ATTENTION_Q_HEADS;
constexpr int HKV     = ATTENTION_KV_HEADS;
constexpr int GQA     = HQ / HKV;
constexpr int DH      = 256;     // head_dim
constexpr int MT      = 64;      // pair rows per block (narrow tile)
constexpr int KT      = 32;      // KV positions per smem tile
constexpr int NW      = 4;       // warps per block
constexpr int NT      = NW * 32;
constexpr int M_CAP   = 64;      // max query rows per segment
constexpr int PAD     = 8;       // smem row padding vs ldmatrix banks
constexpr int LDS     = DH + PAD;
constexpr float SCALE = 0.0625f; // 1/sqrt(256)

// n8 slices and k16 slices over DH.
constexpr int NSLICE = DH / 8;    // 32
constexpr int KSLICE = DH / 16;   // 16

static_assert(DH % 16 == 0, "DH must be a multiple of 16 (k16 MMA slices)");
static_assert(DH % 8  == 0, "DH must be a multiple of 8 (n8 MMA slices)");
static_assert(LDS % 8 == 0, "LDS must stay a multiple of 8 for 16B row alignment");
static_assert(MT * NSLICE % NT == 0, "Q tile staging must divide the block");
static_assert(KT * NSLICE % NT == 0, "KV tile staging must divide the block");

// Narrow-tile dynamic shared memory: Q [MT][LDS], K [KT][LDS], V [KT][LDS],
// causal mask [64] u64. The host opts this in before the first launch.
constexpr int SMEM_NARROW = (MT + 2 * KT) * LDS * 2 + M_CAP * 8;
static_assert(SMEM_NARROW == 68096, "narrow-tile shared memory budget");

// Partial RoPE: only the first ROTARY dims rotate, in (j, j+ROTARY/2) pairs.
#ifndef ATTENTION_ROTARY_DIM
#define ATTENTION_ROTARY_DIM 64
#endif
constexpr int ROTARY = ATTENTION_ROTARY_DIM;
constexpr int RHALF  = ROTARY / 2;
constexpr float THETA = 1.0e7f;

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return (unsigned)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2(unsigned (&r)[2], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2_t(unsigned (&r)[2], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_16816(float (&d)[4], const unsigned (&a)[4],
                                          const unsigned (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// Pack two fp32 into one bf16x2 register (round-to-nearest).
__device__ __forceinline__ unsigned pack_bf16x2(float lo, float hi) {
  const __nv_bfloat162 t = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const unsigned*>(&t);
}
// cp.async 16B copy; sz == 0 zero-fills the destination (out-of-extent rows).
__device__ __forceinline__ void cp16(void* dst, const void* src, int sz) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               ::"r"(smem_u32(dst)), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cp_wait0() {
  asm volatile("cp.async.wait_group 0;\n");
}

// Device-side invariants only. prepare() does the host-side shape/stride work.
__device__ __forceinline__ void guard_meta(const int4* meta) {
  const int prefix     = meta[0].x;
  const int valid_rows = meta[0].y;
  const int key_count  = meta[0].z;
  const int capacity   = meta[0].w;
  const int causal     = meta[1].x;
  if (causal != 1) __trap();
  if (prefix < 0 || valid_rows <= 0 || key_count != prefix + valid_rows) __trap();
  if (key_count > capacity) __trap();
}
// The plan's Pmax must agree with the capacity the device reports.
__device__ __forceinline__ void guard_pmax(const int4* meta, int key_chunk, int Pmax) {
  const int capacity = meta[0].w;
  const int want = (int)(((long)capacity + key_chunk - 1) / key_chunk);
  if (Pmax != want) __trap();
}
// A wrong-shape launch traps.
__device__ __forceinline__ void guard_geometry(int seg_rows, int key_chunk,
                                               int block_threads, int p1) {
  if (seg_rows < 1 || seg_rows > M_CAP) __trap();
  if (key_chunk < KT || (key_chunk % KT) != 0) __trap();
  if (block_threads != (p1 ? NT : DH)) __trap();
  if (gridDim.y != (p1 ? HKV : 1) || gridDim.z != (p1 ? (GQA * seg_rows + MT - 1) / MT : 1))
    __trap();
}

// Split the [M,HQ,512] q_gate projection into Q and gate views: per head, Q at offset 0
// and gate at offset 256 within the head's 512-wide span.
// One thread per (row, head, dim) writes every element of both views; rows at or beyond
// the device valid_rows return.
extern "C" __global__ void tcmqa_d256_qg_split(const __nv_bfloat16* __restrict__ q_gate,
                                               __nv_bfloat16* __restrict__ q,
                                               __nv_bfloat16* __restrict__ gate,
                                               const int4* __restrict__ meta,
                                               int qg_tok_stride, int q_tok_stride,
                                               int g_tok_stride) {
  guard_meta(meta);
  const int rows = meta[0].y;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * HQ * DH) return;
  const int d   = i % DH;
  const int h   = (i / DH) % HQ;
  const int row = i / (DH * HQ);
  const int head_width = gate ? 2 * DH : DH;
  const __nv_bfloat16* src = q_gate + (size_t)row * qg_tok_stride + (size_t)h * head_width + d;
  q[(size_t)row * q_tok_stride + (size_t)h * DH + d] = src[0];
  if (gate) gate[(size_t)row * g_tok_stride + (size_t)h * DH + d] = src[DH];
}

// Rotation of one (j, j+RHALF) pair at the given fp32 angle; both halves are read before
// either is written.
__device__ __forceinline__ void rope_apply(bf16& a, bf16& b, float ang) {
  const float c = cosf(ang), s = sinf(ang);
  const float af = __bfloat162float(a), bf = __bfloat162float(b);
  a = __float2bfloat16(af * c - bf * s);
  b = __float2bfloat16(bf * c + af * s);
}

// Angle of pair half_idx at coordinate pos: pos * theta^(-2*half_idx/ROTARY), fp32.
__device__ __forceinline__ float rope_angle(int pos, int half_idx) {
  const float inv = 1.0f / powf(THETA, float(2 * half_idx) / float(ROTARY));
  return float(pos) * inv;
}

// Interleaved mRoPE (model definition apply_interleaved_mrope, mrope_section (11, 11, 10)):
// of the RHALF = 32 pairs, H owns slice(1, 33, 3), W owns slice(2, 30, 3), T the rest.
__host__ __device__ constexpr int rope_axis(int j) {
  return (j % 3 == 1 && j < 33) ? 1 : ((j % 3 == 2 && j < 30) ? 2 : 0);
}
constexpr int rope_axis_count(int axis) {
  int n = 0;
  for (int j = 0; j < RHALF; ++j) n += (rope_axis(j) == axis);
  return n;
}
static_assert(RHALF != 32 ||
                  (rope_axis_count(0) == 11 && rope_axis_count(1) == 11 &&
                   rope_axis_count(2) == 10),
              "mrope_section (11, 11, 10) over RHALF = 32");

// In-place partial RoPE on Q ([M,HQ,256] BF16 view) at absolute position prefix + row.
// Writes only dims [0, ROTARY); qg_split writes the full view first.
extern "C" __global__ void tcmqa_d256_rope_q(__nv_bfloat16* __restrict__ q,
                                             const int4* __restrict__ meta,
                                             int q_tok_stride) {
  guard_meta(meta);
  const int rows = meta[0].y;
  const int prefix = meta[0].x;
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= rows * HQ * RHALF) return;
  const int j = pair % RHALF;
  const int h = (pair / RHALF) % HQ;
  const int r = pair / (RHALF * HQ);
  bf16* row = q + (size_t)r * q_tok_stride + (size_t)h * DH;
  rope_apply(row[j], row[j + RHALF], rope_angle(prefix + r, j));
}

// Partial RoPE on normalised K, then scatter the head row into the head-major K cache.
// One thread per cache dim: dims < RHALF write the rotated (j, j+RHALF) pair; dims >= ROTARY
// copy through.
extern "C" __global__ void tcmqa_d256_rope_k_scatter(const __nv_bfloat16* __restrict__ k,
                                                     __nv_bfloat16* __restrict__ k_cache,
                                                     const int4* __restrict__ meta,
                                                     int k_tok_stride,
                                                     int k_head_stride, int k_pos_stride) {
  guard_meta(meta);
  const int rows = meta[0].y;
  const int prefix = meta[0].x;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * HKV * DH) return;
  const int d = i % DH;
  const int h = (i / DH) % HKV;
  const int r = i / (DH * HKV);
  const int pos = prefix + r;
  const bf16* src = k + (size_t)r * k_tok_stride + (size_t)h * DH;
  bf16* dst = k_cache + (size_t)h * k_head_stride + (size_t)pos * k_pos_stride;
  if (d < RHALF) {
    bf16 a = src[d], b = src[d + RHALF];
    rope_apply(a, b, rope_angle(pos, d));
    dst[d] = a;
    dst[d + RHALF] = b;
  } else if (d >= ROTARY) {
    dst[d] = src[d];
  }
}

// Three-axis tcmqa_d256_rope_q: each pair's angle uses the row's (T, H, W) coordinate from
// axes ([rows, 3] i32, row-major); rows still come from the meta.
extern "C" __global__ void tcmqa_d256_rope_q_axes(__nv_bfloat16* __restrict__ q,
                                                  const int4* __restrict__ meta,
                                                  const int* __restrict__ axes,
                                                  int q_tok_stride) {
  guard_meta(meta);
  const int rows = meta[0].y;
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= rows * HQ * RHALF) return;
  const int j = pair % RHALF;
  const int h = (pair / RHALF) % HQ;
  const int r = pair / (RHALF * HQ);
  bf16* row = q + (size_t)r * q_tok_stride + (size_t)h * DH;
  rope_apply(row[j], row[j + RHALF],
             rope_angle(axes[(size_t)r * 3 + rope_axis(j)], j));
}

// Three-axis tcmqa_d256_rope_k_scatter: the rotation coordinate comes from axes; the
// destination slot is still prefix + r.
extern "C" __global__ void tcmqa_d256_rope_k_scatter_axes(
    const __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ k_cache,
    const int4* __restrict__ meta, const int* __restrict__ axes, int k_tok_stride,
    int k_head_stride, int k_pos_stride) {
  guard_meta(meta);
  const int rows = meta[0].y;
  const int prefix = meta[0].x;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * HKV * DH) return;
  const int d = i % DH;
  const int h = (i / DH) % HKV;
  const int r = i / (DH * HKV);
  const int pos = prefix + r;
  const bf16* src = k + (size_t)r * k_tok_stride + (size_t)h * DH;
  bf16* dst = k_cache + (size_t)h * k_head_stride + (size_t)pos * k_pos_stride;
  if (d < RHALF) {
    bf16 a = src[d], b = src[d + RHALF];
    rope_apply(a, b, rope_angle(axes[(size_t)r * 3 + rope_axis(d)], d));
    dst[d] = a;
    dst[d + RHALF] = b;
  } else if (d >= ROTARY) {
    dst[d] = src[d];
  }
}

// V has no RoPE; copy straight into the cache.
extern "C" __global__ void tcmqa_d256_v_scatter(const __nv_bfloat16* __restrict__ v,
                                                __nv_bfloat16* __restrict__ v_cache,
                                                const int4* __restrict__ meta,
                                                int v_tok_stride,
                                                int v_head_stride, int v_pos_stride) {
  guard_meta(meta);
  const int rows = meta[0].y;
  const int prefix = meta[0].x;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * HKV * DH) return;
  const int d = i % DH;
  const int h = (i / DH) % HKV;
  const int r = i / (DH * HKV);
  const int pos = prefix + r;
  v_cache[(size_t)h * v_head_stride + (size_t)pos * v_pos_stride + d] =
      v[(size_t)r * v_tok_stride + (size_t)h * DH + d];
}

// Phase 1: block (part, kv, rt) sweeps key chunk part of kv-head kv in KT-position tiles and
// reduces each of its MT pair rows to one (max, sum, acc[DH]) partial.
// Pair p = q_in_group*seg_rows + local_row; the tile covers pairs [rt*MT, rt*MT+MT).
// Warp w owns MMA rows [w*16, w*16+16).
// Launch: grid(Pmax, HKV, ceil(GQA*seg_rows/MT)), block NT,
//         dynamic smem SMEM_NARROW (> 48 KiB, host opts in).
template<bool COMPACT = false>
__device__ __forceinline__ void phase1_body(
    const bf16* __restrict__ q, const bf16* __restrict__ kc, const bf16* __restrict__ vc,
    float* __restrict__ p_max, float* __restrict__ p_sum, float* __restrict__ p_out,
    const unsigned long long* __restrict__ causal_mask, const int4* __restrict__ meta,
    int q_tok_stride, int k_head_stride, int k_pos_stride,
    int v_head_stride, int v_pos_stride,
    int key_chunk, int Pmax, int seg_rows, int row_base, KvCacheView cache = {}) {
  const int part = blockIdx.x;
  const int kv   = blockIdx.y;
  const int rt   = blockIdx.z;

  const int prefix = meta[0].x;
  const int rows   = meta[0].y;

  // Segment geometry. L is the absolute position of this segment's local row 0;
  // only m of the seg_rows rows are live in this call.
  const int m = min(seg_rows, rows - row_base);
  const int L = prefix + row_base;
  const int Pactive = (L + m + key_chunk - 1) / key_chunk;
  // Both exits are block-uniform and precede every barrier.
  if (m <= 0) return;
  if (part >= Pactive) return;

  const int pos_beg = part * key_chunk;
  const int pos_end = min(pos_beg + key_chunk, L + m);

  const int tid  = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int nrows = GQA * seg_rows;   // pair rows in this segment
  const int row0  = rt * MT;

  extern __shared__ __align__(16) bf16 tcm_sm[];
  bf16* qsm = tcm_sm;
  bf16* ksm = qsm + MT * LDS;
  bf16* vsm = ksm + KT * LDS;
  unsigned long long* msm = reinterpret_cast<unsigned long long*>(vsm + KT * LDS);

  // Stage the causal mask and the Q row tile (16B copies).
  for (int i = tid; i < M_CAP; i += NT) msm[i] = causal_mask[i];
  for (int idx = tid; idx < MT * NSLICE; idx += NT) {
    const int r = idx / NSLICE, seg = idx % NSLICE;
    const int p = row0 + r;
    uint4 val = make_uint4(0u, 0u, 0u, 0u);
    if (p < nrows) {
      const int lr = p % seg_rows;
      const int qg = p / seg_rows;
      if (lr < m)
        val = *reinterpret_cast<const uint4*>(
            q + (size_t)lr * q_tok_stride + (size_t)(kv * GQA + qg) * DH + seg * 8);
    }
    *reinterpret_cast<uint4*>(qsm + r * LDS + seg * 8) = val;
  }
  __syncthreads();

  // Per-lane rows (D-fragment layout): rows lane/4 and lane/4+8 of the warp's 16-row slab.
  const int rl = warp * 16 + (lane >> 2);
  const int rh = rl + 8;
  const int pl = row0 + rl;
  const int ph = row0 + rh;
  const int lrl = pl % seg_rows;   // local row, in [0, seg_rows)
  const int lrh = ph % seg_rows;

  float acc[NSLICE][4];
  float mx[2] = {-3.0e38f, -3.0e38f};   // finite sentinel: never NaN
  float ls[2] = {0.0f, 0.0f};
#pragma unroll
  for (int n = 0; n < NSLICE; ++n)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[n][j] = 0.0f;

  // K/V tiles are staged with cp.async. The prologue stages K(0); per tile, CP1
  // waits for K(t), barriers, and stages V(t) during the S mma and softmax; CP2
  // waits for V(t), barriers, and stages K(t+1) during the PV mma. Each stage
  // overwrites the tile consumed before the preceding barrier.
  const int ntiles = (pos_end - pos_beg + KT - 1) / KT;

  auto stage_kv = [&](const bf16* src, int head_stride, int pos_stride, int t, bf16* dst, bool is_k) {
    const int tt0 = pos_beg + t * KT;
#pragma unroll
    for (int rep = 0; rep < KT * NSLICE / NT; ++rep) {
      const int idx = tid + rep * NT;
      const int r = idx / NSLICE, seg = idx % NSLICE;
      const int pos = tt0 + r;
      const bool live = pos < pos_end;
      if constexpr (COMPACT) {
        for(int j=0;j<8;++j) dst[r*LDS+seg*8+j]=live ? kv_load(cache,is_k,kv,pos,seg*8+j) : __float2bfloat16_rn(0.0f);
      } else {
        const bf16* g = live
            ? src + (size_t)kv * head_stride + (size_t)pos * pos_stride + seg * 8
            : src;
        cp16(dst + r * LDS + seg * 8, g, live ? 16 : 0);
      }
    }
  };

  if (ntiles > 0) {
    stage_kv(kc, k_head_stride, k_pos_stride, 0, ksm, true);
    cp_commit();
  }

  for (int t = 0; t < ntiles; ++t) {
    const int t0 = pos_beg + t * KT;
    cp_wait0();                                       // K(t) landed
    __syncthreads();                                  // K(t) visible; V(t-1) consumed
    stage_kv(vc, v_head_stride, v_pos_stride, t, vsm, false);
    cp_commit();                                      // V(t) flies during S + softmax

    // S = Q_w * K_tile^T: KSLICE k16 x 4 n8 mma, per-warp row slab.
    float sf[4][4];
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int j = 0; j < 4; ++j) sf[n][j] = 0.0f;
#pragma unroll
    for (int kk = 0; kk < KSLICE; ++kk) {
      unsigned qa[4];
      // A 16x16 at Q rows [warp*16, +16) x dims [kk*16, +16):
      // lane -> row (lane & 15), col segment (lane >> 4).
      {
        const int rq = warp * 16 + (lane & 15);
        ldsm_x4(qa, qsm + rq * LDS + (2 * kk + (lane >> 4)) * 8);
      }
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        unsigned kb[2];
        // B k16(d) x n8(pos), plain ldmatrix on K[pos][d]:
        // mat0 lanes 0-7: pos n*8+lane%8, d kk*16; mat1 lanes 8-15: d kk*16+8.
        {
          const int rk = n * 8 + (lane & 7);
          ldsm_x2(kb, ksm + rk * LDS + (2 * kk + ((lane >> 3) & 1)) * 8);
        }
        mma_16816(sf[n], qa, kb);
      }
    }

#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int j = 0; j < 4; ++j) sf[n][j] *= SCALE;

    // Visibility: pos < L is unconditional; above L the bit comes from causal_mask[row].
    if (t0 + KT > L) {
      const unsigned long long al = msm[lrl];
      const unsigned long long ah = msm[lrh];
#pragma unroll
      for (int n = 0; n < 4; ++n) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int pos = t0 + n * 8 + (lane & 3) * 2 + (j & 1);
          const int jj = pos - L;
          const bool vis = (jj < 0) || (jj < seg_rows && ((j < 2 ? al : ah) >> jj) & 1ull);
          if (!vis) sf[n][j] = -INFINITY;
        }
      }
    }

    // Online softmax per row; quad reductions.
    float pn[4][4];                                   // P fp32 before bf16 pack
    float rmax[2] = {-3.0e38f, -3.0e38f};
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      rmax[0] = fmaxf(rmax[0], fmaxf(sf[n][0], sf[n][1]));
      rmax[1] = fmaxf(rmax[1], fmaxf(sf[n][2], sf[n][3]));
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      rmax[half] = fmaxf(rmax[half], __shfl_xor_sync(0xffffffffu, rmax[half], 1));
      rmax[half] = fmaxf(rmax[half], __shfl_xor_sync(0xffffffffu, rmax[half], 2));
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const float m_new = fmaxf(mx[half], rmax[half]);
      const float cf = __expf(mx[half] - m_new);
      mx[half] = m_new;
      float rsum = 0.0f;
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        const float p0 = __expf(sf[n][half * 2 + 0] - m_new);
        const float p1 = __expf(sf[n][half * 2 + 1] - m_new);
        pn[n][half * 2 + 0] = p0;
        pn[n][half * 2 + 1] = p1;
        rsum += p0 + p1;
      }
      rsum += __shfl_xor_sync(0xffffffffu, rsum, 1);
      rsum += __shfl_xor_sync(0xffffffffu, rsum, 2);
      ls[half] = ls[half] * cf + rsum;
      // rescale the accumulators of this row half
#pragma unroll
      for (int n = 0; n < NSLICE; ++n) {
        acc[n][half * 2 + 0] *= cf;
        acc[n][half * 2 + 1] *= cf;
      }
    }

    // CP2: V(t) landed; stage K(t+1) into the consumed K tile.
    cp_wait0();                                       // V(t) landed
    __syncthreads();                                  // V(t) visible; K(t) consumed
    if (t + 1 < ntiles) {
      stage_kv(kc, k_head_stride, k_pos_stride, t + 1, ksm, true);
      cp_commit();                                    // K(t+1) flies during PV
    }

    // O += P * V: P bf16 repack, then 2 k16 x NSLICE n8 mma.
    unsigned pa[2][4];
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {                  // k16 = pos [kk*16, +16)
      pa[kk][0] = pack_bf16x2(pn[kk * 2 + 0][0], pn[kk * 2 + 0][1]);
      pa[kk][1] = pack_bf16x2(pn[kk * 2 + 0][2], pn[kk * 2 + 0][3]);
      pa[kk][2] = pack_bf16x2(pn[kk * 2 + 1][0], pn[kk * 2 + 1][1]);
      pa[kk][3] = pack_bf16x2(pn[kk * 2 + 1][2], pn[kk * 2 + 1][3]);
    }
#pragma unroll
    for (int nd = 0; nd < NSLICE; ++nd) {             // dim n8 slices
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        unsigned vb[2];
        // B k16(pos) x n8(dim), transposed ldmatrix on V[pos][dim]
        // mat0 lanes 0-7: pos kk*16+lane%8, dim nd*8;
        // mat1 lanes 8-15: pos kk*16+8+lane%8.
        ldsm_x2_t(vb, vsm + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LDS + nd * 8);
        mma_16816(acc[nd], pa[kk], vb);
      }
    }
  }

  // One partial per owned pair; dead rows never write.
  const size_t part0 = (size_t)(kv * Pmax + part) * (size_t)nrows;
  const bool live_l = pl < nrows && lrl < m;
  const bool live_h = ph < nrows && lrh < m;
  const size_t el = (size_t)(pl / seg_rows) * seg_rows + lrl;
  const size_t eh = (size_t)(ph / seg_rows) * seg_rows + lrh;
  if ((lane & 3) == 0) {
    if (live_l) { p_max[part0 + el] = mx[0]; p_sum[part0 + el] = ls[0]; }
    if (live_h) { p_max[part0 + eh] = mx[1]; p_sum[part0 + eh] = ls[1]; }
  }
#pragma unroll
  for (int nd = 0; nd < NSLICE; ++nd) {
    const int c0 = nd * 8 + (lane & 3) * 2;
    if (live_l)
      *reinterpret_cast<float2*>(p_out + (part0 + el) * DH + c0) =
          make_float2(acc[nd][0], acc[nd][1]);
    if (live_h)
      *reinterpret_cast<float2*>(p_out + (part0 + eh) * DH + c0) =
          make_float2(acc[nd][2], acc[nd][3]);
  }
}

extern "C" __global__ void __launch_bounds__(NT) tcmqa_d256_phase1(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    float* __restrict__ p_max, float* __restrict__ p_sum, float* __restrict__ p_out,
    const unsigned long long* __restrict__ causal_mask,
    const int4* __restrict__ meta,
    int q_tok_stride, int k_head_stride, int k_pos_stride,
    int v_head_stride, int v_pos_stride,
    int key_chunk, int Pmax, int seg_rows, int row_base) {
  guard_meta(meta);
  guard_pmax(meta, key_chunk, Pmax);
  guard_geometry(seg_rows, key_chunk, blockDim.x, 1);
  if (causal_mask == nullptr) __trap();
  phase1_body<>(q, k_cache, v_cache, p_max, p_sum, p_out, causal_mask, meta,
              q_tok_stride, k_head_stride, k_pos_stride, v_head_stride, v_pos_stride,
              key_chunk, Pmax, seg_rows, row_base);
}

extern "C" __global__ void __launch_bounds__(NT) tcmqa_d256_phase1_compact(
    const bf16* q, const bf16* kc, const bf16* vc, float* p_max, float* p_sum,
    float* p_out, const unsigned long long* mask, const int4* meta,
    int q_stride,int kh,int kp,int vh,int vp,int chunk,int pmax,int rows,int base,
    KvCacheView cache) {
  guard_meta(meta);guard_pmax(meta,chunk,pmax);guard_geometry(rows,chunk,blockDim.x,1);
  if(!mask || !cache.cursor || cache.capacity!=meta[0].w) __trap();
  phase1_body<true>(q,kc,vc,p_max,p_sum,p_out,mask,meta,q_stride,kh,kp,vh,vp,chunk,pmax,rows,base,cache);
}

extern "C" __global__ void tcmqa_d256_write_ring(
    const bf16* k,const bf16* v,const int4* meta,const int* axes,
    int k_stride,int v_stride,KvCacheView cache) {
  guard_meta(meta);
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=meta[0].y*HKV*DH) return;
  const int d=i%DH,h=(i/DH)%HKV,r=i/(HKV*DH),pos=meta[0].x+r;
  const size_t dst=((size_t)h*cache.ring+kv_slot(pos,cache.ring))*DH;
  const bf16* src=k+(size_t)r*k_stride+h*DH;
  if(!isfinite(__bfloat162float(src[d])) || !isfinite(__bfloat162float(v[(size_t)r*v_stride+h*DH+d]))) __trap();
  if(d<RHALF) {
    bf16 a=src[d],b=src[d+RHALF];
    const int coordinate=axes ? axes[(size_t)r*3+rope_axis(d)] : pos;
    rope_apply(a,b,rope_angle(coordinate,d));
    cache.k_ring[dst+d]=a;cache.k_ring[dst+d+RHALF]=b;
  } else if(d>=ROTARY) cache.k_ring[dst+d]=src[d];
  cache.v_ring[dst+d]=v[(size_t)r*v_stride+h*DH+d];
}

// Phase 2: LSE-rescale the Pactive partials of one (local row, q head) into the ungated
// BF16 core output; the sigmoid gate is applied by the caller.
// Launch: grid(HQ*seg_rows), block DH (one thread per output dim),
//         dynamic smem 2*Pactive floats + DH reduction slots.
__device__ __forceinline__ void phase2_body(
    const float* __restrict__ p_max, const float* __restrict__ p_sum,
    const float* __restrict__ p_out, bf16* __restrict__ core_out,
    const int4* __restrict__ meta,
    int o_tok_stride, int key_chunk, int Pmax, int seg_rows, int row_base) {
  extern __shared__ float sm[];
  const int eh = blockIdx.x;
  const int lrow = eh / HQ;
  const int qh   = eh % HQ;
  const int tid  = threadIdx.x;

  const int prefix = meta[0].x;
  const int rows   = meta[0].y;
  const int m = min(seg_rows, rows - row_base);
  // Block-uniform; precedes every barrier.
  if (m <= 0 || lrow >= m) return;
  const int L = prefix + row_base;
  const int Pactive = (L + m + key_chunk - 1) / key_chunk;

  const int kv = qh / GQA;
  const int qig = qh % GQA;
  const size_t base = (size_t)(kv * Pmax) * (size_t)(GQA * seg_rows)
                    + (size_t)qig * seg_rows + (size_t)lrow;
  const size_t step = (size_t)GQA * seg_rows;   // partial stride per key chunk

  float* sm_ = sm;
  float* sl_ = sm + Pactive;
  float* red = sm + 2 * Pactive;

  for (int h = tid; h < Pactive; h += DH) {
    sm_[h] = p_max[base + (size_t)h * step];
    sl_[h] = p_sum[base + (size_t)h * step];
  }
  __syncthreads();

  float mx = -3.0e38f;
  for (int h = tid; h < Pactive; h += DH) mx = fmaxf(mx, sm_[h]);
  red[tid] = mx;
  __syncthreads();
  for (int s = DH / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
    __syncthreads();
  }
  const float Mx = red[0];
  __syncthreads();   // all threads have read red[0] before red[] is reused

  float pv = 0.0f;
  for (int h = tid; h < Pactive; h += DH) pv += sl_[h] * __expf(sm_[h] - Mx);
  red[tid] = pv;
  __syncthreads();
  for (int s = DH / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  const float lse = red[0];   // sum_h l_h * exp(m_h - Mx)
  __syncthreads();            // all threads have read red[0] before red[] is reused

  const int d = tid;          // blockDim.x == DH
  float acc = 0.0f;
  for (int h = 0; h < Pactive; ++h)
    acc = fmaf(__expf(sm_[h] - Mx), p_out[(base + (size_t)h * step) * DH + d], acc);

  const float inv = (lse > 0.0f) ? (1.0f / lse) : 0.0f;
  core_out[(size_t)lrow * o_tok_stride + (size_t)qh * DH + d] =
      __float2bfloat16(acc * inv);
}

extern "C" __global__ void __launch_bounds__(DH) tcmqa_d256_phase2(
    const float* __restrict__ p_max, const float* __restrict__ p_sum,
    const float* __restrict__ p_out, __nv_bfloat16* __restrict__ core_out,
    const int4* __restrict__ meta,
    int o_tok_stride, int key_chunk, int Pmax, int seg_rows, int row_base) {
  guard_meta(meta);
  guard_pmax(meta, key_chunk, Pmax);
  guard_geometry(seg_rows, key_chunk, blockDim.x, 0);
  if (gridDim.x != HQ * seg_rows) __trap();
  phase2_body(p_max, p_sum, p_out, core_out, meta, o_tok_stride, key_chunk, Pmax,
              seg_rows, row_base);
}

}  // namespace tcmqa_d256
