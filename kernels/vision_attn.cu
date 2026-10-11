// Qwen3.8 vision attention: MHA 16 x 72, non-causal, per-image blocks, BF16.
//
// Reference geometry (transformers 5.10.2, modeling_qwen3_5.py:905-985):
// 16 Q heads = 16 KV heads, head_dim 72, scaling head_dim**-0.5, is_causal = False,
// no mask, no dropout, no softcap, no Q/K norm. Sequences are per image, taken
// from cu_seqlens; a key outside its own image block carries no weight.
//
// D = 72 = 9 x 8: QK^T runs 9 k8 steps (mma.m16n8k8) and P*V runs 9 n8 slices,
// so neither operand needs padding lanes. The smem row stride is 144 B = 36
// words, which puts the 8 rows of an ldmatrix on 8 distinct bank groups
// (36r mod 32 = 4r), i.e. conflict-free without padding.
//
// Layouts (both owned by the producers, not by this kernel):
//   q, k, v  head-major   [head][row][72]   -- K/V tiles are contiguous copies
//   o        token-major  [row][head][72]   -- what the following projection needs
//
// Launch: grid(ceil(max_block_rows / MT), HQ, nblocks), block 128 (4 warps),
//         static smem 28160 B. One CTA owns MT query rows of one head.
#include <cuda_bf16.h>
#include <math_constants.h>
#include <stdint.h>

namespace vision_attn {

using bf16 = __nv_bfloat16;

// ---------------------------------------------------------------- shape ----
constexpr int HQ  = 16;            // q heads == kv heads (MHA)
constexpr int D   = 72;            // head_dim
constexpr int LDS = D;             // smem row stride in elements (144 B)
constexpr int KS  = D / 8;         // 9 QK k8 steps
constexpr int NS  = D / 8;         // 9 P*V n8 dim slices
// fp32 nearest to 72**-0.5 (the double is 0.11785113019775792). A named constant
// so the value is one line, not an expression re-derived per build.
constexpr float SCALE = 0.11785113066434860f;

static_assert(D % 8 == 0, "D must be a multiple of 8 (k8 QK steps, n8 PV slices)");
static_assert(LDS % 8 == 0, "row stride must stay 16B-aligned for ldmatrix/cp.async");

// The tile is a compile-time parameter of the body, not a runtime value: one
// instantiation per (query rows, key tile, warps). Q tile, K tile and V tile
// only -- the row statistics stay in registers.
//
// The three instantiations below are the ones Q38_VISION_DESIGN.md §11.3 argues
// about; its arithmetic intensity (AI = MT) is a prediction and the A4000 timing
// is what decides. The tall tiles get a launch-bounds minimum of 1: at 256 or 512
// threads, asking for 2 CTAs would cap registers at 65536/(NT*2) and spill.

// --------------------------------------------------------------- helpers ----
__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return (unsigned)__cvta_generic_to_shared(p);
}
// m8n8.x2: lanes 0-7 address matrix 0's rows, lanes 8-15 matrix 1's rows.
__device__ __forceinline__ void ldsm_x2(unsigned (&r)[2], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2_t(unsigned (&r)[2], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1]) : "r"(smem_u32(p)));
}
// m8n8.x1: lanes 0-7 address the matrix's 8 rows.
__device__ __forceinline__ void ldsm_x1(unsigned& r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
               : "=r"(r) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_1688(float (&d)[4], const unsigned (&a)[2], unsigned b) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(b));
}
__device__ __forceinline__ void mma_16816(float (&d)[4], const unsigned (&a)[4],
                                          const unsigned (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ unsigned pack_bf16x2(float lo, float hi) {
  const __nv_bfloat162 t = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const unsigned*>(&t);
}
// cp.async 16B copy; sz == 0 zero-fills (rows outside the block's real extent).
__device__ __forceinline__ void cp16(void* dst, const void* src, int sz) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               ::"r"(smem_u32(dst)), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;\n"); }

// The only visibility bound on this path: a key past its image block has no
// weight. There is no causal structure and no caller-supplied mask.
__device__ __forceinline__ bool vision_key_visible(int q_local, int k_local, int n) {
  return k_local < n;
}

// ----------------------------------------------------------------- guards ---
// Device-side geometry checks. The host plan owns the shape arithmetic; these
// trap on a launch that would otherwise write outside the block's rows.
template <int KT>
__device__ __forceinline__ void guard_geometry(const int* cu, const int* kb, int nblocks) {
  if (nblocks <= 0) __trap();
  if (cu[0] != 0 || kb[0] != 0) __trap();
  for (int b = 0; b < nblocks; ++b) {
    const int n = cu[b + 1] - cu[b];
    const int klen = kb[b + 1] - kb[b];
    if (n <= 0) __trap();                       // an empty block has no l to divide by
    if (klen < n) __trap();                     // K/V must cover the block
    if (klen > n + KT - 1) __trap();            // and be padded by less than one tile
    if (klen % KT != 0) __trap();               // tiles are whole
    if (kb[b] % KT != 0) __trap();              // and tile-aligned in the global tensor
  }
}

// -------------------------------------------------------------- the kernel ---
template <int MT, int KT, int NW>
__device__ __forceinline__ void vision_attn_body(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
    bf16* __restrict__ o, const int* __restrict__ cu, const int* __restrict__ kb,
    int nblocks) {
  constexpr int NT = NW * 32;                  // threads
  constexpr int SN = KT / 8;                   // key n8 tiles per key tile
  constexpr int VN = KT / 16;                  // P*V k16 steps
  constexpr int SMEM_ELEMS = (MT + 2 * KT) * LDS;
  static_assert(KT % 16 == 0, "KT must be a multiple of 16 (P*V k16 steps)");
  static_assert(MT % 16 == 0 && MT / 16 == NW, "one warp per 16-row slab");
  static_assert(SMEM_ELEMS * 2 <= 99 * 1024, "shared memory per CTA");
  const int qt = blockIdx.x;        // query tile
  const int h  = blockIdx.y;        // head
  const int b  = blockIdx.z;        // image block
  if (h >= HQ || b >= nblocks) return;

  const int q_base = cu[b];
  const int n      = cu[b + 1] - q_base;      // this block's rows = patches
  const int k_base = kb[b];
  const int qr0    = q_base + qt * MT;        // first global query row of this CTA
  const int m_rows = min(MT, cu[b + 1] - qr0);
  if (m_rows <= 0) return;                    // block- and grid-uniform exit

  const int tid  = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  __shared__ __align__(16) bf16 vsm[SMEM_ELEMS];
  bf16* qsm = vsm;
  bf16* ksm = qsm + MT * LDS;
  bf16* vtm = ksm + KT * LDS;

  // ---- Q tile: rows [qr0, qr0 + MT) of this head, zero-filled past the block ----
  for (int idx = tid; idx < MT * (LDS / 8); idx += NT) {
    const int r = idx / (LDS / 8), c = idx % (LDS / 8);
    uint4 val = make_uint4(0u, 0u, 0u, 0u);
    if (r < m_rows)
      val = *reinterpret_cast<const uint4*>(q + ((size_t)h * (size_t)cu[nblocks] + qr0 + r) * D + c * 8);
    *reinterpret_cast<uint4*>(qsm + r * LDS + c * 8) = val;
  }
  __syncthreads();

  // Q fragments, held for the whole kernel: 9 k8 steps x 2 registers.
  unsigned qa[KS][2];
#pragma unroll
  for (int kk = 0; kk < KS; ++kk) {
    const int rq = warp * 16 + (lane & 15);
    ldsm_x2(qa[kk], qsm + rq * LDS + kk * 8);
  }

  // Per-lane rows: the m16n8 C fragment puts row (lane>>2) in / lane>>2 + 8 in
  // elements 0,1 / 2,3. Both halves carry their own row max and row sum.
  const int rl = warp * 16 + (lane >> 2);
  const int rh = rl + 8;
  const int ql = qt * MT + rl;                 // block-local query rows
  const int qh = qt * MT + rh;

  float acc[NS][4];
#pragma unroll
  for (int nd = 0; nd < NS; ++nd)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[nd][j] = 0.0f;
  float mx[2] = {-3.0e38f, -3.0e38f};          // finite sentinel: never NaN
  float ls[2] = {0.0f, 0.0f};

  // ---- sweep the block's keys in KT tiles ----
  // K(t) is staged, then V(t) during the S mma and softmax, then K(t+1) during
  // the P*V mma: one buffer each, refilled only after its consumer is done.
  const int ntiles = (n + KT - 1) / KT;
  auto stage = [&](const bf16* src, int t, bf16* dst) {
    const int t0 = t * KT;
#pragma unroll
    for (int rep = 0; rep < (KT * (LDS / 8) + NT - 1) / NT; ++rep) {
      const int idx = tid + rep * NT;
      if (idx >= KT * (LDS / 8)) break;
      const int r = idx / (LDS / 8), c = idx % (LDS / 8);
      const int kl = t0 + r;                       // block-local key row
      const bool live = kl < n;
      const bf16* g = live
          ? src + ((size_t)h * (size_t)(kb[nblocks]) + (size_t)(k_base + kl)) * D + c * 8
          : src;
      cp16(dst + r * LDS + c * 8, g, live ? 16 : 0);
    }
  };

  if (ntiles > 0) { stage(k, 0, ksm); cp_commit(); }

  for (int t = 0; t < ntiles; ++t) {
    cp_wait0();                                    // K(t) landed
    __syncthreads();                               // K(t) visible; V(t-1) consumed
    stage(v, t, vtm);
    cp_commit();                                   // V(t) flies during S + softmax

    // ---- S = Q @ K^T: 8 key n8 tiles x 9 k8 steps ----
    float sf[SN][4];
#pragma unroll
    for (int n8 = 0; n8 < SN; ++n8)
#pragma unroll
      for (int j = 0; j < 4; ++j) sf[n8][j] = 0.0f;
#pragma unroll
    for (int kk = 0; kk < KS; ++kk) {
#pragma unroll
      for (int n8 = 0; n8 < SN; ++n8) {
        unsigned kb8;
        // B is k8(dim) x n8(key) with keys as the smem rows of K[key][dim].
        ldsm_x1(kb8, ksm + (n8 * 8 + (lane & 7)) * LDS + kk * 8);
        mma_1688(sf[n8], qa[kk], kb8);
      }
    }

#pragma unroll
    for (int n8 = 0; n8 < SN; ++n8)
#pragma unroll
      for (int j = 0; j < 4; ++j) sf[n8][j] *= SCALE;

    // ---- visibility: no key past the block's extent ----
#pragma unroll
    for (int n8 = 0; n8 < SN; ++n8) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int kl = t * KT + n8 * 8 + (lane & 3) * 2 + (j & 1);
        const int q_loc = (j < 2) ? ql : qh;
        sf[n8][j] = vision_key_visible(q_loc, kl, n) ? sf[n8][j] : -CUDART_INF_F;
      }
    }

    // ---- online softmax, one row half per reduction ----
    float pn[SN][4];
    float rmax[2] = {-3.0e38f, -3.0e38f};
#pragma unroll
    for (int n8 = 0; n8 < SN; ++n8) {
      rmax[0] = fmaxf(rmax[0], fmaxf(sf[n8][0], sf[n8][1]));
      rmax[1] = fmaxf(rmax[1], fmaxf(sf[n8][2], sf[n8][3]));
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
      for (int n8 = 0; n8 < SN; ++n8) {
        const float p0 = __expf(sf[n8][half * 2 + 0] - m_new);
        const float p1 = __expf(sf[n8][half * 2 + 1] - m_new);
        pn[n8][half * 2 + 0] = p0;
        pn[n8][half * 2 + 1] = p1;
        rsum += p0 + p1;
      }
      rsum += __shfl_xor_sync(0xffffffffu, rsum, 1);
      rsum += __shfl_xor_sync(0xffffffffu, rsum, 2);
      ls[half] = ls[half] * cf + rsum;
#pragma unroll
      for (int nd = 0; nd < NS; ++nd) {
        acc[nd][half * 2 + 0] *= cf;
        acc[nd][half * 2 + 1] *= cf;
      }
    }

    cp_wait0();                                    // V(t) landed
    __syncthreads();                               // V(t) visible; K(t) consumed
    if (t + 1 < ntiles) { stage(k, t + 1, ksm); cp_commit(); }

    // ---- O += P @ V: 4 k16 steps x 9 dim slices; P is rounded to bf16 here ----
    for (int s = 0; s < VN; ++s) {
      unsigned pa[4];
      pa[0] = pack_bf16x2(pn[s * 2 + 0][0], pn[s * 2 + 0][1]);
      pa[1] = pack_bf16x2(pn[s * 2 + 0][2], pn[s * 2 + 0][3]);
      pa[2] = pack_bf16x2(pn[s * 2 + 1][0], pn[s * 2 + 1][1]);
      pa[3] = pack_bf16x2(pn[s * 2 + 1][2], pn[s * 2 + 1][3]);
#pragma unroll
      for (int nd = 0; nd < NS; ++nd) {
        unsigned vb[2];
        // B is k16(key) x n8(dim) on V[key][dim]: rows are keys, so the operand
        // needs the transposed load. mat0 lanes 0-7: key s*16 + lane&7;
        // mat1 lanes 8-15: key s*16 + 8 + lane&7.
        ldsm_x2_t(vb, vtm + (s * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LDS + nd * 8);
        mma_16816(acc[nd], pa, vb);
      }
    }
  }

  // ---- write out: one bf16 row per (row, head), token-major ----
  const bool live_l = rl < m_rows;
  const bool live_h = rh < m_rows;
  const float inv_l = live_l ? 1.0f / ls[0] : 0.0f;
  const float inv_h = live_h ? 1.0f / ls[1] : 0.0f;
#pragma unroll
  for (int nd = 0; nd < NS; ++nd) {
    const int c0 = nd * 8 + (lane & 3) * 2;
    if (live_l)
      *reinterpret_cast<unsigned*>(o + ((size_t)(qr0 + rl) * HQ + h) * D + c0) =
          pack_bf16x2(acc[nd][0] * inv_l, acc[nd][1] * inv_l);
    if (live_h)
      *reinterpret_cast<unsigned*>(o + ((size_t)(qr0 + rh) * HQ + h) * D + c0) =
          pack_bf16x2(acc[nd][2] * inv_h, acc[nd][3] * inv_h);
  }
}

}  // namespace vision_attn

// Three instantiations of the same source. `vision_attn_kernel` keeps its name
// because it is the one already landed in kernels/; the tall tiles are new.
#define VISION_ATTN_ENTRY(NAME, MT, KT, NW, MINB)                          \
  extern "C" __global__ void __launch_bounds__((NW) * 32, MINB) NAME(       \
      const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k, \
      const __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ o,   \
      const int* __restrict__ cu_seqlens, const int* __restrict__ k_bases,  \
      int nblocks) {                                                        \
    if (gridDim.y != vision_attn::HQ) __trap();                             \
    if (blockDim.x != (NW) * 32) __trap();                                  \
    vision_attn::guard_geometry<KT>(cu_seqlens, k_bases, nblocks);          \
    vision_attn::vision_attn_body<MT, KT, NW>(q, k, v, o, cu_seqlens, k_bases, nblocks); \
  }

// The tile constants, named once. The instantiations below and the host-visible
// table are both built from them, so the driver's K/V padding and the manifest's
// k_t come from the same source as the kernel that consumes them. A copy on the
// host side once said 64 where m256's template said 32; the device guard trapped
// it, and this removes the possibility rather than relying on the guard.
constexpr int TILE_M64_ROWS = 64, TILE_M64_KEYS = 64, TILE_M64_WARPS = 4, TILE_M64_MINB = 2;

// Row order is the tile order the driver's selector and the harness use.
//
// ONE tile. The 128- and 256-row instantiations were built, verified and measured
// (P1V_STATUS §9.1): the tall tiles were never faster, and at small n they were up
// to 60% slower, because their better arithmetic intensity did not pay for dropping
// from 3 CTAs/SM to 1. The template parameterisation stays, so a tall tile can come
// back the moment a held-clock measurement shows a large-n win worth having.
extern "C" const char* const kVisionTileNames[1] = {"m64"};
extern "C" const int kVisionTileSpec[1][3] = {
    {TILE_M64_ROWS, TILE_M64_KEYS, TILE_M64_WARPS},
};

VISION_ATTN_ENTRY(vision_attn_kernel, TILE_M64_ROWS, TILE_M64_KEYS, TILE_M64_WARPS, TILE_M64_MINB)
