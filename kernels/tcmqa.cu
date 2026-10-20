// TCMQA: tensor-core multi-query GQA flash-decoding attention. M query tokens
// share one KV cache; tree visibility via a per-row ancestor bitset. S and P*V
// run on bf16 mma.sync.m16n8k16 with fp32 accumulation; the online softmax is
// fp32 in registers per MMA fragment row.
//
// Shape: 16 q-heads / 2 kv-heads (GQA 8), head_dim 128, scale = 1/sqrt(128).
// Every entry point __trap()s when n_q_heads != 16, n_kv_heads != 2, gqa != 8,
// head_dim != 128, M outside [1, MQ_MAX=64], chunk not a positive multiple of
// 64, or (phase 1) blockDim.x/gridDim.y/gridDim.z mismatch or anc == nullptr.
//
// Semantics:
//   The KV caches hold prefix positions [0, L) plus the M new tokens'
//   positions [L, L+M). Query token i attends all prefix positions plus draft
//   position L+j iff bit j of anc[i] is set; positions >= L+M are never
//   visible. Tiles fully below L skip the mask.
//
//   q  : bf16 [M][16][128]; row (i, qh) at q + i*q_tok_stride + qh*q_head_stride
//   K,V: bf16, separate tensors, head_dim-contiguous rows
//   o  : bf16 [M][16][128]; row (i, qh) at o + i*o_tok_stride + qh*o_head_stride
//   anc: M u64 (device); bit j of row i = draft position L+j visible.
//
// Pairs: pair index p = qh_local*M + i in [0, 8M) per kv head. Pair p occupies
// MMA row (p - rt*MT) of row-tile rt; rows with p >= 8M are computed but never
// written.
//
// Two-phase split-KV:
//   phase 1: grid (nchunks, 2, ceil(8M/64)), block 128 = 4 warps. Block
//            (ch, kv, rt) sweeps chunk ch of kv-head kv in KT=32-position
//            tiles; warp w owns MMA rows [w*16, w*16+16) and writes one
//            partial (m, l, acc[128]) per pair.
//   phase 1 wide (_w, _ws): grid (nchunks, 2, ceil(8M/128)), block 256 = 8
//            warps sharing one 128-row pair tile; outputs are bitwise
//            identical to the 64-row variant.
//   phase 2: grid (16*M); one block merges the nchunks partials of one
//            (query, head) pair by LSE rescaling and writes o.
//
// Per-warp per-K-tile flow:
//   S[16][32] = Q_w[16][128] * K_tile^T[128][32]: 8 k16 x 4 n8 mma; A =
//               ldmatrix.x4 on Q smem, B = ldmatrix.x2 on K smem rows [pos][d].
//   softmax   : each fragment row is held by a 4-lane quad; row reductions are
//               2 quad shuffles. Masked elements are -INF; the finite -3e38
//               running-max sentinel keeps m - m' NaN-free.
//   P bf16    : S D fragments repack in-register to A fragments.
//   O += P*V  : 2 k16 x 16 n8 mma; B = ldmatrix.x2.trans on V smem rows [pos][dim].
//
// Thread map: lane l holds D-fragment elements (row l/4, cols (l%4)*2+{0,1})
// and (row l/4+8, same cols) per n8 slice; m/l are replicated across each
// row's quad.
//
// Smem (dynamic): Q [MT][LDSQ] bf16 | K [32][LDSK] | V [32][136] | anc[64] u64.
//   phase 1: (64+32+32)*136*2 + 512 = 35328 B.
//   _w:      (128+32+32)*136*2 + 512 = 52736 B.
//   _ws:     50176 B (Q/K unpadded and XOR-swizzled).
//   _w and _ws require cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize).
//   phase 2: block = threads2 (power of two in [128, 1024]),
//            smem = (2*nchunks + threads2) * 4 B <= 48 KB.
// Out-of-extent tile rows are zero-filled on load.
//
// KV layouts (strides in bf16 elements, multiples of 8):
//   position-major [pos][kv][d]:  head_stride = 128,     pos_stride = 256
//   head-major     [kv][pos][d]:  head_stride = Lh*128,  pos_stride = 128
//   Address: kv*head_stride + pos*pos_stride + d. K and V strides are
//   independent. q/o strides are multiples of 8.
//
// Scratch (caller-allocated fp32; NC = ceil((L+M)/chunk); NP = 16*NC*M):
//   scratch[0    .. NP)         = m   [NP]
//   scratch[NP   .. 2*NP)       = l   [NP]
//   scratch[2*NP .. NP*(2+128)) = acc [NP][128]
//   partial index = (kv*NC + ch)*8*M + p
//   Size: 16 * NC * M * 130 floats. Each partial is written by exactly one block.
//
// CUDA Graph: entry points read L from len_dev. M, chunk and the anc pointer
// are fixed per capture; M and chunk are identical in both phases. Phase 1 may
// launch with grid.x = nchunks_max; blocks with blockIdx.x >= ceil((L+M)/chunk)
// return before writing.
#include <cuda_runtime.h>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

constexpr int HQ  = 16;                 // q heads
constexpr int HKV = 2;                  // kv heads
constexpr int GQA = HQ / HKV;           // 8 q heads per kv head
constexpr int DH  = 128;                // head_dim
constexpr int MT  = 64;                 // MMA rows per block (pair tile)
constexpr int KT  = 32;                 // KV positions per smem tile
constexpr int NW  = 4;                  // warps per block
constexpr int NT  = NW * 32;            // 128 threads
constexpr int MQ_MAX = 64;              // guard range for query tokens: 1..MQ_MAX
constexpr int PAD = 8;                  // smem row padding (elements) vs ldmatrix banks
constexpr int LDS = DH + PAD;           // padded smem row stride = 136
constexpr float SCALE = 0.08838834764831845f;   // 1/sqrt(128)

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return (unsigned)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
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
// cp.async 16B copy; sz = 0 zero-fills the destination (dead rows).
__device__ __forceinline__ void cp16(void* dst, const void* src, int sz) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(smem_u32(dst)), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cp_wait0() {
  asm volatile("cp.async.wait_group 0;\n");
}

// Block (ch, kv, rt): sweeps chunk ch of kv-head kv in KT-pos tiles; warp w
// owns MMA rows [w*16, w*16+16) of the MT_-row pair tile (MT_ = 64 with
// NT_ = 128 threads, or MT_ = 128 with NT_ = 256).
// LDSQ/LDSK/LDSV: Q/K/V tile row strides in elements. SWZ: Q/K 16B segment
// index is XORed with (row & 7).
template <int MT_, int NT_, int LDSQ, int LDSK, int LDSV, bool SWZ>
__device__ __forceinline__ void tcmqa_phase1_body(
    const bf16* __restrict__ q, const bf16* __restrict__ Kc, const bf16* __restrict__ Vc,
    float* __restrict__ p_m, float* __restrict__ p_l, float* __restrict__ p_o,
    const unsigned long long* __restrict__ anc,
    int L, int M, int chunk, int nchunks,
    int q_tok_stride, int q_head_stride,
    int k_head_stride, int k_pos_stride, int v_head_stride, int v_pos_stride) {
  const int ch  = blockIdx.x;
  const int kv  = blockIdx.y;
  const int rt  = blockIdx.z;
  const int pos_beg = ch * chunk;
  const int Lext = L + M;                             // total KV extent
  const int pos_end = min(pos_beg + chunk, Lext);

  const int tid   = threadIdx.x;
  const int warp  = tid >> 5;
  const int lane  = tid & 31;
  const int P     = GQA * M;                          // live pairs per kv head
  const int row0  = rt * MT_;                         // block's first pair

  extern __shared__ __align__(16) bf16 tcm_sm[];
  bf16* qsm = tcm_sm;                                 // [MT_][LDSQ]
  bf16* ksm = tcm_sm + MT_ * LDSQ;                    // [KT][LDSK]
  bf16* vsm = ksm + KT * LDSK;                        // [KT][LDSV]
  unsigned long long* asm_ = reinterpret_cast<unsigned long long*>(vsm + KT * LDS);

  // ---- stage anc (M u64) and the Q row tile (cooperative 16B copies) ----
  for (int i = tid; i < M; i += NT_) asm_[i] = anc[i];
  {
    const int nvec = MT_ * (DH / 8);                  // 16B chunks per row-slab
    for (int idx = tid; idx < nvec; idx += NT_) {
      const int r = idx >> 4, seg = idx & 15;
      const int p = row0 + r;
      uint4 val = make_uint4(0u, 0u, 0u, 0u);
      if (p < P)
        val = *reinterpret_cast<const uint4*>(q + (size_t)(p % M) * q_tok_stride +
                                              (size_t)(kv * GQA + p / M) * q_head_stride +
                                              seg * 8);
      const int qseg = SWZ ? ((seg ^ (r & 7)) * 8) : (seg * 8);
      *reinterpret_cast<uint4*>(qsm + r * LDSQ + qseg) = val;
    }
  }
  __syncthreads();

  // Per-lane row indices (D-fragment layout): rows lane/4 and lane/4+8 of the
  // warp's 16-row slab.
  const int rl = warp * 16 + (lane >> 2);             // row within tile (lo half)
  const int rh = rl + 8;                              // row within tile (hi half)
  const int pl = row0 + rl;                           // global pair (lo)
  const int ph = row0 + rh;                           // global pair (hi)
  const int il = pl % M;                              // query index (lo row)
  const int ih = ph % M;                              // query index (hi row)

  float acc[16][4];                                   // 16 n8-slices x 4 fp32
  float mx[2] = {-3.0e38f, -3.0e38f};                 // finite sentinel: never NaN
  float ls[2] = {0.0f, 0.0f};
#pragma unroll
  for (int n = 0; n < 16; ++n)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[n][j] = 0.0f;

  // cp.async pipeline: the prologue stages K(0); per tile, CP1 waits for K(t)
  // and stages V(t) over S + softmax; CP2 waits for V(t) and stages K(t+1)
  // over PV. Each stage overwrites the tile consumed before the preceding
  // barrier.
  const int ntiles = (pos_end > pos_beg) ? ((pos_end - pos_beg + KT - 1) / KT) : 0;

  // Stage one 32-pos tile of a KV tensor into `dst` ([KT][LDS]) via cp.async
  // 16B copies; out-of-extent rows are zero-filled (src-size 0).
  auto stage_kv = [&](const bf16* src, int head_stride, int pos_stride, int t,
                      bf16* dst, int stride, bool swz) {
    const int tt0 = pos_beg + t * KT;
#pragma unroll
    for (int rep = 0; rep < KT * (DH / 8) / NT_; ++rep) {   // 512 % NT_ == 0
      const int idx = tid + rep * NT_;
      if (idx < KT * (DH / 8)) {
        const int r = idx >> 4, seg = idx & 15;
        const int pos = tt0 + r;
        const bool live = pos < pos_end;
        const bf16* g = live ? src + (size_t)kv * head_stride +
                                     (size_t)pos * pos_stride + seg * 8
                             : src;
        const int dseg = swz ? ((seg ^ (r & 7)) * 8) : (seg * 8);
        cp16(dst + r * stride + dseg, g, live ? 16 : 0);
      }
    }
  };

  if (ntiles > 0) { stage_kv(Kc, k_head_stride, k_pos_stride, 0, ksm, LDSK, SWZ); cp_commit(); }

  for (int t = 0; t < ntiles; ++t) {
    const int t0 = pos_beg + t * KT;
    cp_wait0();                                       // K(t) landed (newest group)
    __syncthreads();                                  // K(t) visible; V(t-1) consumed by all
    stage_kv(Vc, v_head_stride, v_pos_stride, t, vsm, LDSV, false);
    cp_commit();                                      // V(t) flies during S + softmax

    // ---- S = Q_w * K_tile^T : 8 k16 x 4 n8 mma, per-warp row slab ----
    float sf[4][4];                                   // 4 n8-slices x 4 fp32
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int j = 0; j < 4; ++j) sf[n][j] = 0.0f;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      unsigned qa[4];
      // A 16x16 at Q rows [warp*16, +16) x dims [kk*16, +16):
      // lane -> row (lane%16), col (lane/16)*8.
      {
        const int rq = warp * 16 + (lane & 15);
        const int sq = SWZ ? ((2 * kk + (lane >> 4)) ^ (rq & 7)) : (2 * kk + (lane >> 4));
        ldsm_x4(qa, qsm + rq * LDSQ + sq * 8);
      }
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        unsigned kb[2];
        // B k16(d) x n8(pos), plain ldmatrix on K[pos][d]:
        // mat0 lanes 0-7: pos n*8+lane%8, d kk*16; mat1 lanes 8-15: d kk*16+8.
        {
          const int rk = n * 8 + (lane & 7);
          const int sk = SWZ ? ((2 * kk + ((lane >> 3) & 1)) ^ (rk & 7)) : (2 * kk + ((lane >> 3) & 1));
          ldsm_x2(kb, ksm + rk * LDSK + sk * 8);
        }
        mma_16816(sf[n], qa, kb);
      }
    }

#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int j = 0; j < 4; ++j) sf[n][j] *= SCALE;

    // ---- mask (only tiles overlapping the draft positions) ----
    if (t0 + KT > L) {
      const unsigned long long al = asm_[il];
      const unsigned long long ah = asm_[ih];
#pragma unroll
      for (int n = 0; n < 4; ++n) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int pos = t0 + n * 8 + (lane & 3) * 2 + (j & 1);
          const int jj = pos - L;
          const bool vis = (jj < 0) || (jj < M && ((j < 2 ? al : ah) >> jj) & 1ull);
          if (!vis) sf[n][j] = -INFINITY;
        }
      }
    }

    // ---- online softmax (per row; quad reductions) ----
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
#pragma unroll
      for (int n = 0; n < 16; ++n) {
        acc[n][half * 2 + 0] *= cf;
        acc[n][half * 2 + 1] *= cf;
      }
    }

    // ---- CP2: V(t) landed; stage K(t+1) into the just-consumed K tile ----
    cp_wait0();                                       // V(t) landed
    __syncthreads();                                  // V(t) visible; K(t) consumed by all
    if (t + 1 < ntiles) {
      stage_kv(Kc, k_head_stride, k_pos_stride, t + 1, ksm, LDSK, SWZ);
      cp_commit();                                    // K(t+1) flies during PV
    }

    // ---- O += P * V : P bf16 repack, then 2 k16 x 16 n8 mma ----
    unsigned pa[2][4];
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {                  // k16 = pos [kk*16, +16)
      pa[kk][0] = pack_bf16x2(pn[kk * 2 + 0][0], pn[kk * 2 + 0][1]);
      pa[kk][1] = pack_bf16x2(pn[kk * 2 + 0][2], pn[kk * 2 + 0][3]);
      pa[kk][2] = pack_bf16x2(pn[kk * 2 + 1][0], pn[kk * 2 + 1][1]);
      pa[kk][3] = pack_bf16x2(pn[kk * 2 + 1][2], pn[kk * 2 + 1][3]);
    }
#pragma unroll
    for (int nd = 0; nd < 16; ++nd) {                 // dim n8 slices
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

  // ---- write one partial per owned pair (dead rows never write) ----
  const size_t part0 = (size_t)(kv * nchunks + ch) * (size_t)P;
  if ((lane & 3) == 0) {
    if (pl < P) { p_m[part0 + pl] = mx[0]; p_l[part0 + pl] = ls[0]; }
    if (ph < P) { p_m[part0 + ph] = mx[1]; p_l[part0 + ph] = ls[1]; }
  }
#pragma unroll
  for (int nd = 0; nd < 16; ++nd) {
    const int c0 = nd * 8 + (lane & 3) * 2;
    if (pl < P)
      *reinterpret_cast<float2*>(p_o + (part0 + pl) * DH + c0) =
          make_float2(acc[nd][0], acc[nd][1]);
    if (ph < P)
      *reinterpret_cast<float2*>(p_o + (part0 + ph) * DH + c0) =
          make_float2(acc[nd][2], acc[nd][3]);
  }
}

// One block per (query, head) pair; blockIdx.x = eh = qi*n_q_heads + qh.
// Partial index: part(ch) = (kv*nchunks + ch)*8*M + qh_local*M + qi,
// stride 8*M per chunk.
__device__ __forceinline__ void tcmqa_phase2_body(const float* __restrict__ p_m,
                                                  const float* __restrict__ p_l,
                                                  const float* __restrict__ p_o,
                                                  bf16* __restrict__ o, int nchunks,
                                                  int M, int n_q_heads,
                                                  int o_tok_stride, int o_head_stride) {
  extern __shared__ float sm[];
  const int eh  = blockIdx.x;
  const int qi  = eh / n_q_heads;
  const int qh  = eh % n_q_heads;
  const int kv  = qh / GQA;
  const int qhl = qh % GQA;
  const int tid = threadIdx.x;
  const int nt  = blockDim.x;
  const int G2  = nt / DH;
  const int g   = tid / DH;
  const int d   = tid % DH;

  float* sm_ = sm;
  float* sl_ = sm + nchunks;
  float* red = sm + 2 * nchunks;

  const size_t pbase  = (size_t)kv * nchunks * (GQA * M) + (size_t)qhl * M + qi;
  const size_t cstep  = (size_t)GQA * M;               // partial stride per chunk
  for (int h = tid; h < nchunks; h += nt) {
    sm_[h] = p_m[pbase + (size_t)h * cstep];
    sl_[h] = p_l[pbase + (size_t)h * cstep];
  }
  __syncthreads();

  float mm = -3.0e38f;
  for (int h = tid; h < nchunks; h += nt) mm = fmaxf(mm, sm_[h]);
  red[tid] = mm;
  __syncthreads();
  for (int s = nt / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
    __syncthreads();
  }
  const float Mx = red[0];

  float pv = 0.0f;
  for (int h = tid; h < nchunks; h += nt) pv += sl_[h] * __expf(sm_[h] - Mx);
  __syncthreads();      // every thread has consumed red[0] before red[] is reused
  red[tid] = pv;
  __syncthreads();
  for (int s = nt / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  const float lse = red[0];                            // sum_h l_h * exp(m_h - Mx)
  __syncthreads();      // same barrier before red[] is reused by the G2 reduction

  float acc = 0.0f;
  for (int h = g; h < nchunks; h += G2) {
    const float w = __expf(sm_[h] - Mx);
    acc = fmaf(w, p_o[(pbase + (size_t)h * cstep) * DH + d], acc);
  }
  if (G2 > 1) {
    red[tid] = acc;
    __syncthreads();
#pragma unroll
    for (int k = 1; k < G2; ++k) acc += red[k * DH + d];
  }

  const float inv = (lse > 0.0f) ? (1.0f / lse) : 0.0f;
  // only group 0 holds the reduced acc
  if (g == 0) o[(size_t)qi * o_tok_stride + (size_t)qh * o_head_stride + d] =
      __float2bfloat16(acc * inv);
}

__device__ __forceinline__ void tcmqa_phase2_short_body(
    const float* __restrict__ p_m, const float* __restrict__ p_l,
    const float* __restrict__ p_o, bf16* __restrict__ o, int nchunks,
    int M, int o_tok_stride, int o_head_stride) {
  extern __shared__ float sm[];
  __shared__ float stats[2];
  const int tid = threadIdx.x;
  const int qi = blockIdx.x / HQ;
  const int qh = blockIdx.x % HQ;
  const int g = tid / DH;
  const int d = tid % DH;
  const int groups = blockDim.x / DH;
  const size_t step = (size_t)GQA * M;
  const size_t base = (size_t)(qh / GQA) * nchunks * step
                   + (size_t)(qh % GQA) * M + qi;
  float* red = sm + 2 * nchunks;

  if (tid < warpSize) {
    const int hi = tid + warpSize;
    float m0 = -3.0e38f, m1 = -3.0e38f;
    float l0 = 0.0f, l1 = 0.0f;
    if (tid < nchunks) {
      sm[tid] = p_m[base + (size_t)tid * step];
      m0 = fmaxf(m0, sm[tid]);
      l0 = p_l[base + (size_t)tid * step];
    }
    if (hi < nchunks) {
      sm[hi] = p_m[base + (size_t)hi * step];
      m1 = fmaxf(m1, sm[hi]);
      l1 = p_l[base + (size_t)hi * step];
    }
    float mx = fmaxf(m0, m1);
    for (int s = warpSize / 2; s > 0; s >>= 1) {
      const float other = __shfl_down_sync(0xffffffffu, mx, s);
      if (tid < s) mx = fmaxf(mx, other);
    }
    mx = __shfl_sync(0xffffffffu, mx, 0);
    float p0 = 0.0f, p1 = 0.0f;
    if (tid < nchunks) p0 += l0 * __expf(sm[tid] - mx);
    if (hi < nchunks) p1 += l1 * __expf(sm[hi] - mx);
    float sum = p0 + p1;
    for (int s = warpSize / 2; s > 0; s >>= 1) {
      const float other = __shfl_down_sync(0xffffffffu, sum, s);
      if (tid < s) sum += other;
    }
    if (tid == 0) {
      stats[0] = mx;
      stats[1] = sum;
    }
  }
  __syncthreads();

  const float mx = stats[0];
  const float sum = stats[1];
  float acc = 0.0f;
  for (int h = g; h < nchunks; h += groups) {
    const float w = __expf(sm[h] - mx);
    acc = fmaf(w, p_o[(base + (size_t)h * step) * DH + d], acc);
  }
  red[tid] = acc;
  __syncthreads();
  if (g == 0) {
    for (int k = 1; k < groups; ++k) acc += red[k * DH + d];
    const float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    o[(size_t)qi * o_tok_stride + (size_t)qh * o_head_stride + d] =
        __float2bfloat16(acc * inv);
  }
}

#define FD_GUARD_P1 do {                                                                    \
  if (head_dim != DH || n_q_heads != HQ || n_kv_heads != HKV || gqa != GQA ||               \
      M < 1 || M > MQ_MAX || chunk < 64 || (chunk % 64) != 0 ||                             \
      blockDim.x != NT || gridDim.y != HKV || gridDim.z != (GQA * M + MT - 1) / MT ||       \
      anc == nullptr) __trap();                                                             \
} while (0)
// Wide variant: 256 threads, one 128-row pair tile per block (grid.z bucket).
#define FD_GUARD_P1W do {                                                                   \
  if (head_dim != DH || n_q_heads != HQ || n_kv_heads != HKV || gqa != GQA ||               \
      M < 1 || M > MQ_MAX || chunk < 64 || (chunk % 64) != 0 ||                             \
      blockDim.x != 256 || gridDim.y != HKV || gridDim.z != (GQA * M + 127) / 128 ||        \
      anc == nullptr) __trap();                                                             \
} while (0)
#define FD_GUARD_P2 do {                                                                    \
  if (head_dim != DH || n_q_heads != HQ || n_kv_heads != HKV || gqa != GQA ||               \
      M < 1 || M > MQ_MAX || chunk < 64 || (chunk % 64) != 0) __trap();                     \
} while (0)

template <int MT_, int NT_>
__device__ __forceinline__ void tcmqa_phase1_entry(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
    float* __restrict__ scratch, const unsigned long long* __restrict__ anc,
    int L, int M, int chunk,
    int q_tok_stride, int q_head_stride,
    int k_head_stride, int k_pos_stride, int v_head_stride, int v_pos_stride) {
  const int nchunks = (L + M + chunk - 1) / chunk;
  // Block-uniform, before any __syncthreads: blocks beyond nchunks write nothing.
  if (blockIdx.x >= nchunks) return;
  float* p_m = scratch;
  float* p_l = scratch + (size_t)HKV * nchunks * GQA * M;
  float* p_o = p_l + (size_t)HKV * nchunks * GQA * M;
  tcmqa_phase1_body<MT_, NT_, LDS, LDS, LDS, false>(q, k, v, p_m, p_l, p_o, anc, L, M,
                                                     chunk, nchunks,
                              q_tok_stride, q_head_stride,
                              k_head_stride, k_pos_stride, v_head_stride, v_pos_stride);
}

extern "C" __global__ void __launch_bounds__(NT) tcmqa_phase1(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
    float* __restrict__ scratch, const int* __restrict__ len_dev, int M,
    int n_q_heads, int n_kv_heads, int gqa, int head_dim,
    int q_tok_stride, int q_head_stride,
    int k_head_stride, int k_pos_stride, int v_head_stride, int v_pos_stride,
    int chunk, const unsigned long long* __restrict__ anc) {
  FD_GUARD_P1;
  tcmqa_phase1_entry<MT, NT>(q, k, v, scratch, anc, *len_dev, M, chunk,
                             q_tok_stride, q_head_stride,
                             k_head_stride, k_pos_stride, v_head_stride, v_pos_stride);
}

// Wide pair-tile variant: one 128-row tile per block; 8 warps share one K/V
// smem tile set.
extern "C" __global__ void __launch_bounds__(256) tcmqa_phase1_w(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
    float* __restrict__ scratch, const int* __restrict__ len_dev, int M,
    int n_q_heads, int n_kv_heads, int gqa, int head_dim,
    int q_tok_stride, int q_head_stride,
    int k_head_stride, int k_pos_stride, int v_head_stride, int v_pos_stride,
    int chunk, const unsigned long long* __restrict__ anc) {
  FD_GUARD_P1W;
  tcmqa_phase1_entry<128, 256>(q, k, v, scratch, anc, *len_dev, M, chunk,
                               q_tok_stride, q_head_stride,
                               k_head_stride, k_pos_stride, v_head_stride, v_pos_stride);
}

__device__ __forceinline__ void tcmqa_phase2_entry(float* __restrict__ scratch,
                                                   bf16* __restrict__ o, int L, int M,
                                                   int chunk, int n_q_heads,
                                                   int o_tok_stride, int o_head_stride) {
  const int nchunks = (L + M + chunk - 1) / chunk;
  const size_t nparts = (size_t)HKV * nchunks * GQA * M;
  const float* p_m = scratch;
  const float* p_l = scratch + nparts;
  const float* p_o = scratch + 2 * nparts;
  tcmqa_phase2_body(p_m, p_l, p_o, o, nchunks, M, n_q_heads, o_tok_stride, o_head_stride);
}

extern "C" __global__ void tcmqa_phase2(float* __restrict__ scratch,
                                        bf16* __restrict__ o,
                                        const int* __restrict__ len_dev, int M,
                                        int n_q_heads, int n_kv_heads, int gqa,
                                        int head_dim,
                                        int o_tok_stride, int o_head_stride,
                                        int chunk) {
  FD_GUARD_P2;
  tcmqa_phase2_entry(scratch, o, *len_dev, M, chunk, n_q_heads, o_tok_stride, o_head_stride);
}

__device__ __forceinline__ void tcmqa_phase2_short_entry(
    float* __restrict__ scratch, bf16* __restrict__ o, int L, int M,
    int chunk, int o_tok_stride, int o_head_stride) {
  const int nchunks = (L + M + chunk - 1) / chunk;
  if (L < 0 || nchunks < 1 || nchunks > 2 * warpSize ||
      (blockDim.x != 256 && blockDim.x != 512) ||
      blockDim.y != 1 || blockDim.z != 1 ||
      gridDim.x != HQ * M || gridDim.y != 1 || gridDim.z != 1) __trap();
  const size_t nparts = (size_t)HKV * nchunks * GQA * M;
  tcmqa_phase2_short_body(scratch, scratch + nparts, scratch + 2 * nparts,
                         o, nchunks, M, o_tok_stride, o_head_stride);
}

extern "C" __global__ void tcmqa_phase2_short(
    float* __restrict__ scratch, bf16* __restrict__ o,
    const int* __restrict__ len_dev, int M,
    int n_q_heads, int n_kv_heads, int gqa, int head_dim,
    int o_tok_stride, int o_head_stride, int chunk) {
  FD_GUARD_P2;
  tcmqa_phase2_short_entry(scratch, o, *len_dev, M, chunk,
                          o_tok_stride, o_head_stride);
}

// Reduced-smem wide variant: Q/K tiles unpadded (stride DH, swizzled), V tile
// stride LDS; smem = (128 + 32) * 128 * 2 + 32 * 136 * 2 + 512 = 50176 B.
__device__ __forceinline__ void tcmqa_phase1_entry_ws(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
    float* __restrict__ scratch, const unsigned long long* __restrict__ anc,
    int L, int M, int chunk,
    int q_tok_stride, int q_head_stride,
    int k_head_stride, int k_pos_stride, int v_head_stride, int v_pos_stride) {
  const int nchunks = (L + M + chunk - 1) / chunk;
  if (blockIdx.x >= nchunks) return;
  float* p_m = scratch;
  float* p_l = scratch + (size_t)HKV * nchunks * GQA * M;
  float* p_o = p_l + (size_t)HKV * nchunks * GQA * M;
  tcmqa_phase1_body<128, 256, DH, DH, LDS, true>(q, k, v, p_m, p_l, p_o, anc, L, M,
                                                chunk, nchunks,
                                 q_tok_stride, q_head_stride,
                                 k_head_stride, k_pos_stride, v_head_stride,
                                 v_pos_stride);
}

extern "C" __global__ void __launch_bounds__(256) tcmqa_phase1_ws(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
    float* __restrict__ scratch, const int* __restrict__ len_dev, int M,
    int n_q_heads, int n_kv_heads, int gqa, int head_dim,
    int q_tok_stride, int q_head_stride,
    int k_head_stride, int k_pos_stride, int v_head_stride, int v_pos_stride,
    int chunk, const unsigned long long* __restrict__ anc) {
  FD_GUARD_P1W;
  tcmqa_phase1_entry_ws(q, k, v, scratch, anc, *len_dev, M, chunk,
                        q_tok_stride, q_head_stride,
                        k_head_stride, k_pos_stride, v_head_stride, v_pos_stride);
}

#undef FD_GUARD_P1
#undef FD_GUARD_P1W
#undef FD_GUARD_P2
