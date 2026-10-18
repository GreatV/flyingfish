// DSpark Markov-head distribution kernel (tree verification). One launch pair
// serves a batch of P requests; markov_w2 is read once per launch and shared
// by all P acts. Phase 1 uses mma.m16n8k16 bf16 -> fp32.
//
// Semantics (two bf16 rounding points):
//   bias[p][v] = bf16( fp32 dot( w2[v][0:256], w1[req_tok[p]][0:256] ) )
//   s[p][v]    = bias[p][v] + B[req_row[p]][v], BF16 or FP32 per caller binding
//   score      = __bfloat162float(s)
//   top-4[p]   = the 4 largest scores; strictly greater wins, tie -> smaller
//                token id
//   logp[p][j] = (s[p][top4[p][j]] - m) - logsum, where m = max_v s[p][v]
//                and logsum = log(Σ_v exp(s[p][v] - m))
//   lse[p]     = m + logsum
//
// Batch invariance: one request's top4/logp/lse are bitwise identical for
// every batch size P and every batch position. All instances compute bias
// with the same mma instruction, fragment values and k order per (act, v);
// (m, l) partials merge in a canonical order (per 32-v group butterfly offs
// 4/8/16, groups folded ascending); top-4 is a deterministic function of the
// score set; phase 2's merge order does not depend on PT.
//
// Inputs / outputs:
//   B       bf16 [block_rows][V] base logits rows; row per request from req_row
//   w1      bf16 [V][256] (markov_w1); the parent embedding is w1[parent]
//   w2      bf16 [V][256] (markov_w2), row-major, 16B-aligned rows
//   req_tok u32  [P] parent tokens (device)
//   req_row i32  [P] B row indices (device)
//   top4    u32  [P][4] out
//   logp    fp32 [P][4] out
//   lse     fp32 [P] out
//
// Two-phase structure (all smem static; dynamic smem = 0):
//   phase 1 (markov2_top4_phase1_*p{1,8,64}): grid = dim3(NB), block = 256.
//     Block blk owns vocab rows [blk*VT, +VT) (VT = 256, NB = ceil(V/VT))
//     and writes, per act p < P, its local top-4 (4 (score, id) pairs) and
//     its lse partial (m, l) to caller scratch. Slots p >= P are not written.
//   phase 2 (markov2_top4_phase2): grid = dim3(P), block = 256. Reads exactly
//     NB partials per act and writes top4/logp/lse.
//
// Scratch (caller-allocated; PT = compiled max acts of the phase-1 instance):
//   scratch_top4: [NB][PT][4] of (float score, u32 id) -- 8 B each
//     element (blk, p, j) at float offset ((size_t)blk * PT + p) * 8 + j * 2
//   scratch_lse:  [NB][PT][2] fp32 (m, l)
//     element (blk, p) at float offset ((size_t)blk * PT + p) * 2
//   bytes = NB * PT * 40
// Every needed scratch element is written by exactly one phase-1 block.
//
// Every entry point __trap()s on a wrong-shape launch:
//   phase 1: V < 4 || in != 256 || P < 1 || P > PT ||
//            blockDim.x != 256 || gridDim.x != ceil(V/256)
//   phase 2: V < 4 || P < 1 || P > 64 || blockDim.x != 256
//
// CUDA Graph: P and PT are fixed per capture; the request arrays and output
// buffers are pointers fixed per capture whose contents change between
// replays. No device scalars are read; no global state.
#include <cuda_runtime.h>
#include <cuda_bf16.h>

constexpr int MK2_DIM   = 256;
constexpr int MK2_VT    = 256;                  // vocab rows per phase-1 block
constexpr int MK2_NT    = 256;                  // block size, both phases

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float markov_score(float dot, bf16 base, int rounding) {
  const bf16 bias = __float2bfloat16(dot);
  return rounding == 0 ? __bfloat162float(__hadd(bias, base))
                       : __bfloat162float(bias) + __bfloat162float(base);
}

#define MK2_GUARD_P1(PT) do {                                                 \
  if (V < 4 || score_rounding < 0 || score_rounding > 1 || in != MK2_DIM || P < 1 || P > PT ||                  \
      blockDim.x != MK2_NT || gridDim.x != (V + MK2_VT - 1) / MK2_VT) __trap();                \
} while (0)

// Test-only score dump: with -DMK2_DUMP_S and a non-null g_mk2_dump, every
// computed s is also stored at [act*V + v]. Compiles to nothing otherwise.
#ifdef MK2_DUMP_S
__device__ float* g_mk2_dump;
// The host installs g_mk2_dump by launching this kernel.
extern "C" __global__ void mk2_set_dump(float* p) { g_mk2_dump = p; }
#define MK2_DUMP_S_WRITE(act, v, s)                                          \
  do { if (g_mk2_dump) g_mk2_dump[(size_t)(act) * V + (v)] = (s); } while (0)
#else
#define MK2_DUMP_S_WRITE(act, v, s) ((void)0)
#endif

namespace mk2p1 {

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
__device__ __forceinline__ void mma_16816(float (&d)[4], const unsigned (&a)[4],
                                          const unsigned (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void cp16(void* dst, const void* src, int sz) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(smem_u32(dst)), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
__device__ __forceinline__ void cp_wait0() { cp_wait<0>(); }

// Sorted top-4 as a register-resident list, score-desc / id-asc (strictly
// greater wins, tie -> smaller id). Accessed only with literal indices, never
// address-taken; insertion bubbles the candidate through the four slots.
__device__ __forceinline__ void t4_bubble(float& S, unsigned& I, float& s, unsigned& i) {
  const bool better = (s > S) || (s == S && i < I);
  const float ts = S; const unsigned ti = I;
  S = better ? s : S; I = better ? i : I;
  s = better ? ts : s; i = better ? ti : i;
}
struct T4 {
  float s[4]; unsigned i[4];
  __device__ __forceinline__ void init() {
#pragma unroll
    for (int j = 0; j < 4; ++j) { s[j] = -INFINITY; i[j] = 0xFFFFFFFFu; }
  }
  __device__ __forceinline__ void insert(float sv, unsigned iv) {
    t4_bubble(s[0], i[0], sv, iv);
    t4_bubble(s[1], i[1], sv, iv);
    t4_bubble(s[2], i[2], sv, iv);
    t4_bubble(s[3], i[3], sv, iv);
  }
};

// 16B-segment XOR swizzle within a 512 B smem row: logical 16-byte segment
// `seg` of row `row` lives at physical segment (seg ^ (row % 32)).
__device__ __forceinline__ int swz(int seg, int row) { return seg ^ (row & 31); }

// Per-lane running state: top-4 + (m, l) for each of the 2 act columns the
// lane holds under the m16n8 D layout, plus the in-warp merge across the 8
// lanes that share the same two act columns.
struct ActState {
  T4 t[2];
  float m[2], l[2];
  __device__ __forceinline__ void init() {
    t[0].init(); t[1].init();
    m[0] = m[1] = -3.0e38f; l[0] = l[1] = 0.0f;
  }
  template <int Q>
  __device__ __forceinline__ void add(float s, unsigned v) {
    t[Q].insert(s, v);
    const float mn = fmaxf(m[Q], s);
    l[Q] = l[Q] * __expf(m[Q] - mn) + __expf(s - mn);
    m[Q] = mn;
  }
  template <int Q>
  __device__ __forceinline__ void merge_q(int off) {
    float ps[4]; unsigned pi[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      ps[j] = __shfl_xor_sync(0xffffffffu, t[Q].s[j], off);
      pi[j] = __shfl_xor_sync(0xffffffffu, t[Q].i[j], off);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) t[Q].insert(ps[j], pi[j]);
    const float pm = __shfl_xor_sync(0xffffffffu, m[Q], off);
    const float pl = __shfl_xor_sync(0xffffffffu, l[Q], off);
    const float mn = fmaxf(m[Q], pm);
    l[Q] = l[Q] * __expf(m[Q] - mn) + pl * __expf(pm - mn);
    m[Q] = mn;
  }
  __device__ __forceinline__ void merge_lanes() {
#pragma unroll
    for (int off = 4; off <= 16; off <<= 1) {
      merge_q<0>(off);
      merge_q<1>(off);
    }
  }
  // (m, l) only, same off sequence and formula as merge_lanes; the canonical
  // 32-v-group merge.
  __device__ __forceinline__ void merge_lanes_ml() {
#pragma unroll
    for (int off = 4; off <= 16; off <<= 1) {
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const float pm = __shfl_xor_sync(0xffffffffu, m[q], off);
        const float pl = __shfl_xor_sync(0xffffffffu, l[q], off);
        const float mn = fmaxf(m[q], pm);
        l[q] = l[q] * __expf(m[q] - mn) + pl * __expf(pm - mn);
        m[q] = mn;
      }
    }
  }
};

}  // namespace mk2p1

using namespace mk2p1;

// PT = 64. Warp w owns acts [w*8, w*8+8) and sweeps all 256 block rows in
// 16-row chunks. w2 is cp.async-staged through a 4-buffer swizzled pipeline
// in the caller-provided w2sm ([4][16][256] bf16 = 32768 B); the e
// B-fragments are loaded once from gmem into registers.
__device__ __forceinline__ void markov2_p1_body64(
    const bf16* __restrict__ B, const bf16* __restrict__ w1c,
    const bf16* __restrict__ w2c, const unsigned* __restrict__ req_tok,
    const int* __restrict__ req_row, float* __restrict__ scratch_top4,
    float* __restrict__ scratch_lse, int P, int V, int score_rounding, bf16* __restrict__ w2sm) {
  constexpr int PT = 64;
  constexpr int NBUF = 4;                           // w2 chunk buffers
  constexpr int CH = 16;                            // rows per chunk

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int vBlk0 = blockIdx.x * MK2_VT;

  // Per-lane fragment coordinates (m16n8 D layout): rows r0/r0+8 of an
  // m-tile, act columns c0/c0+1 of the warp's n8 tile.
  const int r0 = lane >> 2, c0 = (lane & 3) * 2;
  const int act0 = warp * 8 + c0, act1 = act0 + 1;
  const int rr0 = (act0 < P) ? req_row[act0] : 0;
  const int rr1 = (act1 < P) ? req_row[act1] : 0;
  const bf16* Brow0 = B + (size_t)rr0 * V;
  const bf16* Brow1 = B + (size_t)rr1 * V;

  // Hoist this warp's 16 k16 B-fragments (acts [w*8, w*8+8)) from gmem:
  // b0 = e[act = w*8 + lane/4][kk*16 + (lane%4)*2 +{0,1}], b1 = k + 8.
  unsigned efr[16][2];
  {
    const int arow = warp * 8 + r0;
    const unsigned tok = (arow < P) ? req_tok[arow] : 0u;
    const bf16* er = w1c + (size_t)tok * MK2_DIM + (lane & 3) * 2;
#pragma unroll
    for (int kk = 0; kk < 16; ++kk) {
      efr[kk][0] = *reinterpret_cast<const unsigned*>(er + kk * 16);
      efr[kk][1] = *reinterpret_cast<const unsigned*>(er + kk * 16 + 8);
    }
  }

  // Stage one 16-row w2 chunk into buffer `buf` (512 x 16B = 2 per thread).
  auto stage_w2 = [&](int c, int buf) {
    const bf16* g = w2c + (size_t)(vBlk0 + c * CH) * MK2_DIM;
    bf16* d = w2sm + buf * (CH * MK2_DIM);
#pragma unroll
    for (int rep = 0; rep < 2; ++rep) {
      const int idx = tid + rep * MK2_NT;
      const int r = idx >> 5, seg = idx & 31;
      cp16(d + r * MK2_DIM + swz(seg, r) * 8, g + r * MK2_DIM + seg * 8, vBlk0 + c * CH + r < V ? 16 : 0);
    }
  };

#pragma unroll
  for (int c = 0; c < NBUF - 1; ++c) { stage_w2(c, c); cp_commit(); }

  ActState st;
  st.init();
  // Canonical (m, l): after every 32-v group (two 16-row chunks) the group
  // partial is merged in-warp (merge_lanes_ml) and folded into the block
  // accumulators in ascending-group order by the lanes with r0 == 0.
  float mb[2] = {-3.0e38f, -3.0e38f}, lb[2] = {0.0f, 0.0f};

  for (int c = 0; c < MK2_VT / CH; ++c) {
    cp_wait<NBUF - 2>();          // chunk c resident (empty groups pad the tail)
    __syncthreads();                    // ...and all threads consumed c-1
    if (c + NBUF - 1 < MK2_VT / CH) stage_w2(c + NBUF - 1, (c + NBUF - 1) % NBUF);
    cp_commit();                        // real or empty group: count stays NBUF-1
    const bf16* buf = w2sm + (c % NBUF) * (CH * MK2_DIM);

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int kk = 0; kk < 16; ++kk) {
      unsigned wa[4];
      // A 16x16 at chunk rows [0,16) x k [kk*16,+16):
      // lane -> row lane%16, col kk*16 + (lane/16)*8 (swizzled).
      ldsm_x4(wa, buf + (lane & 15) * MK2_DIM +
                        swz((kk * 2 + (lane >> 4)) & 31, lane & 15) * 8);
      mma_16816(acc, wa, efr[kk]);
    }

    // Two bf16 rounding points, then running top-4 / lse per act column.
    const int vg = vBlk0 + c * CH + r0;
    if (act0 < P && vg < V) {
      const float sv = markov_score(acc[0], Brow0[vg], score_rounding);
      st.add<0>(sv, (unsigned)vg);
      MK2_DUMP_S_WRITE(act0, vg, sv);
    }
    if (act1 < P && vg < V) {
      const float sv = markov_score(acc[1], Brow1[vg], score_rounding);
      st.add<1>(sv, (unsigned)vg);
      MK2_DUMP_S_WRITE(act1, vg, sv);
    }
    if (act0 < P && vg + 8 < V) {
      const float sv = markov_score(acc[2], Brow0[vg + 8], score_rounding);
      st.add<0>(sv, (unsigned)(vg + 8));
      MK2_DUMP_S_WRITE(act0, vg + 8, sv);
    }
    if (act1 < P && vg + 8 < V) {
      const float sv = markov_score(acc[3], Brow1[vg + 8], score_rounding);
      st.add<1>(sv, (unsigned)(vg + 8));
      MK2_DUMP_S_WRITE(act1, vg + 8, sv);
    }
    if (c & 1) {                          // 32-v group complete
      st.merge_lanes_ml();
      if (r0 == 0) {                      // lanes 0..3 fold group c>>1
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          const float mn = fmaxf(mb[q], st.m[q]);
          lb[q] = lb[q] * __expf(mb[q] - mn) + st.l[q] * __expf(st.m[q] - mn);
          mb[q] = mn;
        }
      }
      st.m[0] = st.m[1] = -3.0e38f; st.l[0] = st.l[1] = 0.0f;
    }
  }

  st.merge_lanes();

  // Lanes 0..3 (r0 == 0) hold act columns lane*2, lane*2+1 after the merge.
  if (lane < 4) {
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int act = warp * 8 + lane * 2 + q;
      if (act < P) {
        const size_t b4 = ((size_t)blockIdx.x * PT + act) * 8;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          scratch_top4[b4 + j * 2] = st.t[q].s[j];
          *reinterpret_cast<unsigned*>(scratch_top4 + b4 + j * 2 + 1) = st.t[q].i[j];
        }
        const size_t bl = ((size_t)blockIdx.x * PT + act) * 2;
        scratch_lse[bl] = mb[q];
        scratch_lse[bl + 1] = lb[q];
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(MK2_NT, 3) markov2_top4_phase1_p64(
    const bf16* __restrict__ B, const bf16* __restrict__ w1c,
    const bf16* __restrict__ w2c, const unsigned* __restrict__ req_tok,
    const int* __restrict__ req_row, float* __restrict__ scratch_top4,
    float* __restrict__ scratch_lse, int V, int in, int P, int score_rounding) {
  MK2_GUARD_P1(64);
  __shared__ __align__(16) bf16 w2sm[4 * 16 * MK2_DIM];   // 32768 B swizzled
  markov2_p1_body64(B, w1c, w2c, req_tok, req_row, scratch_top4, scratch_lse,
                    P, V, score_rounding, w2sm);
}

// Small-PT body (PT = 8 / PT = 1): both run the same mma code; e rows p >= P
// are zero-filled and padding acts are never written. Warp w owns v rows
// [blk*256 + w*32, +32) (2 m16 tiles), all acts, and is exactly one 32-v
// group. The 8 v-split warp partials merge through smem in ascending-warp
// order before one warp per act writes scratch.
// Static smem: pool 32768 (caller's) + e 4096 + merge 2560 B.
template <int PTB>
__device__ __forceinline__ void markov2_p1_small_body(
    const bf16* __restrict__ B, const bf16* __restrict__ w1c,
    const bf16* __restrict__ w2c, const unsigned* __restrict__ req_tok,
    const int* __restrict__ req_row, float* __restrict__ scratch_top4,
    float* __restrict__ scratch_lse, int P, int V, int score_rounding, bf16* __restrict__ w2sm) {
  __shared__ __align__(16) bf16 esm[8 * MK2_DIM];         // 4096 B swizzled
  // Merge arrays are indexed by act (0..7), independent of the scratch
  // stride PTB: [8][8], 2560 B.
  __shared__ float wsc[8][8][4];                          // per-warp top4
  __shared__ unsigned wid[8][8][4];
  __shared__ float wml[8][8][2];                          // per-warp (m, l)

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int vBlk0 = blockIdx.x * MK2_VT;

  // Stage the e rows (acts); rows p >= P are zero-filled via src-size 0.
  {
    const int idx = tid;                                  // 8*32 = 256 = NT
    const int p = idx >> 5, seg = idx & 31;
    const unsigned tok = (p < P) ? req_tok[p] : 0u;
    cp16(esm + p * MK2_DIM + swz(seg, p) * 8,
         w1c + (size_t)tok * MK2_DIM + seg * 8, (p < P) ? 16 : 0);
    cp_commit();
    cp_wait0();
    __syncthreads();
  }

  const int r0 = lane >> 2, c0 = (lane & 3) * 2;
  const int act0 = c0, act1 = c0 + 1;
  const int rr0 = (act0 < P) ? req_row[act0] : 0;
  const int rr1 = (act1 < P) ? req_row[act1] : 0;
  const bf16* Brow0 = B + (size_t)rr0 * V;
  const bf16* Brow1 = B + (size_t)rr1 * V;

  ActState st;
  st.init();

  const bf16* erow = esm + (lane & 7) * MK2_DIM;
  const int eseg = lane & 7;

  // Per-warp w2 pipeline: warp w stages its own [32 rows][32 k] chunks into
  // its own double buffer in the pool (2 x [32][32] bf16 = 4096 B per warp);
  // wait_group + __syncwarp per chunk, no block-wide barrier in the sweep.
  // 16B-segment swizzle: seg ^ ((row >> 1) & 3).
  bf16* const myw2 = w2sm + warp * (2 * 32 * 32);

  // Stage chunk c of this warp's slice (rows [w*32, +32), k [c*32, +32)):
  // 32 rows x 4 x 16B = 4 cp.async per lane.
  auto stage_w2 = [&](int c, int buf) {
    const bf16* g = w2c + (size_t)(vBlk0 + warp * 32) * MK2_DIM + c * 32;
    bf16* d = myw2 + buf * (32 * 32);
#pragma unroll
    for (int rep = 0; rep < 4; ++rep) {
      const int idx = lane + rep * 32;
      const int r = idx >> 2, seg = idx & 3;
      cp16(d + r * 32 + ((seg ^ ((r >> 1) & 3)) << 3),
           g + (size_t)r * MK2_DIM + seg * 8, vBlk0 + warp * 32 + r < V ? 16 : 0);
    }
  };
  stage_w2(0, 0);
  cp_commit();

  // Both m-tile accumulators persist across the k-chunk sweep; per acc the
  // k16 slices arrive ascending kk = 0..15.
  float acc[2][4];
#pragma unroll
  for (int t = 0; t < 2; ++t)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[t][j] = 0.0f;

  const int trow = lane & 15;                           // m-tile row in [0,16)
  const int rsw = (trow >> 1) & 3;                      // row swizzle term
  const int trw1 = trow + 16;                           // tile t=1 row
  const int rsw1 = (trw1 >> 1) & 3;

  for (int c = 0; c < 8; ++c) {
    cp_wait0();                           // this lane's chunk-c copies landed
    __syncwarp();                         // ...and the warp's too
    if (c + 1 < 8) { stage_w2(c + 1, (c + 1) & 1); cp_commit(); }
    const bf16* buf = myw2 + (c & 1) * (32 * 32);
#pragma unroll
    for (int kq = 0; kq < 2; ++kq) {      // k16 slices 2c, 2c+1 of the chunk
      const int kk = c * 2 + kq;
      unsigned eb[2], wa[4];
      // B k16 x n8(acts): plain ldmatrix on the e rows.
      ldsm_x2(eb, erow + swz((kk * 2 + (lane >> 3)) & 31, eseg) * 8);
      // A 16x16 tiles at rows [warp*32 + t*16, +16) x k [kk*16, +16):
      // lane -> row lane%16 of the m-tile, k-offset kq*16 + (lane/16)*8.
      const int ks = kq * 2 + (lane >> 4);                // logical 16B seg
      ldsm_x4(wa, buf + trow * 32 + ((ks ^ rsw) << 3));
      mma_16816(acc[0], wa, eb);
      ldsm_x4(wa, buf + trw1 * 32 + ((ks ^ rsw1) << 3));
      mma_16816(acc[1], wa, eb);
    }
  }

  // Epilogue (t = 0 then t = 1): the two bf16 rounding points, then running
  // top-4 / lse per act column.
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    const int vg = vBlk0 + warp * 32 + t * 16 + r0;
    if (act0 < P && vg < V) {
      const float sv = markov_score(acc[t][0], Brow0[vg], score_rounding);
      st.add<0>(sv, (unsigned)vg);
      MK2_DUMP_S_WRITE(act0, vg, sv);
    }
    if (act1 < P && vg < V) {
      const float sv = markov_score(acc[t][1], Brow1[vg], score_rounding);
      st.add<1>(sv, (unsigned)vg);
      MK2_DUMP_S_WRITE(act1, vg, sv);
    }
    if (act0 < P && vg + 8 < V) {
      const float sv = markov_score(acc[t][2], Brow0[vg + 8], score_rounding);
      st.add<0>(sv, (unsigned)(vg + 8));
      MK2_DUMP_S_WRITE(act0, vg + 8, sv);
    }
    if (act1 < P && vg + 8 < V) {
      const float sv = markov_score(acc[t][3], Brow1[vg + 8], score_rounding);
      st.add<1>(sv, (unsigned)(vg + 8));
      MK2_DUMP_S_WRITE(act1, vg + 8, sv);
    }
  }

  st.merge_lanes();

  // Lanes 0..3 hold all 8 acts after the in-warp merge; warp w owns act w's
  // cross-warp merge.
  if (lane < 4) {
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int act = lane * 2 + q;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        wsc[warp][act][j] = st.t[q].s[j];
        wid[warp][act][j] = st.t[q].i[j];
      }
      wml[warp][act][0] = st.m[q];
      wml[warp][act][1] = st.l[q];
    }
  }
  __syncthreads();
  if (lane == 0) {
    const int act = warp;
    T4 t4; t4.init();
    float m = -3.0e38f, l = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
#pragma unroll
      for (int j = 0; j < 4; ++j) t4.insert(wsc[w][act][j], wid[w][act][j]);
      const float bm = wml[w][act][0], bl = wml[w][act][1];
      const float mn = fmaxf(m, bm);
      l = l * __expf(m - mn) + bl * __expf(bm - mn);
      m = mn;
    }
    if (act < P) {
      const size_t b4 = ((size_t)blockIdx.x * PTB + act) * 8;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        scratch_top4[b4 + j * 2] = t4.s[j];
        *reinterpret_cast<unsigned*>(scratch_top4 + b4 + j * 2 + 1) = t4.i[j];
      }
      const size_t bl = ((size_t)blockIdx.x * PTB + act) * 2;
      scratch_lse[bl] = m;
      scratch_lse[bl + 1] = l;
    }
  }
}

template <int PTB>
__device__ __forceinline__ void markov2_p1_small_body_preF1(
    const bf16* __restrict__ B, const bf16* __restrict__ w1c,
    const bf16* __restrict__ w2c, const unsigned* __restrict__ req_tok,
    const int* __restrict__ req_row, float* __restrict__ scratch_top4,
    float* __restrict__ scratch_lse, int P, int V, int score_rounding) {
  __shared__ __align__(16) bf16 esm[8 * MK2_DIM];         // 4096 B swizzled
  // Merge arrays are indexed by act (0..7), independent of the scratch
  // stride PTB: [8][8], 2560 B.
  __shared__ float wsc[8][8][4];                          // per-warp top4
  __shared__ unsigned wid[8][8][4];
  __shared__ float wml[8][8][2];                          // per-warp (m, l)

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int vBlk0 = blockIdx.x * MK2_VT;

  // Stage the e rows (acts); rows p >= P are zero-filled via src-size 0.
  {
    const int idx = tid;                                  // 8*32 = 256 = NT
    const int p = idx >> 5, seg = idx & 31;
    const unsigned tok = (p < P) ? req_tok[p] : 0u;
    cp16(esm + p * MK2_DIM + swz(seg, p) * 8,
         w1c + (size_t)tok * MK2_DIM + seg * 8, (p < P) ? 16 : 0);
    cp_commit();
    cp_wait0();
    __syncthreads();
  }

  const int r0 = lane >> 2, c0 = (lane & 3) * 2;
  const int act0 = c0, act1 = c0 + 1;
  const int rr0 = (act0 < P) ? req_row[act0] : 0;
  const int rr1 = (act1 < P) ? req_row[act1] : 0;
  const bf16* Brow0 = B + (size_t)rr0 * V;
  const bf16* Brow1 = B + (size_t)rr1 * V;

  ActState st;
  st.init();

  const bf16* erow = esm + (lane & 7) * MK2_DIM;
  const int eseg = lane & 7;

  // gmem A-fragment row bases: lane covers rows vt0 + r0 and vt0 + r0 + 8,
  // column pairs c0k = (lane%4)*2 (+8 for a2/a3) of each k16 slice.
  const int coff = (lane & 3) * 2;

#pragma unroll
  for (int t = 0; t < 2; ++t) {
    const int vt0 = vBlk0 + warp * 32 + t * 16;
    const bf16* rA = w2c + (size_t)(vt0 + r0) * MK2_DIM + coff;
    const bf16* rB_ = rA + 8 * MK2_DIM;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int kk = 0; kk < 16; ++kk) {
      unsigned wa[4], eb[2];
      wa[0] = vt0 + r0 < V ? *reinterpret_cast<const unsigned*>(rA + kk * 16) : 0u;
      wa[1] = vt0 + r0 + 8 < V ? *reinterpret_cast<const unsigned*>(rB_ + kk * 16) : 0u;
      wa[2] = vt0 + r0 < V ? *reinterpret_cast<const unsigned*>(rA + kk * 16 + 8) : 0u;
      wa[3] = vt0 + r0 + 8 < V ? *reinterpret_cast<const unsigned*>(rB_ + kk * 16 + 8) : 0u;
      ldsm_x2(eb, erow + swz((kk * 2 + (lane >> 3)) & 31, eseg) * 8);
      mma_16816(acc, wa, eb);
    }

    const int vg = vt0 + r0;
    if (act0 < P && vg < V) {
      const float sv = markov_score(acc[0], Brow0[vg], score_rounding);
      st.add<0>(sv, (unsigned)vg);
      MK2_DUMP_S_WRITE(act0, vg, sv);
    }
    if (act1 < P && vg < V) {
      const float sv = markov_score(acc[1], Brow1[vg], score_rounding);
      st.add<1>(sv, (unsigned)vg);
      MK2_DUMP_S_WRITE(act1, vg, sv);
    }
    if (act0 < P && vg + 8 < V) {
      const float sv = markov_score(acc[2], Brow0[vg + 8], score_rounding);
      st.add<0>(sv, (unsigned)(vg + 8));
      MK2_DUMP_S_WRITE(act0, vg + 8, sv);
    }
    if (act1 < P && vg + 8 < V) {
      const float sv = markov_score(acc[3], Brow1[vg + 8], score_rounding);
      st.add<1>(sv, (unsigned)(vg + 8));
      MK2_DUMP_S_WRITE(act1, vg + 8, sv);
    }
  }

  st.merge_lanes();

  // Lanes 0..3 hold all 8 acts after the in-warp merge; warp w owns act w's
  // cross-warp merge.
  if (lane < 4) {
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int act = lane * 2 + q;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        wsc[warp][act][j] = st.t[q].s[j];
        wid[warp][act][j] = st.t[q].i[j];
      }
      wml[warp][act][0] = st.m[q];
      wml[warp][act][1] = st.l[q];
    }
  }
  __syncthreads();
  if (lane == 0) {
    const int act = warp;
    T4 t4; t4.init();
    float m = -3.0e38f, l = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
#pragma unroll
      for (int j = 0; j < 4; ++j) t4.insert(wsc[w][act][j], wid[w][act][j]);
      const float bm = wml[w][act][0], bl = wml[w][act][1];
      const float mn = fmaxf(m, bm);
      l = l * __expf(m - mn) + bl * __expf(bm - mn);
      m = mn;
    }
    if (act < P) {
      const size_t b4 = ((size_t)blockIdx.x * PTB + act) * 8;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        scratch_top4[b4 + j * 2] = t4.s[j];
        *reinterpret_cast<unsigned*>(scratch_top4 + b4 + j * 2 + 1) = t4.i[j];
      }
      const size_t bl = ((size_t)blockIdx.x * PTB + act) * 2;
      scratch_lse[bl] = m;
      scratch_lse[bl + 1] = l;
    }
  }
}

extern "C" __global__ void __launch_bounds__(MK2_NT, 2) markov2_top4_phase1_p8(
    const bf16* __restrict__ B, const bf16* __restrict__ w1c,
    const bf16* __restrict__ w2c, const unsigned* __restrict__ req_tok,
    const int* __restrict__ req_row, float* __restrict__ scratch_top4,
    float* __restrict__ scratch_lse, int V, int in, int P, int score_rounding) {
  MK2_GUARD_P1(8);
  __shared__ __align__(16) bf16 w2sm[4 * 16 * MK2_DIM];
  markov2_p1_small_body<8>(B, w1c, w2c, req_tok, req_row, scratch_top4,
                           scratch_lse, P, V, score_rounding, w2sm);
}

// PT = 1: the small body with scratch stride 1.
extern "C" __global__ void __launch_bounds__(MK2_NT, 2) markov2_top4_phase1_p1(
    const bf16* __restrict__ B, const bf16* __restrict__ w1c,
    const bf16* __restrict__ w2c, const unsigned* __restrict__ req_tok,
    const int* __restrict__ req_row, float* __restrict__ scratch_top4,
    float* __restrict__ scratch_lse, int V, int in, int P, int score_rounding) {
  MK2_GUARD_P1(1);
  __shared__ __align__(16) bf16 w2sm[4 * 16 * MK2_DIM];
  markov2_p1_small_body<1>(B, w1c, w2c, req_tok, req_row, scratch_top4,
                           scratch_lse, P, V, score_rounding, w2sm);
}

// The small body with direct 32-bit global loads of w2 in place of the
// cp.async stages. Same guards and launch contract as markov2_top4_phase1_p{8,1};
// outputs are bitwise identical to them.
extern "C" __global__ void __launch_bounds__(MK2_NT, 2) markov2_top4_phase1_preF1_p8(
    const bf16* __restrict__ B, const bf16* __restrict__ w1c,
    const bf16* __restrict__ w2c, const unsigned* __restrict__ req_tok,
    const int* __restrict__ req_row, float* __restrict__ scratch_top4,
    float* __restrict__ scratch_lse, int V, int in, int P, int score_rounding) {
  MK2_GUARD_P1(8);
  markov2_p1_small_body_preF1<8>(B, w1c, w2c, req_tok, req_row, scratch_top4,
                                 scratch_lse, P, V, score_rounding);
}

extern "C" __global__ void __launch_bounds__(MK2_NT, 2) markov2_top4_phase1_preF1_p1(
    const bf16* __restrict__ B, const bf16* __restrict__ w1c,
    const bf16* __restrict__ w2c, const unsigned* __restrict__ req_tok,
    const int* __restrict__ req_row, float* __restrict__ scratch_top4,
    float* __restrict__ scratch_lse, int V, int in, int P, int score_rounding) {
  MK2_GUARD_P1(1);
  markov2_p1_small_body_preF1<1>(B, w1c, w2c, req_tok, req_row, scratch_top4,
                                 scratch_lse, P, V, score_rounding);
}

#undef MK2_GUARD_P1

// Per act p = blockIdx.x: thread t folds phase-1 partials b in {t, t+256} &
// [0,510) into a register-resident T4 and an online (m, l); a 5-round shfl
// butterfly merges each warp's lanes; warp partials cross smem and warp 0
// merges the 8 lists with a 3-round butterfly. Lane 0 of warp 0 writes
// lse[p], top4[p][*] and logp[p][*].
__device__ __forceinline__ void markov2_p2_body(
    const float* __restrict__ scratch_top4, const float* __restrict__ scratch_lse,
    unsigned* __restrict__ top4, float* __restrict__ logp, float* __restrict__ lse,
    int P, int PT, int V) {
  __shared__ float ws_sc[8][4];                  // per-warp top4 partials
  __shared__ unsigned ws_id[8][4];
  __shared__ float ws_ml[8][2];                  // per-warp (m, l)

  const int p = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;

  T4 t4; t4.init();
  float m = -3.0e38f, l = 0.0f;
#pragma unroll
  for (int b = tid; b < (V + MK2_VT - 1) / MK2_VT; b += MK2_NT) {
    const float4 qa = *reinterpret_cast<const float4*>(scratch_top4 + ((size_t)b * PT + p) * 8);
    const float4 qb = *reinterpret_cast<const float4*>(scratch_top4 + ((size_t)b * PT + p) * 8 + 4);
    t4.insert(qa.x, __float_as_uint(qa.y));
    t4.insert(qa.z, __float_as_uint(qa.w));
    t4.insert(qb.x, __float_as_uint(qb.y));
    t4.insert(qb.z, __float_as_uint(qb.w));
    const float2 ml = *reinterpret_cast<const float2*>(scratch_lse + ((size_t)b * PT + p) * 2);
    const float mn = fmaxf(m, ml.x);
    l = l * __expf(m - mn) + ml.y * __expf(ml.x - mn);
    m = mn;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    float ps[4]; unsigned pi[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      ps[j] = __shfl_xor_sync(0xffffffffu, t4.s[j], off);
      pi[j] = __shfl_xor_sync(0xffffffffu, t4.i[j], off);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) t4.insert(ps[j], pi[j]);
    const float pm = __shfl_xor_sync(0xffffffffu, m, off);
    const float pl = __shfl_xor_sync(0xffffffffu, l, off);
    const float mn = fmaxf(m, pm);
    l = l * __expf(m - mn) + pl * __expf(pm - mn);
    m = mn;
  }
  if (lane == 0) {
#pragma unroll
    for (int j = 0; j < 4; ++j) { ws_sc[warp][j] = t4.s[j]; ws_id[warp][j] = t4.i[j]; }
    ws_ml[warp][0] = m; ws_ml[warp][1] = l;
  }
  __syncthreads();
  if (warp == 0) {
    T4 w4; w4.init();
    float wm = -3.0e38f, wl = 0.0f;
    if (lane < 8) {
#pragma unroll
      for (int j = 0; j < 4; ++j) { w4.s[j] = ws_sc[lane][j]; w4.i[j] = ws_id[lane][j]; }
      wm = ws_ml[lane][0]; wl = ws_ml[lane][1];
    }
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
      float ps[4]; unsigned pi[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        ps[j] = __shfl_xor_sync(0xffffffffu, w4.s[j], off);
        pi[j] = __shfl_xor_sync(0xffffffffu, w4.i[j], off);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) w4.insert(ps[j], pi[j]);
      const float pm = __shfl_xor_sync(0xffffffffu, wm, off);
      const float pl = __shfl_xor_sync(0xffffffffu, wl, off);
      const float mn = fmaxf(wm, pm);
      wl = wl * __expf(wm - mn) + pl * __expf(pm - mn);
      wm = mn;
    }
    if (lane == 0) {
      const float logsum = logf(wl);
      lse[p] = wm + logsum;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        top4[(size_t)p * 4 + j] = w4.i[j];
        // (score - m) - logsum; logsum is not rounded into lse first.
        logp[(size_t)p * 4 + j] = (w4.s[j] - wm) - logsum;
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(MK2_NT) markov2_top4_phase2(
    const float* __restrict__ scratch_top4, const float* __restrict__ scratch_lse,
    unsigned* __restrict__ top4, float* __restrict__ logp, float* __restrict__ lse,
    int V, int P, int PT) {
  if (V < 4 || P < 1 || P > 64 || blockDim.x != MK2_NT) __trap();
  markov2_p2_body(scratch_top4, scratch_lse, top4, logp, lse, P, PT, V);
}

extern "C" __global__ void sum_argmax(const bf16* base, const bf16* bias,
                                       void* scores, unsigned* token,
                                       int vocab, int rounding) {
  if (blockDim.x != 1024 || vocab < 4 || rounding < 0 || rounding > 1) __trap();
  __shared__ float values[32];
  __shared__ unsigned ids[32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  float best = -__int_as_float(0x7f800000);
  unsigned id = 0xffffffffu;
  const bool aligned = ((reinterpret_cast<size_t>(base) | reinterpret_cast<size_t>(bias) |
                         reinterpret_cast<size_t>(scores)) & 15u) == 0;
  size_t scalar_begin = 0;
  if (aligned) {
    union Pack { uint4 words; bf16 values[8]; };
    const int vectors = vocab / 8;
    for (int i = tid; i < vectors; i += 1024) {
      Pack a, b, rounded;
      a.words = reinterpret_cast<const uint4*>(base)[i];
      b.words = reinterpret_cast<const uint4*>(bias)[i];
      float combined[8];
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        const unsigned token_id = unsigned(i) * 8u + unsigned(j);
        const float score = markov_score(__bfloat162float(b.values[j]), a.values[j], rounding);
        combined[j] = score;
        rounded.values[j] = __float2bfloat16(score);
        if (score > best || (score == best && token_id < id)) { best = score; id = token_id; }
      }
      if (scores && rounding == 0) reinterpret_cast<uint4*>(scores)[i] = rounded.words;
      else if (scores) {
        reinterpret_cast<float4*>(scores)[2 * i] = make_float4(combined[0],combined[1],combined[2],combined[3]);
        reinterpret_cast<float4*>(scores)[2 * i + 1] = make_float4(combined[4],combined[5],combined[6],combined[7]);
      }
    }
    scalar_begin = size_t(vectors) * 8;
  }
  for (size_t v = scalar_begin + tid; v < size_t(vocab); v += 1024) {
    const float score = markov_score(__bfloat162float(bias[v]), base[v], rounding);
    if (scores && rounding == 0) static_cast<bf16*>(scores)[v] = __float2bfloat16(score);
    else if (scores) static_cast<float*>(scores)[v] = score;
    if (score > best || (score == best && (unsigned)v < id)) { best = score; id = unsigned(v); }
  }
  for (int offset = 16; offset; offset >>= 1) {
    const float other = __shfl_down_sync(0xffffffffu,best,offset);
    const unsigned other_id = __shfl_down_sync(0xffffffffu,id,offset);
    if (lane + offset < 32 && (other > best || (other == best && other_id < id))) { best = other; id = other_id; }
  }
  if (lane == 0) { values[warp] = best; ids[warp] = id; }
  __syncthreads();
  if (warp == 0) {
    best = values[lane]; id = ids[lane];
    for (int offset = 16; offset; offset >>= 1) {
      const float other = __shfl_down_sync(0xffffffffu,best,offset);
      const unsigned other_id = __shfl_down_sync(0xffffffffu,id,offset);
      if (lane + offset < 32 && (other > best || (other == best && other_id < id))) { best = other; id = other_id; }
    }
    if (lane == 0) *token = id;
  }
}

extern "C" __global__ void concat_dot_sigmoid(const bf16* hidden, const bf16* embeddings,
                                               const unsigned* anchor, const unsigned* previous,
                                               const bf16* weight, const bf16* bias,
                                               float* output, int rows, int width, int rank,
                                               int vocab) {
  if (blockDim.x != 256 || rows < 1 || rows > 64 || width < 1 || rank < 1) __trap();
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (row >= rows) return;
  const unsigned parent = row == 0 ? *anchor : previous[row - 1];
  if (parent >= (unsigned)vocab) __trap();
  float sum = 0.0f;
  for (int i = tid; i < width + rank; i += 256) {
    const bf16 value = i < width ? hidden[(size_t)row * width + i]
                                 : embeddings[(size_t)parent * rank + i - width];
    sum += __bfloat162float(value) * __bfloat162float(weight[i]);
  }
  for (int offset = 16; offset; offset >>= 1) sum += __shfl_down_sync(0xffffffffu,sum,offset);
  __shared__ float partials[8];
  if (lane == 0) partials[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    sum = lane < 8 ? partials[lane] : 0.0f;
    for (int offset = 16; offset; offset >>= 1) sum += __shfl_down_sync(0xffffffffu,sum,offset);
    if (lane == 0) {
      const float logit = __bfloat162float(__float2bfloat16(sum + __bfloat162float(*bias)));
      output[row] = __bfloat162float(__float2bfloat16(1.0f / (1.0f + expf(-logit))));
    }
  }
}
