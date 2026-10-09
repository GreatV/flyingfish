// ============================================================================
// markov.cu -- DSpark Markov-head distribution kernel (tree verification).
// One launch pair serves a batch of P requests; the markov_w2 table is read
// exactly once per launch and shared by all P acts. All phase-1 instances
// use tensor cores (mma.m16n8k16 bf16 -> fp32).
//
// Semantics (two bf16 rounding points):
//   bias[p][v] = bf16( fp32 dot( w2[v][0:256], w1[req_tok[p]][0:256] ) )
//   s[p][v]    = __hadd(bias[p][v], B[req_row[p]][v])           (bf16 add)
//   score      = __bfloat162float(s)
//   top-4[p]   = the 4 largest scores; strictly greater wins, tie -> smaller
//                token id
//   lse[p]     = fp32 logsumexp over the full vocab of s[p]
//   logp[p][j] = (s[p][top4[p][j]] - m) - logsum, where m = max_v s[p][v]
//                and logsum = log(Σ_v exp(s[p][v] - m))  (stable form)
//   lse[p]     = m + logsum
//
// BATCH INVARIANCE: one request's top4/logp/lse are bitwise identical for
// every batch size P and every batch position. All instances compute bias
// with the same mma instruction, the same fragment values and the same k
// order per (act, v); the lse (m, l) partials are merged in a canonical
// order in every instance (per 32-v group butterfly offs 4/8/16, groups
// folded ascending); top-4 is a deterministic function of the score set
// (insertion order does not matter); phase 2's merge order does not depend
// on PT.
//
// Inputs / outputs:
//   B       bf16 [7][V] base logits rows; row per request from req_row
//   w1      bf16 [V][256] (markov_w1); the parent embedding is w1[parent]
//   w2      bf16 [V][256] (markov_w2), row-major, 16B-aligned rows
//   req_tok u32  [P] parent tokens (device)
//   req_row i32  [P] B row indices (device)
//   top4    u32  [P][4] out
//   logp    fp32 [P][4] out
//   lse     fp32 [P] out
//
// ABI CHANGE relative to the pre-v2 revision of this file: symbols renamed
// markov_top4_phase1_p{1,8,64} -> markov2_top4_phase1_p{1,8,64} and
// markov_top4_phase2 -> markov2_top4_phase2; phase-1 launch is now
// grid = dim3(510), block = 256 with all shared memory static (dynamic
// smem = 0, no cudaFuncSetAttribute opt-in) instead of grid = dim3(16320)
// with dynamic smem; phase-2 launch is now grid = dim3(P), block = 256,
// static smem (dynamic smem = 0) instead of block = 1024 with 40960 B
// dynamic; caller scratch shrinks to NB*PT*40 bytes with NB = 510 (was
// 16320). Argument lists and math semantics are unchanged.
//
// Two-phase structure:
//   phase 1 (markov2_top4_phase1_p{1,8,64}): grid = dim3(NB), block = 256.
//     Block blk owns vocab rows [blk*VT, +VT) (VT = 256, NB = V/VT = 510)
//     and writes, per act p < P, its local top-4 (4 (score, id) pairs) and
//     its lse partial (m, l) to caller scratch. Slots p >= P are not written.
//   phase 2 (markov2_top4_phase2): grid = dim3(P), block = 256. Each thread
//     folds two of the NB block partials per act (vectorized reads,
//     score-desc/id-asc merge; lse rescale), 5-round in-warp shfl
//     butterflies, one __syncthreads, then warp 0 merges the 8 warp lists
//     and writes top4/logp/lse. Reads exactly NB partials per act. The
//     lse/logp merge association differs from the previous (smem halving
//     tree) revision: top-4 ids are unchanged (deterministic total order),
//     lse/logp can differ in the last fp32 ulp.
//
// Scratch (caller-allocated; PT = compiled max acts of the phase-1 instance):
//   scratch_top4: [NB][PT][4] of (float score, u32 id) -- 8 B each
//     element (blk, p, j) at float offset ((size_t)blk * PT + p) * 8 + j * 2
//   scratch_lse:  [NB][PT][2] fp32 (m, l)
//     element (blk, p) at float offset ((size_t)blk * PT + p) * 2
//   bytes = NB * PT * 40  (510*64*40 = 1305600 B at PT = 64)
// Every needed scratch element is written by exactly one phase-1 block.
//
// FD_GUARD: every entry point calls __trap() on a wrong-shape launch:
//   phase 1: V != 130560 || in != 256 || P < 1 || P > PT ||
//            blockDim.x != 256 || gridDim.x != 510
//   phase 2: V != 130560 || P < 1 || P > 64 || blockDim.x != 256
//
// CUDA Graph: P and PT are ordinary kernel arguments, fixed per capture; the
// request arrays and output buffers are pointers fixed per capture whose
// CONTENTS the engine updates between replays. No device scalars are read;
// no global state.
//
// SIGNATURE TABLE (extern "C" __global__; all pointer args __restrict__):
//   markov2_top4_phase1_p{1,8,64}   -- PT = compiled max acts
//     void(const __nv_bfloat16* B, const __nv_bfloat16* w1,
//          const __nv_bfloat16* w2, const unsigned* req_tok,
//          const int* req_row, float* scratch_top4, float* scratch_lse,
//          int V, int in, int P)
//     launch: grid = dim3(510), block = 256, dynamic smem = 0
//   markov2_top4_phase2
//     void(const float* scratch_top4, const float* scratch_lse,
//          unsigned* top4, float* logp, float* lse, int V, int P, int PT)
//     launch: grid = dim3(P), block = 256, dynamic smem = 0
// ============================================================================
#include <cuda_runtime.h>
#include <cuda_bf16.h>

constexpr int MK2_VOCAB = 130560;
constexpr int MK2_DIM   = 256;
constexpr int MK2_VT    = 256;                  // vocab rows per phase-1 block
constexpr int MK2_NB2   = MK2_VOCAB / MK2_VT;   // 510 phase-1 blocks
constexpr int MK2_NT    = 256;                  // block size, both phases

using bf16 = __nv_bfloat16;

#define MK2_GUARD_P1(PT) do {                                                 \
  if (V != MK2_VOCAB || in != MK2_DIM || P < 1 || P > PT ||                  \
      blockDim.x != MK2_NT || gridDim.x != MK2_NB2) __trap();                \
} while (0)

// Test-only score dump: with -DMK2_DUMP_S and a non-null g_mk2_dump, every
// computed s is also stored at [act*V + v]. Compiles to nothing otherwise.
#ifdef MK2_DUMP_S
__device__ float* g_mk2_dump;
// g_mk2_dump lives in this TU; the host installs the pointer by launching
// this kernel (cudaMemcpyToSymbol across TUs fails without -rdc).
extern "C" __global__ void mk2_set_dump(float* p) { g_mk2_dump = p; }
#define MK2_DUMP_S_WRITE(act, v, s)                                          \
  do { if (g_mk2_dump) g_mk2_dump[(size_t)(act) * MK2_VOCAB + (v)] = (s); } while (0)
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
// greater wins, tie -> smaller id); same comparator as v1's top4_insert.
// The list is only ever accessed with literal indices and never address-
// taken (passing float* to an insert helper demotes it to local memory);
// insertion bubbles the candidate through the four slots, which is
// equivalent to v1's shift-insert.
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
  // (m, l) only, same off sequence and formula as merge_lanes. Used for the
  // canonical 32-v-group merge: identical inputs produce identical bits in
  // every instance.
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

// =========================================================== PT = 64 =======
// Warp w owns acts [w*8, w*8+8) and sweeps all 256 block rows in 16-row
// chunks. w2 is cp.async-staged through a 4-buffer swizzled smem pipeline
// ([4][16][256] = 32768 B static smem). DEVIATION from "e rows staged once
// in smem": the e B-fragments are loaded straight from gmem into registers
// once per launch (32 x LDG.32 per lane; the 64 e rows are a 32 KB
// L2-resident working set, so w1 DRAM traffic is still each row once per
// launch, and the sector pairs are fully consumed through L1). Reason:
// the 48 KB static-smem budget cannot hold both e[64][256] (32 KB) and a
// useful w2 pipeline; with e in smem the block is capped at 2 blocks/SM
// (33% occupancy) and measures ~57% of DRAM peak on A4000 (latency-bound,
// phase-synchronized staging bursts). At 32768 B/block three blocks are
// resident per SM (24 warps), which restores DRAM saturation. Dynamic smem
// with an opt-in is not an option: the fixed ABI launch contract does not
// document one.
// Body shared by the fixed-P export and the device-P dispatcher (devp below).
// The w2 pipeline buffer is caller-provided (16384 bf16 = 32768 B) so the
// devp kernel can share one allocation across the three body branches.
__device__ __forceinline__ void markov2_p1_body64(
    const bf16* __restrict__ B, const bf16* __restrict__ w1c,
    const bf16* __restrict__ w2c, const unsigned* __restrict__ req_tok,
    const int* __restrict__ req_row, float* __restrict__ scratch_top4,
    float* __restrict__ scratch_lse, int P, bf16* __restrict__ w2sm) {
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
  const bf16* Brow0 = B + (size_t)rr0 * MK2_VOCAB;
  const bf16* Brow1 = B + (size_t)rr1 * MK2_VOCAB;

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
      cp16(d + r * MK2_DIM + swz(seg, r) * 8, g + r * MK2_DIM + seg * 8, 16);
    }
  };

#pragma unroll
  for (int c = 0; c < NBUF - 1; ++c) { stage_w2(c, c); cp_commit(); }

  ActState st;
  st.init();
  // Canonical (m, l): after every 32-v group (two 16-row chunks) the group
  // partial is merged in-warp (merge_lanes_ml, offs 4/8/16) and folded into
  // the block accumulators in ascending-group order by the lanes with
  // r0 == 0. A lane holds the same four v of a group in every instance, so
  // the block (m, l) partial is bitwise identical across instances.
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
    if (act0 < P) {
      const bf16 sv = __hadd(__float2bfloat16(acc[0]), Brow0[vg]);
      st.add<0>(__bfloat162float(sv), (unsigned)vg);
      MK2_DUMP_S_WRITE(act0, vg, __bfloat162float(sv));
    }
    if (act1 < P) {
      const bf16 sv = __hadd(__float2bfloat16(acc[1]), Brow1[vg]);
      st.add<1>(__bfloat162float(sv), (unsigned)vg);
      MK2_DUMP_S_WRITE(act1, vg, __bfloat162float(sv));
    }
    if (act0 < P) {
      const bf16 sv = __hadd(__float2bfloat16(acc[2]), Brow0[vg + 8]);
      st.add<0>(__bfloat162float(sv), (unsigned)(vg + 8));
      MK2_DUMP_S_WRITE(act0, vg + 8, __bfloat162float(sv));
    }
    if (act1 < P) {
      const bf16 sv = __hadd(__float2bfloat16(acc[3]), Brow1[vg + 8]);
      st.add<1>(__bfloat162float(sv), (unsigned)(vg + 8));
      MK2_DUMP_S_WRITE(act1, vg + 8, __bfloat162float(sv));
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
    float* __restrict__ scratch_lse, int V, int in, int P) {
  MK2_GUARD_P1(64);
  __shared__ __align__(16) bf16 w2sm[4 * 16 * MK2_DIM];   // 32768 B swizzled
  markov2_p1_body64(B, w1c, w2c, req_tok, req_row, scratch_top4, scratch_lse,
                    P, w2sm);
}

// ==================================================== PT = 8 / PT = 1 ======
// Shared tensor-core body for the small-PT instances. PT = 1 runs the same
// mma code as PT = 8 (e rows p >= P are zero-filled, padding acts are never
// written), so one request's bias and s bits are identical whether it runs
// alone or inside a batch. Warp w owns v rows [blk*256 + w*32, +32) (2 m16
// tiles), all acts. The w2 A-operands are cp.async-staged PER WARP: warp w
// streams its own 32-row slice in [32 rows][32 k] chunks through its own
// double buffer inside the caller-provided 32768 B pool (8 warps x 2 x
// [32][32] x 2 B), wait_group + __syncwarp per chunk (the gemm_skinny
// per-warp staging idiom) -- there is no block-wide barrier in the sweep,
// so warps free-run. 16B-segment swizzle seg ^ ((row >> 1) & 3) keeps the
// ldmatrix reads bank-conflict-free; A operands via ldmatrix.x4 on the
// warp's own two m16 tiles. Per-acc k order stays kk = 0..15 ascending with
// the same fragment values as the former gmem-direct loads, so outputs are
// bitwise identical. w2 DRAM traffic is exactly V*256*2 B once per launch.
// e rows are staged in smem (4096 B swizzled) via ldmatrix.x2, one ldsm per
// k16 slice shared by both tiles. Each warp covers exactly one 32-v group:
// its (m, l) partial after the in-warp butterfly is the canonical group
// partial; the 8 v-split warp partials merge through smem in ascending-warp
// order before one warp per act writes scratch.
// Static smem: pool 32768 (caller's) + e 4096 + merge 2560 B.
template <int PTB>
__device__ __forceinline__ void markov2_p1_small_body(
    const bf16* __restrict__ B, const bf16* __restrict__ w1c,
    const bf16* __restrict__ w2c, const unsigned* __restrict__ req_tok,
    const int* __restrict__ req_row, float* __restrict__ scratch_top4,
    float* __restrict__ scratch_lse, int P, bf16* __restrict__ w2sm) {
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
  const bf16* Brow0 = B + (size_t)rr0 * MK2_VOCAB;
  const bf16* Brow1 = B + (size_t)rr1 * MK2_VOCAB;

  ActState st;
  st.init();

  const bf16* erow = esm + (lane & 7) * MK2_DIM;
  const int eseg = lane & 7;

  // Per-warp w2 pipeline: warp w stages its own 32 rows x 32 k chunks into
  // its own double buffer inside the pool (2 x [32][32] bf16 = 4096 B per
  // warp, 32768 B total); cp.async per lane + wait_group + __syncwarp (the
  // gemm_skinny per-warp idiom) -- no block-wide barrier in the sweep, so
  // warps free-run. 16B-segment swizzle seg ^ ((row >> 1) & 3) keeps the
  // ldmatrix reads bank-conflict-free. Same fragment values and ascending
  // per-acc k order as the former gmem-direct loads -> bitwise-identical.
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
           g + (size_t)r * MK2_DIM + seg * 8, 16);
    }
  };
  stage_w2(0, 0);
  cp_commit();

  // Both m-tile accumulators persist across the k-chunk sweep; per acc the
  // k16 slices arrive ascending kk = 0..15 with the same fragment values as
  // the former gmem-direct A loads.
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
      // B k16 x n8(acts): plain ldmatrix on the e rows (unchanged path).
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

  // Epilogue (t = 0 then t = 1, same per-lane add order as before): the two
  // bf16 rounding points, then running top-4 / lse per act column.
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    const int vg = vBlk0 + warp * 32 + t * 16 + r0;
    if (act0 < P) {
      const bf16 sv = __hadd(__float2bfloat16(acc[t][0]), Brow0[vg]);
      st.add<0>(__bfloat162float(sv), (unsigned)vg);
      MK2_DUMP_S_WRITE(act0, vg, __bfloat162float(sv));
    }
    if (act1 < P) {
      const bf16 sv = __hadd(__float2bfloat16(acc[t][1]), Brow1[vg]);
      st.add<1>(__bfloat162float(sv), (unsigned)vg);
      MK2_DUMP_S_WRITE(act1, vg, __bfloat162float(sv));
    }
    if (act0 < P) {
      const bf16 sv = __hadd(__float2bfloat16(acc[t][2]), Brow0[vg + 8]);
      st.add<0>(__bfloat162float(sv), (unsigned)(vg + 8));
      MK2_DUMP_S_WRITE(act0, vg + 8, __bfloat162float(sv));
    }
    if (act1 < P) {
      const bf16 sv = __hadd(__float2bfloat16(acc[t][3]), Brow1[vg + 8]);
      st.add<1>(__bfloat162float(sv), (unsigned)(vg + 8));
      MK2_DUMP_S_WRITE(act1, vg + 8, __bfloat162float(sv));
    }
  }

  st.merge_lanes();

  // v is split across the 8 warps (nvs = 8): stage each warp's 32-row
  // partials (lanes 0..3 hold all 8 acts after the in-warp merge) and merge
  // across warps through smem. Warp w then owns act w's block merge.
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
    float* __restrict__ scratch_lse, int V, int in, int P) {
  MK2_GUARD_P1(8);
  // 32768 B pool for the small body's w2 pipeline (sized to share one
  // allocation with body64 inside the devp dispatcher).
  __shared__ __align__(16) bf16 w2sm[4 * 16 * MK2_DIM];
  markov2_p1_small_body<8>(B, w1c, w2c, req_tok, req_row, scratch_top4,
                           scratch_lse, P, w2sm);
}

// ============================================================ PT = 1 =======
// The same tensor-core body instantiated with scratch stride 1: a request
// produces bitwise-identical top4/logp/lse whether it runs alone or inside
// a batch.
extern "C" __global__ void __launch_bounds__(MK2_NT, 2) markov2_top4_phase1_p1(
    const bf16* __restrict__ B, const bf16* __restrict__ w1c,
    const bf16* __restrict__ w2c, const unsigned* __restrict__ req_tok,
    const int* __restrict__ req_row, float* __restrict__ scratch_top4,
    float* __restrict__ scratch_lse, int V, int in, int P) {
  MK2_GUARD_P1(1);
  __shared__ __align__(16) bf16 w2sm[4 * 16 * MK2_DIM];   // shared pool (as p8)
  markov2_p1_small_body<1>(B, w1c, w2c, req_tok, req_row, scratch_top4,
                           scratch_lse, P, w2sm);
}

#undef MK2_GUARD_P1

// ======================================================== phase 2 ==========
// Per act p = blockIdx.x: thread t folds the MK2_NB2 = 510 phase-1 partials
// b in {t, t+256} & [0,510) (vectorized: 2 x float4 + 1 x float2 per
// partial) into a private register-resident top-4 (T4; score desc, tie ->
// smaller id) and an online (m, l) with __expf rescaling; a 5-round shfl
// butterfly merges each warp's lanes; warp partials cross smem (one
// __syncthreads) and warp 0 merges the 8 lists with a 3-round butterfly.
// Lane 0 of warp 0 writes lse[p] = m + logsum, top4[p][*] and the stable
// logp form logp[j] = (score_j - m) - logsum.
// The lse/logp merge association differs from the pre-warp-tree revision:
// top-4 ids are unchanged (deterministic total order), lse/logp can differ
// in the last fp32 ulp.
__device__ __forceinline__ void markov2_p2_body(
    const float* __restrict__ scratch_top4, const float* __restrict__ scratch_lse,
    unsigned* __restrict__ top4, float* __restrict__ logp, float* __restrict__ lse,
    int P, int PT) {
  __shared__ float ws_sc[8][4];                  // per-warp top4 partials
  __shared__ unsigned ws_id[8][4];
  __shared__ float ws_ml[8][2];                  // per-warp (m, l)

  const int p = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;

  // T4 is register-resident (an address-taken float* list would be demoted
  // to local memory); comparator: strictly greater wins, tie -> smaller id.
  T4 t4; t4.init();
  float m = -3.0e38f, l = 0.0f;
#pragma unroll
  for (int b = tid; b < MK2_NB2; b += MK2_NT) {
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
        // stable form: (score - m) - log(sum); score <= m, so the first
        // subtraction loses no bits to cancellation and logsum is never
        // rounded into the fp32 lse before it is subtracted.
        logp[(size_t)p * 4 + j] = (w4.s[j] - wm) - logsum;
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(MK2_NT) markov2_top4_phase2(
    const float* __restrict__ scratch_top4, const float* __restrict__ scratch_lse,
    unsigned* __restrict__ top4, float* __restrict__ logp, float* __restrict__ lse,
    int V, int P, int PT) {
  if (V != MK2_VOCAB || P < 1 || P > 64 || blockDim.x != MK2_NT) __trap();
  markov2_p2_body(scratch_top4, scratch_lse, top4, logp, lse, P, PT);
}
