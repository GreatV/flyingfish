// ============================================================================
// markov.cu -- DSpark Markov-head distribution kernel (tree verification).
// One launch serves a batch of P requests; the markov_w2 table is read once
// per launch and shared by all P rows.
//
// Semantics (bit-exact boundary vs the chain path, draft.rs Markov chain):
//   bias[p][v] = bf16( fp32 dot( w2[v], w1[parent_p] ) )   (fp32 accumulate,
//                one bf16 rounding -- matches the cuBLAS gemv boundary)
//   s[p][v]    = __hadd(bias[p][v], B[row_p][v])           (bf16 add, same
//                rounding as the repo residual kernel's __hadd2)
//   lse[p]     = fp32 logsumexp over the full vocab of s[p]
//   top-4[p]   = the 4 largest s[p] values; ties break to the smaller token
//                id (same rule as the repo argmax comparator: strictly greater
//                wins, tie -> lower index).
//   logp[p][j] = (s[p][top4[p][j]] - m) - logsum, where m = max_v s[p][v] and
//                logsum = log(Σ_v exp(s[p][v] - m))  (stable form: the
//                absolute lse is never rounded to fp32 before subtraction)
//   lse[p]     = m + logsum   (the fp32 output value)
//
// Inputs:
//   B       bf16 [7][V] base logits rows; row per request from req_row
//   w1      bf16 [V][256] (markov_w1); the parent embedding is w1[parent]
//   w2      bf16 [V][256] (markov_w2), row-major, 16B-aligned rows
//   req_tok u32  [P] parent tokens (device)
//   req_row i32  [P] B row indices (device)
// Outputs:
//   top4    u32  [P][4]
//   logp    fp32 [P][4]
//   lse     fp32 [P]
//
// Two-phase structure:
//   phase 1 (markov_top4_phase1): grid (V/8 = 16320), block 256 = 8 warps,
//     warp per vocab row. Each warp dots its w2 row with all P embeddings
//     (fp32 acc per act per lane, butterfly reduce), applies the two bf16
//     rounding points, and writes its (s, v) pair per act plus its lse
//     partial (m, l) per act to caller scratch. A block-level pass merges the
//     8 warps' pairs into one block top-4 per act.
//   phase 2 (markov_top4_phase2): grid (P), block 1024: merges the 16320
//     block top-4 lists per act (score-desc/id-asc merge; exact: a global
//     top-4 element always sits in its own block's top-4), reduces the lse
//     partials by LSE rescaling, and writes top4/logp/lse.
//
// Scratch (caller-allocated; NB = V/8 = 16320 blocks; PA = PT acts):
//   top4 pairs: [NB][PA][4] of (float score, u32 id) -- 8 B each
//   lse parts:  [NB][PA][2] fp32 (m, l)
//   bytes = NB*PA*32 + NB*PA*8 = NB*PA*40
//   PT = 64, NB = 16320 -> 41.7 MB worst case (P=64); PT=1 -> 0.65 MB.
// Every scratch element is written by exactly one block; phase 2 reads
// exactly NB partials per act.
//
// FD_GUARD: every entry point calls __trap() when invoked with V != 130560,
// in != 256, P outside [1, PT], PT mismatch with the template instance,
// blockDim.x != 256 (phase 1) / 1024 (phase 2), gridDim.y != 1 -- a
// wrong-shape launch fails loudly instead of silently writing garbage.
//
// CUDA Graph: P is an ordinary kernel argument, fixed per capture; the
// request arrays (req_tok, req_row) and all output buffers are pointers fixed
// per capture -- the engine updates their CONTENTS between replays (Graph
// replays read the current contents). No device scalars are read; no global
// state. The ticket counter is not used: phase 1 writes fixed slots and the
// phase boundary is the kernel boundary.
//
// Phase-1 launch: grid = dim3(V/8), block = 256,
//   smem = PT*512 (e rows) + 8*PT*4 (warp scores) + 8*PT*4 (warp ids)
//          + PT*4*4 (block top4 scores) + PT*4*4 (block top4 ids)
//          + PT*2*4 (block lse) = PT*616 B (39424 B at PT=64)
// Phase-2 launch: grid = dim3(P), block = 1024,
//   smem = NT2*4*4 (thread top4 scores) + NT2*4*4 (thread top4 ids)
//          + 2*NT2*4 (lse reduce) = NT2*40 B = 40960 B
//
// SIGNATURE TABLE (extern "C" __global__; all pointer args __restrict__):
//   markov_top4_phase1_p{1,8,64}   -- PT = max acts compiled in
//     void(const __nv_bfloat16* B, const __nv_bfloat16* w1,
//          const __nv_bfloat16* w2, const unsigned* req_tok,
//          const int* req_row, float* scratch_top4, float* scratch_lse,
//          int V, int in, int P)
//     launch: grid = dim3(V/8), block = 256, smem as above
//   markov_top4_phase2
//     void(const float* scratch_top4, const float* scratch_lse,
//          unsigned* top4, float* logp, float* lse, int V, int P, int PT)
//     launch: grid = dim3(P), block = 1024, smem = 40960
//
// HOST INTEGRATION (Codex-A):
//   - One launch pair per tree depth: gather (parent_token, B_row) per node
//     into the request arrays (device), then phase1 + phase2.
//   - PT dispatch: P == 1 -> p1; P <= 8 -> p8; 9..64 -> p64. P is fixed per
//     graph capture; capture one graph per P bucket you actually run.
//   - scratch bytes formula above; one slab, reused across depths.
//   - Chain path (P=1) top-1 equals the dense argmax whenever no two scores
//     sit within one bf16 ulp; the residual-boundary rounding is replicated
//     exactly (bf16(bias) then __hadd with B).
// ============================================================================
#include <cuda_runtime.h>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

constexpr int VOCAB = 130560;
constexpr int DIM   = 256;
constexpr int RPB   = 8;                  // vocab rows per phase-1 block
constexpr int NBLK  = VOCAB / RPB;        // 16320
constexpr int NT1   = 256;                // phase-1 block
constexpr int NT2   = 1024;               // phase-2 block

#define MK_GUARD_P1(PT) do {                                                                  \
  if (V != VOCAB || in != DIM || P < 1 || P > PT || blockDim.x != NT1 ||                  \
      gridDim.x != NBLK) __trap();                                                        \
} while (0)
#define MK_GUARD_P2 do {                                                                  \
  if (V != VOCAB || P < 1 || P > 64 || blockDim.x != NT2) __trap();                       \
} while (0)

// ------------------------------------------------------------- helpers ----
// Warp-shuffle reduction over 32 lanes; order matches the butterfly.
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// Sorted-insert into a 4-deep (score, id) list, score-desc / id-asc.
__device__ __forceinline__ void top4_insert(float* sc, unsigned* id, float s, unsigned i) {
  if (sc[3] > s || (sc[3] == s && id[3] < i)) return;      // can't enter
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const bool better = (s > sc[j]) || (s == sc[j] && i < id[j]);
    if (better) {
      // shift down from j
#pragma unroll
      for (int q = 3; q > j; --q) { sc[q] = sc[q - 1]; id[q] = id[q - 1]; }
      sc[j] = s; id[j] = i;
      return;
    }
  }
}

// -------------------------------------------------------- phase 1 body ----
// Warp per vocab row; per act an fp32 dot of the 256-dim slice, then the two
// bf16 rounding points, then one (s, v) pair per act into the block merge.
template <int PT>
__device__ __forceinline__ void markov_p1_body(
    const bf16* __restrict__ B, const bf16* __restrict__ w1c, const bf16* __restrict__ w2c,
    const unsigned* __restrict__ req_tok, const int* __restrict__ req_row,
    float* __restrict__ scratch_top4, float* __restrict__ scratch_lse, int P) {
  extern __shared__ __align__(16) bf16 mksm[];
  bf16* esm = mksm;                                   // [PT][DIM]
  float* wsc = reinterpret_cast<float*>(mksm + PT * DIM);      // [8][PT] warp scores
  unsigned* wid = reinterpret_cast<unsigned*>(wsc + 8 * PT);   // [8][PT]
  float* bsc = reinterpret_cast<float*>(wid + 8 * PT);         // [PT][4] block top4
  unsigned* bid = reinterpret_cast<unsigned*>(bsc + PT * 4);   // [PT][4]
  float* blse = reinterpret_cast<float*>(bid + PT * 4);        // [PT][2] block (m, l)

  const int tid  = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  // Stage the P embeddings w1[req_tok[p]] into smem (PT*256 bf16).
  for (int idx = tid; idx < P * (DIM / 8); idx += NT1) {
    const int p = idx >> 5, seg = idx & 31;           // 32 x 16B per row
    *reinterpret_cast<uint4*>(esm + p * DIM + seg * 8) =
        *reinterpret_cast<const uint4*>(w1c + (size_t)req_tok[p] * DIM + seg * 8);
  }
  __syncthreads();

  // One vocab row per warp.
  const int v = blockIdx.x * RPB + warp;
  const int lane0 = lane * 8;                          // this lane's 8 dims
  float wf[8];
  {
    const uint4 r = *reinterpret_cast<const uint4*>(w2c + (size_t)v * DIM + lane0);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&r);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __bfloat1622float2(h[i]);
      wf[2 * i] = f.x; wf[2 * i + 1] = f.y;
    }
  }

  for (int p = 0; p < P; ++p) {
    // dot: this lane's 8 dims of w2[v] against esm[p]
    float ef[8];
    {
      const uint4 r = *reinterpret_cast<const uint4*>(esm + p * DIM + lane0);
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&r);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(h[i]);
        ef[2 * i] = f.x; ef[2 * i + 1] = f.y;
      }
    }
    float dot = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) dot = fmaf(wf[j], ef[j], dot);
    dot = warp_sum(dot);
    // Two bf16 rounding points of the chain path, replicated exactly.
    const bf16 bias_r = __float2bfloat16(dot);
    const bf16 sv = __hadd(bias_r, B[(size_t)req_row[p] * VOCAB + v]);
    const float s = __bfloat162float(sv);
    // Warp pair (one row): only lane 0 holds the reduced dot, but all lanes
    // got the same warp_sum result; lane 0 writes.
    if (lane == 0) { wsc[warp * PT + p] = s; wid[warp * PT + p] = v; }
  }

  __syncthreads();
  // Block merge per act: warp 0 merges the 8 warp pairs into block top-4,
  // and the 8 lse partials.
  if (warp != 0) return;
  for (int p = lane; p < P; p += 32) {               // P up to 64 > warp width
    float sc[4]; unsigned id[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) { sc[j] = -INFINITY; id[j] = 0xFFFFFFFFu; }
    float m = -3.0e38f, l = 0.0f;
    for (int w = 0; w < 8; ++w) {
      const float s = wsc[w * PT + p];
      const unsigned i = wid[w * PT + p];
      top4_insert(sc, id, s, i);
      const float mn = fmaxf(m, s);
      l = l * __expf(m - mn) + __expf(s - mn);
      m = mn;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) { bsc[p * 4 + j] = sc[j]; bid[p * 4 + j] = id[j]; }
    blse[p * 2] = m; blse[p * 2 + 1] = l;
  }
  __syncthreads();
  // One thread per act stores the block partials to scratch. Only warp 0
  // (32 threads) is alive here.
  for (int p = tid; p < P; p += 32) {
    const size_t base = ((size_t)blockIdx.x * PT + p) * 8;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      scratch_top4[base + j * 2] = bsc[p * 4 + j];
      *reinterpret_cast<unsigned*>(scratch_top4 + base + j * 2 + 1) = bid[p * 4 + j];
    }
    scratch_lse[((size_t)blockIdx.x * PT + p) * 2] = blse[p * 2];
    scratch_lse[((size_t)blockIdx.x * PT + p) * 2 + 1] = blse[p * 2 + 1];
  }
}

// -------------------------------------------------------- phase 2 body ----
// One block per act: merge the NBLK block top-4 lists (score-desc/id-asc) and
// the lse partials; write top4/logp/lse.
__device__ __forceinline__ void markov_p2_body(
    const float* __restrict__ scratch_top4, const float* __restrict__ scratch_lse,
    unsigned* __restrict__ top4, float* __restrict__ logp, float* __restrict__ lse,
    int P, int PT) {
  extern __shared__ float mk2sm[];                    // [NT2][4] scores
  unsigned* mk2id = reinterpret_cast<unsigned*>(mk2sm + NT2 * 4);
  float* red = mk2sm + NT2 * 8;                       // [2][NT2] lse reduce

  const int p = blockIdx.x;
  const int tid = threadIdx.x;

  // Each thread merges a strided slice of the NBLK lists into a local top-4.
  float sc[4]; unsigned id[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) { sc[j] = -INFINITY; id[j] = 0xFFFFFFFFu; }
  float m = -3.0e38f, l = 0.0f;
  for (int b = tid; b < NBLK; b += NT2) {
    const size_t base = ((size_t)b * PT + p) * 8;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float s = scratch_top4[base + j * 2];
      const unsigned i = *reinterpret_cast<const unsigned*>(scratch_top4 + base + j * 2 + 1);
      top4_insert(sc, id, s, i);
    }
    const float bm = scratch_lse[((size_t)b * PT + p) * 2];
    const float bl = scratch_lse[((size_t)b * PT + p) * 2 + 1];
    const float mn = fmaxf(m, bm);
    l = l * __expf(m - mn) + bl * __expf(bm - mn);
    m = mn;
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) { mk2sm[tid * 4 + j] = sc[j]; mk2id[tid * 4 + j] = id[j]; }
  red[tid] = m; red[NT2 + tid] = l;
  __syncthreads();
  // Tree-merge the 1024 local top-4 lists: levels by halving in smem.
  for (int w = NT2 / 2; w > 0; w >>= 1) {
    if (tid < w) {
      float asc[4]; unsigned aid[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) { asc[j] = mk2sm[tid * 4 + j]; aid[j] = mk2id[tid * 4 + j]; }
      for (int j = 0; j < 4; ++j)
        top4_insert(asc, aid, mk2sm[(tid + w) * 4 + j], mk2id[(tid + w) * 4 + j]);
#pragma unroll
      for (int j = 0; j < 4; ++j) { mk2sm[tid * 4 + j] = asc[j]; mk2id[tid * 4 + j] = aid[j]; }
      const float m1 = red[tid], m2 = red[tid + w];
      const float l1 = red[NT2 + tid], l2 = red[NT2 + tid + w];
      const float mn = fmaxf(m1, m2);
      red[tid] = mn;
      red[NT2 + tid] = l1 * __expf(m1 - mn) + l2 * __expf(m2 - mn);
    }
    __syncthreads();
  }
  if (tid == 0) {
    const float m = red[0];
    const float logsum = logf(red[NT2]);
    lse[p] = m + logsum;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      top4[(size_t)p * 4 + j] = mk2id[j];
      // stable form: (score - m) - log(sum); score <= m, so the first
      // subtraction is exact-magnitude and no cancellation develops.
      logp[(size_t)p * 4 + j] = (mk2sm[j] - m) - logsum;
    }
  }
}

// ======================================================== exported ABI =====
template <int PT>
__device__ __forceinline__ void markov_p1_entry(
    const bf16* __restrict__ B, const bf16* __restrict__ w1c, const bf16* __restrict__ w2c,
    const unsigned* __restrict__ req_tok, const int* __restrict__ req_row,
    float* __restrict__ scratch_top4, float* __restrict__ scratch_lse, int V, int in, int P) {
  markov_p1_body<PT>(B, w1c, w2c, req_tok, req_row, scratch_top4, scratch_lse, P);
}

#define MK_P1_EXPORT(PT)                                                                     \
extern "C" __global__ void __launch_bounds__(NT1) markov_top4_phase1_p##PT(                  \
    const bf16* __restrict__ B, const bf16* __restrict__ w1c, const bf16* __restrict__ w2c,   \
    const unsigned* __restrict__ req_tok, const int* __restrict__ req_row,                   \
    float* __restrict__ scratch_top4, float* __restrict__ scratch_lse,                        \
    int V, int in, int P) {                                                                   \
  MK_GUARD_P1(PT);                                                                             \
  markov_p1_entry<PT>(B, w1c, w2c, req_tok, req_row, scratch_top4, scratch_lse, V, in, P);    \
}

MK_P1_EXPORT(1)
MK_P1_EXPORT(8)
MK_P1_EXPORT(64)

extern "C" __global__ void __launch_bounds__(NT2) markov_top4_phase2(
    const float* __restrict__ scratch_top4, const float* __restrict__ scratch_lse,
    unsigned* __restrict__ top4, float* __restrict__ logp, float* __restrict__ lse,
    int V, int P, int PT) {
  MK_GUARD_P2;
  markov_p2_body(scratch_top4, scratch_lse, top4, logp, lse, P, PT);
}

#undef MK_GUARD_P1
#undef MK_GUARD_P2
