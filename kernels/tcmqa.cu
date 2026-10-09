// ============================================================================
// tcmqa.cu -- TENSOR-CORE multi-query GQA flash-decoding attention (TCMQA)
// for MiniCPM5-2B speculative-decoding verification: M query tokens share one
// KV cache; tree-structured visibility via a per-row ancestor bitset.
// Scores and the P*V product run on bf16 tensor cores (mma.sync.m16n8k16);
// the online softmax is fp32 in registers (FA2-style, per MMA fragment row).
//
// Model shape (MiniCPM5-2B): GQA 16 q-heads / 2 kv-heads (group size 8),
// head_dim 128, bf16 data with fp32 accumulation, scale = 1/sqrt(128).
// FD_GUARD: every entry point calls __trap() when invoked with
// n_q_heads != 16, n_kv_heads != 2, gqa != 8, head_dim != 128, M outside
// [1, MQ_MAX=64], chunk not a positive multiple of 64, blockDim.x != 128,
// gridDim.y != 2, gridDim.z != ceil(8M/64), or anc == nullptr -- a
// wrong-shape launch fails loudly (the trap surfaces as a CUDA error at the
// next synchronization) instead of silently leaving outputs unwritten.
//
// Semantics:
//   The KV caches hold prefix positions [0, L) PLUS the M new tokens'
//   positions [L, L+M); total extent L+M. Query token i attends all prefix
//   positions plus draft position L+j iff bit j of anc[i] is set. The
//   block-causal case is anc[i] = bits 0..i set (built host-side; anc is a
//   plain kernel argument, fixed per graph capture). Positions >= L+M are
//   never visible (covered by the bit-count guard). Mask logic runs only in
//   the tiles overlapping [L, L+M): tiles fully below L take the no-mask fast
//   path.
//
//   q  : bf16 [M][16][128]; row of (query i, head qh) at
//        q + i*q_tok_stride + qh*q_head_stride  (strides in bf16 elements)
//   K,V: bf16, separate tensors, head_dim-contiguous rows
//   o  : bf16 [M][16][128]; row of (i, qh) at o + i*o_tok_stride + qh*o_head_stride
//
// Pairs: global pair index p = qh_local*M + i in [0, 8M) per kv head
// (qh_local in [0,8), i in [0,M)). Pair p occupies MMA row (p - rt*64) of
// row-tile rt. Blocks with row-tile rt cover pairs [rt*64, rt*64+64); rows
// with p >= 8M are dead (computed but never written).
//
// Two-phase split-KV (flash decoding):
//   phase 1: grid (nchunks, 2, ceil(8M/64)); block = 128 threads = 4 warps.
//            Block (ch, kv, rt) sweeps chunk ch of kv-head kv in KT=32-pos
//            tiles; warp w owns MMA rows [w*16, w*16+16) of the tile (all 128
//            dims of those rows), runs the online softmax per row, and writes
//            one partial (m, l, acc[128]) per pair to caller scratch. No
//            cross-warp reductions: warps own disjoint rows.
//   phase 1 wide (tcmqa_phase1_w): grid (nchunks, 2, ceil(8M/128)); block =
//            256 threads = 8 warps sharing one 128-row pair tile; warp w owns
//            MMA rows [w*16, w*16+16) of it. At M = 16 grid.z = 1, so the
//            chunk's K/V is read once per launch (the 64-row variant reads it
//            once per row-tile, i.e. twice at M = 16). Every pair's per-value
//            arithmetic is closed inside its owning warp and identical to the
//            64-row variant, so outputs are bitwise identical.
//   phase 2: grid (16*M); one block merges the nchunks partials of one
//            (query, head) pair by LSE rescaling and writes o.
//
// Per-warp per-K-tile flow (KT=32 positions, KTR = 32):
//   S[16][32]  = Q_w[16][128] * K_tile^T[128][32]: 8 k16-slices x 4 n8-slices
//                of mma.m16n8k16; A = ldmatrix.x4 on the Q smem tile, B =
//                ldmatrix.x2 (plain) on K smem rows [pos][d] (the col-major
//                k16 x n8 B fragment equals the row-major [n][k] tile).
//   softmax    : per row, max/sum over the row's S elements: each m16n8 fp32
//                fragment row is held by a 4-lane quad (2 elements per lane
//                per n8 slice), so row reductions are 2 quad shuffles.
//                m' = max(m, rowmax); cf = expf(m - m'); l = l*cf + rowsum;
//                acc *= cf. Masked elements are set to -INF before the update
//                (exact no-op: expf(-INF - m') = 0 with the finite -3e38
//                sentinel keeping m - m' NaN-free).
//   P bf16     : S fragments' D registers repack in-register to A fragments
//                (a k16 A fragment is two adjacent n8 D fragments, lane-wise).
//   O += P*V   : 2 k16-slices (32 pos) x 16 n8-slices (128 dims); B =
//                ldmatrix.x2.trans on V smem rows [pos][dim] (the col-major
//                k16 x n8 B fragment needs the transpose of the [k][n] tile).
//
// Thread map (phase 1): 128 threads = 4 warps x 32 lanes. Warp w owns rows
// [w*16, w*16+16); within a warp, lane l holds D-fragment elements
// (row l/4, cols (l%4)*2+{0,1}) and (row l/4+8, same cols) per n8 slice.
// m/l are replicated across each row's quad.
//
// Smem (dynamic, <= 48 KB, no cudaFuncSetAttribute needed):
//   Q tile  [64][136] bf16 (row stride padded +8 against ldmatrix bank
//   conflicts) | K tile [32][136] | V tile [32][136] | anc[64] u64.
//   Total = (64+32+32)*136*2 + 512 = 35328 B.
//   The wide variant's 128-row Q tile makes it (128+32+32)*136*2 + 512 =
//   52736 B > 48 KB: the host must opt in with cudaFuncSetAttribute(
//   cudaFuncAttributeMaxDynamicSharedMemorySize) before launching
//   tcmqa_phase1_w.
// Out-of-extent tile rows are zero-filled on load (NaN-safe: P is 0 there and
// 0*0 = 0 in the PV MMA).
//
// KV layouts (strides in bf16 ELEMENTS, multiples of 8 for 16B alignment):
//   position-major [pos][kv][d]:  head_stride = 128,     pos_stride = 256
//   head-major     [kv][pos][d]:  head_stride = Lh*128,  pos_stride = 128
//   Address: kv*head_stride + pos*pos_stride + d.  K and V strides are
//   independent arguments. q/o strides must likewise be multiples of 8
//   (contiguous defaults: tok_stride = 16*128 = 2048, head_stride = 128).
//
// Scratch (caller-allocated fp32, one buffer; NC = nchunks = ceil((L+M)/
// chunk); NP = 2*NC*8*M = 16*NC*M total partials):
//   scratch[0        .. NP)         = m   [NP]
//   scratch[NP       .. 2*NP)       = l   [NP]
//   scratch[2*NP     .. NP*(2+128)) = acc [NP][128]
//   partial index part = ((kv*NC + ch)*8 + qh_local)*M + i
//                      = (kv*NC + ch)*8*M + p   (p = qh_local*M + i)
//   Required size: 16 * nchunks * M * 130 floats.
//   Every (kv, ch, p) partial is written by exactly one block (the row-tile
//   owning p), so phase 2 reads exactly nchunks defined partials per
//   (query, head).
//
// CUDA Graph: the graph-safe entry points (non-_host) read L from a device
// scalar (const int* len_dev); the engine updates that scalar to change L
// between graph replays. M, chunk and the anc pointer are ordinary kernel
// arguments: fixed per capture, and M/chunk MUST be identical in both phases
// (both derive nchunks = ceil((L+M)/chunk) from L, M, chunk). The _host
// variants take L by value. No global state; every buffer is caller-allocated
// and passed per launch. Capacity-sized graphs: phase 1 may be launched with
// grid.x = nchunks_max while the replayed L needs fewer chunks; blocks with
// blockIdx.x >= ceil((L+M)/chunk) return before writing anything.
//
// Phase-1 launch:
//   grid  = dim3(nchunks, 2, ceil(8M/64))
//   block = 128 (fixed; guarded)
//   smem  = 35328 bytes
// Phase-1 wide launch (tcmqa_phase1_w):
//   grid  = dim3(nchunks, 2, ceil(8M/128))
//   block = 256 (fixed; guarded)
//   smem  = 52736 bytes (> 48 KB: cudaFuncSetAttribute opt-in required)
// Phase-2 launch:
//   grid  = (16*M)
//   block = threads2, a multiple of 128 (<= 1024)
//   smem  = (2*nchunks + threads2) * 4 bytes, <= 48 KB
// nchunks must be identical in both phases for the same (L, M, chunk).
//
// HOST INTEGRATION (Codex-A):
//   - phase-1: grid = dim3(nchunks, 2, ceil(8M/64)), block = 128,
//     smem = 35328 B (< 48 KB: no cudaFuncSetAttribute opt-in needed).
//   - phase-1 wide (tcmqa_phase1_w): grid = dim3(nchunks, 2, ceil(8M/128)),
//     block = 256, smem = 52736 B; call cudaFuncSetAttribute(
//     cudaFuncAttributeMaxDynamicSharedMemorySize, 52736) once at setup.
//     Outputs are bitwise identical to the 64-row variant; scratch layout and
//     phase 2 are unchanged.
//   - phase-2: grid = dim3(16*M), block = threads2 (multiple of 128, <= 1024),
//     smem = (2*nchunks + threads2)*4 B <= 49152.
//   - scratch bytes = 16 * nchunks * M * 130 * 4 (fp32), one slab; e.g.
//     32k/M64/c2048: 16*17*64*130*4 = 9.0 MB.
//   - chunk must be a positive multiple of 64 and identical in both phases.
//   - anc: M uint64 (device), bit j of row i = draft position L+j visible.
//     Chain/causal: anc[i] = (1ull << (i+1)) - 1; DSpark draft (full block):
//     anc[i] = (1ull << M) - 1. Built once per (M, topology); the pointer is
//     fixed per graph capture (update the contents, not the pointer).
//   - chunk candidates per (ctx, M) for setup calibration (4090 measured,
//     best first; A4000 calibrates separately -- its L2 is 4 MiB):
//       1k : M7/8 {64,128}   M16 {64,128}   M32 {64,128}   M64 {128,64}
//       8k : M7/8 {128,256}  M16 {256,128}  M32 {256,128}  M64 {512,256}
//       32k: M7/8 {256,512}  M16 {512,1024} M32 {1024,512} M64 {2048,1024}
//     v1 flash_decode_mq (kernels/attention.cu) stays a calibration candidate
//     for M <= 16 (it wins e.g. 1k m7F: 13.2 vs 14.2 us on 4090); choose per
//     (capacity bucket, M) by measurement, not by assumption.
//
// SIGNATURE TABLE (all symbols are extern "C" __global__; all pointer args
// are __restrict__; NC = nchunks = ceil((L+M)/chunk)).
// Failure semantics: wrong-shape calls or anc == nullptr __trap(); phase-1
// grid.x may exceed nchunks (capacity-sized graphs) -- excess blocks exit
// before any write.
//
//   tcmqa_phase1
//     void(const __nv_bfloat16* q, const __nv_bfloat16* k,
//          const __nv_bfloat16* v, float* scratch, const int* len_dev, int M,
//          int n_q_heads, int n_kv_heads, int gqa, int head_dim,
//          int q_tok_stride, int q_head_stride,
//          int k_head_stride, int k_pos_stride,
//          int v_head_stride, int v_pos_stride, int chunk,
//          const unsigned long long* anc)
//     launch: grid = dim3(nchunks, 2, ceil(8M/64)), block = 128, smem = 35328
//   tcmqa_phase1_host
//     void(... same, with int L in place of const int* len_dev ...)
//     launch: same as tcmqa_phase1
//   tcmqa_phase1_w
//     void(... same argument list as tcmqa_phase1 ...)
//     launch: grid = dim3(nchunks, 2, ceil(8M/128)), block = 256,
//             smem = 52736 (cudaFuncSetAttribute opt-in required)
//   tcmqa_phase1_w_host
//     void(... same, with int L in place of const int* len_dev ...)
//     launch: same as tcmqa_phase1_w
//   tcmqa_phase2
//     void(float* scratch, __nv_bfloat16* o, const int* len_dev, int M,
//          int n_q_heads, int n_kv_heads, int gqa, int head_dim,
//          int o_tok_stride, int o_head_stride, int chunk)
//     launch: grid = dim3(16*M), block = threads2 (multiple of 128, <= 1024),
//             smem = (2*nchunks + threads2)*4 bytes, <= 49152
//   tcmqa_phase2_host
//     void(float* scratch, __nv_bfloat16* o, int L, int M,
//          int n_q_heads, int n_kv_heads, int gqa, int head_dim,
//          int o_tok_stride, int o_head_stride, int chunk)
//     launch: same as tcmqa_phase2
// ============================================================================
#include <cuda_runtime.h>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

// ------------------------------------------------------- model / shape ----
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
constexpr float SCALE = 0.08838834764831845f;   // 1/sqrt(128); sqrtf() is not constexpr

// ------------------------------------------------------------- helpers ----
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

// -------------------------------------------------------- phase 1 body ----
// Block (ch, kv, rt): sweeps chunk ch of kv-head kv in KT-pos tiles; warp w
// owns MMA rows [w*16, w*16+16) of the MT_-row pair tile (MT_ = 64 with
// NT_ = 128 threads, or MT_ = 128 with NT_ = 256).
template <int MT_, int NT_>
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
  bf16* qsm = tcm_sm;                                 // [MT_][LDS]
  bf16* ksm = tcm_sm + MT_ * LDS;                     // [KT][LDS]
  bf16* vsm = ksm + KT * LDS;                         // [KT][LDS]
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
      *reinterpret_cast<uint4*>(qsm + r * LDS + seg * 8) = val;
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

  // ---- sweep the chunk in KT-position tiles ----
  // K/V tiles are staged with cp.async. The prologue stages K(0); per tile,
  // CP1 waits for K(t), barriers, and stages V(t) during the S mma and
  // softmax; CP2 waits for V(t), barriers, and stages K(t+1) during the PV
  // mma. Each stage overwrites the tile consumed before the preceding
  // barrier.
  const int ntiles = (pos_end > pos_beg) ? ((pos_end - pos_beg + KT - 1) / KT) : 0;

  // Stage one 32-pos tile of a KV tensor into `dst` ([KT][LDS]) via cp.async
  // 16B copies; out-of-extent rows are zero-filled (src-size 0).
  auto stage_kv = [&](const bf16* src, int head_stride, int pos_stride, int t,
                      bf16* dst) {
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
        cp16(dst + r * LDS + seg * 8, g, live ? 16 : 0);
      }
    }
  };

  if (ntiles > 0) { stage_kv(Kc, k_head_stride, k_pos_stride, 0, ksm); cp_commit(); }

  for (int t = 0; t < ntiles; ++t) {
    const int t0 = pos_beg + t * KT;
    cp_wait0();                                       // K(t) landed (newest group)
    __syncthreads();                                  // K(t) visible; V(t-1) consumed by all
    stage_kv(Vc, v_head_stride, v_pos_stride, t, vsm);
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
      ldsm_x4(qa, qsm + (warp * 16 + (lane & 15)) * LDS + kk * 16 + (lane >> 4) * 8);
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        unsigned kb[2];
        // B k16(d) x n8(pos), plain ldmatrix on K[pos][d]:
        // mat0 lanes 0-7: pos n*8+lane%8, d kk*16; mat1 lanes 8-15: d kk*16+8.
        ldsm_x2(kb, ksm + (n * 8 + (lane & 7)) * LDS + kk * 16 + (lane >> 3) * 8);
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
      // rescale the accumulators of this row half
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
      stage_kv(Kc, k_head_stride, k_pos_stride, t + 1, ksm);
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

// -------------------------------------------------------- phase 2 body ----
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

// ======================================================== exported ABI =====
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
  // Capacity-sized graph launches may carry more CTAs than this L needs.
  // Block-uniform, before any __syncthreads: excess blocks must not write
  // partials (part = (kv*nchunks + ch)*8*M + p would index into the next tile).
  if (blockIdx.x >= nchunks) return;
  float* p_m = scratch;
  float* p_l = scratch + (size_t)HKV * nchunks * GQA * M;
  float* p_o = p_l + (size_t)HKV * nchunks * GQA * M;
  tcmqa_phase1_body<MT_, NT_>(q, k, v, p_m, p_l, p_o, anc, L, M, chunk, nchunks,
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

extern "C" __global__ void __launch_bounds__(NT) tcmqa_phase1_host(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
    float* __restrict__ scratch, int L, int M,
    int n_q_heads, int n_kv_heads, int gqa, int head_dim,
    int q_tok_stride, int q_head_stride,
    int k_head_stride, int k_pos_stride, int v_head_stride, int v_pos_stride,
    int chunk, const unsigned long long* __restrict__ anc) {
  FD_GUARD_P1;
  tcmqa_phase1_entry<MT, NT>(q, k, v, scratch, anc, L, M, chunk,
                             q_tok_stride, q_head_stride,
                             k_head_stride, k_pos_stride, v_head_stride, v_pos_stride);
}

// Wide pair-tile variant: one 128-row tile per block (8 warps share one K/V
// smem tile set), so at M = 16 the KV chunk is read once per launch.
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

extern "C" __global__ void __launch_bounds__(256) tcmqa_phase1_w_host(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
    float* __restrict__ scratch, int L, int M,
    int n_q_heads, int n_kv_heads, int gqa, int head_dim,
    int q_tok_stride, int q_head_stride,
    int k_head_stride, int k_pos_stride, int v_head_stride, int v_pos_stride,
    int chunk, const unsigned long long* __restrict__ anc) {
  FD_GUARD_P1W;
  tcmqa_phase1_entry<128, 256>(q, k, v, scratch, anc, L, M, chunk,
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

extern "C" __global__ void tcmqa_phase2_host(float* __restrict__ scratch,
                                             bf16* __restrict__ o, int L, int M,
                                             int n_q_heads, int n_kv_heads, int gqa,
                                             int head_dim,
                                             int o_tok_stride, int o_head_stride,
                                             int chunk) {
  FD_GUARD_P2;
  tcmqa_phase2_entry(scratch, o, L, M, chunk, n_q_heads, o_tok_stride, o_head_stride);
}

#undef FD_GUARD_P1
#undef FD_GUARD_P1W
#undef FD_GUARD_P2
