// ============================================================================
// attention.cu -- GQA flash-decoding attention for ONE decode token.
// Multi-query (M-token verification) kernels are in the second half of this
// file (symbols flash_decode_mq_*).
//
// Model shape (MiniCPM5-2B): GQA 16 q-heads / 2 kv-heads (group size 8),
// head_dim 128, bf16 data with fp32 accumulation, scale = 1/sqrt(128).
// Causal decode: the single token attends to all L cached KV positions.
// FD_GUARD: every entry point calls __trap() when invoked with
// n_q_heads != 16, n_kv_heads != 2, gqa != 8, or head_dim != 128 --
// a wrong-shape launch fails loudly (the trap surfaces as a CUDA error at
// the next synchronization) instead of silently leaving outputs unwritten.
//
//   q  : bf16 [16][128] (one token, contiguous rows)
//   K,V: bf16, separate tensors, head_dim-contiguous rows
//   o  : bf16 [16][128]
//
// Two-phase split-KV (flash decoding):
//   phase 1: grid (nchunks, n_q_heads/QP); one block runs an online softmax
//            (running max m, running sum l, fp32 acc[128]) over one chunk of
//            `chunk` KV positions for QP q-heads that share one kv head, and
//            writes the partial (m, l, acc[128]) to caller scratch. The
//            per-group partials inside a block are merged with an
//            online-softmax merge (M = max, rescale by exp(m - M)) in two
//            levels: intra-warp butterfly shuffle, then cross-warp through
//            dynamic shared memory.
//   phase 2: grid (n_q_heads); one block merges the nchunks partials of one
//            q head by LSE rescaling and writes o[16][128] bf16.
//
// Thread map (phase 1): TP=16 lanes cooperate on one key position; lane l
// owns dims [l*8, l*8+8), so K and V each load as one 16B vector per lane at
// head_dim=128. The q.k dot is a butterfly reduction over the 16 lanes; m/l
// are replicated per group while the 128-dim accumulator is sharded 8 dims
// per lane. QP=8 packs all 8 q-heads of a kv head into one block so each K/V
// row is read once instead of 8 times.
//
// KV layouts (same kernel, only the strides differ; strides are in bf16
// ELEMENTS and must be multiples of 8 so the 16B vector loads stay aligned):
//   position-major [pos][kv][d]:  head_stride = 128,     pos_stride = 256
//   head-major     [kv][pos][d]:  head_stride = Lh*128,  pos_stride = 128
//     (Lh = L for a compact cache, or Lmax for a fixed max-len cache slab)
//   Address: kv*head_stride + pos*pos_stride + d.  K and V strides are
//   independent arguments.
//
// Scratch (caller-allocated fp32, one buffer; NQ = n_q_heads, NC = nchunks =
// ceil(L/chunk)):
//   scratch[0         .. NQ*NC)          = m           [NQ][NC]
//   scratch[NQ*NC     .. 2*NQ*NC)        = l           [NQ][NC]
//   scratch[2*NQ*NC   .. NQ*NC*(2+128))  = acc         [NQ][NC][128]
//   Required size: n_q_heads * nchunks * (2 + 128) floats.
//
// CUDA Graph: the graph-safe entry points (non-_host) read L from a device
// scalar (const int* len_dev); the engine updates that scalar to change L
// between graph replays. The _host variants take L by value. No global
// state; every buffer is caller-allocated and passed per launch.
// Capacity-sized graphs: phase 1 may be launched with grid.x = nchunks_max
// (for the longest captured L) while the replayed L needs fewer chunks;
// blocks with blockIdx.x >= ceil(L/chunk) return before writing anything,
// so other heads' partials in scratch are never touched.
//
// Phase-1 launch REQUIRES dynamic shared memory for the cross-warp merge:
//   smem  = qpack * (threads/32) * (128+2) * 4 bytes,  smem <= 48 KB
//   block = threads, a multiple of 32
//   grid  = (nchunks, n_q_heads/qpack)
// Phase-2 launch:
//   grid  = n_q_heads (= 16)
//   block = threads2, a multiple of 128 (<= 1024)
//   smem  = (2*nchunks + threads2) * 4 bytes,  smem <= 48 KB
// nchunks must be identical in both phases for the same (L, chunk).
//
// Measured (RTX 4090, sm_89, CUDA 13.2; cold L2 via rotating KV replicas,
// 288 MiB footprint; per-iteration CUDA events, median; full two-phase
// pipeline; threads=256):
//   ctx1k  qp2/chunk128: 10.2 us, 103 GB/s
//   ctx8k  qp2/chunk256: 18.4 us, 456 GB/s
//   ctx32k qp2/chunk512: 46.1 us, 728 GB/s  (76% of the 958 GB/s plateau)
//   max_abs err vs fp64 CPU reference <= 6.1e-5 across the sweep shapes.
//
// SIGNATURE TABLE (all symbols are extern "C" __global__; all pointer args
// are __restrict__; QP = q-heads per block; nchunks = ceil(L/chunk)).
// Failure semantics: wrong-shape calls __trap(); phase-1 grid.x may exceed
// nchunks (capacity-sized graphs) -- excess blocks exit before any write.
//
//   flash_decode_phase1_q1  -- QP=1
//     void(const __nv_bfloat16* q, const __nv_bfloat16* k,
//          const __nv_bfloat16* v, float* scratch, const int* len_dev,
//          int n_q_heads, int n_kv_heads, int gqa, int head_dim,
//          int k_head_stride, int k_pos_stride,
//          int v_head_stride, int v_pos_stride, int chunk)
//     launch: grid = dim3(nchunks, 16), block = threads (multiple of 32),
//             smem = 1*(threads/32)*130*4 bytes, <= 49152
//   flash_decode_phase1_q1_host -- QP=1
//     void(const __nv_bfloat16* q, const __nv_bfloat16* k,
//          const __nv_bfloat16* v, float* scratch, int L,
//          int n_q_heads, int n_kv_heads, int gqa, int head_dim,
//          int k_head_stride, int k_pos_stride,
//          int v_head_stride, int v_pos_stride, int chunk)
//     launch: same as flash_decode_phase1_q1
//   flash_decode_phase1_q2  -- QP=2
//     void(const __nv_bfloat16* q, const __nv_bfloat16* k,
//          const __nv_bfloat16* v, float* scratch, const int* len_dev,
//          int n_q_heads, int n_kv_heads, int gqa, int head_dim,
//          int k_head_stride, int k_pos_stride,
//          int v_head_stride, int v_pos_stride, int chunk)
//     launch: grid = dim3(nchunks, 8), block = threads (multiple of 32),
//             smem = 2*(threads/32)*130*4 bytes, <= 49152
//   flash_decode_phase1_q2_host -- QP=2
//     void(const __nv_bfloat16* q, const __nv_bfloat16* k,
//          const __nv_bfloat16* v, float* scratch, int L,
//          int n_q_heads, int n_kv_heads, int gqa, int head_dim,
//          int k_head_stride, int k_pos_stride,
//          int v_head_stride, int v_pos_stride, int chunk)
//     launch: same as flash_decode_phase1_q2
//   flash_decode_phase1_q4  -- QP=4
//     void(const __nv_bfloat16* q, const __nv_bfloat16* k,
//          const __nv_bfloat16* v, float* scratch, const int* len_dev,
//          int n_q_heads, int n_kv_heads, int gqa, int head_dim,
//          int k_head_stride, int k_pos_stride,
//          int v_head_stride, int v_pos_stride, int chunk)
//     launch: grid = dim3(nchunks, 4), block = threads (multiple of 32),
//             smem = 4*(threads/32)*130*4 bytes, <= 49152
//   flash_decode_phase1_q4_host -- QP=4
//     void(const __nv_bfloat16* q, const __nv_bfloat16* k,
//          const __nv_bfloat16* v, float* scratch, int L,
//          int n_q_heads, int n_kv_heads, int gqa, int head_dim,
//          int k_head_stride, int k_pos_stride,
//          int v_head_stride, int v_pos_stride, int chunk)
//     launch: same as flash_decode_phase1_q4
//   flash_decode_phase1_q8  -- QP=8
//     void(const __nv_bfloat16* q, const __nv_bfloat16* k,
//          const __nv_bfloat16* v, float* scratch, const int* len_dev,
//          int n_q_heads, int n_kv_heads, int gqa, int head_dim,
//          int k_head_stride, int k_pos_stride,
//          int v_head_stride, int v_pos_stride, int chunk)
//     launch: grid = dim3(nchunks, 2), block = threads (multiple of 32),
//             smem = 8*(threads/32)*130*4 bytes, <= 49152
//   flash_decode_phase1_q8_host -- QP=8
//     void(const __nv_bfloat16* q, const __nv_bfloat16* k,
//          const __nv_bfloat16* v, float* scratch, int L,
//          int n_q_heads, int n_kv_heads, int gqa, int head_dim,
//          int k_head_stride, int k_pos_stride,
//          int v_head_stride, int v_pos_stride, int chunk)
//     launch: same as flash_decode_phase1_q8
//   flash_decode_phase2
//     void(float* scratch, __nv_bfloat16* o, const int* len_dev,
//          int n_q_heads, int n_kv_heads, int gqa, int head_dim, int chunk)
//     launch: grid = dim3(16), block = threads2 (multiple of 128, <= 1024),
//             smem = (2*nchunks + threads2)*4 bytes, <= 49152
//   flash_decode_phase2_host
//     void(float* scratch, __nv_bfloat16* o, int L,
//          int n_q_heads, int n_kv_heads, int gqa, int head_dim, int chunk)
//     launch: same as flash_decode_phase2
// ============================================================================
#include <cuda_runtime.h>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

// ------------------------------------------------------- model / shape ----
constexpr int HQ  = 16;                 // q heads
constexpr int HKV = 2;                  // kv heads
constexpr int GQA = HQ / HKV;           // 8 q heads per kv head
constexpr int DH  = 128;                // head_dim
constexpr int TP  = 16;                 // lanes cooperating on one key position
constexpr float SCALE = 0.08838834764831845f;   // 1/sqrt(128); sqrtf() is not constexpr

// ------------------------------------------------------------- helpers ----
// Load 8 bf16 (one 16B vector) and widen to 8 floats.
__device__ __forceinline__ void ld_bf8(const bf16* __restrict__ p, float* out) {
  const uint4 r = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&r);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(h[i]);
    out[2 * i]     = f.x;
    out[2 * i + 1] = f.y;
  }
}
__device__ __forceinline__ void ld_bf4(const bf16* __restrict__ p, float* out) {
  const uint2 r = *reinterpret_cast<const uint2*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&r);
  const float2 f0 = __bfloat1622float2(h[0]);
  const float2 f1 = __bfloat1622float2(h[1]);
  out[0] = f0.x; out[1] = f0.y; out[2] = f1.x; out[3] = f1.y;
}
__device__ __forceinline__ void st_f8(float* __restrict__ p, const float* v) {
  *reinterpret_cast<float4*>(p)     = make_float4(v[0], v[1], v[2], v[3]);
  *reinterpret_cast<float4*>(p + 4) = make_float4(v[4], v[5], v[6], v[7]);
}
__device__ __forceinline__ void st_f4(float* __restrict__ p, const float* v) {
  *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
}
// VPR bf16 in / VPR fp32 out. Uses 16B loads whenever VPR is a multiple of 8
// (head_dim=128 -> VPR=8 -> exactly one 16B load per lane per tensor).
template <int VPR>
__device__ __forceinline__ void ld_vec(const bf16* __restrict__ p, float* out) {
  static_assert(VPR % 4 == 0, "VPR must be a multiple of 4 for vectorized loads");
  if constexpr (VPR % 8 == 0) {
#pragma unroll
    for (int j = 0; j < VPR; j += 8) ld_bf8(p + j, out + j);
  } else {
#pragma unroll
    for (int j = 0; j < VPR; j += 4) ld_bf4(p + j, out + j);
  }
}
template <int VPR>
__device__ __forceinline__ void st_vec(float* __restrict__ p, const float* v) {
  if constexpr (VPR % 8 == 0) {
#pragma unroll
    for (int j = 0; j < VPR; j += 8) st_f8(p + j, v + j);
  } else {
#pragma unroll
    for (int j = 0; j < VPR; j += 4) st_f4(p + j, v + j);
  }
}

// -------------------------------------------------------- phase 1 body ----
// QP = q heads per block (all sharing one kv head), U = key positions per block
// iteration per lane group (software pipeline depth).
template <int DH, int QP, int U>
__device__ __forceinline__ void fd_phase1_body(
    const bf16* __restrict__ q, const bf16* __restrict__ Kc, const bf16* __restrict__ Vc,
    float* __restrict__ p_m, float* __restrict__ p_l, float* __restrict__ p_o,
    int L, int chunk, int nchunks, int gqa,
    int k_head_stride, int k_pos_stride, int v_head_stride, int v_pos_stride) {
  constexpr int VPR = DH / TP;
  const int ch      = blockIdx.x;
  const int qh0     = blockIdx.y * QP;
  const int kv      = qh0 / gqa;                    // QP divides gqa => all QP heads share kv
  const int pos_beg = ch * chunk;
  const int pos_end = min(pos_beg + chunk, L);

  const int tid  = threadIdx.x;
  const int lane = tid % TP;
  const int grp  = tid / TP;
  const int dim0 = lane * VPR;
  const int ngrp = blockDim.x / TP;

  const bf16* __restrict__ Krow = Kc + kv * k_head_stride + dim0;
  const bf16* __restrict__ Vrow = Vc + kv * v_head_stride + dim0;

  float qr[QP][VPR];
  float acc[QP][VPR];
  float mx[QP], ls[QP];
#pragma unroll
  for (int h = 0; h < QP; ++h) {
    ld_vec<VPR>(q + (qh0 + h) * DH + dim0, qr[h]);
    mx[h] = -3.0e38f;                              // finite sentinel: never NaN in m - m'
    ls[h] = 0.0f;
#pragma unroll
    for (int j = 0; j < VPR; ++j) acc[h][j] = 0.0f;
  }

  // Block-uniform trip count so every lane stays converged for the shuffles.
  const int step  = ngrp * U;
  const int niter = (pos_end > pos_beg) ? ((pos_end - pos_beg + step - 1) / step) : 0;

  for (int it = 0; it < niter; ++it) {
    const int base = pos_beg + grp + it * step;

    // Issue all U K loads before touching any of them (memory parallelism).
    float kf[U][VPR];
    bool  ok[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int pos = base + u * ngrp;
      ok[u] = (pos < pos_end);
      ld_vec<VPR>(Krow + (size_t)(ok[u] ? pos : 0) * k_pos_stride, kf[u]);
    }

#pragma unroll
    for (int u = 0; u < U; ++u) {
      float s[QP];
#pragma unroll
      for (int h = 0; h < QP; ++h) s[h] = 0.0f;
#pragma unroll
      for (int j = 0; j < VPR; ++j)
#pragma unroll
        for (int h = 0; h < QP; ++h) s[h] = fmaf(qr[h][j], kf[u][j], s[h]);

      // Butterfly reduce the dot product across the TP lanes of this group.
      // off < TP <= 16, so a warp's two groups reduce independently.
#pragma unroll
      for (int h = 0; h < QP; ++h)
#pragma unroll
        for (int off = TP / 2; off > 0; off >>= 1)
          s[h] += __shfl_xor_sync(0xffffffffu, s[h], off);

      float cf[QP], pf[QP];
#pragma unroll
      for (int h = 0; h < QP; ++h) {
        const float sp = ok[u] ? s[h] * SCALE : -INFINITY;   // dead slot -> no-op softmax step
        const float mn = fmaxf(mx[h], sp);
        cf[h] = __expf(mx[h] - mn);
        pf[h] = __expf(sp - mn);
        mx[h] = mn;
        ls[h] = ls[h] * cf[h] + pf[h];
      }

      const int pos = base + u * ngrp;
      float vf[VPR];
      ld_vec<VPR>(Vrow + (size_t)(ok[u] ? pos : 0) * v_pos_stride, vf);
#pragma unroll
      for (int h = 0; h < QP; ++h)
#pragma unroll
        for (int j = 0; j < VPR; ++j)
          acc[h][j] = fmaf(acc[h][j], cf[h], pf[h] * vf[j]);
    }
  }

  // ---- Merge the per-group partials into one block-level partial. ----
  // Groups processed disjoint position sets, so this is an online-softmax
  // merge, not a plain sum: M=max, rescale by exp(m-M), then accumulate.
  // Level 1: the two TP-lane groups inside each warp exchange via shuffles
  // (butterfly: both halves end up holding the merged value).
#pragma unroll
  for (int h = 0; h < QP; ++h) {
    const float pm = __shfl_xor_sync(0xffffffffu, mx[h], TP);
    const float pl = __shfl_xor_sync(0xffffffffu, ls[h], TP);
    const float M  = fmaxf(mx[h], pm);
    const float c  = __expf(mx[h] - M);        // 1 when both sentinels
    const float pc = __expf(pm - M);
    float pa[VPR];
#pragma unroll
    for (int j = 0; j < VPR; ++j) pa[j] = __shfl_xor_sync(0xffffffffu, acc[h][j], TP);
    ls[h] = ls[h] * c + pl * pc;
#pragma unroll
    for (int j = 0; j < VPR; ++j) acc[h][j] = acc[h][j] * c + pa[j] * pc;
    mx[h] = M;
  }
  // Level 2: cross-warp merge through dynamic shared memory.
  // smem layout: acc [QP][nw][DH] | m [QP][nw] | l [QP][nw].
  const int warp   = tid / 32;
  const int lane32 = tid % 32;
  const int nw     = blockDim.x / 32;
  extern __shared__ float fd_sm[];
  float* sa  = fd_sm;                              // [QP][nw][DH]
  float* smm = sa + QP * nw * DH;                  // [QP][nw]
  float* sll = smm + QP * nw;                      // [QP][nw]
  if (lane32 < TP) {   // one group per warp writes (both halves hold the same merged shard)
#pragma unroll
    for (int h = 0; h < QP; ++h) {
      if (lane32 == 0) { smm[h * nw + warp] = mx[h]; sll[h * nw + warp] = ls[h]; }
#pragma unroll
      for (int j = 0; j < VPR; ++j) sa[(h * nw + warp) * DH + lane32 * VPR + j] = acc[h][j];
    }
  }
  __syncthreads();
  // Only lanes 0..TP-1 of warp 0 merge across warps and store the partial.
  if (warp != 0 || lane32 >= TP) return;
#pragma unroll
  for (int h = 0; h < QP; ++h) {
    for (int w = 1; w < nw; ++w) {
      const float pm = smm[h * nw + w], pl = sll[h * nw + w];
      const float M  = fmaxf(mx[h], pm);
      const float c  = __expf(mx[h] - M), pc = __expf(pm - M);
      ls[h] = ls[h] * c + pl * pc;
#pragma unroll
      for (int j = 0; j < VPR; ++j)
        acc[h][j] = acc[h][j] * c + sa[(h * nw + w) * DH + lane * VPR + j] * pc;
      mx[h] = M;
    }
  }

  // Partials: m,l are [n_q_heads][nchunks] (phase 2 reads them coalesced),
  // acc is [n_q_heads][nchunks][head_dim].
#pragma unroll
  for (int h = 0; h < QP; ++h) {
    const size_t part = (size_t)(qh0 + h) * nchunks + ch;
    if (lane == 0) { p_m[part] = mx[h]; p_l[part] = ls[h]; }
    st_vec<VPR>(p_o + part * DH + dim0, acc[h]);
  }
}

// -------------------------------------------------------- phase 2 body ----
// One block per q head; blockDim.x = DH * G2 where G2 groups split the chunk
// range (G2 = blockDim.x / DH).  Dynamic shared: 2*nchunks (m,l) + blockDim.
template <int DH>
__device__ __forceinline__ void fd_phase2_body(const float* __restrict__ p_m,
                                               const float* __restrict__ p_l,
                                               const float* __restrict__ p_o,
                                               bf16* __restrict__ o, int nchunks) {
  extern __shared__ float sm[];
  const int qh  = blockIdx.x;
  const int tid = threadIdx.x;
  const int nt  = blockDim.x;
  const int G2  = nt / DH;
  const int g   = tid / DH;
  const int d   = tid % DH;

  float* sm_ = sm;
  float* sl_ = sm + nchunks;
  float* red = sm + 2 * nchunks;

  for (int h = tid; h < nchunks; h += nt) {
    sm_[h] = p_m[qh * nchunks + h];
    sl_[h] = p_l[qh * nchunks + h];
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
  const float M = red[0];

  float pv = 0.0f;
  for (int h = tid; h < nchunks; h += nt) pv += sl_[h] * __expf(sm_[h] - M);
  __syncthreads();      // every thread has consumed red[0] before red[] is reused
  red[tid] = pv;
  __syncthreads();
  for (int s = nt / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  const float lse = red[0];                        // sum_h l_h * exp(m_h - M)
  __syncthreads();      // same barrier before red[] is reused by the G2 reduction

  float acc = 0.0f;
  for (int h = g; h < nchunks; h += G2) {
    const float w = __expf(sm_[h] - M);
    acc = fmaf(w, p_o[((size_t)qh * nchunks + h) * DH + d], acc);
  }
  if (G2 > 1) {
    red[tid] = acc;
    __syncthreads();
#pragma unroll
    for (int k = 1; k < G2; ++k) acc += red[k * DH + d];
  }

  const float inv = (lse > 0.0f) ? (1.0f / lse) : 0.0f;
  if (g == 0) o[qh * DH + d] = __float2bfloat16(acc * inv);   // only group 0 holds the reduced acc
}

// ======================================================== exported ABI =====
// Concrete extern "C" __global__ entry points. q_pack is baked into the symbol
// name; every other dimension and every stride is an explicit argument; the
// scratch buffer is caller-allocated; L comes from a device scalar so these are
// CUDA-Graph-capturable.
#define FD_GUARD do { if (head_dim != DH || n_q_heads != HQ || n_kv_heads != HKV || gqa != GQA) __trap(); } while (0)

template <int DH, int QP>
__device__ __forceinline__ void fd_phase1_entry(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
    float* __restrict__ scratch, int L, int chunk, int n_q_heads, int gqa,
    int k_head_stride, int k_pos_stride, int v_head_stride, int v_pos_stride) {
  constexpr int U = (QP == 1) ? 2 : 1;            // pipeline depth baked per variant
  const int nchunks = (L + chunk - 1) / chunk;
  // Capacity-sized graph launches may carry more CTAs than this L needs.
  // Block-uniform, before any __syncthreads: excess blocks must not write
  // partials (part = qh*nchunks + ch would index into the next head).
  if (blockIdx.x >= nchunks) return;
  float* p_m = scratch;
  float* p_l = scratch + (size_t)n_q_heads * nchunks;
  float* p_o = scratch + 2 * (size_t)n_q_heads * nchunks;
  fd_phase1_body<DH, QP, U>(q, k, v, p_m, p_l, p_o, L, chunk, nchunks, gqa,
                            k_head_stride, k_pos_stride, v_head_stride, v_pos_stride);
}

#define FD_P1_EXPORTS(QP)                                                                   \
extern "C" __global__ void flash_decode_phase1_q##QP(                                       \
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,      \
    float* __restrict__ scratch, const int* __restrict__ len_dev,                           \
    int n_q_heads, int n_kv_heads, int gqa, int head_dim,                                   \
    int k_head_stride, int k_pos_stride, int v_head_stride, int v_pos_stride, int chunk) {  \
  FD_GUARD;                                                                                 \
  fd_phase1_entry<DH, QP>(q, k, v, scratch, *len_dev, chunk, n_q_heads, gqa,               \
                           k_head_stride, k_pos_stride, v_head_stride, v_pos_stride);       \
}                                                                                           \
extern "C" __global__ void flash_decode_phase1_q##QP##_host(                                \
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,      \
    float* __restrict__ scratch, int L,                                                      \
    int n_q_heads, int n_kv_heads, int gqa, int head_dim,                                   \
    int k_head_stride, int k_pos_stride, int v_head_stride, int v_pos_stride, int chunk) {  \
  FD_GUARD;                                                                                 \
  fd_phase1_entry<DH, QP>(q, k, v, scratch, L, chunk, n_q_heads, gqa,                      \
                           k_head_stride, k_pos_stride, v_head_stride, v_pos_stride);       \
}

FD_P1_EXPORTS(1)
FD_P1_EXPORTS(2)
FD_P1_EXPORTS(4)
FD_P1_EXPORTS(8)

template <int DH>
__device__ __forceinline__ void fd_phase2_entry(float* __restrict__ scratch,
                                                bf16* __restrict__ o, int L, int chunk,
                                                int n_q_heads) {
  const int nchunks = (L + chunk - 1) / chunk;
  const float* p_m = scratch;
  const float* p_l = scratch + (size_t)n_q_heads * nchunks;
  const float* p_o = scratch + 2 * (size_t)n_q_heads * nchunks;
  fd_phase2_body<DH>(p_m, p_l, p_o, o, nchunks);
}

extern "C" __global__ void flash_decode_phase2(float* __restrict__ scratch,
                                                bf16* __restrict__ o,
                                                const int* __restrict__ len_dev,
                                                int n_q_heads, int n_kv_heads, int gqa,
                                                int head_dim, int chunk) {
  FD_GUARD;
  fd_phase2_entry<DH>(scratch, o, *len_dev, chunk, n_q_heads);
}

extern "C" __global__ void flash_decode_phase2_host(float* __restrict__ scratch,
                                                    bf16* __restrict__ o, int L,
                                                    int n_q_heads, int n_kv_heads, int gqa,
                                                    int head_dim, int chunk) {
  FD_GUARD;
  fd_phase2_entry<DH>(scratch, o, L, chunk, n_q_heads);
}

#undef FD_P1_EXPORTS
#undef FD_GUARD

// ============================================================================
// Multi-query flash decode for speculative-decoding verification: M query
// tokens share one KV cache, block-causal among the M new positions. The
// per-position online-softmax machinery is identical to the single-token
// kernels above.
//
// Semantics:
//   The KV caches hold prefix positions [0, L) PLUS the M new tokens'
//   positions [L, L+M) (written by the qkv path before attention); total
//   extent L+M. Query token i (0-based, 0 <= i < M) attends exactly the
//   positions [0, L+i] inclusive: all prefix plus new positions 0..i.
//
//   q  : bf16 [M][16][128]; row of (query qi, head qh) at
//        q + qi*q_tok_stride + qh*q_head_stride  (strides in bf16 elements)
//   K,V: same layouts and stride rules as the single-token kernels
//   o  : bf16 [M][16][128]; row of (qi, qh) at
//        o + qi*o_tok_stride + qh*o_head_stride
//
// Two-phase split-KV (flash decoding), same structure as the single-token
// kernels:
//   phase 1: grid (nchunks, n_q_heads/QP, M); block (ch, y, qi) runs an online
//            softmax over chunk ch for QP q-heads that share one kv head of
//            query token qi and writes the partial (m, l, acc[128]) to caller
//            scratch. The causal mask is folded into the trip count:
//            pos_end = min(pos_beg + chunk, L + qi + 1); positions p > L+qi
//            get score -INFINITY, an exact no-op in the online softmax
//            (pf = expf(-inf - mn) = 0, cf = expf(mx - mn) = 1 with the finite
//            -3e38 sentinel keeping m - m' NaN-free). Chunks fully above a
//            query's range run zero iterations and still write their partial
//            as the sentinel, so phase 2 always reads exactly nchunks defined
//            partials per (query, head) and merges dead chunks as no-ops.
//   phase 2: grid (n_q_heads*M); block eh = qi*n_q_heads + qh merges the
//            nchunks partials of one (query, head) pair by LSE rescaling and
//            writes o. Same merge math as the single-token kernel.
//
// Scratch (caller-allocated fp32, one buffer; NQ = n_q_heads = 16,
// NC = nchunks = ceil((L+M)/chunk), row index eh = qi*NQ + qh):
//   scratch[0         .. M*NQ*NC)          = m   [M*NQ][NC]
//   scratch[M*NQ*NC   .. 2*M*NQ*NC)        = l   [M*NQ][NC]
//   scratch[2*M*NQ*NC .. M*NQ*NC*(2+128))  = acc [M*NQ][NC][128]
//   Required size: M * n_q_heads * nchunks * (2 + 128) floats.
//   Every partial is written by exactly one block; every o row is written by
//   exactly one phase-2 block.
//
// CUDA Graph: the graph-safe entry points (non-_host) read L from a device
// scalar (const int* len_dev); the engine updates that scalar to change L
// between graph replays. M is an ordinary kernel argument: fixed per capture,
// and MUST be identical in both phases (both derive nchunks = ceil((L+M)/
// chunk) from L, M, chunk). The _host variants take L by value. No global
// state; every buffer is caller-allocated and passed per launch.
// Capacity-sized graphs: phase 1 may be launched with grid.x = nchunks_max
// while the replayed L needs fewer chunks; blocks with blockIdx.x >=
// ceil((L+M)/chunk) return before writing anything.
//
// Phase-1 launch REQUIRES dynamic shared memory for the cross-warp merge:
//   smem  = qpack * (threads/32) * 130 * 4 bytes,  smem <= 48 KB
//   block = threads, a multiple of 32
//   grid  = (nchunks, n_q_heads/qpack, M)
// Phase-2 launch:
//   grid  = (n_q_heads * M)
//   block = threads2, a multiple of 128 (<= 1024)
//   smem  = (2*nchunks + threads2) * 4 bytes,  smem <= 48 KB
// nchunks must be identical in both phases for the same (L, M, chunk).
//
// SIGNATURE TABLE (all symbols are extern "C" __global__; all pointer args
// are __restrict__; QP = q-heads per block; nchunks = ceil((L+M)/chunk);
// grid.y = 16/QP, grid.z = M for phase 1).
// Failure semantics: wrong-shape calls or M outside [1,16] __trap(); phase-1
// grid.x may exceed nchunks (capacity-sized graphs) -- excess blocks exit
// before any write.
//
//   flash_decode_mq_phase1_q1  -- QP=1
//     void(const __nv_bfloat16* q, const __nv_bfloat16* k,
//          const __nv_bfloat16* v, float* scratch, const int* len_dev, int M,
//          int n_q_heads, int n_kv_heads, int gqa, int head_dim,
//          int q_tok_stride, int q_head_stride,
//          int k_head_stride, int k_pos_stride,
//          int v_head_stride, int v_pos_stride, int chunk)
//     launch: grid = dim3(nchunks, 16, M), block = threads (multiple of 32),
//             smem = 1*(threads/32)*130*4 bytes, <= 49152
//   flash_decode_mq_phase1_q1_host -- QP=1
//     void(const __nv_bfloat16* q, const __nv_bfloat16* k,
//          const __nv_bfloat16* v, float* scratch, int L, int M,
//          int n_q_heads, int n_kv_heads, int gqa, int head_dim,
//          int q_tok_stride, int q_head_stride,
//          int k_head_stride, int k_pos_stride,
//          int v_head_stride, int v_pos_stride, int chunk)
//     launch: same as flash_decode_mq_phase1_q1
//   flash_decode_mq_phase1_q2  -- QP=2
//     void(... same argument list as flash_decode_mq_phase1_q1 ...)
//     launch: grid = dim3(nchunks, 8, M), block = threads,
//             smem = 2*(threads/32)*130*4 bytes, <= 49152
//   flash_decode_mq_phase1_q2_host -- QP=2, L by value
//   flash_decode_mq_phase1_q4  -- QP=4
//     launch: grid = dim3(nchunks, 4, M), block = threads,
//             smem = 4*(threads/32)*130*4 bytes, <= 49152
//   flash_decode_mq_phase1_q4_host -- QP=4, L by value
//   flash_decode_mq_phase1_q8  -- QP=8
//     launch: grid = dim3(nchunks, 2, M), block = threads,
//             smem = 8*(threads/32)*130*4 bytes, <= 49152
//   flash_decode_mq_phase1_q8_host -- QP=8, L by value
//   flash_decode_mq_phase2
//     void(float* scratch, __nv_bfloat16* o, const int* len_dev, int M,
//          int n_q_heads, int n_kv_heads, int gqa, int head_dim,
//          int o_tok_stride, int o_head_stride, int chunk)
//     launch: grid = dim3(16*M), block = threads2 (multiple of 128, <= 1024),
//             smem = (2*nchunks + threads2)*4 bytes, <= 49152
//   flash_decode_mq_phase2_host
//     void(float* scratch, __nv_bfloat16* o, int L, int M,
//          int n_q_heads, int n_kv_heads, int gqa, int head_dim,
//          int o_tok_stride, int o_head_stride, int chunk)
//     launch: same as flash_decode_mq_phase2
// ============================================================================

constexpr int MQ_MAX = 16;              // guard range for query tokens: 1..MQ_MAX

// ------------------------------------------------- multi-query phase 1 ----
// QP = q heads per block (all sharing one kv head), U = key positions per block
// iteration per lane group (software pipeline depth). blockIdx.z = qi selects
// the query token; the causal mask is folded into pos_end.
template <int DH, int QP, int U>
__device__ __forceinline__ void fdmq_phase1_body(
    const bf16* __restrict__ q, const bf16* __restrict__ Kc, const bf16* __restrict__ Vc,
    float* __restrict__ p_m, float* __restrict__ p_l, float* __restrict__ p_o,
    int L, int chunk, int nchunks, int n_q_heads, int gqa,
    int q_tok_stride, int q_head_stride,
    int k_head_stride, int k_pos_stride, int v_head_stride, int v_pos_stride) {
  constexpr int VPR = DH / TP;
  const int ch      = blockIdx.x;
  const int qh0     = blockIdx.y * QP;
  const int qi      = blockIdx.z;                     // query token, 0 <= qi < M
  const int kv      = qh0 / gqa;                      // QP divides gqa => all QP heads share kv
  const int pos_beg = ch * chunk;
  // Query qi attends [0, L+qi] inclusive; p > L+qi is masked (dead slot below).
  const int pos_end = min(pos_beg + chunk, L + qi + 1);

  const int tid  = threadIdx.x;
  const int lane = tid % TP;
  const int grp  = tid / TP;
  const int dim0 = lane * VPR;
  const int ngrp = blockDim.x / TP;

  const bf16* __restrict__ Krow = Kc + kv * k_head_stride + dim0;
  const bf16* __restrict__ Vrow = Vc + kv * v_head_stride + dim0;
  const bf16* __restrict__ Qrow = q + (size_t)qi * q_tok_stride + dim0;

  float qr[QP][VPR];
  float acc[QP][VPR];
  float mx[QP], ls[QP];
#pragma unroll
  for (int h = 0; h < QP; ++h) {
    ld_vec<VPR>(Qrow + (qh0 + h) * q_head_stride, qr[h]);
    mx[h] = -3.0e38f;                              // finite sentinel: never NaN in m - m'
    ls[h] = 0.0f;
#pragma unroll
    for (int j = 0; j < VPR; ++j) acc[h][j] = 0.0f;
  }

  // Block-uniform trip count so every lane stays converged for the shuffles.
  // niter = 0 for chunks fully above this query's range: the block then writes
  // the sentinel partial (m = -3e38, l = 0, acc = 0), a no-op for phase 2.
  const int step  = ngrp * U;
  const int niter = (pos_end > pos_beg) ? ((pos_end - pos_beg + step - 1) / step) : 0;

  for (int it = 0; it < niter; ++it) {
    const int base = pos_beg + grp + it * step;

    // Issue all U K loads before touching any of them (memory parallelism).
    float kf[U][VPR];
    bool  ok[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int pos = base + u * ngrp;
      ok[u] = (pos < pos_end);
      ld_vec<VPR>(Krow + (size_t)(ok[u] ? pos : 0) * k_pos_stride, kf[u]);
    }

#pragma unroll
    for (int u = 0; u < U; ++u) {
      float s[QP];
#pragma unroll
      for (int h = 0; h < QP; ++h) s[h] = 0.0f;
#pragma unroll
      for (int j = 0; j < VPR; ++j)
#pragma unroll
        for (int h = 0; h < QP; ++h) s[h] = fmaf(qr[h][j], kf[u][j], s[h]);

      // Butterfly reduce the dot product across the TP lanes of this group.
      // off < TP <= 16, so a warp's two groups reduce independently.
#pragma unroll
      for (int h = 0; h < QP; ++h)
#pragma unroll
        for (int off = TP / 2; off > 0; off >>= 1)
          s[h] += __shfl_xor_sync(0xffffffffu, s[h], off);

      float cf[QP], pf[QP];
#pragma unroll
      for (int h = 0; h < QP; ++h) {
        const float sp = ok[u] ? s[h] * SCALE : -INFINITY;   // dead slot -> no-op softmax step
        const float mn = fmaxf(mx[h], sp);
        cf[h] = __expf(mx[h] - mn);
        pf[h] = __expf(sp - mn);
        mx[h] = mn;
        ls[h] = ls[h] * cf[h] + pf[h];
      }

      const int pos = base + u * ngrp;
      float vf[VPR];
      ld_vec<VPR>(Vrow + (size_t)(ok[u] ? pos : 0) * v_pos_stride, vf);
#pragma unroll
      for (int h = 0; h < QP; ++h)
#pragma unroll
        for (int j = 0; j < VPR; ++j)
          acc[h][j] = fmaf(acc[h][j], cf[h], pf[h] * vf[j]);
    }
  }

  // ---- Merge the per-group partials into one block-level partial. ----
  // Groups processed disjoint position sets, so this is an online-softmax
  // merge, not a plain sum: M=max, rescale by exp(m-M), then accumulate.
  // Level 1: the two TP-lane groups inside each warp exchange via shuffles
  // (butterfly: both halves end up holding the merged value).
#pragma unroll
  for (int h = 0; h < QP; ++h) {
    const float pm = __shfl_xor_sync(0xffffffffu, mx[h], TP);
    const float pl = __shfl_xor_sync(0xffffffffu, ls[h], TP);
    const float M  = fmaxf(mx[h], pm);
    const float c  = __expf(mx[h] - M);        // 1 when both sentinels
    const float pc = __expf(pm - M);
    float pa[VPR];
#pragma unroll
    for (int j = 0; j < VPR; ++j) pa[j] = __shfl_xor_sync(0xffffffffu, acc[h][j], TP);
    ls[h] = ls[h] * c + pl * pc;
#pragma unroll
    for (int j = 0; j < VPR; ++j) acc[h][j] = acc[h][j] * c + pa[j] * pc;
    mx[h] = M;
  }
  // Level 2: cross-warp merge through dynamic shared memory.
  // smem layout: acc [QP][nw][DH] | m [QP][nw] | l [QP][nw].
  const int warp   = tid / 32;
  const int lane32 = tid % 32;
  const int nw     = blockDim.x / 32;
  extern __shared__ float fdmq_sm[];
  float* sa  = fdmq_sm;                            // [QP][nw][DH]
  float* smm = sa + QP * nw * DH;                  // [QP][nw]
  float* sll = smm + QP * nw;                      // [QP][nw]
  if (lane32 < TP) {   // one group per warp writes (both halves hold the same merged shard)
#pragma unroll
    for (int h = 0; h < QP; ++h) {
      if (lane32 == 0) { smm[h * nw + warp] = mx[h]; sll[h * nw + warp] = ls[h]; }
#pragma unroll
      for (int j = 0; j < VPR; ++j) sa[(h * nw + warp) * DH + lane32 * VPR + j] = acc[h][j];
    }
  }
  __syncthreads();
  // Only lanes 0..TP-1 of warp 0 merge across warps and store the partial.
  if (warp != 0 || lane32 >= TP) return;
#pragma unroll
  for (int h = 0; h < QP; ++h) {
    for (int w = 1; w < nw; ++w) {
      const float pm = smm[h * nw + w], pl = sll[h * nw + w];
      const float M  = fmaxf(mx[h], pm);
      const float c  = __expf(mx[h] - M), pc = __expf(pm - M);
      ls[h] = ls[h] * c + pl * pc;
#pragma unroll
      for (int j = 0; j < VPR; ++j)
        acc[h][j] = acc[h][j] * c + sa[(h * nw + w) * DH + lane * VPR + j] * pc;
      mx[h] = M;
    }
  }

  // Partials: m,l are [M*n_q_heads][nchunks] (phase 2 reads them coalesced),
  // acc is [M*n_q_heads][nchunks][head_dim]; row eh = qi*n_q_heads + qh.
#pragma unroll
  for (int h = 0; h < QP; ++h) {
    const size_t part = ((size_t)qi * n_q_heads + (qh0 + h)) * nchunks + ch;
    if (lane == 0) { p_m[part] = mx[h]; p_l[part] = ls[h]; }
    st_vec<VPR>(p_o + part * DH + dim0, acc[h]);
  }
}

// ------------------------------------------------- multi-query phase 2 ----
// One block per (query, head) pair; blockIdx.x = eh = qi*n_q_heads + qh.
// blockDim.x = DH * G2 where G2 groups split the chunk range
// (G2 = blockDim.x / DH).  Dynamic shared: 2*nchunks (m,l) + blockDim.
template <int DH>
__device__ __forceinline__ void fdmq_phase2_body(const float* __restrict__ p_m,
                                                 const float* __restrict__ p_l,
                                                 const float* __restrict__ p_o,
                                                 bf16* __restrict__ o, int nchunks,
                                                 int n_q_heads,
                                                 int o_tok_stride, int o_head_stride) {
  extern __shared__ float fdmq_sm2[];
  const int eh  = blockIdx.x;
  const int qi  = eh / n_q_heads;
  const int qh  = eh % n_q_heads;
  const int tid = threadIdx.x;
  const int nt  = blockDim.x;
  const int G2  = nt / DH;
  const int g   = tid / DH;
  const int d   = tid % DH;

  float* sm_ = fdmq_sm2;
  float* sl_ = fdmq_sm2 + nchunks;
  float* red = fdmq_sm2 + 2 * nchunks;

  for (int h = tid; h < nchunks; h += nt) {
    sm_[h] = p_m[eh * nchunks + h];
    sl_[h] = p_l[eh * nchunks + h];
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
  const float M = red[0];

  float pv = 0.0f;
  for (int h = tid; h < nchunks; h += nt) pv += sl_[h] * __expf(sm_[h] - M);
  __syncthreads();      // every thread has consumed red[0] before red[] is reused
  red[tid] = pv;
  __syncthreads();
  for (int s = nt / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  const float lse = red[0];                        // sum_h l_h * exp(m_h - M)
  __syncthreads();      // same barrier before red[] is reused by the G2 reduction

  float acc = 0.0f;
  for (int h = g; h < nchunks; h += G2) {
    const float w = __expf(sm_[h] - M);
    acc = fmaf(w, p_o[((size_t)eh * nchunks + h) * DH + d], acc);
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

// ============================================ exported multi-query ABI =====
#define FD_GUARD_MQ do { if (head_dim != DH || n_q_heads != HQ || n_kv_heads != HKV || gqa != GQA || M < 1 || M > MQ_MAX) __trap(); } while (0)

template <int DH, int QP>
__device__ __forceinline__ void fdmq_phase1_entry(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
    float* __restrict__ scratch, int L, int M, int chunk, int n_q_heads, int gqa,
    int q_tok_stride, int q_head_stride,
    int k_head_stride, int k_pos_stride, int v_head_stride, int v_pos_stride) {
  constexpr int U = (QP == 1) ? 2 : 1;            // pipeline depth baked per variant
  const int nchunks = (L + M + chunk - 1) / chunk;
  // Capacity-sized graph launches may carry more CTAs than this L needs.
  // Block-uniform, before any __syncthreads: excess blocks must not write
  // partials (part = (qi*NQ+qh)*nchunks + ch would index into the next row).
  if (blockIdx.x >= nchunks) return;
  float* p_m = scratch;
  float* p_l = scratch + (size_t)M * n_q_heads * nchunks;
  float* p_o = scratch + 2 * (size_t)M * n_q_heads * nchunks;
  fdmq_phase1_body<DH, QP, U>(q, k, v, p_m, p_l, p_o, L, chunk, nchunks, n_q_heads, gqa,
                              q_tok_stride, q_head_stride,
                              k_head_stride, k_pos_stride, v_head_stride, v_pos_stride);
}

#define FD_MQ_P1_EXPORTS(QP)                                                                \
extern "C" __global__ void flash_decode_mq_phase1_q##QP(                                    \
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,      \
    float* __restrict__ scratch, const int* __restrict__ len_dev, int M,                    \
    int n_q_heads, int n_kv_heads, int gqa, int head_dim,                                   \
    int q_tok_stride, int q_head_stride,                                                    \
    int k_head_stride, int k_pos_stride, int v_head_stride, int v_pos_stride, int chunk) {  \
  FD_GUARD_MQ;                                                                              \
  fdmq_phase1_entry<DH, QP>(q, k, v, scratch, *len_dev, M, chunk, n_q_heads, gqa,          \
                            q_tok_stride, q_head_stride,                                    \
                            k_head_stride, k_pos_stride, v_head_stride, v_pos_stride);      \
}                                                                                           \
extern "C" __global__ void flash_decode_mq_phase1_q##QP##_host(                             \
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,      \
    float* __restrict__ scratch, int L, int M,                                               \
    int n_q_heads, int n_kv_heads, int gqa, int head_dim,                                   \
    int q_tok_stride, int q_head_stride,                                                    \
    int k_head_stride, int k_pos_stride, int v_head_stride, int v_pos_stride, int chunk) {  \
  FD_GUARD_MQ;                                                                              \
  fdmq_phase1_entry<DH, QP>(q, k, v, scratch, L, M, chunk, n_q_heads, gqa,                 \
                            q_tok_stride, q_head_stride,                                    \
                            k_head_stride, k_pos_stride, v_head_stride, v_pos_stride);      \
}

FD_MQ_P1_EXPORTS(1)
FD_MQ_P1_EXPORTS(2)
FD_MQ_P1_EXPORTS(4)
FD_MQ_P1_EXPORTS(8)

template <int DH>
__device__ __forceinline__ void fdmq_phase2_entry(float* __restrict__ scratch,
                                                  bf16* __restrict__ o, int L, int M,
                                                  int chunk, int n_q_heads,
                                                  int o_tok_stride, int o_head_stride) {
  const int nchunks = (L + M + chunk - 1) / chunk;
  const float* p_m = scratch;
  const float* p_l = scratch + (size_t)M * n_q_heads * nchunks;
  const float* p_o = scratch + 2 * (size_t)M * n_q_heads * nchunks;
  fdmq_phase2_body<DH>(p_m, p_l, p_o, o, nchunks, n_q_heads, o_tok_stride, o_head_stride);
}

extern "C" __global__ void flash_decode_mq_phase2(float* __restrict__ scratch,
                                                  bf16* __restrict__ o,
                                                  const int* __restrict__ len_dev, int M,
                                                  int n_q_heads, int n_kv_heads, int gqa,
                                                  int head_dim,
                                                  int o_tok_stride, int o_head_stride,
                                                  int chunk) {
  FD_GUARD_MQ;
  fdmq_phase2_entry<DH>(scratch, o, *len_dev, M, chunk, n_q_heads,
                        o_tok_stride, o_head_stride);
}

extern "C" __global__ void flash_decode_mq_phase2_host(float* __restrict__ scratch,
                                                       bf16* __restrict__ o, int L, int M,
                                                       int n_q_heads, int n_kv_heads, int gqa,
                                                       int head_dim,
                                                       int o_tok_stride, int o_head_stride,
                                                       int chunk) {
  FD_GUARD_MQ;
  fdmq_phase2_entry<DH>(scratch, o, L, M, chunk, n_q_heads,
                        o_tok_stride, o_head_stride);
}

#undef FD_MQ_P1_EXPORTS
#undef FD_GUARD_MQ
