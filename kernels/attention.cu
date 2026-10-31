// GQA flash-decoding attention: single-token decode (flash_decode_*) and M-token
// block-causal verification (flash_decode_mq_*).
// Shape: head_dim 128, n_q_heads = gqa * n_kv_heads, gqa % QP == 0, bf16 data, fp32
// accumulation, scale = 1/sqrt(128). FD_GUARD calls __trap() on any other shape.
//   q  : bf16 [n_q_heads][128]
//   K,V: bf16, separate tensors, head_dim-contiguous rows
//   o  : bf16 [n_q_heads][128]
// Phase 1: grid (nchunks, n_q_heads/QP); one block runs an online softmax over `chunk`
//   KV positions for QP q-heads sharing one kv head and writes the partial
//   (m, l, acc[128]) to scratch; per-group partials merge by intra-warp shuffle, then
//   cross-warp through dynamic smem.
// Phase 2: grid (n_q_heads); one block merges the nchunks partials of one q head by
//   LSE rescaling and writes o.
// Phase-1 thread map: TP=16 lanes per key position; lane l owns dims [l*8, l*8+8).
// KV address: kv*head_stride + pos*pos_stride + d; strides in bf16 elements, multiples of 8.
//   position-major [pos][kv][d]: head_stride = 128,    pos_stride = 256
//   head-major     [kv][pos][d]: head_stride = Lh*128, pos_stride = 128
// Scratch (fp32; NQ = n_q_heads, NC = nchunks = ceil(L/chunk)):
//   m [NQ][NC] | l [NQ][NC] | acc [NQ][NC][128]; size NQ*NC*(2+128) floats.
// Entry points read L from the device scalar len_dev.
// Phase-1 grid.x may exceed nchunks; blocks with blockIdx.x >= nchunks return before any write.
// Phase-1 launch: block = threads (multiple of 32), grid = (nchunks, n_q_heads/qpack),
//   smem = qpack*(threads/32)*(128+2)*4 bytes <= 48 KB.
// Phase-2 launch: grid = n_q_heads, block = threads2 (power of two in [128, 1024]),
//   smem = (2*nchunks + threads2)*4 bytes <= 48 KB.
// nchunks is identical in both phases for the same (L, chunk).
#include "gpu_compat.cuh"
#include <stdint.h>

using bf16 = __nv_bfloat16;

// One CTA/query/head; BF16 projections and relative table, FP32 logits and weighted values.
extern "C" __global__ void relative_window_attention_d128(
    const bf16* q, const bf16* k, const bf16* v, const bf16* relative, const bf16* dim_scale,
    const uint8_t* mask, bf16* output, uint32_t rows, uint32_t heads,
    uint32_t chunk, uint32_t left, float q_scale, float k_scale, float cap, float invalid) {
    const uint32_t context = chunk + left - 1, relative_rows = context / 2 + 1;
    if (blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 ||
        !q || !k || !v || !relative || !dim_scale || !mask || !output || !rows || !heads ||
        !chunk || left < 2 || context > 32 || left - 1 > relative_rows || !(cap > 0.0f) ||
        gridDim.x != rows || gridDim.y != heads || gridDim.z != 1) { __trap(); return; }
    const uint32_t row = blockIdx.x, head = blockIdx.y, d = threadIdx.x, hidden = heads * 128;
    const int first = int(row / chunk * chunk) - int(left) + 1;
    __shared__ float reduction[8], probabilities[32];
    const float query = __fmul_rn(__fmul_rn(__bfloat162float(q[uint64_t(row)*hidden+head*128+d]), q_scale),
        __bfloat162float(dim_scale[d]));
    for (uint32_t c = 0; c < context; ++c) {
        const int key = first + int(c), lag = int(row) - key;
        const bool valid = key >= 0 && key < int(rows) && mask[key] && lag >= 0 && lag < int(left)-1;
        float ac = 0.0f, bd = 0.0f;
        if (valid) {
            ac = __fmul_rn(query, __fmul_rn(__bfloat162float(k[uint64_t(key)*hidden+head*128+d]), k_scale));
            bd = __fmul_rn(query, __bfloat162float(relative[uint64_t(int(relative_rows)-1-lag)*hidden+head*128+d]));
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            ac += __shfl_down_sync(0xffffffffu, ac, offset);
            bd += __shfl_down_sync(0xffffffffu, bd, offset);
        }
        if ((d & 31) == 0) { reduction[d/32] = ac; reduction[4+d/32] = bd; }
        __syncthreads();
        if (d == 0) {
            const float content = (reduction[0]+reduction[1])+(reduction[2]+reduction[3]);
            const float position = (reduction[4]+reduction[5])+(reduction[6]+reduction[7]);
            probabilities[c] = valid ? cap*tanhf((content+position)/cap) : invalid;
        }
        __syncthreads();
    }
    if (d == 0) {
        float peak = -INFINITY;
        for (uint32_t c = 0; c < context; ++c) peak = fmaxf(peak, probabilities[c]);
        float sum = 0.0f;
        for (uint32_t c = 0; c < context; ++c) { probabilities[c] = expf(probabilities[c]-peak); sum += probabilities[c]; }
        for (uint32_t c = 0; c < context; ++c) probabilities[c] /= sum;
    }
    __syncthreads();
    float result = 0.0f;
    for (uint32_t c = 0; c < context; ++c) {
        const int key = first + int(c);
        if (key >= 0 && key < int(rows)) result = __fmaf_rn(probabilities[c],
            __bfloat162float(v[uint64_t(key)*hidden+head*128+d]), result);
    }
    output[uint64_t(row)*hidden+head*128+d] = __float2bfloat16_rn(result);
}

constexpr int DH  = 128;                // head_dim
constexpr int TP  = 16;                 // lanes cooperating on one key position
constexpr float SCALE = 0.08838834764831845f;   // 1/sqrt(128)

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
// VPR bf16 in, VPR fp32 out; 16B loads when VPR % 8 == 0, else 8B loads.
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

// QP = q heads per block sharing one kv head; U = key positions per lane group per iteration.
template <int DH, int QP, int U, bool WINDOW>
__device__ __forceinline__ void fd_phase1_body(
    const bf16* __restrict__ q, const bf16* __restrict__ Kc, const bf16* __restrict__ Vc,
    float* __restrict__ p_m, float* __restrict__ p_l, float* __restrict__ p_o,
    int L, int chunk, int nchunks, int gqa,
    int k_head_stride, int k_pos_stride, int v_head_stride, int v_pos_stride,
    int first_pos, int ring_tokens, float scale = SCALE) {
  constexpr int TP = DH == 256 ? 32 : 16;
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
  const int ngrp = DH == 256 ? 4 : blockDim.x / TP;
  extern __shared__ float fd_sm[];
  if constexpr (DH == 256) {
    bf16* shared_q = reinterpret_cast<bf16*>(fd_sm);
    if (tid < 32) {
#pragma unroll
      for (int h = 0; h < QP; ++h)
        reinterpret_cast<uint4*>(shared_q + h * DH)[tid] =
            reinterpret_cast<const uint4*>(q + (qh0 + h) * DH)[tid];
    }
    __syncthreads();
  }

  const bf16* __restrict__ Krow = Kc + kv * k_head_stride + dim0;
  const bf16* __restrict__ Vrow = Vc + kv * v_head_stride + dim0;

  float qr[QP][VPR];
  float acc[QP][VPR];
  float mx[QP], ls[QP];
#pragma unroll
  for (int h = 0; h < QP; ++h) {
    if constexpr (DH == 256)
      ld_vec<VPR>(reinterpret_cast<bf16*>(fd_sm) + h * DH + dim0, qr[h]);
    else
      ld_vec<VPR>(q + (qh0 + h) * DH + dim0, qr[h]);
    mx[h] = -3.0e38f;                              // finite sentinel: never NaN in m - m'
    ls[h] = 0.0f;
#pragma unroll
    for (int j = 0; j < VPR; ++j) acc[h][j] = 0.0f;
  }

  // Trip count is block-uniform; all lanes are converged at the shuffles.
  const int step  = ngrp * U;
  const int niter = (pos_end > pos_beg) ? ((pos_end - pos_beg + step - 1) / step) : 0;

  for (int it = 0; it < niter; ++it) {
    const int base = pos_beg + grp + it * step;

    // All U K loads issue before the first use.
    float kf[U][VPR];
    bool  ok[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int pos = base + u * ngrp;
      ok[u] = (pos < pos_end);
      int cache_pos = pos;
      if constexpr (WINDOW) {
        cache_pos = first_pos + pos;
        if (cache_pos >= ring_tokens) cache_pos -= ring_tokens;
      }
      ld_vec<VPR>(Krow + (size_t)(ok[u] ? cache_pos : 0) * k_pos_stride, kf[u]);
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

      // Butterfly over the TP lanes of one group; off < TP keeps a warp's two groups independent.
#pragma unroll
      for (int h = 0; h < QP; ++h)
#pragma unroll
        for (int off = TP / 2; off > 0; off >>= 1)
          s[h] += __shfl_xor_sync(FF_WARP_MASK, s[h], off);

      float cf[QP], pf[QP];
#pragma unroll
      for (int h = 0; h < QP; ++h) {
        if constexpr (DH == 256) { cf[h] = 0.0f; pf[h] = 0.0f; }
        if (DH != 256 || lane == 0) {
          const float sp = ok[u] ? s[h] * scale : -INFINITY;
          const float mn = fmaxf(mx[h], sp);
          cf[h] = __expf(mx[h] - mn);
          pf[h] = __expf(sp - mn);
          mx[h] = mn;
          ls[h] = ls[h] * cf[h] + pf[h];
        }
        if constexpr (DH == 256) {
          cf[h] = __shfl_sync(FF_WARP_MASK, cf[h], 0);
          pf[h] = __shfl_sync(FF_WARP_MASK, pf[h], 0);
        }
      }

      const int pos = base + u * ngrp;
      int cache_pos = pos;
      if constexpr (WINDOW) {
        cache_pos = first_pos + pos;
        if (cache_pos >= ring_tokens) cache_pos -= ring_tokens;
      }
      float vf[VPR];
      ld_vec<VPR>(Vrow + (size_t)(ok[u] ? cache_pos : 0) * v_pos_stride, vf);
#pragma unroll
      for (int h = 0; h < QP; ++h)
#pragma unroll
        for (int j = 0; j < VPR; ++j)
          acc[h][j] = fmaf(acc[h][j], cf[h], pf[h] * vf[j]);
    }
  }

  // Online-softmax merge of the per-group partials (disjoint position sets).
  // Level 1: the two TP-lane groups of a warp exchange via shuffles; both halves hold the result.
  if constexpr (TP == 16) {
#pragma unroll
  for (int h = 0; h < QP; ++h) {
    const float pm = __shfl_xor_sync(FF_WARP_MASK, mx[h], TP);
    const float pl = __shfl_xor_sync(FF_WARP_MASK, ls[h], TP);
    const float M  = fmaxf(mx[h], pm);
    const float c  = __expf(mx[h] - M);        // 1 when both sentinels
    const float pc = __expf(pm - M);
    float pa[VPR];
#pragma unroll
    for (int j = 0; j < VPR; ++j) pa[j] = __shfl_xor_sync(FF_WARP_MASK, acc[h][j], TP);
    ls[h] = ls[h] * c + pl * pc;
#pragma unroll
    for (int j = 0; j < VPR; ++j) acc[h][j] = acc[h][j] * c + pa[j] * pc;
    mx[h] = M;
  }
  } else {
#pragma unroll
    for (int h = 0; h < QP; ++h) {
      mx[h] = __shfl_sync(FF_WARP_MASK, mx[h], 0);
      ls[h] = __shfl_sync(FF_WARP_MASK, ls[h], 0);
    }
  }
  // Level 2: cross-warp merge through dynamic shared memory.
  const int warp   = tid / 32;
  const int lane32 = tid % 32;
  const int nw     = DH == 256 ? 4 : blockDim.x / 32;
  float* sa  = fd_sm + (DH == 256 ? QP * DH / 2 : 0); // [QP][nw][DH]
  float* smm = sa + QP * nw * DH;                  // [QP][nw]
  float* sll = smm + QP * nw;                      // [QP][nw]
  if (lane32 < TP) {   // one group per warp writes; both halves hold the same merged shard
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

  // Partials: m, l [n_q_heads][nchunks]; acc [n_q_heads][nchunks][head_dim].
#pragma unroll
  for (int h = 0; h < QP; ++h) {
    const size_t part = (size_t)(qh0 + h) * nchunks + ch;
    if (lane == 0) { p_m[part] = mx[h]; p_l[part] = ls[h]; }
    st_vec<VPR>(p_o + part * DH + dim0, acc[h]);
  }
}

// One block per q head; blockDim.x = DH*G2, G2 groups split the chunk range.
// Dynamic smem: 2*nchunks (m, l) + blockDim.x floats.
template <int DH, bool SINK>
__device__ __forceinline__ void fd_phase2_body(const float* __restrict__ p_m,
                                               const float* __restrict__ p_l,
                                               const float* __restrict__ p_o,
                                               const bf16* __restrict__ sinks,
                                               bf16* __restrict__ o, int nchunks,
                                               float* __restrict__ output_lse = nullptr) {
  extern __shared__ float sm[];
  const int qh  = blockIdx.x;
  const int tid = threadIdx.x;
  const int nt  = DH == 256 ? 256 : blockDim.x;
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
  float sink = -3.0e38f;
  if constexpr (SINK) sink = __bfloat162float(sinks[qh]);
  float M = red[0];
  if constexpr (SINK) M = fmaxf(M, sink);

  float pv = 0.0f;
  for (int h = tid; h < nchunks; h += nt) pv += sl_[h] * __expf(sm_[h] - M);
  if constexpr (SINK) {
    if (tid == 0) pv += __expf(sink - M);
  }
  __syncthreads();      // all threads have read red[0] before red[] is reused
  red[tid] = pv;
  __syncthreads();
  for (int s = nt / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  const float lse = red[0];                        // sum_h l_h * exp(m_h - M)
  __syncthreads();      // all threads have read red[0] before the G2 reduction reuses red[]

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
  if (output_lse && tid == 0) output_lse[qh] = M + logf(lse);
}

#define FD_RING_256(QP) \
extern "C" __global__ void flash_decode_ring_d256_q##QP( \
    const bf16* q,const bf16* k,const bf16* v,float* scratch, \
    const int* position,int heads,int kv_heads,int ring,int chunk,float scale) { \
  const int p = *position; \
  if (blockDim.x != 128 || p < 0 || p == 2147483647 || ring <= 0 || chunk <= 0 || \
      heads <= 0 || kv_heads <= 0 || heads % kv_heads || \
      (heads / kv_heads) % QP || !(scale > 0.0f)) {__trap();return;} \
  const int length = min(p + 1,ring); \
  const int parts = (length + chunk - 1) / chunk; \
  if (blockIdx.x >= parts) return; \
  fd_phase1_body<256,QP,(QP == 1 ? 2 : 1),true>( \
      q,k,v,scratch,scratch + heads * parts,scratch + 2 * heads * parts, \
      length,chunk,parts,heads / kv_heads,ring * 256,256,ring * 256,256, \
      (p + 1 - length) % ring,ring,scale); \
}
FD_RING_256(1)
FD_RING_256(2)
FD_RING_256(4)
FD_RING_256(8)

extern "C" __global__ void flash_decode_ring_d256_merge(
    const float* scratch,bf16* out,float* lse,const int* position,int heads,int ring,int chunk) {
  const int p = *position;
  if (blockDim.x != 256 || p < 0 || p == 2147483647 || ring <= 0 || chunk <= 0 || heads <= 0) {__trap();return;}
  const int parts = (min(p + 1,ring) + chunk - 1) / chunk;
  fd_phase2_body<256,false>(scratch,scratch + heads * parts,
      scratch + 2 * heads * parts,nullptr,out,parts,lse);
}

// QP is baked into the symbol name; every other dimension and stride is an argument.
#define FD_GUARD(QP) do { \
  if (head_dim != DH || n_q_heads <= 0 || n_kv_heads <= 0 || \
      (n_q_heads % n_kv_heads) != 0 || gqa != (n_q_heads / n_kv_heads) || \
      (gqa % (QP)) != 0) __trap(); \
} while (0)

template <int DH, int QP>
__device__ __forceinline__ void fd_phase1_entry(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
    float* __restrict__ scratch, int L, int chunk, int n_q_heads, int gqa,
    int k_head_stride, int k_pos_stride, int v_head_stride, int v_pos_stride) {
  constexpr int U = (QP == 1) ? 2 : 1;            // pipeline depth baked per variant
  const int nchunks = (L + chunk - 1) / chunk;
  // Block-uniform, before any __syncthreads: blocks past nchunks write no partials.
  if (blockIdx.x >= nchunks) return;
  float* p_m = scratch;
  float* p_l = scratch + (size_t)n_q_heads * nchunks;
  float* p_o = scratch + 2 * (size_t)n_q_heads * nchunks;
  fd_phase1_body<DH, QP, U, false>(
      q, k, v, p_m, p_l, p_o, L, chunk, nchunks, gqa,
      k_head_stride, k_pos_stride, v_head_stride, v_pos_stride, 0, 1);
}

template <int DH, int QP>
__device__ __forceinline__ void fd_window_phase1_entry(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
    float* __restrict__ scratch, int L, int chunk, int n_q_heads, int gqa,
    int k_head_stride, int k_pos_stride, int v_head_stride, int v_pos_stride,
    const int* __restrict__ position, int ring_tokens) {
  constexpr int U = (QP == 1) ? 2 : 1;
  const int nchunks = (L + chunk - 1) / chunk;
  if (blockIdx.x >= nchunks) return;
  const int logical_length = *position + 1;
  const int first_pos = (logical_length - L) % ring_tokens;
  float* p_m = scratch;
  float* p_l = scratch + (size_t)n_q_heads * nchunks;
  float* p_o = scratch + 2 * (size_t)n_q_heads * nchunks;
  fd_phase1_body<DH, QP, U, true>(
      q, k, v, p_m, p_l, p_o, L, chunk, nchunks, gqa,
      k_head_stride, k_pos_stride, v_head_stride, v_pos_stride,
      first_pos, ring_tokens);
}

#define FD_P1_EXPORTS(QP)                                                                   \
extern "C" __global__ void flash_decode_phase1_q##QP(                                       \
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,      \
    float* __restrict__ scratch, const int* __restrict__ len_dev,                           \
    int n_q_heads, int n_kv_heads, int gqa, int head_dim,                                   \
    int k_head_stride, int k_pos_stride, int v_head_stride, int v_pos_stride, int chunk) {  \
  FD_GUARD(QP);                                                                             \
  fd_phase1_entry<DH, QP>(q, k, v, scratch, *len_dev, chunk, n_q_heads, gqa,               \
                           k_head_stride, k_pos_stride, v_head_stride, v_pos_stride);       \
}

FD_P1_EXPORTS(1)
FD_P1_EXPORTS(2)
FD_P1_EXPORTS(4)
FD_P1_EXPORTS(8)

#define FD_WINDOW_P1_EXPORTS(QP)                                                              \
extern "C" __global__ void flash_decode_window_phase1_q##QP(                                  \
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,        \
    float* __restrict__ scratch, const int* __restrict__ len_dev,                             \
    int n_q_heads, int n_kv_heads, int gqa, int head_dim,                                     \
    int k_head_stride, int k_pos_stride, int v_head_stride, int v_pos_stride, int chunk,      \
    const int* position, int ring_tokens) {                                                    \
  FD_GUARD(QP);                                                                                \
  fd_window_phase1_entry<DH, QP>(                                                             \
      q, k, v, scratch, *len_dev, chunk, n_q_heads, gqa,                                      \
      k_head_stride, k_pos_stride, v_head_stride, v_pos_stride, position, ring_tokens);       \
}

FD_WINDOW_P1_EXPORTS(1)
FD_WINDOW_P1_EXPORTS(2)
FD_WINDOW_P1_EXPORTS(4)
FD_WINDOW_P1_EXPORTS(8)

extern "C" __global__ void attention_sink_rescale(
    const float* __restrict__ lse, const bf16* __restrict__ sinks,
    bf16* __restrict__ output, int rows, int heads) {
  if (rows < 1 || heads < 1 || (blockDim.x & 31) != 0) __trap();
  const long long extent = (long long)rows * heads * 128;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < extent;
       i += (long long)gridDim.x * blockDim.x) {
    const int row = (int)(i / (heads * 128));
    const int head = (int)(i / 128) % heads;
    const float logit = __bfloat162float(sinks[head]);
    const float denominator = 1.0f + __expf(logit - lse[(long long)head * rows + row]);
    output[i] = __float2bfloat16(__bfloat162float(output[i]) / denominator);
  }
}


template <int DH>
__device__ __forceinline__ void fd_phase2_entry(float* __restrict__ scratch,
                                                bf16* __restrict__ o, int L, int chunk,
                                                int n_q_heads) {
  const int nchunks = (L + chunk - 1) / chunk;
  const float* p_m = scratch;
  const float* p_l = scratch + (size_t)n_q_heads * nchunks;
  const float* p_o = scratch + 2 * (size_t)n_q_heads * nchunks;
  fd_phase2_body<DH, false>(p_m, p_l, p_o, nullptr, o, nchunks);
}

template <int DH>
__device__ __forceinline__ void fd_sink_phase2_entry(
    float* __restrict__ scratch, bf16* __restrict__ o, const bf16* __restrict__ sinks,
    int L, int chunk, int n_q_heads) {
  const int nchunks = (L + chunk - 1) / chunk;
  const float* p_m = scratch;
  const float* p_l = scratch + (size_t)n_q_heads * nchunks;
  const float* p_o = scratch + 2 * (size_t)n_q_heads * nchunks;
  fd_phase2_body<DH, true>(p_m, p_l, p_o, sinks, o, nchunks);
}

extern "C" __global__ void flash_decode_phase2(float* __restrict__ scratch,
                                                bf16* __restrict__ o,
                                                const int* __restrict__ len_dev,
                                                int n_q_heads, int n_kv_heads, int gqa,
                                                int head_dim, int chunk) {
  FD_GUARD(1);
  fd_phase2_entry<DH>(scratch, o, *len_dev, chunk, n_q_heads);
}

extern "C" __global__ void flash_decode_sink_phase2(float* __restrict__ scratch,
                                                    bf16* __restrict__ o,
                                                    const bf16* __restrict__ sinks,
                                                    const int* __restrict__ len_dev,
                                                    int n_q_heads, int n_kv_heads, int gqa,
                                                    int head_dim, int chunk) {
  FD_GUARD(1);
  fd_sink_phase2_entry<DH>(scratch, o, sinks, *len_dev, chunk, n_q_heads);
}

#undef FD_P1_EXPORTS
#undef FD_WINDOW_P1_EXPORTS
#undef FD_GUARD

// Multi-query flash decode: M query tokens share one KV cache, block-causal among the
// M new positions. KV holds positions [0, L+M); query qi (0 <= qi < M) attends
// [0, L+qi] inclusive.
//   q  : bf16 [M][16][128]; row (qi, qh) at q + qi*q_tok_stride + qh*q_head_stride (bf16 elements)
//   K,V: same layouts and stride rules as the single-token kernels
//   o  : bf16 [M][16][128]; row (qi, qh) at o + qi*o_tok_stride + qh*o_head_stride
// Phase 1: grid (nchunks, n_q_heads/QP, M); pos_end = min(pos_beg + chunk, L + qi + 1).
//   Chunks fully above a query's range run zero iterations and write the sentinel partial,
//   so phase 2 reads nchunks defined partials per (query, head).
// Phase 2: grid (n_q_heads*M); block eh = qi*n_q_heads + qh merges one (query, head) row.
// Scratch (fp32; NQ = 16, NC = nchunks = ceil((L+M)/chunk), row eh = qi*NQ + qh):
//   m [M*NQ][NC] | l [M*NQ][NC] | acc [M*NQ][NC][128]; size M*NQ*NC*(2+128) floats.
// M is identical in both phases; FD_GUARD_MQ calls __trap() on M outside [1, 16].
// Phase-1 grid.x may exceed nchunks; blocks with blockIdx.x >= nchunks return before any write.
// Phase-1 launch: block = threads (multiple of 32), grid = (nchunks, n_q_heads/qpack, M),
//   smem = qpack*(threads/32)*130*4 bytes <= 48 KB.
// Phase-2 launch: grid = n_q_heads*M, block = threads2 (power of two in [128, 1024]),
//   smem = (2*nchunks + threads2)*4 bytes <= 48 KB.

constexpr int MQ_MAX = 16;              // guard range for query tokens: 1..MQ_MAX

// QP = q heads per block sharing one kv head; U = key positions per lane group per
// iteration; blockIdx.z = query token.
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

  // Trip count is block-uniform; niter = 0 above this query's range writes the sentinel
  // partial (m = -3e38, l = 0, acc = 0).
  const int step  = ngrp * U;
  const int niter = (pos_end > pos_beg) ? ((pos_end - pos_beg + step - 1) / step) : 0;

  for (int it = 0; it < niter; ++it) {
    const int base = pos_beg + grp + it * step;

    // All U K loads issue before the first use.
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

      // Butterfly over the TP lanes of one group; off < TP keeps a warp's two groups independent.
#pragma unroll
      for (int h = 0; h < QP; ++h)
#pragma unroll
        for (int off = TP / 2; off > 0; off >>= 1)
          s[h] += __shfl_xor_sync(FF_WARP_MASK, s[h], off);

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

  // Online-softmax merge of the per-group partials (disjoint position sets).
  // Level 1: the two TP-lane groups of a warp exchange via shuffles; both halves hold the result.
#pragma unroll
  for (int h = 0; h < QP; ++h) {
    const float pm = __shfl_xor_sync(FF_WARP_MASK, mx[h], TP);
    const float pl = __shfl_xor_sync(FF_WARP_MASK, ls[h], TP);
    const float M  = fmaxf(mx[h], pm);
    const float c  = __expf(mx[h] - M);        // 1 when both sentinels
    const float pc = __expf(pm - M);
    float pa[VPR];
#pragma unroll
    for (int j = 0; j < VPR; ++j) pa[j] = __shfl_xor_sync(FF_WARP_MASK, acc[h][j], TP);
    ls[h] = ls[h] * c + pl * pc;
#pragma unroll
    for (int j = 0; j < VPR; ++j) acc[h][j] = acc[h][j] * c + pa[j] * pc;
    mx[h] = M;
  }
  // Level 2: cross-warp merge through dynamic shared memory.
  const int warp   = tid / 32;
  const int lane32 = tid % 32;
  const int nw     = blockDim.x / 32;
  extern __shared__ float fdmq_sm[];
  float* sa  = fdmq_sm;                            // [QP][nw][DH]
  float* smm = sa + QP * nw * DH;                  // [QP][nw]
  float* sll = smm + QP * nw;                      // [QP][nw]
  if (lane32 < TP) {   // one group per warp writes; both halves hold the same merged shard
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

  // Partials: m, l [M*n_q_heads][nchunks]; acc [M*n_q_heads][nchunks][head_dim];
  // row eh = qi*n_q_heads + qh.
#pragma unroll
  for (int h = 0; h < QP; ++h) {
    const size_t part = ((size_t)qi * n_q_heads + (qh0 + h)) * nchunks + ch;
    if (lane == 0) { p_m[part] = mx[h]; p_l[part] = ls[h]; }
    st_vec<VPR>(p_o + part * DH + dim0, acc[h]);
  }
}

// One block per (query, head); blockIdx.x = eh = qi*n_q_heads + qh; blockDim.x = DH*G2,
// G2 groups split the chunk range. Dynamic smem: 2*nchunks (m, l) + blockDim.x floats.
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
  __syncthreads();      // all threads have read red[0] before red[] is reused
  red[tid] = pv;
  __syncthreads();
  for (int s = nt / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  const float lse = red[0];                        // sum_h l_h * exp(m_h - M)
  __syncthreads();      // all threads have read red[0] before the G2 reduction reuses red[]

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

#define FD_GUARD_MQ(QP) do { \
  if (head_dim != DH || n_q_heads <= 0 || n_kv_heads <= 0 || \
      (n_q_heads % n_kv_heads) != 0 || gqa != (n_q_heads / n_kv_heads) || \
      (gqa % (QP)) != 0 || M < 1 || M > MQ_MAX) __trap(); \
} while (0)

template <int DH, int QP>
__device__ __forceinline__ void fdmq_phase1_entry(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
    float* __restrict__ scratch, int L, int M, int chunk, int n_q_heads, int gqa,
    int q_tok_stride, int q_head_stride,
    int k_head_stride, int k_pos_stride, int v_head_stride, int v_pos_stride) {
  constexpr int U = (QP == 1) ? 2 : 1;            // pipeline depth baked per variant
  const int nchunks = (L + M + chunk - 1) / chunk;
  // Block-uniform, before any __syncthreads: blocks past nchunks write no partials.
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
  FD_GUARD_MQ(QP);                                                                          \
  fdmq_phase1_entry<DH, QP>(q, k, v, scratch, *len_dev, M, chunk, n_q_heads, gqa,          \
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
  FD_GUARD_MQ(1);
  fdmq_phase2_entry<DH>(scratch, o, *len_dev, M, chunk, n_q_heads,
                        o_tok_stride, o_head_stride);
}

#undef FD_MQ_P1_EXPORTS
#undef FD_GUARD_MQ

#if defined(__HIP__) && defined(__GFX11__)
namespace prefill {
struct Problem {
  const bf16* q;
  const bf16* k;
  const bf16* v;
  bf16* out;
  float* lse;
  uint32_t rows, capacity, q_stride, q_heads, group, start;
  const int* position;
};
static_assert(sizeof(Problem) == 72);
static_assert(__builtin_offsetof(Problem, position) == 64);
using B = short __attribute__((ext_vector_type(16)));
using A = float __attribute__((ext_vector_type(8)));

// D256 uses eight-lane FP32 QK dots and BF16 WMMA PV.
// D128 reuses K storage for scores and keeps two BF16 probability planes.
template<int D, int MODE, bool EXPLICIT = false, bool PARTIAL = false, int L = D>
__device__ void run(Problem p, uint32_t extent, float scale = 0.0f,
    float* partials = nullptr, uint32_t parts = 1, uint32_t chunk = 0,
    const uint32_t* blocks = nullptr, uint32_t origin = 0, uint32_t keys = 0) {
  if (p.position) {
    const int start = *p.position;
    if (start < 0) __trap();
    p.start = min(static_cast<uint32_t>(start), p.start);
  }
  constexpr int BQ = D == 128 ? 32 : 16;
  constexpr int BK = D == 128 ? 64 : (D == 256 ? 32 : 16);
  constexpr int KS = D == 128 ? D : D+8;
  constexpr int PLANES = D == 256 && !EXPLICIT ? 1 : 2;
  constexpr int PANELS = D == 512 ? 8 : 4;
  __shared__ __align__(16) unsigned short memory[BQ*D + BK*KS + BK*D +
      PLANES*BQ*BK + 2*BQ + (D != 128 ? 2*BQ*BK : 0)];
  auto* qs = memory;
  auto* ks = qs + BQ*D;
  auto* vs = ks + BK*KS;
  auto* high = vs + BK*D;
  auto* low = high + BQ*BK;
  auto* alpha = reinterpret_cast<float*>(high + PLANES*BQ*BK);
  auto* score_memory = D != 128 ? alpha+BQ : reinterpret_cast<float*>(ks);
  const int tid = threadIdx.x, wave = tid/32, lane = tid%32;
  const uint32_t head = blockIdx.y, kv = head/p.group;
  const uint32_t query_base = PARTIAL ? 0 : blockIdx.x*BQ;
  const uint32_t partition = PARTIAL ? blockIdx.x : 0;
  if (blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 ||
      gridDim.x != (PARTIAL ? parts : (p.rows+BQ-1)/BQ) || gridDim.y != p.q_heads || gridDim.z != 1 ||
      p.group == 0 || p.rows == 0 || p.start > p.capacity || p.rows > p.capacity-p.start ||
      (PARTIAL && (p.rows > BQ || parts == 0 || chunk == 0 || chunk % BK != 0))) __trap();
  const auto bounds = [&](uint32_t row) {
    const uint32_t q = p.start+row;
    return make_uint2(min(q > extent ? q-extent : 0, blocks[2*(origin+q)] > origin ? blocks[2*(origin+q)]-origin : 0),
        max(q+1, blocks[2*(origin+q)+1] > origin ? blocks[2*(origin+q)+1]-origin : 0));
  };
  const uint32_t end = MODE >= 4 ? keys : MODE == 1 ? extent : (MODE == 2 ? max(extent,p.start+p.rows) : p.start+p.rows);
  uint32_t first = MODE == 3 && p.start+query_base > extent
      ? (p.start+query_base-extent)/BK*BK : 0;
  if constexpr (MODE == 5) first = bounds(query_base).x/BK*BK;
  uint32_t limit = MODE >= 4 ? min(end,bounds(min(query_base+BQ-1,p.rows-1)).y) : MODE == 1 ? end : min(end, MODE == 2
      ? max(extent,p.start+query_base+BQ) : p.start+query_base+BQ);
  if constexpr (PARTIAL) {
    first = max(first,partition*chunk);
    limit = min(limit,(partition+1)*chunk);
  }
  for (int cell = tid; cell < BQ*D; cell += 128) {
    const uint32_t row = query_base + cell/D;
    qs[cell] = row < p.rows && cell%D < L ? reinterpret_cast<const unsigned short*>(p.q)[
        (size_t)row*p.q_stride + head*L + cell%D] : 0;
  }
  float peak = -INFINITY, denominator = 0.0f, correction = 0.0f;
  A output[PANELS] = {};
  const int q_slab = D == 128 ? wave/2 : 0;
  const int channel = D == 128 ? (wave%2)*64 : wave*(D/4);
  __syncthreads();
  for (uint32_t base = first; base < limit; base += BK) {
    for (int cell = tid; cell < BK*D; cell += 128) {
      const uint32_t key = base + cell/D;
      const size_t offset = ((size_t)kv*p.capacity + key)*L + cell%D;
      ks[(cell/D)*KS+cell%D] = key < end && cell%D < L ? reinterpret_cast<const unsigned short*>(p.k)[offset] : 0;
      vs[cell] = key < end && cell%D < L ? reinterpret_cast<const unsigned short*>(p.v)[offset] : 0;
    }
    __syncthreads();
    const auto store_score = [&](int row, int column, float value) {
      const uint32_t key = base+column, query = p.start+query_base+row;
      const bool visible = query_base+row < p.rows && key < end &&
          (MODE >= 4 ? key < bounds(query_base+row).y && (MODE != 5 || key >= bounds(query_base+row).x) :
              (MODE == 1 || (MODE == 2 ? key < max(extent,query+1) : key <= query)) &&
              (MODE != 3 || query-key <= extent));
      score_memory[row*BK+column] = visible ? value*(EXPLICIT ? scale : (D == 256 ? 0.0625f : SCALE)) : -INFINITY;
    };
    if constexpr (D != 128) {
      for (int cell = tid/8; cell < BQ*BK; cell += 16) {
        float dot = 0.0f;
#pragma unroll
        for (int dimension = tid%8; dimension < D; dimension += 8) {
          const float q = __uint_as_float(uint32_t(qs[(cell/BK)*D+dimension]) << 16);
          const float k = __uint_as_float(uint32_t(ks[(cell%BK)*KS+dimension]) << 16);
          dot = fmaf(q,k,dot);
        }
        dot += __shfl_down(dot,4,8);
        dot += __shfl_down(dot,2,8);
        dot += __shfl_down(dot,1,8);
        if (tid%8 == 0) store_score(cell/BK,cell%BK,dot);
      }
    } else {
      A scores[2] = {};
      const int key_slab = (wave%2)*2;
#pragma unroll
      for (int part = 0; part < D/16; ++part) {
        B a;
#pragma unroll
        for (int j = 0; j < 16; ++j) a[j] = qs[(q_slab*16+lane%16)*D+part*16+j];
#pragma unroll
        for (int tile = 0; tile < 2; ++tile) {
          B b;
#pragma unroll
          for (int j = 0; j < 16; ++j) b[j] = ks[((key_slab+tile)*16+lane%16)*KS+part*16+j];
          scores[tile] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a,b,scores[tile]);
        }
      }
      __syncthreads();
#pragma unroll
      for (int tile = 0; tile < 2; ++tile)
#pragma unroll
        for (int j = 0; j < 8; ++j)
          store_score(q_slab*16+2*j+lane/16,(key_slab+tile)*16+lane%16,scores[tile][j]);
    }
    __syncthreads();
    if (tid < BQ) {
      float next = peak;
      for (int key = 0; key < BK; ++key) next = fmaxf(next,score_memory[tid*BK+key]);
      const float rescale = isfinite(next) ? __expf(peak-next) : 0.0f;
      denominator *= rescale;
      correction *= rescale;
      for (int key = 0; key < BK; ++key) {
        const float value = isfinite(next) ? __expf(score_memory[tid*BK+key]-next) : 0.0f;
        const float y = value-correction, sum = denominator+y;
        correction = (sum-denominator)-y;
        denominator = sum;
        const bf16 hi = __float2bfloat16(value);
        high[tid*BK+key] = reinterpret_cast<const unsigned short&>(hi);
        if constexpr (PLANES == 2) {
          const bf16 lo = __float2bfloat16(value-__bfloat162float(hi));
          low[tid*BK+key] = reinterpret_cast<const unsigned short&>(lo);
        }
      }
      peak = next;
      alpha[tid] = rescale;
    }
    __syncthreads();
#pragma unroll
    for (int panel = 0; panel < PANELS; ++panel) {
#pragma unroll
      for (int j = 0; j < 8; ++j) output[panel][j] *= alpha[q_slab*16+2*j+lane/16];
    }
#pragma unroll
    for (int tile = 0; tile < BK/16; ++tile) {
      B ph, pl;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        ph[j] = high[(q_slab*16+lane%16)*BK+tile*16+j];
        if constexpr (PLANES == 2) pl[j] = low[(q_slab*16+lane%16)*BK+tile*16+j];
      }
#pragma unroll
      for (int panel = 0; panel < PANELS; ++panel) {
        B value;
#pragma unroll
        for (int j = 0; j < 16; ++j) value[j] = vs[(tile*16+j)*D+channel+panel*16+lane%16];
        output[panel] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ph,value,output[panel]);
        if constexpr (PLANES == 2)
          output[panel] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pl,value,output[panel]);
      }
    }
    __syncthreads();
  }
  if (tid < BQ) {
    alpha[tid] = denominator > 0 ? 1.0f/denominator : 0.0f;
    if (query_base+tid < p.rows) {
      if constexpr (PARTIAL) {
        const size_t offset = (((size_t)(query_base+tid)*p.q_heads+head)*parts+partition)*514;
        partials[offset+512] = peak;
        partials[offset+513] = denominator;
      } else {
        p.lse[(size_t)head*p.rows+query_base+tid] = peak+logf(denominator);
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int panel = 0; panel < PANELS; ++panel) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint32_t row = query_base+q_slab*16+2*j+lane/16;
      if (row < p.rows) {
        const size_t dimension = channel+panel*16+lane%16;
        if constexpr (PARTIAL) {
          partials[(((size_t)row*p.q_heads+head)*parts+partition)*514+dimension] = output[panel][j];
        } else {
          if (dimension < L) p.out[((size_t)row*p.q_heads+head)*L+dimension] =
              __float2bfloat16(output[panel][j]*alpha[q_slab*16+2*j+lane/16]);
        }
      }
    }
  }
}
}
extern "C" __global__ void attention_prefill_d256(prefill::Problem p) { prefill::run<256,0>(p,0); }
extern "C" __global__ void attention_prefill_d72_bidirectional(prefill::Problem p, uint32_t keys, float scale) { prefill::run<128,1,true,false,72>(p,keys,scale); }
extern "C" __global__ void attention_prefill_d256_blocks(prefill::Problem p, uint32_t keys, uint32_t window, float scale, const uint32_t* blocks, uint32_t origin) {
  if (window == UINT32_MAX) prefill::run<256,4,true>(p,window,scale,nullptr,1,0,blocks,origin,keys);
  else prefill::run<256,5,true>(p,window,scale,nullptr,1,0,blocks,origin,keys);
}
extern "C" __global__ void attention_prefill_d512_blocks(prefill::Problem p, uint32_t keys, float scale, const uint32_t* blocks, uint32_t origin) { prefill::run<512,4,true>(p,UINT32_MAX,scale,nullptr,1,0,blocks,origin,keys); }
extern "C" __global__ void attention_prefill_d256_window(prefill::Problem p, uint32_t window, float scale) { prefill::run<256,3,true>(p,window,scale); }
extern "C" __global__ void attention_prefill_d512(prefill::Problem p, float scale) { prefill::run<512,0,true>(p,0,scale); }
extern "C" __global__ void attention_prefill_d512_bidirectional(prefill::Problem p, uint32_t keys, float scale) { prefill::run<512,1,true>(p,keys,scale); }
extern "C" __global__ void attention_partial_d512(prefill::Problem p, float scale, float* partials, uint32_t parts, uint32_t chunk) { prefill::run<512,0,true,true>(p,0,scale,partials,parts,chunk); }
extern "C" __global__ void attention_merge_d512(prefill::Problem p, const float* partials, uint32_t parts) {
  const uint32_t row = blockIdx.x, head = blockIdx.y, tid = threadIdx.x;
  if (blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 || gridDim.z != 1 ||
      gridDim.x != p.rows || gridDim.y != p.q_heads || parts == 0) __trap();
  const size_t base = ((size_t)row*p.q_heads+head)*parts*514;
  float peak = -INFINITY;
  for (uint32_t part = 0; part < parts; ++part) peak = fmaxf(peak,partials[base+part*514+512]);
  float sum = 0.0f, correction = 0.0f;
  float output[4] = {};
  for (uint32_t part = 0; part < parts; ++part) {
    const float* source = partials+base+part*514;
    if (source[513] > 0.0f) {
      const float weight = __expf(source[512]-peak);
      const float value = weight*source[513]-correction, next = sum+value;
      correction = (next-sum)-value;
      sum = next;
#pragma unroll
      for (int j = 0; j < 4; ++j) output[j] = fmaf(weight,source[tid*4+j],output[j]);
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) p.out[((size_t)row*p.q_heads+head)*512+tid*4+j] = __float2bfloat16(output[j]/sum);
  if (tid == 0) p.lse[(size_t)head*p.rows+row] = peak+logf(sum);
}
extern "C" __global__ void attention_prefill_d128(prefill::Problem p) { prefill::run<128,0>(p,0); }
extern "C" __global__ void attention_prefill_d128_bidirectional(prefill::Problem p, uint32_t keys) { prefill::run<128,1>(p,keys); }
extern "C" __global__ void attention_prefill_d256_bidirectional(prefill::Problem p, uint32_t keys, float scale) { prefill::run<256,1,true>(p,keys,scale); }
extern "C" __global__ void attention_prefill_d128_prefix(prefill::Problem p, uint32_t prefix) { prefill::run<128,2>(p,prefix); }
extern "C" __global__ void attention_prefill_d128_window(prefill::Problem p, uint32_t window) { prefill::run<128,3>(p,window); }
#endif
