// Skinny GEMM on bf16 tensor cores: M activation rows share one streaming weight read.
//   Y[m, out] = X[m, in] @ W[out, in]^T
//   W: bf16 row-major [out,in], row stride ldw elements (packed: ldw == in).
//   X: bf16 row-major [m,in] packed.  Y: bf16 row-major [m,out].
//   fp32 accumulation (MMA D fragments), one bf16 rounding at the store.
// FD_GUARD_SKINNY calls __trap() on m outside [1,16], in % 256 != 0, ldw % 8 != 0, or
// blockDim.x != 128.
// Block = 4 warps over 16 consecutive W rows (the m16 tile of mma.m16n8k16); warp w sweeps
// k in [w*in/4, (w+1)*in/4); the warps merge fp32 partials through smem at the end.
// Per warp per 64-wide in-chunk: 4 k16 slices; A = ldmatrix.x4 on the W tile (row-major
// [16 rows][64 in]); B = ldmatrix.x2 on the X chunk (row-major [act][in-slice]).
// Activation rows >= m are zero-filled at staging and never stored. W/X staging is
// cp.async 16B copies, double-buffered.
// Launch: grid = ceil(out / 16), block = 128,
//   dynamic smem = 2 x 9216 (W) + 2 x 9216 (X) + 4096 (merge) = 40960 B.
// W and X are 16B-aligned. Every Y element is written by exactly one block.
#include <cuda_runtime.h>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

constexpr int RT  = 16;                 // W rows per block (MMA m16 tile)
constexpr int NW  = 4;                  // warps per block (in-dim quarters)
constexpr int NT_ = NW * 32;
constexpr int CK  = 64;                 // in-chunk width (k elements)
constexpr int PAD = 8;                  // smem row padding vs ldmatrix banks
constexpr int LDS = CK + PAD;           // padded smem row stride = 72
constexpr int MA  = 16;                 // max acts (one or two n8 tiles)

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

// smem plan (byte counts): wt double buffer 2 x NW*RT*LDS*2; xt double buffer
// 2 x NW*MA*LDS*2; merge area NW*RT*MA fp32.
constexpr int WT_BYTES = NW * RT * LDS * 2;        // 9216 per buffer
constexpr int XT_BYTES = NW * MA * LDS * 2;        // 9216 per buffer
constexpr int MG_OFF = 2 * (WT_BYTES + XT_BYTES);  // 36864; + merge 4096 = 40960 total

#define FD_GUARD_SKINNY do {                                                            \
  if (blockDim.x != NT_ || in_features % (NW * CK) != 0 || (ldw & 7) != 0 ||            \
      m < 1 || m > MA) __trap();                                                        \
} while (0)

extern "C" __global__ void __launch_bounds__(NT_) gemm_skinny_bf16(
    const bf16* __restrict__ W, const bf16* __restrict__ X,
    bf16* __restrict__ Y, int out_features, int in_features,
    long ldw, int m) {
  FD_GUARD_SKINNY;

  const int tid  = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int row0 = blockIdx.x * RT;                 // this block's first W row
  const int nck  = in_features / (NW * CK);         // 64-wide in-chunks per warp
  const int kbeg = warp * (in_features / NW);       // warp's k-quarter start
  const int nacts = (m + 7) / 8;                    // n8 act tiles: 1 or 2

  extern __shared__ __align__(16) bf16 smm[];
  bf16* wt = smm;                                   // [2][NW][RT][LDS]
  bf16* xt = smm + 2 * (WT_BYTES / 2);              // [2][NW][MA][LDS]
  float* mg = reinterpret_cast<float*>(smm + MG_OFF / 2);   // [NW][RT][MA]

  // Per-warp tile bases.
  bf16* const wt0 = wt + warp * RT * LDS;
  bf16* const wt1 = wt0 + WT_BYTES / 2;
  bf16* const xt0 = xt + warp * MA * LDS;
  bf16* const xt1 = xt0 + XT_BYTES / 2;

  // Per-warp cp.async staging of one (wt, xt) pair for chunk c: 16B per
  // cp.async, src-size 0 zero-fills dead rows (OOB W rows / acts >= m).
  // W tile: 16 rows x 64 k = 2048 B = 128 x 16B -> 4 per lane.
  // X chunk: MA rows x 64 k -> MA*8 x 16B -> 4 per lane.
  auto stage_chunk = [&](int c, int buf) {
    bf16* const wbuf = buf ? wt1 : wt0;
    bf16* const xbuf = buf ? xt1 : xt0;
    const int k0 = kbeg + c * CK;
    const uint4* w0 = reinterpret_cast<const uint4*>(W) + (size_t)row0 * (ldw >> 3) + (k0 >> 3);
    const long wstride = ldw >> 3;                  // uint4 elements per W row
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int idx = lane + i * 32;                // 128 x 16B
      const int r = idx >> 3, seg = idx & 7;        // 16 rows x 8 segs (64 k)
      const bool live = row0 + r < out_features;
      const uint4* src = live ? w0 + (size_t)r * wstride + seg : w0;
      const unsigned dst = smem_u32(wbuf + r * LDS + seg * 8);
      const int sz = live ? 16 : 0;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                   :: "r"(dst), "l"(src), "r"(sz));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int idx = lane + i * 32;
      const int r = idx >> 3, seg = idx & 7;
      const bool live = r < m;
      const bf16* src = live ? X + (size_t)r * in_features + k0 + seg * 8 : X;
      const unsigned dst = smem_u32(xbuf + r * LDS + seg * 8);
      const int sz = live ? 16 : 0;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                   :: "r"(dst), "l"(src), "r"(sz));
    }
    asm volatile("cp.async.commit_group;\n");
  };

  float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  stage_chunk(0, 0);
  for (int c = 0; c < nck; ++c) {
    const int buf = c & 1;
    if (c + 1 < nck) stage_chunk(c + 1, buf ^ 1);   // prefetch next chunk
    // buffer `buf` is ready when at most (has-next ? 1 : 0) groups are pending
    if (c + 1 < nck) asm volatile("cp.async.wait_group 1;\n");
    else             asm volatile("cp.async.wait_group 0;\n");
    __syncwarp();
    bf16* const wbuf = buf ? wt1 : wt0;
    bf16* const xbuf = buf ? xt1 : xt0;
#pragma unroll
    for (int kk = 0; kk < CK / 16; ++kk) {          // 4 k16 slices
      unsigned wa[4];
      // A 16x16 at rows [0,RT) x k [kk*16,+16) of the warp's W tile:
      // lane -> row lane%16, col kk*16 + (lane/16)*8.
      ldsm_x4(wa, wbuf + (lane & 15) * LDS + kk * 16 + (lane >> 4) * 8);
      unsigned xb[2];
      ldsm_x2(xb, xbuf + (lane & 7) * LDS + kk * 16 + (lane >> 3) * 8);
      mma_16816(acc0, wa, xb);
      if (nacts > 1) {
        ldsm_x2(xb, xbuf + (8 + (lane & 7)) * LDS + kk * 16 + (lane >> 3) * 8);
        mma_16816(acc1, wa, xb);
      }
    }
  }

  // ---- cross-warp merge: each warp writes its fp32 partials, warp 0 sums --
  // D frag: lane l holds (row l/4, col (l%4)*2+{0,1}) and (row l/4+8, ...).
  __syncthreads();
  const int c0 = (lane & 3) * 2;
  mg[((warp * RT) + (lane >> 2)) * MA + c0] = acc0[0];
  mg[((warp * RT) + (lane >> 2)) * MA + c0 + 1] = acc0[1];
  mg[((warp * RT) + (lane >> 2) + 8) * MA + c0] = acc0[2];
  mg[((warp * RT) + (lane >> 2) + 8) * MA + c0 + 1] = acc0[3];
  if (nacts > 1) {
    mg[((warp * RT) + (lane >> 2)) * MA + c0 + 8] = acc1[0];
    mg[((warp * RT) + (lane >> 2)) * MA + c0 + 9] = acc1[1];
    mg[((warp * RT) + (lane >> 2) + 8) * MA + c0 + 8] = acc1[2];
    mg[((warp * RT) + (lane >> 2) + 8) * MA + c0 + 9] = acc1[3];
  }
  __syncthreads();
  if (warp != 0) return;
  // warp 0: sum the 4 quarters and store bf16. lane covers (row, act) pairs.
  for (int idx = lane; idx < RT * m; idx += 32) {
    const int r = idx / m, a = idx % m;
    const int row = row0 + r;
    if (row >= out_features) continue;
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < NW; ++w) s += mg[(w * RT + r) * MA + a];
    Y[(size_t)a * out_features + row] = __float2bfloat16(s);
  }
}

#undef FD_GUARD_SKINNY

extern "C" __global__ __launch_bounds__(128) void gemv_bf16_after_bias(
    const bf16* __restrict__ W, const bf16* __restrict__ X,
    const bf16* __restrict__ bias, bf16* __restrict__ Y,
    int out_features, int in_features, int rows) {
  if (blockDim.x != 128 || rows != 1 || in_features % 32 != 0) {
    __trap(); return;
  }
  const int lane = threadIdx.x & 31;
  const int column = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (column >= out_features) return;
  const bf16* w = W + (size_t)column * in_features;
  float a = 0.0f, b = 0.0f;
  for (int k = lane * 2; k < in_features; k += 64) {
    const float2 ww = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(w + k));
    const float2 xx = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(X + k));
    a = fmaf(ww.x, xx.x, a);
    b = fmaf(ww.y, xx.y, b);
  }
  a += b;
  #pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1)
    a += __shfl_down_sync(0xffffffff, a, delta);
  if (lane == 0) Y[column] = __hadd(__float2bfloat16_rn(a), bias[column]);
}

// Tile [64 activation rows,128 output columns,32 reduction elements], two stages.
extern "C" __global__ __launch_bounds__(256) void gemm_bf16_after_bias(
    const bf16* __restrict__ W, const bf16* __restrict__ X,
    const bf16* __restrict__ bias, bf16* __restrict__ Y,
    int out_features, int in_features, int rows) {
  constexpr int K = 32, LD = 40, WM = 128, XM = 64;
  constexpr int STAGE = (WM + XM) * LD;
  if (blockDim.x != 256 || in_features % K != 0 || out_features % 8 != 0 || rows <= 1) {
    __trap(); return;
  }
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int column0 = blockIdx.x * WM, row0 = blockIdx.y * XM;
  extern __shared__ __align__(16) bf16 tiles[];
  auto stage = [&](int chunk, int buffer) {
    bf16* wt = tiles + buffer * STAGE;
    bf16* xt = wt + WM * LD;
    #pragma unroll
    for (int i = tid; i < WM * (K / 8); i += 256) {
      const int r = i / (K / 8), c = (i % (K / 8)) * 8;
      const bool live = column0 + r < out_features;
      const bf16* source = live ? W + (size_t)(column0 + r) * in_features + chunk * K + c : W;
      const unsigned target = smem_u32(wt + r * LD + c);
      const int bytes = live ? 16 : 0;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                   :: "r"(target), "l"(source), "r"(bytes));
    }
    #pragma unroll
    for (int i = tid; i < XM * (K / 8); i += 256) {
      const int r = i / (K / 8), c = (i % (K / 8)) * 8;
      const bool live = row0 + r < rows;
      const bf16* source = live ? X + (size_t)(row0 + r) * in_features + chunk * K + c : X;
      const unsigned target = smem_u32(xt + r * LD + c);
      const int bytes = live ? 16 : 0;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                   :: "r"(target), "l"(source), "r"(bytes));
    }
    asm volatile("cp.async.commit_group;\n");
  };
  float acc[8][4] = {{0.0f}};
  const int chunks = in_features / K;
  stage(0, 0);
  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int buffer = chunk & 1;
    if (chunk + 1 < chunks) stage(chunk + 1, buffer ^ 1);
    if (chunk + 1 < chunks) asm volatile("cp.async.wait_group 1;\n");
    else asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();
    const bf16* wt = tiles + buffer * STAGE;
    const bf16* xt = wt + WM * LD;
    #pragma unroll
    for (int kk = 0; kk < K / 16; ++kk) {
      unsigned wa[4];
      ldsm_x4(wa, wt + (warp * 16 + (lane & 15)) * LD + kk * 16 + (lane >> 4) * 8);
      #pragma unroll
      for (int n = 0; n < 8; ++n) {
        unsigned xb[2];
        ldsm_x2(xb, xt + (n * 8 + (lane & 7)) * LD + kk * 16 + (lane >> 3) * 8);
        mma_16816(acc[n], wa, xb);
      }
    }
    __syncthreads();
  }
  const int column = column0 + warp * 16 + (lane >> 2);
  const int r = row0 + (lane & 3) * 2;
  #pragma unroll
  for (int n = 0; n < 8; ++n) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c = column + (i / 2) * 8;
      const int row = r + n * 8 + i % 2;
      if (c < out_features && row < rows)
        Y[(size_t)row * out_features + c] = __hadd(__float2bfloat16_rn(acc[n][i]), bias[c]);
    }
  }
}
