// ============================================================================
// gemm_skinny.cu -- skinny GEMM (M activation rows share one streaming weight
// read) on bf16 tensor cores, for the speculative-decoding verification path.
//
//   Y[m, out] = X[m, in] @ W[out, in]^T
//   W: bf16 row-major [out,in], row stride ldw elements (packed: ldw == in).
//   X: bf16 row-major [m,in] packed.  Y: bf16 row-major [m,out].
//   fp32 accumulation (MMA D fragments), one bf16 rounding at the store.
//
// FD_GUARD: every entry point calls __trap() when invoked with m outside
// [1,16], in % 64 != 0, ldw % 8 != 0, or blockDim.x != 128 -- a wrong-shape
// launch fails loudly (the trap surfaces as a CUDA error at the next
// synchronization) instead of silently computing garbage.
//
// Kernel organization: block = 128 threads = 4 warps; each block covers 16
// consecutive W rows (the m16 tile of mma.m16n8k16). Warps split the IN
// dimension into quarters (warp w sweeps k in [w*in/4, (w+1)*in/4)) so a
// block streams 16 rows x in once; cross-warp fp32 merge through smem at the
// end. Per warp per 64-wide in-chunk: 4 k16-slices of mma.m16n8k16; A =
// ldmatrix.x4 on the W smem tile (row-major [16 rows][64 in]); B =
// ldmatrix.x2 (plain) on the X smem chunk (row-major [act][in-slice] -- a
// col-major k16 x n8 B fragment equals a row-major [n][k] tile). Activation
// columns beyond m are zero-filled at staging and never stored. W/X chunk
// staging uses cp.async 16B copies, double-buffered (buffer for chunk c+1 is
// committed before chunk c is consumed).
//
// Launch contract:
//   gridDim.x = ceil(out / 16), blockDim.x = 128 (fixed; guarded).
//   smem = W tile 4 warps x 16 rows x (64+8) x 2 B (double-buffered x2)
//        + X chunk 4 warps x 16 acts x (64+8) x 2 B (double-buffered x2)
//        + merge [4][16][16] fp32
//        = 18432 + 18432 + 4096 = 40960 B (<= 48 KB, no opt-in needed).
// Alignment: W and X 16B-aligned; in % 256 == 0 (NW*CK); ldw % 8 == 0. No device-side
// scalar is read and all arguments are fixed per call, so the launch is
// CUDA-Graph capturable; no global state. Every Y element is written by
// exactly one block.
//
// SIGNATURE TABLE (extern "C" __global__; all pointer args __restrict__):
//   gemm_skinny_bf16
//     void(const __nv_bfloat16* W, const __nv_bfloat16* X,
//          __nv_bfloat16* Y, int out_features, int in_features,
//          long ldw, int m)
//     launch: grid = dim3(ceil(out/16)), block = 128, smem = 40960
//
// HOST INTEGRATION (Codex-A):
//   - Selection between this kernel and the cuBLAS path is a setup-time
//     calibration per (device, shape, M bucket): measure both, persist the
//     winner. (4090 measured: skinny wins for M >= 7 in-graph; A4000 picks
//     cuBLAS.) Do not hardcode an M threshold.
//   - Weight buffers are the existing [out][in] row-major bf16 blocks; X/Y
//     are the engine's packed [rows][dim] activation buffers.
// ============================================================================
#include <cuda_runtime.h>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

constexpr int RT  = 16;                 // W rows per block (MMA m16 tile)
constexpr int NW  = 4;                  // warps per block (in-dim quarters)
constexpr int NT_ = NW * 32;            // 128 threads
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
