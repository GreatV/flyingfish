#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800) || defined(FF_FP16_GEMM_BODY)
// sm75: m16n8k16 bf16 is two m16n8k8.row.col.f32.f16.f16.f32 steps.
// cp.async.cg 16B is ld.global.cg.v4.u32 plus st.shared.v4.u32.
// A segment with fewer than 8 live elements is scalar u16 loads.
// W and X elements are FP16. Accumulators are F32.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

constexpr int RT = 16;
constexpr int NW = 4;
constexpr int NT_ = NW * 32;
constexpr int CK = 64;
constexpr int PAD = 8;
constexpr int LDS = CK + PAD;
constexpr int MA = 16;
constexpr int SPAN = NW * CK;

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
// PTX m16n8k16 A registers are two m16n8k8 fragments: (a0,a1) then (a2,a3).
// B registers are the same split along K.
__device__ __forceinline__ void mma_1688(float (&d)[4], unsigned a0, unsigned a1, unsigned b0) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(b0));
}

constexpr int WT_BYTES = NW * RT * LDS * 2;
constexpr int XT_BYTES = NW * MA * LDS * 2;
constexpr int MG_OFF = 2 * (WT_BYTES + XT_BYTES);

__device__ __forceinline__ void stage8(void* dst, const void* src, int alive) {
  unsigned x = 0u, y = 0u, z = 0u, w = 0u;
  if (alive == 8 && (reinterpret_cast<uintptr_t>(src) & 15u) == 0u) {
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                 : "l"(src));
  } else if (alive > 0) {
    unsigned short e0 = 0, e1 = 0, e2 = 0, e3 = 0, e4 = 0, e5 = 0, e6 = 0, e7 = 0;
    const unsigned short* p = reinterpret_cast<const unsigned short*>(src);
    if (alive > 0) e0 = p[0];
    if (alive > 1) e1 = p[1];
    if (alive > 2) e2 = p[2];
    if (alive > 3) e3 = p[3];
    if (alive > 4) e4 = p[4];
    if (alive > 5) e5 = p[5];
    if (alive > 6) e6 = p[6];
    if (alive > 7) e7 = p[7];
    x = (unsigned)e0 | ((unsigned)e1 << 16);
    y = (unsigned)e2 | ((unsigned)e3 << 16);
    z = (unsigned)e4 | ((unsigned)e5 << 16);
    w = (unsigned)e6 | ((unsigned)e7 << 16);
  }
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(smem_u32(dst)), "r"(x), "r"(y), "r"(z), "r"(w));
}

template <typename Output>
__device__ __forceinline__ void skinny_project(const __half* __restrict__ W, const __half* __restrict__ X,
                                                Output* __restrict__ Y, int out_features, int in_features, long ldw,
                                                int m, int tiny) {
  if (blockDim.x != NT_ || m < 1 || out_features < 1 || in_features < 1 || ldw < in_features) __trap();

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (tiny) {
    if ((ldw & 1) != 0 || ((uintptr_t)W & 3u) != 0u || ((uintptr_t)X & 3u) != 0u) __trap();
    if (gridDim.x != out_features || gridDim.y != m || gridDim.z != 1) __trap();
    const int output = blockIdx.x;
    const int row = blockIdx.y;
    const __half* weights = W + (size_t)output * ldw;
    const __half* input = X + (size_t)row * in_features;
    float sum = 0.0f;
    for (int k = 2 * tid; k + 1 < in_features; k += 2 * NT_) {
      const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(weights + k));
      const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(input + k));
      sum = fmaf(wv.x, xv.x, sum);
      sum = fmaf(wv.y, xv.y, sum);
    }
    if ((in_features & 1) != 0 && tid == 0) {
      const int k = in_features - 1;
      sum = fmaf(__half2float(weights[k]), __half2float(input[k]), sum);
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, delta);
    extern __shared__ float partials[];
    if (lane == 0) partials[warp] = sum;
    __syncthreads();
    if (tid == 0) {
      float total = 0.0f;
#pragma unroll
      for (int w = 0; w < NW; ++w) total += partials[w];
      if constexpr (std::is_same_v<Output, __half>)
        Y[(size_t)row * out_features + output] = __float2half_rn(total);
      else
        Y[(size_t)row * out_features + output] = total;
    }
    return;
  }
  if (gridDim.x != (out_features + RT - 1) / RT || gridDim.y != 1 || gridDim.z != 1) __trap();

  const int row0 = blockIdx.x * RT;
  const int nck = (in_features + SPAN - 1) / SPAN;
  const int kbeg = warp * nck * CK;

  extern __shared__ __align__(16) __half smm[];
  __half* wt = smm;
  __half* xt = smm + 2 * (WT_BYTES / 2);
  float* mg = reinterpret_cast<float*>(smm + MG_OFF / 2);
  __half* const wt0 = wt + warp * RT * LDS;
  __half* const wt1 = wt0 + WT_BYTES / 2;
  __half* const xt0 = xt + warp * MA * LDS;
  __half* const xt1 = xt0 + XT_BYTES / 2;

  auto stage_w = [&](int c, int buf) {
    __half* const wbuf = buf ? wt1 : wt0;
    const int k0 = kbeg + c * CK;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int idx = lane + i * 32;
      const int r = idx >> 3, seg = idx & 7;
      const int k = k0 + seg * 8;
      const bool row_live = row0 + r < out_features;
      int alive = 0;
      const __half* src = W;
      if (row_live && k < in_features) {
        alive = in_features - k;
        if (alive > 8) alive = 8;
        src = W + (size_t)(row0 + r) * ldw + k;
      }
      stage8(wbuf + r * LDS + seg * 8, src, alive);
    }
    __syncwarp();
  };
  auto stage_x = [&](int c, int buf, int m0, int mt) {
    __half* const xbuf = buf ? xt1 : xt0;
    const int k0 = kbeg + c * CK;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int idx = lane + i * 32;
      const int r = idx >> 3, seg = idx & 7;
      const int k = k0 + seg * 8;
      int alive = 0;
      const __half* src = X;
      if (r < mt && k < in_features) {
        alive = in_features - k;
        if (alive > 8) alive = 8;
        src = X + (size_t)(m0 + r) * in_features + k;
      }
      stage8(xbuf + r * LDS + seg * 8, src, alive);
    }
    __syncwarp();
  };

  for (int m0 = 0; m0 < m; m0 += MA) {
    const int mt = m - m0 < MA ? m - m0 : MA;
    const int nacts = (mt + 7) / 8;
    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    stage_w(0, 0);
    stage_x(0, 0, m0, mt);
    for (int c = 0; c < nck; ++c) {
      const int buf = c & 1;
      if (c + 1 < nck) {
        stage_w(c + 1, buf ^ 1);
        stage_x(c + 1, buf ^ 1, m0, mt);
      }
      __half* const wbuf = buf ? wt1 : wt0;
      __half* const xbuf = buf ? xt1 : xt0;
#pragma unroll
      for (int kk = 0; kk < CK / 16; ++kk) {
        unsigned wa[4];
        ldsm_x4(wa, wbuf + (lane & 15) * LDS + kk * 16 + (lane >> 4) * 8);
        unsigned xb[2];
        ldsm_x2(xb, xbuf + (lane & 7) * LDS + kk * 16 + (lane >> 3) * 8);
        mma_1688(acc0, wa[0], wa[1], xb[0]);
        mma_1688(acc0, wa[2], wa[3], xb[1]);
        if (nacts > 1) {
          ldsm_x2(xb, xbuf + (8 + (lane & 7)) * LDS + kk * 16 + (lane >> 3) * 8);
          mma_1688(acc1, wa[0], wa[1], xb[0]);
          mma_1688(acc1, wa[2], wa[3], xb[1]);
        }
      }
    }
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
    if (warp == 0) {
      for (int idx = lane; idx < RT * mt; idx += 32) {
        const int r = idx / mt, a = idx % mt;
        const int row = row0 + r;
        if (row >= out_features) continue;
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; ++w) s += mg[(w * RT + r) * MA + a];
        if constexpr (std::is_same_v<Output, __half>)
          Y[(size_t)(m0 + a) * out_features + row] = __float2half_rn(s);
        else
          Y[(size_t)(m0 + a) * out_features + row] = s;
      }
    }
    __syncthreads();
  }
}

extern "C" __global__ __launch_bounds__(NT_) void gemm_skinny_bf16(const __half* __restrict__ W,
                                                                   const __half* __restrict__ X,
                                                                   __half* __restrict__ Y, int out_features,
                                                                   int in_features, long ldw, int m, int tiny) {
  skinny_project(W, X, Y, out_features, in_features, ldw, m, tiny);
}

extern "C" __global__ __launch_bounds__(NT_) void gemm_skinny_bf16_f32(const __half* __restrict__ W,
                                                                       const __half* __restrict__ X,
                                                                       float* __restrict__ Y, int out_features,
                                                                       int in_features, long ldw, int m) {
  skinny_project(W, X, Y, out_features, in_features, ldw, m, 0);
}
#else /* ff-sm80 */
#if defined(__HIP__)
#include "gpu_compat.cuh"
#include <stdint.h>

using bf16 = __nv_bfloat16;
using Bf16Fragment = short __attribute__((ext_vector_type(16)));
using AccFragment = float __attribute__((ext_vector_type(8)));

// Four wave32 groups reduce disjoint K intervals; W[out,ldw], X[m,in], Y[m,out].
__device__ __forceinline__ void gemm_skinny_hip(
    const bf16* __restrict__ W, const bf16* __restrict__ X,
    bf16* __restrict__ Y, const bf16* __restrict__ bias,
    int out_features, int in_features, unsigned long ldw, int m) {
    if (blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.x != (out_features + 15) / 16 || gridDim.y != 1 || gridDim.z != 1 ||
        m < 1 || m > 16 || in_features < 1 || out_features < 1 || ldw < in_features)
        __trap();
    __shared__ unsigned short staging[4096];
    const int tid = threadIdx.x;
    const int wave = tid / 32;
    const int lane = tid % 32;
    const int base = blockIdx.x * 16;
    const int panels = (in_features + 127) / 128;
    unsigned short* weights = staging + wave * 1024;
    unsigned short* inputs = weights + 512;
    AccFragment acc = {};
    for (int panel = 0; panel < panels; ++panel) {
        const int begin = (wave * panels + panel) * 32;
        for (int cell = lane; cell < 512; cell += 32) {
            const int row = cell / 32;
            const int k = begin + cell % 32;
            weights[cell] = base + row < out_features && k < in_features
                ? reinterpret_cast<const unsigned short*>(W)[(size_t)(base + row) * ldw + k] : 0;
            inputs[cell] = row < m && k < in_features
                ? reinterpret_cast<const unsigned short*>(X)[(size_t)row * in_features + k] : 0;
        }
        __syncthreads();
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            Bf16Fragment a, b;
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                a[k] = weights[(lane % 16) * 32 + half * 16 + k];
                b[k] = inputs[(lane % 16) * 32 + half * 16 + k];
            }
            acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
        }
        __syncthreads();
    }
    float* partials = reinterpret_cast<float*>(staging);
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        const int row = 2 * g + lane / 16;
        partials[wave * 256 + row * 16 + lane % 16] = acc[g];
    }
    __syncthreads();
    for (int cell = tid; cell < 256; cell += 128) {
        const int output = base + cell / 16;
        const int row = cell % 16;
        if (output < out_features && row < m) {
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < 4; ++w) sum += partials[w * 256 + cell];
            if (bias) sum += __bfloat162float(bias[output]);
            Y[(size_t)row * out_features + output] = __float2bfloat16(sum);
        }
    }
}
extern "C" __global__ __launch_bounds__(128) void gemm_skinny_bf16(
    const bf16* W, const bf16* X, bf16* Y, const bf16* bias,
    int out_features, int in_features, unsigned long ldw, int m) {
    gemm_skinny_hip(W, X, Y, bias, out_features, in_features, ldw, m);
}
extern "C" __global__ __launch_bounds__(128) void gemm_skinny_bf16_f32(
    const bf16* W, const bf16* X, float* Y,
    int out_features, int in_features, unsigned long ldw, int m) {
    if (blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.x != (out_features + 15) / 16 || gridDim.y != 1 || gridDim.z != 1 ||
        m < 1 || m > 16 || in_features < 1 || out_features < 1 || ldw < in_features)
        __trap();
    const int output = blockIdx.x * 16 + threadIdx.x / 8;
    const int lane = threadIdx.x % 8;
    if (output >= out_features) return;
    for (int row = 0; row < m; ++row) {
        float sum = 0.0f;
        for (int k = lane; k < in_features; k += 8)
            sum = fmaf(__bfloat162float(W[(size_t)output * ldw + k]),
                       __bfloat162float(X[(size_t)row * in_features + k]), sum);
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(FF_WARP_MASK, sum, offset, 8);
        if (lane == 0) Y[(size_t)row * out_features + output] = sum;
    }
}

// Row-count-invariant RDNA3 wave32 BF16 WMMA GEMM: Y[rows,out] = alpha * (X @ W^T) + beta * Y.
// K-loop starts at 0 without workgroup staggering or split-K; beta in {0, 1}.
extern "C" __global__ __launch_bounds__(128) void gemm_bf16_row_invariant(
    const bf16* __restrict__ W, const bf16* __restrict__ X, bf16* __restrict__ Y,
    int rows, int out_features, int in_features, float alpha, float beta) {
    if (blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.x != (out_features + 63) / 64 || gridDim.y != (rows + 63) / 64 ||
        gridDim.z != 1 || rows < 1 || out_features < 1 || in_features < 1 ||
        !W || !X || !Y || !(beta == 0.0f || beta == 1.0f)) {
        __trap();
        return;
    }
    constexpr int BK = 32;
    constexpr int STRIDE = BK + 8;
    __shared__ __align__(16) unsigned short ws[64 * STRIDE];
    __shared__ __align__(16) unsigned short xs[64 * STRIDE];
    const int tid = threadIdx.x;
    const int wave = tid / 32;
    const int lane = tid % 32;
    const int wm = wave / 2;
    const int wn = wave % 2;
    const int out_base = blockIdx.x * 64;
    const int row_base = blockIdx.y * 64;
    const auto* w_u16 = reinterpret_cast<const unsigned short*>(W);
    const auto* x_u16 = reinterpret_cast<const unsigned short*>(X);
    const bool vec_ok = (in_features & 7) == 0 &&
        (reinterpret_cast<uintptr_t>(W) & 15) == 0 &&
        (reinterpret_cast<uintptr_t>(X) & 15) == 0;
    AccFragment acc[2][2] = {};
    for (int k0 = 0; k0 < in_features; k0 += BK) {
        if (vec_ok && k0 + BK <= in_features) {
            for (int v = tid; v < 64 * (BK / 8); v += 128) {
                const int r = v / (BK / 8);
                const int c8 = v % (BK / 8);
                const int dst = r * STRIDE + c8 * 8;
                const int o = out_base + r;
                const int xr = row_base + r;
                *reinterpret_cast<uint4*>(ws + dst) = o < out_features
                    ? *reinterpret_cast<const uint4*>(w_u16 + (size_t)o * in_features + k0 + c8 * 8)
                    : make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(xs + dst) = xr < rows
                    ? *reinterpret_cast<const uint4*>(x_u16 + (size_t)xr * in_features + k0 + c8 * 8)
                    : make_uint4(0, 0, 0, 0);
            }
        } else {
            for (int cell = tid; cell < 64 * BK; cell += 128) {
                const int r = cell / BK;
                const int c = cell % BK;
                const int k = k0 + c;
                const int o = out_base + r;
                const int xr = row_base + r;
                ws[r * STRIDE + c] = (o < out_features && k < in_features)
                    ? w_u16[(size_t)o * in_features + k] : 0;
                xs[r * STRIDE + c] = (xr < rows && k < in_features)
                    ? x_u16[(size_t)xr * in_features + k] : 0;
            }
        }
        __syncthreads();
#pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
            const int koff = kh * 16;
            Bf16Fragment a[2], b[2];
#pragma unroll
            for (int im = 0; im < 2; ++im) {
                const unsigned short* p = ws + (wm * 32 + im * 16 + lane % 16) * STRIDE + koff;
                uint4 lo = *reinterpret_cast<const uint4*>(p);
                uint4 hi = *reinterpret_cast<const uint4*>(p + 8);
                __builtin_memcpy(&a[im], &lo, 16);
                __builtin_memcpy(reinterpret_cast<char*>(&a[im]) + 16, &hi, 16);
            }
#pragma unroll
            for (int in_t = 0; in_t < 2; ++in_t) {
                const unsigned short* p = xs + (wn * 32 + in_t * 16 + lane % 16) * STRIDE + koff;
                uint4 lo = *reinterpret_cast<const uint4*>(p);
                uint4 hi = *reinterpret_cast<const uint4*>(p + 8);
                __builtin_memcpy(&b[in_t], &lo, 16);
                __builtin_memcpy(reinterpret_cast<char*>(&b[in_t]) + 16, &hi, 16);
            }
#pragma unroll
            for (int im = 0; im < 2; ++im) {
#pragma unroll
                for (int in_t = 0; in_t < 2; ++in_t) {
                    acc[im][in_t] =
                        __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a[im], b[in_t], acc[im][in_t]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int im = 0; im < 2; ++im) {
#pragma unroll
        for (int in_t = 0; in_t < 2; ++in_t) {
            const int r = row_base + wn * 32 + in_t * 16 + lane % 16;
            if (r < rows) {
#pragma unroll
                for (int g = 0; g < 8; ++g) {
                    const int o = out_base + wm * 32 + im * 16 + 2 * g + lane / 16;
                    if (o < out_features) {
                        const size_t idx = (size_t)r * out_features + o;
                        float val = alpha * acc[im][in_t][g];
                        if (beta != 0.0f) val = fmaf(beta, __bfloat162float(Y[idx]), val);
                        Y[idx] = __float2bfloat16(val);
                    }
                }
            }
        }
    }
}

#else
// Skinny GEMM on bf16 tensor cores: M activation rows share one streaming weight read.
//   Y[m, out] = X[m, in] @ W[out, in]^T
//   W: bf16 row-major [out,in], row stride ldw elements (packed: ldw == in).
//   X: bf16 row-major [m,in] packed.  Y: bf16 row-major [m,out].
//   fp32 accumulation (MMA D fragments), one bf16 rounding at the store.
// Regular MMA requires in % 256 == 0 and ldw % 8 == 0.
// Block = 4 warps over 16 consecutive W rows (the m16 tile of mma.m16n8k16); warp w sweeps
// k in [w*in/4, (w+1)*in/4); the warps merge fp32 partials through smem at the end.
// Per warp per 64-wide in-chunk: 4 k16 slices; A = ldmatrix.x4 on the W tile (row-major
// [16 rows][64 in]); B = ldmatrix.x2 on the X chunk (row-major [act][in-slice]).
// Activation rows >= m are zero-filled at staging and never stored. W/X staging is
// cp.async 16B copies, double-buffered.
// Launch: grid = ceil(out / 16), block = 128,
//   dynamic smem = 2 x 9216 (W) + 2 x 9216 (X) + 4096 (merge) = 40960 B.
// W and X are 16B-aligned. Every Y element is written by exactly one block.
// tiny: grid [out,m,1], 128 threads, four FP32 shared partials per CTA.
// tiny uses fixed lane-strided BF16 pairs and fixed FP32 reductions for every m.
// tiny requires positive even in, even ldw >= in, and 4B-aligned W/X bases.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <type_traits>

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

template<typename Output>
__device__ __forceinline__ void skinny_project(
    const bf16* __restrict__ W, const bf16* __restrict__ X,
    Output* __restrict__ Y, int out_features, int in_features,
    long ldw, int m, int tiny) {
  if (blockDim.x != NT_ || m < 1 || m > MA || out_features < 1 ||
      in_features < 1 || ldw < in_features) __trap();

  const int tid  = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (tiny) {
    if ((in_features & 1) != 0 || (ldw & 1) != 0 ||
        ((size_t)W & 3) != 0 || ((size_t)X & 3) != 0) __trap();
    if (gridDim.x != out_features || gridDim.y != m || gridDim.z != 1) __trap();
    const int output = blockIdx.x;
    const int row = blockIdx.y;
    const bf16* weights = W + (size_t)output * ldw;
    const bf16* input = X + (size_t)row * in_features;
    float sum = 0.0f;
    for (int k = 2 * tid; k < in_features; k += 2 * NT_) {
      const float2 w = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(weights + k));
      const float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(input + k));
      sum = fmaf(w.x, x.x, sum);
      sum = fmaf(w.y, x.y, sum);
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      sum += __shfl_down_sync(0xffffffffu, sum, delta);
    }
    extern __shared__ float partials[];
    if (lane == 0) partials[warp] = sum;
    __syncthreads();
    if (tid == 0) {
      float total = 0.0f;
#pragma unroll
      for (int w = 0; w < NW; ++w) total += partials[w];
      if constexpr (std::is_same_v<Output, bf16>)
        Y[(size_t)row * out_features + output] = __float2bfloat16(total);
      else Y[(size_t)row * out_features + output] = total;
    }
    return;
  }
  if (in_features % (NW * CK) != 0 || (ldw & 7) != 0) __trap();
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
    if constexpr (std::is_same_v<Output, bf16>)
      Y[(size_t)a * out_features + row] = __float2bfloat16(s);
    else Y[(size_t)a * out_features + row] = s;
  }
}

extern "C" __global__ __launch_bounds__(NT_) void gemm_skinny_bf16(
    const bf16* __restrict__ W, const bf16* __restrict__ X,
    bf16* __restrict__ Y, int out_features, int in_features,
    long ldw, int m, int tiny) {
    skinny_project(W, X, Y, out_features, in_features, ldw, m, tiny);
}

extern "C" __global__ __launch_bounds__(NT_) void gemm_skinny_bf16_f32(
    const bf16* __restrict__ W, const bf16* __restrict__ X,
    float* __restrict__ Y, int out_features, int in_features,
    long ldw, int m) {
    skinny_project(W, X, Y, out_features, in_features, ldw, m, 0);
}

#undef FD_GUARD_SKINNY

#endif
#endif /* ff-sm80 */
