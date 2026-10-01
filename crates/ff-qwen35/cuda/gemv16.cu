// 16-bit grouped GEMV for ff-qwen35 raw-weight checkpoints: up to 4
// segments over one x in a single launch, warp-per-row, uint4 weight
// reads (8 elements per load), f32 activation and accumulation.
// Deterministic per row: sequential lane passes, fixed shuffle fold,
// no split-K and no atomics. Rows per block = warps per block = 8.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define RPB16 8

template <typename W> struct Widen;
template <> struct Widen<__nv_bfloat16> {
    static __device__ __forceinline__ float at(const uint4 &v, int i) {
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(&v)[i]);
    }
};
template <> struct Widen<__half> {
    static __device__ __forceinline__ float at(const uint4 &v, int i) {
        return __half2float(reinterpret_cast<const __half *>(&v)[i]);
    }
};

// One warp per row; the block's warp index picks the row inside the
// RPB16 row tile. in_dim is a multiple of 8 (host asserts).
template <typename W>
__device__ __forceinline__ void gemv16_row(
    const unsigned char *__restrict__ w, float *__restrict__ y,
    int rows, int row_base, const float *__restrict__ x, int in_dim)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = row_base + warp;
    if (row >= rows) return;
    const uint4 *prow =
        reinterpret_cast<const uint4 *>(w + (long long)row * in_dim * 2);
    const int tiles = in_dim >> 3;
    float acc = 0.0f;
    #pragma unroll 4
    for (int t = lane; t < tiles; t += 32) {
        const uint4 pk = prow[t];
        const float4 xa = *reinterpret_cast<const float4 *>(x + t * 8);
        const float4 xb = *reinterpret_cast<const float4 *>(x + t * 8 + 4);
        acc += Widen<W>::at(pk, 0) * xa.x;
        acc += Widen<W>::at(pk, 1) * xa.y;
        acc += Widen<W>::at(pk, 2) * xa.z;
        acc += Widen<W>::at(pk, 3) * xa.w;
        acc += Widen<W>::at(pk, 4) * xb.x;
        acc += Widen<W>::at(pk, 5) * xb.y;
        acc += Widen<W>::at(pk, 6) * xb.z;
        acc += Widen<W>::at(pk, 7) * xb.w;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}

template <typename W>
__device__ __forceinline__ void gemv16_group(
    const unsigned char *__restrict__ w0, float *__restrict__ y0, int rows0,
    const unsigned char *__restrict__ w1, float *__restrict__ y1, int rows1,
    const unsigned char *__restrict__ w2, float *__restrict__ y2, int rows2,
    const unsigned char *__restrict__ w3, float *__restrict__ y3, int rows3,
    const float *__restrict__ x, int in_dim)
{
    const int pad0 = (rows0 + RPB16 - 1) & ~(RPB16 - 1);
    const int pad1 = (rows1 + RPB16 - 1) & ~(RPB16 - 1);
    const int pad2 = (rows2 + RPB16 - 1) & ~(RPB16 - 1);

    const unsigned char *w;
    float *y;
    int rows;
    int row_base;
    {
        int g = blockIdx.x * RPB16;
        int seg = 0;
        if (g >= pad0) { g -= pad0; seg = 1; }
        if (seg == 1 && g >= pad1) { g -= pad1; seg = 2; }
        if (seg == 2 && g >= pad2) { g -= pad2; seg = 3; }
        row_base = g;
        switch (seg) {
        case 0: w = w0; y = y0; rows = rows0; break;
        case 1: w = w1; y = y1; rows = rows1; break;
        case 2: w = w2; y = y2; rows = rows2; break;
        default: w = w3; y = y3; rows = rows3; break;
        }
        if (row_base >= rows) return;
    }
    gemv16_row<W>(w, y, rows, row_base, x, in_dim);
}

extern "C" __global__ void qwen_gemv16_group4_bf16(
    const unsigned char *__restrict__ w0, float *__restrict__ y0, int rows0,
    const unsigned char *__restrict__ w1, float *__restrict__ y1, int rows1,
    const unsigned char *__restrict__ w2, float *__restrict__ y2, int rows2,
    const unsigned char *__restrict__ w3, float *__restrict__ y3, int rows3,
    const float *__restrict__ x, int in_dim)
{
    gemv16_group<__nv_bfloat16>(w0, y0, rows0, w1, y1, rows1,
        w2, y2, rows2, w3, y3, rows3, x, in_dim);
}

extern "C" __global__ void qwen_gemv16_group4_f16(
    const unsigned char *__restrict__ w0, float *__restrict__ y0, int rows0,
    const unsigned char *__restrict__ w1, float *__restrict__ y1, int rows1,
    const unsigned char *__restrict__ w2, float *__restrict__ y2, int rows2,
    const unsigned char *__restrict__ w3, float *__restrict__ y3, int rows3,
    const float *__restrict__ x, int in_dim)
{
    gemv16_group<__half>(w0, y0, rows0, w1, y1, rows1,
        w2, y2, rows2, w3, y3, rows3, x, in_dim);
}
