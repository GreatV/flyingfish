// MMA GEMM for prefill projections (prefill-design §2.1). One skeleton
// per weight policy; accumulation is f32. y[t][r] = sum_k x[t][k] *
// w[r][k]; x token-major [T, in], w row-major [rows, in]. Tile: BM=64
// rows x BT=64 tokens x BK=64, 4 warps at 32x32. Staging is cp.async
// double-buffered; A and 16-bit B fragments load via ldmatrix; int4 B
// dequantises in registers (scale*code stays f32 until the bf16
// fragment, one rounding). BK equals the int4 group size, so K-tile kt
// is group kt: int4_mma converts x to bf16 while staging, reduces each
// token's f32 group sum before the conversion, and folds
// biases[r][kt] * sumx[t] into the accumulator — no xsum/bias_part
// kernels and no [rows, T] intermediate.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define BM 64
#define BT 64
#define BK 64

// m16n8k16 lane mappings (PTX ISA): group g = lane>>2, tig = lane&3.
__device__ __forceinline__ int lane_g() { return (threadIdx.x & 31) >> 2; }
__device__ __forceinline__ int lane_tig() { return threadIdx.x & 3; }

__device__ __forceinline__ void cp16(void* smem, const void* gmem, bool valid)
{
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    const int sz = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(sz));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait1() { asm volatile("cp.async.wait_group 1;\n"); }
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;\n"); }

// ldmatrix.x4: reg i comes from the 8x8 matrix addressed by lanes 8i..8i+7.
__device__ __forceinline__ void ldsm4(
    const void* smem, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3)
{
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(s));
}

// A fragment addresses for one m16k16 at (tile_m, ks): lanes 0-15 give
// rows tile_m..tile_m+15 at col ks*16 (+8 for lanes 16-31).
__device__ __forceinline__ const void* a_addr(
    const void* sa, int tile_m, int ks, int stride)
{
    const int lane = threadIdx.x & 31;
    const int row = tile_m + (lane & 15);
    const int col = ks * 16 + (lane >> 4) * 8;
    return static_cast<const char*>(sa) + (row * stride + col) * 2;
}

// B fragment addresses covering two n8 tiles at (tile_n, ks): matrix j
// is tile (j/2), k-half (j%2). SMEM is [N][K] row-major; the mma B
// operand's (k-pair, n) pairs are adjacent there, so no .trans.
__device__ __forceinline__ const void* b_addr(
    const void* sb, int tile_n, int ks, int stride)
{
    const int lane = threadIdx.x & 31;
    const int row = tile_n + (lane & 7) + ((lane >> 4) << 3);
    const int col = ks * 16 + (((lane >> 3) & 1) << 3);
    return static_cast<const char*>(sb) + (row * stride + col) * 2;
}

__device__ __forceinline__ void mma_bf16(float c[4], const unsigned a[4], const unsigned b[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma_f16(float c[4], const unsigned a[4], const unsigned b[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// The C fragment's (row, col) inside the tile, then into y [T, rows].
__device__ __forceinline__ void c_store(
    float* __restrict__ y, const float c[4], int m0, int n0, int t0, int r0,
    int rows, int t_max)
{
    const int g = lane_g();
    const int t2 = lane_tig() * 2;
    const int t_row = t0 + m0 + g;
    const int t_row2 = t_row + 8;
    const int r_col = r0 + n0 + t2;
    if (t_row < t_max && r_col < rows) {
        y[(long long)t_row * rows + r_col] = c[0];
        if (r_col + 1 < rows)
            y[(long long)t_row * rows + r_col + 1] = c[1];
    }
    if (t_row2 < t_max && r_col < rows) {
        y[(long long)t_row2 * rows + r_col] = c[2];
        if (r_col + 1 < rows)
            y[(long long)t_row2 * rows + r_col + 1] = c[3];
    }
}

extern "C" __global__ void x_to_bf16(
    const float* __restrict__ x, __nv_bfloat16* __restrict__ xb, long long total)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total)
        xb[i] = __float2bfloat16(x[i]);
}

extern "C" __global__ void x_to_f16(
    const float* __restrict__ x, __half* __restrict__ xh, long long total)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total)
        xh[i] = __float2half(x[i]);
}

// Stage one x tile (BM x BK bf16/f16, row-major) via cp.async.
template <typename T>
__device__ __forceinline__ void stage_x(
    T* sa, const T* __restrict__ xb, int t0, int k0, int tokens, int in_dim)
{
    for (int j = threadIdx.x; j < BM * (BK / 8); j += blockDim.x) {
        const int r = j / (BK / 8);
        const int cc = j % (BK / 8);
        const bool valid = t0 + r < tokens;
        const T* src = valid ? xb + (long long)(t0 + r) * in_dim + k0 + cc * 8 : xb;
        cp16(sa + j * 8, src, valid);
    }
}

// Stage one x tile (BT tokens x BK) converting f32 to bf16 on the way
// in, and reduce each token's group sum in f32 before the conversion.
// 128 threads walk the tile in float4 chunks; row r's 16 lanes reduce
// via a width-16 shuffle tree, so ssumx[r] is deterministic.
// Stage one w tile (BM rows x BK, row-major over n) via cp.async.
template <typename T>
__device__ __forceinline__ void stage_w(
    T* sb, const T* __restrict__ w, int r0, int k0, int rows, int in_dim)
{
    for (int j = threadIdx.x; j < BM * (BK / 8); j += blockDim.x) {
        const int r = j / (BK / 8);
        const int cc = j % (BK / 8);
        const bool valid = r0 + r < rows;
        const T* src = valid ? w + (long long)(r0 + r) * in_dim + k0 + cc * 8 : w;
        cp16(sb + j * 8, src, valid);
    }
}

__device__ __forceinline__ void acc_zero(float acc[2][4][4])
{
    #pragma unroll
    for (int mt = 0; mt < 2; mt++)
        #pragma unroll
        for (int nt = 0; nt < 4; nt++)
            #pragma unroll
            for (int i = 0; i < 4; i++) acc[mt][nt][i] = 0.0f;
}

__device__ __forceinline__ void c_store_all(
    float* __restrict__ y, const float acc[2][4][4], int wm, int wn, int t0,
    int r0, int rows, int tokens)
{
    #pragma unroll
    for (int mt = 0; mt < 2; mt++)
        #pragma unroll
        for (int nt = 0; nt < 4; nt++)
            c_store(y, acc[mt][nt], wm + mt * 16, wn + nt * 8, t0, r0, rows, tokens);
}

// Per-group f32 sums of x, [groups, T]; the deleted xsum's body,
// reproduced verbatim so the bias term keeps its exact reduction order.
extern "C" __global__ void int4_group_sums(
    const float* __restrict__ x,      // [T, in_dim]
    float* __restrict__ xsum,         // [groups, T]
    int in_dim, int groups)
{
    const int t = blockIdx.x;
    const float* row = x + (long long)t * in_dim;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    for (int g = warp; g < groups; g += blockDim.x / 32) {
        float acc = 0.0f;
        for (int k = lane; k < 64; k += 32) acc += row[g * 64 + k];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) xsum[(long long)g * gridDim.x + t] = acc;
    }
}

extern "C" __global__ void bf16_mma(
    const __nv_bfloat16* __restrict__ w,   // [rows, in]
    const __nv_bfloat16* __restrict__ xb,  // [T, in]
    float* __restrict__ y,                 // [T, rows] f32
    int rows, int in_dim, int tokens)
{
    __shared__ __nv_bfloat16 sa[2][BM * BK];
    __shared__ __nv_bfloat16 sb[2][BM * BK];
    const int t0 = blockIdx.x * BT;
    const int r0 = blockIdx.y * BM;
    const int warp = threadIdx.x >> 5;
    const int wm = (warp & 1) * 32;
    const int wn = (warp >> 1) * 32;
    const int tiles = (in_dim + BK - 1) / BK;

    float acc[2][4][4];
    acc_zero(acc);

    stage_x(sa[0], xb, t0, 0, tokens, in_dim);
    stage_w(sb[0], w, r0, 0, rows, in_dim);
    cp_commit();
    for (int kt = 0; kt < tiles; kt++) {
        if (kt + 1 < tiles) {
            stage_x(sa[(kt + 1) & 1], xb, t0, (kt + 1) * BK, tokens, in_dim);
            stage_w(sb[(kt + 1) & 1], w, r0, (kt + 1) * BK, rows, in_dim);
            cp_commit();
            cp_wait1();
        } else {
            cp_wait0();
        }
        __syncthreads();
        const int buf = kt & 1;
        #pragma unroll
        for (int ks = 0; ks < BK / 16; ks++) {
            #pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                unsigned a[4];
                ldsm4(a_addr(sa[buf], wm + mt * 16, ks, BK), a[0], a[1], a[2], a[3]);
                #pragma unroll
                for (int np = 0; np < 2; np++) {
                    unsigned b[4];
                    ldsm4(b_addr(sb[buf], wn + np * 16, ks, BK), b[0], b[1], b[2], b[3]);
                    mma_bf16(acc[mt][2 * np], a, b);
                    mma_bf16(acc[mt][2 * np + 1], a, b + 2);
                }
            }
        }
        __syncthreads();
    }
    c_store_all(y, acc, wm, wn, t0, r0, rows, tokens);
}

// int4 B fragment for one m16n8k16: lane reads its two packed words from
// SMEM codes [BM][BK/8] and dequantises in f32 until the bf16 fragment.
// codes word (ks*2) holds k = ks*16..+7, word (ks*2+1) the next 8.
__device__ __forceinline__ void b_frag_int4(
    const unsigned* __restrict__ scodes, const __nv_bfloat16* __restrict__ sscale,
    int tile_n, int ks, unsigned out[2])
{
    const int g = lane_g();
    const int tig = lane_tig();
    const int n = tile_n + g;
    const unsigned w0 = scodes[n * (BK / 8) + ks * 2];
    const unsigned w1 = scodes[n * (BK / 8) + ks * 2 + 1];
    const float scale = __bfloat162float(sscale[n]);
    const unsigned c0 = (w0 >> (8 * tig)) & 0xFu;
    const unsigned c1 = (w0 >> (8 * tig + 4)) & 0xFu;
    const unsigned c2 = (w1 >> (8 * tig)) & 0xFu;
    const unsigned c3 = (w1 >> (8 * tig + 4)) & 0xFu;
    const __nv_bfloat16 b0 = __float2bfloat16(scale * (float)c0);
    const __nv_bfloat16 b1 = __float2bfloat16(scale * (float)c1);
    const __nv_bfloat16 b2 = __float2bfloat16(scale * (float)c2);
    const __nv_bfloat16 b3 = __float2bfloat16(scale * (float)c3);
    out[0] = (unsigned)__bfloat16_as_ushort(b0) | ((unsigned)__bfloat16_as_ushort(b1) << 16);
    out[1] = (unsigned)__bfloat16_as_ushort(b2) | ((unsigned)__bfloat16_as_ushort(b3) << 16);
}

extern "C" __global__ __launch_bounds__(128, 4) void int4_mma(
    const unsigned* __restrict__ packed,      // [rows, in/8]
    const __nv_bfloat16* __restrict__ scales, // [rows, in/64]
    const __nv_bfloat16* __restrict__ biases, // [rows, in/64]
    const float* __restrict__ xs,             // [in/64, T] group sums, f32
    const __nv_bfloat16* __restrict__ xb,     // [T, in]
    float* __restrict__ y,                    // [T, rows] f32
    int rows, int in_dim, int tokens)
{
    __shared__ __nv_bfloat16 sa[2][BM * BK];
    __shared__ unsigned scodes[2][BM * (BK / 8)];
    __shared__ __nv_bfloat16 sscale[2][BM];
    const int t0 = blockIdx.x * BT;
    const int r0 = blockIdx.y * BM;
    const int warp = threadIdx.x >> 5;
    const int wm = (warp & 1) * 32;
    const int wn = (warp >> 1) * 32;
    const int words = in_dim / 8;
    const int groups = in_dim / 64;
    const int tiles = (in_dim + BK - 1) / BK;

    float acc[2][4][4];
    acc_zero(acc);
    // The bias term accumulates separately from the dot, in tile order,
    // and joins once at the store — the deleted bias_part summed the
    // same sequence on its own before the single add.
    float bacc[2][4][4];
    acc_zero(bacc);

    auto stage = [&](int kt, int buf) {
        const int k0 = kt * BK;
        stage_x(sa[buf], xb, t0, k0, tokens, in_dim);
        for (int j = threadIdx.x; j < BM * 2; j += blockDim.x) {
            const int r = j / 2;
            const int cc = j % 2;
            const bool valid = r0 + r < rows;
            const unsigned* src = valid
                ? packed + (long long)(r0 + r) * words + k0 / 8 + cc * 4
                : packed;
            cp16(scodes[buf] + j * 4, src, valid);
        }
        for (int r = threadIdx.x; r < BM; r += blockDim.x)
            sscale[buf][r] = (r0 + r < rows)
                ? scales[(long long)(r0 + r) * groups + k0 / 64]
                : __float2bfloat16(0.0f);
    };

    stage(0, 0);
    cp_commit();
    for (int kt = 0; kt < tiles; kt++) {
        if (kt + 1 < tiles) {
            stage(kt + 1, (kt + 1) & 1);
            cp_commit();
            cp_wait1();
        } else {
            cp_wait0();
        }
        __syncthreads();
        const int buf = kt & 1;
        #pragma unroll
        for (int ks = 0; ks < BK / 16; ks++) {
            #pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                unsigned a[4];
                ldsm4(a_addr(sa[buf], wm + mt * 16, ks, BK), a[0], a[1], a[2], a[3]);
                #pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    unsigned b[2];
                    b_frag_int4(scodes[buf], sscale[buf], wn + nt * 8, ks, b);
                    mma_bf16(acc[mt][nt], a, b);
                }
            }
        }
        // K-tile kt is group kt: fold biases[r][kt] * xs[kt][t] into the
        // bias accumulator, in the deleted bias_part's ascending-group
        // order; the fragment's (token, row) pairs follow c_store.
        const int g = lane_g();
        const int t2 = lane_tig() * 2;
        #pragma unroll
        for (int mt = 0; mt < 2; mt++) {
            const int t_row = wm + mt * 16 + g;
            const int t_row2 = t_row + 8;
            const float sx0 = (t0 + t_row < tokens) ? xs[(long long)kt * tokens + t0 + t_row] : 0.0f;
            const float sx8 = (t0 + t_row2 < tokens) ? xs[(long long)kt * tokens + t0 + t_row2] : 0.0f;
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                const int r_col = wn + nt * 8 + t2;
                const int gi = (long long)(r0 + r_col) * groups + kt;
                const int gi2 = (long long)(r0 + r_col + 1) * groups + kt;
                const float b0 = (r0 + r_col < rows)
                    ? __bfloat162float(biases[gi])
                    : 0.0f;
                const float b1 = (r0 + r_col + 1 < rows)
                    ? __bfloat162float(biases[gi2])
                    : 0.0f;
                bacc[mt][nt][0] += b0 * sx0;
                bacc[mt][nt][1] += b1 * sx0;
                bacc[mt][nt][2] += b0 * sx8;
                bacc[mt][nt][3] += b1 * sx8;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int mt = 0; mt < 2; mt++)
        #pragma unroll
        for (int nt = 0; nt < 4; nt++)
            #pragma unroll
            for (int i = 0; i < 4; i++) acc[mt][nt][i] += bacc[mt][nt][i];
    c_store_all(y, acc, wm, wn, t0, r0, rows, tokens);
}

extern "C" __global__ void f16_mma(
    const __half* __restrict__ w,   // [rows, in]
    const __half* __restrict__ xh,  // [T, in]
    float* __restrict__ y,          // [T, rows] f32
    int rows, int in_dim, int tokens)
{
    __shared__ __half sa[2][BM * BK];
    __shared__ __half sb[2][BM * BK];
    const int t0 = blockIdx.x * BT;
    const int r0 = blockIdx.y * BM;
    const int warp = threadIdx.x >> 5;
    const int wm = (warp & 1) * 32;
    const int wn = (warp >> 1) * 32;
    const int tiles = (in_dim + BK - 1) / BK;

    float acc[2][4][4];
    acc_zero(acc);

    stage_x(sa[0], xh, t0, 0, tokens, in_dim);
    stage_w(sb[0], w, r0, 0, rows, in_dim);
    cp_commit();
    for (int kt = 0; kt < tiles; kt++) {
        if (kt + 1 < tiles) {
            stage_x(sa[(kt + 1) & 1], xh, t0, (kt + 1) * BK, tokens, in_dim);
            stage_w(sb[(kt + 1) & 1], w, r0, (kt + 1) * BK, rows, in_dim);
            cp_commit();
            cp_wait1();
        } else {
            cp_wait0();
        }
        __syncthreads();
        const int buf = kt & 1;
        #pragma unroll
        for (int ks = 0; ks < BK / 16; ks++) {
            #pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                unsigned a[4];
                ldsm4(a_addr(sa[buf], wm + mt * 16, ks, BK), a[0], a[1], a[2], a[3]);
                #pragma unroll
                for (int np = 0; np < 2; np++) {
                    unsigned b[4];
                    ldsm4(b_addr(sb[buf], wn + np * 16, ks, BK), b[0], b[1], b[2], b[3]);
                    mma_f16(acc[mt][2 * np], a, b);
                    mma_f16(acc[mt][2 * np + 1], a, b + 2);
                }
            }
        }
        __syncthreads();
    }
    c_store_all(y, acc, wm, wn, t0, r0, rows, tokens);
}
