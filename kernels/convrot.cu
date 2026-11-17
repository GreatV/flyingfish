#include "gpu_compat.cuh"

// H4 = [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]].
__device__ int convrot_h4(int row, int col) {
    const int h[16] = {1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1};
    return h[row * 4 + col];
}

// Regular Hadamard entry, group a power of 4. Magnitude is (1/2) ^ log4(group).
__device__ float convrot_hadamard(int group, int row, int col) {
    int sign = 1;
    float scale = 1.0f;
    int n = group;
    while (n > 1) {
        n /= 4;
        const int rb = row / n;
        const int cb = col / n;
        row %= n;
        col %= n;
        sign *= convrot_h4(rb, cb);
        scale = __fmul_rn(scale, 0.5f);
    }
    return __fmul_rn(static_cast<float>(sign), scale);
}

__device__ __forceinline__ unsigned convrot_smem(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void convrot_ldsm_x4(unsigned (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(convrot_smem(p)));
}

__device__ __forceinline__ void convrot_ldsm_x2(unsigned (&r)[2], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(convrot_smem(p)));
}

__device__ __forceinline__ void convrot_mma_16816(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Below 17 rows: one thread writes one output. The rotated group is an f32 dot of bf16
// values, rounded back to bf16. s_x = max(|xR|, 1e-12) / 127. q is rint(xR / s_x).
extern "C" __global__ void convrot_linear(
    const float* __restrict__ x, const int8_t* __restrict__ qweight, const float* __restrict__ wscale,
    __nv_bfloat16* __restrict__ y, int rows, int k, int n, int group) {
    if (rows >= 17) __trap();
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= rows * n) return;
    const int row = idx / n;
    const int col = idx - row * n;
    const float* src = x + static_cast<long long>(row) * k;
    float peak = 0.0f;
    for (int start = 0; start < k; start += group) {
        for (int gcol = 0; gcol < group; ++gcol) {
            float sum = 0.0f;
            for (int i = 0; i < group; ++i) {
                const float value = __bfloat162float(__float2bfloat16(src[start + i]));
                sum = __fadd_rn(sum, __fmul_rn(value, convrot_hadamard(group, i, gcol)));
            }
            peak = fmaxf(peak, fabsf(__bfloat162float(__float2bfloat16(sum))));
        }
    }
    const float scale = __fdiv_rn(fmaxf(peak, 1e-12f), 127.0f);
    int acc = 0;
    const int8_t* weight = qweight + static_cast<long long>(col) * k;
    for (int start = 0; start < k; start += group) {
        for (int gcol = 0; gcol < group; ++gcol) {
            float sum = 0.0f;
            for (int i = 0; i < group; ++i) {
                const float value = __bfloat162float(__float2bfloat16(src[start + i]));
                sum = __fadd_rn(sum, __fmul_rn(value, convrot_hadamard(group, i, gcol)));
            }
            const float xr = __bfloat162float(__float2bfloat16(sum));
            float q = rintf(__fdiv_rn(xr, scale));
            q = fminf(fmaxf(q, -127.0f), 127.0f);
            acc += static_cast<int>(q) * static_cast<int>(weight[start + gcol]);
        }
    }
    y[static_cast<long long>(row) * n + col] =
        __float2bfloat16(__fmul_rn(__fmul_rn(static_cast<float>(acc), scale), wscale[col]));
}

// 17 rows and above. One block owns 16 rows and one group.
// The activation is bf16. The dot is mma.m16n8k16, bf16 inputs, f32 accumulator, K groups of 16 in order.
// grid: (ceil(rows/16), 1, k/group). block: 256.
// Dynamic smem is 16*group bf16, then 8*8*16 bf16. Bytes are 32*group+2048.
extern "C" __global__ void convrot_group_dot(const __nv_bfloat16* __restrict__ x, float* __restrict__ y, int rows,
                                             int k, int group) {
    if (rows < 17 || group < 16 || (group & 15) || (k % group)) __trap();
    constexpr int STRIDE = 16;
    extern __shared__ __align__(16) __nv_bfloat16 tile[];
    __nv_bfloat16* atile = tile;
    __nv_bfloat16* bsmem = tile + 16 * group;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row0 = static_cast<int>(blockIdx.x) * 16;
    const int x_col0 = static_cast<int>(blockIdx.z) * group;
    const int count = 16 * group;
    for (int i = static_cast<int>(threadIdx.x); i < count; i += static_cast<int>(blockDim.x)) {
        const int r = i / group;
        const int c = i - r * group;
        const int gr = row0 + r;
        __nv_bfloat16 v = __float2bfloat16(0.f);
        if (gr < rows) v = x[static_cast<long long>(gr) * k + x_col0 + c];
        const int panel = c >> 4;
        const int local = c & 15;
        atile[panel * (16 * STRIDE) + r * STRIDE + local] = v;
    }
    __syncthreads();
    __nv_bfloat16* bt = bsmem + warp * (8 * STRIDE);
    const int nsub = group >> 3;
    for (int sub = warp; sub < nsub; sub += 8) {
        const int n0 = sub << 3;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int k0 = 0; k0 < group; k0 += 16) {
            for (int i = lane; i < 8 * 16; i += 32) {
                const int n = i >> 4;
                const int c = i & 15;
                bt[n * STRIDE + c] = __float2bfloat16(convrot_hadamard(group, k0 + c, n0 + n));
            }
            __syncwarp();
            unsigned a[4];
            const __nv_bfloat16* ap = atile + (k0 >> 4) * (16 * STRIDE);
            convrot_ldsm_x4(a, ap + (lane & 15) * STRIDE + (lane >> 4) * 8);
            unsigned b[2];
            convrot_ldsm_x2(b, bt + (lane & 7) * STRIDE + ((lane >> 3) & 1) * 8);
            convrot_mma_16816(acc, a, b);
            __syncwarp();
        }
        const int rbase = lane >> 2;
        const int cbase = (lane & 3) * 2;
        const int rows_out[4] = {rbase, rbase, rbase + 8, rbase + 8};
        const int cols_out[4] = {cbase, cbase + 1, cbase, cbase + 1};
        for (int t = 0; t < 4; ++t) {
            const int r = rows_out[t];
            const int c = cols_out[t];
            if (row0 + r < rows && n0 + c < group)
                y[static_cast<long long>(row0 + r) * k + x_col0 + n0 + c] = acc[t];
        }
    }
}

// Block 256. Dynamic smem is tile_rows * k f32. Each live row is loaded once, coalesced.
// Warp peaks reuse that row's first 8 floats, then the reciprocal does. k >= 8.
// s = max(peak, 1e-12) / 127. q = rint(v * (1/s)), clamped to [-127, 127].
constexpr int CONVROT_QUANT_THREADS = 256;

extern "C" __global__ void convrot_quantize(const float* __restrict__ xr, int8_t* __restrict__ codes,
                                            float* __restrict__ scales, int rows, int k, int tile_rows) {
    extern __shared__ float quant_tile[];
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int row0 = static_cast<int>(blockIdx.x) * tile_rows;
    for (int r = 0; r < tile_rows; ++r) {
        const int row = row0 + r;
        if (row < rows) {
            const float* src = xr + static_cast<long long>(row) * k;
            float* dst = quant_tile + static_cast<long long>(r) * k;
            for (int col = tid; col < k; col += CONVROT_QUANT_THREADS) dst[col] = src[col];
        }
    }
    __syncthreads();
    for (int r = 0; r < tile_rows; ++r) {
        const int row = row0 + r;
        const bool live = row < rows;
        float* row_tile = quant_tile + static_cast<long long>(r) * k;
        float peak = 0.f;
        if (live) {
            for (int col = tid; col < k; col += CONVROT_QUANT_THREADS)
                peak = fmaxf(peak, fabsf(row_tile[col]));
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, off));
        __syncthreads();
        float saved = 0.f;
        if (live && tid < 8) saved = row_tile[tid];
        __syncthreads();
        if (live && lane == 0) row_tile[wid] = peak;
        __syncthreads();
        if (live && tid == 0) {
            float reduced = 0.f;
            for (int w = 0; w < 8; ++w) reduced = fmaxf(reduced, row_tile[w]);
            const float scale = __fdiv_rn(fmaxf(reduced, 1e-12f), 127.f);
            scales[row] = scale;
            row_tile[0] = __fdiv_rn(1.f, scale);
        }
        __syncthreads();
        float inv = 0.f;
        if (live) inv = row_tile[0];
        __syncthreads();
        if (live && tid < 8) row_tile[tid] = saved;
        __syncthreads();
        if (live) {
            int8_t* dst = codes + static_cast<long long>(row) * k;
            for (int col = tid; col < k; col += CONVROT_QUANT_THREADS) {
                float q = rintf(__fmul_rn(row_tile[col], inv));
                q = fminf(fmaxf(q, -127.f), 127.f);
                dst[col] = static_cast<int8_t>(static_cast<int>(q));
            }
        }
        __syncthreads();
    }
}

// m16n8k32.row.col.s32.s8.s8. BM 128, BN 256, BK 64, stride 80, 2 stages, 512 threads.
// Dynamic smem 61440. grid (ceil(n/256), ceil(rows/128)).
constexpr int CONVROT_BM = 128;
constexpr int CONVROT_BN = 256;
constexpr int CONVROT_BK = 64;
constexpr int CONVROT_STRIDE = 80;
constexpr int CONVROT_STAGE = (CONVROT_BM + CONVROT_BN) * CONVROT_STRIDE;
static_assert(CONVROT_STAGE * 2 == 61440, "convrot product smem");

__device__ __forceinline__ void convrot_cp16(void* dst, const void* src, int sz) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(convrot_smem(dst)), "l"(src), "r"(sz));
}

__device__ __forceinline__ void convrot_cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int PENDING>
__device__ __forceinline__ void convrot_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(PENDING) : "memory");
}

__device__ __forceinline__ unsigned convrot_lds(const int8_t* p) {
    unsigned v;
    asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(v) : "r"(convrot_smem(p)));
    return v;
}

__device__ __forceinline__ void convrot_mma_s8(int& d0, int& d1, int& d2, int& d3, unsigned a0, unsigned a1,
                                               unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One 16-byte cp.async per A row segment and two per B row. srcSize 0 zero-fills an OOB row.
__device__ __forceinline__ void convrot_stage(int8_t* smem, const int8_t* __restrict__ codes,
                                              const int8_t* __restrict__ qweight, int rows, int k, int n, int m0,
                                              int n0, int k0, int stage) {
    int8_t* a = smem + stage * CONVROT_STAGE;
    int8_t* b = a + CONVROT_BM * CONVROT_STRIDE;
    const int t = static_cast<int>(threadIdx.x);
    const int ar = t >> 2;
    const int aseg = t & 3;
    const int gr = m0 + ar;
    const int8_t* asrc = codes;
    int asz = 0;
    if (gr < rows) {
        asrc = codes + (static_cast<long long>(gr) * k + k0 + aseg * 16);
        asz = 16;
    }
    convrot_cp16(a + ar * CONVROT_STRIDE + aseg * 16, asrc, asz);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int linear = t + i * 512;
        const int bc = linear >> 2;
        const int bseg = linear & 3;
        const int gc = n0 + bc;
        const int8_t* bsrc = qweight;
        int bsz = 0;
        if (gc < n) {
            bsrc = qweight + (static_cast<long long>(gc) * k + k0 + bseg * 16);
            bsz = 16;
        }
        convrot_cp16(b + bc * CONVROT_STRIDE + bseg * 16, bsrc, bsz);
    }
}

// Lane groupID = lane>>2, thread = lane&3. A u32 at (group, thread*4), (group+8, thread*4),
// (group, thread*4+16), (group+8, thread*4+16). B u32 at column group, K thread*4 and +16.
__device__ __forceinline__ void convrot_mma_tile(const int8_t* stage, int (&acc)[2][8][4], int warp_m, int warp_n,
                                                 int group, int t4) {
    const int8_t* a = stage;
    const int8_t* b = stage + CONVROT_BM * CONVROT_STRIDE;
    const int row_base = warp_m * 32;
    const int col_base = warp_n * 64;
#pragma unroll
    for (int ki = 0; ki < 2; ++ki) {
        const int kbase = ki * 32 + t4;
        unsigned af[2][4];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            const int r = row_base + mi * 16 + group;
            af[mi][0] = convrot_lds(a + (r * CONVROT_STRIDE + kbase));
            af[mi][1] = convrot_lds(a + ((r + 8) * CONVROT_STRIDE + kbase));
            af[mi][2] = convrot_lds(a + (r * CONVROT_STRIDE + kbase + 16));
            af[mi][3] = convrot_lds(a + ((r + 8) * CONVROT_STRIDE + kbase + 16));
        }
        unsigned bf[8][2];
#pragma unroll
        for (int ni = 0; ni < 8; ++ni) {
            const int c = col_base + ni * 8 + group;
            bf[ni][0] = convrot_lds(b + (c * CONVROT_STRIDE + kbase));
            bf[ni][1] = convrot_lds(b + (c * CONVROT_STRIDE + kbase + 16));
        }
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
            for (int ni = 0; ni < 8; ++ni) {
                convrot_mma_s8(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3], af[mi][0],
                               af[mi][1], af[mi][2], af[mi][3], bf[ni][0], bf[ni][1]);
            }
        }
    }
}

__device__ __forceinline__ void convrot_store_one(__nv_bfloat16* __restrict__ y, const float* __restrict__ scales,
                                                  const float* __restrict__ wscale, int rows, int n, int r, int c,
                                                  int acc) {
    if (r < rows && c < n) {
        const float v = __fmul_rn(__fmul_rn(static_cast<float>(acc), scales[r]), wscale[c]);
        y[static_cast<long long>(r) * n + c] = __float2bfloat16(v);
    }
}

// ptxas sm_89: 124 registers, 0 spills.
extern "C" __global__ void __maxnreg__(124) convrot_product(const int8_t* __restrict__ codes, const float* __restrict__ scales,
                                           const int8_t* __restrict__ qweight, const float* __restrict__ wscale,
                                           __nv_bfloat16* __restrict__ y, int rows, int k, int n) {
    if (rows <= 0 || n <= 0 || k <= 0 || (k & (CONVROT_BK - 1))) __trap();
    extern __shared__ __align__(16) int8_t smem[];
    const int m0 = static_cast<int>(blockIdx.y) * CONVROT_BM;
    const int n0 = static_cast<int>(blockIdx.x) * CONVROT_BN;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int warp_m = warp >> 2;
    const int warp_n = warp & 3;
    const int group = lane >> 2;
    const int t4 = (lane & 3) << 2;
    const int t2 = (lane & 3) << 1;
    int acc[2][8][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 8; ++ni) {
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[mi][ni][e] = 0;
        }
    }
    const int tiles = k >> 6;
    int stage = 0;
    convrot_stage(smem, codes, qweight, rows, k, n, m0, n0, 0, 0);
    convrot_cp_commit();
    for (int kt = 0; kt < tiles; ++kt) {
        if (kt + 1 < tiles) {
            convrot_stage(smem, codes, qweight, rows, k, n, m0, n0, (kt + 1) * CONVROT_BK, stage ^ 1);
            convrot_cp_commit();
            convrot_cp_wait<1>();
        } else {
            convrot_cp_wait<0>();
        }
        __syncthreads();
        convrot_mma_tile(smem + stage * CONVROT_STAGE, acc, warp_m, warp_n, group, t4);
        __syncthreads();
        stage ^= 1;
    }
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 8; ++ni) {
            const int r0 = m0 + warp_m * 32 + mi * 16 + group;
            const int c0 = n0 + warp_n * 64 + ni * 8 + t2;
            convrot_store_one(y, scales, wscale, rows, n, r0, c0, acc[mi][ni][0]);
            convrot_store_one(y, scales, wscale, rows, n, r0, c0 + 1, acc[mi][ni][1]);
            convrot_store_one(y, scales, wscale, rows, n, r0 + 8, c0, acc[mi][ni][2]);
            convrot_store_one(y, scales, wscale, rows, n, r0 + 8, c0 + 1, acc[mi][ni][3]);
        }
    }
}
