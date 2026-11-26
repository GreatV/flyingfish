#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800 && !defined(__HIP__)
// sm75: cp.async.cg 16B is ld.global.cg.v4.u32 plus st.shared.v4.u32. mma.bf16 and mma.m16n8k32 trap.
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
    asm volatile("trap;" ::"f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]), "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                 "r"(b[0]), "r"(b[1]));
}


__device__ __forceinline__ uint4 convrot_load_u128(const void* p) {
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
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

// m16n8k32.row.col.s32.s8.s8. BM 128, BN 256, BK 64, stride 80, 2 stages, 512 threads.
// Dynamic smem 61440 N-fast, 49152 M-fast.
constexpr int CONVROT_BM = 128;
constexpr int CONVROT_BN = 256;
constexpr int CONVROT_BK = 64;
constexpr int CONVROT_STRIDE = 80;
template <bool SWIZZLED>
constexpr int CONVROT_PRODUCT_STRIDE = SWIZZLED ? CONVROT_BK : CONVROT_STRIDE;
template <bool SWIZZLED>
constexpr int CONVROT_STAGE = (CONVROT_BM + CONVROT_BN) * CONVROT_PRODUCT_STRIDE<SWIZZLED>;
static_assert(CONVROT_STAGE<false> * 2 == 61440 && CONVROT_STAGE<true> * 2 == 49152, "convrot product smem");

template <bool SWIZZLED>
__device__ __forceinline__ int convrot_segment_offset(int row, int column) {
    if constexpr (SWIZZLED) {
        return row * CONVROT_BK + (column ^ (((row >> 1) & 3) << 4));
    } else {
        return row * CONVROT_STRIDE + column;
    }
}

__device__ __forceinline__ void convrot_cp16(void* dst, const void* src, int sz) {
    unsigned x = 0u, y = 0u, z = 0u, w = 0u;
    if (sz == 16) {
        asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                     : "l"(src));
    }
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(convrot_smem(dst)), "r"(x), "r"(y), "r"(z), "r"(w));
}


__device__ __forceinline__ void convrot_cp_commit() {
    asm volatile("" ::: "memory");
}


template <int PENDING>
__device__ __forceinline__ void convrot_cp_wait() {
    static_assert(PENDING == 0 || PENDING == 1, "convrot stages");
    __syncthreads();
}


__device__ __forceinline__ unsigned convrot_lds(const int8_t* p) {
    unsigned v;
    asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(v) : "r"(convrot_smem(p)));
    return v;
}

__device__ __forceinline__ void convrot_mma_s8(int& d0, int& d1, int& d2, int& d3, unsigned a0, unsigned a1,
                                               unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("trap;" ::"r"(d0), "r"(d1), "r"(d2), "r"(d3), "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0),
                 "r"(b1));
}


// One 16-byte load per A row segment and two per B row. sz == 0 writes zeros for an OOB row.
template <bool SWIZZLED>
__device__ __forceinline__ void convrot_stage(int8_t* smem, const int8_t* __restrict__ codes,
                                              const int8_t* __restrict__ qweight, int rows, int k, int n, int m0,
                                              int n0, int k0, int stage) {
    int8_t* a = smem + stage * CONVROT_STAGE<SWIZZLED>;
    int8_t* b = a + CONVROT_BM * CONVROT_PRODUCT_STRIDE<SWIZZLED>;
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
    convrot_cp16(a + convrot_segment_offset<SWIZZLED>(ar, aseg * 16), asrc, asz);
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
        convrot_cp16(b + convrot_segment_offset<SWIZZLED>(bc, bseg * 16), bsrc, bsz);
    }
}

// Lane groupID = lane>>2, thread = lane&3. A u32 at (group, thread*4), (group+8, thread*4),
// (group, thread*4+16), (group+8, thread*4+16). B u32 at column group, K thread*4 and +16.
template <bool MATRIX_LOAD, bool SWIZZLED>
__device__ __forceinline__ void convrot_mma_tile(const int8_t* stage, int (&acc)[2][8][4], int warp_m, int warp_n,
                                                 int group, int t4, int lane) {
    const int8_t* a = stage;
    const int8_t* b = stage + CONVROT_BM * CONVROT_PRODUCT_STRIDE<SWIZZLED>;
    const int row_base = warp_m * 32;
    const int col_base = warp_n * 64;
#pragma unroll
    for (int ki = 0; ki < 2; ++ki) {
        const int kbase = ki * 32 + t4;
        unsigned af[2][4];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            if constexpr (MATRIX_LOAD) {
                const int r = row_base + mi * 16 + (lane & 15);
                const int k_panel = ki * 32 + (lane >> 4) * 16;
                convrot_ldsm_x4(af[mi], a + convrot_segment_offset<SWIZZLED>(r, k_panel));
            } else {
                const int r = row_base + mi * 16 + group;
                af[mi][0] = convrot_lds(a + (r * CONVROT_STRIDE + kbase));
                af[mi][1] = convrot_lds(a + ((r + 8) * CONVROT_STRIDE + kbase));
                af[mi][2] = convrot_lds(a + (r * CONVROT_STRIDE + kbase + 16));
                af[mi][3] = convrot_lds(a + ((r + 8) * CONVROT_STRIDE + kbase + 16));
            }
        }
        unsigned bf[8][2];
        if constexpr (MATRIX_LOAD) {
#pragma unroll
            for (int ni = 0; ni < 8; ni += 2) {
                const int c = col_base + ni * 8 + (lane & 7) + (lane >> 4) * 8;
                const int k_panel = ki * 32 + ((lane >> 3) & 1) * 16;
                unsigned pair[4];
                convrot_ldsm_x4(pair, b + convrot_segment_offset<SWIZZLED>(c, k_panel));
                bf[ni][0] = pair[0];
                bf[ni][1] = pair[1];
                bf[ni + 1][0] = pair[2];
                bf[ni + 1][1] = pair[3];
            }
        } else {
#pragma unroll
            for (int ni = 0; ni < 8; ++ni) {
                const int c = col_base + ni * 8 + group;
                bf[ni][0] = convrot_lds(b + (c * CONVROT_STRIDE + kbase));
                bf[ni][1] = convrot_lds(b + (c * CONVROT_STRIDE + kbase + 16));
            }
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

__device__ __forceinline__ float convrot_ldg_f32(const float* p, bool live) {
    float v = 0.f;
    if (live) {
        asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(v) : "l"(p));
    }
    return v;
}

__device__ __forceinline__ void convrot_store_one(__nv_bfloat16* __restrict__ y, int rows, int n, int r, int c, int acc,
                                                  float sx, float sw) {
    if (r < rows && c < n) {
        const float v = __fmul_rn(__fmul_rn(static_cast<float>(acc), sx), sw);
        y[static_cast<long long>(r) * n + c] = __float2bfloat16(v);
    }
}

template <bool MATRIX_LOAD, bool SWIZZLED = false>
__device__ __forceinline__ void convrot_product_tile(
    const int8_t* __restrict__ codes, const float* __restrict__ scales,
    const int8_t* __restrict__ qweight, const float* __restrict__ wscale,
    __nv_bfloat16* __restrict__ y, int rows, int k, int n, int m_tile, int n_tile) {
    static_assert(!SWIZZLED || MATRIX_LOAD);
    if (rows <= 0 || n <= 0 || k <= 0 || (k & (CONVROT_BK - 1))) __trap();
    extern __shared__ __align__(16) int8_t smem[];
    const int m0 = m_tile * CONVROT_BM;
    const int n0 = n_tile * CONVROT_BN;
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
    convrot_stage<SWIZZLED>(smem, codes, qweight, rows, k, n, m0, n0, 0, 0);
    convrot_cp_commit();
    for (int kt = 0; kt < tiles; ++kt) {
        if (kt + 1 < tiles) {
            convrot_stage<SWIZZLED>(smem, codes, qweight, rows, k, n, m0, n0, (kt + 1) * CONVROT_BK, stage ^ 1);
            convrot_cp_commit();
            convrot_cp_wait<1>();
        } else {
            convrot_cp_wait<0>();
        }
        __syncthreads();
        convrot_mma_tile<MATRIX_LOAD, SWIZZLED>(smem + stage * CONVROT_STAGE<SWIZZLED>, acc, warp_m, warp_n, group, t4, lane);
        __syncthreads();
        stage ^= 1;
    }
    // 4 row scales and 16 column scales, one global load each.
    float sx[2][2];
    float sw[8][2];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        const int r0 = m0 + warp_m * 32 + mi * 16 + group;
        sx[mi][0] = convrot_ldg_f32(scales + r0, r0 < rows);
        sx[mi][1] = convrot_ldg_f32(scales + (r0 + 8), (r0 + 8) < rows);
    }
#pragma unroll
    for (int ni = 0; ni < 8; ++ni) {
        const int c0 = n0 + warp_n * 64 + ni * 8 + t2;
        sw[ni][0] = convrot_ldg_f32(wscale + c0, c0 < n);
        sw[ni][1] = convrot_ldg_f32(wscale + (c0 + 1), (c0 + 1) < n);
    }
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 8; ++ni) {
            const int r0 = m0 + warp_m * 32 + mi * 16 + group;
            const int c0 = n0 + warp_n * 64 + ni * 8 + t2;
            convrot_store_one(y, rows, n, r0, c0, acc[mi][ni][0], sx[mi][0], sw[ni][0]);
            convrot_store_one(y, rows, n, r0, c0 + 1, acc[mi][ni][1], sx[mi][0], sw[ni][1]);
            convrot_store_one(y, rows, n, r0 + 8, c0, acc[mi][ni][2], sx[mi][1], sw[ni][0]);
            convrot_store_one(y, rows, n, r0 + 8, c0 + 1, acc[mi][ni][3], sx[mi][1], sw[ni][1]);
        }
    }
}

extern "C" __global__ void __maxnreg__(124) convrot_product(
    const int8_t* codes, const float* scales, const int8_t* qweight, const float* wscale,
    __nv_bfloat16* y, int rows, int k, int n) {
    convrot_product_tile<false>(codes, scales, qweight, wscale, y, rows, k, n, blockIdx.y, blockIdx.x);
}

extern "C" __global__ void __maxnreg__(124) convrot_product_n_fast_ldsm(
    const int8_t* codes, const float* scales, const int8_t* qweight, const float* wscale,
    __nv_bfloat16* y, int rows, int k, int n) {
    convrot_product_tile<true>(codes, scales, qweight, wscale, y, rows, k, n, blockIdx.y, blockIdx.x);
}
#else /* ff-sm80 */
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

#if !defined(__HIP__)
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

__device__ __forceinline__ uint4 convrot_load_u128(const void* p) {
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}
#endif

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

#if !defined(__HIP__)
// m16n8k32.row.col.s32.s8.s8. BM 128, BN 256, BK 64, stride 80, 2 stages, 512 threads.
// Dynamic smem 61440 N-fast, 49152 M-fast.
constexpr int CONVROT_BM = 128;
constexpr int CONVROT_BN = 256;
constexpr int CONVROT_BK = 64;
constexpr int CONVROT_STRIDE = 80;
template <bool SWIZZLED>
constexpr int CONVROT_PRODUCT_STRIDE = SWIZZLED ? CONVROT_BK : CONVROT_STRIDE;
template <bool SWIZZLED>
constexpr int CONVROT_STAGE = (CONVROT_BM + CONVROT_BN) * CONVROT_PRODUCT_STRIDE<SWIZZLED>;
static_assert(CONVROT_STAGE<false> * 2 == 61440 && CONVROT_STAGE<true> * 2 == 49152, "convrot product smem");

template <bool SWIZZLED>
__device__ __forceinline__ int convrot_segment_offset(int row, int column) {
    if constexpr (SWIZZLED) {
        return row * CONVROT_BK + (column ^ (((row >> 1) & 3) << 4));
    } else {
        return row * CONVROT_STRIDE + column;
    }
}

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
template <bool SWIZZLED>
__device__ __forceinline__ void convrot_stage(int8_t* smem, const int8_t* __restrict__ codes,
                                              const int8_t* __restrict__ qweight, int rows, int k, int n, int m0,
                                              int n0, int k0, int stage) {
    int8_t* a = smem + stage * CONVROT_STAGE<SWIZZLED>;
    int8_t* b = a + CONVROT_BM * CONVROT_PRODUCT_STRIDE<SWIZZLED>;
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
    convrot_cp16(a + convrot_segment_offset<SWIZZLED>(ar, aseg * 16), asrc, asz);
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
        convrot_cp16(b + convrot_segment_offset<SWIZZLED>(bc, bseg * 16), bsrc, bsz);
    }
}

// Lane groupID = lane>>2, thread = lane&3. A u32 at (group, thread*4), (group+8, thread*4),
// (group, thread*4+16), (group+8, thread*4+16). B u32 at column group, K thread*4 and +16.
template <bool MATRIX_LOAD, bool SWIZZLED>
__device__ __forceinline__ void convrot_mma_tile(const int8_t* stage, int (&acc)[2][8][4], int warp_m, int warp_n,
                                                 int group, int t4, int lane) {
    const int8_t* a = stage;
    const int8_t* b = stage + CONVROT_BM * CONVROT_PRODUCT_STRIDE<SWIZZLED>;
    const int row_base = warp_m * 32;
    const int col_base = warp_n * 64;
#pragma unroll
    for (int ki = 0; ki < 2; ++ki) {
        const int kbase = ki * 32 + t4;
        unsigned af[2][4];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            if constexpr (MATRIX_LOAD) {
                const int r = row_base + mi * 16 + (lane & 15);
                const int k_panel = ki * 32 + (lane >> 4) * 16;
                convrot_ldsm_x4(af[mi], a + convrot_segment_offset<SWIZZLED>(r, k_panel));
            } else {
                const int r = row_base + mi * 16 + group;
                af[mi][0] = convrot_lds(a + (r * CONVROT_STRIDE + kbase));
                af[mi][1] = convrot_lds(a + ((r + 8) * CONVROT_STRIDE + kbase));
                af[mi][2] = convrot_lds(a + (r * CONVROT_STRIDE + kbase + 16));
                af[mi][3] = convrot_lds(a + ((r + 8) * CONVROT_STRIDE + kbase + 16));
            }
        }
        unsigned bf[8][2];
        if constexpr (MATRIX_LOAD) {
#pragma unroll
            for (int ni = 0; ni < 8; ni += 2) {
                const int c = col_base + ni * 8 + (lane & 7) + (lane >> 4) * 8;
                const int k_panel = ki * 32 + ((lane >> 3) & 1) * 16;
                unsigned pair[4];
                convrot_ldsm_x4(pair, b + convrot_segment_offset<SWIZZLED>(c, k_panel));
                bf[ni][0] = pair[0];
                bf[ni][1] = pair[1];
                bf[ni + 1][0] = pair[2];
                bf[ni + 1][1] = pair[3];
            }
        } else {
#pragma unroll
            for (int ni = 0; ni < 8; ++ni) {
                const int c = col_base + ni * 8 + group;
                bf[ni][0] = convrot_lds(b + (c * CONVROT_STRIDE + kbase));
                bf[ni][1] = convrot_lds(b + (c * CONVROT_STRIDE + kbase + 16));
            }
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

__device__ __forceinline__ float convrot_ldg_f32(const float* p, bool live) {
    float v = 0.f;
    if (live) {
        asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(v) : "l"(p));
    }
    return v;
}

__device__ __forceinline__ void convrot_store_one(__nv_bfloat16* __restrict__ y, int rows, int n, int r, int c, int acc,
                                                  float sx, float sw) {
    if (r < rows && c < n) {
        const float v = __fmul_rn(__fmul_rn(static_cast<float>(acc), sx), sw);
        y[static_cast<long long>(r) * n + c] = __float2bfloat16(v);
    }
}

template <bool MATRIX_LOAD, bool SWIZZLED = false, bool ONE_BARRIER = false>
__device__ __forceinline__ void convrot_product_tile(
    const int8_t* __restrict__ codes, const float* __restrict__ scales,
    const int8_t* __restrict__ qweight, const float* __restrict__ wscale,
    __nv_bfloat16* __restrict__ y, int rows, int k, int n, int m_tile, int n_tile) {
    static_assert(!SWIZZLED || MATRIX_LOAD);
    if (rows <= 0 || n <= 0 || k <= 0 || (k & (CONVROT_BK - 1))) __trap();
    extern __shared__ __align__(16) int8_t smem[];
    const int m0 = m_tile * CONVROT_BM;
    const int n0 = n_tile * CONVROT_BN;
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
    convrot_stage<SWIZZLED>(smem, codes, qweight, rows, k, n, m0, n0, 0, 0);
    convrot_cp_commit();
    for (int kt = 0; kt < tiles; ++kt) {
        if constexpr (ONE_BARRIER) {
            convrot_cp_wait<0>();
            // The current slot is ready and every warp has finished reading the opposite slot.
            __syncthreads();
        }
        if (kt + 1 < tiles) {
            convrot_stage<SWIZZLED>(smem, codes, qweight, rows, k, n, m0, n0, (kt + 1) * CONVROT_BK, stage ^ 1);
            convrot_cp_commit();
            if constexpr (!ONE_BARRIER) convrot_cp_wait<1>();
        } else if constexpr (!ONE_BARRIER) {
            convrot_cp_wait<0>();
        }
        if constexpr (!ONE_BARRIER) __syncthreads();
        convrot_mma_tile<MATRIX_LOAD, SWIZZLED>(smem + stage * CONVROT_STAGE<SWIZZLED>, acc, warp_m, warp_n, group, t4, lane);
        if constexpr (!ONE_BARRIER) __syncthreads();
        stage ^= 1;
    }
    // 4 row scales and 16 column scales, one global load each.
    float sx[2][2];
    float sw[8][2];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        const int r0 = m0 + warp_m * 32 + mi * 16 + group;
        sx[mi][0] = convrot_ldg_f32(scales + r0, r0 < rows);
        sx[mi][1] = convrot_ldg_f32(scales + (r0 + 8), (r0 + 8) < rows);
    }
#pragma unroll
    for (int ni = 0; ni < 8; ++ni) {
        const int c0 = n0 + warp_n * 64 + ni * 8 + t2;
        sw[ni][0] = convrot_ldg_f32(wscale + c0, c0 < n);
        sw[ni][1] = convrot_ldg_f32(wscale + (c0 + 1), (c0 + 1) < n);
    }
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 8; ++ni) {
            const int r0 = m0 + warp_m * 32 + mi * 16 + group;
            const int c0 = n0 + warp_n * 64 + ni * 8 + t2;
            convrot_store_one(y, rows, n, r0, c0, acc[mi][ni][0], sx[mi][0], sw[ni][0]);
            convrot_store_one(y, rows, n, r0, c0 + 1, acc[mi][ni][1], sx[mi][0], sw[ni][1]);
            convrot_store_one(y, rows, n, r0 + 8, c0, acc[mi][ni][2], sx[mi][1], sw[ni][0]);
            convrot_store_one(y, rows, n, r0 + 8, c0 + 1, acc[mi][ni][3], sx[mi][1], sw[ni][1]);
        }
    }
}

extern "C" __global__ void __maxnreg__(124) convrot_product(
    const int8_t* codes, const float* scales, const int8_t* qweight, const float* wscale,
    __nv_bfloat16* y, int rows, int k, int n) {
    convrot_product_tile<false>(codes, scales, qweight, wscale, y, rows, k, n, blockIdx.y, blockIdx.x);
}

#if __CUDA_ARCH__ == 1200
extern "C" __global__ void __maxnreg__(124) convrot_product_m_fast(
    const int8_t* codes, const float* scales, const int8_t* qweight, const float* wscale,
    __nv_bfloat16* y, int rows, int k, int n) {
    const unsigned m_tile = blockIdx.x + blockIdx.z * gridDim.x;
    if (m_tile * CONVROT_BM >= static_cast<unsigned>(rows)) return;
    convrot_product_tile<true, true>(codes, scales, qweight, wscale, y, rows, k, n, m_tile, blockIdx.y);
}
#endif

extern "C" __global__ void __maxnreg__(124) convrot_product_n_fast_ldsm(
    const int8_t* codes, const float* scales, const int8_t* qweight, const float* wscale,
    __nv_bfloat16* y, int rows, int k, int n) {
    convrot_product_tile<true>(codes, scales, qweight, wscale, y, rows, k, n, blockIdx.y, blockIdx.x);
}

#if __CUDA_ARCH__ == 1200
extern "C" __global__ void __maxnreg__(124) convrot_product_m_fast_one_barrier(
    const int8_t* codes, const float* scales, const int8_t* qweight, const float* wscale,
    __nv_bfloat16* y, int rows, int k, int n) {
    const unsigned m_tile = blockIdx.x + blockIdx.z * gridDim.x;
    if (m_tile * CONVROT_BM >= static_cast<unsigned>(rows)) return;
    convrot_product_tile<true, true, true>(codes, scales, qweight, wscale, y, rows, k, n, m_tile, blockIdx.y);
}
#endif

extern "C" __global__ void __maxnreg__(124) convrot_product_n_fast_swizzled(
    const int8_t* codes, const float* scales, const int8_t* qweight, const float* wscale,
    __nv_bfloat16* y, int rows, int k, int n) {
    convrot_product_tile<true, true>(codes, scales, qweight, wscale, y, rows, k, n, blockIdx.y, blockIdx.x);
}

static_assert(CONVROT_STAGE<true> * 3 == 73728 && CONVROT_STAGE<true> * 4 == 98304, "convrot pipelined smem");

__device__ __forceinline__ void convrot_cp16_u(unsigned smem_dst, const void* src, int sz) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_dst), "l"(src), "r"(sz));
}

__device__ __forceinline__ void convrot_ldsm_x4_u(unsigned (&r)[4], unsigned smem_addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_addr));
}

template <bool M_STRIPED = false>
__device__ __forceinline__ void convrot_tile_origin(int& m0, int& n0) {
    if constexpr (M_STRIPED) {
        const int m_tile = static_cast<int>(blockIdx.x + blockIdx.z * gridDim.x);
        const int raw_n = static_cast<int>(blockIdx.y);
        const int n_tile = (blockIdx.z & 1u) ? (static_cast<int>(gridDim.y) - 1 - raw_n) : raw_n;
        m0 = m_tile * CONVROT_BM;
        n0 = n_tile * CONVROT_BN;
    } else {
        const int m_tiles = static_cast<int>(gridDim.y);
        const int n_tiles = static_cast<int>(gridDim.x);
        constexpr int GROUP_M = 8;
        const int m_group_id = static_cast<int>(blockIdx.y) / GROUP_M;
        const int m_group_start = m_group_id * GROUP_M;
        const int group_h = min(GROUP_M, m_tiles - m_group_start);
        const int local_id = (static_cast<int>(blockIdx.y) - m_group_start) * n_tiles + static_cast<int>(blockIdx.x);
        const int m_tile = m_group_start + (local_id % group_h);
        const int raw_n = local_id / group_h;
        const int n_tile = (m_group_id & 1) ? (n_tiles - 1 - raw_n) : raw_n;
        m0 = m_tile * CONVROT_BM;
        n0 = n_tile * CONVROT_BN;
    }
}

template <bool VEC32>
__device__ __forceinline__ void convrot_store_row_pair(__nv_bfloat16* __restrict__ y, int rows, int n, int r, int c0,
                                                       int a0, int a1, float sx, float sw0, float sw1) {
    if constexpr (VEC32) {
        if (r < rows && c0 < n) {
            const float v0 = __fmul_rn(__fmul_rn(static_cast<float>(a0), sx), sw0);
            const float v1 = __fmul_rn(__fmul_rn(static_cast<float>(a1), sx), sw1);
            *reinterpret_cast<__nv_bfloat162*>(y + static_cast<long long>(r) * n + c0) = __floats2bfloat162_rn(v0, v1);
        }
    } else {
        convrot_store_one(y, rows, n, r, c0, a0, sx, sw0);
        convrot_store_one(y, rows, n, r, c0 + 1, a1, sx, sw1);
    }
}

template <bool VEC32>
__device__ __forceinline__ void convrot_store_epilogue(__nv_bfloat16* __restrict__ y, const int (&acc)[2][8][4],
                                                       const float (&sx)[2][2], const float (&sw)[8][2], int rows,
                                                       int n, int m0, int n0, int warp_m, int warp_n, int group,
                                                       int t2) {
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 8; ++ni) {
            const int r0 = m0 + warp_m * 32 + mi * 16 + group;
            const int c0 = n0 + warp_n * 64 + ni * 8 + t2;
            convrot_store_row_pair<VEC32>(y, rows, n, r0, c0, acc[mi][ni][0], acc[mi][ni][1], sx[mi][0], sw[ni][0],
                                          sw[ni][1]);
            convrot_store_row_pair<VEC32>(y, rows, n, r0 + 8, c0, acc[mi][ni][2], acc[mi][ni][3], sx[mi][1], sw[ni][0],
                                          sw[ni][1]);
        }
    }
}

constexpr int CONVROT_OUT_STRIDE = 264;

__device__ __forceinline__ void convrot_pack_smem_mi(
    __nv_bfloat16* __restrict__ out_smem, const int (&acc_mi)[8][4], const float (&sx_mi)[2],
    const float (&sw)[8][2], int sr, int sc_base) {
#pragma unroll
    for (int ni = 0; ni < 8; ++ni) {
        const int sc = sc_base + ni * 8;
        const float p0 = __fmul_rn(__fmul_rn(static_cast<float>(acc_mi[ni][0]), sx_mi[0]), sw[ni][0]);
        const float p1 = __fmul_rn(__fmul_rn(static_cast<float>(acc_mi[ni][1]), sx_mi[0]), sw[ni][1]);
        *reinterpret_cast<__nv_bfloat162*>(out_smem + sr * CONVROT_OUT_STRIDE + sc) =
            __floats2bfloat162_rn(p0, p1);
        const float q0 = __fmul_rn(__fmul_rn(static_cast<float>(acc_mi[ni][2]), sx_mi[1]), sw[ni][0]);
        const float q1 = __fmul_rn(__fmul_rn(static_cast<float>(acc_mi[ni][3]), sx_mi[1]), sw[ni][1]);
        *reinterpret_cast<__nv_bfloat162*>(out_smem + (sr + 8) * CONVROT_OUT_STRIDE + sc) =
            __floats2bfloat162_rn(q0, q1);
    }
}

__device__ __forceinline__ void convrot_flush_smem_row(
    const __nv_bfloat16* __restrict__ out_smem, __nv_bfloat16* __restrict__ y, int sr, int gr,
    int rows, int n, int col_local, int gc, bool aligned128) {
    if (gr >= rows) return;
    const uint4 v = *reinterpret_cast<const uint4*>(out_smem + sr * CONVROT_OUT_STRIDE + col_local);
    if (aligned128 && gc < n) {
        *reinterpret_cast<uint4*>(y + static_cast<long long>(gr) * n + gc) = v;
    } else {
        const __nv_bfloat16* vp = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            if (gc + e < n) y[static_cast<long long>(gr) * n + gc + e] = vp[e];
        }
    }
}

template <bool FULL_SMEM>
__device__ __forceinline__ void convrot_store_epilogue_smem128(
    int8_t* __restrict__ smem, __nv_bfloat16* __restrict__ y, const int (&acc)[2][8][4],
    const float (&sx)[2][2], const float (&sw)[8][2], int rows, int n, int m0, int n0,
    int warp, int warp_m, int warp_n, int group, int t2, int lane) {
    static_assert((FULL_SMEM ? 128 : 64) * CONVROT_OUT_STRIDE * sizeof(__nv_bfloat16) <= (FULL_SMEM ? 73728 : 49152),
                  "convrot epilogue smem extent");
    __nv_bfloat16* out_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    const int sc_base = warp_n * 64 + t2;
    const int col_local = lane << 3;
    const int gc = n0 + col_local;
    const bool aligned128 = ((n & 7) == 0) && ((reinterpret_cast<unsigned long long>(y) & 15ULL) == 0ULL);

    if constexpr (FULL_SMEM) {
        __syncthreads();
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            convrot_pack_smem_mi(out_smem, acc[mi], sx[mi], sw, warp_m * 32 + mi * 16 + group, sc_base);
        }
        __syncthreads();
#pragma unroll
        for (int step = 0; step < 8; ++step) {
            const int sr = warp + step * 16;
            convrot_flush_smem_row(out_smem, y, sr, m0 + sr, rows, n, col_local, gc, aligned128);
        }
    } else {
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            __syncthreads();
            convrot_pack_smem_mi(out_smem, acc[mi], sx[mi], sw, warp_m * 16 + group, sc_base);
            __syncthreads();
#pragma unroll
            for (int step = 0; step < 4; ++step) {
                const int sr = warp + step * 16;
                const int gr = m0 + ((sr >> 4) << 5) + (mi << 4) + (sr & 15);
                convrot_flush_smem_row(out_smem, y, sr, gr, rows, n, col_local, gc, aligned128);
            }
        }
    }
}

template <int STAGES, bool M_STRIPED = false>
__device__ __forceinline__ void convrot_product_tile_pipelined(
    const int8_t* __restrict__ codes, const float* __restrict__ scales,
    const int8_t* __restrict__ qweight, const float* __restrict__ wscale,
    __nv_bfloat16* __restrict__ y, int rows, int k, int n) {
    static_assert(STAGES >= 2, "convrot pipelined stages");
    if (rows <= 0 || n <= 0 || k <= 0 || (k & (CONVROT_BK - 1))) __trap();
    extern __shared__ __align__(16) int8_t smem[];
    constexpr int STAGE_BYTES = CONVROT_STAGE<true>;
    constexpr int TOTAL_BYTES = STAGES * STAGE_BYTES;

    const int t = static_cast<int>(threadIdx.x);
    const int lane = t & 31;
    const int warp = t >> 5;
    const int warp_m = warp >> 2;
    const int warp_n = warp & 3;
    const int group = lane >> 2;
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

    int init_m0 = 0;
    int init_n0 = 0;
    convrot_tile_origin<M_STRIPED>(init_m0, init_n0);
    if constexpr (M_STRIPED) {
        if (init_m0 >= rows) return;
    }

    const int ar = t >> 2;
    const int aseg = (t & 3) * 16;
    const bool a_live = (init_m0 + ar) < rows;
    const int asz = a_live ? 16 : 0;
    const int8_t* a_g = a_live ? (codes + (static_cast<long long>(init_m0 + ar) * k + aseg)) : codes;
    const int a_step = asz << 2;

    const int bc0 = t >> 2;
    const int bseg0 = (t & 3) * 16;
    const bool b0_live = (init_n0 + bc0) < n;
    const int b0sz = b0_live ? 16 : 0;
    const int8_t* b0_g = b0_live ? (qweight + (static_cast<long long>(init_n0 + bc0) * k + bseg0)) : qweight;
    const int b0_step = b0sz << 2;

    const int linear1 = t + 512;
    const int bc1 = linear1 >> 2;
    const int bseg1 = (linear1 & 3) * 16;
    const bool b1_live = (init_n0 + bc1) < n;
    const int b1sz = b1_live ? 16 : 0;
    const int8_t* b1_g = b1_live ? (qweight + (static_cast<long long>(init_n0 + bc1) * k + bseg1)) : qweight;
    const int b1_step = b1sz << 2;

    const unsigned smem_base = convrot_smem(smem);
    const unsigned aw = smem_base + convrot_segment_offset<true>(ar, aseg);
    const unsigned b0w = smem_base + CONVROT_BM * CONVROT_BK + convrot_segment_offset<true>(bc0, bseg0);

    const int r_ldsm = warp_m * 32 + (lane & 15);
    const int ka_ldsm = (lane >> 4) * 16;
    const int ra_off0 = convrot_segment_offset<true>(r_ldsm, ka_ldsm);

    const int c_ldsm = warp_n * 64 + (lane & 7) + (lane >> 4) * 8;
    const int kb_ldsm = ((lane >> 3) & 1) * 16;
    const int rb_off0 = CONVROT_BM * CONVROT_BK + convrot_segment_offset<true>(c_ldsm, kb_ldsm);

    const int tiles = k >> 6;
    unsigned w_off = 0;
#pragma unroll
    for (int p = 0; p < STAGES - 1; ++p) {
        if (p < tiles) {
            convrot_cp16_u(aw + w_off, a_g, asz);
            convrot_cp16_u(b0w + w_off, b0_g, b0sz);
            convrot_cp16_u(b0w + 8192 + w_off, b1_g, b1sz);
            a_g += a_step;
            b0_g += b0_step;
            b1_g += b1_step;
        }
        convrot_cp_commit();
        w_off += STAGE_BYTES;
    }
    convrot_cp_wait<STAGES - 2>();
    __syncthreads();

    int issue_rem = tiles - (STAGES - 1);
    unsigned r_off = 0;
#pragma unroll 1
    for (int kt = 0; kt < tiles; ++kt) {
        const unsigned a_rd0 = r_off + ra_off0;
        const unsigned a_rd1 = r_off + (ra_off0 ^ 32);
        const unsigned b_rd0 = r_off + rb_off0;
        const unsigned b_rd1 = r_off + (rb_off0 ^ 32);
        const bool do_issue = issue_rem > 0;

        unsigned af[2][2][4];
        unsigned bf[8][2];

        convrot_ldsm_x4_u(af[0][0], a_rd0);
        convrot_ldsm_x4_u(af[0][1], a_rd0 + 1024);
        {
            unsigned pair[4];
            convrot_ldsm_x4_u(pair, b_rd0);
            bf[0][0] = pair[0]; bf[0][1] = pair[1]; bf[1][0] = pair[2]; bf[1][1] = pair[3];
            convrot_ldsm_x4_u(pair, b_rd0 + 1024);
            bf[2][0] = pair[0]; bf[2][1] = pair[1]; bf[3][0] = pair[2]; bf[3][1] = pair[3];
            convrot_ldsm_x4_u(pair, b_rd0 + 2048);
            bf[4][0] = pair[0]; bf[4][1] = pair[1]; bf[5][0] = pair[2]; bf[5][1] = pair[3];
            convrot_ldsm_x4_u(pair, b_rd0 + 3072);
            bf[6][0] = pair[0]; bf[6][1] = pair[1]; bf[7][0] = pair[2]; bf[7][1] = pair[3];
        }

#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                convrot_mma_s8(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                               af[0][mi][0], af[0][mi][1], af[0][mi][2], af[0][mi][3], bf[ni][0], bf[ni][1]);
            }
        }
        if (do_issue) {
            convrot_cp16_u(aw + w_off, a_g, asz);
            convrot_cp16_u(b0w + w_off, b0_g, b0sz);
            convrot_cp16_u(b0w + 8192 + w_off, b1_g, b1sz);
            a_g += a_step;
            b0_g += b0_step;
            b1_g += b1_step;
        }
        convrot_ldsm_x4_u(af[1][0], a_rd1);

#pragma unroll
        for (int ni = 2; ni < 4; ++ni) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                convrot_mma_s8(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                               af[0][mi][0], af[0][mi][1], af[0][mi][2], af[0][mi][3], bf[ni][0], bf[ni][1]);
            }
        }
        convrot_ldsm_x4_u(af[1][1], a_rd1 + 1024);
        {
            unsigned pair[4];
            convrot_ldsm_x4_u(pair, b_rd1);
            bf[0][0] = pair[0]; bf[0][1] = pair[1]; bf[1][0] = pair[2]; bf[1][1] = pair[3];
        }

#pragma unroll
        for (int ni = 4; ni < 6; ++ni) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                convrot_mma_s8(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                               af[0][mi][0], af[0][mi][1], af[0][mi][2], af[0][mi][3], bf[ni][0], bf[ni][1]);
            }
        }
        {
            unsigned pair[4];
            convrot_ldsm_x4_u(pair, b_rd1 + 1024);
            bf[2][0] = pair[0]; bf[2][1] = pair[1]; bf[3][0] = pair[2]; bf[3][1] = pair[3];
        }

#pragma unroll
        for (int ni = 6; ni < 8; ++ni) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                convrot_mma_s8(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                               af[0][mi][0], af[0][mi][1], af[0][mi][2], af[0][mi][3], bf[ni][0], bf[ni][1]);
            }
        }
        {
            unsigned pair[4];
            convrot_ldsm_x4_u(pair, b_rd1 + 2048);
            bf[4][0] = pair[0]; bf[4][1] = pair[1]; bf[5][0] = pair[2]; bf[5][1] = pair[3];
        }

#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                convrot_mma_s8(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                               af[1][mi][0], af[1][mi][1], af[1][mi][2], af[1][mi][3], bf[ni][0], bf[ni][1]);
            }
        }
        {
            unsigned pair[4];
            convrot_ldsm_x4_u(pair, b_rd1 + 3072);
            bf[6][0] = pair[0]; bf[6][1] = pair[1]; bf[7][0] = pair[2]; bf[7][1] = pair[3];
        }

#pragma unroll
        for (int ni = 2; ni < 8; ++ni) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                convrot_mma_s8(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                               af[1][mi][0], af[1][mi][1], af[1][mi][2], af[1][mi][3], bf[ni][0], bf[ni][1]);
            }
        }

        if (kt + 1 < tiles) {
            convrot_cp_commit();
            convrot_cp_wait<STAGES - 2>();
            __syncthreads();
            --issue_rem;
            r_off += STAGE_BYTES;
            if (r_off == TOTAL_BYTES) r_off = 0;
            w_off += STAGE_BYTES;
            if (w_off == TOTAL_BYTES) w_off = 0;
        }
    }
    convrot_cp_wait<0>();

    int m0 = 0;
    int n0 = 0;
    convrot_tile_origin<M_STRIPED>(m0, n0);

    float sx[2][2];
    float sw[8][2];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        const int r0 = m0 + warp_m * 32 + mi * 16 + group;
        sx[mi][0] = convrot_ldg_f32(scales + r0, r0 < rows);
        sx[mi][1] = convrot_ldg_f32(scales + (r0 + 8), (r0 + 8) < rows);
    }
#pragma unroll
    for (int ni = 0; ni < 8; ++ni) {
        const int c0 = n0 + warp_n * 64 + ni * 8 + t2;
        sw[ni][0] = convrot_ldg_f32(wscale + c0, c0 < n);
        sw[ni][1] = convrot_ldg_f32(wscale + (c0 + 1), (c0 + 1) < n);
    }
    if constexpr (M_STRIPED) {
        convrot_store_epilogue_smem128<TOTAL_BYTES >= 128 * CONVROT_OUT_STRIDE * sizeof(__nv_bfloat16)>(
            smem, y, acc, sx, sw, rows, n, m0, n0, warp, warp_m, warp_n, group, t2, lane);
    } else if ((n & 1) == 0 && (reinterpret_cast<unsigned long long>(y) & 3ULL) == 0ULL) {
        convrot_store_epilogue<true>(y, acc, sx, sw, rows, n, m0, n0, warp_m, warp_n, group, t2);
    } else {
        convrot_store_epilogue<false>(y, acc, sx, sw, rows, n, m0, n0, warp_m, warp_n, group, t2);
    }
}

extern "C" __global__ void __maxnreg__(124) convrot_product_n_fast_swizzled_s3(
    const int8_t* codes, const float* scales, const int8_t* qweight, const float* wscale,
    __nv_bfloat16* y, int rows, int k, int n) {
    convrot_product_tile_pipelined<3>(codes, scales, qweight, wscale, y, rows, k, n);
}

extern "C" __global__ void __maxnreg__(124) convrot_product_n_fast_swizzled_s4(
    const int8_t* codes, const float* scales, const int8_t* qweight, const float* wscale,
    __nv_bfloat16* y, int rows, int k, int n) {
    convrot_product_tile_pipelined<4>(codes, scales, qweight, wscale, y, rows, k, n);
}

#if __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 1200
extern "C" __global__ void __maxnreg__(124) convrot_product_m_fast_s2(
    const int8_t* codes, const float* scales, const int8_t* qweight, const float* wscale,
    __nv_bfloat16* y, int rows, int k, int n) {
    convrot_product_tile_pipelined<2, true>(codes, scales, qweight, wscale, y, rows, k, n);
}

extern "C" __global__ void __maxnreg__(124) convrot_product_m_fast_s3(
    const int8_t* codes, const float* scales, const int8_t* qweight, const float* wscale,
    __nv_bfloat16* y, int rows, int k, int n) {
    convrot_product_tile_pipelined<3, true>(codes, scales, qweight, wscale, y, rows, k, n);
}

extern "C" __global__ void __maxnreg__(124) convrot_product_m_fast_s4(
    const int8_t* codes, const float* scales, const int8_t* qweight, const float* wscale,
    __nv_bfloat16* y, int rows, int k, int n) {
    convrot_product_tile_pipelined<4, true>(codes, scales, qweight, wscale, y, rows, k, n);
}
#endif
#else
// gfx11 wave32 v_wmma_i32_16x16x16_iu8. BK=64, 2 swizzled LDS stages.
// 128x128x64: 256 threads, 32768 B LDS (2 WGs/WGP). 128x256x64: 512 threads, 49152 B LDS (1 WG/WGP).
__device__ __forceinline__ int convrot_hip_swiz(int row, int col_bytes) {
    return row * 64 + (col_bytes ^ (((row >> 1) & 3) << 4));
}

template <int BM, int BN, int THREADS>
__device__ __forceinline__ void convrot_product_tile_hip(
    const int8_t* __restrict__ codes, const float* __restrict__ scales,
    const int8_t* __restrict__ qweight, const float* __restrict__ wscale,
    __nv_bfloat16* __restrict__ y, int rows, int k, int n, int m_tile, int n_tile) {
    using I32Vec4 = int __attribute__((ext_vector_type(4)));
    using I32Vec8 = int __attribute__((ext_vector_type(8)));
    static_assert(BM == 128 && (BN == 128 || BN == 256) && (THREADS == 256 || THREADS == 512));
    constexpr int BK = 64;
    constexpr int STAGE_A = BM * BK;
    constexpr int STAGE_B = BN * BK;
    constexpr int STAGE_BYTES = STAGE_A + STAGE_B;
    constexpr int A_LOADS = (BM * (BK / 16)) / THREADS;
    constexpr int B_LOADS = (BN * (BK / 16)) / THREADS;
    constexpr int WARPS_N = BN / 64;
    if (rows <= 0 || n <= 0 || k <= 0 || (k & (BK - 1))) __trap();
    extern __shared__ __align__(16) int8_t smem[];
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_m = warp / WARPS_N;
    const int warp_n = warp % WARPS_N;
    const int m0 = m_tile * BM;
    const int n0 = n_tile * BN;
    const int row_base = warp_m * 32;
    const int col_base = warp_n * 64;
    const int lane_r = lane & 15;
    const int half = lane >> 4;
    I32Vec8 acc[2][4];
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            acc[mi][ni] = {0, 0, 0, 0, 0, 0, 0, 0};
        }
    }
    uint4 reg_a[A_LOADS];
    uint4 reg_b[B_LOADS];
    auto prefetch_global = [&](int k0) {
        #pragma unroll
        for (int i = 0; i < A_LOADS; ++i) {
            const int linear = tid + i * THREADS;
            const int ar = linear >> 2;
            const int aseg = linear & 3;
            const int gr = m0 + ar;
            reg_a[i] = (gr < rows)
                           ? *reinterpret_cast<const uint4*>(codes + static_cast<long long>(gr) * k + k0 + (aseg << 4))
                           : make_uint4(0u, 0u, 0u, 0u);
        }
        #pragma unroll
        for (int i = 0; i < B_LOADS; ++i) {
            const int linear = tid + i * THREADS;
            const int bc = linear >> 2;
            const int bseg = linear & 3;
            const int gc = n0 + bc;
            reg_b[i] =
                (gc < n)
                    ? *reinterpret_cast<const uint4*>(qweight + static_cast<long long>(gc) * k + k0 + (bseg << 4))
                    : make_uint4(0u, 0u, 0u, 0u);
        }
    };
    auto store_lds = [&](int stage_idx) {
        int8_t* sa = smem + stage_idx * STAGE_BYTES;
        int8_t* sb = sa + STAGE_A;
        #pragma unroll
        for (int i = 0; i < A_LOADS; ++i) {
            const int linear = tid + i * THREADS;
            const int ar = linear >> 2;
            const int aseg = linear & 3;
            *reinterpret_cast<uint4*>(sa + convrot_hip_swiz(ar, aseg << 4)) = reg_a[i];
        }
        #pragma unroll
        for (int i = 0; i < B_LOADS; ++i) {
            const int linear = tid + i * THREADS;
            const int bc = linear >> 2;
            const int bseg = linear & 3;
            *reinterpret_cast<uint4*>(sb + convrot_hip_swiz(bc, bseg << 4)) = reg_b[i];
        }
    };
    const int tiles = k >> 6;
    int stage = 0;
    prefetch_global(0);
    store_lds(0);
    __syncthreads();
    for (int kt = 0; kt < tiles; ++kt) {
        if (kt + 1 < tiles) {
            prefetch_global((kt + 1) * BK);
        }
        const int8_t* sa = smem + stage * STAGE_BYTES;
        const int8_t* sb = sa + STAGE_A;
        #pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
            const int k_off = ki << 4;
            I32Vec4 af[2];
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const uint4 v =
                    *reinterpret_cast<const uint4*>(sa + convrot_hip_swiz(row_base + mi * 16 + lane_r, k_off));
                __builtin_memcpy(&af[mi], &v, sizeof(v));
            }
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const uint4 v =
                    *reinterpret_cast<const uint4*>(sb + convrot_hip_swiz(col_base + ni * 16 + lane_r, k_off));
                I32Vec4 bf;
                __builtin_memcpy(&bf, &v, sizeof(v));
                acc[0][ni] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af[0], true, bf, acc[0][ni], false);
                acc[1][ni] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af[1], true, bf, acc[1][ni], false);
            }
        }
        if (kt + 1 < tiles) {
            store_lds(stage ^ 1);
            __syncthreads();
            stage ^= 1;
        }
    }
    float sw[4];
    #pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
        const int c = n0 + col_base + ni * 16 + lane_r;
        sw[ni] = (c < n) ? wscale[c] : 0.0f;
    }
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        #pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int r = m0 + row_base + mi * 16 + 2 * g + half;
            if (r < rows) {
                const float sx = scales[r];
                #pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    const int c = n0 + col_base + ni * 16 + lane_r;
                    if (c < n) {
                        const float v = __fmul_rn(__fmul_rn(static_cast<float>(acc[mi][ni][g]), sx), sw[ni]);
                        y[static_cast<long long>(r) * n + c] = __float2bfloat16(v);
                    }
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 2) void convrot_product(
    const int8_t* codes, const float* scales, const int8_t* qweight, const float* wscale,
    __nv_bfloat16* y, int rows, int k, int n) {
    convrot_product_tile_hip<128, 128, 256>(codes, scales, qweight, wscale, y, rows, k, n, static_cast<int>(blockIdx.y),
                                            static_cast<int>(blockIdx.x));
}

extern "C" __global__ __launch_bounds__(256, 2) void convrot_product_m_fast(
    const int8_t* codes, const float* scales, const int8_t* qweight, const float* wscale,
    __nv_bfloat16* y, int rows, int k, int n) {
    if (rows <= 0) __trap();
    const unsigned m_tile = blockIdx.x + blockIdx.z * gridDim.x;
    if (m_tile * 128u >= static_cast<unsigned>(rows)) return;
    convrot_product_tile_hip<128, 128, 256>(codes, scales, qweight, wscale, y, rows, k, n, static_cast<int>(m_tile),
                                            static_cast<int>(blockIdx.y));
}

extern "C" __global__ __launch_bounds__(512, 1) void convrot_product_n_fast_ldsm(
    const int8_t* codes, const float* scales, const int8_t* qweight, const float* wscale,
    __nv_bfloat16* y, int rows, int k, int n) {
    convrot_product_tile_hip<128, 256, 512>(codes, scales, qweight, wscale, y, rows, k, n, static_cast<int>(blockIdx.y),
                                            static_cast<int>(blockIdx.x));
}
#endif
#endif /* ff-sm80 */

// Fused 1-pass Radix-4 regular Hadamard + per-row INT8 quantize (CUDA sm75+ and HIP wave32).
// Grid: (rows, 1, 1). Each warp processes 256-element chunks (8 consecutive elements per lane).
// H4/2 cancel-free butterfly: a0 = x0+x1, d0 = x0-x1, a1 = x2+x3, d1 = x2-x3; y = 0.5 * (a ± d_other) in f32 RN.
__device__ __forceinline__ uint4 convrot_fwht_load_u128(const void* p) {
#if defined(__HIP__)
    return *reinterpret_cast<const uint4*>(p);
#else
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
#endif
}

template <bool SILU>
__device__ __forceinline__ void convrot_fwht_load_row8(const __nv_bfloat16* __restrict__ x_row, int k, int col,
                                                       float (&v)[8]) {
    if (col < k) {
        const uint4 lower_u = convrot_fwht_load_u128(x_row + col);
        const auto* lower = reinterpret_cast<const __nv_bfloat16*>(&lower_u);
        if constexpr (SILU) {
            const uint4 upper_u = convrot_fwht_load_u128(x_row + k + col);
            const auto* up = reinterpret_cast<const __nv_bfloat16*>(&upper_u);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float u = __bfloat162float(up[j]);
                const __nv_bfloat16 gated = __hmul(__float2bfloat16(u / (1.0f + expf(-u))), lower[j]);
                v[j] = __bfloat162float(gated);
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                v[j] = __bfloat162float(lower[j]);
            }
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[j] = 0.0f;
        }
    }
}

template <bool PROBE>
__device__ __forceinline__ void convrot_fwht_warp(float (&v)[8], int stages, int lane, int mutant) {
    const int eff_stages = (PROBE && mutant == 1) ? (stages > 0 ? stages - 1 : 0) : stages;
    if (eff_stages >= 1) {
        #pragma unroll
        for (int base = 0; base < 8; base += 4) {
            const float x0 = v[base + 0];
            const float x1 = v[base + 1];
            const float x2 = v[base + 2];
            const float x3 = v[base + 3];
            const float a0 = __fadd_rn(x0, x1);
            const float d0 = __fsub_rn(x0, x1);
            const float a1 = __fadd_rn(x2, x3);
            const float d1 = __fsub_rn(x2, x3);
            v[base + 0] = __fadd_rn(a0, (PROBE && mutant == 2 && base == 0) ? a1 : d1);
            v[base + 1] = __fsub_rn(a0, d1);
            v[base + 2] = __fadd_rn(a1, d0);
            v[base + 3] = __fsub_rn(a1, d0);
        }
    }
    if (eff_stages >= 2) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float lo = v[j];
            const float hi = v[j + 4];
            const float a = __fadd_rn(lo, hi);
            const float d = __fsub_rn(lo, hi);
            const float d_other = __shfl_xor_sync(FF_WARP_MASK, d, 1);
            v[j] = __fadd_rn(a, d_other);
            v[j + 4] = __fsub_rn(a, d_other);
        }
    }
    if (eff_stages >= 3) {
        const bool odd3 = (lane & 2) != 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float x = v[j];
            const float p = __shfl_xor_sync(FF_WARP_MASK, x, 2);
            const float a = __fadd_rn(x, p);
            const float d = odd3 ? __fsub_rn(p, x) : __fsub_rn(x, p);
            const float d_other = __shfl_xor_sync(FF_WARP_MASK, d, 4);
            v[j] = odd3 ? __fsub_rn(a, d_other) : __fadd_rn(a, d_other);
        }
    }
    if (eff_stages >= 4) {
        const bool odd4 = (lane & 8) != 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float x = v[j];
            const float p = __shfl_xor_sync(FF_WARP_MASK, x, 8);
            const float a = __fadd_rn(x, p);
            const float d = odd4 ? __fsub_rn(p, x) : __fsub_rn(x, p);
            const float d_other = __shfl_xor_sync(FF_WARP_MASK, d, 16);
            v[j] = odd4 ? __fsub_rn(a, d_other) : __fadd_rn(a, d_other);
        }
    }
    const float norm = (stages == 4) ? 0.0625f : (stages == 3 ? 0.125f : (stages == 2 ? 0.25f : 0.5f));
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = __fmul_rn(v[j], norm);
    }
}

__device__ __forceinline__ uint2 convrot_quantize_pack8(const float (&v)[8], float inv) {
    unsigned lo = 0u;
    unsigned hi = 0u;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float q0 = rintf(__fmul_rn(v[j], inv));
        q0 = fminf(fmaxf(q0, -127.0f), 127.0f);
        const unsigned b0 = static_cast<unsigned>(static_cast<unsigned char>(static_cast<int8_t>(static_cast<int>(q0))));
        lo |= b0 << (j * 8);

        float q1 = rintf(__fmul_rn(v[j + 4], inv));
        q1 = fminf(fmaxf(q1, -127.0f), 127.0f);
        const unsigned b1 = static_cast<unsigned>(static_cast<unsigned char>(static_cast<int8_t>(static_cast<int>(q1))));
        hi |= b1 << (j * 8);
    }
    return make_uint2(lo, hi);
}

template <bool SILU, int MAX_CHUNKS, bool PROBE>
__device__ __forceinline__ void convrot_fwht_quantize_reg(
    const __nv_bfloat16* __restrict__ x, float* __restrict__ rotated, int8_t* __restrict__ codes,
    float* __restrict__ scales, int rows, int k, int group, int mutant) {
    const unsigned ugroup = static_cast<unsigned>(group);
    const int threads = static_cast<int>(blockDim.x);
    const int warps = threads >> 5;
    if (rows <= 0 || k <= 0 || ugroup < 4u || ugroup > 256u || (ugroup & (ugroup - 1u)) != 0u ||
        ((__ffs(static_cast<int>(ugroup)) - 1) & 1) != 0 ||
        (static_cast<unsigned>(k) & (ugroup - 1u)) != 0u || (k & 7) != 0 ||
        (threads & 31) != 0 || threads <= 0 || threads > 1024 ||
        ((k + 255) >> 8) > MAX_CHUNKS * warps ||
        ((reinterpret_cast<unsigned long long>(x) & 15ull) != 0ull) ||
        ((reinterpret_cast<unsigned long long>(codes) & 7ull) != 0ull)) {
        __trap();
    }
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    const int stages = (__ffs(static_cast<int>(ugroup)) - 1) >> 1;
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const __nv_bfloat16* x_row = x + static_cast<long long>(row) * (SILU ? 2LL * k : k);
    float out[MAX_CHUNKS][8];
    float peak = 0.0f;
    #pragma unroll
    for (int c = 0; c < MAX_CHUNKS; ++c) {
        const int chunk = wid + c * warps;
        if ((chunk << 8) < k) {
            const int col = (chunk << 8) + (lane << 3);
            convrot_fwht_load_row8<SILU>(x_row, k, col, out[c]);
            convrot_fwht_warp<PROBE>(out[c], stages, lane, mutant);
            if (col < k) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    peak = fmaxf(peak, fabsf(out[c][j]));
                }
            }
        }
    }
    __shared__ float warp_peaks[32];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        peak = fmaxf(peak, __shfl_xor_sync(FF_WARP_MASK, peak, off));
    }
    if (lane == 0) warp_peaks[wid] = peak;
    __syncthreads();
    float w = (lane < warps) ? warp_peaks[lane] : 0.0f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        w = fmaxf(w, __shfl_xor_sync(FF_WARP_MASK, w, off));
    }
    const float peak_clamped = fmaxf(w, 1e-12f);
    if (tid == 0) scales[row] = __fdiv_rn(peak_clamped, 127.0f);
    const float inv = __fdiv_rn(127.0f, peak_clamped);
    int8_t* dst = codes + static_cast<long long>(row) * k;
    float* rdst = PROBE ? (rotated + static_cast<long long>(row) * k) : nullptr;
    #pragma unroll
    for (int c = 0; c < MAX_CHUNKS; ++c) {
        const int chunk = wid + c * warps;
        const int col = (chunk << 8) + (lane << 3);
        if (col < k) {
            if constexpr (PROBE) {
                *reinterpret_cast<float4*>(rdst + col) =
                    make_float4(out[c][0], out[c][1], out[c][2], out[c][3]);
                *reinterpret_cast<float4*>(rdst + col + 4) =
                    make_float4(out[c][4], out[c][5], out[c][6], out[c][7]);
            }
            *reinterpret_cast<uint2*>(dst + col) = convrot_quantize_pack8(out[c], inv);
        }
    }
}

template <bool SILU, bool PROBE>
__device__ __forceinline__ void convrot_fwht_quantize_smem_impl(
    const __nv_bfloat16* __restrict__ x, float* __restrict__ rotated, int8_t* __restrict__ codes,
    float* __restrict__ scales, int rows, int k, int group, int mutant) {
    const unsigned ugroup = static_cast<unsigned>(group);
    const int threads = static_cast<int>(blockDim.x);
    const int warps = threads >> 5;
    if (rows <= 0 || k <= 0 || ugroup < 4u || (ugroup & (ugroup - 1u)) != 0u ||
        ((__ffs(static_cast<int>(ugroup)) - 1) & 1) != 0 ||
        (static_cast<unsigned>(k) & (ugroup - 1u)) != 0u || (k & 7) != 0 ||
        (threads & 31) != 0 || threads <= 0 || threads > 1024 ||
        ((reinterpret_cast<unsigned long long>(x) & 15ull) != 0ull) ||
        ((reinterpret_cast<unsigned long long>(codes) & 7ull) != 0ull)) {
        __trap();
    }
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    extern __shared__ __align__(16) float fwht_smem[];
    __shared__ float warp_peaks[32];
    const int stages = (__ffs(static_cast<int>(ugroup)) - 1) >> 1;
    const int in_warp_stages = stages < 4 ? stages : 4;
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int chunks = (k + 255) >> 8;
    const __nv_bfloat16* x_row = x + static_cast<long long>(row) * (SILU ? 2LL * k : k);
    float peak = 0.0f;
    for (int chunk = wid; chunk < chunks; chunk += warps) {
        const int col = (chunk << 8) + (lane << 3);
        float v[8];
        convrot_fwht_load_row8<SILU>(x_row, k, col, v);
        convrot_fwht_warp<PROBE>(v, in_warp_stages, lane, stages > 4 ? 0 : mutant);
        if (col < k) {
            *reinterpret_cast<float4*>(fwht_smem + col) = make_float4(v[0], v[1], v[2], v[3]);
            *reinterpret_cast<float4*>(fwht_smem + col + 4) = make_float4(v[4], v[5], v[6], v[7]);
            if (stages <= 4) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    peak = fmaxf(peak, fabsf(v[j]));
                }
            }
        }
    }
    if (stages > 4) {
        __syncthreads();
        const int eff_stages = (PROBE && mutant == 1) ? (stages - 1) : stages;
        const int total_bfly = k >> 2;
        for (int s = 4; s < eff_stages; ++s) {
            const int shift = s << 1;
            const int stride = 1 << shift;
            const int mask = stride - 1;
            for (int b = tid; b < total_bfly; b += threads) {
                const int blk = b >> shift;
                const int off = b & mask;
                const int idx = (blk << (shift + 2)) + off;
                const float x0 = fwht_smem[idx];
                const float x1 = fwht_smem[idx + stride];
                const float x2 = fwht_smem[idx + (stride << 1)];
                const float x3 = fwht_smem[idx + stride * 3];
                const float a0 = __fmul_rn(0.5f, __fadd_rn(x0, x1));
                const float d0 = __fmul_rn(0.5f, __fsub_rn(x0, x1));
                const float a1 = __fmul_rn(0.5f, __fadd_rn(x2, x3));
                const float d1 = __fmul_rn(0.5f, __fsub_rn(x2, x3));
                const float y0 = __fadd_rn(a0, (PROBE && mutant == 2 && s == 4 && idx == 0) ? a1 : d1);
                const float y1 = __fsub_rn(a0, d1);
                const float y2 = __fadd_rn(a1, d0);
                const float y3 = __fsub_rn(a1, d0);
                fwht_smem[idx] = y0;
                fwht_smem[idx + stride] = y1;
                fwht_smem[idx + (stride << 1)] = y2;
                fwht_smem[idx + stride * 3] = y3;
                if (s + 1 == eff_stages) {
                    peak = fmaxf(peak, fmaxf(fmaxf(fabsf(y0), fabsf(y1)), fmaxf(fabsf(y2), fabsf(y3))));
                }
            }
            __syncthreads();
        }
        if (PROBE && eff_stages == 4) {
            for (int col = tid; col < k; col += threads) {
                peak = fmaxf(peak, fabsf(fwht_smem[col]));
            }
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        peak = fmaxf(peak, __shfl_xor_sync(FF_WARP_MASK, peak, off));
    }
    if (lane == 0) warp_peaks[wid] = peak;
    __syncthreads();
    float w = (lane < warps) ? warp_peaks[lane] : 0.0f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        w = fmaxf(w, __shfl_xor_sync(FF_WARP_MASK, w, off));
    }
    const float peak_clamped = fmaxf(w, 1e-12f);
    if (tid == 0) scales[row] = __fdiv_rn(peak_clamped, 127.0f);
    const float inv = __fdiv_rn(127.0f, peak_clamped);
    int8_t* dst = codes + static_cast<long long>(row) * k;
    float* rdst = PROBE ? (rotated + static_cast<long long>(row) * k) : nullptr;
    for (int col = tid << 3; col < k; col += threads << 3) {
        const float4 a = *reinterpret_cast<const float4*>(fwht_smem + col);
        const float4 b = *reinterpret_cast<const float4*>(fwht_smem + col + 4);
        if constexpr (PROBE) {
            *reinterpret_cast<float4*>(rdst + col) = a;
            *reinterpret_cast<float4*>(rdst + col + 4) = b;
        }
        const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        *reinterpret_cast<uint2*>(dst + col) = convrot_quantize_pack8(v, inv);
    }
}

#if defined(__HIP__)
#define CONVROT_FWHT_BOUNDS __launch_bounds__(1024, 1)
#else
#define CONVROT_FWHT_BOUNDS
#endif

extern "C" __global__ CONVROT_FWHT_BOUNDS void convrot_fwht_quantize_1(
    const __nv_bfloat16* x, int8_t* codes, float* scales, int rows, int k, int group) {
    convrot_fwht_quantize_reg<false, 1, false>(x, nullptr, codes, scales, rows, k, group, 0);
}

extern "C" __global__ CONVROT_FWHT_BOUNDS void convrot_fwht_quantize_2(
    const __nv_bfloat16* x, int8_t* codes, float* scales, int rows, int k, int group) {
    convrot_fwht_quantize_reg<false, 2, false>(x, nullptr, codes, scales, rows, k, group, 0);
}

extern "C" __global__ CONVROT_FWHT_BOUNDS void convrot_fwht_quantize_4(
    const __nv_bfloat16* x, int8_t* codes, float* scales, int rows, int k, int group) {
    convrot_fwht_quantize_reg<false, 4, false>(x, nullptr, codes, scales, rows, k, group, 0);
}

extern "C" __global__ CONVROT_FWHT_BOUNDS void convrot_fwht_quantize_silu_1(
    const __nv_bfloat16* gu, int8_t* codes, float* scales, int rows, int k, int group) {
    convrot_fwht_quantize_reg<true, 1, false>(gu, nullptr, codes, scales, rows, k, group, 0);
}

extern "C" __global__ CONVROT_FWHT_BOUNDS void convrot_fwht_quantize_silu_2(
    const __nv_bfloat16* gu, int8_t* codes, float* scales, int rows, int k, int group) {
    convrot_fwht_quantize_reg<true, 2, false>(gu, nullptr, codes, scales, rows, k, group, 0);
}

extern "C" __global__ CONVROT_FWHT_BOUNDS void convrot_fwht_quantize_silu_4(
    const __nv_bfloat16* gu, int8_t* codes, float* scales, int rows, int k, int group) {
    convrot_fwht_quantize_reg<true, 4, false>(gu, nullptr, codes, scales, rows, k, group, 0);
}

extern "C" __global__ CONVROT_FWHT_BOUNDS void convrot_fwht_quantize_smem(
    const __nv_bfloat16* x, int8_t* codes, float* scales, int rows, int k, int group) {
    convrot_fwht_quantize_smem_impl<false, false>(x, nullptr, codes, scales, rows, k, group, 0);
}

extern "C" __global__ CONVROT_FWHT_BOUNDS void convrot_fwht_quantize_silu_smem(
    const __nv_bfloat16* gu, int8_t* codes, float* scales, int rows, int k, int group) {
    convrot_fwht_quantize_smem_impl<true, false>(gu, nullptr, codes, scales, rows, k, group, 0);
}

extern "C" __global__ CONVROT_FWHT_BOUNDS void convrot_fwht_quantize_probe(
    const __nv_bfloat16* x, float* rotated, int8_t* codes, float* scales, int rows, int k, int group, int mutant) {
    convrot_fwht_quantize_reg<false, 4, true>(x, rotated, codes, scales, rows, k, group, mutant);
}

extern "C" __global__ CONVROT_FWHT_BOUNDS void convrot_fwht_quantize_silu_probe(
    const __nv_bfloat16* gu, float* rotated, int8_t* codes, float* scales, int rows, int k, int group, int mutant) {
    convrot_fwht_quantize_reg<true, 4, true>(gu, rotated, codes, scales, rows, k, group, mutant);
}

extern "C" __global__ CONVROT_FWHT_BOUNDS void convrot_fwht_quantize_smem_probe(
    const __nv_bfloat16* x, float* rotated, int8_t* codes, float* scales, int rows, int k, int group, int mutant) {
    convrot_fwht_quantize_smem_impl<false, true>(x, rotated, codes, scales, rows, k, group, mutant);
}
