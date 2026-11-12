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

// 17 rows and above. One warp owns a 16x8 tile of one group.
// The dot is mma.m16n8k16, bf16 inputs, f32 accumulator, K groups of 16 in order.
// grid: (ceil(rows/16), group/8, k/group). block: 32.
extern "C" __global__ void convrot_group_dot(const float* __restrict__ x, float* __restrict__ y, int rows, int k,
                                             int group) {
    if (rows < 17 || group < 16 || (group & 15) || (k % group)) __trap();
    constexpr int STRIDE = 16;
    __shared__ __nv_bfloat16 asmem[16 * STRIDE];
    __shared__ __nv_bfloat16 bsmem[8 * STRIDE];
    const int lane = threadIdx.x;
    const int row0 = blockIdx.x * 16;
    const int n0 = blockIdx.y * 8;
    const int x_col0 = blockIdx.z * group;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k0 = 0; k0 < group; k0 += 16) {
        for (int i = lane; i < 16 * 16; i += 32) {
            const int r = i >> 4;
            const int c = i & 15;
            const int gr = row0 + r;
            float v = 0.f;
            if (gr < rows) v = x[static_cast<long long>(gr) * k + x_col0 + k0 + c];
            asmem[r * STRIDE + c] = __float2bfloat16(v);
        }
        for (int i = lane; i < 8 * 16; i += 32) {
            const int n = i >> 4;
            const int c = i & 15;
            bsmem[n * STRIDE + c] = __float2bfloat16(convrot_hadamard(group, k0 + c, n0 + n));
        }
        __syncwarp();
        unsigned a[4];
        convrot_ldsm_x4(a, asmem + (lane & 15) * STRIDE + (lane >> 4) * 8);
        unsigned b[2];
        convrot_ldsm_x2(b, bsmem + (lane & 7) * STRIDE + ((lane >> 3) & 1) * 8);
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

// s_x = max(|dot|, 1e-12) / 127. q = rint(dot * (1/s_x)), clamped to [-127, 127].
extern "C" __global__ void convrot_quantize(const float* __restrict__ xr, int8_t* __restrict__ codes,
                                            float* __restrict__ scales, int rows, int k) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;
    const float* src = xr + static_cast<long long>(row) * k;
    float peak = 0.f;
    for (int i = 0; i < k; ++i) peak = fmaxf(peak, fabsf(src[i]));
    const float scale = __fdiv_rn(fmaxf(peak, 1e-12f), 127.f);
    scales[row] = scale;
    const float inv = __fdiv_rn(1.f, scale);
    int8_t* dst = codes + static_cast<long long>(row) * k;
    for (int i = 0; i < k; ++i) {
        float q = rintf(__fmul_rn(src[i], inv));
        q = fminf(fmaxf(q, -127.f), 127.f);
        dst[i] = static_cast<int8_t>(static_cast<int>(q));
    }
}

extern "C" __global__ void convrot_product(const int8_t* __restrict__ codes, const float* __restrict__ scales,
                                           const int8_t* __restrict__ qweight, const float* __restrict__ wscale,
                                           __nv_bfloat16* __restrict__ y, int rows, int k, int n) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= rows * n) return;
    const int row = idx / n;
    const int col = idx - row * n;
    const int8_t* xq = codes + static_cast<long long>(row) * k;
    const int8_t* weight = qweight + static_cast<long long>(col) * k;
    int acc = 0;
    for (int i = 0; i < k; ++i) acc += static_cast<int>(xq[i]) * static_cast<int>(weight[i]);
    const float scale = scales[row];
    y[static_cast<long long>(row) * n + col] =
        __float2bfloat16(__fmul_rn(__fmul_rn(static_cast<float>(acc), scale), wscale[col]));
}
