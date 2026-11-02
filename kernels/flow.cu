// Rectified-flow Euler step on F32 rows with a data-ward velocity:
//   x[i] = r * x[i] + q * (x[i] + s * v[i])
// s = 1 - t of the evaluation, r = sigma_next / sigma, q = 1 - r; every product and sum rounds separately (no
// contraction). Flat grid-stride launch.
#include "gpu_compat.cuh"

extern "C" __global__ void euler_blend_f32(float* x, const float* v, unsigned long long count, float s,
                                           float r, float q) {
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        const float sample = x[i];
        const float denoised = __fadd_rn(sample, __fmul_rn(s, v[i]));
        x[i] = __fadd_rn(__fmul_rn(r, sample), __fmul_rn(q, denoised));
    }
}
