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

// form 0: alpha*x-beta*v; form 1: (alpha*x-x0)/beta.
extern "C" __global__ void flow_x0_f32(const float* x, const float* prediction,
    float* output, unsigned long long count, float alpha, float beta, int form) {
    if (!x || !prediction || !output || !count || !(beta > 0) || form < 0 || form > 1) { __trap(); return; }
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (unsigned long long)gridDim.x * blockDim.x) {
        const float left = __fmul_rn(alpha, x[i]);
        output[i] = form == 0 ? __fsub_rn(left, __fmul_rn(beta, prediction[i]))
            : __fdiv_rn(__fsub_rn(left, prediction[i]), beta);
    }
}

extern "C" __global__ void flow_rescale_f32(float* x0, const float* std_pos,
    const float* std_cfg, unsigned long long count, float amount) {
    const float scale = __fdiv_rn(*std_pos, *std_cfg);
    if (!isfinite(scale)) { __trap(); return; }
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (unsigned long long)gridDim.x * blockDim.x) {
        const float scaled = __fmul_rn(x0[i], scale);
        x0[i] = __fadd_rn(__fmul_rn(amount, scaled), __fmul_rn(__fsub_rn(1.0f, amount), x0[i]));
    }
}
