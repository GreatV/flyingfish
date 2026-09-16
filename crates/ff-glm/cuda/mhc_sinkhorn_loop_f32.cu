// Fused Sinkhorn: one thread per token, normalizing dim 2 then dim 1.
// Preserve candle's halving-tree sums: streams=4 gives (m0+m2)+(m1+m3).
// Requires power-of-two streams, IEEE division, --prec-div=true and --ftz=false;
// do not use fast-math or __fdividef. Rust falls back for unsupported stream counts.
// See math/sinkhorn.rs for bitwise tests and build.rs for the PTX target.

#define GLM_MHC_MAX_STREAMS 8

extern "C" __global__ void glm_mhc_sinkhorn_loop_f32_v1(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int tokens,
    const int streams,
    const int rounds,
    const float eps)
{
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= tokens) return;
    const int n = streams * streams;
    const float* in = input + (long long)token * n;
    float m[GLM_MHC_MAX_STREAMS * GLM_MHC_MAX_STREAMS];
    for (int i = 0; i < n; ++i) m[i] = in[i];
    for (int round = 0; round < rounds; ++round) {
        for (int i = 0; i < streams; ++i) {
            float v[GLM_MHC_MAX_STREAMS];
            for (int j = 0; j < streams; ++j) v[j] = m[i * streams + j];
            for (int s = streams >> 1; s > 0; s >>= 1)
                for (int k = 0; k < s; ++k)
                    v[k] += v[k + s];
            const float den = v[0] + eps;
            for (int j = 0; j < streams; ++j) m[i * streams + j] = m[i * streams + j] / den;
        }
        for (int j = 0; j < streams; ++j) {
            float v[GLM_MHC_MAX_STREAMS];
            for (int i = 0; i < streams; ++i) v[i] = m[i * streams + j];
            for (int s = streams >> 1; s > 0; s >>= 1)
                for (int k = 0; k < s; ++k)
                    v[k] += v[k + s];
            const float den = v[0] + eps;
            for (int i = 0; i < streams; ++i) m[i * streams + j] = m[i * streams + j] / den;
        }
    }
    float* out = output + (long long)token * n;
    for (int i = 0; i < n; ++i) out[i] = m[i];
}
