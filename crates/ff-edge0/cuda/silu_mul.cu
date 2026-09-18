// Elementwise silu(g) * u over N lanes, in place into g (f32). Used between
// the gate+up GEMVs and the down GEMVs to keep the whole MoE block on one
// x upload and one sync: y[i] = g[i]/(1+exp(-g[i])) * u[i].

extern "C" __global__ void edge0_silu_mul(
    const float* __restrict__ g,
    const float* __restrict__ u,
    float* __restrict__ y,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float gv = g[i];
        y[i] = gv / (1.0f + __expf(-gv)) * u[i];
    }
}
