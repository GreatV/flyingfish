// Token-parallel norms for prefill slabs: one block per token row.
// Per-row arithmetic is identical to the scalar edge0_rmsnorm_zc and
// edge0_add_rmsnorm_zc — same strides, same reduction trees, same thread
// counts — so a row's result is bit-identical to the scalar kernels.

extern "C" __global__ void qwen_rmsnorm_rows(
    const float* __restrict__ x,    // [T, n]
    const float* __restrict__ w,    // [n]
    float* __restrict__ out,        // [T, n]
    int n,
    float eps)
{
    x += (long long)blockIdx.x * n;
    out += (long long)blockIdx.x * n;
    __shared__ float partials[256];
    const int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = x[i];
        acc += v * v;
    }
    partials[tid] = acc;
    __syncthreads();
    #pragma unroll
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) partials[tid] += partials[tid + off];
        __syncthreads();
    }
    const float inv = rsqrtf(partials[0] / (float)n + eps);
    for (int i = tid; i < n; i += blockDim.x)
        out[i] = x[i] * inv * (1.0f + w[i]);
}

// Launch with n.min(1024) threads, matching the scalar variant's rule.
extern "C" __global__ void qwen_add_rmsnorm_rows(
    float* __restrict__ acc,        // [T, n]
    const float* __restrict__ delta, // [T, n]
    const float* __restrict__ w,    // [n]
    float* __restrict__ out,        // [T, n]
    int n,
    float eps)
{
    acc += (long long)blockIdx.x * n;
    delta += (long long)blockIdx.x * n;
    out += (long long)blockIdx.x * n;
    __shared__ float partials[1024];
    const int tid = threadIdx.x;
    float sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float s = acc[i] + delta[i];
        acc[i] = s;
        sq += s * s;
    }
    if (tid < 1024) partials[tid] = sq;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (tid < off) partials[tid] += partials[tid + off];
        __syncthreads();
    }
    const float inv = rsqrtf(partials[0] / (float)n + eps);
    for (int i = tid; i < n; i += blockDim.x)
        out[i] = acc[i] * inv * (1.0f + w[i]);
}

extern "C" __global__ void copy_row(const float* source, float* out, int offset, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) out[i] = source[offset + i];
}
