// One block per row, matching cuda_mean in math/normalization.rs:
// - accumulate each vector slot in iteration order, then sum slots sequentially;
// - reduce lanes with a halving tree, then reduce groups with another;
// - compute mean_scale in F64 on the host, cast to F32, then apply +eps and rsqrtf;
// - use --fmad=false to preserve rounding between the square and accumulation,
//   and the same --prec-sqrt/--ftz flags as glm_rsqrt_f32.
// The host guarantees iterations * lanes * vector == width, so no padding is needed.
// See math/normalization.rs for bitwise tests and build.rs for the PTX target.

#define GLM_NORM_MAX_LANES 512

extern "C" __global__ void glm_normalized_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int width,
    const int vector,
    const int block_x,
    const int groups,
    const int iterations,
    const float mean_scale,
    const float eps)
{
    const long long row = blockIdx.x;
    const float* in = input + row * (long long)width;
    __shared__ float partial[GLM_NORM_MAX_LANES];
    __shared__ float shared_inv;
    const int lane = threadIdx.x;
    const int lanes = block_x * groups;
    if (lane < lanes) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int it = 0; it < iterations; ++it) {
            for (int s = 0; s < vector; ++s) {
                const int index = (it * lanes + lane) * vector + s;
                const float v = in[index];
                acc[s] += v * v;
            }
        }
        float sum = acc[0];
        for (int s = 1; s < vector; ++s) sum += acc[s];
        partial[lane] = sum;
    }
    __syncthreads();
    // Match cuda_mean's [rows, groups, block_x] layout.
    const int k = lane % block_x;
    const int g = lane / block_x;
    for (int off = block_x >> 1; off > 0; off >>= 1) {
        if (lane < lanes && k < off) partial[lane] += partial[lane + off];
        __syncthreads();
    }
    for (int off = groups >> 1; off > 0; off >>= 1) {
        if (lane < lanes && k == 0 && g < off)
            partial[lane] += partial[(g + off) * block_x];
        __syncthreads();
    }
    if (lane == 0) shared_inv = rsqrtf(partial[0] * mean_scale + eps);
    __syncthreads();
    const float inv = shared_inv;
    float* out = output + row * (long long)width;
    for (int i = lane; i < width; i += blockDim.x) out[i] = in[i] * inv;
}
