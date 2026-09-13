// Exact BF16 LayerNorm forward for the released Qwen3-VL vision profiles.
//
// This is a narrow, specialized transcription of PyTorch's vectorized CUDA
// LayerNorm path at commit 7269437d655783a26cba32aa88195b741ff496aa:
//   aten/src/ATen/native/cuda/layer_norm_kernel.cu
//   aten/src/ATen/native/cuda/thread_constants.h
//   c10/cuda/CUDAMathCompat.h
//
// PyTorch is distributed under its BSD-style license. See
// https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/LICENSE
//
// Reduction order, vec4 accesses, launch geometry, rsqrtf, and expression
// order are numerical-contract details. Algebraically equivalent rewrites are
// not allowed because the BF16 result must match the official runtime bitwise.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kVectorWidth = 4;

template <typename T, int Width>
struct alignas(sizeof(T) * Width) AlignedVector {
  T values[Width];
};

struct WelfordData {
  float mean;
  float sigma2;
  float count;
};

__device__ __forceinline__ WelfordData welford_online_sum(
    float value,
    const WelfordData& current) {
  const float delta = value - current.mean;
  const float new_count = current.count + 1.0f;
  const float new_mean = current.mean + delta * (1.0f / new_count);
  return {
      new_mean,
      current.sigma2 + delta * (value - new_mean),
      new_count};
}

// Keep PyTorch's data_b/data_a parameter order and exact expression tree.
__device__ __forceinline__ WelfordData welford_combine(
    const WelfordData data_b,
    const WelfordData data_a) {
  const float delta = data_b.mean - data_a.mean;
  const float count = data_a.count + data_b.count;
  if (count > 0.0f) {
    const float coefficient = 1.0f / count;
    const float n_a = data_a.count * coefficient;
    const float n_b = data_b.count * coefficient;
    const float mean = n_a * data_a.mean + n_b * data_b.mean;
    const float sigma2 = data_a.sigma2 + data_b.sigma2 +
                         delta * delta * data_a.count * n_b;
    return {mean, sigma2, count};
  }
  return {0.0f, 0.0f, count};
}

__device__ __forceinline__ WelfordData compute_stats(
    const __nv_bfloat16* __restrict__ input,
    int width,
    float* shared) {
  using Vector = AlignedVector<__nv_bfloat16, kVectorWidth>;
  const auto* input_vectors = reinterpret_cast<const Vector*>(input);
  const int thread_count = blockDim.x * blockDim.y;
  const int thread_index = threadIdx.x + threadIdx.y * blockDim.x;
  const int vector_count = width / kVectorWidth;
  WelfordData value{0.0f, 0.0f, 0.0f};

  for (int index = thread_index; index < vector_count; index += thread_count) {
    const Vector data = input_vectors[index];
#pragma unroll
    for (int lane = 0; lane < kVectorWidth; ++lane) {
      value = welford_online_sum(static_cast<float>(data.values[lane]), value);
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const WelfordData other{
        __shfl_down_sync(0xffffffffu, value.mean, offset),
        __shfl_down_sync(0xffffffffu, value.sigma2, offset),
        __shfl_down_sync(0xffffffffu, value.count, offset)};
    value = welford_combine(value, other);
  }

  // Pinned PyTorch launches four 32-thread warps and combines the upper half
  // into the lower half through this exact shared-memory tree.
  float* mean_sigma = shared;
  float* counts = shared + blockDim.y;
  for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
    if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
      const int write_y = threadIdx.y - offset;
      mean_sigma[2 * write_y] = value.mean;
      mean_sigma[2 * write_y + 1] = value.sigma2;
      counts[write_y] = value.count;
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y < offset) {
      const WelfordData other{
          mean_sigma[2 * threadIdx.y],
          mean_sigma[2 * threadIdx.y + 1],
          counts[threadIdx.y]};
      value = welford_combine(value, other);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    mean_sigma[0] = value.mean;
    mean_sigma[1] = value.sigma2 / static_cast<float>(width);
  }
  __syncthreads();
  return {mean_sigma[0], mean_sigma[1], 0.0f};
}

}  // namespace

extern "C" __global__ void qwen_layer_norm_bf16(
    int width,
    float epsilon,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ output) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  const __nv_bfloat16* row_input = input + row * width;
  const WelfordData stats = compute_stats(row_input, width, shared);
  const float reciprocal_standard_deviation =
      rsqrtf(stats.sigma2 + epsilon);

  using Vector = AlignedVector<__nv_bfloat16, kVectorWidth>;
  const auto* input_vectors = reinterpret_cast<const Vector*>(row_input);
  const auto* weight_vectors = reinterpret_cast<const Vector*>(weight);
  const auto* bias_vectors = reinterpret_cast<const Vector*>(bias);
  auto* output_vectors = reinterpret_cast<Vector*>(output + row * width);
  const int thread_count = blockDim.x * blockDim.y;
  const int thread_index = threadIdx.x + threadIdx.y * blockDim.x;
  const int vector_count = width / kVectorWidth;

  for (int index = thread_index; index < vector_count; index += thread_count) {
    const Vector data = input_vectors[index];
    Vector result;
#pragma unroll
    for (int lane = 0; lane < kVectorWidth; ++lane) {
      result.values[lane] =
          static_cast<float>(weight_vectors[index].values[lane]) *
              (reciprocal_standard_deviation *
               (static_cast<float>(data.values[lane]) - stats.mean)) +
          static_cast<float>(bias_vectors[index].values[lane]);
    }
    output_vectors[index] = result;
  }
}
