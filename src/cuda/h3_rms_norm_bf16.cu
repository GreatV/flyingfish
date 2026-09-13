// A narrowly scoped BF16 RMSNorm forward kernel for official H3 parity.
//
// This is a clean, specialized transcription of the RMSNorm vectorized CUDA
// path in PyTorch, pinned at commit
// 7269437d655783a26cba32aa88195b741ff496aa:
//   aten/src/ATen/native/cuda/layer_norm_kernel.cu
//   aten/src/ATen/native/cuda/thread_constants.h
//
// The upstream code is distributed under PyTorch's BSD-style license. See
// https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/LICENSE
//
// The launch geometry, vec4 load/store width, reduction order, rsqrtf, and
// multiplication order intentionally match that pinned implementation. Do not
// replace them with an algebraically equivalent expression: BF16 output bits
// are part of the official-parity contract.

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

__device__ __forceinline__ WelfordData rms_online_sum(
    float value,
    const WelfordData& current) {
  return {0.0f, current.sigma2 + value * value, 0.0f};
}

__device__ __forceinline__ WelfordData rms_combine(
    const WelfordData data_b,
    const WelfordData data_a) {
  return {0.0f, data_b.sigma2 + data_a.sigma2, 0.0f};
}

__device__ __forceinline__ WelfordData compute_rms_stats(
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
      value = rms_online_sum(static_cast<float>(data.values[lane]), value);
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const WelfordData other{
        __shfl_down_sync(0xffffffffu, value.mean, offset),
        __shfl_down_sync(0xffffffffu, value.sigma2, offset),
        __shfl_down_sync(0xffffffffu, value.count, offset)};
    value = rms_combine(value, other);
  }

  // PyTorch launches four 32-thread warps. Keep its upper-warp-write then
  // lower-warp-merge tree, including the data_b + data_a operand order.
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
      value = rms_combine(value, other);
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

extern "C" __global__ void h3_rms_norm_bf16(
    int width,
    float epsilon,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  const __nv_bfloat16* row_input = input + row * width;
  const WelfordData stats = compute_rms_stats(row_input, width, shared);
  const float inverse_root_mean_square = rsqrtf(stats.sigma2 + epsilon);

  using Vector = AlignedVector<__nv_bfloat16, kVectorWidth>;
  const auto* input_vectors = reinterpret_cast<const Vector*>(row_input);
  const auto* weight_vectors = reinterpret_cast<const Vector*>(weight);
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
          (inverse_root_mean_square * static_cast<float>(data.values[lane]));
    }
    output_vectors[index] = result;
  }
}

// Qwen's unfused RMSNorm calls torch.rsqrt as a distinct F32 pointwise op.
// Keep that operation distinct from sqrt+reciprocal, which is not bitwise
// equivalent on CUDA.
extern "C" __global__ void qwen_rsqrt_f32(
    int elements,
    const float* __restrict__ input,
    float* __restrict__ output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = rsqrtf(input[index]);
  }
}
