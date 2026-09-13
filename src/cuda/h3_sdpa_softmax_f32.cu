// F32 persistent softmax for official H3 native-math SDPA parity.
//
// This is a clean transcription of the unmasked F32 forward dispatch in
// PyTorch commit 7269437d655783a26cba32aa88195b741ff496aa:
//   aten/src/ATen/native/cuda/PersistentSoftmax.cuh
//   aten/src/ATen/native/cuda/SoftMax.cu
//
// The upstream code is distributed under PyTorch's BSD-style license. See
// https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/LICENSE
//
// The H3 wrappers cover exactly the complete persistent dispatch range,
// widths 1..=2048 (log2_elements 0..=11). The Qwen wrappers additionally
// transcribe SoftMax.cu's regular register kernel for widths 2049..=9216
// (1024 threads and reg_count 3..=9). There is no fallback approximation.

#include <cuda_runtime.h>
#include <math.h>

namespace {

template <int Log2Elements>
__device__ __forceinline__ void persistent_softmax_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int width) {
  constexpr int kNextPowerOfTwo = 1 << Log2Elements;
  constexpr int kWarpSize = kNextPowerOfTwo < 32 ? kNextPowerOfTwo : 32;
  constexpr int kIterations = kNextPowerOfTwo / kWarpSize;
  constexpr int kBatchesPerWarp = kNextPowerOfTwo <= 128 ? 2 : 1;

  const int first_row =
      (blockDim.y * blockIdx.x + threadIdx.y) * kBatchesPerWarp;
  int local_rows = rows - first_row;
  if (local_rows > kBatchesPerWarp) {
    local_rows = kBatchesPerWarp;
  }
  const int lane = threadIdx.x;
  float elements[kBatchesPerWarp][kIterations];

#pragma unroll
  for (int local_row = 0; local_row < kBatchesPerWarp; ++local_row) {
    const int row_width = local_row >= local_rows ? 0 : width;
#pragma unroll
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      const int column = lane + iteration * kWarpSize;
      elements[local_row][iteration] =
          column < row_width
          ? input[(first_row + local_row) * width + column]
          : -__int_as_float(0x7f800000);
    }
  }

  float maximum[kBatchesPerWarp];
#pragma unroll
  for (int local_row = 0; local_row < kBatchesPerWarp; ++local_row) {
    maximum[local_row] = elements[local_row][0];
#pragma unroll
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      maximum[local_row] = maximum[local_row] > elements[local_row][iteration]
          ? maximum[local_row]
          : elements[local_row][iteration];
    }
  }
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int local_row = 0; local_row < kBatchesPerWarp; ++local_row) {
      const float other = __shfl_xor_sync(
          0xffffffffu, maximum[local_row], offset, kWarpSize);
      maximum[local_row] = maximum[local_row] < other
          ? other
          : maximum[local_row];
    }
  }

  float sums[kBatchesPerWarp] = {0.0f};
#pragma unroll
  for (int local_row = 0; local_row < kBatchesPerWarp; ++local_row) {
#pragma unroll
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      elements[local_row][iteration] =
          expf(elements[local_row][iteration] - maximum[local_row]);
      sums[local_row] += elements[local_row][iteration];
    }
  }
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int local_row = 0; local_row < kBatchesPerWarp; ++local_row) {
      const float other = __shfl_xor_sync(
          0xffffffffu, sums[local_row], offset, kWarpSize);
      sums[local_row] = sums[local_row] + other;
    }
  }

#pragma unroll
  for (int local_row = 0; local_row < kBatchesPerWarp; ++local_row) {
    if (local_row >= local_rows) {
      break;
    }
#pragma unroll
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      const int column = lane + iteration * kWarpSize;
      if (column < width) {
        output[(first_row + local_row) * width + column] =
            elements[local_row][iteration] / sums[local_row];
      } else {
        break;
      }
    }
  }
}

__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other = __shfl_down_sync(0xffffffffu, value, offset);
    value = value < other ? other : value;
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

// Exact specialization of PyTorch blockReduceWarp for a 1024-thread block.
// The first reduction produces 32 warp values and the first warp then reduces
// those values in the same shuffle-down order as block_reduce.cuh.
template <bool IsMax>
__device__ __forceinline__ float regular_block_reduce(
    float* shared, float value) {
  constexpr float kLowest = -3.402823466e+38F;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = IsMax ? warp_reduce_max(value) : warp_reduce_sum(value);
  __syncthreads();
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  value = threadIdx.x < 32 ? shared[lane] : (IsMax ? kLowest : 0.0f);
  if (warp == 0) {
    value = IsMax ? warp_reduce_max(value) : warp_reduce_sum(value);
  }
  if (threadIdx.x == 0) {
    shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

template <int RegisterCount>
__device__ __forceinline__ void regular_softmax_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long width) {
  extern __shared__ float shared[];
  float values[RegisterCount];
  input += static_cast<long long>(blockIdx.x) * width;
  output += static_cast<long long>(blockIdx.x) * width;

  float thread_max = -3.402823466e+38F;
#pragma unroll
  for (int register_index = 0; register_index < RegisterCount; ++register_index) {
    const int offset = threadIdx.x + register_index * blockDim.x;
    if (offset < width) {
      values[register_index] = input[offset];
      thread_max = thread_max < values[register_index]
          ? values[register_index]
          : thread_max;
    }
  }
  const float maximum = regular_block_reduce<true>(shared, thread_max);

  float thread_sum = 0.0f;
#pragma unroll
  for (int register_index = 0; register_index < RegisterCount; ++register_index) {
    const int offset = threadIdx.x + register_index * blockDim.x;
    if (offset < width) {
      thread_sum += expf(values[register_index] - maximum);
    }
  }
  const float sum = regular_block_reduce<false>(shared, thread_sum);

#pragma unroll
  for (int register_index = 0; register_index < RegisterCount; ++register_index) {
    const int offset = threadIdx.x + register_index * blockDim.x;
    if (offset < width) {
      output[offset] = expf(values[register_index] - maximum) / sum;
    }
  }
}

}  // namespace

#define DEFINE_PERSISTENT_SOFTMAX_WRAPPER(LOG2)                              \
  extern "C" __global__ void h3_sdpa_softmax_f32_log2_##LOG2(               \
      const float* __restrict__ input,                                       \
      float* __restrict__ output,                                            \
      int rows,                                                              \
      int width) {                                                           \
    persistent_softmax_forward<LOG2>(input, output, rows, width);            \
  }

DEFINE_PERSISTENT_SOFTMAX_WRAPPER(0)
DEFINE_PERSISTENT_SOFTMAX_WRAPPER(1)
DEFINE_PERSISTENT_SOFTMAX_WRAPPER(2)
DEFINE_PERSISTENT_SOFTMAX_WRAPPER(3)
DEFINE_PERSISTENT_SOFTMAX_WRAPPER(4)
DEFINE_PERSISTENT_SOFTMAX_WRAPPER(5)
DEFINE_PERSISTENT_SOFTMAX_WRAPPER(6)
DEFINE_PERSISTENT_SOFTMAX_WRAPPER(7)
DEFINE_PERSISTENT_SOFTMAX_WRAPPER(8)
DEFINE_PERSISTENT_SOFTMAX_WRAPPER(9)
DEFINE_PERSISTENT_SOFTMAX_WRAPPER(10)
DEFINE_PERSISTENT_SOFTMAX_WRAPPER(11)

#define DEFINE_QWEN_REGULAR_SOFTMAX_WRAPPER(REGISTERS)                      \
  extern "C" __global__ void qwen_softmax_regular_f32_reg_##REGISTERS(     \
      const float* __restrict__ input,                                      \
      float* __restrict__ output,                                           \
      long long width) {                                                    \
    regular_softmax_forward<REGISTERS>(input, output, width);               \
  }

DEFINE_QWEN_REGULAR_SOFTMAX_WRAPPER(3)
DEFINE_QWEN_REGULAR_SOFTMAX_WRAPPER(4)
DEFINE_QWEN_REGULAR_SOFTMAX_WRAPPER(5)
DEFINE_QWEN_REGULAR_SOFTMAX_WRAPPER(6)
DEFINE_QWEN_REGULAR_SOFTMAX_WRAPPER(7)
DEFINE_QWEN_REGULAR_SOFTMAX_WRAPPER(8)
DEFINE_QWEN_REGULAR_SOFTMAX_WRAPPER(9)
