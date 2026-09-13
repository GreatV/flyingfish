// F32 width-128 mean reduction matching PyTorch's CUDA Reduce.cuh path.
//
// Pinned source:
//   PyTorch 7269437d655783a26cba32aa88195b741ff496aa
//   aten/src/ATen/native/cuda/ReduceMomentKernel.cu
//   aten/src/ATen/native/cuda/Reduce.cuh
//   aten/src/ATen/native/SharedReduceOps.h
//
// PyTorch is distributed under its BSD-style license; see
// https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/LICENSE
//
// This is deliberately limited to the contiguous F32, last-dimension width
// 128 configuration used by released Qwen3-VL language q/k head RMSNorm.
// PyTorch vectorizes the input by four, assigns one 32-lane warp to each
// output row, accumulates the four vector lanes in order, then uses descending
// warp-shuffle offsets 16,8,4,2,1 before multiplying by 1/128.

namespace {

struct alignas(16) Float4 {
  float values[4];
};

__device__ __forceinline__ float add(float left, float right) {
  return __fadd_rn(left, right);
}

}  // namespace

extern "C" __global__ __launch_bounds__(512, 4)
void qwen_head_mean_width128_f32(
    int rows,
    const float* __restrict__ input,
    float* __restrict__ output) {
  // PyTorch's fixed ReduceConfig is block=(32,16), with threadIdx.y selecting
  // one output and threadIdx.x selecting one aligned float4 input vector.
  const int row = static_cast<int>(blockIdx.x) * 16
      + static_cast<int>(threadIdx.y);
  if (row >= rows) {
    return;
  }

  const auto* vectors = reinterpret_cast<const Float4*>(input + row * 128);
  const Float4 values = vectors[threadIdx.x];
  float sum = 0.0f;
#pragma unroll
  for (int lane = 0; lane < 4; ++lane) {
    sum = add(sum, values.values[lane]);
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum = add(sum, __shfl_down_sync(0xffffffffu, sum, offset));
  }
  if (threadIdx.x == 0) {
    output[row] = __fmul_rn(sum, 0.0078125f);
  }
}

// For 357/1935 rows of width 5120, ReduceConfig also chooses (32,16):
// 160 scalar values per thread is below the 256 threshold for block-y
// reduction. Keep four independent accumulators across all 40 float4 loads,
// then combine them before the warp reduction (Reduce.cuh).
extern "C" __global__ __launch_bounds__(512, 4)
void qwen_hidden_mean_width5120_f32(
    int rows,
    const float* __restrict__ input,
    float* __restrict__ output) {
  const int row = static_cast<int>(blockIdx.x) * 16 + static_cast<int>(threadIdx.y);
  if (row >= rows) return;
  const auto* vectors = reinterpret_cast<const Float4*>(input + row * 5120);
  float accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int index = threadIdx.x; index < 1280; index += 32) {
    const Float4 values = vectors[index];
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      accumulators[lane] = add(accumulators[lane], values.values[lane]);
    }
  }
  float sum = accumulators[0];
#pragma unroll
  for (int lane = 1; lane < 4; ++lane) sum = add(sum, accumulators[lane]);
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum = add(sum, __shfl_down_sync(0xffffffffu, sum, offset));
  }
  if (threadIdx.x == 0) output[row] = __fmul_rn(sum, 1.0f / 5120.0f);
}
