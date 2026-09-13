// BF16 scalar multiplication matching PyTorch's CUDA BinaryMulKernel.
//
// Pinned source:
//   PyTorch 7269437d655783a26cba32aa88195b741ff496aa
//   aten/src/ATen/native/cuda/BinaryMulKernel.cu
//   aten/src/ATen/native/cuda/Loops.cuh
//
// PyTorch is distributed under its BSD-style license; see
// https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/LICENSE

using BFloat16Bits = unsigned short;

__device__ __forceinline__ float bf16_to_float(BFloat16Bits value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16);
}

__device__ __forceinline__ BFloat16Bits float_to_bf16(float value) {
  BFloat16Bits result;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(result) : "f"(value));
  return result;
}

extern "C" __global__ void qwen_attention_scale_bf16(
    int elements,
    float scale,
    const BFloat16Bits* __restrict__ input,
    BFloat16Bits* __restrict__ output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = float_to_bf16(bf16_to_float(input[index]) * scale);
  }
}
