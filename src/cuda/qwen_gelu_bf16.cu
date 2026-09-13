// Exact BF16 GELU forward kernels for the released Qwen3-VL vision profiles.
//
// This is a specialized transcription of PyTorch commit
// 7269437d655783a26cba32aa88195b741ff496aa:
//   aten/src/ATen/native/cuda/ActivationGeluKernel.cu
//
// PyTorch is distributed under its BSD-style license. See
// https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/LICENSE
//
// The block MLP uses approximate="tanh" through the released config while
// patch mergers instantiate nn.GELU() and therefore use approximate="none".
// Expression order and the single final BF16 cast are parity-critical.

// Keep this translation unit NVRTC-self-contained. In particular, do not
// import CUDA toolkit headers from a different release than the pinned NVRTC
// 13.0/libdevice implementation.
using BFloat16Bits = unsigned short;

__device__ __forceinline__ float bf16_to_float(BFloat16Bits value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16);
}

__device__ __forceinline__ BFloat16Bits float_to_bf16(float value) {
  BFloat16Bits result;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(result) : "f"(value));
  return result;
}

extern "C" __global__ void qwen_gelu_tanh_bf16(
    int elements,
    const BFloat16Bits* __restrict__ input,
    BFloat16Bits* __restrict__ output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    using Opmath = float;
    constexpr Opmath kBeta = Opmath(0.79788456080286535588);
    constexpr Opmath kKappa = 0.044715;
    const Opmath value = bf16_to_float(input[index]);
    const Opmath value_cube = value * value * value;
    const Opmath inner = kBeta * (value + kKappa * value_cube);
    output[index] = float_to_bf16(
        Opmath(0.5) * value * (Opmath(1) + tanhf(inner)));
  }
}

extern "C" __global__ void qwen_gelu_erf_bf16(
    int elements,
    const BFloat16Bits* __restrict__ input,
    BFloat16Bits* __restrict__ output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    using Opmath = float;
    constexpr Opmath kAlpha = Opmath(0.70710678118654752440);
    const Opmath value = bf16_to_float(input[index]);
    output[index] = float_to_bf16(
        value * Opmath(0.5) * (Opmath(1) + erff(value * kAlpha)));
  }
}
