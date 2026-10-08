#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

inline void ff_cuda_check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

#define C10_CUDA_CHECK(call) ff_cuda_check(call)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
