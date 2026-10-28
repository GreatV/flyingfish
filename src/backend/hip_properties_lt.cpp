#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>

struct Properties {
    char name[256];
    char architecture[256];
    uint8_t uuid[16];
    uint64_t optin_shared_bytes;
    int32_t warp, compute_units, max_threads, shared_bytes, unit_shared_bytes;
    int32_t l2_bytes, memory_clock_khz, memory_bus_bits, cooperative;
    int32_t driver_version, runtime_version;
    int32_t max_grid_y;
    int32_t async_engines, host_register_supported;
    int32_t max_grid_x;
};
static_assert(sizeof(Properties) == 600 && alignof(Properties) == 8);
static_assert(offsetof(Properties, optin_shared_bytes) == 528);
static_assert(offsetof(Properties, warp) == 536);
static_assert(offsetof(Properties, max_grid_y) == 580);
static_assert(offsetof(Properties, async_engines) == 584);
static_assert(offsetof(Properties, max_grid_x) == 592);
static_assert(hipErrorOutOfMemory == 2);
static_assert(hipErrorNotReady == 600);
static_assert(hipHostMallocPortable == 1 && hipHostRegisterPortable == 1);
static_assert(sizeof(hipblasLtMatmulAlgo_t) == 24 && alignof(hipblasLtMatmulAlgo_t) == 8);
static_assert(sizeof(hipblasLtMatmulHeuristicResult_t) == 56);
static_assert(alignof(hipblasLtMatmulHeuristicResult_t) == 8);
static_assert(offsetof(hipblasLtMatmulHeuristicResult_t, workspaceSize) == 24);
static_assert(offsetof(hipblasLtMatmulHeuristicResult_t, state) == 32);
static_assert(offsetof(hipblasLtMatmulHeuristicResult_t, wavesCount) == 36);
static_assert(HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK == 0);
static_assert(HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES == 1);
static_assert(hipStreamCaptureModeThreadLocal == 1);
static_assert(hipErrorLaunchFailure == 719);
static_assert(HIP_R_32F == 0 && HIP_R_16BF == 14);
static_assert(HIPBLAS_COMPUTE_32F == 2);
static_assert(HIP_R_64F == 1 && HIPBLAS_COMPUTE_64F == 4);
static_assert(HIPBLAS_OP_N == 111 && HIPBLAS_OP_T == 112);
static_assert(HIPBLAS_GEMM_DEFAULT == 160 && HIPBLASLT_EPILOGUE_BIAS == 4);
static_assert(HIPBLASLT_MATMUL_DESC_TRANSA == 0 && HIPBLASLT_MATMUL_DESC_EPILOGUE == 2);
static_assert(HIPBLASLT_MATMUL_DESC_BIAS_POINTER == 3 && HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE == 4);
static_assert(HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES == 1);

extern "C" int ff_hip_device_properties(int device, Properties* output) {
    hipDeviceProp_t properties{};
    auto status = hipGetDeviceProperties(&properties, device);
    if (status != hipSuccess) return status;
    std::memcpy(output->name, properties.name, sizeof(output->name));
    std::memcpy(output->architecture, properties.gcnArchName, sizeof(output->architecture));
    std::memcpy(output->uuid, &properties.uuid, sizeof(output->uuid));
    output->optin_shared_bytes = properties.sharedMemPerBlockOptin;
    const hipDeviceAttribute_t attributes[] = {
        hipDeviceAttributeWarpSize, hipDeviceAttributeMultiprocessorCount,
        hipDeviceAttributeMaxThreadsPerBlock, hipDeviceAttributeMaxSharedMemoryPerBlock,
        hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, hipDeviceAttributeL2CacheSize,
        hipDeviceAttributeMemoryClockRate, hipDeviceAttributeMemoryBusWidth,
        hipDeviceAttributeCooperativeLaunch, hipDeviceAttributeMaxGridDimY,
        hipDeviceAttributeAsyncEngineCount, hipDeviceAttributeHostRegisterSupported,
        hipDeviceAttributeMaxGridDimX
    };
    int* values[] = {&output->warp, &output->compute_units, &output->max_threads,
        &output->shared_bytes, &output->unit_shared_bytes, &output->l2_bytes,
        &output->memory_clock_khz, &output->memory_bus_bits, &output->cooperative,
        &output->max_grid_y, &output->async_engines, &output->host_register_supported, &output->max_grid_x};
    for (size_t i = 0; i < sizeof(attributes) / sizeof(attributes[0]); ++i) {
        status = hipDeviceGetAttribute(values[i], attributes[i], device);
        if (status != hipSuccess) return status;
    }
    status = hipDriverGetVersion(&output->driver_version);
    if (status != hipSuccess) return status;
    return hipRuntimeGetVersion(&output->runtime_version);
}

extern "C" int ff_hip_lt_index(hipblasLtMatmulAlgo_t* algorithm) {
    return hipblaslt_ext::getIndexFromAlgo(*algorithm);
}

extern "C" int ff_hip_lt_supported(hipblasLtHandle_t handle,
    hipblasLtMatmulDesc_t desc, hipblasLtMatrixLayout_t a,
    hipblasLtMatrixLayout_t b, hipblasLtMatrixLayout_t c,
    hipblasLtMatmulAlgo_t* algorithm, size_t* workspace) {
    const float alpha = 1.0f, beta = 0.0f;
    return hipblaslt_ext::matmulIsAlgoSupported(handle, desc, &alpha, a, b,
        &beta, c, c, *algorithm, *workspace);
}
