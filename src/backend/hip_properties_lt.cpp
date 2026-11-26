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
    int32_t max_grid_z;
    int32_t clock_khz;
    int32_t reserved;
};
static_assert(sizeof(Properties) == 608 && alignof(Properties) == 8);
static_assert(offsetof(Properties, optin_shared_bytes) == 528);
static_assert(offsetof(Properties, warp) == 536);
static_assert(offsetof(Properties, max_grid_y) == 580);
static_assert(offsetof(Properties, async_engines) == 584);
static_assert(offsetof(Properties, max_grid_x) == 592);
static_assert(offsetof(Properties, max_grid_z) == 596);
static_assert(offsetof(Properties, clock_khz) == 600);
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
static_assert(HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES == 3);
static_assert(HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES == 8);
static_assert(hipStreamCaptureModeThreadLocal == 1);
static_assert(hipErrorLaunchFailure == 719);
static_assert(HIP_R_32F == 0 && HIP_R_16BF == 14);
static_assert(HIPBLAS_COMPUTE_32F == 2);
static_assert(HIP_R_64F == 1 && HIPBLAS_COMPUTE_64F == 7);
static_assert(HIPBLAS_OP_N == 111 && HIPBLAS_OP_T == 112);
static_assert(HIPBLAS_STATUS_NOT_SUPPORTED == 7);
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
        hipDeviceAttributeMaxGridDimX, hipDeviceAttributeMaxGridDimZ,
        hipDeviceAttributeClockRate
    };
    int* values[] = {&output->warp, &output->compute_units, &output->max_threads,
        &output->shared_bytes, &output->unit_shared_bytes, &output->l2_bytes,
        &output->memory_clock_khz, &output->memory_bus_bits, &output->cooperative,
        &output->max_grid_y, &output->async_engines, &output->host_register_supported,
        &output->max_grid_x, &output->max_grid_z, &output->clock_khz};
    output->reserved = 0;
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
    float beta, hipblasLtMatmulAlgo_t* algorithm, size_t* workspace) {
    const float alpha = 1.0f;
    return hipblaslt_ext::matmulIsAlgoSupported(handle, desc, &alpha, a, b,
        &beta, c, c, *algorithm, *workspace);
}

extern "C" int ff_hip_lt_algo_names(hipblasLtHandle_t handle,
    hipblasLtMatmulAlgo_t* algorithm,
    char* solution, size_t solution_len,
    char* kernel, size_t kernel_len) {
    if (!algorithm || (!solution && solution_len) || (!kernel && kernel_len)) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    auto copy_str = [](const std::string& src, char* dst, size_t cap) -> bool {
        if (!dst) return cap == 0;
        if (src.size() >= cap) return false;
        std::memcpy(dst, src.data(), src.size());
        dst[src.size()] = '\0';
        return true;
    };
    if (!copy_str(hipblaslt_ext::getSolutionNameFromAlgo(handle, *algorithm), solution, solution_len) ||
        !copy_str(hipblaslt_ext::getKernelNameFromAlgo(handle, *algorithm), kernel, kernel_len)) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    return HIPBLAS_STATUS_SUCCESS;
}

static hipError_t measure_probe_delta(
    size_t bytes,
    size_t* delta_out,
    const char** failure,
    const char* no_drop_msg,
    const char* changed_msg
) {
    size_t free_before = 0, total_before = 0;
    auto status = hipMemGetInfo(&free_before, &total_before);
    if (status != hipSuccess) return status;
    if (free_before < bytes) {
        if (failure) *failure = "insufficient free VRAM for granularity probe";
        return hipErrorOutOfMemory;
    }

    void* ptr = nullptr;
    status = hipMalloc(&ptr, bytes);
    if (status != hipSuccess) return status;
    size_t free_during = 0, total_during = 0;
    status = hipMemGetInfo(&free_during, &total_during);
    auto free_status = hipFree(ptr);
    if (status != hipSuccess) return status;
    if (free_status != hipSuccess) return free_status;
    if (free_during >= free_before) {
        if (failure) *failure = no_drop_msg;
        return hipErrorUnknown;
    }

    size_t free_after = 0, total_after = 0;
    status = hipMemGetInfo(&free_after, &total_after);
    if (status != hipSuccess) return status;
    if (free_after != free_before) {
        if (failure) *failure = changed_msg;
        return hipErrorUnknown;
    }

    *delta_out = free_before - free_during;
    return hipSuccess;
}

extern "C" int ff_hip_allocation_granularity(int device, size_t* out, const char** failure) {
    if (failure) *failure = nullptr;
    if (!out) {
        if (failure) *failure = "null allocation granularity output pointer";
        return hipErrorInvalidValue;
    }
    auto status = hipSetDevice(device);
    if (status != hipSuccess) return status;
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;
    size_t vmm_gran = 0;
    status = hipMemGetAllocationGranularity(&vmm_gran, &prop, hipMemAllocationGranularityRecommended);
    if (status != hipSuccess) return status;
    if (vmm_gran == 0) {
        if (failure) *failure = "VMM allocation granularity is zero";
        return hipErrorInvalidValue;
    }

    // Sense physical hipMalloc backing chunk granularity (2 MiB on gfx1100 vs 4 KiB VMM query).
    size_t chunk = 0;
    status = measure_probe_delta(
        vmm_gran,
        &chunk,
        failure,
        "first hipMalloc probe did not reduce free VRAM (pool reuse or external free)",
        "free memory changed during probe (another allocation on the device)"
    );
    if (status != hipSuccess) return status;
    if (chunk < vmm_gran) {
        if (failure) *failure = "chunk below VMM granularity";
        return hipErrorUnknown;
    }
    if ((chunk % vmm_gran) != 0) {
        if (failure) *failure = "chunk not a multiple of VMM granularity";
        return hipErrorUnknown;
    }

    size_t free_avail = 0, total_avail = 0;
    status = hipMemGetInfo(&free_avail, &total_avail);
    if (status != hipSuccess) return status;
    if (chunk <= SIZE_MAX - vmm_gran && free_avail >= chunk + vmm_gran) {
        size_t delta2 = 0;
        status = measure_probe_delta(
            chunk + vmm_gran,
            &delta2,
            failure,
            "second hipMalloc probe did not reduce free VRAM (pool reuse or external free)",
            "free memory changed during second probe (another allocation on the device)"
        );
        if (status != hipSuccess) return status;
        if (delta2 <= chunk) {
            if (failure) *failure = "two-step probe allocation did not exceed single chunk";
            return hipErrorUnknown;
        }
        size_t step = delta2 - chunk;
        if ((step % vmm_gran) != 0) {
            if (failure) *failure = "second-step allocation increment not a multiple of VMM granularity";
            return hipErrorUnknown;
        }
        if (step > chunk) chunk = step;
    }

    *out = chunk;
    return hipSuccess;
}

extern "C" int ff_hip_default_mem_pool_usage(int device, uint64_t* reserved, uint64_t* used) {
    if (!reserved || !used) return hipErrorInvalidValue;
    hipMemPool_t pool = nullptr;
    auto status = hipDeviceGetDefaultMemPool(&pool, device);
    if (status != hipSuccess) return status;
    status = hipMemPoolGetAttribute(pool, hipMemPoolAttrReservedMemCurrent, reserved);
    if (status != hipSuccess) return status;
    return hipMemPoolGetAttribute(pool, hipMemPoolAttrUsedMemCurrent, used);
}

extern "C" int ff_hip_reset_default_mem_pool_peak(int device) {
    hipMemPool_t pool = nullptr;
    auto status = hipDeviceGetDefaultMemPool(&pool, device);
    if (status != hipSuccess) return status;
    uint64_t zero = 0;
    return hipMemPoolSetAttribute(pool, hipMemPoolAttrUsedMemHigh, &zero);
}

extern "C" int ff_hip_default_mem_pool_peak(int device, uint64_t* used, uint64_t* peak) {
    if (!used || !peak) return hipErrorInvalidValue;
    hipMemPool_t pool = nullptr;
    auto status = hipDeviceGetDefaultMemPool(&pool, device);
    if (status != hipSuccess) return status;
    status = hipMemPoolGetAttribute(pool, hipMemPoolAttrUsedMemCurrent, used);
    if (status != hipSuccess) return status;
    return hipMemPoolGetAttribute(pool, hipMemPoolAttrUsedMemHigh, peak);
}
