// device_context.h
#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>

/*
 * DeviceContext encapsulates CUDA resources associated with a specific GPU
 * device, including a CUDA stream, cuBLAS handle, and cuRAND generator.
 */
struct DeviceContext {
    int device_id = 0;               // GPU device ID
    cudaStream_t stream = nullptr;   // CUDA stream handle
    cublasHandle_t cublas = nullptr; // cuBLAS handle
    curandGenerator_t rng = nullptr; // cuRAND generator handle

    // Init the device context for the specified device ID.
    void init(int dev_id, bool use_tensor, bool use_doubles);
    // Shutdown and release all resources.
    void shutdown();
    // return free and total memory on device
    void mem_info(size_t &free_bytes, size_t &total_bytes) const;
};
