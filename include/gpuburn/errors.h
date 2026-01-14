// errors.h
// wrapper for CUDA/cuBLAS/cuRAND function calling with error checking
#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <stdexcept>
#include <string>

inline void cuda_check(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " +
                                 cudaGetErrorString(e));
    }
}

inline const char *cublas_status_string(cublasStatus_t s) {
#if defined(CUBLAS_VERSION)
    const char *p = cublasGetStatusString(s);
    return p ? p : "CUBLAS_STATUS_<unknown>";
#else
    (void)s;
    return "CUBLAS_STATUS_<unknown>";
#endif
}

inline void cublas_check(cublasStatus_t s, const char *what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": " +
                                 cublas_status_string(s));
    }
}

inline void curand_check(curandStatus_t s, const char *what) {
    if (s != CURAND_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": CURAND_STATUS_" +
                                 std::to_string((int)s));
    }
}
