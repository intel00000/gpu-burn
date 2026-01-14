// stress_gemm.cpp
// same idea based on the original gpu_burn
#include "gpuburn/modes/stress_gemm.h"
#include "gpuburn/device_context.h"
#include "gpuburn/errors.h"
#include "gpuburn/kernels/gemm_kernels.h"
#include <cstdio>
#include <type_traits>

using namespace gpuburn;

template <typename T> void StressGemm<T>::release() noexcept {
    if (device_id_ < 0)
        return;
    cudaSetDevice(device_id_);
    if (done_evt_) {
        cudaEventDestroy(done_evt_);
        done_evt_ = nullptr;
    }
    if (dC_) {
        cudaFree(dC_);
        dC_ = nullptr;
    }
    if (dA_) {
        cudaFree(dA_);
        dA_ = nullptr;
    }
    if (dB_) {
        cudaFree(dB_);
        dB_ = nullptr;
    }
    if (dFaulty_) {
        cudaFree(dFaulty_);
        dFaulty_ = nullptr;
    }
    if (hFaultyPinned_) {
        cudaFreeHost(hFaultyPinned_);
        hFaultyPinned_ = nullptr;
    }
}

template <typename T>
void StressGemm<T>::init(DeviceContext &dev, const StressConfig &cfg) {
    device_id_ = dev.device_id;
    use_tensor_ = cfg.use_tensor;
    release();
    // matrix size from config
    size_ = cfg.matrix_size;
    elems_ = size_ * size_;
    mat_bytes_ = sizeof(T) * elems_;

    const size_t use_bytes = resolve_use_bytes(dev, cfg.use_bytes);
    const size_t mat_replicas = use_bytes / mat_bytes_;
    if (mat_replicas < 3) // need at least A+B+1xC, ie 3 matrices
        throw std::runtime_error(
            "Not enough VRAM for A+B+at least 1xC matrices");
    iters_ = mat_replicas - 2; // number of C matrices
    std::printf("GPU %d init: matrix=%zu MB, iters=%zu, using~%zu MB\n",
                dev.device_id, mat_bytes_ / (1024 * 1024), iters_,
                mat_replicas * mat_bytes_ / (1024 * 1024));

    cuda_check(cudaSetDevice(dev.device_id), "cudaSetDevice(init)");
    cuda_check(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1),
               "cudaDeviceSetCacheConfig(L1)");

    try {
        cuda_check(cudaEventCreateWithFlags(&done_evt_, cudaEventDisableTiming),
                   "cudaEventCreate(done)");
        cuda_check(cudaMalloc((void **)&dA_, mat_bytes_), "cudaMalloc(A)");
        cuda_check(cudaMalloc((void **)&dB_, mat_bytes_), "cudaMalloc(B)");
        cuda_check(cudaMalloc((void **)&dC_, iters_ * mat_bytes_),
                   "cudaMalloc(C)");
        cuda_check(cudaMalloc((void **)&dFaulty_, sizeof(int)),
                   "cudaMalloc(faulty)");
        cuda_check(cudaHostAlloc((void **)&hFaultyPinned_, sizeof(int),
                                 cudaHostAllocDefault),
                   "cudaHostAlloc(faulty)");

        if constexpr (std::is_same_v<T, float>) {
            curand_check(curandGenerateUniform(dev.rng, (float *)dA_, elems_),
                         "curand(A)");
            curand_check(curandGenerateUniform(dev.rng, (float *)dB_, elems_),
                         "curand(B)");
            launch_scale_float((float *)dA_, elems_, 2.0f, -1.0f, dev.stream);
            launch_scale_float((float *)dB_, elems_, 2.0f, -1.0f, dev.stream);
        } else {
            curand_check(
                curandGenerateUniformDouble(dev.rng, (double *)dA_, elems_),
                "curand(A)");
            curand_check(
                curandGenerateUniformDouble(dev.rng, (double *)dB_, elems_),
                "curand(B)");
            launch_scale_double((double *)dA_, elems_, 2.0, -1.0, dev.stream);
            launch_scale_double((double *)dB_, elems_, 2.0, -1.0, dev.stream);
        }

        cuda_check(cudaGetLastError(), "kernel(scale)");
        cuda_check(cudaStreamSynchronize(dev.stream), "sync(init)");
    } catch (...) {
        release();
        throw;
    }
}

template <typename T> PerfModel StressGemm<T>::perf_model() const {
    // FLOPs for reporting, likely overestimated
    // 2*N^3 for square naive GEMM (N^3 multiplies + N^3 adds)
    // Original gpu_burn-drv.cpp used OPS_PER_MUL = 1100048498688 for SIZE=8192
    //      (measured via Visual Profiler, equals 2*N^3 * 2049/2048)
    const double ops = 2.0 * (double)size_ * (double)size_ * (double)size_;
    return PerfModel{MetricKind::Gflops, ops};
}

template <typename T> StepResult StressGemm<T>::step(DeviceContext &dev) {
    cuda_check(cudaSetDevice(dev.device_id), "cudaSetDevice(step)");

    // Float: TF32 or FP32
    // Double: FP64
    for (size_t i = 0; i < iters_; ++i) {
        T *Ci = dC_ + i * elems_;
        if constexpr (std::is_same_v<T, float>) {
            const float alpha = 1.0f, beta = 0.0f;
            const cublasComputeType_t compute_type =
                use_tensor_ ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
            cublas_check(cublasGemmEx(dev.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      (int)size_, (int)size_, (int)size_,
                                      &alpha, dA_, CUDA_R_32F, (int)size_, dB_,
                                      CUDA_R_32F, (int)size_, &beta, Ci,
                                      CUDA_R_32F, (int)size_, compute_type,
                                      CUBLAS_GEMM_DEFAULT),
                         "cublasGemmEx(float)");
        } else {
            const double alpha = 1.0, beta = 0.0;
            cublas_check(cublasGemmEx(dev.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      (int)size_, (int)size_, (int)size_,
                                      &alpha, dA_, CUDA_R_64F, (int)size_, dB_,
                                      CUDA_R_64F, (int)size_, &beta, Ci,
                                      CUDA_R_64F, (int)size_,
                                      CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT),
                         "cublasGemmEx(double)");
        }
    }

    // validate results, check for consistency across iters_
    cuda_check(cudaMemsetAsync(dFaulty_, 0, sizeof(int), dev.stream),
               "memset(faulty)");
    if constexpr (std::is_same_v<T, float>)
        launch_compare_float((const float *)dC_, dFaulty_, iters_, elems_,
                             dev.stream);
    else
        launch_compare_double((const double *)dC_, dFaulty_, iters_, elems_,
                              dev.stream);

    cuda_check(cudaGetLastError(), "kernel(compare)");
    cuda_check(cudaMemcpyAsync(hFaultyPinned_, dFaulty_, sizeof(int),
                               cudaMemcpyDeviceToHost, dev.stream),
               "DtoH(faulty)");

    cuda_check(cudaEventRecord(done_evt_, dev.stream), "eventRecord(done)");
    cuda_check(cudaEventSynchronize(done_evt_), "eventSync(done)");

    StepResult r;
    r.units = iters_;
    r.errors = (uint64_t)(*hFaultyPinned_);
    return r;
}

// Explicit instantiation
template class StressGemm<float>;
template class StressGemm<double>;
