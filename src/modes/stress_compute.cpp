// stress_compute.cpp
#include "gpuburn/modes/stress_compute.h"
#include "gpuburn/device_context.h"
#include "gpuburn/errors.h"
#include "gpuburn/kernels/compute_kernels.h"
#include <cstdio>
#include <type_traits>

using namespace gpuburn;

template <typename T> void StressCompute<T>::release() noexcept {
    if (device_id_ < 0)
        return;
    cudaSetDevice(device_id_);
    if (done_evt_) {
        cudaEventDestroy(done_evt_);
        done_evt_ = nullptr;
    }
    if (dData_) {
        cudaFree(dData_);
        dData_ = nullptr;
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
void StressCompute<T>::init(DeviceContext &dev, const StressConfig &cfg) {
    device_id_ = dev.device_id;
    release();

    // If didn't specify -m, use default, clamp to range
    buffer_bytes_ = (cfg.use_bytes == 0)
                        ? kComputeDefaultBytes
                        : resolve_use_bytes(dev, cfg.use_bytes);
    if (buffer_bytes_ < kComputeMinBytes)
        buffer_bytes_ = kComputeMinBytes;
    if (buffer_bytes_ > kComputeMaxBytes)
        buffer_bytes_ = kComputeMaxBytes;

    elements_ = buffer_bytes_ / sizeof(T) / 2;
    if (elements_ < 1024)
        throw std::runtime_error("Not enough VRAM for compute stress buffers");
    std::printf(
        "GPU %d init: compute stress, %zu elements/half, ~%llu MB total\n",
        dev.device_id, elements_, buffer_bytes_ / 1024ull / 1024ull);

    cuda_check(cudaSetDevice(dev.device_id), "cudaSetDevice(init)");
    cuda_check(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1),
               "cudaDeviceSetCacheConfig(L1)");

    try {
        cuda_check(cudaEventCreateWithFlags(&done_evt_, cudaEventDisableTiming),
                   "cudaEventCreate(done)");
        cuda_check(cudaMalloc((void **)&dData_, buffer_bytes_),
                   "cudaMalloc(data)");
        cuda_check(cudaMalloc((void **)&dFaulty_, sizeof(int)),
                   "cudaMalloc(faulty)");
        cuda_check(cudaHostAlloc((void **)&hFaultyPinned_, sizeof(int),
                                 cudaHostAllocDefault),
                   "cudaHostAlloc(faulty)");

        // Init both halves
        cuda_check(cudaMemsetAsync(dFaulty_, 0, sizeof(int), dev.stream),
                   "memset(faulty)");
        if constexpr (std::is_same_v<T, float>) {
            launch_compute_verify_reinit_float((float *)dData_, dFaulty_,
                                               elements_, seed_, dev.stream);
        } else {
            launch_compute_verify_reinit_double((double *)dData_, dFaulty_,
                                                elements_, seed_, dev.stream);
        }
        cuda_check(cudaStreamSynchronize(dev.stream), "sync(init)");
    } catch (...) {
        release();
        throw;
    }
}

template <typename T> PerfModel StressCompute<T>::perf_model() const {
    // Per iteration: 8 stress_step calls (4 lanes × 2 runs)
    // Per stress_step: 11 FP ops + 6 SFU ops = 17 ops
    // Total: 8 × 17 = 136 ops per element per iteration
    const double ops = 136.0 * (double)elements_ * (double)iterations_;
    return PerfModel{MetricKind::Gflops, ops};
}

template <typename T> StepResult StressCompute<T>::step(DeviceContext &dev) {
    cuda_check(cudaSetDevice(dev.device_id), "cudaSetDevice(step)");
    cuda_check(cudaMemsetAsync(dFaulty_, 0, sizeof(int), dev.stream),
               "memset(faulty)");
    if constexpr (std::is_same_v<T, float>) {
        launch_compute_stress_2x_verify_reinit_float((float *)dData_, dFaulty_,
                                                     elements_, iterations_,
                                                     seed_, dev.stream);
    } else {
        launch_compute_stress_2x_verify_reinit_double(
            (double *)dData_, dFaulty_, elements_, iterations_, seed_,
            dev.stream);
    }
    seed_++;
    cuda_check(cudaGetLastError(), "kernel(compute stress)");
    cuda_check(cudaMemcpyAsync(hFaultyPinned_, dFaulty_, sizeof(int),
                               cudaMemcpyDeviceToHost, dev.stream),
               "DtoH(faulty)");

    cuda_check(cudaEventRecord(done_evt_, dev.stream), "eventRecord(done)");
    cuda_check(cudaEventSynchronize(done_evt_), "eventSync(done)");

    StepResult r;
    r.units = 1;
    r.errors = (uint64_t)(*hFaultyPinned_);
    return r;
}

// Explicit instantiation
template class StressCompute<float>;
template class StressCompute<double>;
