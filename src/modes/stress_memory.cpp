#include "gpuburn/modes/stress_memory.h"
#include "gpuburn/device_context.h"
#include "gpuburn/errors.h"
#include "gpuburn/kernels/memory_kernels.h"
#include <cstdio>
#include <curand_kernel.h>
#include <type_traits>

using namespace gpuburn;

template <typename T> void StressMemory<T>::release() noexcept {
    if (device_id_ < 0)
        return;
    cudaSetDevice(device_id_);

    if (done_evt_) {
        cudaEventDestroy(done_evt_);
        done_evt_ = nullptr;
    }
    if (dRef_) {
        cudaFree(dRef_);
        dRef_ = nullptr;
    }
    if (dTest_) {
        cudaFree(dTest_);
        dTest_ = nullptr;
    }
    if (dFaulty_) {
        cudaFree(dFaulty_);
        dFaulty_ = nullptr;
    }
    if (dRandStates_) {
        cudaFree(dRandStates_);
        dRandStates_ = nullptr;
    }
    if (hFaultyPinned_) {
        cudaFreeHost(hFaultyPinned_);
        hFaultyPinned_ = nullptr;
    }
    if (hPCIeSrc_) {
        cudaFreeHost(hPCIeSrc_);
        hPCIeSrc_ = nullptr;
    }
    if (hPCIeDst_) {
        cudaFreeHost(hPCIeDst_);
        hPCIeDst_ = nullptr;
    }
}

template <typename T>
void StressMemory<T>::init(DeviceContext &dev, const StressConfig &cfg) {
    device_id_ = dev.device_id;
    release();

    const size_t use_bytes = resolve_use_bytes(dev, cfg.use_bytes);
    // Need 2 buffers (ref + test) for validation
    buffer_bytes_ = use_bytes / 2;
    elements_ = buffer_bytes_ / sizeof(T);
    if (elements_ < 1024)
        throw std::runtime_error("Not enough VRAM for memory stress buffers");
    pcie_buffer_bytes_ = buffer_bytes_ < kMaxPCIeBufferBytes
                             ? buffer_bytes_
                             : kMaxPCIeBufferBytes;
    stride_ = (elements_ / 64) + 997;

    std::printf("GPU %d init: memory stress, %zu elements, ~%llu MB total "
                "(PCIe: %llu MB)\n",
                dev.device_id, elements_,
                (2 * buffer_bytes_) / 1024ull / 1024ull,
                pcie_buffer_bytes_ / 1024ull / 1024ull);
    cuda_check(cudaSetDevice(dev.device_id), "cudaSetDevice(init)");
    try {
        cuda_check(cudaEventCreateWithFlags(&done_evt_, cudaEventDisableTiming),
                   "cudaEventCreate(done)");
        cuda_check(cudaMalloc((void **)&dRef_, buffer_bytes_),
                   "cudaMalloc(ref)");
        cuda_check(cudaMalloc((void **)&dTest_, buffer_bytes_),
                   "cudaMalloc(test)");
        cuda_check(cudaMalloc((void **)&dFaulty_, sizeof(int)),
                   "cudaMalloc(faulty)");

        // Allocate curandState
        cuda_check(
            cudaMalloc(&dRandStates_, n_rand_states_ * sizeof(curandState)),
            "cudaMalloc(rand_states)");
        // Init cuRAND states
        launch_init_rand_states(dRandStates_, n_rand_states_, seed_,
                                dev.stream);
        cuda_check(cudaHostAlloc((void **)&hFaultyPinned_, sizeof(int),
                                 cudaHostAllocDefault),
                   "cudaHostAlloc(faulty)");

        // Allocate host pinned buffers for PCIe transfers
        cuda_check(cudaHostAlloc((void **)&hPCIeSrc_, pcie_buffer_bytes_,
                                 cudaHostAllocDefault),
                   "cudaHostAlloc(pcie_src)");
        cuda_check(cudaHostAlloc((void **)&hPCIeDst_, pcie_buffer_bytes_,
                                 cudaHostAllocDefault),
                   "cudaHostAlloc(pcie_dst)");

        // Init with data using cuRAND
        if constexpr (std::is_same_v<T, float>) {
            curand_check(
                curandGenerateUniform(dev.rng, (float *)dRef_, elements_),
                "curand(ref)");
            cuda_check(cudaMemcpyAsync(dTest_, dRef_, buffer_bytes_,
                                       cudaMemcpyDeviceToDevice, dev.stream),
                       "memcpy(ref->test)");
            cuda_check(cudaMemcpyAsync(hPCIeSrc_, dRef_, pcie_buffer_bytes_,
                                       cudaMemcpyDeviceToHost, dev.stream),
                       "memcpy(d2h init pcie_src)");
        } else {
            curand_check(curandGenerateUniformDouble(dev.rng, (double *)dRef_,
                                                     elements_),
                         "curand(ref)");
            cuda_check(cudaMemcpyAsync(dTest_, dRef_, buffer_bytes_,
                                       cudaMemcpyDeviceToDevice, dev.stream),
                       "memcpy(ref->test)");
            cuda_check(cudaMemcpyAsync(hPCIeSrc_, dRef_, pcie_buffer_bytes_,
                                       cudaMemcpyDeviceToHost, dev.stream),
                       "memcpy(d2h init pcie_src)");
        }
        cuda_check(cudaStreamSynchronize(dev.stream), "sync(init)");
    } catch (...) {
        release();
        throw;
    }
}

template <typename T> PerfModel StressMemory<T>::perf_model() const {
    // Approximate bytes per step across 4 patterns:
    // Pattern 0 (sequential): 2 buffers × cycles × 2 (R+W) per element
    // Pattern 1 (random): cycles × read-only
    // Pattern 2 (stride): cycles × read-only
    // Pattern 3 (PCIe): transfers (approximated as part of average)
    const double cycles_per_elem =
        (4.0 * (double)sequential_cycles_ + (double)random_cycles_ +
         (double)stride_cycles_) /
        4.0;
    const double bytes =
        cycles_per_elem * (double)elements_ * (double)sizeof(T);
    return PerfModel{MetricKind::Gbytes, bytes};
}

template <typename T> StepResult StressMemory<T>::step(DeviceContext &dev) {
    cuda_check(cudaSetDevice(dev.device_id), "cudaSetDevice(step)");
    // Cycle through patterns: sequential, random, stride, PCIe
    const int pattern = pattern_index_ % 4;
    pattern_index_++;

    // Reinit before sequential pattern to prevent drift
    if (pattern == 0) {
        if constexpr (std::is_same_v<T, float>) {
            curand_check(
                curandGenerateUniform(dev.rng, (float *)dRef_, elements_),
                "curand(ref)");
            cuda_check(cudaMemcpyAsync(dTest_, dRef_, buffer_bytes_,
                                       cudaMemcpyDeviceToDevice, dev.stream),
                       "memcpy(ref->test)");
        } else {
            curand_check(curandGenerateUniformDouble(dev.rng, (double *)dRef_,
                                                     elements_),
                         "curand(ref)");
            cuda_check(cudaMemcpyAsync(dTest_, dRef_, buffer_bytes_,
                                       cudaMemcpyDeviceToDevice, dev.stream),
                       "memcpy(ref->test)");
        }
    }

    if constexpr (std::is_same_v<T, float>) {
        switch (pattern) {
        case 0:
            // Sequential: run on dRef_ 16x, then dTest_ 16x, then compare
            launch_memory_sequential_float((float *)dRef_, elements_,
                                           sequential_cycles_, dev.stream);
            launch_memory_sequential_float((float *)dTest_, elements_,
                                           sequential_cycles_, dev.stream);
            break;
        case 1:
            // Random: read-only 16x, no check
            launch_memory_random_float((float *)dRef_, dRandStates_, elements_,
                                       random_cycles_, dev.stream);
            break;
        case 2:
            // Stride: read-only 16x, no check
            launch_memory_stride_float((float *)dRef_, elements_, stride_,
                                       stride_cycles_, dev.stream);
            break;
        case 3:
            // PCIe: transfer to two host buffers and back, then compare
            cuda_check(cudaMemcpyAsync(dRef_, hPCIeSrc_, pcie_buffer_bytes_,
                                       cudaMemcpyHostToDevice, dev.stream),
                       "H2D(ref)");
            cuda_check(cudaMemcpyAsync(dTest_, hPCIeSrc_, pcie_buffer_bytes_,
                                       cudaMemcpyHostToDevice, dev.stream),
                       "H2D(test)");
            cuda_check(cudaMemcpyAsync(hPCIeDst_, dRef_, pcie_buffer_bytes_,
                                       cudaMemcpyDeviceToHost, dev.stream),
                       "D2H(dst)");
            break;
        }
        // Validate sequential and PCIe patterns
        if (pattern == 0 || pattern == 3) {
            cuda_check(cudaMemsetAsync(dFaulty_, 0, sizeof(int), dev.stream),
                       "memset(faulty)");
            size_t compare_elems =
                (pattern == 3) ? (pcie_buffer_bytes_ / sizeof(T)) : elements_;
            launch_compare_memory_float((const float *)dRef_,
                                        (const float *)dTest_, dFaulty_,
                                        compare_elems, dev.stream);
        }
    } else {
        switch (pattern) {
        case 0:
            launch_memory_sequential_double((double *)dRef_, elements_,
                                            sequential_cycles_, dev.stream);
            launch_memory_sequential_double((double *)dTest_, elements_,
                                            sequential_cycles_, dev.stream);
            break;
        case 1:
            launch_memory_random_double((double *)dRef_, dRandStates_,
                                        elements_, random_cycles_, dev.stream);
            break;
        case 2:
            launch_memory_stride_double((double *)dRef_, elements_, stride_,
                                        stride_cycles_, dev.stream);
            break;
        case 3:
            cuda_check(cudaMemcpyAsync(dRef_, hPCIeSrc_, pcie_buffer_bytes_,
                                       cudaMemcpyHostToDevice, dev.stream),
                       "H2D(ref)");
            cuda_check(cudaMemcpyAsync(dTest_, hPCIeSrc_, pcie_buffer_bytes_,
                                       cudaMemcpyHostToDevice, dev.stream),
                       "H2D(test)");
            cuda_check(cudaMemcpyAsync(hPCIeDst_, dRef_, pcie_buffer_bytes_,
                                       cudaMemcpyDeviceToHost, dev.stream),
                       "D2H(dst)");
            break;
        }
        if (pattern == 0 || pattern == 3) {
            cuda_check(cudaMemsetAsync(dFaulty_, 0, sizeof(int), dev.stream),
                       "memset(faulty)");
            size_t compare_elems =
                (pattern == 3) ? (pcie_buffer_bytes_ / sizeof(T)) : elements_;
            launch_compare_memory_double((const double *)dRef_,
                                         (const double *)dTest_, dFaulty_,
                                         compare_elems, dev.stream);
        }
    }
    cuda_check(cudaGetLastError(), "kernel(memory stress)");
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

template class StressMemory<float>;
template class StressMemory<double>;
