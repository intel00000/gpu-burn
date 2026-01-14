// stress_memory.h
#pragma once
#include "gpuburn/config.h"
#include "gpuburn/stress_mode.h"
#include <cuda_runtime.h>

template <typename T> class StressMemory final : public StressMode {
  public:
    StressMemory() = default;
    ~StressMemory() override { release(); }

    std::string_view name() const override { return "memory"; }
    void init(DeviceContext &dev, const StressConfig &cfg) override;
    StepResult step(DeviceContext &dev) override;
    PerfModel perf_model() const override;

  private:
    int device_id_ = -1;
    const size_t n_rand_states_ = 256 * 256;
    const unsigned int seed_ = 42;
    const int sequential_cycles_ = 64;
    const int random_cycles_ = 64;
    const int stride_cycles_ = 64;

    size_t elements_ = 0;          // Number of elements
    size_t buffer_bytes_ = 0;      // Size of each buffer
    size_t pcie_buffer_bytes_ = 0; // Size of PCIe host buffers
    size_t stride_ = 0;            // Stride for cache thrashing

    T *dRef_ = nullptr;      // Reference buffer (first run)
    T *dTest_ = nullptr;     // Test buffer (validation run)
    int *dFaulty_ = nullptr; // Error counter
    int *hFaultyPinned_ = nullptr;
    void *dRandStates_ = nullptr;

    T *hPCIeSrc_ = nullptr; // Host source for H2D
    T *hPCIeDst_ = nullptr; // Host destination for D2H

    cudaEvent_t done_evt_ = nullptr;

    int pattern_index_ = 0;

    void release() noexcept;
};
